"""Prepare a source-locked BMC Research Note; never submit or invent declarations."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

import build_path_b_documents as path_b
import build_research_documents as documents
from docx import Document
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Pt, RGBColor
from selectivity import json_bytes, publish
from structure_comparison import verified_source_inventory

PACKAGE = Path("research/submission")
STATEMENTS = (
    "final_byline_and_addresses",
    "ethics_approval_and_consent",
    "consent_for_publication",
    "competing_interests",
    "funding",
    "author_contributions",
    "acknowledgements",
    "ai_assistance_disclosure",
    "prior_publication_and_submission_history",
    "label_and_decoy_provenance_disposition",
    "rights_permissions_and_code_license",
)
CONFIRMATIONS = (
    "contributor_history_resolved",
    "all_authors_approve_and_accept_accountability",
    "no_concurrent_submission",
    "all_retained_material_reuse_cleared",
    "source_provenance_disposition_approved",
    "human_review_of_ai_assisted_work_complete",
    "journal_fees_and_license_terms_reviewed",
)
MAX_ADDITIONAL_BYTES = 20_000_000


def release_gaps(record: dict[str, Any]) -> list[str]:
    gaps = []
    if record.get("status") != "author_confirmed":
        gaps.append("status: author confirmation required")
    statements, confirmations = record.get("statements", {}), record.get("confirmations", {})
    if not isinstance(statements, dict) or not isinstance(confirmations, dict):
        raise ValueError("Author statements and confirmations must be objects")
    for name in STATEMENTS:
        value = statements.get(name)
        if (
            not isinstance(value, str)
            or not value.strip()
            or re.search(r"\[(?:MISSING|OWNER|TODO|INSERT)|\bTBD\b", value, re.IGNORECASE)
        ):
            gaps.append(name)
    gaps.extend(name for name in CONFIRMATIONS if confirmations.get(name) is not True)
    return gaps


def article_checks(text: str) -> dict[str, Any]:
    headings = (
        "Abstract",
        "Keywords",
        "Introduction",
        "Main text",
        "Limitations",
        "List of abbreviations",
        "References",
    )
    positions = []
    for heading in headings:
        marker = f"## {heading}\n"
        if text.count(marker) != 1:
            raise ValueError(f"Missing or duplicate journal section: {heading}")
        positions.append(text.index(marker))
    if positions != sorted(positions):
        raise ValueError("Journal sections are out of order")
    abstract = text.split("## Abstract\n")[1].split("## Keywords\n")[0]
    if any(abstract.count(f"### {h}\n") != 1 for h in ("Objective", "Results")):
        raise ValueError("Abstract requires Objective and Results")
    if re.search(r"\[\d", abstract):
        raise ValueError("Abstract must not contain reference citations")
    body = text.split("## Introduction\n")[1].split("## List of abbreviations\n")[0]
    prose = "\n".join(line for line in body.splitlines() if not line.startswith(("|", "<!--")))
    abstract_words, body_words = len(abstract.split()), len(prose.split())
    keywords = text.split("## Keywords\n")[1].split("## Introduction\n")[0].strip().split(";")
    tables = [b for b in documents.read_blocks(body) if b[0] == "table"]
    if abstract_words > 200 or body_words > 2000 or not 3 <= len(keywords) <= 10:
        raise ValueError("Journal word or keyword limit exceeded")
    if len(tables) != 3 or re.search(r"^Figure \d", text, re.MULTILINE):
        raise ValueError("This note requires exactly three tables and no main figures")
    for number in range(1, 4):
        match = re.search(rf"^Table {number}\. (.*?)\n\n(.*?)\n\n<!--", text, re.S | re.M)
        if not match or len(match[1].split()) > 15 or len(match[2].split()) > 300:
            raise ValueError("Invalid table title or legend")
    return {
        "abstract_words_including_subheadings": abstract_words,
        "body_words_including_headings_titles_legends_excluding_cells": body_words,
        "keywords": len(keywords),
        "main_tables": len(tables),
        "main_figures": 0,
    }


def format_journal_docx(path: Path) -> None:
    doc = Document(str(path))
    doc.styles["Normal"].font.size = Pt(12)
    for name in ("Title", "Heading 1", "Heading 2", "Heading 3"):
        doc.styles[name].font.color.rgb = RGBColor(0, 0, 0)
    doc.styles["Title"].font.size = Pt(16)
    for paragraph in doc.paragraphs:
        paragraph.paragraph_format.line_spacing = 2
        paragraph.paragraph_format.page_break_before = False
    for table in doc.tables:
        table.style = "Table Grid"
        for row in table.rows:
            for cell in row.cells:
                for paragraph in cell.paragraphs:
                    paragraph.paragraph_format.line_spacing = 2
                    paragraph.paragraph_format.keep_with_next = False
    for section in doc.sections:
        numbering = OxmlElement("w:lnNumType")
        for key, value in {"countBy": "1", "start": "0", "restart": "continuous"}.items():
            numbering.set(qn(f"w:{key}"), value)
        section._sectPr.append(numbering)
        footer = section.footer.paragraphs[0]
        footer.text = "Page "
        field = OxmlElement("w:fldSimple")
        field.set(qn("w:instr"), "PAGE")
        footer._p.append(field)
    doc.save(str(path))


def zip_payload(files: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with ZipFile(buffer, "w", ZIP_DEFLATED, compresslevel=9) as archive:
        for name, payload in sorted(files.items()):
            if Path(name).is_absolute() or ".." in Path(name).parts:
                raise ValueError("Unsafe archive path")
            info = ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, payload)
    return buffer.getvalue()


def source_archives(repository: Path, generated: Path) -> tuple[bytes, bytes]:
    if (repository / ".git").exists():
        names = set(
            subprocess.check_output(["git", "ls-files", "-z"], cwd=repository).decode().split("\0")
        ) - {""}
    else:
        names = set(verified_source_inventory(repository))
    names.update(
        str(p.relative_to(repository)) for p in (repository / PACKAGE).rglob("*") if p.is_file()
    )
    names.update({"scripts/build_bmc_submission.py", "tests/test_bmc_submission.py"})
    groups: list[dict[str, bytes]] = [{}, {}]
    hashes: dict[str, str] = {}
    for name in sorted(names):
        path = repository / name
        if path.is_symlink() or any(p.is_symlink() for p in path.parents):
            raise ValueError(f"Symlink refused in source archive: {name}")
        payload = path.read_bytes()
        hashes[name] = hashlib.sha256(payload).hexdigest()
        group = int(name.endswith((".ccp4.gz", "-sf.cif.gz")))
        groups[group][f"source/{name}"] = payload
    groups[0]["source/SHA256SUMS.json"] = json_bytes(hashes)
    for path in sorted(generated.iterdir()):
        groups[0][f"generated/path_b/{path.name}"] = path.read_bytes()
    groups[0]["REPRODUCE.txt"] = (
        b"Extract Additional_file_2.zip and Additional_file_3.zip into the same directory.\n"
        b"The source/ directory contains the checkout; SHA256SUMS.json covers both archives.\n"
        b"From source/: follow README.md to install the pinned Python environment, then run\n"
        b".venv/bin/python scripts/build_bmc_submission.py --output ../reproduced-note\n"
        b"Use a new output directory. No training, download, submission or public deposit occurs.\n"
        b"Historical manuscripts, claims, AI reviews and manifests are provenance.\n"
        b"The reading proof and internal author/policy files are not manuscript uploads.\n"
        b"Rights and author confirmations remain prerequisites to external distribution.\n"
    )
    return zip_payload(groups[0]), zip_payload(groups[1])


def declarations(record: dict[str, Any], archive_hashes: dict[str, str]) -> str:
    statement = record["statements"]
    pairs = (
        ("Ethics approval and consent to participate", statement["ethics_approval_and_consent"]),
        ("Consent for publication", statement["consent_for_publication"]),
        (
            "Availability of data and materials",
            "The source snapshots, code and recorded results "
            "supporting this reanalysis accompany Additional files 2 and 3. The historical "
            "18,412-compound library and raw catalytic replicate vectors are unavailable. "
            + " ".join(f"{name} SHA-256: {digest}." for name, digest in archive_hashes.items()),
        ),
        ("Competing interests", statement["competing_interests"]),
        ("Funding", statement["funding"]),
        ("Authors' contributions", statement["author_contributions"]),
        ("Acknowledgements", statement["acknowledgements"]),
        ("AI assistance", statement["ai_assistance_disclosure"]),
    )
    return "## Declarations\n\n" + "\n\n".join(f"### {h}\n\n{v}" for h, v in pairs) + "\n\n"


def build(repository: Path, output: Path, *, require_author_confirmation: bool = False) -> Path:
    if output.is_symlink() or (output.exists() and (not output.is_dir() or any(output.iterdir()))):
        raise ValueError("Submission output must be a new or empty directory")
    path_b.locked_inputs(repository, repository / "research/results")
    package = repository / PACKAGE
    text = (package / "manuscript.md").read_text()
    inputs = [
        repository / name
        for name in (
            "research/structure_comparison/results/observed_proximity.json",
            "research/results/selectivity.json",
            "research/structure_comparison/model_support/results/model_support.json",
            "research/results/controls.json",
        )
    ]
    blocks = path_b.quantitative_blocks(*(json.loads(p.read_text()) for p in inputs))
    path_b.synchronize(text, {k: blocks[k] for k in ("sites", "paired", "controls")})
    checks = article_checks(text)
    record = json.loads((package / "author_confirmation.json").read_text())
    gaps = release_gaps(record)
    if require_author_confirmation and gaps:
        raise ValueError("Author release blocked: " + ", ".join(gaps))
    cover = (package / "cover_letter.md").read_text()
    if len(cover.split("Dear Editors,\n", 1)[1].split()) != 250:
        raise ValueError("Cover-letter scientific core must be 250 words")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".bmc-note-", dir=output.parent) as folder:
        staging = Path(folder)
        baseline = staging / "path_b"
        documents.build(
            repository / "research/manuscript.md",
            repository / "research/results",
            baseline,
            profile="path-b",
            repository=repository,
            structure_input=inputs[0],
            structure_manifest=inputs[0].parent / "derived/derived_manifest.json",
            handoff_directory=repository / "research/structure_comparison",
        )
        # The supplement keeps its own original numbered references and scientific limitations.
        original = (repository / "research/manuscript.md").read_text()
        methods = re.search(r"^## 2\..*?\n(.*?)^## 3\.", original, re.M | re.S)
        if methods is None:
            raise ValueError("Cannot locate source-locked structural methods")
        supplement = (baseline / "diagnostic_supplement.md").read_text()
        supplement = re.sub(
            r"^Author affiliation, funding, COI,.*?\n\n", "", supplement, flags=re.M | re.S
        )
        supplement = (
            "# Additional file 1: Supplementary methods and diagnostic results\n\n"
            "This supplement preserves scientific limitations and source-reported uncertainty. "
            "Author declarations and approvals are handled in the main-document release.\n\n"
            "## Extended structural and endpoint methods\n\n"
            + methods[1]
            + "\n\n"
            + supplement
            + "\n\n## Supplementary references\n\n"
            + original.split("## References\n", 1)[1]
        )
        (staging / "Additional_file_1.md").write_text(supplement)
        figures = []
        for heading, path, caption in _supplement_figures(baseline):
            figures.append((heading, path, caption))
        documents.render_document(
            documents.read_blocks(supplement),
            staging,
            figures,
            path_b_layout=True,
            stem="Additional_file_1",
            title="Supplementary methods and diagnostics",
        )
        first, second = source_archives(repository, baseline)
        archives = {"Additional_file_2.zip": first, "Additional_file_3.zip": second}
        for name, payload in archives.items():
            (staging / name).write_bytes(payload)
        if not gaps:
            text = re.sub(
                r"^AUTHOR-REVIEW DRAFT\..*?$",
                record["statements"]["final_byline_and_addresses"],
                text,
                flags=re.M,
            )
            text = text.replace(
                "## References\n",
                declarations(
                    record,
                    {
                        name: hashlib.sha256(payload).hexdigest()
                        for name, payload in archives.items()
                    },
                )
                + "## References\n",
            )
            cover = "Dear Editors,\n" + cover.split("Dear Editors,\n", 1)[1]
            cover += "\n\nCompeting interests: " + record["statements"]["competing_interests"]
            cover += (
                "\n\nPrior versions and submissions: "
                + record["statements"]["prior_publication_and_submission_history"]
            )
            cover += (
                "\n\nAll authors have approved this manuscript and accept accountability. "
                "No concurrent submission is active.\n"
            )
        (staging / "BMC_research_note.md").write_text(text)
        (staging / "Cover_letter.md").write_text(cover)
        for stem, content in (("BMC_research_note", text), ("Cover_letter", cover)):
            _, word = documents.render_document(
                documents.read_blocks(content),
                staging,
                [],
                path_b_layout=True,
                stem=stem,
                title="BMC Research Notes preparation",
                table_width_overrides={
                    "Control / method": [155, 105, 110, 110],
                    "PDB": [40, 45, 115, 44, 44, 48, 74, 70],
                },
            )
            format_journal_docx(word)
        for name in (
            "checklist.md",
            "author_actions.md",
            "reviewer_candidates.md",
            "claim_traceability.md",
            "author_confirmation.json",
        ):
            (staging / name).write_bytes((package / name).read_bytes())
        for name in ("Additional_file_1.pdf", *archives):
            if (staging / name).stat().st_size > MAX_ADDITIONAL_BYTES:
                raise ValueError(f"Additional file exceeds 20 MB: {name}")
        payloads = {p.name: p.read_bytes() for p in staging.iterdir() if p.is_file()}
        manifest = {
            "journal": "BMC Research Notes",
            "article_type": "Research note",
            "author_release_checks_complete": not gaps,
            "submission_ready": not gaps,
            "submitted": False,
            "biological_validation": False,
            "author_owned_gaps": gaps,
            "qualification": (
                "Checks verify documented confirmations, not their truth or editorial acceptance."
            ),
            "article_checks": checks,
            "cover_letter_scientific_core_words": 250,
            "source_lock_sha256": path_b.LOCK_SHA256,
            "files": {
                name: {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}
                for name, data in sorted(payloads.items())
            },
        }
        payloads["submission-manifest.json"] = json_bytes(manifest)
        publish(output, payloads, "submission-manifest.json")
    return output / "submission-manifest.json"


def _supplement_figures(baseline: Path) -> list[tuple[str, Path, str]]:
    """Recover exact captions and image relationships from the generated supplement DOCX."""
    doc = Document(str(baseline / "Path_B_diagnostic_supplement.docx"))
    figures: list[tuple[str, Path, str]] = []
    for index, paragraph in enumerate(doc.paragraphs):
        drawings = paragraph._p.xpath(".//w:drawing")
        if not drawings:
            continue
        if len(drawings) != 1 or not 0 < index < len(doc.paragraphs) - 1:
            raise ValueError("Supplement figure/caption mapping is incomplete")
        heading, caption = doc.paragraphs[index - 1].text, doc.paragraphs[index + 1].text
        if not heading.startswith("Figure S") or not caption:
            raise ValueError("Supplement figure/caption mapping is incomplete")
        blips = paragraph._p.xpath(".//a:blip")
        if len(blips) != 1:
            raise ValueError("Ambiguous supplement image")
        relationship = blips[0].get(qn("r:embed"))
        if relationship is None:
            raise ValueError("Missing supplement image relationship")
        part = doc.part.related_parts[relationship]
        path = baseline / f"supplement-image-{len(figures)}.png"
        path.write_bytes(part.blob)
        figures.append((heading, path, caption))
    if not figures or len(figures) != len(doc.inline_shapes):
        raise ValueError("Supplement figure/caption mapping is incomplete")
    return figures


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, default=documents.ROOT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--require-author-confirmation", action="store_true")
    args = parser.parse_args()
    build(
        args.repository, args.output, require_author_confirmation=args.require_author_confirmation
    )


if __name__ == "__main__":
    main()
