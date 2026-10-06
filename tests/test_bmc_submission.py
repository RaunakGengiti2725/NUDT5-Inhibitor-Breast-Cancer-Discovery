from __future__ import annotations

import copy
import gzip
import hashlib
import io
import json
from pathlib import Path
from typing import Any
from xml.etree import ElementTree
from zipfile import ZipFile

import pytest
from build_bmc_submission import (
    CONFIRMATIONS,
    MAX_ADDITIONAL_BYTES,
    PACKAGE,
    STATEMENTS,
    article_checks,
    build,
    declarations,
    format_journal_docx,
    release_gaps,
    source_archives,
    zip_payload,
)
from build_research_documents import ROOT
from docx import Document


def test_article_limits_and_required_baselines() -> None:
    text = (ROOT / PACKAGE / "manuscript.md").read_text()
    checks = article_checks(text)
    assert checks["abstract_words_including_subheadings"] <= 200
    assert checks["body_words_including_headings_titles_legends_excluding_cells"] <= 2000
    assert checks["main_tables"] == 3 and checks["keywords"] == 6
    for name in ("Constant_0_5", "Train_prevalence", "tpsa_only_lr", "hba_only_lr", "Property_LR"):
        assert name in text
    assert "3.756 Å" in text and "3.251 Å" in text
    assert "0.600" in text and "No finite bound" in text
    assert "zero-occupancy" in text and "0.78" in text
    assert "## Declarations" not in text


@pytest.mark.parametrize(
    "mutation",
    ["abstract", "body", "sections", "objective", "citation", "title", "legend", "display"],
)
def test_format_violations_fail(mutation: str) -> None:
    text = (ROOT / PACKAGE / "manuscript.md").read_text()
    if mutation == "abstract":
        text = text.replace("### Objective\n", "### Objective\n" + "word " * 201)
    elif mutation == "body":
        text = text.replace("## Introduction\n", "## Introduction\n" + "word " * 2001)
    elif mutation == "sections":
        text += "\n## Abstract\n"
    elif mutation == "objective":
        text = text.replace("### Objective\n", "### Background\n")
    elif mutation == "citation":
        text = text.replace("### Objective\n", "### Objective\n[1]")
    elif mutation == "title":
        text = text.replace("Table 1. Deposited", "Table 1. " + "word " * 16 + "Deposited")
    elif mutation == "legend":
        text = text.replace("Site identifiers are", "word " * 301 + "Site identifiers are")
    else:
        text += "\nFigure 1. Extra display\n"
    with pytest.raises(ValueError):
        article_checks(text)


def test_author_release_is_blocked_without_facts(tmp_path: Path) -> None:
    record = json.loads((ROOT / PACKAGE / "author_confirmation.json").read_text())
    assert set(STATEMENTS).issubset(release_gaps(record))
    assert set(CONFIRMATIONS).issubset(release_gaps(record))
    with pytest.raises(ValueError, match="Author release blocked"):
        build(ROOT, tmp_path / "release", require_author_confirmation=True)
    assert not (tmp_path / "release").exists()


def approved_fixture() -> dict[str, Any]:
    return {
        "status": "author_confirmed",
        "statements": {name: f"Test fixture statement for {name}" for name in STATEMENTS},
        "confirmations": dict.fromkeys(CONFIRMATIONS, True),
    }


@pytest.mark.parametrize("bad", [None, "", "   ", "[MISSING: affiliation]", "TBD", 1])
def test_missing_and_placeholder_statements_refused(bad: Any) -> None:
    record = approved_fixture()
    record["statements"]["funding"] = bad
    assert "funding" in release_gaps(record)


@pytest.mark.parametrize("bad", [False, None, "true", 1])
def test_confirmations_must_be_explicit_booleans(bad: Any) -> None:
    record = approved_fixture()
    record["confirmations"]["contributor_history_resolved"] = bad
    assert "contributor_history_resolved" in release_gaps(record)


def test_author_statements_are_not_invented() -> None:
    record = approved_fixture()
    original = copy.deepcopy(record)
    assert release_gaps(record) == []
    text = declarations(record, {"Additional_file_2.zip": "a" * 64})
    for key in (
        "funding",
        "competing_interests",
        "author_contributions",
        "ai_assistance_disclosure",
    ):
        assert record["statements"][key] in text
    assert "Not applicable" not in text and "no funding" not in text
    assert record == original


def test_docx_is_editable_double_spaced_and_numbered(tmp_path: Path) -> None:
    path = tmp_path / "note.docx"
    doc = Document()
    doc.add_paragraph("Text")
    doc.add_table(rows=2, cols=3)
    doc.save(str(path))
    format_journal_docx(path)
    with ZipFile(path) as archive:
        xml = ElementTree.fromstring(archive.read("word/document.xml"))
        ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
        spacing = xml.findall(".//w:spacing", ns)
        assert spacing and all(n.attrib[f"{{{ns['w']}}}line"] == "480" for n in spacing)
        assert xml.find(".//w:lnNumType", ns) is not None
        assert b'"PAGE"' in archive.read("word/footer1.xml")
        assert b'w:type="page"' not in archive.read("word/document.xml")


def test_archives_are_deterministic_and_reject_traversal() -> None:
    files = {"a": b"first", "b": b"second"}
    assert zip_payload(files) == zip_payload(dict(reversed(list(files.items()))))
    with pytest.raises(ValueError, match="Unsafe"):
        zip_payload({"../outside": b"no"})


def test_official_source_captures_match_hashes() -> None:
    folder = ROOT / PACKAGE / "sources"
    manifest = json.loads((folder / "manifest.json").read_text())
    assert len(manifest) == 10
    for name, record in manifest.items():
        assert hashlib.sha256((folder / name).read_bytes()).hexdigest() == record["sha256"]
        assert record["url"].startswith("https://")
        assert (
            hashlib.sha256(gzip.decompress((folder / name).read_bytes())).hexdigest()
            == (record["decoded_sha256"])
        )


def test_source_archive_without_git_and_missing_map_part(tmp_path: Path) -> None:
    root = tmp_path / "source"
    for name in ("scripts/build_bmc_submission.py", "tests/test_bmc_submission.py", "a.ccp4.gz"):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"fixture")
    inventory = {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in root.rglob("*")
        if p.is_file()
    }
    (root / "SHA256SUMS.json").write_text(json.dumps(inventory))
    generated = tmp_path / "generated"
    generated.mkdir()
    first, second = source_archives(root, generated)
    with ZipFile(io.BytesIO(first)) as archive:
        assert "source/scripts/build_bmc_submission.py" in archive.namelist()
        assert "source/a.ccp4.gz" not in archive.namelist()
    with ZipFile(io.BytesIO(second)) as archive:
        assert "source/a.ccp4.gz" in archive.namelist()
    (root / "a.ccp4.gz").write_bytes(b"changed")
    with pytest.raises(ValueError, match="inventory mismatch"):
        source_archives(root, generated)


def test_nonempty_output_refused(tmp_path: Path) -> None:
    (tmp_path / "sentinel").write_text("keep")
    with pytest.raises(ValueError, match="new or empty"):
        build(ROOT, tmp_path)
    assert (tmp_path / "sentinel").read_text() == "keep"


def test_full_package_build(tmp_path: Path) -> None:
    output = tmp_path / "note"
    manifest = json.loads(build(ROOT, output).read_text())
    assert manifest["submission_ready"] is False and manifest["submitted"] is False
    assert manifest["author_owned_gaps"]
    assert not manifest["biological_validation"]
    for name, record in manifest["files"].items():
        payload = (output / name).read_bytes()
        assert hashlib.sha256(payload).hexdigest() == record["sha256"]
        assert len(payload) == record["bytes"]
    for name in ("Additional_file_1.pdf", "Additional_file_2.zip", "Additional_file_3.zip"):
        assert (output / name).stat().st_size <= MAX_ADDITIONAL_BYTES
    with ZipFile(output / "BMC_research_note.docx") as archive:
        xml = archive.read("word/document.xml")
        assert b"AUTHOR-REVIEW DRAFT" in xml
        assert b"lnNumType" in xml
    extracted = tmp_path / "extracted"
    for name in ("Additional_file_2.zip", "Additional_file_3.zip"):
        with ZipFile(output / name) as archive:
            archive.extractall(extracted)
    source = extracted / "source"
    inventory = json.loads((source / "SHA256SUMS.json").read_text())
    for name, digest in inventory.items():
        assert hashlib.sha256((source / name).read_bytes()).hexdigest() == digest
    supplement = (output / "Additional_file_1.md").read_text()
    assert "## Extended structural and endpoint methods" in supplement
    assert "## Supplementary references" in supplement
    assert "The journal-specific guide was inaccessible" not in supplement
    doc = Document(str(output / "BMC_research_note.docx"))
    assert doc.tables[2].columns[0].width is not None
    assert doc.tables[2].columns[0].width.pt == 155
    assert len(Document(str(output / "Additional_file_1.docx")).inline_shapes) == 10
    rebuilt = tmp_path / "rebuilt_without_git"
    second_manifest = json.loads(build(source, rebuilt).read_text())
    assert second_manifest["article_checks"] == manifest["article_checks"]
    assert (rebuilt / "BMC_research_note.md").read_bytes() == (
        output / "BMC_research_note.md"
    ).read_bytes()
