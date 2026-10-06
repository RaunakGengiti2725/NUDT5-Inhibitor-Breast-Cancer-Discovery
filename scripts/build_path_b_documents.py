"""Source-locked Path B documents; no model fitting or scientific-input mutation."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any

import build_extension_figures as extensions
import build_research_documents as documents
import build_selectivity_figures as paired_figures
import build_structure_comparison_figures as geometry
import build_structure_model_support as support
from selectivity import artifacts, json_bytes
from sklearn.metrics import roc_auc_score
from structure_comparison import json_bytes as structure_json_bytes

LOCK_SHA256 = "f06fd2bd37f203b3d9f59285bd66718ef25427d5298d2ac9d287a763ec3c865c"


def locked_inputs(repository: Path, results: Path) -> list[Path]:
    lock = repository / "research/path_b/frozen_inputs.json"
    if lock.is_symlink() or hashlib.sha256(lock.read_bytes()).hexdigest() != LOCK_SHA256:
        raise ValueError("Path B input lock is not trusted")
    inputs = [lock]
    for name, expected in json.loads(lock.read_text())["files"].items():
        path = (
            results / Path(name).name if name.startswith("research/results/") else repository / name
        )
        if any(p.is_symlink() for p in (path, *path.parents)) or (
            hashlib.sha256(path.read_bytes()).hexdigest() != expected
        ):
            raise ValueError(f"Path B frozen input mismatch: {name}")
        inputs.append(path)
    return inputs


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    return "\n".join(
        "| " + " | ".join(row) + " |" for row in [headers, ["---"] * len(headers), *rows]
    )


def diagnostic_controls(controls: dict[str, Any]) -> list[dict[str, Any]]:
    evaluation = controls["evaluations"]["full_valid_set"]
    rows = evaluation["predictions"]
    if evaluation["split"] != "exact_scaffold" or evaluation["seed"] != 42:
        raise ValueError("Path B requires the recorded exact-scaffold seed-42 design")
    if [r["index"] for r in rows] != list(range(evaluation["n"])):
        raise ValueError("Diagnostic score indices are not aligned")
    for fold in evaluation["folds"]:
        if fold["group_overlap"] or fold["scaffold_overlap"]:
            raise ValueError("Diagnostic fold overlap")
        if sorted(r["index"] for r in rows if r["fold"] == fold["fold"]) != sorted(
            fold["test_indices"]
        ):
            raise ValueError("Diagnostic fold assignments are not aligned")
    labels = [r["label"] for r in rows]
    records = []
    methods = (
        "Constant_0_5",
        "Train_prevalence",
        "Property_LR",
        "Nearest_active",
        "Tanimoto_kNN",
        "RF",
        "GBT",
        "SVM_RBF",
        "Equal_mean",
    )
    for name in methods:
        scores = [r["scores"][name] for r in rows]
        auc = float(roc_auc_score(labels, scores))
        if not math.isclose(auc, evaluation["methods"][name]["auc"], abs_tol=1e-12):
            raise ValueError(f"Diagnostic metric drift: {name}")
        records.append(
            {
                "method": name,
                "auc": auc,
                "scores": scores,
                "source": f"/evaluations/full_valid_set/predictions/*/scores/{name}",
            }
        )
    for name, ablation in evaluation["ablations"].items():
        if not name.endswith("_only_lr"):
            continue
        auc = float(roc_auc_score(labels, ablation["scores"]))
        if not math.isclose(auc, ablation["metrics"]["auc"], abs_tol=1e-12):
            raise ValueError(f"Diagnostic metric drift: {name}")
        records.append(
            {
                "method": name,
                "auc": auc,
                "scores": ablation["scores"],
                "source": f"/evaluations/full_valid_set/ablations/{name}/scores",
            }
        )
    return records


def endpoint_text(endpoint: dict[str, Any]) -> str:
    if endpoint["status"] == "right_censored":
        return f">{endpoint['bound']:g}"
    if endpoint["status"] == "numeric":
        return str(endpoint["table1_text"])
    return "Untested"


def quantitative_blocks(
    structural: dict[str, Any],
    paired: dict[str, Any],
    model_support: dict[str, Any],
    controls: dict[str, Any],
) -> dict[str, str]:
    counts = Counter(r["geometry_status"] for r in structural["residue_proximity"])
    rscc = [s["report"]["metrics"]["rscc"]["value"] for s in model_support["sites"]]
    rsr = [s["report"]["metrics"]["rsr"]["value"] for s in model_support["sites"]]
    witnesses = sorted(
        [
            r
            for r in structural["residue_proximity"]
            if r["residue_identity"]["pdb_id"] == "8RIY"
            and r["residue_identity"]["auth_seq_id"] == "51"
            and r["within_5_0A"]
        ],
        key=lambda r: r["residue_identity"]["auth_asym_id"],
    )
    if len(witnesses) != 2:
        raise ValueError("Expected both local Arg51 witnesses")
    witness_text = []
    for row in witnesses:
        witness = row["minimum_witness_pairs"][0]
        atom = witness["protein_atom"]
        witness_text.append(
            f"8RIY auth chain {atom['auth_asym_id']} (label {atom['label_asym_id']}): "
            f"Arg51 {atom['label_atom_id']} to W0O {witness['ligand_atom']['label_atom_id']} "
            f"is {witness['distance_A']:.3f} Å; witness occupancy {atom['occupancy']:g}; "
            f"residue status {row['geometry_status']}."
        )
    compound9 = next(r for r in paired["rows"] if r["source_compound"] == "9")
    abstract = (
        "Deposited inhibitor complexes describe local geometry but cannot by themselves test "
        "energetic dependence. We reanalyse all "
        + str(len(structural["sites"]))
        + " W0O sites in published Nudix hydrolase 5 (NUDT5) and NUDT14 structures "
        "8RIY/8OTV, retaining both "
        "receptor chains, occupancy, alternate conformers and missingness. Official validation "
        f"reports give ligand real-space correlation coefficient (RSCC) "
        f"{min(rscc):.3f}–{max(rscc):.3f} and real-space R (RSR) "
        f"{min(rsr):.3f}–{max(rsr):.3f}. Fixed sampling and slices of precomputed PDBe maps "
        "provide limited model-support inspection, not independent density validation. "
        f"The nearest retained 8RIY Arg51 witnesses differ: backbone N at "
        f"{witnesses[0]['observed_min_distance_A']:.3f} Å in auth chain AAA and side-chain CD at "
        f"{witnesses[1]['observed_min_distance_A']:.3f} Å in BBB. AAA has a zero-occupancy CZ "
        "and report-listed local geometry problems. BBB CD has fractional occupancy. "
        "These observations qualify the published chain-specific account without testing or "
        "refuting its energetic interpretation. Published compound-9 catalytic means give "
        "NUDT14/NUDT5 half-maximal inhibitory concentration (IC50) ratio "
        f"{compound9['ratio']['point']:.3f}; unequal reaction times, "
        "shared control normalization and unresolved replication wording prevent an affinity "
        "or general selectivity interpretation. Crystal copies are nonindependent. Source-label "
        "controls are retained only as diagnostics. The contribution is descriptive and "
        "methodological; no new inhibitor, biological experiment or prospective validation "
        "is reported."
    )
    site_rows = []
    for site in model_support["sites"]:
        identity, metrics = site["site_identity"], site["report"]["metrics"]
        site_rows.append(
            [
                identity["pdb_id"],
                identity["target"],
                f"{identity['label_asym_id']} / {identity['auth_asym_id']} / "
                f"{identity['auth_seq_id']}",
                metrics["rscc"]["raw"],
                metrics["rsr"]["raw"],
                metrics["NatomsEDS"]["raw"],
            ]
        )
    paired_rows = []
    for row in paired["rows"]:
        if not row["source_pair_has_both_endpoints"]:
            continue
        ratio = row["ratio"]
        value = (
            f"{ratio['point']:.6f}"
            if ratio["point"] is not None
            else f"{ratio['comparator']}{ratio['bound']:.6f}"
            if ratio["bound"] is not None
            else "No finite bound"
        )
        paired_rows.append(
            [
                row["source_compound"],
                endpoint_text(row["endpoints"]["NUDT5"]),
                endpoint_text(row["endpoints"]["NUDT14"]),
                value,
            ]
        )
    diagnostic_rows = diagnostic_controls(controls)
    return {
        "abstract": abstract,
        "geometry": (
            f"The fixed-contract extraction retains {len(structural['residue_proximity']):,} "
            f"residue-conformer rows: {counts['observed']:,} observed, "
            f"{counts['partial_observed']} partial and {counts['refused']} refused/null. "
            f"It retains {len(structural['atom_pairs_within_5A']):,} atom pairs within 5 Å. "
            "These are coordinate records, not independent observations. Refused rows are not "
            "no-contact results. Partial minima use retained atoms only.\n\n"
            + " ".join(witness_text)
        ),
        "sites": markdown_table(
            ["PDB", "Target", "W0O label / auth / residue", "RSCC", "RSR", "Report atoms"],
            site_rows,
        ),
        "paired": markdown_table(
            [
                "Compound",
                "NUDT5 IC50, µM (mean ± SD or bound)",
                "NUDT14 IC50, µM (mean ± SD or bound)",
                "R or strict bound",
            ],
            paired_rows,
        ),
        "controls": markdown_table(
            ["Control / method", "Pooled ROC-AUC"],
            [[r["method"], f"{r['auc']:.4f}"] for r in diagnostic_rows],
        ),
    }


def synchronize(text: str, blocks: dict[str, str], *, check: bool = True) -> str:
    for name, value in blocks.items():
        start, end = f"<!-- path-b:{name}:start -->", f"<!-- path-b:{name}:end -->"
        if text.count(start) != 1 or text.count(end) != 1:
            raise ValueError(f"Missing or duplicate Path B quantitative block: {name}")
        pattern = re.compile(re.escape(start) + r"\n.*?\n" + re.escape(end), re.DOTALL)
        replacement = f"{start}\n{value}\n{end}"
        updated, n = pattern.subn(replacement, text)
        if n != 1 or (check and updated != text):
            raise ValueError(f"Manuscript quantitative drift: {name}")
        text = updated
    return text


def build(
    manuscript: Path,
    results: Path,
    output: Path,
    paired: dict[str, Any],
    structural: dict[str, Any],
    repository: Path,
) -> tuple[Path, Path, list[Path]]:
    inputs = locked_inputs(repository, results)
    model_support, maps = support.build(repository)
    recorded = repository / support.PACKAGE / "results/model_support.json"
    if structure_json_bytes(model_support) != recorded.read_bytes():
        raise ValueError("Model-support result does not match trusted source regeneration")
    for name in ("benchmark.json", "controls.json", "transfer.json"):
        (output / name).write_bytes((results / name).read_bytes())
    for name in (
        "manifest.json",
        "controls-manifest.json",
        "transfer-manifest.json",
        "selectivity-manifest.json",
    ):
        (output / f"historical-{name}").write_bytes((results / name).read_bytes())
    controls = json.loads((results / "controls.json").read_text())
    blocks = quantitative_blocks(structural, paired, model_support, controls)
    text = synchronize(manuscript.read_text(), blocks)
    package = repository / "research/path_b"
    supplementary = package / "diagnostic_supplement.md"
    inputs.extend([Path(__file__), supplementary, recorded])
    inputs.extend(p for p in (repository / support.PACKAGE).rglob("*") if p.is_file())
    inputs.append(Path(support.__file__))
    figures = []
    for number, target in enumerate(("NUDT5", "NUDT14"), start=1):
        for name, payload in geometry.render(
            structural, target=target, document_layout=True
        ).items():
            (output / name).write_bytes(payload)
        figures.append(
            (
                f"Figure {number}. {target} observed geometry",
                output / f"{geometry.FIGURE}_{target}.png",
                documents.STRUCTURE_CAPTION,
            )
        )
    for name, payload in {
        **geometry.tables(structural),
        **support.tables(model_support),
        **support.figures(model_support, maps),
    }.items():
        (output / name).write_bytes(payload)
    map_figures = sorted(output.glob("*_slices.png"))
    for letter, path in zip("AB", [p for p in map_figures if "_51_" in p.name], strict=True):
        figures.append(
            (
                f"Figure 3{letter}. Arg51 fixed map slices",
                path,
                "Precomputed PDBe maps, fixed slice planes and atom sampling only. "
                "Model-dependent maps are not independent ligand validation. "
                "No omit/polder map, rerefinement or energetic inference. "
                "Map values have no pass/fail support threshold.",
            )
        )
    (output / "manuscript.md").write_text(text)
    pdf, word = documents.render_document(
        documents.read_blocks(text),
        output,
        figures,
        path_b_layout=True,
        stem="Path_B_manuscript",
        title="Deposited structural pharmacology",
    )
    diagnostics = [
        (
            "Figure S1. Original-label diagnostics",
            documents.diagnostic_figure(results, output),
            "Source labels versus unassayed presumed negatives; no target-recognition "
            "inference. Different designs are separate diagnostics; no population interval.",
        )
    ]
    extension_figures = extensions.build_figures(results, output)
    for number, (path, caption) in enumerate(extension_figures, start=2):
        caption = re.sub(r"^Figure \d+\.", f"Figure S{number}.", caption)
        diagnostics.append((f"Figure S{number}. Diagnostic context", path, caption))
    annex = extensions.write_tables(results, output).read_text()
    for name, payload in {**artifacts(paired), **paired_figures.render(paired)}.items():
        (output / name).write_bytes(payload)
    for name in paired_figures.FIGURE_NAMES:
        diagnostics.append(
            (
                f"Figure S{len(diagnostics) + 1}. Retrospective paired scores",
                output / f"{name}.png",
                "Reported-mean catalytic ratios only; "
                "unequal protocols and unresolved covariance. Bounds are not confidence "
                "intervals. Frozen scores are not selectivity predictions.",
            )
        )
    for path in map_figures:
        if "_51_" not in path.name:
            diagnostics.append(
                (
                    f"Figure S{len(diagnostics) + 1}. W0O fixed map slices",
                    path,
                    "All deposited W0O sites retained. Precomputed model-dependent "
                    "maps; limited inspection, no independent density validation.",
                )
            )
    supplement_text = supplementary.read_text() + "\n\n" + annex
    supplement_text += "\n\n" + support.summary(model_support).decode()
    (output / "diagnostic_supplement.md").write_text(supplement_text)
    documents.render_document(
        documents.read_blocks(supplement_text),
        output,
        diagnostics,
        path_b_layout=True,
        stem="Path_B_diagnostic_supplement",
        title="Path B diagnostic supplement",
    )
    provenance = {
        "design": "full_valid_set/exact_scaffold/seed42",
        "source": "research/results/controls.json",
        "methods": diagnostic_controls(controls),
        "rows": controls["evaluations"]["full_valid_set"]["predictions"],
    }
    (output / "diagnostic_control_sources.json").write_bytes(json_bytes(provenance))
    for name, value in blocks.items():
        (output / f"quantitative_{name}.md").write_text(value + "\n")
        if name in {"sites", "paired", "controls"}:
            table = documents.read_blocks(value)[0][1]
            buffer = io.StringIO()
            csv.writer(buffer, lineterminator="\n").writerows(table)
            (output / f"table_{name}.csv").write_text(buffer.getvalue())
    (output / "model_support.json").write_bytes(json_bytes(model_support))
    for name in ("author_requests.md", "laboratory_specification.md", "current_gaps.md"):
        path = package / name
        if not path.read_bytes().strip():
            raise ValueError(f"Empty Path B handoff: {name}")
        inputs.append(path)
        (output / name).write_bytes(path.read_bytes())
    return pdf, word, inputs
