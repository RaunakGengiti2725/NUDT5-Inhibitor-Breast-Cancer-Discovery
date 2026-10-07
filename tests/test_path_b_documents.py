from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any
from zipfile import ZipFile

import pytest
from build_path_b_documents import (
    diagnostic_controls,
    locked_inputs,
    quantitative_blocks,
    synchronize,
)
from build_research_documents import ROOT, build

PACKAGE = ROOT / "research/path_b"
STRUCTURE = ROOT / "research/structure_comparison"


def inputs() -> tuple[dict[str, Any], ...]:
    return tuple(
        json.loads(p.read_text())
        for p in (
            STRUCTURE / "results/observed_proximity.json",
            ROOT / "research/results/selectivity.json",
            STRUCTURE / "model_support/results/model_support.json",
            ROOT / "research/results/controls.json",
        )
    )


def test_every_quantitative_block_matches_frozen_sources() -> None:
    blocks = quantitative_blocks(*inputs())
    text = (ROOT / "research/manuscript.md").read_text()
    assert synchronize(text, blocks) == text
    assert "0.928–0.952" in blocks["abstract"]
    assert "3.756" in blocks["abstract"] and "3.251" in blocks["abstract"]
    assert "ROC-AUC" not in blocks["abstract"] and "TPSA" not in blocks["abstract"]
    assert "Property_LR" in blocks["controls"] and "tpsa_only_lr" in blocks["controls"]
    assert "Train_prevalence" in blocks["controls"] and "Constant_0_5" in blocks["controls"]
    assert "NA" not in blocks["paired"] and ">50" in blocks["paired"]


@pytest.mark.parametrize("name", ["abstract", "geometry", "sites", "paired", "controls"])
def test_changed_or_missing_quantitative_blocks_fail(name: str) -> None:
    blocks = quantitative_blocks(*inputs())
    text = (ROOT / "research/manuscript.md").read_text()
    with pytest.raises(ValueError, match="quantitative drift"):
        synchronize(text.replace(blocks[name], blocks[name] + " changed"), blocks)
    with pytest.raises(ValueError, match="Missing or duplicate"):
        synchronize(text.replace(f"<!-- path-b:{name}:start -->", ""), blocks)
    with pytest.raises(ValueError, match="Missing or duplicate"):
        synchronize(text + f"<!-- path-b:{name}:end -->", blocks)


def test_numeric_source_change_drifts_abstract_and_tables() -> None:
    structural, paired, support, controls = inputs()
    support["sites"][0]["report"]["metrics"]["rscc"]["value"] = 0.5
    blocks = quantitative_blocks(structural, paired, support, controls)
    with pytest.raises(ValueError, match="abstract"):
        synchronize((ROOT / "research/manuscript.md").read_text(), blocks)


@pytest.mark.parametrize("mutation", ["auc", "index", "fold", "overlap", "seed", "ablation"])
def test_diagnostic_metric_and_alignment_corruption_fails(mutation: str) -> None:
    controls = inputs()[-1]
    evaluation = controls["evaluations"]["full_valid_set"]
    if mutation == "auc":
        evaluation["methods"]["Equal_mean"]["auc"] = 0.0
    elif mutation == "ablation":
        evaluation["ablations"]["tpsa_only_lr"]["metrics"]["auc"] = 0.0
    elif mutation == "index":
        evaluation["predictions"][0]["index"] = 99
    elif mutation == "fold":
        evaluation["predictions"][0]["fold"] = 99
    elif mutation == "overlap":
        evaluation["folds"][0]["group_overlap"] = ["overlap"]
    else:
        evaluation["seed"] = 41
    with pytest.raises(ValueError):
        diagnostic_controls(controls)


def test_frozen_inputs_are_trusted_not_self_consistent(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    shutil.copytree(PACKAGE, repo / "research/path_b")
    lock = repo / "research/path_b/frozen_inputs.json"
    data = json.loads(lock.read_text())
    data["files"] = {}
    lock.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="not trusted"):
        locked_inputs(repo, ROOT / "research/results")


@pytest.mark.parametrize("name", ["controls.json", "benchmark.json", "transfer.json"])
def test_changed_frozen_diagnostics_fail(tmp_path: Path, name: str) -> None:
    results = tmp_path / "results"
    shutil.copytree(ROOT / "research/results", results)
    (results / name).write_text("{}")
    with pytest.raises(ValueError, match="frozen input mismatch"):
        locked_inputs(ROOT, results)


def build_path_b(output: Path) -> tuple[Path, Path]:
    return build(
        ROOT / "research/manuscript.md",
        ROOT / "research/results",
        output,
        profile="path-b",
        structure_input=STRUCTURE / "results/observed_proximity.json",
        structure_manifest=STRUCTURE / "results/derived/derived_manifest.json",
    )


def test_path_b_separates_figures_and_records_all_inputs_without_fitting(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import pipeline
    import sklearn.ensemble
    import sklearn.linear_model
    import sklearn.svm
    from docx import Document

    def forbidden(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("No model fitting is authorized")

    monkeypatch.setattr(pipeline, "fit_scores", forbidden)
    monkeypatch.setattr(sklearn.ensemble.RandomForestClassifier, "fit", forbidden)
    monkeypatch.setattr(sklearn.linear_model.LogisticRegression, "fit", forbidden)
    monkeypatch.setattr(sklearn.svm.SVC, "fit", forbidden)
    output = tmp_path / "documents"
    pdf, word = build_path_b(output)
    assert pdf.read_bytes().startswith(b"%PDF")
    for paragraph in Document(str(word)).paragraphs:
        if paragraph._p.xpath(".//w:drawing"):
            assert paragraph.paragraph_format.line_spacing == 1
            assert paragraph.paragraph_format.keep_with_next

    with ZipFile(word) as archive:
        main = archive.read("word/document.xml").decode()
        assert len([n for n in archive.namelist() if n.startswith("word/media/")]) == 4
    with ZipFile(output / "Path_B_diagnostic_supplement.docx") as archive:
        supplement = archive.read("word/document.xml").decode()
    assert main.index("Figure 1. NUDT5") < main.index("Figure 2. NUDT14") < main.index("Figure 3A.")
    assert "Figure S1." not in main and "Figure S1." in supplement
    assert "Figure S2. Property" in supplement and "Figure 2. Property" not in supplement
    assert "<w:tblHeader" in main and "<w:tblHeader" in supplement
    assert "<w:cantSplit" in main and "<w:cantSplit" in supplement
    assert "path-b:abstract" not in main and "Generated from frozen inputs" not in main
    manifest = json.loads((output / "documents-manifest.json").read_text())
    assert manifest["arguments"]["profile"] == "path-b"
    assert "frozen_inputs.json" in json.dumps(manifest["files"])
    for name, artifact in manifest["artifacts"].items():
        assert hashlib.sha256((output / name).read_bytes()).hexdigest() == artifact["sha256"]
    for name in (
        "repair-response.md",
        "repair-reviews_manifest.json",
        "diagnostic_reexpressions.json",
        "arg51_functional_group_minima.json",
        "withdrawn_historical_assertions.json",
        "final_hits.csv",
    ):
        assert name in manifest["artifacts"]
    assert (output / "repair-AI-R1-review.md").read_bytes() == (
        PACKAGE / "repair/AI-R1-review.md"
    ).read_bytes()
    assert (output / "final_hits.csv").read_bytes() == (ROOT / "final_hits.csv").read_bytes()
    assert "AAA CZ (occupancy 0) is outside all three" in main
    assert "Observed source-label inclusion" in supplement
    assert "Pairs pooled / within" in main
    scores = json.loads((output / "diagnostic_control_sources.json").read_text())
    assert len(scores["rows"]) == 45 and len(scores["methods"]) == 16
    assert (output / "laboratory_specification.md").read_bytes() == (
        PACKAGE / "laboratory_specification.md"
    ).read_bytes()


def test_path_b_cannot_opt_out_of_structural_requirements(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Required structural"):
        build(
            ROOT / "research/manuscript.md",
            ROOT / "research/results",
            tmp_path / "out",
            profile="path-b",
            require_structure=False,
        )
    assert not (tmp_path / "out").exists()


def test_path_b_late_failure_does_not_publish(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import build_research_documents

    def fail(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("render failure")

    monkeypatch.setattr(build_research_documents, "render_document", fail)
    with pytest.raises(RuntimeError, match="render failure"):
        build_path_b(tmp_path / "out")
    assert not (tmp_path / "out").exists()
    assert not list(tmp_path.glob(".nudt5-documents-*"))


def test_archive_hashes_and_gap_ids_are_preserved() -> None:
    import csv

    archive = PACKAGE / "audit_archive"
    manifest = json.loads((archive / "archive_manifest.json").read_text())
    for name, digest in manifest["files"].items():
        assert hashlib.sha256((archive / name).read_bytes()).hexdigest() == digest
    rows = list(csv.DictReader((archive / "unified_claim_table.csv").open()))
    assert len(rows) == 1041
    assert {r["#"] for r in rows if r["Status"] == "must be cut"} == {
        "C-0100",
        "C-0167",
        "C-0327",
        "C-0471",
    }
    gaps = (PACKAGE / "current_gaps.md").read_text()
    for row in csv.DictReader((archive / "material_gap_register.csv").open()):
        assert next(iter(row.values())) in gaps
    assert (
        "OUT-OF-SCOPE" in gaps
        and "usual failure point"
        not in (PACKAGE / "laboratory_specification.md").read_text().split("## 1. Question")[1]
    )
