from __future__ import annotations

import csv
import gzip
import hashlib
import json
from typing import Any

import pytest
from build_path_b_documents import arg51_group_minima, arg51_group_text, quantitative_blocks
from build_research_documents import ROOT
from path_b_diagnostics import (
    auc_estimands,
    conformal_sparsity,
    diagnostic_reexpressions,
    group_resampling,
)

PACKAGE = ROOT / "research/path_b"


def frozen(name: str) -> dict[str, Any]:
    value: dict[str, Any] = json.loads((ROOT / "research/results" / name).read_text())
    return value


def test_auc_estimands_distinguish_between_model_and_within_model_pairs() -> None:
    rows = [{"label": y, "fold": fold} for fold in (0, 1) for y in (0, 1)]
    result = auc_estimands(rows, [0.8, 0.9, 0.1, 0.2])
    assert result == {
        "pooled_auc": 0.75,
        "within_fold_auc": 1.0,
        "pooled_pairs": 4,
        "within_fold_pairs": 2,
    }
    tied = auc_estimands(rows, [0.5] * 4)
    assert tied["pooled_auc"] == tied["within_fold_auc"] == 0.5
    one_class = auc_estimands([{"label": 1, "fold": 0}], [0.8])
    assert one_class["pooled_auc"] is None and one_class["within_fold_auc"] is None
    with pytest.raises(ValueError, match="align"):
        auc_estimands(rows, [0.1])
    with pytest.raises(ValueError, match="align"):
        auc_estimands([], [])


def test_frozen_estimands_sparsity_and_grouping_sensitivity() -> None:
    result = diagnostic_reexpressions(frozen("controls.json"))
    full = result["full_valid_set"]
    assert full["methods"]["Train_prevalence"]["within_fold_auc"] == 0.5
    assert full["methods"]["Equal_mean"]["pooled_pairs"] == 494
    assert full["methods"]["Equal_mean"]["within_fold_pairs"] == 87
    assert full["methods"]["Equal_mean"]["within_fold_auc"] == pytest.approx(84 / 87)
    assert full["conformal"]["levels"][0]["both_forced"] == 22
    similarity = result["similarity_component_split"]["conformal"]
    assert all(row["counts"]["1"] == 2 for row in similarity["calibration"])
    assert all(row["minimum_pvalue"]["1"] == 1 / 3 for row in similarity["calibration"])
    assert [row["set_size_counts"]["2"] for row in similarity["levels"]] == [43, 42]
    assert all(row["forced_inclusion"]["1"] == 45 for row in similarity["levels"])
    scaffold, component = result["similarity_split_grouping_sensitivity"]
    assert scaffold["percentile_range_95"] == pytest.approx([0.933870523415978, 1.0])
    assert component["percentile_range_95"] == pytest.approx([0.9285714285714286, 1.0])
    assert scaffold["single_class_draws"] == 0 and component["single_class_draws"] == 17
    assert not scaffold["model_refitted"] and not component["model_refitted"]


@pytest.mark.parametrize("mutation", ["minimum", "set"])
def test_conformal_corruption_is_rejected(mutation: str) -> None:
    conformal = frozen("controls.json")["evaluations"]["full_valid_set"]["conformal"]
    if mutation == "minimum":
        conformal["assignments"][0]["minimum_pvalue"]["1"] = 0.9
    else:
        conformal["predictions"][0]["sets"]["0.1"] = [99]
    with pytest.raises(ValueError, match="Conformal"):
        conformal_sparsity(conformal)


def test_single_class_and_empty_group_resampling() -> None:
    row = {"label": 1, "scores": {"Equal_mean": 0.8}, "group": "a"}
    result = group_resampling([row], "group", draws=10)
    assert result["percentile_range_95"] is None
    assert result["single_class_draws"] == 10
    with pytest.raises(ValueError, match="groups"):
        group_resampling([], "group")


def test_report_fields_ratios_and_functional_group_radii() -> None:
    structure = json.loads(
        (ROOT / "research/structure_comparison/results/observed_proximity.json").read_text()
    )
    support = json.loads(
        (
            ROOT / "research/structure_comparison/model_support/results/model_support.json"
        ).read_text()
    )
    blocks = quantitative_blocks(
        structure, frozen("selectivity.json"), support, frozen("controls.json")
    )
    assert "EDIAm" in blocks["sites"] and "OPIA (%)" in blocks["sites"]
    assert blocks["sites"].count("| 8RIY |") == 4
    assert blocks["sites"].count("| 8OTV |") == 2
    assert "0.199" in blocks["sites"] and "45.450" in blocks["sites"]
    assert "1.182796" not in blocks["paired"] and "1.18" in blocks["paired"]
    assert "494 / 87" in blocks["controls"]
    minima = arg51_group_minima(structure)
    assert [r["distance_A"] for r in minima] == pytest.approx(
        [3.7555613429, 4.0010894766, 4.7617397031, 3.9465373430, 3.2512725201, 4.6063983762]
    )
    assert minima[4]["occupancy"] == 0.78
    assert "AAA aliphatic side chain | No | No | Yes | Yes" in arg51_group_text(structure)
    assert "AAA guanidinium | No | No | No | Yes" in arg51_group_text(structure)
    structure["atom_pairs_within_5A"] = []
    with pytest.raises(ValueError, match="both chains"):
        arg51_group_minima(structure)


def test_review_archives_definitions_and_withdrawal_provenance() -> None:
    repair = PACKAGE / "repair"
    manifest = json.loads((repair / "reviews_manifest.json").read_text())
    assert manifest["reviewed_commit"] == "22830642f7d33ca3319872b5ba4c513baff50154"
    response = (repair / "response.md").read_text()
    for name, entry in manifest["files"].items():
        assert hashlib.sha256((repair / name).read_bytes()).hexdigest() == entry["sha256"]
    for review, heading, next_heading in [
        (1, "## Review 1", "## Review 2"),
        (2, "## Review 2", "## Review 3"),
        (3, "## Review 3", "## Objections"),
    ]:
        section = response.split(heading)[1].split(next_heading)[0]
        for objection in json.loads((repair / f"AI-R{review}-objections.json").read_text())[
            "objections"
        ]:
            assert f"| {objection['id']} |" in section
    for entry in json.loads((repair / "definitions_sources.json").read_text()).values():
        if "archive" in entry:
            payload = (repair / entry["archive"]).read_bytes()
            assert hashlib.sha256(payload).hexdigest() == entry["archive_sha256"]
            assert hashlib.sha256(gzip.decompress(payload)).hexdigest() == entry["sha256"]
    withdrawal = json.loads((PACKAGE / "withdrawn_historical_assertions.json").read_text())
    source = ROOT / withdrawal["source_file"]["path"]
    assert hashlib.sha256(source.read_bytes()).hexdigest() == withdrawal["source_file"]["sha256"]
    with source.open(newline="") as handle:
        assert [r["id"] for r in csv.DictReader(handle)] == [r["id"] for r in withdrawal["rows"]]
    assert len(withdrawal["rows"]) == 10
    assert all(r["status"] == "withdrawn_historical_assertion" for r in withdrawal["rows"])
