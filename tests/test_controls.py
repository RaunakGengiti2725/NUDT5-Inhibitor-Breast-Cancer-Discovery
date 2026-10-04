from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from controls import (
    balance_report,
    calibration_bins,
    conformal_pvalues,
    conformal_sets,
    evaluate,
    extended_metrics,
    labels_of,
    main,
    match_property_pairs,
    paired_auc_interval,
    property_matrix,
    similarity_bins,
    similarity_components,
    standardize,
    tanimoto_matrix,
    univariate_descriptor_auc,
)
from pipeline import ROOT, read_compounds, scaffold_splits, smiles_to_fp
from sklearn.metrics import brier_score_loss, f1_score, matthews_corrcoef


def test_probability_metrics_and_undefined_precision() -> None:
    labels = np.array([0, 1, 0, 1])
    values = np.array([0.1, 0.8, 0.4, 0.6])
    result = extended_metrics(labels, values)
    assert result["auc"] == result["average_precision"] == 1
    assert result["brier"] == pytest.approx(brier_score_loss(labels, values))
    assert result["f1_at_0_5"] == f1_score(labels, values >= 0.5)
    assert result["mcc_at_0_5"] == matthews_corrcoef(labels, values >= 0.5)
    assert extended_metrics(labels, np.zeros(4))["precision_at_0_5"] is None
    assert extended_metrics(labels, np.ones(4))["npv_at_0_5"] is None
    with pytest.raises(ValueError):
        extended_metrics(labels, np.array([0.0, 1.0, 2.0, 0.5]))


def test_calibration_endpoints_and_complete_partition() -> None:
    labels = np.array([0, 1, 0, 1])
    values = np.array([0.0, 0.2, 0.8, 1.0])
    rows = calibration_bins(labels, values)
    assert sum(row["n"] for row in rows["bins"]) == 4
    assert rows["bins"][-1]["n"] == 2
    assert calibration_bins(labels, labels.astype(float))["expected_calibration_error"] == 0
    with pytest.raises(ValueError):
        calibration_bins(labels, values, 0)


def test_similarity_bins_include_every_boundary_without_undefined_auc() -> None:
    similarity = np.array([0.0, 0.3, 0.5, 0.7, 0.85, 1.0])
    labels = np.array([0, 0, 1, 0, 1, 0])
    rows = similarity_bins(labels, np.ones(6) * 0.5, similarity)
    assert sum(row["n"] for row in rows) == 6
    assert all(row["auc"] is None for row in rows[:-1])
    assert rows[-1]["auc"] == 0.5


def test_tanimoto_connected_components_bound_cross_group_similarity() -> None:
    features = np.array([[1.0, 1.0, 0.0], [1.0, 1.0, 1.0], [0.0, 1.0, 1.0], [0.0, 0.0, 1.0]])
    groups = similarity_components(features, 0.65)
    assert groups[0] == groups[1] == groups[2]
    assert groups[3] != groups[0]
    similarity = tanimoto_matrix(features, features)
    for i in range(4):
        for j in range(4):
            if groups[i] != groups[j]:
                assert similarity[i, j] < 0.65
    with pytest.raises(ValueError):
        similarity_components(features, 0)


def test_matching_deterministic_without_replacement_or_balance_promise() -> None:
    records, _ = read_compounds(ROOT / "compounds.csv")
    labels = labels_of(records)
    values = standardize(property_matrix(records))
    pairs = match_property_pairs(records, labels, values)
    assert len(pairs) == 19
    assert len({r["negative_index"] for r in pairs}) == len(pairs)
    assert pairs == match_property_pairs(records, labels, values)
    assert balance_report(values, labels, range(45))["positives"] == 19
    np.testing.assert_array_equal(standardize(np.ones((4, 7))), np.zeros((4, 7)))
    with pytest.raises(ValueError):
        standardize(np.array([]))


def test_univariate_report_is_descriptive_not_direction_tuned_oof() -> None:
    labels = np.array([0, 0, 1, 1])
    values = np.tile(np.arange(4)[:, None], (1, 7)).astype(float)
    for result in univariate_descriptor_auc(labels, values).values():
        assert result["auc_higher_is_positive"] == result["auc_best_direction"] == 1


def test_conformal_plus_one_ties_and_unseen_class() -> None:
    reference = np.array([0.1, 0.2, 0.2])
    assert conformal_pvalues(reference, 0.8) == pytest.approx(0.75)
    assert conformal_pvalues(reference, 0) == 0.25
    assert conformal_pvalues(reference, 1) == 1
    assert conformal_pvalues(np.array([], dtype=float), 0.5) == 1


def test_conformal_calibration_is_group_disjoint_and_sets_reconstruct() -> None:
    records, _ = read_compounds(ROOT / "compounds.csv")
    labels = labels_of(records)
    groups = [r.scaffold for r in records]
    features = np.stack([smiles_to_fp(r.smiles) for r in records])
    result = conformal_sets(features, labels, groups, scaffold_splits(labels, groups, 5, 42), 42)
    assert result["status"] == "computed"
    assert len(result["predictions"]) == len(records)
    for row in result["assignments"]:
        proper, calibrate, test = [
            set(row[key]) for key in ("proper_indices", "calibration_indices", "test_indices")
        ]
        assert not (proper & calibrate or proper & test or calibrate & test)
        assert not {groups[i] for i in proper} & {groups[i] for i in calibrate}
        for key in ("0", "1"):
            assert row["minimum_pvalue"][key] == 1 / (row["calibration_class_counts"][key] + 1)
    for row in result["predictions"]:
        for alpha in (0.1, 0.2):
            assert row["sets"][str(alpha)] == [c for c in (0, 1) if row["pvalues"][c] > alpha]
    for level in result["levels"]:
        alpha_key = str(level["alpha"])
        covered = [r["label"] in r["sets"][alpha_key] for r in result["predictions"]]
        assert level["empirical_coverage"] == np.mean(covered)


def test_paired_fixed_score_interval_preserves_pairing() -> None:
    labels = np.array([0, 1, 0, 1])
    scores = np.array([0.1, 0.9, 0.2, 0.8])
    result = paired_auc_interval(labels, scores, scores, ["a", "b", "c", "d"], 42, 100)
    assert result["auc_difference"] == 0
    assert result["conditional_percentile_95"] == [0, 0]


@pytest.fixture(scope="module")
def full_evaluation() -> dict[str, Any]:
    records, _ = read_compounds(ROOT / "compounds.csv")
    return evaluate(records, range(len(records)), 42, diagnostics=False)


def test_extended_oof_matches_original_and_contains_reproducible_assignments(
    full_evaluation: dict[str, Any],
) -> None:
    recorded = json.loads((ROOT / "research/results/benchmark.json").read_text())
    for method, expected in recorded["scaffold"]["metrics"].items():
        assert full_evaluation["methods"][method]["auc"] == expected["auc"]
    assert full_evaluation["methods"]["Constant_0_5"]["auc"] == 0.5
    assert {r["index"] for r in full_evaluation["predictions"]} == set(range(45))
    for row in full_evaluation["predictions"]:
        assert row["max_training_tanimoto"] >= row["max_active_tanimoto"]
    for fold in full_evaluation["folds"]:
        assert not fold["group_overlap"]
        assert not fold["scaffold_overlap"]


def test_controls_cli_requires_acknowledgement_and_preserves_outputs(tmp_path: Path) -> None:
    out = tmp_path / "out"
    assert main(["--output", str(out), "--allow-invalid"]) == 2
    assert not out.exists()
    out.mkdir()
    marker = out / "keep.txt"
    marker.write_text("original")
    assert main(["--output", str(out), "--allow-invalid", "--acknowledge-unverified-labels"]) == 2
    assert marker.read_text() == "original"
