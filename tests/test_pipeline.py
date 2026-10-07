from __future__ import annotations

import csv
import itertools
import json
import math
import subprocess
import sys
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
from pipeline import (
    ROOT,
    bedroc,
    candidate_audit,
    compute_props,
    consensus_scores,
    deduplicate,
    enrichment_factor,
    main,
    metrics,
    molecule,
    permutation_pvalue,
    read_compounds,
    scaffold_splits,
    smiles_to_fp,
)
from rdkit.ML.Scoring.Scoring import CalcBEDROC


def write_compounds(path: Path, rows: list[list[str]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "smiles", "label"])
        writer.writerows(rows)


def test_source_dataset_counts_invalid_and_overlap() -> None:
    records, issues = read_compounds(ROOT / "compounds.csv")
    candidates, candidate_issues = read_compounds(ROOT / "final_hits.csv", labelled=False)
    assert len(records) == 45
    assert sum(record.label == 1 for record in records) == 19
    assert [row["id"] for row in issues] == ["ACT-18"]
    assert not candidate_issues
    audited = {row["id"]: row for row in candidate_audit(records, candidates)}
    assert audited["NC5-02"]["training_identity_matches"] == ["ACT-19"]
    assert audited["NC5-02"]["max_training_tanimoto"] == 1
    assert audited["NC5-10"]["veber_pass"] is False


def test_explicit_validation_and_canonical_identity(tmp_path: Path) -> None:
    path = tmp_path / "molecules.csv"
    write_compounds(
        path,
        [
            ["A", "CCO", "1"],
            ["B", "OCC", "1"],
            ["C", "", "0"],
            ["D", "CCC", "invalid"],
            ["A", "CC", "0"],
        ],
    )
    records, issues = read_compounds(path)
    assert len(issues) == 3
    unique, duplicates = deduplicate(records)
    assert len(unique) == 1
    assert duplicates[0]["ids"] == ["A", "B"]
    with pytest.raises(ValueError, match="Conflicting"):
        deduplicate([records[0], replace(records[1], label=0)])


def test_schema_and_empty_input(tmp_path: Path) -> None:
    path = tmp_path / "bad.csv"
    path.write_text("id,wrong\nX,CCC\n")
    with pytest.raises(ValueError, match="required columns"):
        read_compounds(path)
    write_compounds(path, [])
    with pytest.raises(ValueError, match="no valid records"):
        read_compounds(path)
    path.write_text("id,smiles,label\nA,CCC,0,extra\nB,CCO,1\n")
    records, issues = read_compounds(path)
    assert len(records) == len(issues) == 1


@pytest.mark.parametrize("smiles", ["", "not-a-smiles", "C(C)(C)(C)(C)C"])
def test_invalid_molecules_raise(smiles: str) -> None:
    with pytest.raises(ValueError):
        molecule(smiles)
    with pytest.raises(ValueError):
        smiles_to_fp(smiles)
    with pytest.raises(ValueError):
        compute_props(smiles)


def test_fingerprint_and_properties() -> None:
    assert smiles_to_fp("CCO").shape == (2048,)
    np.testing.assert_array_equal(smiles_to_fp("CCO"), smiles_to_fp("OCC"))
    assert compute_props("CCO")["mw"] == pytest.approx(46.069)
    assert compute_props("CCO")["fsp3"] == 1
    with pytest.raises(ValueError):
        smiles_to_fp("CC", nbits=0)


def test_bedroc_against_rdkit_for_every_small_ranking() -> None:
    for size in range(2, 10):
        for labels in itertools.product([0, 1], repeat=size):
            if min(labels) == max(labels):
                continue
            scores = np.arange(size, 0, -1)
            reference = cast(Callable[[list[list[int]], int, float], float], CalcBEDROC)
            expected = reference([[label] for label in labels], 0, 20)
            assert bedroc(labels, scores) == pytest.approx(expected, abs=1e-12)


def test_rank_metric_ties_are_order_invariant() -> None:
    labels = np.array([1, 0, 0, 1])
    tied = np.ones(4)
    assert enrichment_factor(labels, tied) == 1
    expected = np.mean([bedroc(order, [4, 3, 2, 1]) for order in itertools.permutations(labels)])
    assert bedroc(labels, tied) == pytest.approx(expected)
    assert bedroc(labels[::-1], tied) == pytest.approx(expected)
    assert enrichment_factor([1, 0, 1, 0], [2, 2, 1, 0], 0.25) == 1


def test_metric_endpoints_and_finite_resolution() -> None:
    assert bedroc([1, 1, 0, 0], [4, 3, 2, 1]) == pytest.approx(1)
    assert bedroc([0, 0, 1, 1], [4, 3, 2, 1]) == pytest.approx(0)
    assert enrichment_factor([1, 0, 0], [3, 2, 1], 1) == 1
    values = metrics([1, 0, 1], [3, 2, 1])
    assert values["ef_1pct_k"] == 1
    assert permutation_pvalue(1, [0] * 30) == pytest.approx(1 / 31)
    assert permutation_pvalue(1, [1] * 5) == 1


@pytest.mark.parametrize(
    ("labels", "scores"),
    [
        ([], []),
        ([1], [1]),
        ([0, 0], [1, 0]),
        ([1, 0], [1]),
        ([1, 0], [math.nan, 0]),
        ([0.5, 0], [1, 0]),
        ([[1, 0]], [[1, 0]]),
    ],
)
def test_invalid_ranking_inputs(labels: Any, scores: Any) -> None:
    for function in (bedroc, enrichment_factor, metrics):
        with pytest.raises(ValueError):
            function(labels, scores)


@pytest.mark.parametrize("value", [0, -1, math.nan, math.inf])
def test_invalid_metric_parameters(value: float) -> None:
    with pytest.raises(ValueError):
        bedroc([1, 0], [1, 0], value)
    with pytest.raises(ValueError):
        enrichment_factor([1, 0], [1, 0], value)
    with pytest.raises(ValueError):
        permutation_pvalue(1, [])


def test_consensus_fixed_mean_and_batch_independence() -> None:
    result = consensus_scores({"A": [1, 0], "B": [0, 1]}, {"A": 3, "B": 1})
    np.testing.assert_allclose(result, [0.75, 0.25])
    individual = consensus_scores({"A": [1], "B": [0]}, {"A": 3, "B": 1})
    assert result[0] == individual[0]
    np.testing.assert_allclose(consensus_scores({"A": [0.5, 0.5], "B": [0.5, 0.5]}), [0.5, 0.5])


@pytest.mark.parametrize(
    "scores,weights",
    [
        ({}, None),
        ({"A": []}, None),
        ({"A": [2]}, None),
        ({"A": [math.nan]}, None),
        ({"A": [1], "B": [1, 0]}, None),
        ({"A": [1]}, {"A": -1}),
        ({"A": [1]}, {"A": 0}),
        ({"A": [1]}, {"B": 1}),
        ({"A": [1]}, {"A": math.inf}),
    ],
)
def test_invalid_consensus(scores: Any, weights: Any) -> None:
    with pytest.raises(ValueError):
        consensus_scores(scores, weights)


def test_scaffold_splits_have_no_leakage_and_cover_every_row() -> None:
    records, _ = read_compounds(ROOT / "compounds.csv")
    labels = np.array([record.label for record in records], dtype=np.int64)
    groups = [record.scaffold for record in records]
    splits = scaffold_splits(labels, groups, 5, 42)
    indices: list[int] = []
    for train, test in splits:
        assert not {groups[index] for index in train} & {groups[index] for index in test}
        assert set(labels[train]) == {0, 1}
        indices.extend(test)
    assert sorted(indices) == list(range(45))
    for _train, test in scaffold_splits(labels, groups, 5, 42):
        np.testing.assert_array_equal(splits.pop(0)[1], test)
    with pytest.raises(ValueError):
        scaffold_splits(labels, ["one"] * len(labels), 5, 42)


def test_audit_cli_preserves_sources_and_refuses_overwrite(tmp_path: Path) -> None:
    before = (ROOT / "compounds.csv").read_bytes()
    output = tmp_path / "audit"
    assert main(["audit", "--output", str(output)]) == 0
    report = json.loads((output / "audit.json").read_text())
    assert report["unique_records"] == 45
    assert len(report["invalid_records"]) == 1
    assert main(["audit", "--output", str(output)]) == 2
    assert (ROOT / "compounds.csv").read_bytes() == before
    assert "files" in json.loads((output / "manifest.json").read_text())


def test_benchmark_requires_acknowledgement_and_invalid_opt_in(tmp_path: Path) -> None:
    output = tmp_path / "blocked"
    assert main(["benchmark", "--output", str(output)]) == 2
    assert main(["benchmark", "--output", str(output), "--acknowledge-unverified-labels"]) == 2
    assert not output.exists()


def test_cli_from_other_working_directory_and_import_has_no_io(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/scripts/pipeline.py"),
            "audit",
            "--output",
            str(tmp_path / "run"),
        ],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    path = str(ROOT / "scripts/scripts")
    subprocess.run(
        [sys.executable, "-c", f"import sys; sys.path.insert(0, {path!r}); import pipeline"],
        cwd=tmp_path,
        check=True,
    )
    assert sorted(p.name for p in tmp_path.iterdir()) == ["run"]


def test_series_split_is_metadata_based_not_positional() -> None:
    from pipeline import series_splits

    records, _ = read_compounds(ROOT / "compounds.csv")
    records, _ = deduplicate(records)
    splits = series_splits(records, 42)
    assert len(splits) == 2
    tested: list[int] = []
    for train, test in splits:
        train_series = {records[i].source_row["series"] for i in train if records[i].label == 1}
        test_series = {records[i].source_row["series"] for i in test if records[i].label == 1}
        assert not train_series & test_series
        assert len(test_series) == 1
        assert {records[i].label for i in test} == {0, 1}
        tested.extend(test)
    assert sorted(tested) == list(range(45))
    with pytest.raises(ValueError, match="positive-series"):
        series_splits(
            [replace(r, source_row={**r.source_row, "series": "single"}) for r in records], 42
        )


def test_cluster_resampling_discloses_fixed_predictions() -> None:
    from pipeline import auc_resampling

    result = auc_resampling(
        np.array([1, 0, 1, 0]),
        np.array([0.9, 0.1, 0.8, 0.2]),
        ["a", "b", "c", "d"],
        seed=42,
        draws=100,
    )
    assert result["conditional_auc_percentile_95"] == [1, 1]
    assert result["valid_draws"] + result["one_class_draws"] == 100
    assert result["model_refitted"] is False
    with pytest.raises(ValueError):
        auc_resampling(np.array([1, 0]), np.array([1.0, 0.0]), [], 42)


def test_fold_metrics_join_on_id_not_input_order() -> None:
    from pipeline import fold_statistics

    records, _ = read_compounds(ROOT / "compounds.csv")
    selection = [records[0], records[-1], records[-2]]
    assignments = [
        {"id": selection[2].identifier, "fold": 1},
        {"id": selection[1].identifier, "fold": 0},
        {"id": selection[0].identifier, "fold": 0},
    ]
    result = fold_statistics(selection, assignments, {"method": np.array([0.9, 0.1, 0.2])})
    assert result["0"]["method"]["auc"] == 1
    assert result["1"]["auc"] is None
    assert result["1"]["n"] == 1


def test_oof_predictions_deterministic_without_train_test_overlap() -> None:
    from pipeline import out_of_fold

    records, _ = read_compounds(ROOT / "compounds.csv")
    records, _ = deduplicate(records)
    first, assignments = out_of_fold(records, split="series", folds=5, seed=42)
    second, repeated = out_of_fold(records, split="series", folds=5, seed=42)
    assert assignments == repeated
    assert len(assignments) == len(records)
    for method, values in first.items():
        np.testing.assert_array_equal(values, second[method])
        assert np.isfinite(values).all()
        assert ((values >= 0) & (values <= 1)).all()


def test_permutation_diagnostic_has_nonzero_finite_pvalues() -> None:
    from pipeline import out_of_fold, randomized_label_diagnostic

    records, _ = read_compounds(ROOT / "compounds.csv")
    records, _ = deduplicate(records)
    scores, _ = out_of_fold(records, split="molecule", folds=2, seed=42)
    labels = np.array([record.label for record in records])
    observed = {name: float(metrics(labels, values)["auc"]) for name, values in scores.items()}
    result = randomized_label_diagnostic(records, observed, permutations=2, folds=2, seed=42)
    assert result["minimum_pvalue"] == 1 / 3
    for method in result["methods"].values():
        assert len(method["null_auc"]) == 2
        assert method["p_plus_one"] >= 1 / 3


def test_json_writer_never_replaces_an_existing_result(tmp_path: Path) -> None:
    from pipeline import write_json

    path = tmp_path / "output.json"
    write_json(path, {"original": True})
    with pytest.raises(FileExistsError):
        write_json(path, {"replacement": True})
    assert json.loads(path.read_text()) == {"original": True}


def test_candidate_audit_distinguishes_active_and_all_training_reference() -> None:
    records, _ = read_compounds(ROOT / "compounds.csv")
    candidates, _ = read_compounds(ROOT / "final_hits.csv", labelled=False)
    results = {r["id"]: r for r in candidate_audit(records, candidates)}
    assert results["NC5-02"]["max_active_tanimoto"] == 1
    assert results["NC5-02"]["nearest_active_id"] == "ACT-19"
    assert sum(row["max_active_tanimoto"] >= 0.25 for row in results.values()) == 3
    assert results["NC5-06"]["max_active_tanimoto"] < 0.25
    assert "unverified" in results["NC5-06"]["status"]


def test_invalid_json_values_do_not_leave_a_partial_file(tmp_path: Path) -> None:
    from pipeline import write_json

    path = tmp_path / "output.json"
    with pytest.raises(ValueError):
        write_json(path, {"value": float("nan")})
    assert not path.exists()


def test_duplicate_column_names_are_rejected_before_any_record_is_used(tmp_path: Path) -> None:
    path = tmp_path / "duplicate.csv"
    for header in (
        "id,smiles,label,label",
        "id,smiles,smiles,label",
        "id,smiles,label,series,series",
    ):
        path.write_text(header + "\nX,CCO,0,1\n")
        with pytest.raises(ValueError, match="duplicate column names"):
            read_compounds(path)
    assert main(["audit", "--compounds", str(path), "--output", str(tmp_path / "out")]) == 2
    assert not (tmp_path / "out").exists()


def test_unterminated_quote_fails_closed_instead_of_swallowing_records(tmp_path: Path) -> None:
    path = tmp_path / "unclosed.csv"
    path.write_text('id,smiles,label,notes\nA,CCO,1,"unclosed\nB,CCC,0,note\n')
    with pytest.raises(ValueError, match="malformed CSV content"):
        read_compounds(path)
    assert main(["audit", "--compounds", str(path), "--output", str(tmp_path / "out")]) == 2
    assert not (tmp_path / "out").exists()
    path.write_text('id,smiles,label,notes\nA,CCO,1,"properly quoted\nmultiline"\nB,CCC,0,plain\n')
    records, issues = read_compounds(path)
    assert [record.identifier for record in records] == ["A", "B"]
    assert records[0].source_row["notes"] == "properly quoted\nmultiline"
    assert not issues


def test_atomic_json_failure_leaves_no_partial_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import os

    from pipeline import write_json

    def fail_publish(source: Path, target: Path) -> None:
        assert json.loads(source.read_text()) == {"complete": True}
        assert not target.exists()
        raise OSError("simulated publication failure")

    monkeypatch.setattr(os, "link", fail_publish)
    with pytest.raises(OSError, match="simulated"):
        write_json(tmp_path / "output.json", {"complete": True})
    assert not list(tmp_path.iterdir())
