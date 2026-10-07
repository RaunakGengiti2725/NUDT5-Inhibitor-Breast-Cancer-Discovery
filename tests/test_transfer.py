from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import pytest
from pipeline import ROOT, deduplicate, read_compounds
from transfer import (
    identity,
    identity_matches,
    main,
    measured_challenge,
    potency,
    probe_challenge,
    read_source_csv,
)


def test_identity_preserves_stereo_and_flags_parent_equivalence() -> None:
    left, right = identity("N[C@H](C)C(=O)O"), identity("N[C@@H](C)C(=O)O")
    assert left["inchikey"] != right["inchikey"]
    assert left["canonical_smiles"] != right["canonical_smiles"]
    acid = identity("CC(=O)O")
    salt = identity("CC(=O)[O-].[Na+]")
    assert salt["fragment_count"] == 2
    assert salt["neutral_fragment_parent"] == acid["neutral_fragment_parent"]
    matches = identity_matches(salt, [{"id": "acid", **acid}])
    assert matches["canonical_smiles"] == []
    assert matches["neutral_fragment_parent"] == ["acid"]
    assert identity("CC(O)C(=O)O")["unspecified_tetrahedral_centers"] == 1


@pytest.mark.parametrize("bad", ["", "not-a-smiles"])
def test_invalid_identity_fails(bad: str) -> None:
    with pytest.raises(ValueError):
        identity(bad)


def test_potency_never_equates_untested_with_inactive() -> None:
    assert potency("not tested") == ("untested", None)
    assert potency("not active") == ("inactive_gt50_uM", None)
    assert potency("2.04 ± 0.240") == ("numeric_ic50_uM", 2.04)
    for invalid in ("-1", "nan", "0"):
        with pytest.raises(ValueError):
            potency(invalid)


@pytest.fixture(scope="module")
def biochemical_challenge() -> dict[str, Any]:
    records, _ = read_compounds(ROOT / "compounds.csv")
    records, _ = deduplicate(records)
    return measured_challenge(records, ROOT / "research/source_assays.csv", 42)


def test_measured_challenge_has_no_overlap_or_untested_negatives(
    biochemical_challenge: dict[str, Any],
) -> None:
    result = biochemical_challenge
    assert len(result["rows"]) == 23
    assert len(result["eligible_ids"]) == 10
    assert set(result["eligible_ids"]).isdisjoint({"10", "11"})
    for row in result["rows"]:
        if row["id"] in result["eligible_ids"]:
            assert not any(row["training_matches"].values())
            assert row["assay_status"] != "untested"
        if row["id"] in {"10", "11"}:
            assert row["exclusion_reason"] == "training_identity_or_parent_overlap"
    assert result["threshold_sensitivity_uM"]["50.0"]["positives"] == 5
    assert result["threshold_sensitivity_uM"]["1.0"]["positives"] == 2


def test_all_measured_threshold_scores_are_unchanged(biochemical_challenge: dict[str, Any]) -> None:
    import numpy as np
    from controls import extended_metrics

    result = biochemical_challenge
    rows = [r for r in result["rows"] if r["id"] in result["eligible_ids"]]
    scores = np.array([r["scores"]["Equal_mean"] for r in rows])
    for cutoff, entry in result["threshold_sensitivity_uM"].items():
        labels = np.array(
            [int(r["ic50_uM"] is not None and r["ic50_uM"] < float(cutoff)) for r in rows]
        )
        assert entry["metrics"]["Equal_mean"] == extended_metrics(labels, scores)


def test_probe_pair_reports_six_graphs_and_separate_measurements() -> None:
    records, _ = read_compounds(ROOT / "compounds.csv")
    records, _ = deduplicate(records)
    result = probe_challenge(records, ROOT / "research/external/nudt5_measured_ledger.csv", 42)
    assert len(result["rows"]) == 6
    assert len({row["id"] for row in result["rows"]}) == 6
    strong = next(r for r in result["rows"] if r["id"] == "MRK-952")
    weak = next(r for r in result["rows"] if r["id"] == "MRK-952-NC")
    assert len(strong["source_measurements"]) == len(weak["source_measurements"]) == 2
    assert not any(strong["training_matches"].values())
    for name in strong["scores"]:
        assert (
            result["MRK_pair_contrast"][name]["difference"]
            == strong["scores"][name] - weak["scores"][name]
        )


def test_source_csv_fails_closed_and_cli_does_not_write_without_ack(tmp_path: Path) -> None:
    path = tmp_path / "bad.csv"
    path.write_text("id,id\na,b\n")
    with pytest.raises(ValueError):
        read_source_csv(path)
    assert main(["--output", str(tmp_path / "out")]) == 2
    assert not (tmp_path / "out").exists()


def test_reference_substitution_is_separate_and_preserves_unaffected_records() -> None:
    from transfer import substitute_references

    original, _ = read_compounds(ROOT / "compounds.csv")
    original, _ = deduplicate(original)
    fixed, changes = substitute_references(original, ROOT / "research/reference_structures.csv")
    assert len(fixed) == len(original) == 45
    assert {row["id"] for row in changes} == {"ACT-01", "ACT-02"}
    for before, after in zip(original, fixed, strict=True):
        assert before.identifier == after.identifier
        assert before.label == after.label
        assert before.source_row == after.source_row
        if before.identifier in {"ACT-01", "ACT-02"}:
            assert before.canonical_smiles != after.canonical_smiles
        else:
            assert before == after
    th5427 = next(row for row in changes if row["id"] == "ACT-01")
    assert th5427["replacement"]["formula"] == "C20H20Cl2N8O3"
    assert th5427["original"]["formula"] == "C19H18Cl2N8O3"


def test_missing_source_columns_and_unmatched_reference_fail(tmp_path: Path) -> None:
    from transfer import substitute_references

    path = tmp_path / "source.csv"
    path.write_text("training_id,source_smiles,source_url\nBAD,CC,https://example.org\n")
    with pytest.raises(ValueError, match="requires columns"):
        read_source_csv(path, ("compound_name",))
    records, _ = read_compounds(ROOT / "compounds.csv")
    with pytest.raises(ValueError, match="must all occur"):
        substitute_references(records, path)


@pytest.mark.parametrize("defect", ["duplicate_id", "conflicting_alias", "censor_bound"])
def test_corrupted_source_cannot_reweight_or_relabel_measurements(
    tmp_path: Path, defect: str
) -> None:
    rows = read_source_csv(ROOT / "research/source_assays.csv")
    if defect == "duplicate_id":
        rows.append(dict(rows[3]))
    elif defect == "conflicting_alias":
        rows.append(
            {**rows[0], "source_compound": "alias", "nudt5_ic50_uM_as_reported": "not active"}
        )
    else:
        for row in rows:
            if row["nudt5_ic50_uM_as_reported"] == "not active":
                row["inactive_definition"] = "IC50 >10 µM; not tested is distinct from inactive"
    path = tmp_path / "altered.csv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    records, _ = read_compounds(ROOT / "compounds.csv")
    training, _ = deduplicate(records)
    with pytest.raises(ValueError, match="unique compound|censoring"):
        measured_challenge(training, path, 42)
