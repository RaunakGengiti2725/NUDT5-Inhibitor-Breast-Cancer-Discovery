"""Scientific-record regression checks, not biological validation."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pipeline
import pytest
import transfer
from regenerate_identity import EXPECTED, INPUTS, build

ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture(scope="module")
def evidence() -> dict[str, Any]:
    return build(ROOT)


def test_all_ten_original_candidates_and_only_one_bounded_overlap(evidence: dict[str, Any]) -> None:
    assert [r["id"] for r in evidence["candidates"]] == [f"NC5-{i:02d}" for i in range(1, 11)]
    for pool in [
        "original_training",
        "balikci_23",
        "archived_database_records",
    ]:
        assert all(v == ["NC5-02"] for v in evidence["candidate_overlap_sets"][pool].values())
    for pool in ["authenticated_ccd", "archived_measured_ledger"]:
        assert all(v == [] for v in evidence["candidate_overlap_sets"][pool].values())
    hit = evidence["candidates"][1]
    assert hit["matches"]["original_training"]["canonical_smiles"] == ["ACT-19"]
    assert hit["matches"]["balikci_23"]["canonical_smiles"] == ["11"]
    assert hit["matches"]["archived_pubchem_cids"]["canonical_smiles"] == []
    assert hit["matches"]["archived_pubchem_cids"]["neutral_fragment_parent"] == []
    assert hit["matches"]["archived_pubchem_cids"]["canonical_parent_tautomer"] == ["22346757"]
    assert evidence["candidate_overlap_sets"]["archived_pubchem_cids"] == {
        "canonical_smiles": [],
        "neutral_fragment_parent": [],
        "canonical_parent_tautomer": ["NC5-02"],
    }


def test_invalid_act18_is_preserved_without_a_replacement(evidence: dict[str, Any]) -> None:
    assert [r["id"] for r in evidence["invalid_training"]] == ["ACT-18"]
    assert len(evidence["pools"]["original_training"]) == 45
    assert evidence["invalid_candidates"] == []
    assert evidence["candidate_duplicates"] == []
    assert evidence["training_duplicates"] == []
    assert evidence["input_sha256"] | EXPECTED == evidence["input_sha256"]


def test_four_named_graphs_are_distinct_and_958_synonym_is_not_identity(
    evidence: dict[str, Any],
) -> None:
    refs = {r["id"]: r for r in evidence["pools"]["authenticated_ccd"]}
    hit = evidence["candidates"][1]
    assert len({r["inchikey"] for r in [*refs.values(), hit]}) == 4
    assert refs["958"]["ccd_synonyms"] == "TH5427"
    assert refs["958"]["authenticated_name"] == "TH1713"
    assert refs["958"]["formula"] == "C19H21N7O3"
    assert refs["9CH"]["formula"] == "C20H20Cl2N8O3"
    assert refs["W0O"]["formula"] == "C23H24N6O"
    assert refs["W0O"]["pdb_accessions"] == ["8RIY", "8OTV"]
    sources = {r["id"]: r for r in evidence["pools"]["balikci_23"]}
    assert refs["W0O"]["canonical_smiles"] == sources["9"]["canonical_smiles"]
    assert refs["9CH"]["canonical_smiles"] == sources["6"]["canonical_smiles"]
    assert hit["canonical_smiles"] == sources["11"]["canonical_smiles"]


def test_original_reference_graphs_are_not_authenticated_references(
    evidence: dict[str, Any],
) -> None:
    changes = evidence["reference_substitutions_identity_only"]
    assert {r["id"] for r in changes} == {"ACT-01", "ACT-02"}
    for r in changes:
        assert r["original"]["canonical_smiles"] != r["replacement"]["canonical_smiles"]
    assert evidence["candidates"][0]["authenticated_TH5427_tanimoto"] == pytest.approx(32 / 83)
    overlaps = evidence["source_to_training"]
    assert {r["source_compound"] for r in overlaps if any(r["original_training"].values())} == {
        "10",
        "11",
    }
    assert {
        r["source_compound"]
        for r in overlaps
        if any(r["authenticated_reference_sensitivity"].values())
    } == {"6", "10", "11"}


def test_existing_audit_and_fresh_summary_agree(evidence: dict[str, Any]) -> None:
    audit = json.loads((ROOT / "research/results/audit.json").read_text())
    assert [r["original_candidate_audit"] for r in evidence["candidates"]] == audit["candidates"]
    assert evidence["invalid_training"] == audit["invalid_records"]
    summary = json.loads((Path(__file__).parent / "identity_summary.json").read_text())
    assert evidence == summary
    assert evidence["pubchem_linkage"] == json.loads(
        (ROOT / "research/external/pubchem/audit.json").read_text()
    )


def test_no_fitting_is_called(monkeypatch: pytest.MonkeyPatch) -> None:
    def prohibited(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("Model fitting is prohibited in this source check")

    monkeypatch.setattr(pipeline, "fit_scores", prohibited)
    monkeypatch.setattr(transfer, "fit_scores", prohibited)
    assert len(build(ROOT)["candidates"]) == 10


@pytest.mark.parametrize("input_name", list(EXPECTED))
def test_changed_original_bytes_fail_closed(tmp_path: Path, input_name: str) -> None:
    for name in INPUTS:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, path)
    with (tmp_path / input_name).open("a") as handle:
        handle.write("\n")
    with pytest.raises(ValueError, match="Original input hash mismatch"):
        build(tmp_path)
