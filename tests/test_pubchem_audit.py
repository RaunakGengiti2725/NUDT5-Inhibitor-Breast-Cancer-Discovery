import csv
import json
import shutil
from pathlib import Path

import pytest
from build_pubchem_audit import build_audit, micromolar, validated_snapshots
from pipeline import ROOT, write_json

SNAPSHOTS = ROOT / "research/external/pubchem"
LEDGER = ROOT / "research/external/observed_database_rows.csv"


def test_frozen_pubchem_crosscheck_and_censoring() -> None:
    actual = build_audit(SNAPSHOTS, LEDGER)
    assert actual == json.loads((SNAPSHOTS / "audit.json").read_text())
    assert actual["counts"] == {
        "gene_query_aids": 13,
        "accession_query_aids": 9,
        "union_aids": 21,
        "concise_aids": 7,
        "concise_rows_matched": 26,
        "distinct_cids_matched": 16,
        "ledger_strict_greater_than_rows": 6,
        "missing_numeric_values": 2,
    }
    record = next(r for r in actual["activity_matches"] if r["aid"] == 1440791)
    assert record["pubchem_concise_value_uM"] == "100.0000000000"
    assert record["ledger_relation_not_provided_by_concise"] == ">"


@pytest.mark.parametrize("value", ["NaN", "Infinity", "-Infinity", "0", "-1", "not a number"])
def test_invalid_values_refused(value: str) -> None:
    with pytest.raises(ValueError, match="Positive finite"):
        micromolar(value, "uM")


def test_units_and_missingness() -> None:
    assert micromolar("1000", "nM") == micromolar("1", "uM")
    assert micromolar("", "") is None
    with pytest.raises(ValueError, match="Unsupported"):
        micromolar("1", "mg/mL")


def test_tampered_snapshot_refused(tmp_path: Path) -> None:
    shutil.copytree(SNAPSHOTS, tmp_path / "snapshot")
    (tmp_path / "snapshot/geneid-aids.json").write_text("{}")
    with pytest.raises(ValueError, match="provenance mismatch"):
        validated_snapshots(tmp_path / "snapshot")


@pytest.mark.parametrize("mutation", ["duplicate", "value", "endpoint", "identity"])
def test_ambiguous_or_changed_ledger_refused(tmp_path: Path, mutation: str) -> None:
    with LEDGER.open() as handle:
        rows = list(csv.DictReader(handle))
    row = next(r for r in rows if r["record_id"] == "18034931")
    if mutation == "duplicate":
        rows.append(dict(row))
    elif mutation == "value":
        row["value"] = "99999"
    elif mutation == "endpoint":
        row["standard_endpoint"] = "Kd"
    else:
        row["inchikey"] = "UNMATCHED"
    ledger = tmp_path / "ledger.csv"
    with ledger.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError, match="One-to-one"):
        build_audit(SNAPSHOTS, ledger)


def test_report_is_deterministic_and_nonoverwriting(tmp_path: Path) -> None:
    report = build_audit(SNAPSHOTS, LEDGER)
    first, second = tmp_path / "first.json", tmp_path / "second.json"
    write_json(first, report)
    write_json(second, build_audit(SNAPSHOTS, LEDGER))
    assert first.read_bytes() == second.read_bytes()
    with pytest.raises(FileExistsError):
        write_json(first, report)
