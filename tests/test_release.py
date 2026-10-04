from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from build_release_manifest import build_manifest
from pipeline import ROOT


def test_inventory_hashes_inputs_and_refuses_overwrite(tmp_path: Path) -> None:
    source = tmp_path / "compounds.csv"
    source.write_text("original input")
    generated = tmp_path / "scripts" / "package.egg-info"
    generated.mkdir(parents=True)
    (generated / "PKG-INFO").write_text("Build metadata is not an evidence input")
    output = tmp_path / "inventory.json"
    result = build_manifest(tmp_path, output)
    assert len(result["files"]) == 1
    assert result["files"][0]["sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert result["files"][0]["size_bytes"] == source.stat().st_size
    with pytest.raises(FileExistsError):
        build_manifest(tmp_path, output)


def test_claim_ledger_uses_requested_status_vocabulary() -> None:
    import csv

    allowed = {
        "VERIFIED",
        "PARTIALLY VERIFIED",
        "UNVERIFIED",
        "UNSUPPORTED",
        "CONTRADICTED",
        "REQUIRES DATA",
        "REQUIRES EXPERIMENT",
    }
    rows = list(csv.DictReader((ROOT / "research/claim_ledger.csv").open()))
    assert len(rows) >= 24
    assert {r["verification_status"] for r in rows} <= allowed


def test_source_auc_table_matches_recorded_metrics() -> None:
    transfer = json.loads((ROOT / "research/results/transfer.json").read_text())
    text = (ROOT / "research/manuscript.md").read_text()
    lines = text.split("Table 2.")[1].split("The ordering")[0].splitlines()
    data = [line for line in lines if line.startswith("| ")][1:]
    methods = ["Property_LR", "RF", "SVM_RBF", "Nearest_active", "Equal_mean", "GBT"]
    assert len(data) == len(methods)
    for line, method in zip(data, methods, strict=True):
        values = [float(v.strip()) for v in line.split("|")[3:-1]]
        for value, cutoff in zip(values, ["1.0", "10.0", "50.0"], strict=True):
            recorded = transfer["measured_source_challenge"]["threshold_sensitivity_uM"][cutoff][
                "metrics"
            ][method]["auc"]
            assert abs(value - recorded) <= 0.0005


def test_compressed_authenticated_sources_preserve_original_hashes() -> None:
    import csv
    import gzip

    for row in csv.DictReader((ROOT / "research/reference_structures.csv").open()):
        name = row["source_url"].rsplit("/", 1)[-1]
        content = gzip.decompress((ROOT / "research/reference_sources" / f"{name}.gz").read_bytes())
        assert hashlib.sha256(content).hexdigest() == row["source_sha256"]
