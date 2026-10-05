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


def test_paired_table_matches_all_eligible_recorded_endpoints_and_scores() -> None:
    import math

    result = json.loads((ROOT / "research/results/selectivity.json").read_text())
    text = (ROOT / "research/manuscript.md").read_text()
    lines = text.split("Table 3. Same-paper")[1].split("Compound 9 has")[0].splitlines()
    cells = [
        [v.strip() for v in line.strip("|").split("|")] for line in lines if line.startswith("| ")
    ][1:]
    rows = {r["source_compound"]: r for r in result["rows"]}
    scores = {(r["source_compound"], r["scenario"]): r for r in result["score_rows"]}
    scenarios = ["historical_original_graphs", "stored_authenticated_reference_sensitivity"]
    assert [c[0] for c in cells] == result["summary"]["scenarios"][scenarios[0]]["eligible_ids"]
    assert len(cells) == 6
    for name, nudt5, nudt14, ratio, historical, reference in cells:
        row = rows[name]
        for target, printed in (("NUDT5", nudt5), ("NUDT14", nudt14)):
            endpoint = row["endpoints"][target]
            if endpoint["status"] == "right_censored":
                assert printed == f">{endpoint['bound']:g}"
            else:
                assert float(printed) == endpoint["reported_mean"]
        value = row["ratio"]
        if value["status"] == "double_censored":
            assert ratio == "Not estimable"
            assert value["point"] is None and value["bound"] is None
        elif value["status"] == "upper_bound":
            assert ratio.startswith("<")
            assert math.isclose(float(ratio[1:]), value["bound"], abs_tol=5e-7)
        else:
            assert math.isclose(float(ratio), value["point"], abs_tol=5e-7)
        for scenario, printed in zip(scenarios, (historical, reference), strict=True):
            assert math.isclose(
                float(printed), scores[(name, scenario)]["Equal_mean"], abs_tol=5e-7
            )


def test_new_cli_registration_includes_both_modules() -> None:
    import tomllib

    config = tomllib.loads((ROOT / "pyproject.toml").read_text())
    for name in ("selectivity", "assay"):
        assert config["project"]["scripts"][f"nudt5-{name}"] == f"{name}:main"
        assert name in config["tool"]["setuptools"]["py-modules"]


def test_portable_inventory_excludes_canonical_previous_inventory(tmp_path: Path) -> None:
    research = tmp_path / "research"
    research.mkdir()
    (research / "release_manifest.json").write_text('{"previous": true}')
    (research / "evidence.md").write_text("Evidence boundaries")
    first = build_manifest(tmp_path, tmp_path / "inventory-a.json")
    second = build_manifest(tmp_path, tmp_path / "inventory-b.json")
    assert first == second
    assert [row["filename"] for row in first["files"]] == ["research/evidence.md"]
