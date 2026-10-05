"""SOFTWARE-ONLY rendering/table tests; not biological validation."""

from __future__ import annotations

import copy
import csv
import io
from pathlib import Path
from typing import Any

import build_structure_comparison_figures as figures
import pytest
import structure_comparison as mod
from PIL import Image
from pipeline import ROOT
from test_structure_comparison import CONTRACT, MANIFEST


@pytest.fixture(scope="module")
def result() -> dict[str, Any]:
    return mod.build_result(ROOT, MANIFEST, CONTRACT, ["SOFTWARE VERIFICATION"])


def test_tables_are_complete_and_lossless(result: dict[str, Any]) -> None:
    tables = figures.tables(figures.validate(result))
    assert tables == figures.tables(result)
    residues = list(csv.DictReader(io.StringIO(tables["residue_proximity.csv"].decode())))
    assert len(residues) == len(result["residue_proximity"]) == 1730
    for row, source in zip(residues, result["residue_proximity"], strict=True):
        distance = source["observed_min_distance_A"]
        assert row["observed_min_distance_A"] == ("" if distance is None else repr(distance))
        if distance is not None:
            assert float(row["observed_min_distance_A"]) == distance
        assert row["complete_residue_distance_A"] == ""
    pairs = list(csv.DictReader(io.StringIO(tables["atom_pairs_within_5A.csv"].decode())))
    assert len(pairs) == len(result["atom_pairs_within_5A"])
    assert all(float(p["distance_A"]) <= 5.0 for p in pairs)
    sensitivity = list(csv.DictReader(io.StringIO(tables["radius_sensitivity.csv"].decode())))
    assert len(sensitivity) == 16
    assert all("No pooled count" in r["interpretation_limit"] for r in sensitivity)


def test_figures_are_deterministic_labelled_and_complete(result: dict[str, Any]) -> None:
    first = figures.render(result)
    assert first == figures.render(result)
    assert set(first) == {f"{figures.FIGURE}.{e}" for e in ("png", "svg", "pdf")}
    assert Image.open(io.BytesIO(first[f"{figures.FIGURE}.png"])).format == "PNG"
    assert first[f"{figures.FIGURE}.pdf"].startswith(b"%PDF-")
    svg = first[f"{figures.FIGURE}.svg"].decode()
    for label in (
        "NUDT5",
        "NUDT14",
        "8RIY",
        "8OTV",
        "Å",
        "4.0 (primary)",
        "not homology",
        "not binding energy",
        "Arg51",
        "Leu107",
        "alt A",
        "null/refused",
    ):
        assert label in svg
    for site in result["sites"]:
        assert site["site_identity"]["site_id"].split(":model1")[1] in svg


@pytest.mark.parametrize("mode", ["threshold", "warning", "missing", "null"])
def test_invalid_results_refused(result: dict[str, Any], mode: str) -> None:
    bad = copy.deepcopy(result)
    row = next(r for r in bad["residue_proximity"] if r["observed_min_distance_A"] is not None)
    if mode == "threshold":
        row["within_4_0A"] = not row["within_4_0A"]
    elif mode == "warning":
        bad["warning"] = "changed"
    elif mode == "missing":
        del bad["refusals"]
    else:
        row["observed_min_distance_A"] = None
    with pytest.raises((ValueError, KeyError)):
        figures.validate(bad)


def test_cli_hash_and_overwrite_refusal(tmp_path: Path, result: dict[str, Any]) -> None:
    source = tmp_path / "result.json"
    source.write_bytes(mod.json_bytes(result))
    output = tmp_path / "derived"
    args = ["--input", str(source), "--output", str(output), "--input-sha256"]
    assert figures.main([*args, "0" * 64]) == 2 and not output.exists()
    assert figures.main([*args, mod.digest(source.read_bytes())]) == 0
    manifest = mod.read_json(output / "derived_manifest.json")
    assert manifest["input_sha256"] == mod.digest(source.read_bytes())
    for name, entry in manifest["artifacts"].items():
        assert mod.digest((output / name).read_bytes()) == entry["sha256"]
    before = {p.name: p.read_bytes() for p in output.iterdir()}
    assert figures.main([*args, mod.digest(source.read_bytes())]) == 2
    assert before == {p.name: p.read_bytes() for p in output.iterdir()}


@pytest.mark.parametrize(
    "mode",
    [
        "nan",
        "negative",
        "string",
        "bool",
        "duplicate",
        "witness",
        "occupancy",
        "coordinates",
        "missingness",
        "pair_nan",
        "site",
        "unestimated",
    ],
)
def test_malformed_geometry_refuses_before_render(result: dict[str, Any], mode: str) -> None:
    bad = copy.deepcopy(result)
    row = next(r for r in bad["residue_proximity"] if r["observed_min_distance_A"] is not None)
    if mode in ("nan", "negative", "string", "bool"):
        row["observed_min_distance_A"] = {
            "nan": float("nan"),
            "negative": -1,
            "string": "3.0",
            "bool": True,
        }[mode]
    elif mode == "duplicate":
        bad["residue_proximity"].append(row)
    elif mode == "witness":
        row["minimum_witness_pairs"] = []
    elif mode == "occupancy":
        row["minimum_witness_pairs"][0]["protein_atom"]["occupancy"] = 0
    elif mode == "coordinates":
        atom = row["minimum_witness_pairs"][0]["protein_atom"]
        atom["xyz_A"] = [x + 100 for x in atom["xyz_A"]]
    elif mode == "missingness":
        del row["missing_or_excluded_atoms"]
    elif mode == "pair_nan":
        bad["atom_pairs_within_5A"][0]["distance_A"] = float("nan")
    elif mode == "site":
        row["site_id"] = "invented"
    else:
        row["complete_residue_distance_A"] = 3.0
    with pytest.raises(ValueError):
        figures.validate(bad)


@pytest.mark.parametrize("target", ["NUDT5", "NUDT14"])
def test_readable_target_panels_retain_both_sites_and_all_rows(
    result: dict[str, Any], target: str
) -> None:
    files = figures.render(result, target=target)
    assert files == figures.render(result, target=target)
    svg = files[f"{figures.FIGURE}_{target}.svg"].decode()
    rows = [
        r
        for r in result["residue_proximity"]
        if r["residue_identity"]["pdb_id"] == ("8RIY" if target == "NUDT5" else "8OTV")
    ]
    for row in rows:
        if row["within_5_0A"]:
            assert figures.residue_label(row) in svg
    assert "4.0 (primary)" in svg and "null/refused" in svg
    import xml.etree.ElementTree as ET

    visible = " ".join(" ".join(ET.fromstring(svg).itertext()).split())
    assert figures.CAVEAT in visible
    assert len({r["site_id"] for r in rows}) == 2
