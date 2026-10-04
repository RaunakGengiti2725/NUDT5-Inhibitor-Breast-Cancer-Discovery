from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
from build_extension_figures import build_figures, write_tables
from pipeline import ROOT

RESULTS = ROOT / "research/results"


def test_every_figure_is_written_in_three_formats_from_recorded_results(tmp_path: Path) -> None:
    captions = build_figures(RESULTS, tmp_path)
    assert len(captions) == 3
    for path, caption in captions:
        assert path.stat().st_size > 5000
        for extension in ("pdf", "svg"):
            assert path.with_suffix(f".{extension}").stat().st_size > 1000
        assert all(
            line == line.rstrip() for line in path.with_suffix(".svg").read_text().splitlines()
        )
        assert len(caption) > 200
        assert "Figure" in caption


def test_tables_match_recorded_json_and_never_overwrite(tmp_path: Path) -> None:
    write_tables(RESULTS, tmp_path)
    controls = json.loads((RESULTS / "controls.json").read_text())
    transfer = json.loads((RESULTS / "transfer.json").read_text())
    assert b"\r\n" not in (tmp_path / "all_metrics.csv").read_bytes()
    rows = list(csv.DictReader((tmp_path / "all_metrics.csv").open()))
    recorded = controls["evaluations"]["full_valid_set"]["methods"]
    for method, metrics in recorded.items():
        row = next(r for r in rows if r["design"] == "full_valid_set" and r["method"] == method)
        assert float(row["auc"]) == metrics["auc"]
    for split, evaluation in transfer["reference_sensitivity"]["evaluations"].items():
        for method, metrics in evaluation["metrics"].items():
            row = next(
                r
                for r in rows
                if r["design"] == f"authenticated_reference_{split}" and r["method"] == method
            )
            assert float(row["auc"]) == metrics["auc"]
    assert sum(r["design"].startswith("ablation_") for r in rows) == 51
    candidates = list(csv.DictReader((tmp_path / "candidate_axes.csv").open()))
    assert len(candidates) == len(transfer["candidates"]) == 10
    assert {r["id"] for r in candidates} == {r["id"] for r in transfer["candidates"]}
    text = (tmp_path / "supplementary_results.md").read_text()
    assert "arbitrary" in text
    assert "No coverage guarantee" in text
    with pytest.raises(FileExistsError):
        write_tables(RESULTS, tmp_path)


def test_missing_results_fail_rather_than_fabricate(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        build_figures(tmp_path, tmp_path / "out")
    with pytest.raises(FileNotFoundError):
        write_tables(tmp_path, tmp_path / "out")
