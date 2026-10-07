"""SOFTWARE-ONLY rendering tests, not experiments or biological validation."""

from __future__ import annotations

import copy
import hashlib
import io
import subprocess
import sys
from pathlib import Path
from typing import Any

import build_selectivity_figures as figures
import pytest
import selectivity as mod
from PIL import Image
from pipeline import ROOT
from test_selectivity import software_subset


@pytest.fixture(scope="module")
def result() -> dict[str, Any]:
    return mod.analyze(
        ROOT / "research/selectivity/paired_evidence.json",
        ROOT / "research/results/transfer.json",
        ROOT,
    )[0]


def test_real_figures_are_deterministic_and_explicit(result: dict[str, Any]) -> None:
    first = figures.render(result)
    assert first == figures.render(result)
    assert len(first) == 6
    for name in figures.FIGURE_NAMES:
        assert Image.open(io.BytesIO(first[f"{name}.png"])).size == (2100, 1300)
        assert first[f"{name}.pdf"].startswith(b"%PDF-")
        svg = first[f"{name}.svg"].decode()
        for label in (
            "n=6",
            "3 point ratios",
            "1 strict bound(s)",
            "2 without a finite ratio bound",
            "13: &lt;-1.128",
            "12: both &gt;50",
            "15: both &gt;50",
            "IC50(NUDT14)",
            "IC50(NUDT5)",
            "uncalibrated label score",
            "ratio uncertainty",
            "20/60 min",
            "not fresh external or prospective validation",
        ):
            assert label in svg
        assert "#0072b2" in svg and "#d55e00" in svg
        assert not any(line.rstrip() != line for line in svg.splitlines())


@pytest.mark.parametrize("names", [[], ["2"], ["1"], ["12"], ["13"], ["1", "2"]])
def test_one_zero_eligible_pair_figures(result: dict[str, Any], names: list[str]) -> None:
    small = software_subset(result, names)
    payloads = figures.render(small)
    n = small["summary"]["scenarios"][mod.SCENARIOS[0]]["n"]
    for name in figures.FIGURE_NAMES:
        svg = payloads[f"{name}.svg"].decode()
        assert f"n={n}" in svg and "SOFTWARE TESTS ONLY" in svg
        if not n:
            assert "No eligible pairs" in svg
        if "12" in names:
            assert "12: both &gt;50" in svg
        assert Image.open(io.BytesIO(payloads[f"{name}.png"])).getbbox() is not None


@pytest.mark.parametrize(
    "defect",
    [
        "nonfinite",
        "censored_point",
        "ratio_direction",
        "rank",
        "count",
        "stale_identity",
        "duplicate",
        "scenario",
    ],
)
def test_bad_results_refuse_figures(result: dict[str, Any], defect: str) -> None:
    bad = copy.deepcopy(result)
    if defect == "nonfinite":
        bad["score_rows"][0]["RF"] = float("inf")
    elif defect == "censored_point":
        bad["rows"][11]["ratio"]["point"] = 1
    elif defect == "ratio_direction":
        bad["ratio_definition"] = "IC50_NUDT5 / IC50_NUDT14"
    elif defect == "rank":
        bad["score_rows"][0]["ranks"]["RF"] = 0
    elif defect == "count":
        bad["summary"]["source_graphs"] = 1
    elif defect == "stale_identity":
        bad["score_rows"][0]["canonical_isomeric_smiles"] = "CC"
    elif defect == "duplicate":
        bad["rows"].append(copy.deepcopy(bad["rows"][0]))
    else:
        bad["score_rows"][0]["scenario"] = "chosen_best"
    with pytest.raises(ValueError):
        figures.render(bad)


def test_figure_cli_hash_guard_external_cwd_and_no_overwrite(
    result: dict[str, Any],
    tmp_path: Path,
) -> None:
    source = tmp_path / "recorded.json"
    data = mod.json_bytes(result)
    source.write_bytes(data)
    manifest = tmp_path / "recorded-manifest.json"
    manifest.write_bytes(
        mod.json_bytes(
            {
                "artifacts": {
                    "selectivity.json": {
                        "sha256": hashlib.sha256(data).hexdigest(),
                        "size_bytes": len(data),
                    }
                }
            }
        )
    )
    output = tmp_path / "figures"
    args = [
        sys.executable,
        str(ROOT / "scripts/build_selectivity_figures.py"),
        "--input",
        str(source),
        "--manifest",
        str(manifest),
        "--output",
        str(output),
    ]
    process = subprocess.run(args, cwd=tmp_path, capture_output=True, text=True)
    assert process.returncode == 0, process.stderr
    published = {p.name: p.read_bytes() for p in output.iterdir()}
    assert subprocess.run(args, cwd=tmp_path, capture_output=True).returncode == 2
    assert published == {p.name: p.read_bytes() for p in output.iterdir()}
    source.write_bytes(data + b" ")
    bad_output = tmp_path / "bad"
    assert (
        figures.main(
            ["--input", str(source), "--manifest", str(manifest), "--output", str(bad_output)]
        )
        == 2
    )
    assert not bad_output.exists()


def test_software_only_lower_bound_arrow(result: dict[str, Any]) -> None:
    small = software_subset(result, ["1"])
    row = small["rows"][0]
    row["endpoints"]["NUDT14"].update(
        status="right_censored",
        reported_mean=None,
        reported_sd=None,
        bound=50.0,
        comparator=">",
        bound_strict=True,
        biological_n_for_reported_sd=None,
        author_csv_raw="not active",
        ledger_text="not active",
        table1_text="NA",
    )
    row["ratio"] = mod.ratio(row["endpoints"]["NUDT5"], row["endpoints"]["NUDT14"])
    for score in small["score_rows"]:
        score["ratio_status"] = "lower_bound"
    small["summary"] = mod.summarize(small["rows"], small["score_rows"])
    for name in figures.FIGURE_NAMES:
        svg = figures.render(small)[f"{name}.svg"].decode()
        assert "1: &gt;1.776" in svg
        assert "0 point ratios" in svg and "SOFTWARE TESTS ONLY" in svg
