"""SOFTWARE TESTS ONLY; artificial mutations never enter the measured evidence ledger."""

from __future__ import annotations

import copy
import csv
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import selectivity as mod
from pipeline import ROOT
from transfer import identity

SOURCE = ROOT / "research/selectivity/paired_evidence.json"
PREDICTIONS = ROOT / "research/results/transfer.json"


@pytest.fixture(scope="module")
def result() -> dict[str, Any]:
    return mod.analyze(SOURCE, PREDICTIONS, ROOT)[0]


def software_subset(result: dict[str, Any], names: list[str]) -> dict[str, Any]:
    small = copy.deepcopy(result)
    small["warning"] = "SOFTWARE TESTS ONLY; reduced fixture, not biological Results"
    small["rows"] = [r for r in small["rows"] if r["source_compound"] in names]
    small["score_rows"] = [r for r in small["score_rows"] if r["source_compound"] in names]
    for scenario in mod.SCENARIOS:
        eligible = [
            r
            for r in small["score_rows"]
            if r["scenario"] == scenario and r["paired_diagnostic_eligible"]
        ]
        for row in (r for r in small["score_rows"] if r["scenario"] == scenario):
            row["rank_denominator"] = len(eligible)
            for method in mod.METHODS:
                rank = (
                    1 + sum(r[method] > row[method] for r in eligible)
                    if row["paired_diagnostic_eligible"]
                    else None
                )
                row["ranks"][method] = rank
                row[f"{method}_rank_within_six"] = rank
    small["summary"] = mod.summarize(small["rows"], small["score_rows"])
    return small


def endpoint(state: str, target: str = "NUDT5", mean: float = 2.0) -> dict[str, Any]:
    numeric = state == "numeric"
    censored = state == "right_censored"
    return {
        "target": target,
        "endpoint": "IC50",
        "endpoint_family": "purified_enzyme_catalytic_inhibition",
        "unit": "uM",
        "status": state,
        "reported_mean": mean if numeric else None,
        "reported_sd": 0.1 if numeric else None,
        "bound": 50.0 if censored else None,
        "bound_strict": True if censored else None,
        "comparator": "=" if numeric else ">" if censored else None,
        "biological_n_for_reported_sd": 2 if numeric else None,
    }


@pytest.mark.parametrize(
    "vector", mod.read_json(SOURCE.parent / "contract_test_vectors.json")["cases"]
)
def test_software_contract_vectors(vector: dict[str, Any]) -> None:
    targets = []
    for target in mod.TARGETS:
        raw = vector[f"input_{target}"]
        cell = endpoint(raw["status"], target)
        cell.update(raw)
        targets.append(cell)
    actual = mod.ratio(*targets)
    for key, expected in vector["expected"].items():
        if type(expected) in (int, float):
            assert math.isclose(actual[key], expected, rel_tol=1e-12, abs_tol=1e-12)
        else:
            assert actual[key] == expected


def test_direction_and_equality_boundaries() -> None:
    left, right = endpoint("numeric", mean=2), endpoint("numeric", "NUDT14", 10)
    forward = mod.ratio(left, right)
    left["reported_mean"], right["reported_mean"] = 10, 2
    reverse = mod.ratio(left, right)
    assert forward["point"] == 5
    assert reverse["point"] == 0.2
    assert forward["log10_point"] == pytest.approx(-reverse["log10_point"])
    for first, second, symbol in (
        ("right_censored", "numeric", "<"),
        ("numeric", "right_censored", ">"),
    ):
        value = mod.ratio(endpoint(first, mean=50), endpoint(second, "NUDT14", 50))
        assert value["bound"] == 1 and value["log10_bound"] == 0
        assert value["comparator"] == symbol and value["bound_strict"] is True
        assert value["point"] is None
    left["target"], right["target"] = "NUDT14", "NUDT5"
    with pytest.raises(ValueError, match="target"):
        mod.ratio(left, right)


@pytest.mark.parametrize(
    "field,value",
    [
        ("reported_mean", 0),
        ("reported_mean", -1),
        ("reported_mean", float("inf")),
        ("reported_mean", float("nan")),
        ("reported_mean", "2"),
        ("reported_mean", True),
        ("reported_sd", -0.1),
        ("reported_sd", float("inf")),
        ("reported_sd", None),
        ("endpoint", "KD"),
        ("endpoint", "EC50"),
        ("endpoint_family", "cell_viability"),
        ("unit", "nM"),
        ("comparator", ">="),
        ("bound", 50),
        ("bound_strict", False),
    ],
)
def test_invalid_numeric_endpoints(field: str, value: Any) -> None:
    bad = endpoint("numeric")
    bad[field] = value
    with pytest.raises(ValueError):
        mod.ratio(bad, endpoint("numeric", "NUDT14"))


@pytest.mark.parametrize(
    "state,patch",
    [
        ("right_censored", {"reported_mean": 50}),
        ("right_censored", {"reported_sd": 0}),
        ("right_censored", {"comparator": ">="}),
        ("right_censored", {"bound": 49}),
        ("right_censored", {"bound_strict": False}),
        ("right_censored", {"bound_strict": 1}),
        ("untested", {"reported_mean": 2}),
        ("untested", {"biological_n_for_reported_sd": 2}),
        ("untested", {"bound": 50}),
        ("untested", {"comparator": "="}),
    ],
)
def test_censoring_and_missingness_cannot_be_repaired(state: str, patch: dict[str, Any]) -> None:
    bad = endpoint(state)
    bad.update(patch)
    with pytest.raises(ValueError):
        mod.ratio(bad, endpoint("numeric", "NUDT14"))


@pytest.mark.parametrize(
    "bad",
    [
        "2",
        "nan ± 1",
        "inf ± 1",
        "1e999 ± 0",
        "0 ± 1",
        "-1 ± 1",
        "1 ± -1",
        "1 ± NaN",
        "2junk ± 1",
        "NA",
        ">50",
        "",
        "2 ± 1 extra",
    ],
)
def test_malformed_reported_numbers(bad: str) -> None:
    with pytest.raises(ValueError):
        mod.parse_reported(bad)


def test_source_whitespace_and_na_semantics() -> None:
    assert mod.parse_reported("13.9 ± 0.62 ") == ("numeric", 13.9, 0.62)
    assert mod.parse_reported("NA", table=True) == ("right_censored", None, None)
    assert mod.parse_reported("not tested") == ("untested", None, None)


@pytest.mark.parametrize(
    "text",
    [
        '{"x":1,"x":2}',
        '{"x":{"y":1,"y":2}}',
        '{"x":NaN}',
        '{"x":Infinity}',
        '{"x":-Infinity}',
        '{"x":1e999}',
    ],
)
def test_strict_json(tmp_path: Path, text: str) -> None:
    path = tmp_path / "bad.json"
    path.write_text(text)
    with pytest.raises(ValueError):
        mod.read_json(path)


@pytest.mark.parametrize(
    "text",
    [
        "id,id\na,b\n",
        "id,\na,b\n",
        "id,value\na,b,c\n",
        "id,value\na\n",
        'id,value\na,"unclosed\n',
        "value,id\nb,a\n",
    ],
)
def test_csv_header_and_shape_fail_closed(tmp_path: Path, text: str) -> None:
    path = tmp_path / "bad.csv"
    path.write_text(text)
    with pytest.raises((ValueError, csv.Error)):
        mod.strict_csv(path, ("id", "value"))


def test_csv_quoted_comma_whitespace_null_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "quoted.csv"
    rows = [{"id": "software,only", "raw": "13.9 ± 0.62 ", "missing": None, "bool": False}]
    path.write_bytes(mod.csv_bytes(rows, ("id", "raw", "missing", "bool")))
    assert mod.strict_csv(path, ("id", "raw", "missing", "bool")) == [
        {"id": "software,only", "raw": "13.9 ± 0.62 ", "missing": "", "bool": "False"}
    ]
    assert b"\r\n" not in path.read_bytes()


@pytest.mark.parametrize(
    "defect", ["unknown", "missing_target", "double_point", "fake_mean", "nonstrict"]
)
def test_release_schema_negative_mutations(defect: str) -> None:
    data = mod.read_json(SOURCE)
    if defect == "unknown":
        data["unexpected"] = 1
    elif defect == "missing_target":
        del data["rows"][0]["endpoints"]["NUDT14"]
    elif defect == "double_point":
        data["rows"][11]["ratio"]["point"] = 1
    elif defect == "fake_mean":
        data["rows"][11]["endpoints"]["NUDT5"]["reported_mean"] = 50
    else:
        data["rows"][11]["endpoints"]["NUDT5"]["bound_strict"] = 1
    with pytest.raises(ValueError):
        mod.validate_schema(data, mod.read_json(SOURCE.parent / "paired_evidence.schema.json"))


def test_schema_rejects_unsupported_keywords() -> None:
    with pytest.raises(ValueError, match="Unsupported"):
        mod.validate_schema({}, {"patternProperties": {}})


def test_real_data_hand_calculations_and_full_accounting(result: dict[str, Any]) -> None:
    rows = {r["source_compound"]: r for r in result["rows"]}
    # Hand transcription from archived Table 1 (tbl1/fx2); no fitted or expected favorable metric.
    numeric = {
        "1": (0.837, 0.329, 0.990, 0.110),
        "9": (0.270, 0.027, 0.162, 0.005),
        "10": (0.487, 0.010, 0.263, 0.031),
        "11": (2.04, 0.240, 0.519, 0.084),
        "14": (13.8, 0.900, 1.64, 0.140),
    }
    for name, (mean5, sd5, mean14, sd14) in numeric.items():
        row = rows[name]
        for target, mean, sd in (("NUDT5", mean5, sd5), ("NUDT14", mean14, sd14)):
            assert row["endpoints"][target]["reported_mean"] == mean
            assert row["endpoints"][target]["reported_sd"] == sd
            assert row["endpoints"][target]["biological_n_for_reported_sd"] == 2
        assert row["ratio"]["point"] == mean14 / mean5
    assert rows["13"]["ratio"]["bound"] == 3.72 / 50
    assert rows["13"]["ratio"]["comparator"] == "<"
    assert rows["13"]["ratio"]["log10_bound"] == pytest.approx(-1.1284270644541214)
    for name in ("12", "15"):
        assert rows[name]["ratio"]["status"] == "double_censored"
        assert rows[name]["ratio"]["point"] is None and rows[name]["ratio"]["bound"] is None
    summary = result["summary"]
    assert summary["source_graphs"] == 23 and summary["endpoint_cells"] == 46
    assert summary["source_paired_n"] == 8
    assert summary["source_ratio_status_counts"] == {
        "point": 5,
        "upper_bound": 1,
        "double_censored": 2,
        "missing_endpoint": 15,
    }
    assert summary["endpoint_status_counts"] == {
        "NUDT5": {"numeric": 7, "right_censored": 5, "untested": 11},
        "NUDT14": {"numeric": 6, "right_censored": 2, "untested": 15},
    }
    for scenario in mod.SCENARIOS:
        assert summary["scenarios"][scenario]["eligible_ids"] == ["1", "9", "12", "13", "14", "15"]
        assert summary["scenarios"][scenario]["ratio_status_counts"] == {
            "point": 3,
            "upper_bound": 1,
            "double_censored": 2,
        }
    assert rows["11"]["candidate_identity_matches"]["canonical_smiles"] == ["NC5-02"]
    assert rows["6"]["reference_sensitivity_training_matches"]["canonical_smiles"] == ["ACT-01"]
    assert len(result["quarantined_training_records"]) == 1
    assert result["historical_prediction_manifest"] == mod.read_json(
        ROOT / "research/results/transfer-manifest.json"
    )


def test_all_frozen_scores_and_tied_ranks_retained(result: dict[str, Any]) -> None:
    frozen = mod.strict_csv(SOURCE.parent / "frozen_scores.csv", mod.SCORE_FIELDS)
    by_key = {(r["source_compound"], r["scenario"]): r for r in result["score_rows"]}
    assert len(by_key) == 46
    for row in frozen:
        out = by_key[(row["source_compound"], row["scenario"])]
        for method in mod.METHODS:
            assert out[method] == float(row[method])
            raw_rank = row[f"{method}_rank_within_six"]
            assert out["ranks"][method] == (int(raw_rank) if raw_rank else None)
    assert by_key[("9", mod.SCENARIOS[0])]["ranks"]["Equal_mean"] == 1
    assert by_key[("9", mod.SCENARIOS[0])]["Equal_mean"] == pytest.approx(0.8112916069767883)


@pytest.mark.parametrize(
    "defect",
    [
        "duplicate_id",
        "duplicate_graph",
        "graph",
        "source_smiles",
        "id",
        "missing",
        "method",
        "nonfinite",
        "overlap",
        "equal_mean",
    ],
)
def test_prediction_joins_fail_closed(result: dict[str, Any], defect: str) -> None:
    predicted = mod.read_json(PREDICTIONS)
    rows = predicted["measured_source_challenge"]["rows"]
    if defect == "duplicate_id":
        rows.append(copy.deepcopy(rows[0]))
    elif defect == "duplicate_graph":
        rows[1]["canonical_smiles"] = rows[0]["canonical_smiles"]
    elif defect == "graph":
        rows[0]["canonical_parent_tautomer"] = "CC"
    elif defect == "source_smiles":
        rows[0]["source"]["source_smiles"] = "CC"
    elif defect == "id":
        rows[0]["id"] = "software-only-alias"
    elif defect == "missing":
        rows.pop()
    elif defect == "method":
        rows[0]["scores"]["fitted_new_score"] = 0.5
    elif defect == "nonfinite":
        rows[0]["scores"]["RF"] = float("nan")
    elif defect == "overlap":
        rows[0]["training_matches"]["canonical_smiles"] = ["ACT-01"]
    else:
        rows[0]["scores"]["Equal_mean"] = 0.1
    with pytest.raises(ValueError):
        mod.join_scores(
            result["rows"],
            predicted,
            mod.strict_csv(ROOT / "research/source_assays.csv", mod.LEDGER_FIELDS),
        )


def test_identity_levels_remain_distinct() -> None:
    left, right = identity("N[C@H](C)C(=O)O"), identity("N[C@@H](C)C(=O)O")
    assert left["canonical_smiles"] != right["canonical_smiles"]
    acid, salt = identity("CC(=O)O"), identity("CC(=O)[O-].[Na+]")
    assert acid["canonical_smiles"] != salt["canonical_smiles"]
    assert acid["neutral_fragment_parent"] == salt["neutral_fragment_parent"]


@pytest.mark.parametrize(
    "defect", ["id", "graph", "source_smiles", "endpoint", "ratio", "identity"]
)
def test_source_semantics_fail_closed(result: dict[str, Any], defect: str) -> None:
    rows = copy.deepcopy(result["rows"])
    if defect == "id":
        rows[1]["source_compound"] = rows[0]["source_compound"]
    elif defect == "graph":
        rows[1]["source_smiles"] = rows[0]["source_smiles"]
    elif defect == "source_smiles":
        rows[0]["source_smiles"] += ".Cl"
    elif defect == "identity":
        rows[0]["identity"]["inchikey"] = "INVALID"
    elif defect == "endpoint":
        rows[0]["endpoints"]["NUDT14"]["unit"] = "nM"
    else:
        rows[11]["ratio"]["point"] = 1
    predicted = mod.read_json(PREDICTIONS)
    training = predicted["training_identity"]
    sensitivity = copy.deepcopy(training)
    for change in predicted["reference_sensitivity"]["changes"]:
        next(r for r in sensitivity if r["id"] == change["id"]).update(change["replacement"])
    candidates = [
        {"id": r["id"], **{k: r[k] for k in rows[0]["identity"]}} for r in predicted["candidates"]
    ]
    with pytest.raises(ValueError):
        mod.validate_source_rows(
            rows,
            mod.strict_csv(ROOT / "research/source_assays.csv", mod.LEDGER_FIELDS),
            training,
            sensitivity,
            candidates,
            result["assay_conditions"],
        )


@pytest.mark.parametrize("names", [[], ["2"], ["1"], ["12"], ["13"], ["1", "2"]])
def test_zero_one_pair_document_and_csv_paths(result: dict[str, Any], names: list[str]) -> None:
    small = software_subset(result, names)
    payloads = mod.artifacts(small)
    text = payloads["selectivity.md"].decode()
    n = sum(r["model_diagnostic_eligible"] for r in small["rows"])
    assert f"n={n} nonoverlap paired compounds" in text
    if not n:
        assert "No eligible pairs" in text
    assert "ratio uncertainty" in text
    assert ">50 (strict)" in text if "12" in names or "13" in names else True
    for name in names:
        assert f"| {name} |" in text
    assert mod.artifacts(small) == payloads


def test_deterministic_analysis_no_fitting(
    result: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    import transfer

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("Model fitting/scoring is forbidden")

    for name in (
        "fit_scores",
        "score_panel",
        "measured_challenge",
        "probe_challenge",
        "reference_sensitivity",
        "run",
    ):
        monkeypatch.setattr(transfer, name, forbidden)
    again, _ = mod.analyze(SOURCE, PREDICTIONS, ROOT)
    assert mod.artifacts(again) == mod.artifacts(result)


def test_output_refusal_and_atomic_cleanup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    output = tmp_path / "out"
    payloads = {"a.csv": b"a\n", "done.json": b"{}\n"}
    mod.publish(output, payloads, "done.json")
    before = {p.name: p.read_bytes() for p in output.iterdir()}
    with pytest.raises(ValueError, match="empty"):
        mod.publish(output, payloads, "done.json")
    assert {p.name: p.read_bytes() for p in output.iterdir()} == before
    destination = tmp_path / "file"
    destination.write_bytes(b"unchanged")
    with pytest.raises(ValueError):
        mod.publish(destination, payloads, "done.json")
    assert destination.read_bytes() == b"unchanged"
    link = os.link
    calls = 0

    def fail_second(src: Any, dst: Any) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("simulated interruption before completion")
        link(src, dst)

    monkeypatch.setattr(os, "link", fail_second)
    with pytest.raises(OSError, match="interruption"):
        mod.publish(tmp_path / "failed", payloads, "done.json")
    assert not (tmp_path / "failed").exists()
    assert not list(tmp_path.glob(".selectivity-*"))


def test_concurrent_writer_is_not_overwritten(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "out"
    link = os.link

    def race(src: Any, dst: Any) -> None:
        Path(dst).write_bytes(b"other writer")
        link(src, dst)

    monkeypatch.setattr(os, "link", race)
    with pytest.raises(FileExistsError):
        mod.publish(output, {"done.json": b"{}"}, "done.json")
    assert (output / "done.json").read_bytes() == b"other writer"


def test_cli_external_cwd_and_overwrite(tmp_path: Path) -> None:
    args = [
        sys.executable,
        str(ROOT / "scripts/scripts/selectivity.py"),
        "--source",
        str(SOURCE),
        "--predictions",
        str(PREDICTIONS),
        "--repository",
        str(ROOT),
        "--output",
        str(tmp_path / "out"),
    ]
    completed = subprocess.run(args, cwd=tmp_path, capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
    path = tmp_path / "out/selectivity-manifest.json"
    manifest = mod.read_json(path)
    for name, data in manifest["artifacts"].items():
        assert mod.sha256(tmp_path / "out" / name) == data["sha256"]
    assert manifest["command"][1:] == args[1:]
    assert subprocess.run(args, cwd=tmp_path, capture_output=True).returncode == 2


def test_stale_predictions_fail_before_publication(tmp_path: Path) -> None:
    path = tmp_path / "stale.json"
    prediction = mod.read_json(PREDICTIONS)
    prediction["measured_source_challenge"]["rows"][0]["scores"]["RF"] = 0.99
    path.write_text(json.dumps(prediction))
    output = tmp_path / "out"
    assert (
        mod.main(["--source", str(SOURCE), "--predictions", str(path), "--output", str(output)])
        == 2
    )
    assert not output.exists()


def test_projection_mutation_and_fabricated_ratio_fail(
    tmp_path: Path, result: dict[str, Any]
) -> None:
    path = tmp_path / "bad.csv"
    rows = mod.long_projection(result["rows"])
    path.write_bytes(mod.csv_bytes(rows, mod.LONG_FIELDS))
    text = path.read_text().replace(",0.837,0.329,", ",0.837junk,0.329,", 1)
    path.write_text(text)
    with pytest.raises(ValueError, match="Malformed"):
        mod.compare_projection(path, mod.LONG_FIELDS, rows)
    bad = copy.deepcopy(result)
    bad["rows"][11]["ratio"]["point"] = 1
    with pytest.raises(ValueError, match="Ratio"):
        mod.artifacts(bad)


def test_write_failure_cleans_staging(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(fd: int) -> None:
        raise OSError("simulated fsync failure")

    monkeypatch.setattr(os, "fsync", fail)
    with pytest.raises(OSError):
        mod.publish(tmp_path / "failed", {"done.json": b"{}"}, "done.json")
    assert not (tmp_path / "failed").exists()
    assert not list(tmp_path.glob(".selectivity-*"))


def test_input_hashes_and_source_snapshots_are_not_optional(tmp_path: Path) -> None:
    import shutil

    base = tmp_path / "curation"
    shutil.copytree(SOURCE.parent, base)
    original = (base / "sources/balikci.xml.gz").read_bytes()
    (base / "sources/balikci.xml.gz").write_bytes(original[:-1] + b"X")
    with pytest.raises(ValueError, match="hash"):
        mod.verify_inputs(base / "paired_evidence.json", PREDICTIONS, ROOT)
    assert not (tmp_path / "out").exists()


def test_output_symlink_refused(tmp_path: Path) -> None:
    target = tmp_path / "target"
    target.mkdir()
    link = tmp_path / "link"
    link.symlink_to(target, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        mod.publish(link, {"done.json": b"{}"}, "done.json")
    assert not list(target.iterdir())


@pytest.mark.parametrize("name", ["scripts/scripts/transfer.py", mod.HISTORICAL_TRANSFER])
def test_repaired_runtime_preserves_both_code_hash_gates(
    name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = mod.sha256
    monkeypatch.setattr(
        mod, "sha256", lambda path: "0" * 64 if path == ROOT / name else original(path)
    )
    with pytest.raises(ValueError, match="Unreviewed transfer runtime|Stale/incompatible original"):
        mod.verify_inputs(SOURCE, PREDICTIONS, ROOT)
