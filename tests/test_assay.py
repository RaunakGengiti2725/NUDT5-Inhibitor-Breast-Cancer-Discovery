"""SOFTWARE TESTS ONLY. No fixture represents a measured experiment or qualification."""

from __future__ import annotations

import copy
import csv
import io
import json
import math
import os
import subprocess
import sys
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any, cast
from unittest.mock import patch

import assay
import numpy as np
import pipeline
import pytest
from numpy.typing import NDArray
from pipeline import write_json as pipeline_write_json
from scipy.optimize import least_squares

CONTRACT = assay.strict_json((assay.ASSAY / "contract.json").read_bytes())
STRESS = assay.strict_json((assay.ASSAY / "stress_matrix.json").read_bytes())


def write_document(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, allow_nan=False, indent=2) + "\n", encoding="utf-8")


@dataclass
class SoftwareFixture:
    root: Path
    manifest: dict[str, Any]
    rows: list[dict[str, str]]

    @property
    def manifest_path(self) -> Path:
        return self.root / "SOFTWARE_TEST_ONLY.manifest.json"

    @property
    def csv_path(self) -> Path:
        return self.root / "SOFTWARE_TEST_ONLY.observations.csv"

    def add_artifact(self, identifier: str, role: str, content: Any) -> None:
        path = self.root / f"SOFTWARE_TEST_ONLY.{identifier}.json"
        write_document(path, content)
        self.manifest["artifacts"].append(
            {
                "id": identifier,
                "path": path.name,
                "sha256": assay.sha256(path.read_bytes()),
                "role": role,
                "origin": self.manifest["origin"],
            }
        )

    def document(self, identifier: str) -> dict[str, Any]:
        artifact = next(a for a in self.manifest["artifacts"] if a["id"] == identifier)
        return assay.strict_json((self.root / artifact["path"]).read_bytes())

    def replace_document(self, identifier: str, content: Any) -> None:
        artifact = next(a for a in self.manifest["artifacts"] if a["id"] == identifier)
        write_document(self.root / artifact["path"], content)
        artifact["sha256"] = assay.sha256((self.root / artifact["path"]).read_bytes())

    def save(self) -> None:
        stream = io.StringIO(newline="")
        writer = csv.DictWriter(stream, fieldnames=CONTRACT["csv_columns"], lineterminator="\n")
        writer.writeheader()
        writer.writerows(self.rows)
        self.csv_path.write_text(stream.getvalue(), encoding="utf-8", newline="")
        write_document(self.manifest_path, self.manifest)

    def report(self) -> dict[str, Any]:
        self.save()
        return assay.analyze(assay.load_package(self.manifest_path, self.csv_path))

    def cli(self, output: Path) -> int:
        return assay.main(
            [
                "--manifest",
                str(self.manifest_path),
                "--observations",
                str(self.csv_path),
                "--output",
                str(output),
            ]
        )


def software_fixture(
    root: Path,
    *,
    bottom: float = 0,
    top: float = 100,
    midpoint: float = 1e-6,
    hill: float = 1,
    doses: list[float] | None = None,
    replicates: int = 2,
) -> SoftwareFixture:
    """Deterministic algebra from stress_matrix, not a simulator of biological variability."""
    root.mkdir(parents=True, exist_ok=True)
    concentrations = doses if doses is not None else np.logspace(-9, -3, 13).tolist()
    manifest: dict[str, Any] = {
        "schema_version": assay.VERSION,
        "origin": "simulated",
        "phase": "software_test",
        "endpoint": assay.ENDPOINT,
        "target": {
            "name": "NUDT5",
            "species": "SOFTWARE_TEST_ONLY",
            "construct_id": "SOFTWARE_TEST_ONLY",
            "construct_artifact_id": "construct",
        },
        "artifacts": [],
        "compounds": {"BLD-0001": "synthetic_test"},
        "plates": {
            "P1": {
                "run_id": "R1",
                "independent_experiment_id": "E1",
                "conditions_artifact_id": "conditions",
                "normalization_id": "N1",
                "normalization_artifact_id": "normalization",
                "normalization_formula": assay.FORMULA,
                "normalization_reviewed": True,
                "qc_status": "software_test_only",
                "qc_artifact_id": "qc",
            }
        },
        "design": [
            {
                "blinded_compound_id": "BLD-0001",
                "plate_id": "P1",
                "concentrations_M": concentrations,
                "technical_replicate_ids": [f"T{i + 1}" for i in range(replicates)],
            }
        ],
        "qualification": {
            "status": "software_test_only",
            "evidence_artifact_id": "qualification",
            "reviewer": "SOFTWARE_TEST_ONLY",
            "policy": STRESS["fixture_policy_only"].copy(),
        },
        "lock": {
            "status": "software_test_only",
            "locked_at_utc": None,
            "first_validation_acquisition_utc": None,
            "artifact_id": None,
            "analysis_code_sha256": None,
            "contract_sha256": None,
            "environment_lock_sha256": None,
        },
    }
    fixture = SoftwareFixture(root, manifest, [])
    for identifier, role in (
        ("construct", "construct"),
        ("qualification", "qualification"),
        ("qc", "plate_qc"),
        ("controls", "raw_controls"),
    ):
        fixture.add_artifact(
            identifier, role, {"warning": assay.SOFTWARE_WARNING, "id": identifier}
        )
    fixture.add_artifact(
        "conditions",
        "conditions",
        {
            "endpoint": assay.ENDPOINT,
            "species": "SOFTWARE_TEST_ONLY",
            "construct_id": "SOFTWARE_TEST_ONLY",
            "enzyme_concentration_M": 1e-9,
            "substrate": "ADP-ribose",
            "substrate_concentration_M": 1e-5,
            "reaction_minutes": 20,
            "preincubation_minutes": 0,
            "temperature_C": 20,
            "pH": 7,
            "buffer_description": "SOFTWARE TESTS ONLY",
            "cosolvent": "SOFTWARE TESTS ONLY",
            "cosolvent_percent": 1,
            "detergent_description": "SOFTWARE TESTS ONLY",
            "readout": "SOFTWARE TESTS ONLY",
            "protocol_id": "SOFTWARE_TEST_ONLY",
        },
    )
    fixture.add_artifact(
        "normalization",
        "normalization",
        {
            "plate_id": "P1",
            "normalization_id": "N1",
            "formula": assay.FORMULA,
            "signal_unit": "SOFTWARE_TEST_ONLY",
            "blank_mean": 0,
            "vehicle_mean": 100,
            "blank_control_ids": ["B1"],
            "vehicle_control_ids": ["V1"],
            "controls_artifact_id": "controls",
            "conversion_description": "SOFTWARE TESTS ONLY; synthetic algebra, not raw conversion",
        },
    )
    for dose_index, concentration in enumerate(concentrations):
        # Stable algebra for the extreme predeclared hill=100 case, without overflow.
        power = hill * math.log10(concentration / midpoint)
        fraction = 1 / (1 + 10**power) if power < 300 else 0.0
        response = bottom + (top - bottom) * fraction
        for technical in range(replicates):
            row_index = dose_index * replicates + technical + 1
            offset = 0 if replicates == 1 else 0.4 * technical / (replicates - 1) - 0.2
            fixture.rows.append(
                {
                    "observation_id": f"O{row_index}",
                    "origin": "simulated",
                    "blinded_compound_id": "BLD-0001",
                    "target": "NUDT5",
                    "independent_experiment_id": "E1",
                    "run_id": "R1",
                    "plate_id": "P1",
                    "well_id": f"W{row_index}",
                    "technical_replicate_id": f"T{technical + 1}",
                    "concentration": str(concentration),
                    "concentration_unit": "M",
                    "response_percent_activity": str(response + offset),
                    "observation_status": "observed",
                    "response_relation": "=",
                    "status_reason": "",
                    "normalization_id": "N1",
                    "source_id": "raw",
                    "source_row": str(row_index),
                }
            )
    fixture.add_artifact(
        "raw",
        "raw_observations",
        {
            "warning": assay.SOFTWARE_WARNING,
            "record_index_convention": "1-based array index in rows",
            "rows": fixture.rows,
        },
    )
    fixture.save()
    return fixture


def assert_refusal(report: dict[str, Any], reason: str | None = None) -> None:
    for curve in report["curves"]:
        fit = curve["fit"]
        assert fit["status"] == "refused", fit
        assert fit["reason"]
        if reason:
            assert reason in fit["reason"], fit
        assert fit["parameters"] is None
        assert fit["relative_midpoint_M"] is None
        assert fit["absolute_50_vehicle_M"] is None
        assert fit["absolute_50_status"] == "not_estimated"
    json.dumps(report, allow_nan=False)


def assert_invalid(fixture: SoftwareFixture) -> None:
    output = fixture.root / "SOFTWARE_TEST_ONLY.rejected.json"
    with patch("assay.least_squares", side_effect=AssertionError("must not fit invalid input")):
        with pytest.raises((ValueError, OSError)):
            assay.load_package(fixture.manifest_path, fixture.csv_path)
        assert fixture.cli(output) == 2
    assert not output.exists()


@pytest.mark.parametrize(
    "bottom,top,absolute", [(0, 100, 1e-6), (65, 100, None), (20, 100, 1e-6 * 50 / 30)]
)
def test_full_and_partial_estimands(
    tmp_path: Path, bottom: float, top: float, absolute: float | None
) -> None:
    fixture = software_fixture(tmp_path, bottom=bottom, top=top)
    report = fixture.report()
    fit = report["curves"][0]["fit"]
    assert fit["status"] == "estimated", fit
    assert fit["relative_midpoint_M"] == pytest.approx(1e-6, rel=1e-6)
    if absolute is None:
        assert fit["absolute_50_vehicle_M"] is None
        assert fit["absolute_50_status"] == "no_finite_crossing_in_fitted_asymptotes"
    else:
        assert fit["absolute_50_vehicle_M"] == pytest.approx(absolute, rel=1e-6)
    assert fit["parameters"]["bottom_percent"] == pytest.approx(bottom, abs=1e-7)
    assert fit["parameters"]["top_percent"] == pytest.approx(top, abs=1e-7)
    assert fit["uncertainty"] == "not_estimated"
    assert fit["diagnostics"]["completed_starts"] == 9
    assert report["n_submitted_independent_experiments"] == 1
    assert report["observations"] == fixture.rows
    assert report["origin"] == "simulated"
    assert "SOFTWARE TESTS ONLY" in report["warning"]
    for key in CONTRACT["report_required_fields"]:
        assert key in report
    for key in CONTRACT["fit_required_fields"]:
        assert key in fit


def test_both_response_range_ends_are_preserved(tmp_path: Path) -> None:
    fixture = software_fixture(tmp_path, bottom=-3, top=103)
    report = fixture.report()
    values = [float(row["response_percent_activity"]) for row in report["observations"]]
    assert min(values) < 0 and max(values) > 100
    assert report["curves"][0]["fit"]["status"] == "estimated"
    assert report["observations"] == fixture.rows
    for aggregate in report["curves"][0]["aggregates"]:
        assert aggregate["technical_sd_pp"] == pytest.approx(math.sqrt(0.08), abs=1e-12)


@pytest.mark.parametrize(
    "case,changes,reason",
    [
        ("flat", {"bottom": 100}, "flat_or_insufficient_response_span"),
        ("inverted", {"bottom": 100, "top": 0}, "inverted_or_non_decreasing"),
        ("unbracketed", {"midpoint": 1}, None),
        ("steep_ill_supported", {"hill": 100}, None),
        ("shallow_ill_supported", {"hill": 0.01}, None),
        ("undersampled", {"doses": [1e-8, 1e-7, 1e-6, 1e-5]}, "undersampled"),
        ("narrow_support", {"doses": np.logspace(-6.001, -5.999, 8).tolist()}, None),
        ("no_plateaus", {"doses": np.logspace(-7, -5, 9).tolist()}, "plateaus_unsupported"),
    ],
)
def test_unsupported_curves(
    tmp_path: Path, case: str, changes: dict[str, Any], reason: str | None
) -> None:
    fixture = software_fixture(tmp_path / case, **changes)
    assert_refusal(fixture.report(), reason)


@pytest.mark.parametrize("status", ["absent", "failed", "untested", "censored"])
@pytest.mark.parametrize("relation,limit", [(">", "70"), ("<=", "5")])
def test_missing_and_censored_rows_refuse_whole_curve(
    tmp_path: Path,
    status: str,
    relation: str,
    limit: str,
) -> None:
    fixture = software_fixture(tmp_path)
    fixture.rows[0].update(
        {
            "observation_status": status,
            "status_reason": "SOFTWARE TEST ONLY reason",
            "response_relation": relation if status == "censored" else "not_applicable",
            "response_percent_activity": limit if status == "censored" else "",
        }
    )
    with patch("assay.least_squares", side_effect=AssertionError("no partial-curve fitting")):
        report = fixture.report()
    assert_refusal(report, "incomplete_or_censored_curve")
    assert report["observations"] == fixture.rows
    assert report["curves"][0]["aggregates"][0]["n_technical_observed"] == 1
    assert report["curves"][0]["observation_states"][status] == 1


@pytest.mark.parametrize(
    "kind", ["omitted", "observation", "well", "technical", "source", "hash_alias"]
)
def test_omitted_or_duplicate_identity_fails(tmp_path: Path, kind: str) -> None:
    fixture = software_fixture(tmp_path)
    if kind == "omitted":
        fixture.rows.pop()
    elif kind in {"observation", "well", "technical", "source"}:
        key = {
            "observation": "observation_id",
            "well": "well_id",
            "technical": "technical_replicate_id",
            "source": "source_row",
        }[kind]
        fixture.rows[1][key] = fixture.rows[0][key]
    else:
        alias = copy.deepcopy(next(a for a in fixture.manifest["artifacts"] if a["id"] == "raw"))
        alias["id"] = "raw_alias"
        fixture.manifest["artifacts"].append(alias)
        fixture.rows[1].update(source_id="raw_alias", source_row="1")
    fixture.save()
    assert_invalid(fixture)


@pytest.mark.parametrize("unit", ["M", "mM", "uM", "nM", "pM"])
def test_equivalent_units(tmp_path: Path, unit: str) -> None:
    fixture = software_fixture(tmp_path)
    expected = fixture.report()["curves"]
    for row in fixture.rows:
        row["concentration"] = str(
            Decimal(row["concentration"]) * Decimal(10) ** -assay.UNIT_POWERS[unit]
        )
        row["concentration_unit"] = unit
    actual = fixture.report()["curves"]
    assert actual == expected
    assert assay.molar_dose("1000", "nM") == assay.molar_dose("1", "uM")


@pytest.mark.parametrize("unit", ["mg/mL", "µM", "μM", "UM", "uM "])
def test_ambiguous_units_fail(tmp_path: Path, unit: str) -> None:
    fixture = software_fixture(tmp_path)
    fixture.rows[0]["concentration_unit"] = unit
    fixture.save()
    assert_invalid(fixture)


@pytest.mark.parametrize(
    "field,value",
    [
        ("concentration", "NaN"),
        ("concentration", "inf"),
        ("concentration", "1e-999"),
        ("concentration", "1e999"),
        ("concentration", "0"),
        ("concentration", "-1"),
        ("concentration", "1_000"),
        ("response_percent_activity", "NaN"),
        ("response_percent_activity", "-inf"),
        ("response_percent_activity", "1e-999"),
        ("response_percent_activity", "1000000"),
        ("response_percent_activity", "-1000000"),
        ("response_percent_activity", ""),
        ("concentration", " 1e-9"),
    ],
)
def test_nonfinite_missing_and_numeric_support(tmp_path: Path, field: str, value: str) -> None:
    fixture = software_fixture(tmp_path)
    fixture.rows[0][field] = value
    fixture.save()
    assert_invalid(fixture)


def test_molar_conversion_underflow_and_float_aliases(tmp_path: Path) -> None:
    with pytest.raises(assay.InputError):
        assay.molar_dose("1e-320", "pM")
    fixture = software_fixture(tmp_path)
    fixture.rows[0]["concentration"] = "0.00000000100000000000000000001"
    fixture.save()
    assert_invalid(fixture)


def add_second_plate(fixture: SoftwareFixture) -> None:
    second = software_fixture(fixture.root / "second", midpoint=1e-5)
    plate = copy.deepcopy(second.manifest["plates"]["P1"])
    plate.update(
        run_id="R2",
        conditions_artifact_id="conditions2",
        normalization_id="N2",
        normalization_artifact_id="normalization2",
    )
    fixture.manifest["plates"]["P2"] = plate
    condition = second.document("conditions")
    condition["reaction_minutes"] = 30
    fixture.add_artifact("conditions2", "conditions", condition)
    normalization = second.document("normalization")
    normalization.update(plate_id="P2", normalization_id="N2")
    fixture.add_artifact("normalization2", "normalization", normalization)
    design = copy.deepcopy(second.manifest["design"][0])
    design["plate_id"] = "P2"
    fixture.manifest["design"].append(design)
    for row in second.rows:
        row.update(
            plate_id="P2",
            run_id="R2",
            normalization_id="N2",
            observation_id="second-" + row["observation_id"],
            source_id="raw2",
        )
    fixture.add_artifact(
        "raw2", "raw_observations", {"warning": assay.SOFTWARE_WARNING, "rows": second.rows}
    )
    fixture.rows.extend(second.rows)


def test_grouped_runs_and_condition_shift(tmp_path: Path) -> None:
    fixture = software_fixture(tmp_path)
    add_second_plate(fixture)
    report = fixture.report()
    assert report["n_submitted_independent_experiments"] == 1
    assert len(report["curves"]) == 2
    for curve, midpoint in zip(report["curves"], [1e-6, 1e-5], strict=True):
        assert curve["n_submitted_independent_experiments"] == 1
        assert curve["fit"]["status"] == "estimated"
        assert curve["fit"]["relative_midpoint_M"] == pytest.approx(midpoint, rel=1e-6)
    assert report["input_manifest"]["plates"]["P1"]["conditions_artifact_id"] != "conditions2"


def test_technical_counts_never_inflate_biological_n(tmp_path: Path) -> None:
    two = software_fixture(tmp_path / "two", replicates=2).report()
    twenty = software_fixture(tmp_path / "twenty", replicates=20).report()
    assert (
        two["n_submitted_independent_experiments"]
        == twenty["n_submitted_independent_experiments"]
        == 1
    )
    assert twenty["curves"][0]["n_submitted_independent_experiments"] == 1
    for left, right in zip(
        two["curves"][0]["aggregates"], twenty["curves"][0]["aggregates"], strict=True
    ):
        assert left["n_technical_observed"] == 2 and right["n_technical_observed"] == 20
        assert left["mean_percent_activity"] == pytest.approx(
            right["mean_percent_activity"], abs=1e-12
        )
    assert twenty["curves"][0]["fit"]["uncertainty"] == "not_estimated"


@pytest.mark.parametrize(
    "kind",
    [
        "ragged",
        "duplicate_header",
        "extra_header",
        "duplicate_json",
        "nan_json",
        "underflow_json",
        "unknown_field",
        "bool_policy",
        "wrong_endpoint",
        "mixed_origin",
        "artifact_origin",
        "hash",
        "path",
        "symlink",
        "wrong_role",
        "run_reassigned",
        "null_root",
        "malformed_csv",
    ],
)
def test_schema_and_provenance_errors(tmp_path: Path, kind: str) -> None:
    fixture = software_fixture(tmp_path)
    if kind == "unknown_field":
        fixture.manifest["chemical_name"] = "must remain blinded"
    elif kind == "bool_policy":
        fixture.manifest["qualification"]["policy"]["min_technical_replicates"] = True
    elif kind == "wrong_endpoint":
        fixture.manifest["endpoint"] = "binding_KD"
    elif kind == "mixed_origin":
        fixture.rows[0]["origin"] = "measured"
    elif kind == "artifact_origin":
        fixture.manifest["artifacts"][0]["origin"] = "measured"
    elif kind == "hash":
        fixture.manifest["artifacts"][0]["sha256"] = "0" * 64
    elif kind == "path":
        fixture.manifest["artifacts"][0]["path"] = "../escaped.json"
    elif kind == "symlink":
        outside = tmp_path.parent / "SOFTWARE_TEST_ONLY.outside.json"
        outside.write_text("{}")
        link = tmp_path / "escape.json"
        link.symlink_to(outside)
        fixture.manifest["artifacts"][0]["path"] = link.name
    elif kind == "wrong_role":
        fixture.rows[0]["source_id"] = "controls"
    elif kind == "run_reassigned":
        add_second_plate(fixture)
        fixture.manifest["plates"]["P2"].update(run_id="R1", independent_experiment_id="E2")
    fixture.save()
    if kind in {"ragged", "duplicate_header", "extra_header", "malformed_csv"}:
        text = fixture.csv_path.read_text()
        if kind == "ragged":
            text += "too,few\n"
        elif kind == "duplicate_header":
            text = text.replace("observation_id,origin", "origin,origin", 1)
        elif kind == "extra_header":
            text = "extra," + text
        else:
            text += '"unclosed\n'
        fixture.csv_path.write_text(text)
    elif kind in {"duplicate_json", "nan_json", "underflow_json", "null_root"}:
        text = fixture.manifest_path.read_text()
        if kind == "duplicate_json":
            text = text.replace(
                '"origin": "simulated",', '"origin": "simulated", "origin": "simulated",', 1
            )
        elif kind == "null_root":
            text = "null"
        else:
            text = text.replace(
                '"min_response_span_pp": 10',
                '"min_response_span_pp": ' + ("NaN" if kind == "nan_json" else "1e-999"),
            )
        fixture.manifest_path.write_text(text)
    assert_invalid(fixture)


@pytest.mark.parametrize(
    "section,key,reason",
    [
        ("qualification", "policy", "qualification_evidence_or_policy_pending"),
        ("qualification", "reviewer", "qualification_evidence_or_policy_pending"),
        ("qualification", "evidence_artifact_id", "qualification_evidence_or_policy_pending"),
        ("target", "species", "species_or_construct_pending"),
        ("target", "construct_id", "species_or_construct_pending"),
        ("target", "construct_artifact_id", "species_or_construct_pending"),
        ("plate", "conditions_artifact_id", "conditions_normalization_or_qc_pending"),
        ("plate", "normalization_artifact_id", "conditions_normalization_or_qc_pending"),
        ("plate", "qc_artifact_id", "conditions_normalization_or_qc_pending"),
    ],
)
def test_pending_qualification_retains_rows_without_solver(
    tmp_path: Path,
    section: str,
    key: str,
    reason: str,
) -> None:
    fixture = software_fixture(tmp_path)
    destination = (
        fixture.manifest["plates"]["P1"] if section == "plate" else fixture.manifest[section]
    )
    destination[key] = None
    with patch("assay.least_squares", side_effect=AssertionError("not qualified")):
        report = fixture.report()
    assert_refusal(report, reason)
    assert report["observations"] == fixture.rows


@pytest.mark.parametrize(
    "field,value,reason",
    [
        ("qc_status", "fail", "plate_qc_not_passed"),
        ("qc_status", "pending", "plate_qc_not_passed"),
        ("normalization_reviewed", False, "conditions_normalization_or_qc_pending"),
    ],
)
def test_qc_and_conversion_review_abstain(
    tmp_path: Path, field: str, value: Any, reason: str
) -> None:
    fixture = software_fixture(tmp_path)
    fixture.manifest["plates"]["P1"][field] = value
    assert_refusal(fixture.report(), reason)


def test_pending_assay_and_too_few_technical_wells(tmp_path: Path) -> None:
    fixture = software_fixture(tmp_path, replicates=1)
    fixture.manifest["qualification"]["status"] = "pending"
    report = fixture.report()
    assert_refusal(report, "assay_qualification_pending")
    assert_refusal(report, "insufficient_technical_replicates")
    assert all(a["technical_sd_pp"] is None for a in report["curves"][0]["aggregates"])


@pytest.mark.parametrize(
    "document,key,value",
    [
        ("conditions", "endpoint", "viability"),
        ("conditions", "species", "other"),
        ("conditions", "construct_id", "other"),
        ("conditions", "substrate", "ATP"),
        ("conditions", "enzyme_concentration_M", 0),
        ("conditions", "reaction_minutes", -1),
        ("conditions", "substrate_concentration_M", True),
        ("conditions", "preincubation_minutes", -1),
        ("conditions", "pH", 15),
        ("conditions", "cosolvent_percent", 101),
        ("conditions", "temperature_C", "unknown"),
        ("conditions", "buffer_description", ""),
        ("conditions", "extra", "unexpected"),
        ("normalization", "vehicle_mean", 0),
        ("normalization", "blank_mean", True),
        ("normalization", "plate_id", "other"),
        ("normalization", "formula", "100*signal/vehicle"),
        ("normalization", "blank_control_ids", []),
        ("normalization", "blank_control_ids", ["V1"]),
        ("normalization", "vehicle_control_ids", ["V1", "V1"]),
        ("normalization", "controls_artifact_id", "raw"),
        ("normalization", "controls_artifact_id", []),
        ("normalization", "controls_artifact_id", None),
        ("normalization", "conversion_description", ""),
    ],
)
def test_metadata_semantics_fail_closed(
    tmp_path: Path, document: str, key: str, value: Any
) -> None:
    fixture = software_fixture(tmp_path)
    content = fixture.document(document)
    content[key] = value
    fixture.replace_document(document, content)
    fixture.save()
    assert_invalid(fixture)


def parser_only_measured_shape(fixture: SoftwareFixture, phase: str = "validation") -> None:
    """Isolated SOFTWARE TEST ONLY parser objects; never analyze or publish measured reports."""
    fixture.manifest.update(origin="measured", phase=phase)
    fixture.manifest["compounds"]["BLD-0001"] = "known_control"
    fixture.manifest["qualification"]["status"] = "qualified"
    fixture.manifest["plates"]["P1"]["qc_status"] = "pass"
    fixture.manifest["lock"]["status"] = "pending"
    for artifact in fixture.manifest["artifacts"]:
        artifact["origin"] = "measured"
    for row in fixture.rows:
        row["origin"] = "measured"
    fixture.save()


def add_parser_only_lock(fixture: SoftwareFixture) -> None:
    parser_only_measured_shape(fixture)
    hashes = {
        "analysis_code_sha256": assay.sha256(Path(assay.__file__).read_bytes()),
        "contract_sha256": assay.sha256((assay.ASSAY / "contract.json").read_bytes()),
        "environment_lock_sha256": assay.sha256((assay.ROOT / "requirements.lock").read_bytes()),
    }
    stamp = "2020-01-01T00:00:00Z"
    fixture.manifest["lock"].update(
        status="locked",
        locked_at_utc=stamp,
        first_validation_acquisition_utc="2020-01-02T00:00:00Z",
        artifact_id="lock",
        **hashes,
    )
    fixture.add_artifact(
        "lock",
        "analysis_lock",
        {
            "locked_at_utc": stamp,
            "plan_sha256": assay.canonical_hash(assay.plan_projection(fixture.manifest)),
            "custodian": "SOFTWARE TESTS ONLY",
            "unblinding_rule": "SOFTWARE TESTS ONLY; never unblind",
            "independent_unit_definition": "SOFTWARE TESTS ONLY",
            "pilot_exclusion_rule": "SOFTWARE TESTS ONLY",
            **hashes,
        },
    )
    fixture.save()


@pytest.mark.parametrize(
    "phase,expected", [("pilot", []), ("validation", ["validation_lock_pending"])]
)
def test_software_parser_only_measured_lock_prerequisites(
    tmp_path: Path, phase: str, expected: list[str]
) -> None:
    fixture = software_fixture(tmp_path)
    parser_only_measured_shape(fixture, phase)
    package = assay.load_package(fixture.manifest_path, fixture.csv_path)
    assert assay.qualification_reasons(package.manifest, "P1") == expected
    assert not list(tmp_path.glob("*report*"))


def test_software_parser_only_valid_lock_and_projection(tmp_path: Path) -> None:
    fixture = software_fixture(tmp_path)
    add_parser_only_lock(fixture)
    package = assay.load_package(fixture.manifest_path, fixture.csv_path)
    assert assay.qualification_reasons(package.manifest, "P1") == []
    initial = assay.canonical_hash(assay.plan_projection(fixture.manifest))
    changed = copy.deepcopy(fixture.manifest)
    changed["plates"]["P1"]["qc_status"] = "fail"
    changed["lock"]["locked_at_utc"] = "later"
    assert assay.canonical_hash(assay.plan_projection(changed)) == initial
    changed["qualification"]["policy"]["max_fit_rms_pp"] += 1
    assert assay.canonical_hash(assay.plan_projection(changed)) != initial


@pytest.mark.parametrize(
    "tamper",
    [
        "policy",
        "timestamp",
        "after_acquisition",
        "naive_time",
        "code",
        "contract",
        "environment",
        "plan",
        "custodian",
        "conditions",
        "construct",
        "role",
        "lock_extra",
        "lock_missing",
    ],
)
def test_software_parser_only_lock_tampering(tmp_path: Path, tamper: str) -> None:
    fixture = software_fixture(tmp_path)
    add_parser_only_lock(fixture)
    if tamper == "policy":
        fixture.manifest["qualification"]["policy"]["min_response_span_pp"] += 1
    elif tamper == "timestamp":
        fixture.manifest["lock"]["locked_at_utc"] = "2020-01-01T01:00:00Z"
    elif tamper == "after_acquisition":
        fixture.manifest["lock"]["first_validation_acquisition_utc"] = "2019-01-01T00:00:00Z"
    elif tamper == "naive_time":
        fixture.manifest["lock"]["locked_at_utc"] = "2020-01-01T00:00:00"
    elif tamper in {"code", "contract", "environment"}:
        key = {
            "code": "analysis_code_sha256",
            "contract": "contract_sha256",
            "environment": "environment_lock_sha256",
        }[tamper]
        fixture.manifest["lock"][key] = "0" * 64
    elif tamper == "conditions":
        content = fixture.document("conditions")
        content["reaction_minutes"] = 30
        fixture.replace_document("conditions", content)
    elif tamper == "construct":
        fixture.replace_document("construct", {"warning": assay.SOFTWARE_WARNING, "changed": True})
    elif tamper == "role":
        fixture.manifest["compounds"]["BLD-0001"] = "candidate"
    elif tamper == "lock_missing":
        fixture.manifest["lock"]["artifact_id"] = None
    else:
        content = fixture.document("lock")
        key = {"plan": "plan_sha256", "custodian": "custodian", "lock_extra": "extra"}[tamper]
        content[key] = "" if tamper == "custodian" else "0" * 64
        fixture.replace_document("lock", content)
    fixture.save()
    assert_invalid(fixture)


@pytest.mark.parametrize(
    "kind",
    [
        "error",
        "unsuccessful",
        "nonfinite",
        "bound",
        "zero_jacobian",
        "rank_deficient",
        "condition",
        "multistart",
    ],
)
def test_optimizer_failure_and_identifiability_refuse(tmp_path: Path, kind: str) -> None:
    fixture = software_fixture(tmp_path)
    original = least_squares
    count = 0

    def injected(*args: Any, **kwargs: Any) -> Any:
        nonlocal count
        if kind == "error":
            raise ValueError("SOFTWARE TEST ONLY injected solver failure")
        result = cast(Any, original)(*args, **kwargs)
        count += 1
        if kind == "unsuccessful":
            result.success = False
        elif kind == "nonfinite":
            result.x[0] = float("nan")
        elif kind == "bound":
            result.active_mask[0] = -1
        elif kind == "zero_jacobian":
            result.jac[:, 0] = 0
        elif kind == "rank_deficient":
            result.jac[:, 1] = result.jac[:, 0]
        elif kind == "multistart":
            result.x[2] += 0.02 * count
        return result

    expected = {
        "error": "optimizer_failure",
        "unsuccessful": "optimizer_failure",
        "nonfinite": "optimizer_failure",
        "bound": "numeric_bound_active",
        "zero_jacobian": "nonidentifiable",
        "rank_deficient": "nonidentifiable",
        "condition": "ill_conditioned",
        "multistart": "nonunique_midpoint",
    }[kind]
    with patch("assay.least_squares", side_effect=injected):
        if kind == "condition":
            with patch("numpy.linalg.cond", return_value=1e9):
                report = fixture.report()
        else:
            report = fixture.report()
    assert_refusal(report, expected)


def test_model_misfit_and_midpoint_extrapolation_refuse(tmp_path: Path) -> None:
    fixture = software_fixture(tmp_path / "misfit")
    for row in fixture.rows:
        index = int(row["source_row"])
        row["response_percent_activity"] = str(
            float(row["response_percent_activity"]) + 10 * (index % 3 - 1)
        )
    assert_refusal(fixture.report(), "model_misfit")
    outside = software_fixture(tmp_path / "outside", midpoint=0.002)
    assert_refusal(outside.report(), "relative_midpoint_unbracketed")


@pytest.mark.parametrize("bottom,top", [(50, 100), (0, 50), (49.999, 100)])
def test_absolute_endpoint_or_unobserved_crossing_not_reported(
    tmp_path: Path, bottom: float, top: float
) -> None:
    report = software_fixture(tmp_path, bottom=bottom, top=top).report()
    fit = report["curves"][0]["fit"]
    assert fit["status"] == "estimated", fit
    assert fit["relative_midpoint_M"] == pytest.approx(1e-6, rel=1e-6)
    assert fit["absolute_50_vehicle_M"] is None
    assert fit["absolute_50_status"] in {
        "no_finite_crossing_in_fitted_asymptotes",
        "observations_do_not_bracket_50",
    }


def test_determinism_row_order_and_provenance(tmp_path: Path) -> None:
    fixture = software_fixture(tmp_path)
    first = fixture.report()
    second = fixture.report()
    assert first["curves"] == second["curves"]
    fixture.rows.reverse()
    reverse = fixture.report()
    assert first["curves"] == reverse["curves"]
    assert (
        first["provenance"]["observations_sha256"] != reverse["provenance"]["observations_sha256"]
    )
    assert reverse["observations"] == fixture.rows
    for key in (
        "analysis_code_sha256",
        "contract_sha256",
        "environment_lock_sha256",
        "manifest_sha256",
        "manifest_schema_sha256",
        "output_writer_sha256",
    ):
        assert len(first["provenance"][key]) == 64
    assert first["provenance"]["python"].startswith("3.12.")
    assert first["provenance"]["numpy"] == "2.2.6"
    assert first["provenance"]["scipy"] == "1.15.3"


def test_cli_from_different_cwd_and_no_overwrite(tmp_path: Path) -> None:
    fixture = software_fixture(tmp_path / "package")
    output = tmp_path / "SOFTWARE_TEST_ONLY.report.json"
    command = [
        sys.executable,
        str(Path(assay.__file__).resolve()),
        "--manifest",
        str(fixture.manifest_path),
        "--observations",
        str(fixture.csv_path),
        "--output",
        str(output),
    ]
    result = subprocess.run(command, cwd=tmp_path, capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    contents = output.read_bytes()
    report = assay.strict_json(contents)
    assert "SOFTWARE TESTS ONLY" in report["warning"]
    assert report["curves"][0]["fit"]["status"] == "estimated"
    assert fixture.cli(output) == 2
    assert output.read_bytes() == contents
    link = tmp_path / "link.json"
    link.symlink_to(output)
    assert fixture.cli(link) == 2
    assert output.read_bytes() == contents
    dangling = tmp_path / "dangling.json"
    dangling.symlink_to(tmp_path / "absent.json")
    assert fixture.cli(dangling) == 2
    assert dangling.is_symlink()


@pytest.mark.parametrize("operation", ["link", "fsync", "read", "missing_parent", "race"])
def test_io_failure_never_publishes_partial_output(tmp_path: Path, operation: str) -> None:
    fixture = software_fixture(tmp_path)
    output = tmp_path / "SOFTWARE_TEST_ONLY.output.json"
    if operation == "missing_parent":
        output = tmp_path / "missing" / output.name
        assert fixture.cli(output) == 2
    elif operation == "read":
        with patch.object(
            Path, "read_bytes", side_effect=OSError("SOFTWARE TEST ONLY read failure")
        ):
            assert fixture.cli(output) == 2
    elif operation == "race":
        link = os.link

        def racing_link(source: Any, destination: Any) -> None:
            Path(destination).write_text("existing result")
            link(source, destination)

        with patch.object(os, "link", side_effect=racing_link):
            assert fixture.cli(output) == 2
        assert output.read_text() == "existing result"
    else:
        with patch.object(os, operation, side_effect=OSError("SOFTWARE TEST ONLY write failure")):
            assert fixture.cli(output) == 2
    if operation != "race":
        assert not output.exists()
    assert not list(tmp_path.glob(".nudt5-*"))


def test_success_exit_can_mean_all_refusals(tmp_path: Path) -> None:
    fixture = software_fixture(tmp_path, bottom=100)
    output = tmp_path / "SOFTWARE_TEST_ONLY.refusal.json"
    assert fixture.cli(output) == 0
    assert_refusal(assay.strict_json(output.read_bytes()), "flat_or_insufficient_response_span")


def test_derivative_matches_independent_finite_difference() -> None:
    parameters: NDArray[np.float64] = np.array(
        [20, math.log(80), -6, math.log(1.2)], dtype=np.float64
    )
    x: NDArray[np.float64] = np.linspace(-9, -3, 13, dtype=np.float64)
    analytic = assay.response_jacobian(parameters, x)
    for column in range(4):
        delta: NDArray[np.float64] = np.zeros(4, dtype=np.float64)
        delta[column] = 1e-5
        finite = (
            assay.response_model(parameters + delta, x)
            - assay.response_model(parameters - delta, x)
        ) / 2e-5
        np.testing.assert_allclose(analytic[:, column], finite, rtol=1e-6, atol=1e-7)


def test_entirely_absent_curve_has_null_aggregates(tmp_path: Path) -> None:
    fixture = software_fixture(tmp_path)
    for row in fixture.rows:
        row.update(
            observation_status="absent",
            response_percent_activity="",
            response_relation="not_applicable",
            status_reason="SOFTWARE TESTS ONLY absent",
        )
    report = fixture.report()
    assert_refusal(report, "incomplete_or_censored_curve")
    assert all(
        a["mean_percent_activity"] is None
        and a["technical_sd_pp"] is None
        and a["n_technical_observed"] == 0
        for a in report["curves"][0]["aggregates"]
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("observation_status", "unknown"),
        ("response_relation", ">"),
        ("status_reason", "unexpected"),
        ("source_row", "0"),
        ("source_row", "1.0"),
        ("target", "NUDT14"),
    ],
)
def test_observation_semantics(tmp_path: Path, field: str, value: str) -> None:
    fixture = software_fixture(tmp_path)
    fixture.rows[0][field] = value
    fixture.save()
    assert_invalid(fixture)


def test_plate_hierarchy_and_control_roles_remain_separate(tmp_path: Path) -> None:
    fixture = software_fixture(tmp_path)
    add_second_plate(fixture)
    fixture.manifest["plates"]["P2"]["run_id"] = "R1"
    for row in fixture.rows:
        row["run_id"] = "R1"
    fixture.manifest["compounds"]["BLD-0001"] = "known_control"
    report = fixture.report()
    assert len(report["curves"]) == 2
    assert report["n_submitted_independent_experiments"] == 1
    assert all(curve["compound_role"] == "known_control" for curve in report["curves"])
    fixture.manifest["plates"]["P2"].update(run_id="R2", independent_experiment_id="E2")
    for row in fixture.rows:
        if row["plate_id"] == "P2":
            row.update(run_id="R2", independent_experiment_id="E2")
    assert fixture.report()["n_submitted_independent_experiments"] == 2


def test_exact_response_zero_and_hundred_not_clipped(tmp_path: Path) -> None:
    fixture = software_fixture(tmp_path)
    fixture.rows[0]["response_percent_activity"] = "100"
    fixture.rows[-1]["response_percent_activity"] = "0"
    report = fixture.report()
    assert report["observations"][0]["response_percent_activity"] == "100"
    assert report["observations"][-1]["response_percent_activity"] == "0"


def test_strict_json_nonfinite_and_nested_duplicate_keys() -> None:
    for text in ('{"a":{"b":1,"b":2}}', '{"a":Infinity}', '{"a":1e999}', '{"a":1e-999}'):
        with pytest.raises(assay.InputError):
            assay.strict_json(text.encode())


def test_all_planned_curves_validated_before_any_fitting(tmp_path: Path) -> None:
    fixture = software_fixture(tmp_path)
    add_second_plate(fixture)
    fixture.rows = [row for row in fixture.rows if row["plate_id"] != "P2"]
    fixture.save()
    assert_invalid(fixture)


def test_fixture_policy_is_not_analyzer_default() -> None:
    assert CONTRACT["pilot_policy_defaults"] is None
    assert CONTRACT["required_pilot_policy"] == list(STRESS["fixture_policy_only"])


def test_raw_bytes_not_normalization_arithmetic_authenticated(tmp_path: Path) -> None:
    fixture = software_fixture(tmp_path)
    package = assay.load_package(fixture.manifest_path, fixture.csv_path)
    assert package.provenance["artifact_hashes"]["raw"] == assay.sha256(
        (tmp_path / "SOFTWARE_TEST_ONLY.raw.json").read_bytes()
    )
    # The program does not parse vendor raw data; conversion_reviewed is a human attestation.
    fixture.rows[0]["response_percent_activity"] = "99"
    fixture.save()
    assert (
        assay.load_package(fixture.manifest_path, fixture.csv_path).observations[0].response == 99
    )


def test_predeclared_stress_matrix_has_executable_coverage() -> None:
    assert len(STRESS["cases"]) == 25
    assert len({case["id"] for case in STRESS["cases"]}) == 25
    for case in STRESS["cases"]:
        assert case["test_functions"]
        assert all(callable(globals().get(name)) for name in case["test_functions"])


def test_atomic_serialization_failure_has_no_staging_file(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        pipeline_write_json(tmp_path / "SOFTWARE_TEST_ONLY.invalid.json", {"value": float("nan")})
    assert not list(tmp_path.iterdir())


def test_explicit_repository_remaps_contract_and_hashes_actual_writer(tmp_path: Path) -> None:
    import shutil

    fixture = software_fixture(tmp_path / "SOFTWARE_TEST_ONLY")
    fixture.save()
    repository = tmp_path / "evidence"
    contracts = repository / "research/assay"
    contracts.mkdir(parents=True)
    for name in ("contract.json", "manifest.schema.json"):
        shutil.copyfile(assay.ASSAY / name, contracts / name)
    shutil.copyfile(assay.ROOT / "requirements.lock", repository / "requirements.lock")
    package = assay.load_package(fixture.manifest_path, fixture.csv_path, repository=repository)
    assert package.provenance["output_writer_sha256"] == assay.sha256(
        Path(pipeline.__file__).read_bytes()
    )
    output = tmp_path / "SOFTWARE_TEST_ONLY.report.json"
    assert (
        assay.main(
            [
                "--manifest",
                str(fixture.manifest_path),
                "--observations",
                str(fixture.csv_path),
                "--repository",
                str(repository),
                "--output",
                str(output),
            ]
        )
        == 0
    )
    report = assay.strict_json(output.read_bytes())
    assert "SOFTWARE TESTS ONLY" in report["warning"]
    (contracts / "contract.json").unlink()
    refused = tmp_path / "refused.json"
    assert (
        assay.main(
            [
                "--manifest",
                str(fixture.manifest_path),
                "--observations",
                str(fixture.csv_path),
                "--repository",
                str(repository),
                "--output",
                str(refused),
            ]
        )
        == 2
    )
    assert not refused.exists()
