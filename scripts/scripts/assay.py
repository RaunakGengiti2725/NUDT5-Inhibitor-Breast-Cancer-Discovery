"""Blinded, descriptive hydrolysis curves; software checks are not assay qualification."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import itertools
import json
import math
import platform
import re
import sys
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from importlib.metadata import version
from pathlib import Path
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray
from pipeline import write_json
from scipy.optimize import least_squares
from scipy.special import expit

ROOT = Path(__file__).resolve().parents[2]
ASSAY = ROOT / "research" / "assay"
VERSION = "nudt5-assay-v1"
ENDPOINT = "biochemical_ADPr_hydrolysis_percent_activity"
FORMULA = "100*(signal-blank)/(vehicle-blank)"
SOFTWARE_WARNING = (
    "SOFTWARE TESTS ONLY; simulated data, not experimental evidence or qualification."
)
NUMBER = re.compile(r"[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?\Z")
UNIT_POWERS = {"M": 0, "mM": -3, "uM": -6, "nM": -9, "pM": -12}
FloatArray = NDArray[np.float64]
CurveKey = tuple[str, str, str, str]


class InputError(ValueError):
    """The complete package must be rejected before any fitting/publication."""


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_hash(value: Any) -> str:
    return sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
        ).encode("utf-8")
    )


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise InputError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def decimal_number(text: str) -> Decimal:
    if not NUMBER.fullmatch(text):
        raise InputError("Expected an unpadded finite ASCII number")
    try:
        value = Decimal(text)
        converted = float(value)
    except (ValueError, OverflowError, InvalidOperation) as exc:
        raise InputError("Invalid number") from exc
    if not math.isfinite(converted) or (value != 0 and converted == 0):
        raise InputError("Nonfinite, underflow or overflow number")
    return value


def _json_float(text: str) -> float:
    return float(decimal_number(text))


def _invalid_constant(text: str) -> Any:
    raise InputError(f"Nonfinite JSON constant: {text}")


def strict_json(data: bytes) -> dict[str, Any]:
    try:
        result = json.loads(
            data.decode("utf-8"),
            object_pairs_hook=_unique_object,
            parse_float=_json_float,
            parse_constant=_invalid_constant,
        )
    except (UnicodeError, ValueError, RecursionError) as exc:
        raise InputError(f"Invalid JSON: {exc}") from exc
    if not isinstance(result, dict):
        raise InputError("JSON object required")
    return result


def _number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise InputError(f"{name}: numeric value required")
    try:
        result = float(value)
    except OverflowError as exc:
        raise InputError(f"{name}: numeric overflow") from exc
    if not math.isfinite(result):
        raise InputError(f"{name}: finite number required")
    return result


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise InputError(f"{name}: nonblank, unpadded string required")
    return value


def _fields(value: dict[str, Any], fields: str, name: str) -> None:
    if set(value) != set(fields.split()):
        raise InputError(f"{name}: exact fields required: {fields}")


def _schema(value: Any, schema: dict[str, Any], name: str = "manifest") -> None:
    """Validate the closed subset used by the shipped manifest schema, not arbitrary schemas."""
    if "anyOf" in schema:
        for option in schema["anyOf"]:
            try:
                _schema(value, option, name)
                return
            except InputError:
                pass
        raise InputError(f"{name}: no permitted schema matches")
    types = schema.get("type", [])
    if isinstance(types, str):
        types = [types]
    checks = {
        "null": value is None,
        "object": isinstance(value, dict),
        "array": isinstance(value, list),
        "string": isinstance(value, str),
        "boolean": isinstance(value, bool),
        "number": isinstance(value, (int, float)) and not isinstance(value, bool),
        "integer": isinstance(value, int) and not isinstance(value, bool),
    }
    if types and not any(checks[t] for t in types):
        raise InputError(f"{name}: wrong type")
    if "enum" in schema and value not in schema["enum"]:
        raise InputError(f"{name}: unexpected value")
    if isinstance(value, dict):
        if not set(schema.get("required", [])).issubset(value):
            raise InputError(f"{name}: missing fields")
        if len(value) < schema.get("minProperties", 0):
            raise InputError(f"{name}: empty object")
        properties = schema.get("properties", {})
        extra = schema.get("additionalProperties", True)
        for key, item in value.items():
            _text(key, name)
            if "propertyNames" in schema:
                _schema(key, schema["propertyNames"], name)
            if key in properties:
                _schema(item, properties[key], f"{name}.{key}")
            elif extra is False:
                raise InputError(f"{name}: unknown field {key}")
            elif isinstance(extra, dict):
                _schema(item, extra, f"{name}.{key}")
    elif isinstance(value, list):
        if len(value) < schema.get("minItems", 0):
            raise InputError(f"{name}: too few items")
        if schema.get("uniqueItems") and len({canonical_hash(v) for v in value}) != len(value):
            raise InputError(f"{name}: duplicate items")
        for item in value:
            _schema(item, schema.get("items", {}), name)
    elif isinstance(value, str):
        _text(value, name)
        if len(value) < schema.get("minLength", 0) or (
            "pattern" in schema and not re.search(schema["pattern"], value)
        ):
            raise InputError(f"{name}: invalid string")
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        number = _number(value, name)
        if ("minimum" in schema and number < schema["minimum"]) or (
            "exclusiveMinimum" in schema and number <= schema["exclusiveMinimum"]
        ):
            raise InputError(f"{name}: outside numeric range")


def molar_dose(text: str, unit: str) -> Decimal:
    if unit not in UNIT_POWERS:
        raise InputError("Unsupported concentration unit")
    value = decimal_number(text)
    sign, digits, exponent = value.as_tuple()
    assert isinstance(exponent, int)
    dose = Decimal((sign, digits, exponent + UNIT_POWERS[unit]))
    number = float(dose)
    if dose <= 0 or not math.isfinite(number) or number == 0:
        raise InputError("Concentration must be positive, finite and representable in M")
    return dose


@dataclass(frozen=True)
class Observation:
    raw: dict[str, str]
    dose_M: Decimal
    response: float | None

    @property
    def curve_key(self) -> CurveKey:
        return (
            self.raw["blinded_compound_id"],
            self.raw["independent_experiment_id"],
            self.raw["run_id"],
            self.raw["plate_id"],
        )


@dataclass(frozen=True)
class PilotPolicy:
    min_response_span_pp: float
    max_fit_rms_pp: float
    plateau_tolerance_pp: float
    min_technical_replicates: int


@dataclass(frozen=True)
class AssayPackage:
    manifest: dict[str, Any]
    observations: tuple[Observation, ...]
    contract: dict[str, Any]
    provenance: dict[str, Any]


def _artifact_ref(
    identifier: Any, role: str, registry: Mapping[str, dict[str, Any]]
) -> dict[str, Any] | None:
    if identifier is None:
        return None
    if (
        not isinstance(identifier, str)
        or identifier not in registry
        or registry[identifier]["role"] != role
    ):
        raise InputError(f"Artifact must reference role {role}")
    return registry[identifier]


def plan_projection(manifest: dict[str, Any]) -> dict[str, Any]:
    """Build the non-circular pre-acquisition commitment specified by contract v1."""
    artifacts = {a["id"]: a for a in manifest["artifacts"]}

    def digest(identifier: str | None) -> str | None:
        return artifacts[identifier]["sha256"] if identifier is not None else None

    return {
        **{
            k: manifest[k]
            for k in (
                "schema_version",
                "origin",
                "phase",
                "endpoint",
                "target",
                "compounds",
                "design",
                "qualification",
            )
        },
        "qualification_evidence_sha256": digest(manifest["qualification"]["evidence_artifact_id"]),
        "construct_sha256": digest(manifest["target"]["construct_artifact_id"]),
        "plates": {
            pid: {
                **{
                    k: plate[k]
                    for k in (
                        "run_id",
                        "independent_experiment_id",
                        "normalization_id",
                        "normalization_formula",
                    )
                },
                "conditions_sha256": digest(plate["conditions_artifact_id"]),
            }
            for pid, plate in manifest["plates"].items()
        },
    }


def _timestamp(text: Any) -> datetime:
    value = _text(text, "UTC timestamp")
    if not re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d(?:\.\d+)?(?:Z|\+00:00)", value):
        raise InputError("Explicit UTC ISO timestamp required")
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise InputError("Invalid UTC timestamp") from exc


def _validate_lock(
    manifest: dict[str, Any],
    registry: dict[str, dict[str, Any]],
    contents: dict[str, bytes],
    hashes: dict[str, str],
) -> None:
    lock = manifest["lock"]
    artifact = _artifact_ref(lock["artifact_id"], "analysis_lock", registry)
    if lock["status"] != "locked":
        if any(lock[k] is not None for k in lock if k != "status"):
            raise InputError("Pending/software lock must not carry an apparent commitment")
        return
    if artifact is None:
        raise InputError("Locked analysis requires its artifact")
    locked = _timestamp(lock["locked_at_utc"])
    acquired = _timestamp(lock["first_validation_acquisition_utc"])
    if locked >= acquired:
        raise InputError("Lock must precede first validation acquisition")
    document = strict_json(contents[artifact["id"]])
    _fields(
        document,
        "locked_at_utc plan_sha256 custodian unblinding_rule independent_unit_definition "
        "pilot_exclusion_rule analysis_code_sha256 contract_sha256 environment_lock_sha256",
        "lock",
    )
    for key in (
        "custodian",
        "unblinding_rule",
        "independent_unit_definition",
        "pilot_exclusion_rule",
    ):
        _text(document[key], key)
    if document["locked_at_utc"] != lock["locked_at_utc"]:
        raise InputError("Lock timestamp mismatch")
    for key in ("analysis_code_sha256", "contract_sha256", "environment_lock_sha256"):
        if document[key] != hashes[key] or lock[key] != hashes[key]:
            raise InputError(f"Lock byte commitment mismatch: {key}")
    if document["plan_sha256"] != canonical_hash(plan_projection(manifest)):
        raise InputError("Lock plan commitment mismatch")


def _validate_conditions(document: dict[str, Any], target: dict[str, Any]) -> None:
    _fields(
        document,
        "endpoint species construct_id enzyme_concentration_M substrate substrate_concentration_M "
        "reaction_minutes preincubation_minutes temperature_C pH buffer_description cosolvent "
        "cosolvent_percent detergent_description readout protocol_id",
        "conditions",
    )
    for key in (
        "species",
        "construct_id",
        "buffer_description",
        "cosolvent",
        "detergent_description",
        "readout",
        "protocol_id",
    ):
        _text(document[key], key)
    if document["endpoint"] != ENDPOINT or document["substrate"] != "ADP-ribose":
        raise InputError("Condition endpoint/substrate mismatch")
    for key in ("species", "construct_id"):
        if target[key] is not None and target[key] != document[key]:
            raise InputError(f"Condition/target {key} mismatch")
    for key in ("enzyme_concentration_M", "substrate_concentration_M", "reaction_minutes"):
        if _number(document[key], key) <= 0:
            raise InputError(f"{key} must be positive")
    if _number(document["preincubation_minutes"], "preincubation") < 0:
        raise InputError("Preincubation must be nonnegative")
    for key, limit in (("pH", 14), ("cosolvent_percent", 100)):
        if not 0 <= _number(document[key], key) <= limit:
            raise InputError(f"{key} outside permitted range")
    _number(document["temperature_C"], "temperature")


def _validate_normalization(
    document: dict[str, Any], pid: str, plate: dict[str, Any], registry: dict[str, dict[str, Any]]
) -> None:
    _fields(
        document,
        "plate_id normalization_id formula signal_unit blank_mean vehicle_mean blank_control_ids "
        "vehicle_control_ids controls_artifact_id conversion_description",
        "normalization",
    )
    if (
        document["plate_id"] != pid
        or document["normalization_id"] != plate["normalization_id"]
        or document["formula"] != FORMULA
    ):
        raise InputError("Normalization/plate mismatch")
    for key in ("signal_unit", "conversion_description"):
        _text(document[key], key)
    denominator = _number(document["vehicle_mean"], "vehicle") - _number(
        document["blank_mean"], "blank"
    )
    if not math.isfinite(denominator) or denominator <= 0:
        raise InputError("Normalization denominator must be positive finite")
    ids: list[str] = []
    for key in ("blank_control_ids", "vehicle_control_ids"):
        if not isinstance(document[key], list) or not document[key]:
            raise InputError("Explicit control IDs required")
        ids.extend(_text(value, key) for value in document[key])
    if len(ids) != len(set(ids)):
        raise InputError("Normalization control IDs must be unique and disjoint")
    if _artifact_ref(document["controls_artifact_id"], "raw_controls", registry) is None:
        raise InputError("Raw control artifact required")


def _validate_metadata(
    manifest: dict[str, Any],
    registry: dict[str, dict[str, Any]],
    contents: dict[str, bytes],
    hashes: dict[str, str],
) -> None:
    simulated = manifest["origin"] == "simulated"
    if simulated != (manifest["phase"] == "software_test"):
        raise InputError("Origin/phase mismatch")
    qualification = manifest["qualification"]
    if simulated:
        if (
            qualification["status"] == "qualified"
            or manifest["lock"]["status"] != "software_test_only"
        ):
            raise InputError("Simulated package requires software-only qualification/lock")
    elif (
        qualification["status"] == "software_test_only"
        or manifest["lock"]["status"] == "software_test_only"
        or "synthetic_test" in manifest["compounds"].values()
    ):
        raise InputError("Measured package cannot claim software-only roles")
    _artifact_ref(manifest["target"]["construct_artifact_id"], "construct", registry)
    _artifact_ref(qualification["evidence_artifact_id"], "qualification", registry)
    runs: dict[str, str] = {}
    for pid, plate in manifest["plates"].items():
        run, experiment = plate["run_id"], plate["independent_experiment_id"]
        if run in runs and runs[run] != experiment:
            raise InputError("A run cannot belong to multiple independent experiments")
        runs[run] = experiment
        if (simulated and plate["qc_status"] == "pass") or (
            not simulated and plate["qc_status"] == "software_test_only"
        ):
            raise InputError("Origin/QC mismatch")
        _artifact_ref(plate["qc_artifact_id"], "plate_qc", registry)
        conditions = _artifact_ref(plate["conditions_artifact_id"], "conditions", registry)
        normalization = _artifact_ref(plate["normalization_artifact_id"], "normalization", registry)
        if conditions:
            _validate_conditions(strict_json(contents[conditions["id"]]), manifest["target"])
        if normalization:
            _validate_normalization(
                strict_json(contents[normalization["id"]]), pid, plate, registry
            )
    _validate_lock(manifest, registry, contents, hashes)


def _read_observations(
    data: bytes,
    manifest: dict[str, Any],
    contract: dict[str, Any],
    registry: dict[str, dict[str, Any]],
) -> tuple[Observation, ...]:
    try:
        records = list(csv.reader(io.StringIO(data.decode("utf-8-sig"), newline=""), strict=True))
    except (UnicodeError, csv.Error) as exc:
        raise InputError(f"Invalid CSV: {exc}") from exc
    if not records or records[0] != contract["csv_columns"] or len(records) == 1:
        raise InputError("Nonempty CSV with exact ordered headers required")
    expected: set[tuple[str, str, Decimal, str]] = set()
    designs: set[tuple[str, str]] = set()
    float_doses: dict[float, Decimal] = {}

    def check_resolution(dose: Decimal) -> None:
        key = float(dose)
        if key in float_doses and float_doses[key] != dose:
            raise InputError("Distinct concentrations collapse at floating-point resolution")
        float_doses[key] = dose

    for design in manifest["design"]:
        cid, pid = design["blinded_compound_id"], design["plate_id"]
        if (
            cid not in manifest["compounds"]
            or pid not in manifest["plates"]
            or (cid, pid) in designs
        ):
            raise InputError("Unknown or duplicate planned curve")
        designs.add((cid, pid))
        doses = [molar_dose(str(d), "M") for d in design["concentrations_M"]]
        if len(set(doses)) != len(doses):
            raise InputError("Duplicate planned concentrations")
        for dose in doses:
            check_resolution(dose)
        expected.update(
            (cid, pid, dose, tid) for dose in doses for tid in design["technical_replicate_ids"]
        )
    seen: set[tuple[str, str, Decimal, str]] = set()
    wells: set[tuple[str, str]] = set()
    observations: set[str] = set()
    sources: set[tuple[str, int]] = set()
    rows: list[Observation] = []
    optional = {"response_percent_activity", "status_reason"}
    for cells in records[1:]:
        if len(cells) != len(contract["csv_columns"]):
            raise InputError("Ragged or empty CSV row")
        row = dict(zip(contract["csv_columns"], cells, strict=True))
        if any(v != v.strip() or (not v and k not in optional) for k, v in row.items()):
            raise InputError("Missing or whitespace-padded CSV cell")
        if row["origin"] != manifest["origin"] or row["target"] != "NUDT5":
            raise InputError("Observation origin/target mismatch")
        plate = manifest["plates"].get(row["plate_id"])
        if plate is None or any(
            row[k] != plate[k] for k in ("run_id", "independent_experiment_id", "normalization_id")
        ):
            raise InputError("Observation/plate registry mismatch")
        dose = molar_dose(row["concentration"], row["concentration_unit"])
        check_resolution(dose)
        key = (row["blinded_compound_id"], row["plate_id"], dose, row["technical_replicate_id"])
        well = (row["plate_id"], row["well_id"])
        if (
            key not in expected
            or key in seen
            or well in wells
            or row["observation_id"] in observations
        ):
            raise InputError("Unplanned/duplicate technical identity, well or observation")
        source = _artifact_ref(row["source_id"], "raw_observations", registry)
        if source is None or not re.fullmatch(r"[1-9][0-9]*", row["source_row"]):
            raise InputError("Positive integer source record index required")
        source_key = (source["sha256"], int(row["source_row"]))
        if source_key in sources:
            raise InputError("Duplicate source bytehash/record index")
        status, relation, reason = (
            row["observation_status"],
            row["response_relation"],
            row["status_reason"],
        )
        response = None
        if status in {"observed", "censored"}:
            response = float(decimal_number(row["response_percent_activity"]))
            if abs(response) >= contract["numeric_policy"]["response_absolute_limit_exclusive_pp"]:
                raise InputError("Response outside numeric support")
            if (status == "observed" and (relation != "=" or reason)) or (
                status == "censored" and (relation not in {"<", "<=", ">", ">="} or not reason)
            ):
                raise InputError("Invalid observation relation/reason")
        elif status not in {"absent", "failed", "untested"} or (
            row["response_percent_activity"] or relation != "not_applicable" or not reason
        ):
            raise InputError("Invalid observation status/value/reason")
        seen.add(key)
        wells.add(well)
        observations.add(row["observation_id"])
        sources.add(source_key)
        rows.append(Observation(row, dose, response))
    if seen != expected:
        raise InputError("Omitted planned wells: encode absent explicitly")
    return tuple(rows)


def load_package(manifest_path: Path, observations_path: Path) -> AssayPackage:
    """Read and validate the entire immutable package before any numerical analysis."""
    manifest_bytes = manifest_path.read_bytes()
    csv_bytes = observations_path.read_bytes()
    contract_bytes = (ASSAY / "contract.json").read_bytes()
    schema_bytes = (ASSAY / "manifest.schema.json").read_bytes()
    contract = strict_json(contract_bytes)
    manifest = strict_json(manifest_bytes)
    _schema(manifest, strict_json(schema_bytes))
    registry: dict[str, dict[str, Any]] = {}
    contents: dict[str, bytes] = {}
    base = manifest_path.resolve().parent
    for artifact in manifest["artifacts"]:
        identifier = artifact["id"]
        relative = Path(artifact["path"])
        path = (base / relative).resolve()
        if relative.is_absolute() or not path.is_relative_to(base) or identifier in registry:
            raise InputError("Artifact path escapes package or duplicate artifact ID")
        if artifact["origin"] != manifest["origin"]:
            raise InputError("Artifact origin mismatch")
        data = path.read_bytes()
        if not data or sha256(data) != artifact["sha256"]:
            raise InputError("Artifact hash mismatch or empty evidence")
        registry[identifier], contents[identifier] = artifact, data
    hashes = {
        "analysis_code_sha256": sha256(Path(__file__).read_bytes()),
        "contract_sha256": sha256(contract_bytes),
        "environment_lock_sha256": sha256((ROOT / "requirements.lock").read_bytes()),
    }
    _validate_metadata(manifest, registry, contents, hashes)
    observations = _read_observations(csv_bytes, manifest, contract, registry)
    provenance = {
        **hashes,
        "manifest_sha256": sha256(manifest_bytes),
        "observations_sha256": sha256(csv_bytes),
        "manifest_schema_sha256": sha256(schema_bytes),
        "output_writer_sha256": sha256((ROOT / "scripts/scripts/pipeline.py").read_bytes()),
        "artifact_hashes": {key: value["sha256"] for key, value in sorted(registry.items())},
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": version("scipy"),
    }
    return AssayPackage(manifest, observations, contract, provenance)


def qualification_reasons(manifest: dict[str, Any], plate_id: str) -> list[str]:
    """Report missing attestations; never infer qualification from a successful fit."""
    qualification, target = manifest["qualification"], manifest["target"]
    plate = manifest["plates"][plate_id]
    simulated = manifest["origin"] == "simulated"
    reasons: list[str] = []
    if any(qualification[k] is None for k in ("policy", "evidence_artifact_id", "reviewer")):
        reasons.append("qualification_evidence_or_policy_pending")
    if any(target[k] is None for k in ("species", "construct_id", "construct_artifact_id")):
        reasons.append("species_or_construct_pending")
    if (
        any(
            plate[k] is None
            for k in ("conditions_artifact_id", "normalization_artifact_id", "qc_artifact_id")
        )
        or not plate["normalization_reviewed"]
    ):
        reasons.append("conditions_normalization_or_qc_pending")
    if qualification["status"] != ("software_test_only" if simulated else "qualified"):
        reasons.append("assay_qualification_pending")
    if plate["qc_status"] != ("software_test_only" if simulated else "pass"):
        reasons.append("plate_qc_not_passed")
    if manifest["phase"] == "validation" and manifest["lock"]["status"] != "locked":
        reasons.append("validation_lock_pending")
    return reasons


def aggregate_observations(observations: Sequence[Observation]) -> list[dict[str, Any]]:
    """Technical means and sample SD only; keep all doses and all submitted units."""
    grouped: dict[Decimal, list[Observation]] = defaultdict(list)
    for observation in observations:
        grouped[observation.dose_M].append(observation)
    aggregates: list[dict[str, Any]] = []
    for dose, rows in sorted(grouped.items()):
        values = [
            r.response
            for r in sorted(rows, key=lambda r: r.raw["technical_replicate_id"])
            if r.raw["observation_status"] == "observed" and r.response is not None
        ]
        n = len(values)
        mean = math.fsum(values) / n if n else None
        sd = (
            math.sqrt(math.fsum((v - mean) ** 2 for v in values) / (n - 1))
            if n >= 2 and mean is not None
            else None
        )
        aggregates.append(
            {
                "concentration_M": float(dose),
                "n_technical_observed": n,
                "mean_percent_activity": mean,
                "technical_sd_pp": sd,
            }
        )
    return aggregates


def response_model(parameters: FloatArray, log10_dose_M: FloatArray) -> FloatArray:
    bottom, log_amplitude, midpoint, log_hill = parameters
    return np.asarray(
        bottom
        + np.exp(log_amplitude)
        * expit(-np.exp(log_hill) * np.log(10.0) * (log10_dose_M - midpoint)),
        dtype=np.float64,
    )


def response_jacobian(parameters: FloatArray, log10_dose_M: FloatArray) -> FloatArray:
    _, log_amplitude, midpoint, log_hill = parameters
    amplitude, hill = np.exp(log_amplitude), np.exp(log_hill)
    delta = log10_dose_M - midpoint
    sigmoid = expit(-hill * np.log(10.0) * delta)
    derivative = amplitude * sigmoid * (1 - sigmoid) * hill * np.log(10.0)
    return np.column_stack(
        (np.ones_like(delta), amplitude * sigmoid, derivative, -derivative * delta)
    )


def refused(reasons: Sequence[str], diagnostics: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "status": "refused",
        "reason": list(reasons),
        "parameters": None,
        "relative_midpoint_M": None,
        "absolute_50_vehicle_M": None,
        "absolute_50_status": "not_estimated",
        "uncertainty": "not_estimated",
        "diagnostics": diagnostics or {},
    }


def fit_response(
    concentrations_M: Sequence[float],
    mean_percent_activity: Sequence[float],
    policy: PilotPolicy,
    numeric: Mapping[str, Any],
) -> dict[str, Any]:
    """Conditional equal-dose 4PL point estimates, never biological precision or extrapolation.

    Call through analyze() for package/qualification checks. This low-level numerical routine
    does not qualify inputs, normalization, materials, or laboratory policy.
    """
    c, y = np.asarray(concentrations_M, dtype=float), np.asarray(mean_percent_activity, dtype=float)
    if (
        c.ndim != 1
        or y.shape != c.shape
        or not np.isfinite(c).all()
        or (c <= 0).any()
        or not np.isfinite(y).all()
        or (np.abs(y) >= numeric["response_absolute_limit_exclusive_pp"]).any()
    ):
        raise InputError("Invalid numerical curve arrays")
    if len(c) != len(np.unique(c)):
        raise InputError("Concentrations must be unique technical means")
    for value in (policy.min_response_span_pp, policy.max_fit_rms_pp, policy.plateau_tolerance_pp):
        if _number(value, "pilot policy") <= 0:
            raise InputError("Positive finite pilot policy required")
    if type(policy.min_technical_replicates) is not int or policy.min_technical_replicates < 1:
        raise InputError("Positive integer technical replicate requirement needed")
    order = np.argsort(c)
    c, y = c[order], y[order]
    if len(c) < numeric["min_concentrations"]:
        return refused(["undersampled"])
    x = np.log10(c)
    span = float(np.ptp(y))
    diagnostics: dict[str, Any] = {"n_concentrations": len(c), "observed_response_span_pp": span}
    if span < policy.min_response_span_pp:
        return refused(["flat_or_insufficient_response_span"], diagnostics)
    if float(np.dot(x - x.mean(), y - y.mean())) >= 0:
        return refused(["inverted_or_non_decreasing"], diagnostics)
    points = numeric["plateau_points_each_end"]
    bottom = float(y[-points:].mean())
    amplitude = float(y[:points].mean()) - bottom
    blo, bhi = numeric["bottom_bounds"]
    alo, ahi = numeric["amplitude_bounds"]
    hlo, hhi = numeric["hill_bounds"]
    extension = numeric["midpoint_search_extension_log10"]
    if not blo < bottom < bhi or not alo < amplitude < ahi:
        return refused(["outside_numeric_support"], diagnostics)
    lower = np.array([blo, math.log(alo), x[0] - extension, math.log(hlo)])
    upper = np.array([bhi, math.log(ahi), x[-1] + extension, math.log(hhi)])
    fits: list[Any] = []
    try:
        for fraction, hill in itertools.product(
            numeric["start_midpoint_fractions"], numeric["start_hills"]
        ):
            initial = np.array(
                [bottom, math.log(amplitude), x[0] + fraction * (x[-1] - x[0]), math.log(hill)]
            )
            result = cast(Any, least_squares)(
                lambda p: response_model(p, x) - y,
                initial,
                jac=lambda p: response_jacobian(p, x),
                bounds=(lower, upper),
                method="trf",
                loss="linear",
                x_scale="jac",
                max_nfev=numeric["max_nfev"],
                ftol=numeric["solver_tolerance"],
                xtol=numeric["solver_tolerance"],
                gtol=numeric["solver_tolerance"],
            )
            if not result.success or not all(
                np.isfinite(v).all() for v in (result.x, result.fun, result.jac, result.cost)
            ):
                return refused(
                    ["optimizer_failure"], {**diagnostics, "completed_starts": len(fits)}
                )
            fits.append(result)
    except (
        ValueError,
        RuntimeError,
        FloatingPointError,
        OverflowError,
        np.linalg.LinAlgError,
    ) as exc:
        return refused(
            ["optimizer_failure"],
            {**diagnostics, "completed_starts": len(fits), "exception_type": type(exc).__name__},
        )
    best = min(fits, key=lambda fit: float(np.dot(fit.fun, fit.fun)))
    sse = float(np.dot(best.fun, best.fun))
    rms = math.sqrt(sse / len(c))
    diagnostics.update(
        {
            "completed_starts": len(fits),
            "residual_rms_pp": rms,
            "sse_pp_squared": sse,
            "residuals_pp": best.fun.tolist(),
        }
    )
    near_bound = np.isclose(best.x, lower, rtol=1e-7, atol=1e-8) | np.isclose(
        best.x, upper, rtol=1e-7, atol=1e-8
    )
    if np.any(best.active_mask) or near_bound.any():
        return refused(["numeric_bound_active"], diagnostics)
    norms = np.linalg.norm(best.jac, axis=0)
    if not np.isfinite(norms).all() or (norms == 0).any():
        return refused(["nonidentifiable"], diagnostics)
    scaled = best.jac / norms
    try:
        rank = int(np.linalg.matrix_rank(scaled))
        condition = float(np.linalg.cond(scaled))
    except np.linalg.LinAlgError:
        return refused(["nonidentifiable"], diagnostics)
    diagnostics.update(
        {
            "column_scaled_jacobian_rank": rank,
            "column_scaled_jacobian_condition": condition if math.isfinite(condition) else None,
        }
    )
    if rank < 4:
        return refused(["nonidentifiable"], diagnostics)
    if not math.isfinite(condition) or condition > numeric["max_column_scaled_jacobian_condition"]:
        return refused(["ill_conditioned"], diagnostics)
    tolerance = (
        numeric["near_optimal_sse_absolute_tolerance"]
        + numeric["near_optimal_sse_relative_tolerance"] * sse
    )
    near = [float(f.x[2]) for f in fits if float(np.dot(f.fun, f.fun)) - sse <= tolerance]
    disagreement = max(near) - min(near)
    diagnostics["near_optimal_midpoint_range_log10"] = disagreement
    if disagreement > numeric["max_near_optimal_midpoint_disagreement_log10"]:
        return refused(["nonunique_midpoint"], diagnostics)
    if rms > policy.max_fit_rms_pp:
        return refused(["model_misfit"], diagnostics)
    bottom, log_amplitude, midpoint, log_hill = (float(v) for v in best.x)
    top, hill = bottom + math.exp(log_amplitude), math.exp(log_hill)
    midpoint_response = (bottom + top) / 2
    side = numeric["min_each_side_of_midpoint"]
    if (
        not x[0] < midpoint < x[-1]
        or min(
            np.count_nonzero(x < midpoint),
            np.count_nonzero(x > midpoint),
            np.count_nonzero(y < midpoint_response),
            np.count_nonzero(y > midpoint_response),
        )
        < side
    ):
        return refused(["relative_midpoint_unbracketed"], diagnostics)
    if (np.abs(y[:points] - top) > policy.plateau_tolerance_pp).any() or (
        np.abs(y[-points:] - bottom) > policy.plateau_tolerance_pp
    ).any():
        return refused(["plateaus_unsupported"], diagnostics)
    absolute = None
    if not bottom < 50 < top:
        absolute_status = "no_finite_crossing_in_fitted_asymptotes"
    elif not float(y.min()) < 50 < float(y.max()):
        absolute_status = "observations_do_not_bracket_50"
    else:
        crossing = midpoint + math.log10((top - 50) / (50 - bottom)) / hill
        if not x[0] < crossing < x[-1]:
            absolute_status = "crossing_outside_tested_range"
        else:
            absolute, absolute_status = 10**crossing, "estimated_condition_specific_apparent_IC50"
    return {
        "status": "estimated",
        "reason": [],
        "parameters": {
            "bottom_percent": bottom,
            "top_percent": top,
            "ln_amplitude": log_amplitude,
            "log10_midpoint_M": midpoint,
            "ln_hill": log_hill,
            "hill": hill,
        },
        "relative_midpoint_M": 10**midpoint,
        "absolute_50_vehicle_M": absolute,
        "absolute_50_status": absolute_status,
        "uncertainty": "not_estimated",
        "diagnostics": diagnostics,
    }


def analyze(package: AssayPackage) -> dict[str, Any]:
    """Analyze a package returned by load_package; retain all rows and all refused curves."""
    manifest = package.manifest
    groups: dict[CurveKey, list[Observation]] = defaultdict(list)
    for observation in package.observations:
        groups[observation.curve_key].append(observation)
    curves: list[dict[str, Any]] = []
    for (cid, experiment, run, pid), rows in sorted(groups.items()):
        aggregates = aggregate_observations(rows)
        reasons = qualification_reasons(manifest, pid)
        states = {
            state: sum(r.raw["observation_status"] == state for r in rows)
            for state in package.contract["statuses"]
        }
        if any(r.raw["observation_status"] != "observed" for r in rows):
            reasons.append("incomplete_or_censored_curve")
        policy_dict = manifest["qualification"]["policy"]
        if policy_dict is not None and any(
            a["n_technical_observed"] < policy_dict["min_technical_replicates"] for a in aggregates
        ):
            reasons.append("insufficient_technical_replicates")
        fit = (
            refused(reasons)
            if reasons
            else fit_response(
                [a["concentration_M"] for a in aggregates],
                [a["mean_percent_activity"] for a in aggregates],
                PilotPolicy(**policy_dict),
                package.contract["numeric_policy"],
            )
        )
        curves.append(
            {
                "blinded_compound_id": cid,
                "compound_role": manifest["compounds"][cid],
                "plate_id": pid,
                "run_id": run,
                "independent_experiment_id": experiment,
                "n_submitted_independent_experiments": 1,
                "observation_states": states,
                "aggregates": aggregates,
                "fit": fit,
            }
        )
    return {
        "schema_version": VERSION,
        "origin": manifest["origin"],
        "phase": manifest["phase"],
        "endpoint": ENDPOINT,
        "warning": SOFTWARE_WARNING
        if manifest["origin"] == "simulated"
        else (
            "Conditional descriptive point estimates only; submitted qualification "
            "and independence "
            "are human attestations, not verified by this software. No biological precision, "
            "efficacy, fresh external validation or completed scientific "
            "qualification is established."
        ),
        "input_manifest": manifest,
        "observations": [r.raw for r in package.observations],
        "curves": curves,
        "n_submitted_independent_experiments": len({key[1] for key in groups}),
        "provenance": {**package.provenance, "generated_at_utc": datetime.now(UTC).isoformat()},
        "analysis_contract": package.contract,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--observations", required=True, type=Path)
    parser.add_argument(
        "--output", required=True, type=Path, help="New JSON file; parent must exist"
    )
    args = parser.parse_args(argv)
    try:
        if args.output.exists() or args.output.is_symlink() or not args.output.parent.is_dir():
            raise InputError("Output must be new, with an existing parent directory")
        package = load_package(args.manifest, args.observations)
        write_json(args.output, analyze(package))
    except (OSError, ValueError, csv.Error) as exc:
        print(f"Assay input/publication error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
