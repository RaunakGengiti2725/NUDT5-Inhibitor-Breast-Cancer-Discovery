"""Retrospective paired IC50 descriptions with frozen scores, never a fitted model."""

from __future__ import annotations

import argparse
import copy
import csv
import gzip
import hashlib
import io
import json
import math
import os
import re
import shutil
import sys
import tempfile
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from pipeline import ROOT, deduplicate, input_manifest, read_compounds
from rdkit import rdBase
from transfer import identity, identity_matches, substitute_references

METHODS = ("RF", "GBT", "SVM_RBF", "Nearest_active", "Property_LR", "Equal_mean")
SCENARIOS = ("historical_original_graphs", "stored_authenticated_reference_sensitivity")
TARGETS = ("NUDT5", "NUDT14")
RATIO_DEFINITION = "IC50_NUDT14 / IC50_NUDT5"
WARNING = (
    "Previously inspected single-publication retrospective evidence; not fresh external or "
    "prospective validation. No models fitted, new experiments, selectivity classifier, "
    "significance tests, ratio confidence intervals or biological qualification."
)
LIMITATIONS = [
    WARNING,
    "R is reported mean catalytic IC50(NUDT14)/reported mean catalytic IC50(NUDT5), "
    "dimensionless; R>1 favors lower reported NUDT5 IC50 under these assay conditions, "
    "not an affinity constant, clinical selectivity or a validated biological class.",
    "Source mean ± SD remains visible at each target. Raw paired replicates and covariance "
    "are unavailable; ratio uncertainty is not estimated. Table 1 reports two biological "
    "replicates; Methods triplicate sets and Figure 1 technical triplicates are not extra "
    "independent biological replicates.",
    "NUDT5 and NUDT14 reaction times differ (20 versus 60 minutes). The shared TH5427 "
    "normalization wording has no clear NUDT14-specific exception. Comparability remains "
    "unresolved; this is not proof of assay failure.",
    "Untested is not inactive. >50 uM is a strict bound, never an exact mean of 50. "
    "Double censoring permits any positive R and identifies no finite bound or direction.",
    "Exact/neutral-parent/parent-tautomer training matches exclude only model diagnostics. "
    "All source rows, known controls and source roles remain visible. Related chemistry "
    "from one already-inspected publication is not an independent validation cohort.",
    "All six frozen methods and both recorded scenarios are retained separately. Scores "
    "are uncalibrated label diagnostics, not IC50 predictions or selectivity probabilities. "
    "There is no outcome-chosen high-score threshold and no pooling of scenarios.",
    "Source disagreements remain unresolved: Figure 1 versus CSV SD precision for 4/5, "
    "two trailing spaces, control/replication ambiguity, all-compounds-tested context and "
    "the reciprocal ratio orientation in the earlier structural report.",
]
LONG_FIELDS = (
    "source_compound",
    "source_smiles",
    "canonical_isomeric_smiles",
    "target",
    "endpoint",
    "endpoint_family",
    "unit",
    "status",
    "comparator",
    "reported_mean",
    "reported_sd",
    "bound",
    "bound_strict",
    "author_csv_raw",
    "ledger_text",
    "table1_text",
    "conditions_id",
    "conditions_application",
    "uncertainty_definition",
    "biological_n_for_reported_sd",
    "raw_replicates_available",
    "paired_target_replicates_available",
    "source_ids",
    "source_locations",
    "discrepancy_ids",
)
SCORE_FIELDS = (
    "source_compound",
    "canonical_isomeric_smiles",
    "scenario",
    "score_source_json_pointer",
    "join_verified",
    "paired_diagnostic_eligible",
    "exclusion_reasons",
    "ratio_status",
    "max_training_tanimoto",
    "nearest_training_id",
    *sorted(METHODS),
    *(f"{method}_rank_within_six" for method in METHODS),
)
LEDGER_FIELDS = (
    "source_compound",
    "source_smiles",
    "canonical_smiles",
    "nudt5_ic50_uM_as_reported",
    "nudt14_ic50_uM_as_reported",
    "repository_identity_matches",
    "source_doi",
    "assay",
    "uncertainty",
    "inactive_definition",
)
NUMBER = r"(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def finite_number(value: Any, *, positive: bool = False) -> float:
    require(type(value) in (int, float), f"Numeric value required, not {value!r}")
    try:
        number = float(value)
    except (OverflowError, ValueError) as exc:
        raise ValueError("Unrepresentable number") from exc
    require(math.isfinite(number), "Nonfinite value")
    require(number > 0 if positive else number >= 0, "Invalid numeric range")
    return number


def read_json(path: Path) -> Any:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            require(key not in result, f"Duplicate JSON key: {key}")
            result[key] = value
        return result

    def reject(value: str) -> Any:
        raise ValueError(f"Nonfinite JSON constant: {value}")

    value = json.loads(
        path.read_text(encoding="utf-8"), object_pairs_hook=pairs, parse_constant=reject
    )
    # This also catches exponent overflow (1e999), which parse_constant does not see.
    json.dumps(value, allow_nan=False)
    return value


def strict_csv(path: Path, fields: Sequence[str]) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, strict=True)
        require(reader.fieldnames == list(fields), f"Exact unique CSV headers required: {path}")
        result = list(reader)
    require(
        all(None not in row and all(v is not None for v in row.values()) for row in result),
        f"Ragged CSV rows: {path}",
    )
    return result


def validate_schema(value: Any, schema: dict[str, Any], path: str = "$") -> None:
    """Validate only the closed keyword subset used by the supplied release schema."""
    supported = {
        "$schema",
        "title",
        "description",
        "type",
        "const",
        "enum",
        "required",
        "properties",
        "additionalProperties",
        "items",
        "minItems",
        "maxItems",
        "minimum",
        "exclusiveMinimum",
        "oneOf",
    }
    require(not set(schema) - supported, f"Unsupported schema keyword at {path}")
    types = schema.get("type", [])
    types = [types] if isinstance(types, str) else types
    matches = {
        "null": value is None,
        "boolean": type(value) is bool,
        "string": isinstance(value, str),
        "number": type(value) in (int, float),
        "integer": type(value) is int,
        "object": isinstance(value, dict),
        "array": isinstance(value, list),
    }
    require(all(t in matches for t in types), f"Unsupported schema type at {path}")
    require(not types or any(matches[t] for t in types), f"Invalid type at {path}")
    if type(value) in (int, float):
        require(math.isfinite(value), f"Nonfinite number at {path}")
        if "minimum" in schema:
            require(value >= schema["minimum"], f"Below minimum at {path}")
        if "exclusiveMinimum" in schema:
            require(value > schema["exclusiveMinimum"], f"Below exclusive minimum at {path}")
    if "const" in schema:
        require(
            json.dumps(value) == json.dumps(schema["const"])
            or (
                type(value) in (int, float)
                and type(schema["const"]) in (int, float)
                and value == schema["const"]
            ),
            f"Invalid constant at {path}",
        )
    if "enum" in schema:
        require(
            any(type(value) is type(v) and value == v for v in schema["enum"]),
            f"Invalid enum at {path}",
        )
    if isinstance(value, dict):
        require(set(schema.get("required", [])).issubset(value), f"Missing keys at {path}")
        properties = schema.get("properties", {})
        if schema.get("additionalProperties") is False:
            require(set(value).issubset(properties), f"Unexpected keys at {path}")
        for key in value.keys() & properties.keys():
            validate_schema(value[key], properties[key], f"{path}/{key}")
    if isinstance(value, list):
        require(len(value) >= schema.get("minItems", 0), f"Too few items at {path}")
        require(len(value) <= schema.get("maxItems", len(value)), f"Too many items at {path}")
        if "items" in schema:
            for i, item in enumerate(value):
                validate_schema(item, schema["items"], f"{path}/{i}")
    if "oneOf" in schema:
        successes = 0
        for alternative in schema["oneOf"]:
            try:
                validate_schema(value, alternative, path)
                successes += 1
            except ValueError:
                pass
        require(successes == 1, f"Expected exactly one schema alternative at {path}")


def validate_endpoint(endpoint: dict[str, Any], target: str) -> None:
    require(endpoint["target"] == target, "Reversed or wrong endpoint target")
    require(endpoint["endpoint"] == "IC50", "Only IC50 endpoints accepted")
    require(
        endpoint["endpoint_family"] == "purified_enzyme_catalytic_inhibition",
        "Incompatible endpoint family",
    )
    require(endpoint["unit"] == "uM", "Incompatible endpoint units")
    state = endpoint["status"]
    if state == "numeric":
        finite_number(endpoint["reported_mean"], positive=True)
        finite_number(endpoint["reported_sd"])
        require(
            endpoint["comparator"] == "="
            and endpoint["bound"] is None
            and endpoint["bound_strict"] is None,
            "Conflicting numeric censoring fields",
        )
    elif state == "right_censored":
        require(
            type(endpoint["bound"]) in (int, float)
            and endpoint["bound"] == 50
            and endpoint["comparator"] == ">"
            and endpoint["bound_strict"] is True,
            "Source requires strict >50 uM",
        )
        require(
            endpoint["reported_mean"] is None and endpoint["reported_sd"] is None,
            "Censored endpoint cannot have fabricated mean/SD",
        )
    elif state == "untested":
        require(
            all(
                endpoint[k] is None
                for k in (
                    "reported_mean",
                    "reported_sd",
                    "comparator",
                    "bound",
                    "bound_strict",
                    "biological_n_for_reported_sd",
                )
            ),
            "Untested endpoint has fabricated values",
        )
    else:
        raise ValueError(f"Unknown endpoint status: {state}")


def ratio(nudt5: dict[str, Any], nudt14: dict[str, Any]) -> dict[str, Any]:
    for endpoint, target in zip((nudt5, nudt14), TARGETS, strict=True):
        validate_endpoint(endpoint, target)
    states = (nudt5["status"], nudt14["status"])
    result: dict[str, Any] = {
        "definition": RATIO_DEFINITION,
        "unit": "dimensionless",
        "status": "point",
        "point": None,
        "bound": None,
        "log10_point": None,
        "log10_bound": None,
        "comparator": None,
        "bound_strict": None,
        "non_estimability_reason": None,
    }
    if "untested" in states:
        result.update(
            status="missing_endpoint", non_estimability_reason="At least one target untested"
        )
    elif states == ("right_censored", "right_censored"):
        result.update(
            status="double_censored",
            non_estimability_reason=(
                "Both IC50 >50 uM permit any positive ratio; "
                "no informative finite bound or direction"
            ),
        )
    else:
        numerator = nudt14["reported_mean"] if states[1] == "numeric" else nudt14["bound"]
        denominator = nudt5["reported_mean"] if states[0] == "numeric" else nudt5["bound"]
        value = finite_number(numerator / denominator, positive=True)
        if states == ("numeric", "numeric"):
            result.update(point=value, log10_point=math.log10(value), comparator="=")
        else:
            upper = states[0] == "right_censored"
            result.update(
                status="upper_bound" if upper else "lower_bound",
                bound=value,
                log10_bound=math.log10(value),
                comparator="<" if upper else ">",
                bound_strict=True,
            )
    return result


def assert_ratio(stored: dict[str, Any], calculated: dict[str, Any]) -> None:
    require(set(stored) == set(calculated), "Ratio fields differ")
    for key, expected in calculated.items():
        observed = stored[key]
        if type(expected) is float:
            require(
                type(observed) in (int, float)
                and math.isfinite(observed)
                and math.isclose(observed, expected, rel_tol=1e-12, abs_tol=1e-12),
                f"Ratio arithmetic mismatch: {key}",
            )
        elif key == "non_estimability_reason" and expected is not None:
            require(isinstance(observed, str) and bool(observed.strip()), "Missing ratio reason")
        else:
            require(
                type(observed) is type(expected) and observed == expected,
                f"Ratio orientation/state mismatch: {key}",
            )


def parse_reported(raw: str, *, table: bool = False) -> tuple[str, float | None, float | None]:
    text = raw.strip()
    if text == "not tested":
        return "untested", None, None
    if text == "not active" or (table and text == "NA"):
        return "right_censored", None, None
    match = re.fullmatch(f"({NUMBER})\\s*±\\s*({NUMBER})", text)
    require(match is not None, f"Malformed reported mean ± SD: {raw!r}")
    assert match is not None
    return "numeric", finite_number(float(match[1]), positive=True), finite_number(float(match[2]))


def unique_rows(rows: Sequence[dict[str, Any]], key: str) -> dict[str, dict[str, Any]]:
    result = {}
    for row in rows:
        value = row[key]
        require(
            isinstance(value, str) and bool(value.strip()) and value == value.strip(),
            f"Invalid {key}",
        )
        require(value not in result, f"Duplicate {key}: {value}")
        result[value] = row
    return result


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def safe_child(root: Path, name: str) -> Path:
    path = (root / name).resolve()
    require(path.is_relative_to(root.resolve()), "Provenance path escapes its root")
    return path


def verify_inputs(
    source: Path, predictions: Path, repository: Path
) -> tuple[dict[str, Any], list[Path]]:
    base = source.parent
    inventory = read_json(base / "artifact_manifest.json")
    provenance = read_json(base / "provenance.json")
    paths = [base / "artifact_manifest.json"]
    entries = unique_rows(inventory["files"], "path")
    require(
        "paired_evidence.json" in entries and "provenance.json" in entries,
        "Incomplete curation inventory",
    )
    for entry in entries.values():
        path = (
            source if entry["path"] == "paired_evidence.json" else safe_child(base, entry["path"])
        )
        require(
            sha256(path) == entry["sha256"] and path.stat().st_size == entry["size_bytes"],
            f"Curation artifact hash/size mismatch: {path}",
        )
        paths.append(path)
    originals = unique_rows(provenance["inputs"], "path")
    require("research/results/transfer.json" in originals, "Prediction hash not recorded")
    for entry in originals.values():
        path = (
            predictions
            if entry["path"] == "research/results/transfer.json"
            else safe_child(repository, entry["path"])
        )
        require(
            sha256(path) == entry["sha256"] and path.stat().st_size == entry["size_bytes"],
            f"Stale/incompatible original input: {path}",
        )
        paths.append(path)
    for entry in provenance["sources"]:
        path = safe_child(base, entry["stored_path"])
        raw = gzip.decompress(path.read_bytes())
        require(
            sha256(path) == entry["sha256_stored"]
            and hashlib.sha256(raw).hexdigest() == entry["sha256_uncompressed"]
            and len(raw) == entry["size_bytes_uncompressed"],
            "Primary snapshot mismatch",
        )
    return provenance, sorted(set(paths))


def long_projection(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "source_compound": row["source_compound"],
            "source_smiles": row["source_smiles"],
            "canonical_isomeric_smiles": row["identity"]["canonical_smiles"],
            **row["endpoints"][target],
        }
        for row in rows
        for target in TARGETS
    ]


def csv_cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return ";".join(str(item) for item in value)
    return str(value)


def compare_projection(
    path: Path, fields: Sequence[str], expected: Sequence[dict[str, Any]]
) -> None:
    actual = strict_csv(path, fields)
    require(len(actual) == len(expected), f"Projection row count differs: {path}")
    for left, right in zip(actual, expected, strict=True):
        for key in fields:
            value = right[key]
            if type(value) is float:
                require(re.fullmatch(NUMBER, left[key]) is not None, f"Malformed CSV number: {key}")
                require(
                    finite_number(float(left[key])) == value, f"Projection number differs: {key}"
                )
            else:
                require(left[key] == csv_cell(value), f"Projection field differs: {key}")


def eligibility(row: dict[str, Any], matches: dict[str, list[str]]) -> list[str]:
    reasons = []
    if any(matches.values()):
        reasons.append("training_exact_parent_or_tautomer_overlap")
    if any(row["endpoints"][target]["status"] == "untested" for target in TARGETS):
        reasons.append("missing_target_endpoint")
    return reasons


def validate_source_rows(
    rows: list[dict[str, Any]],
    ledger: list[dict[str, str]],
    training: Sequence[dict[str, Any]],
    sensitivity: Sequence[dict[str, Any]],
    candidates: Sequence[dict[str, Any]],
    conditions: list[dict[str, Any]],
) -> None:
    source_by_id = unique_rows(rows, "source_compound")
    ledger_by_id = unique_rows(ledger, "source_compound")
    require(list(source_by_id) == list(ledger_by_id), "Source IDs/order differ from ledger")
    seen: set[str] = set()
    condition_ids = unique_rows(conditions, "conditions_id")
    for row in rows:
        name = row["source_compound"]
        original = ledger_by_id[name]
        require(row["source_smiles"] == original["source_smiles"], "Source SMILES mismatch")
        info = identity(row["source_smiles"])
        require(info == row["identity"], f"Recomputed identity conflict: {name}")
        require(info["canonical_smiles"] == original["canonical_smiles"], "Ledger graph conflict")
        require(info["canonical_smiles"] not in seen, "Duplicate source graph")
        seen.add(info["canonical_smiles"])
        for key, panel in (
            ("historical_training_matches", training),
            ("reference_sensitivity_training_matches", sensitivity),
            ("candidate_identity_matches", candidates),
        ):
            require(row[key] == identity_matches(info, panel), f"Overlap conflict: {name}/{key}")
        require(set(row["endpoints"]) == set(TARGETS), "Exactly two target endpoints required")
        for target in TARGETS:
            endpoint = row["endpoints"][target]
            validate_endpoint(endpoint, target)
            condition = condition_ids[endpoint["conditions_id"]]
            require(
                all(
                    endpoint[k] == condition[k]
                    for k in ("target", "unit", "endpoint", "endpoint_family")
                ),
                "Assay mismatch",
            )
            require(
                endpoint["ledger_text"] == original[f"{target.lower()}_ic50_uM_as_reported"],
                "Endpoint ledger text mismatch",
            )
            observed = (endpoint["status"], endpoint["reported_mean"], endpoint["reported_sd"])
            for text in (endpoint["author_csv_raw"], endpoint["ledger_text"]):
                require(parse_reported(text) == observed, "Raw endpoint/numeric conflict")
            if endpoint["table1_text"] is not None:
                require(
                    parse_reported(endpoint["table1_text"], table=True) == observed,
                    "Table 1 endpoint/numeric conflict",
                )
            require(
                endpoint["raw_replicates_available"] is False
                and endpoint["paired_target_replicates_available"] is False,
                "Raw paired replicates unavailable in this release",
            )
        calculated = ratio(row["endpoints"]["NUDT5"], row["endpoints"]["NUDT14"])
        assert_ratio(row["ratio"], calculated)
        require(
            row["ratio_informative"]
            == (calculated["status"] in ("point", "upper_bound", "lower_bound")),
            "Informativeness conflict",
        )
        paired = calculated["status"] != "missing_endpoint"
        require(row["source_pair_has_both_endpoints"] is paired, "Pair status conflict")
        reasons = eligibility(row, row["historical_training_matches"])
        require(
            row["model_exclusion_reasons"] == reasons
            and row["model_diagnostic_eligible"] is (not reasons),
            "Eligibility conflict",
        )


def join_scores(
    rows: list[dict[str, Any]], predictions: dict[str, Any], ledger: list[dict[str, str]]
) -> list[dict[str, Any]]:
    sources = unique_rows(ledger, "source_compound")
    output = []
    for scenario in SCENARIOS:
        historical = scenario == SCENARIOS[0]
        pointer = (
            "/measured_source_challenge/rows"
            if historical
            else ("/reference_sensitivity/measured_source_challenge/rows")
        )
        panel = (
            predictions["measured_source_challenge"]["rows"]
            if historical
            else (predictions["reference_sensitivity"]["measured_source_challenge"]["rows"])
        )
        predicted = unique_rows(panel, "id")
        require(set(predicted) == set(sources), "Missing/extra promised prediction join")
        unique_rows(panel, "canonical_smiles")
        joined = []
        for row in rows:
            name = row["source_compound"]
            prediction = predicted[name]
            require(prediction["source"] == sources[name], f"Score source/ID conflict: {name}")
            require(
                all(prediction[k] == v for k, v in row["identity"].items()),
                f"Score graph/identity conflict: {name}",
            )
            matches = row[
                "historical_training_matches"
                if historical
                else "reference_sensitivity_training_matches"
            ]
            require(prediction["training_matches"] == matches, "Score training-overlap conflict")
            scores = prediction["scores"]
            require(set(scores) == set(METHODS), "Frozen method names differ")
            for value in scores.values():
                require(finite_number(value) <= 1, "Invalid uncalibrated score")
            expected_mean = sum(scores[k] for k in METHODS[:4]) / 4
            require(
                math.isclose(scores["Equal_mean"], expected_mean, rel_tol=1e-14),
                "Equal_mean is not the original four-method mean",
            )
            reasons = eligibility(row, matches)
            joined.append(
                {
                    "source_compound": name,
                    "canonical_isomeric_smiles": row["identity"]["canonical_smiles"],
                    "scenario": scenario,
                    "score_source_json_pointer": f"{pointer}/{panel.index(prediction)}",
                    "join_verified": True,
                    "score_availability": "available_frozen_identity_join",
                    "unavailable_score_reason": None,
                    "paired_diagnostic_eligible": not reasons,
                    "exclusion_reasons": reasons,
                    "ratio_status": row["ratio"]["status"],
                    "max_training_tanimoto": prediction["max_training_tanimoto"],
                    "nearest_training_id": prediction["nearest_training_id"],
                    **scores,
                }
            )
        eligible = [row for row in joined if row["paired_diagnostic_eligible"]]
        for row in joined:
            row["rank_denominator"] = len(eligible)
            row["inclusion_reason"] = (
                "paired_endpoints_without_training_identity_overlap"
                if (row["paired_diagnostic_eligible"])
                else None
            )
            for method in METHODS:
                row[f"{method}_rank_within_six"] = (
                    1 + sum(other[method] > row[method] for other in eligible)
                    if row["paired_diagnostic_eligible"]
                    else None
                )
            # New outputs use cohort-neutral rank keys; legacy keys verify the supplied projection.
            row["ranks"] = {method: row[f"{method}_rank_within_six"] for method in METHODS}
        output.extend(joined)
    return output


def summarize(rows: Sequence[dict[str, Any]], scores: Sequence[dict[str, Any]]) -> dict[str, Any]:
    scenarios = {}
    for scenario in SCENARIOS:
        eligible = [
            r for r in scores if r["scenario"] == scenario and r["paired_diagnostic_eligible"]
        ]
        scenarios[scenario] = {
            "eligible_ids": [r["source_compound"] for r in eligible],
            "n": len(eligible),
            "ratio_status_counts": dict(Counter(r["ratio_status"] for r in eligible)),
            "Equal_mean_descending_ids": [
                r["source_compound"] for r in sorted(eligible, key=lambda r: -r["Equal_mean"])
            ],
        }
    return {
        "source_graphs": len(rows),
        "endpoint_cells": len(rows) * 2,
        "source_paired_n": sum(r["source_pair_has_both_endpoints"] for r in rows),
        "source_ratio_status_counts": dict(Counter(r["ratio"]["status"] for r in rows)),
        "endpoint_status_counts": {
            t: dict(Counter(r["endpoints"][t]["status"] for r in rows)) for t in TARGETS
        },
        "scenarios": scenarios,
    }


def analyze(source: Path, predictions: Path, repository: Path) -> tuple[dict[str, Any], list[Path]]:
    provenance, paths = verify_inputs(source, predictions, repository)
    hashes = {path: sha256(path) for path in paths}
    evidence = read_json(source)
    validate_schema(evidence, read_json(source.parent / "paired_evidence.schema.json"))
    require(evidence["rdkit_version"] == rdBase.rdkitVersion, "Incompatible RDKit identity version")
    stored = read_json(predictions)
    historical = read_json(repository / "research/results/transfer-manifest.json")
    for path in (
        repository / "compounds.csv",
        repository / "final_hits.csv",
        repository / "research/source_assays.csv",
        repository / "research/reference_structures.csv",
        repository / "requirements.lock",
    ):
        suffix = path.relative_to(repository).as_posix()
        matches = [v for k, v in historical["files"].items() if k.endswith("/" + suffix)]
        require(matches == [sha256(path)], f"Historical prediction provenance conflict: {suffix}")
    records, invalid = read_compounds(repository / "compounds.csv")
    records, _ = deduplicate(records)
    fixed, changes = substitute_references(
        records, repository / "research/reference_structures.csv"
    )
    training = [
        {"id": r.identifier, **identity(r.smiles), "original_source": r.source_row} for r in records
    ]
    require(
        training == stored["training_identity"] and len(training) == stored["training_n"],
        "Stale prediction training identities",
    )
    require(changes == stored["reference_sensitivity"]["changes"], "Stale reference sensitivity")
    sensitivity = [{"id": r.identifier, **identity(r.smiles)} for r in fixed]
    candidates, candidate_invalid = read_compounds(repository / "final_hits.csv", labelled=False)
    require(not candidate_invalid, "Invalid candidate graph")
    candidate_info = [{"id": r.identifier, **identity(r.smiles)} for r in candidates]
    ledger = strict_csv(repository / "research/source_assays.csv", LEDGER_FIELDS)
    rows = evidence["rows"]
    conditions = read_json(source.parent / "assay_conditions.json")["conditions"]
    validate_source_rows(rows, ledger, training, sensitivity, candidate_info, conditions)
    compare_projection(source.parent / "target_evidence.csv", LONG_FIELDS, long_projection(rows))
    scores = join_scores(rows, stored, ledger)
    lookup = {(r["source_compound"], r["scenario"]): r for r in scores}
    legacy_order = [lookup[(r["source_compound"], s)] for r in rows for s in SCENARIOS]
    compare_projection(source.parent / "frozen_scores.csv", SCORE_FIELDS, legacy_order)
    result = {
        "schema_version": "1.0.0",
        "warning": WARNING,
        "ratio_definition": RATIO_DEFINITION,
        "ratio_unit": "dimensionless",
        "methods": list(METHODS),
        "rows": copy.deepcopy(rows),
        "score_rows": scores,
        "summary": summarize(rows, scores),
        "limitations": LIMITATIONS,
        "assay_conditions": conditions,
        "source_provenance": provenance,
        "source_disagreements": read_json(source.parent / "source_disagreements.json"),
        "historical_prediction_manifest": historical,
        "quarantined_training_records": invalid,
        "uncertainty": (
            "Source SD retained; ratio uncertainty not estimated without raw paired replicates"
        ),
    }
    for row in result["rows"]:
        row["ratio"] = ratio(row["endpoints"]["NUDT5"], row["endpoints"]["NUDT14"])
    require(
        all(sha256(path) == digest for path, digest in hashes.items()), "Inputs changed during read"
    )
    validate_result(result)
    return result, paths


def validate_result(result: dict[str, Any]) -> None:
    require(
        result["schema_version"] == "1.0.0"
        and result["ratio_definition"] == RATIO_DEFINITION
        and result["ratio_unit"] == "dimensionless",
        "Incompatible result contract",
    )
    require(result["methods"] == list(METHODS), "Result methods differ")
    json.dumps(result, allow_nan=False)
    rows = unique_rows(result["rows"], "source_compound")
    unique_rows([r["identity"] for r in rows.values()], "canonical_smiles")
    for row in rows.values():
        for target in TARGETS:
            cell = row["endpoints"][target]
            require(
                parse_reported(cell["author_csv_raw"])
                == (cell["status"], cell["reported_mean"], cell["reported_sd"]),
                "Result raw/numeric endpoint conflict",
            )
        require(
            row["source_pair_has_both_endpoints"] is (row["ratio"]["status"] != "missing_endpoint"),
            "Result paired status differs",
        )
        reasons = eligibility(row, row["historical_training_matches"])
        require(
            row["model_exclusion_reasons"] == reasons
            and row["model_diagnostic_eligible"] is (not reasons),
            "Result source eligibility differs",
        )
        assert_ratio(row["ratio"], ratio(row["endpoints"]["NUDT5"], row["endpoints"]["NUDT14"]))
    for scenario in SCENARIOS:
        score_rows = [r for r in result["score_rows"] if r["scenario"] == scenario]
        require(
            set(unique_rows(score_rows, "source_compound")) == set(rows), "Result score IDs differ"
        )
        eligible = [r for r in score_rows if r["paired_diagnostic_eligible"]]
        for score in score_rows:
            source = rows[score["source_compound"]]
            matches = source[
                "historical_training_matches"
                if scenario == SCENARIOS[0]
                else "reference_sensitivity_training_matches"
            ]
            reasons = eligibility(source, matches)
            require(
                score["paired_diagnostic_eligible"] is (not reasons)
                and score["exclusion_reasons"] == reasons,
                "Result exclusions differ",
            )
            require(
                score["canonical_isomeric_smiles"] == source["identity"]["canonical_smiles"]
                and score["ratio_status"] == source["ratio"]["status"],
                "Result score join conflict",
            )
            require(score["rank_denominator"] == len(eligible), "Rank denominator differs")
            require(
                score["join_verified"] is True
                and score["score_availability"] == "available_frozen_identity_join"
                and score["unavailable_score_reason"] is None,
                "Result score join unavailable",
            )
            require(
                math.isclose(
                    score["Equal_mean"], sum(score[m] for m in METHODS[:4]) / 4, rel_tol=1e-14
                ),
                "Result Equal_mean differs",
            )
            for method in METHODS:
                require(finite_number(score[method]) <= 1, "Result score outside range")
                rank = (
                    1 + sum(other[method] > score[method] for other in eligible)
                    if not reasons
                    else None
                )
                require(score["ranks"][method] == rank, "Result rank differs")
    require(
        len(result["score_rows"]) == len(rows) * len(SCENARIOS), "Unknown/duplicate result scenario"
    )
    require(
        result["summary"] == summarize(result["rows"], result["score_rows"]), "Result counts differ"
    )


def ratio_text(value: dict[str, Any]) -> str:
    if value["status"] == "point":
        return f"{value['point']:.6g}"
    if value["bound"] is not None:
        return f"{value['comparator']}{value['bound']:.6g} (strict)"
    return "not estimable: both >50" if value["status"] == "double_censored" else "missing endpoint"


def endpoint_text(endpoint: dict[str, Any]) -> str:
    if endpoint["status"] == "numeric":
        return str(endpoint["author_csv_raw"].strip())
    return ">50 (strict)" if endpoint["status"] == "right_censored" else "untested"


def markdown(result: dict[str, Any]) -> str:
    validate_result(result)
    summary = result["summary"]
    lines = [
        "# Retrospective paired NUDT5/NUDT14 evidence",
        "",
        result["warning"],
        "",
        f"All {summary['source_graphs']} source graphs and "
        f"{summary['endpoint_cells']} endpoint cells "
        f"are retained; {summary['source_paired_n']} source rows have both endpoints.",
        "",
        "R = reported mean IC50(NUDT14) / reported mean IC50(NUDT5), dimensionless. "
        "Positive log10 R favors lower NUDT5 reported IC50, not validated selectivity.",
        "",
        "## Complete source pharmacology",
        "",
        "IC50 in µM; ± denotes published SD, not ratio uncertainty.",
        "",
        "| Compound | NUDT5 IC50 (µM) | NUDT14 IC50 (µM) | R / bound | "
        "Historical inclusion or exclusion |",
        "|---|---|---|---|---|",
    ]
    for row in result["rows"]:
        reason = "; ".join(row["model_exclusion_reasons"]) or "paired; no training identity overlap"
        lines.append(
            f"| {row['source_compound']} | {endpoint_text(row['endpoints']['NUDT5'])} | "
            f"{endpoint_text(row['endpoints']['NUDT14'])} | {ratio_text(row['ratio'])} | {reason} |"
        )
    by_id = {r["source_compound"]: r for r in result["rows"]}
    for scenario in SCENARIOS:
        counts = summary["scenarios"][scenario]
        lines.extend(
            [
                "",
                f"## {scenario}",
                "",
                f"n={counts['n']} nonoverlap paired compounds; "
                f"{counts['ratio_status_counts'].get('point', 0)} point ratios. "
                "Uncalibrated scores (within-cohort shared-minimum rank); no ratio ranking.",
                "",
                "| Compound | R / bound | " + " | ".join(METHODS) + " |",
                "|---|---|" + "---|" * len(METHODS),
            ]
        )
        scores = {
            r["source_compound"]: r for r in result["score_rows"] if r["scenario"] == scenario
        }
        if not counts["n"]:
            lines.extend(["", "No eligible pairs; no score/ratio association can be described."])
        for name in counts["Equal_mean_descending_ids"]:
            row = scores[name]
            cells = [f"{row[m]:.6g} ({row['ranks'][m]}/{counts['n']})" for m in METHODS]
            lines.append(
                f"| {name} | {ratio_text(by_id[name]['ratio'])} | " + " | ".join(cells) + " |"
            )
        if counts["n"]:
            top = counts["Equal_mean_descending_ids"][0]
            lines.extend(
                [
                    "",
                    f"The largest displayed Equal_mean is {scores[top]['Equal_mean']:.6g} "
                    f"for compound {top}, with R {ratio_text(by_id[top]['ratio'])}. "
                    "This coexistence is descriptive, not predictive validation.",
                ]
            )
    lines.extend(
        [
            "",
            "## Evidence limits",
            "",
            *[f"- {item}" for item in result["limitations"]],
            "",
            "## Primary evidence inspected in the inherited curation",
            "",
            result["source_provenance"]["primary_citation"],
            "",
        ]
    )
    for source in result["source_provenance"]["sources"]:
        lines.append(
            f"- {source['source_id']}: {source['url']}; "
            f"sections: {', '.join(source['inspected_locations'])}; "
            f"retrieved {source['retrieved_utc']}; decompressed SHA-256 "
            f"`{source['sha256_uncompressed']}`."
        )
    return "\n".join(lines) + "\n"


def csv_bytes(rows: Sequence[dict[str, Any]], fields: Sequence[str]) -> bytes:
    handle = io.StringIO(newline="")
    writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows({k: csv_cell(row[k]) for k in fields} for row in rows)
    return handle.getvalue().encode("utf-8")


def json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def publish(output: Path, payloads: Mapping[str, bytes], completion: str) -> None:
    """Publish exclusively via hard links; the completion manifest is linked last."""
    require(completion in payloads, "Completion manifest required")
    require(
        all(Path(name).name == name and name not in ("", ".", "..") for name in payloads),
        "Flat output filenames required",
    )
    require(not output.is_symlink(), "Output symlink refused")
    created = False
    if output.exists():
        require(output.is_dir() and not any(output.iterdir()), "Output directory must be empty")
    else:
        output.mkdir(parents=True, exist_ok=False)
        created = True
    staging = Path(tempfile.mkdtemp(prefix=".selectivity-", dir=output.parent))
    published: dict[Path, tuple[int, int]] = {}
    try:
        for name, data in payloads.items():
            with (staging / name).open("xb") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
        order = [name for name in payloads if name != completion] + [completion]
        for name in order:
            path = output / name
            os.link(staging / name, path)
            stat = (staging / name).stat()
            published[path] = (stat.st_dev, stat.st_ino)
        fd = os.open(output, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    except BaseException:
        for path, inode in reversed(list(published.items())):
            if path.exists() and (path.stat().st_dev, path.stat().st_ino) == inode:
                path.unlink()
        if created and not any(output.iterdir()):
            output.rmdir()
        raise
    finally:
        shutil.rmtree(staging)


def artifacts(result: dict[str, Any]) -> dict[str, bytes]:
    validate_result(result)
    pharmacology = []
    for row in result["rows"]:
        item: dict[str, Any] = {
            "source_compound": row["source_compound"],
            "source_smiles": row["source_smiles"],
            "canonical_isomeric_smiles": row["identity"]["canonical_smiles"],
            "known_source_role": row["known_source_role"],
            "endpoint": "IC50",
            "endpoint_family": "purified_enzyme_catalytic_inhibition",
            "unit": "uM",
        }
        for target in TARGETS:
            for key in (
                "status",
                "comparator",
                "reported_mean",
                "reported_sd",
                "bound",
                "bound_strict",
                "biological_n_for_reported_sd",
                "author_csv_raw",
                "uncertainty_definition",
            ):
                item[f"{target}_{key}"] = row["endpoints"][target][key]
        item.update({f"ratio_{k}": v for k, v in row["ratio"].items()})
        item["model_diagnostic_eligible"] = row["model_diagnostic_eligible"]
        item["model_exclusion_reasons"] = row["model_exclusion_reasons"]
        pharmacology.append(item)
    # Use stable fields even for an empty SOFTWARE-ONLY document.
    fields = list(pharmacology[0]) if pharmacology else ["source_compound"]
    scores = []
    for scenario in SCENARIOS:
        scenario_rows = [r for r in result["score_rows"] if r["scenario"] == scenario]
        ordered = sorted(
            scenario_rows, key=lambda r: (not r["paired_diagnostic_eligible"], -r["Equal_mean"])
        )
        for row in ordered:
            scores.append(
                {
                    **{k: row[k] for k in SCORE_FIELDS if not k.endswith("_within_six")},
                    "rank_denominator": row["rank_denominator"],
                    **{f"{m}_rank_within_cohort": row["ranks"][m] for m in METHODS},
                    "inclusion_reason": row["inclusion_reason"],
                    "score_availability": row["score_availability"],
                    "unavailable_score_reason": row["unavailable_score_reason"],
                }
            )
    score_fields = list(scores[0]) if scores else ["source_compound", "scenario"]
    return {
        "selectivity.json": json_bytes(result),
        "selectivity.md": markdown(result).encode("utf-8"),
        "pharmacology.csv": csv_bytes(pharmacology, fields),
        "target_endpoints.csv": csv_bytes(long_projection(result["rows"]), LONG_FIELDS),
        "diagnostic_scores.csv": csv_bytes(scores, score_fields),
    }


def run_manifest(
    paths: Sequence[Path], arguments: Mapping[str, Any], payloads: Mapping[str, bytes]
) -> dict[str, Any]:
    manifest = input_manifest(paths, arguments)
    manifest.update(
        schema_version="1.0.0",
        warning=WARNING,
        command=[sys.executable, *sys.argv],
        artifacts={
            k: {"sha256": hashlib.sha256(v).hexdigest(), "size_bytes": len(v)}
            for k, v in payloads.items()
        },
        completion="This manifest is published last; absent manifest means incomplete run",
    )
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Verified paired_evidence.json")
    parser.add_argument(
        "--predictions", type=Path, required=True, help="Frozen transfer.json; never refitted"
    )
    parser.add_argument(
        "--repository", type=Path, default=ROOT, help="Root for immutable recorded input paths"
    )
    parser.add_argument("--output", type=Path, required=True, help="New or empty output directory")
    args = parser.parse_args(argv)
    try:
        require(
            not args.output.exists() or (args.output.is_dir() and not any(args.output.iterdir())),
            "Output directory must be empty",
        )
        result, inputs = analyze(
            args.source.resolve(), args.predictions.resolve(), args.repository.resolve()
        )
        payloads = artifacts(result)
        manifest = run_manifest([*inputs, Path(__file__).resolve()], vars(args), payloads)
        payloads["selectivity-manifest.json"] = json_bytes(manifest)
        publish(args.output, payloads, "selectivity-manifest.json")
    except (ValueError, OSError, csv.Error, KeyError, TypeError, OverflowError) as exc:
        print(f"Selectivity analysis refused: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
