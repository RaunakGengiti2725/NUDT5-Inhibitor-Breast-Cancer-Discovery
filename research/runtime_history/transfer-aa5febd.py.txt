"""Fixed-model, retrospective biochemical-source challenges and identity reporting."""

from __future__ import annotations

import argparse
import csv
import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import numpy as np
from controls import extended_metrics, labels_of, property_matrix, tanimoto_matrix
from pipeline import (
    ROOT,
    Compound,
    candidate_audit,
    compute_props,
    deduplicate,
    fit_scores,
    input_manifest,
    molecule,
    out_of_fold,
    read_compounds,
    smiles_to_fp,
    write_json,
)
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem.Scaffolds import MurckoScaffold
from scipy.stats import rankdata


def identity(smiles: str) -> dict[str, Any]:
    mol = molecule(smiles)
    parent = rdMolStandardize.FragmentParent(mol)
    neutral = rdMolStandardize.Uncharger().uncharge(parent)
    tautomer = rdMolStandardize.TautomerEnumerator().Canonicalize(neutral)
    return {
        "canonical_smiles": Chem.MolToSmiles(mol, isomericSmiles=True),
        "inchi": cast(Any, Chem).MolToInchi(mol),
        "inchikey": cast(Any, Chem).MolToInchiKey(mol),
        "formula": rdMolDescriptors.CalcMolFormula(mol),
        "fragment_count": len(Chem.GetMolFrags(mol)),
        "formal_charge": sum(atom.GetFormalCharge() for atom in mol.GetAtoms()),
        "unspecified_tetrahedral_centers": sum(
            label == "?"
            for _, label in cast(Any, Chem).FindMolChiralCenters(mol, includeUnassigned=True)
        ),
        "neutral_fragment_parent": Chem.MolToSmiles(neutral, isomericSmiles=True),
        "canonical_parent_tautomer": Chem.MolToSmiles(tautomer, isomericSmiles=True),
        "scaffold": cast(Any, MurckoScaffold).MurckoScaffoldSmiles(mol=mol) or "ACYCLIC",
    }


def identity_matches(
    query: dict[str, Any], training: Sequence[dict[str, Any]]
) -> dict[str, list[str]]:
    return {
        key: [row["id"] for row in training if row[key] == query[key]]
        for key in ("canonical_smiles", "neutral_fragment_parent", "canonical_parent_tautomer")
    }


def source_compound(identifier: str, smiles: str, metadata: dict[str, str]) -> Compound:
    row = identity(smiles)
    return Compound(identifier, smiles, row["canonical_smiles"], row["scaffold"], None, metadata)


def read_source_csv(path: Path, required: Sequence[str] = ()) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle, strict=True)
        headers = reader.fieldnames or []
        if not headers or len(set(headers)) != len(headers):
            raise ValueError("Unique source CSV headers required")
        if not set(required).issubset(headers):
            raise ValueError(f"Source CSV requires columns: {sorted(required)}")
        rows = list(reader)
    if not rows or any(None in row or any(value is None for value in row.values()) for row in rows):
        raise ValueError("Nonempty complete source rows required")
    if any(not row[key].strip() for row in rows for key in required):
        raise ValueError("Required source values must be nonempty")
    return rows


def potency(value: str) -> tuple[str, float | None]:
    if value == "not tested":
        return "untested", None
    if value == "not active":
        return "inactive_gt50_uM", None
    numeric = float(value.split("±")[0].strip())
    if not np.isfinite(numeric) or numeric <= 0:
        raise ValueError("Positive finite potency required")
    return "numeric_ic50_uM", numeric


def score_panel(
    training: Sequence[Compound], panel: Sequence[Compound], seed: int
) -> list[dict[str, Any]]:
    features = np.stack([smiles_to_fp(r.smiles) for r in training])
    query = np.stack([smiles_to_fp(r.smiles) for r in panel])
    labels = labels_of(training)
    scores = fit_scores(
        features, labels, query, property_matrix(training), property_matrix(panel), seed
    )
    similarity = tanimoto_matrix(query, features)
    training_identity = [{"id": r.identifier, **identity(r.smiles)} for r in training]
    rows = []
    for index, record in enumerate(panel):
        info = identity(record.smiles)
        nearest = int(np.argmax(similarity[index]))
        active_indices = np.flatnonzero(labels == 1)
        nearest_active = int(active_indices[np.argmax(similarity[index, active_indices])])
        rows.append(
            {
                "id": record.identifier,
                **info,
                "training_matches": identity_matches(info, training_identity),
                "max_training_tanimoto": float(similarity[index, nearest]),
                "nearest_training_id": training[nearest].identifier,
                "max_active_tanimoto": float(similarity[index, nearest_active]),
                "nearest_active_id": training[nearest_active].identifier,
                "training_scaffold_matches": [
                    r.identifier for r in training if r.scaffold == record.scaffold
                ],
                "properties": compute_props(record.smiles),
                "scores": {name: float(values[index]) for name, values in scores.items()},
            }
        )
    return rows


def measured_challenge(training: Sequence[Compound], ledger: Path, seed: int) -> dict[str, Any]:
    source = read_source_csv(
        ledger,
        ("source_compound", "source_smiles", "nudt5_ic50_uM_as_reported", "inactive_definition"),
    )
    ids = [row["source_compound"] for row in source]
    structures = [identity(row["source_smiles"])["canonical_smiles"] for row in source]
    if len(set(ids)) != len(ids) or len(set(structures)) != len(structures):
        raise ValueError("Measured source requires unique compound IDs and chemical identities")
    for source_entry in source:
        if source_entry["nudt5_ic50_uM_as_reported"] == "not active" and not re.fullmatch(
            r"IC50\s*>\s*50\s*[µμu]M", source_entry["inactive_definition"].split(";")[0].strip()
        ):
            raise ValueError("Inactive censoring must explicitly mean IC50 >50 µM")
    panel = [source_compound(row["source_compound"], row["source_smiles"], row) for row in source]
    scored = score_panel(training, panel, seed)
    eligible = []
    for row, original in zip(scored, source, strict=True):
        status, value = potency(original["nudt5_ic50_uM_as_reported"])
        row.update({"assay_status": status, "ic50_uM": value, "source": original})
        row["exclusion_reason"] = (
            "training_identity_or_parent_overlap"
            if any(row["training_matches"].values())
            else "untested"
            if status == "untested"
            else None
        )
        if row["exclusion_reason"] is None:
            eligible.append(row)
    methods = list(scored[0]["scores"])
    sensitivities = {}
    for cutoff in (1.0, 10.0, 50.0):
        labels = np.array(
            [int(row["ic50_uM"] is not None and row["ic50_uM"] < cutoff) for row in eligible],
            dtype=np.int64,
        )
        sensitivities[str(cutoff)] = {
            "n": len(labels),
            "positives": int(labels.sum()),
            "metrics": {
                name: extended_metrics(labels, np.array([row["scores"][name] for row in eligible]))
                for name in methods
            }
            if len(np.unique(labels)) == 2
            else None,
        }
    return {
        "design": (
            "Previously inspected single-publication retrospective challenge; "
            "no tuning. Not external prospective validation or potency "
            "prediction. Original training labels retained."
        ),
        "rows": scored,
        "threshold_sensitivity_uM": sensitivities,
        "eligible_ids": [row["id"] for row in eligible],
        "labels": (
            "Numeric IC50 below cutoff versus measured values above "
            "cutoff/censored inactive >50uM. Untested excluded; known overlap "
            "excluded by exact, neutral parent or canonical parent tautomer."
        ),
    }


def probe_challenge(training: Sequence[Compound], ledger: Path, seed: int) -> dict[str, Any]:
    sources = read_source_csv(
        ledger,
        ("record_id", "compound_name", "source_smiles", "inchikey", "endpoint_family", "value_nM"),
    )
    unique: dict[str, dict[str, str]] = {}
    for source in sources:
        key = identity(source["source_smiles"])["inchikey"]
        if key != source["inchikey"]:
            raise ValueError("Recomputed source InChIKey mismatch")
        unique.setdefault(key, source)
    name_counts = Counter(row["compound_name"] for row in unique.values())
    panel = [
        source_compound(
            row["compound_name"]
            if name_counts[row["compound_name"]] == 1
            else f"{row['compound_name']} [{row['inchikey']}]",
            row["source_smiles"],
            row,
        )
        for row in unique.values()
    ]
    scored = score_panel(training, panel, seed)
    reference = next(r for r in panel if r.identifier == "TH5427 (known reference)")
    reference_features = np.stack([smiles_to_fp(reference.smiles)])
    for row, record in zip(scored, panel, strict=True):
        row["source_measurements"] = [s for s in sources if s["inchikey"] == row["inchikey"]]
        row["authentic_TH5427_tanimoto"] = float(
            tanimoto_matrix(np.stack([smiles_to_fp(record.smiles)]), reference_features)[0, 0]
        )
    by_id = {r["id"]: r for r in scored}
    strong, weak = by_id["MRK-952"], by_id["MRK-952-NC"]
    contrast = {
        name: {
            "MRK952": strong["scores"][name],
            "MRK952_NC": weak["scores"][name],
            "difference": strong["scores"][name] - weak["scores"][name],
        }
        for name in strong["scores"]
    }
    for name in ("mw", "clogp"):
        contrast[f"higher_{name}_baseline"] = {
            "MRK952": strong["properties"][name],
            "MRK952_NC": weak["properties"][name],
            "difference": strong["properties"][name] - weak["properties"][name],
        }
    contrast["authentic_TH5427_similarity"] = {
        "MRK952": strong["authentic_TH5427_tanimoto"],
        "MRK952_NC": weak["authentic_TH5427_tanimoto"],
        "difference": strong["authentic_TH5427_tanimoto"] - weak["authentic_TH5427_tanimoto"],
    }
    return {
        "rows": scored,
        "MRK_pair_contrast": contrast,
        "design": (
            "One exposed probe pair; positive difference agrees with its "
            "published potency ordering but cannot establish potency "
            "prediction. No AUC or significance claim. NC is weakly inhibitory, "
            "not inactive. Species/construct unspecified; NC stereochemistry "
            "assumed by source. Database counter-screen and lysate apparent Kd "
            "are separate evidence tiers/endpoints."
        ),
    }


def candidate_table(
    training: Sequence[Compound], candidates: Sequence[Compound], seed: int
) -> list[dict[str, Any]]:
    rows = score_panel(training, candidates, seed)
    audits = {r["id"]: r for r in candidate_audit(training, candidates)}
    ranks = cast(Any, rankdata)(-np.array([r["scores"]["Equal_mean"] for r in rows]), method="min")
    novelty_ranks = cast(Any, rankdata)(
        np.array([r["max_training_tanimoto"] for r in rows]), method="min"
    )
    for i, row in enumerate(rows):
        row.update(
            {
                "equal_mean_score_rank": int(ranks[i]),
                "training_dissimilarity_rank": int(novelty_ranks[i]),
                "alerts_and_properties": audits[row["id"]],
                "activity_novelty_patent_status": "not established by computational scoring",
                "experimental_role": "published_control_not_discovery"
                if any(row["training_matches"].values())
                else "identity_and_assay_qualification_required",
                "priority_basis": (
                    "No efficacy-based experimental ranking is justified; score "
                    "and training dissimilarity ranks are independent "
                    "descriptive axes."
                ),
            }
        )
    return rows


def substitute_references(
    training: Sequence[Compound], reference_path: Path
) -> tuple[list[Compound], list[dict[str, Any]]]:
    sources = read_source_csv(reference_path, ("training_id", "source_smiles", "source_url"))
    by_id = {row["training_id"]: row for row in sources}
    if len(by_id) != len(sources) or not set(by_id).issubset(r.identifier for r in training):
        raise ValueError("Unique reference IDs must all occur in the training input")
    corrected, changes = [], []
    for record in training:
        if record.identifier not in by_id:
            corrected.append(record)
            continue
        source = by_id[record.identifier]
        replacement = source_compound(record.identifier, source["source_smiles"], record.source_row)
        corrected.append(replace(replacement, label=record.label))
        changes.append(
            {
                "id": record.identifier,
                "original": identity(record.smiles),
                "replacement": identity(replacement.smiles),
                "reference_source": source,
            }
        )
    unique, _ = deduplicate(corrected)
    if len(unique) != len(corrected):
        raise ValueError("Reference substitutions produce duplicate training identities")
    return corrected, changes


def reference_sensitivity(
    training: Sequence[Compound],
    candidates: Sequence[Compound],
    assay: Path,
    probes: Path,
    references: Path,
    seed: int,
) -> dict[str, Any]:
    corrected, changes = substitute_references(training, references)
    labels = labels_of(corrected)
    evaluations = {}
    for split in ("molecule", "scaffold", "series"):
        scores, assignments = out_of_fold(corrected, split=split, folds=5, seed=seed)
        evaluations[split] = {
            "metrics": {name: extended_metrics(labels, values) for name, values in scores.items()},
            "assignments": assignments,
            "predictions": [
                {
                    "id": record.identifier,
                    "label": int(labels[i]),
                    "scores": {name: float(values[i]) for name, values in scores.items()},
                }
                for i, record in enumerate(corrected)
            ],
        }
    return {
        "design": "Secondary sensitivity, not replacement of historical data. Only authenticated "
        "reference graphs change; labels/order/seed/hyperparameters are retained. "
        "Scaffolds and folds are recomputed. No best-outcome selection or new assays.",
        "changes": changes,
        "evaluations": evaluations,
        "measured_source_challenge": measured_challenge(corrected, assay, seed),
        "probe_challenge": probe_challenge(corrected, probes, seed),
        "candidates": candidate_table(corrected, candidates, seed),
    }


def run(
    compounds: Path,
    candidates_path: Path,
    assay_ledger: Path,
    probe_ledger: Path,
    output: Path,
    *,
    seed: int = 42,
    references: Path = ROOT / "research/reference_structures.csv",
    allow_invalid: bool = False,
    acknowledge_unverified_labels: bool = False,
) -> dict[str, Any]:
    if not acknowledge_unverified_labels:
        raise ValueError("Require --acknowledge-unverified-labels")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError("Output must be new or empty")
    records, issues = read_compounds(compounds)
    if issues and not allow_invalid:
        raise ValueError("Invalid records require --allow-invalid")
    training, _ = deduplicate(records)
    candidates, candidate_issues = read_compounds(candidates_path, labelled=False)
    if candidate_issues:
        raise ValueError("Candidate input has invalid records")
    report = {
        "seed": seed,
        "training_n": len(training),
        "training_identity": [
            {"id": r.identifier, **identity(r.smiles), "original_source": r.source_row}
            for r in training
        ],
        "invalid_training_records": issues,
        "normalization_policy": (
            "Original graphs train all models. Salt-neutralized parent and "
            "canonical tautomer are screening keys only; tautomer normalization "
            "may remove stereochemical detail and does not prove "
            "interchangeable biological activity. No salts, charge or "
            "stereochemistry are silently edited in source inputs."
        ),
        "measured_source_challenge": measured_challenge(training, assay_ledger, seed),
        "probe_challenge": probe_challenge(training, probe_ledger, seed),
        "candidates": candidate_table(training, candidates, seed),
        "reference_sensitivity": reference_sensitivity(
            training, candidates, assay_ledger, probe_ledger, references, seed
        ),
    }
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "transfer.json", report)
    write_json(
        output / "manifest.json",
        input_manifest(
            [
                compounds,
                candidates_path,
                assay_ledger,
                probe_ledger,
                references,
                Path(__file__),
                Path(__file__).with_name("controls.py"),
                Path(__file__).with_name("pipeline.py"),
                *[
                    path
                    for path in (ROOT / "requirements.lock", ROOT / "research/extension_design.md")
                    if path.is_file()
                ],
            ],
            {
                "command": "transfer",
                "seed": seed,
                "allow_invalid": allow_invalid,
                "acknowledge_unverified_labels": acknowledge_unverified_labels,
            },
        ),
    )
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compounds", type=Path, default=ROOT / "compounds.csv")
    parser.add_argument("--candidates", type=Path, default=ROOT / "final_hits.csv")
    parser.add_argument("--assay-ledger", type=Path, default=ROOT / "research/source_assays.csv")
    parser.add_argument(
        "--probe-ledger", type=Path, default=ROOT / "research/external/nudt5_measured_ledger.csv"
    )
    parser.add_argument(
        "--references", type=Path, default=ROOT / "research/reference_structures.csv"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--allow-invalid", action="store_true")
    parser.add_argument("--acknowledge-unverified-labels", action="store_true")
    args = parser.parse_args(argv)
    try:
        run(
            args.compounds,
            args.candidates,
            args.assay_ledger,
            args.probe_ledger,
            args.output,
            seed=args.seed,
            references=args.references,
            allow_invalid=args.allow_invalid,
            acknowledge_unverified_labels=args.acknowledge_unverified_labels,
        )
    except (ValueError, OSError, csv.Error) as error:
        print(f"error: {error}")
        return 2
    print(f"Retrospective diagnostic results: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
