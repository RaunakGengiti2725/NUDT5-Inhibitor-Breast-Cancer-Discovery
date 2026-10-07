"""Audit supplied chemical records and run explicitly exploratory label diagnostics.

No measured activity is inferred from a CSV label. Raw inputs are never modified.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray
from rdkit import Chem
from rdkit.Chem import Descriptors, rdFingerprintGenerator, rdMolDescriptors
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.DataStructs.cDataStructs import BulkTanimotoSimilarity, ConvertToNumpyArray
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
ROOT = Path(__file__).resolve().parents[2]
PROPERTY_NAMES = ("mw", "clogp", "tpsa", "hbd", "hba", "nrb", "fsp3")
WARNING = (
    "Exploratory diagnostics of unverified source labels versus untested decoys. "
    "Not evidence of NUDT5 inhibition, scaffold transfer, safety or therapeutic efficacy."
)


@dataclass(frozen=True)
class Compound:
    identifier: str
    smiles: str
    canonical_smiles: str
    scaffold: str
    label: int | None
    source_row: dict[str, str]


def molecule(smiles: str) -> Any:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetNumAtoms() == 0:
        raise ValueError(f"Invalid or empty SMILES: {smiles!r}")
    return mol


def read_compounds(
    path: Path, *, labelled: bool = True
) -> tuple[list[Compound], list[dict[str, Any]]]:
    records: list[Compound] = []
    issues: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle, strict=True)
        try:
            fieldnames = list(reader.fieldnames or [])
        except csv.Error as exc:
            raise ValueError(f"{path.name}: malformed CSV header: {exc}") from exc
        required = {"id", "smiles"} | ({"label"} if labelled else set())
        if not required.issubset(fieldnames):
            raise ValueError(f"{path.name}: required columns are {sorted(required)}")
        repeated = sorted({name for name in fieldnames if fieldnames.count(name) > 1})
        if repeated:
            raise ValueError(f"{path.name}: duplicate column names are ambiguous: {repeated}")
        try:
            rows = list(enumerate(reader, start=2))
        except csv.Error as exc:
            raise ValueError(f"{path.name}: malformed CSV content: {exc}") from exc
        for line, raw in rows:
            if None in raw or any(value is None for value in raw.values()):
                issues.append(
                    {
                        "line": line,
                        "id": raw.get("id"),
                        "error": "Malformed CSV row",
                        "source_row": {str(k): v for k, v in raw.items()},
                    }
                )
                continue
            row = {key: value.strip() for key, value in raw.items()}
            identifier = row["id"]
            error = None
            if not identifier or identifier in seen_ids:
                error = "Missing or duplicate record ID"
            seen_ids.add(identifier)
            if labelled and row["label"] not in {"0", "1"}:
                error = "Label must be exactly 0 or 1"
            try:
                mol = molecule(row["smiles"])
            except ValueError as exc:
                error = str(exc)
            if error:
                issues.append({"line": line, "id": identifier, "error": error, "source_row": row})
                continue
            canonical = Chem.MolToSmiles(mol, isomericSmiles=True)
            scaffold_function = cast(
                Callable[[str | None, Any, bool], str], MurckoScaffold.MurckoScaffoldSmiles
            )
            scaffold = scaffold_function(None, mol, False)
            records.append(
                Compound(
                    identifier,
                    row["smiles"],
                    canonical,
                    scaffold or "ACYCLIC",
                    int(row["label"]) if labelled else None,
                    row,
                )
            )
    if not records:
        raise ValueError(f"{path.name}: no valid records; issues={issues}")
    return records, issues


def deduplicate(records: Sequence[Compound]) -> tuple[list[Compound], list[dict[str, Any]]]:
    by_identity: dict[str, list[Compound]] = defaultdict(list)
    for record in records:
        by_identity[record.canonical_smiles].append(record)
    unique: list[Compound] = []
    duplicates: list[dict[str, Any]] = []
    for identity, group in sorted(by_identity.items()):
        if len({record.label for record in group}) > 1:
            raise ValueError(f"Conflicting labels for structure: {[r.identifier for r in group]}")
        ordered = sorted(group, key=lambda record: record.identifier)
        unique.append(ordered[0])
        if len(ordered) > 1:
            duplicates.append(
                {
                    "canonical_smiles": identity,
                    "ids": [record.identifier for record in ordered],
                    "representative": ordered[0].identifier,
                }
            )
    return sorted(unique, key=lambda record: record.identifier), duplicates


def smiles_to_bitvect(smiles: str, radius: int = 2, nbits: int = 2048) -> Any:
    if radius < 0 or nbits < 1:
        raise ValueError("Fingerprint radius must be nonnegative and bit count positive")
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=nbits)
    return generator.GetFingerprint(molecule(smiles))


def smiles_to_fp(smiles: str, radius: int = 2, nbits: int = 2048) -> FloatArray:
    vector = np.zeros(nbits, dtype=np.float64)
    ConvertToNumpyArray(smiles_to_bitvect(smiles, radius, nbits), vector)
    return vector


def compute_props(smiles: str) -> dict[str, float]:
    mol = molecule(smiles)
    return {
        "mw": float(cast(Any, Descriptors).MolWt(mol)),
        "clogp": float(rdMolDescriptors.CalcCrippenDescriptors(mol)[0]),
        "tpsa": float(rdMolDescriptors.CalcTPSA(mol)),
        "hbd": float(rdMolDescriptors.CalcNumHBD(mol)),
        "hba": float(rdMolDescriptors.CalcNumHBA(mol)),
        "nrb": float(rdMolDescriptors.CalcNumRotatableBonds(mol)),
        "fsp3": float(rdMolDescriptors.CalcFractionCSP3(mol)),
    }


def ranking_inputs(y_true: ArrayLike, y_scores: ArrayLike) -> tuple[IntArray, FloatArray]:
    labels = np.asarray(y_true, dtype=np.float64)
    scores = np.asarray(y_scores, dtype=np.float64)
    if labels.ndim != 1 or scores.shape != labels.shape or len(labels) == 0:
        raise ValueError("Labels and scores must be nonempty, equally sized one-dimensional arrays")
    if not np.isfinite(scores).all() or not np.isin(labels, [0, 1]).all():
        raise ValueError("Scores must be finite and labels binary")
    if len(np.unique(labels)) != 2:
        raise ValueError("Ranking metrics require both classes")
    return labels.astype(np.int64), scores


def expected_rank_labels(y_true: ArrayLike, y_scores: ArrayLike) -> FloatArray:
    """Average over every ordering within exactly tied scores, without favoring IDs."""
    labels, scores = ranking_inputs(y_true, y_scores)
    order = np.argsort(-scores, kind="stable")
    ordered_scores = scores[order]
    ranked = labels[order].astype(np.float64)
    boundaries = np.r_[0, np.flatnonzero(np.diff(ordered_scores)) + 1, len(scores)]
    for start, end in zip(boundaries[:-1], boundaries[1:], strict=True):
        ranked[start:end] = ranked[start:end].mean()
    return ranked


def enrichment_factor(y_true: ArrayLike, y_scores: ArrayLike, frac: float = 0.01) -> float:
    if not math.isfinite(frac) or not 0 < frac <= 1:
        raise ValueError("Fraction must be in (0, 1]")
    ranked = expected_rank_labels(y_true, y_scores)
    count = math.ceil(frac * len(ranked))
    return float(ranked[:count].mean() / ranked.mean())


def bedroc(y_true: ArrayLike, y_scores: ArrayLike, alpha: float = 20.0) -> float:
    if not math.isfinite(alpha) or alpha <= 0:
        raise ValueError("BEDROC alpha must be finite and positive")
    labels, _ = ranking_inputs(y_true, y_scores)
    ranked = expected_rank_labels(y_true, y_scores)
    weights = np.exp(-alpha * (np.arange(len(labels), dtype=np.float64) / len(labels)))
    if alpha < np.finfo(np.float64).eps:
        weights = -np.arange(len(labels), dtype=np.float64) / len(labels)
    elif alpha < 0.01:
        # Remove the common constant before summing; expm1 retains tiny-alpha differences.
        weights = (
            np.expm1(-alpha * (np.arange(len(labels), dtype=np.float64) / len(labels))) / alpha
        )
    active = int(labels.sum())
    best, worst = weights[:active].sum(), weights[-active:].sum()
    if best == worst:
        raise ValueError("BEDROC alpha is too small for numerical precision")
    return float(np.clip((np.dot(ranked, weights) - worst) / (best - worst), 0, 1))


def consensus_scores(
    score_dict: Mapping[str, ArrayLike], weights: Mapping[str, float] | None = None
) -> FloatArray:
    """A fixed weighted mean of bounded scores, not transfer learning or calibrated risk."""
    if not score_dict:
        raise ValueError("Consensus requires at least one score vector")
    names = list(score_dict)
    arrays = [np.asarray(score_dict[name], dtype=np.float64) for name in names]
    if any(
        array.ndim != 1
        or array.size == 0
        or array.shape != arrays[0].shape
        or not np.isfinite(array).all()
        or (array < 0).any()
        or (array > 1).any()
        for array in arrays
    ):
        raise ValueError("Consensus requires equally sized, finite vectors in [0, 1]")
    if weights is None:
        coefficients: FloatArray = np.ones(len(names))
    else:
        if set(weights) != set(names):
            raise ValueError("Weight names must exactly match score names")
        coefficients = np.asarray([weights[name] for name in names], dtype=np.float64)
    if not np.isfinite(coefficients).all() or (coefficients < 0).any() or coefficients.max() <= 0:
        raise ValueError("Weights must be finite, nonnegative, and have positive total")
    coefficients = coefficients / coefficients.max()
    result = np.asarray(
        np.average(np.stack(arrays), axis=0, weights=coefficients), dtype=np.float64
    )
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite consensus arithmetic")
    return result


def permutation_pvalue(observed: float, null_scores: ArrayLike) -> float:
    null = np.asarray(null_scores, dtype=np.float64)
    if (
        null.ndim != 1
        or null.size == 0
        or not np.isfinite(null).all()
        or not math.isfinite(observed)
    ):
        raise ValueError("Permutation scores must be a nonempty finite vector")
    return float((np.count_nonzero(null >= observed) + 1) / (len(null) + 1))


def metrics(y_true: ArrayLike, scores: ArrayLike) -> dict[str, float | int]:
    labels, values = ranking_inputs(y_true, scores)
    return {
        "n": len(labels),
        "positives": int(labels.sum()),
        "auc": float(roc_auc_score(labels, values)),
        "sensitivity_at_0_5": float(np.mean(values[labels == 1] >= 0.5)),
        "specificity_at_0_5": float(np.mean(values[labels == 0] < 0.5)),
        "average_precision": float(average_precision_score(labels, values)),
        "ef_1pct": enrichment_factor(labels, values, 0.01),
        "ef_5pct": enrichment_factor(labels, values, 0.05),
        "ef_1pct_k": math.ceil(len(labels) * 0.01),
        "ef_5pct_k": math.ceil(len(labels) * 0.05),
        "bedroc20": bedroc(labels, values),
    }


def scaffold_splits(
    labels: IntArray, groups: Sequence[str], folds: int, seed: int
) -> list[tuple[IntArray, IntArray]]:
    if len(groups) != len(labels) or folds < 2 or len(set(groups)) < folds:
        raise ValueError("Insufficient aligned scaffold groups or invalid fold count")
    splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
    result: list[tuple[IntArray, IntArray]] = []
    group_array = np.asarray(groups)
    for train, test in splitter.split(np.zeros(len(labels)), labels, groups):
        if len(test) == 0 or len(np.unique(labels[train])) != 2:
            raise ValueError("A scaffold fold is empty or its training set lacks a class")
        if set(group_array[train]) & set(group_array[test]):
            raise ValueError("Scaffold overlap detected between train and test")
        result.append((train, test))
    return result


def series_splits(records: Sequence[Compound], seed: int) -> list[tuple[IntArray, IntArray]]:
    """Hold out each labelled positive series with a disjoint partition of decoys."""
    series = sorted(
        {record.source_row.get("series", "") for record in records if record.label == 1}
    )
    if len(series) < 2 or "" in series:
        raise ValueError("Series holdout requires at least two nonempty positive-series labels")
    negatives = np.asarray(
        [i for i, record in enumerate(records) if record.label == 0], dtype=np.int64
    )
    if len(negatives) < len(series):
        raise ValueError("Series holdout needs at least one test decoy per series")
    rng = np.random.default_rng(seed)
    negative_parts = np.array_split(rng.permutation(negatives), len(series))
    all_indices = np.arange(len(records), dtype=np.int64)
    splits = []
    for group, negative in zip(series, negative_parts, strict=True):
        positive = np.array(
            [
                i
                for i, record in enumerate(records)
                if record.label == 1 and record.source_row["series"] == group
            ],
            dtype=np.int64,
        )
        test = np.sort(np.r_[positive, negative])
        train = np.setdiff1d(all_indices, test)
        splits.append((train, test))
    return splits


def auc_resampling(
    labels: IntArray, scores: FloatArray, groups: Sequence[str], seed: int, draws: int = 1000
) -> dict[str, Any]:
    """Cluster resampling of fixed OOF scores, not a model-retraining confidence interval."""
    if draws < 1 or len(groups) != len(labels):
        raise ValueError("Positive draws and aligned groups are required")
    ranking_inputs(labels, scores)
    group_array = np.asarray(groups)
    identities = np.unique(group_array)
    members = [np.flatnonzero(group_array == group) for group in identities]
    rng = np.random.default_rng(seed)
    aucs = []
    one_class = 0
    for _ in range(draws):
        indices = np.concatenate([members[i] for i in rng.integers(0, len(members), len(members))])
        if len(np.unique(labels[indices])) < 2:
            one_class += 1
            continue
        aucs.append(float(roc_auc_score(labels[indices], scores[indices])))
    if not aucs:
        raise ValueError("All cluster bootstrap draws lack both classes")
    low, high = np.quantile(aucs, [0.025, 0.975])
    return {
        "conditional_auc_percentile_95": [float(low), float(high)],
        "draws": draws,
        "valid_draws": len(aucs),
        "one_class_draws": one_class,
        "unit": "exact Murcko scaffold",
        "model_refitted": False,
        "limitation": "Fixed-OOF-score descriptive interval; ignores training-set uncertainty",
    }


def randomized_label_diagnostic(
    records: Sequence[Compound],
    observed: Mapping[str, float],
    permutations: int,
    folds: int,
    seed: int,
) -> dict[str, Any]:
    if permutations < 1:
        raise ValueError("At least one permutation is required")
    labels = np.asarray([record.label for record in records], dtype=np.int64)
    rng = np.random.default_rng(seed)
    null: dict[str, list[float]] = {name: [] for name in observed}
    for iteration in range(permutations):
        shuffled = rng.permutation(labels)
        permuted = [
            replace(record, label=int(label))
            for record, label in zip(records, shuffled, strict=True)
        ]
        scores, _ = out_of_fold(permuted, split="molecule", folds=folds, seed=seed)
        for name, values in scores.items():
            null[name].append(float(roc_auc_score(shuffled, values)))
        if (iteration + 1) % 10 == 0:
            print(f"Label-permutation diagnostic: {iteration + 1}/{permutations}", flush=True)
    return {
        "permutations": permutations,
        "minimum_pvalue": 1 / (permutations + 1),
        "split": "molecule; stratification recomputed from each permuted label vector",
        "warning": (
            "Unrestricted unique-molecule label-shuffling diagnostic; "
            "correlated chemical series are not assumed biologically exchangeable. "
            "These p-values do not validate biological activity or rule out dataset bias."
        ),
        "methods": {
            name: {
                "observed_auc": observed[name],
                "null_auc": values,
                "p_plus_one": permutation_pvalue(observed[name], values),
            }
            for name, values in null.items()
        },
    }


def fit_scores(
    x_train: FloatArray,
    y_train: IntArray,
    x_test: FloatArray,
    properties_train: FloatArray,
    properties_test: FloatArray,
    seed: int,
) -> dict[str, FloatArray]:
    models: dict[str, RandomForestClassifier | GradientBoostingClassifier | SVC] = {
        "RF": RandomForestClassifier(
            n_estimators=100, max_features="sqrt", random_state=seed, n_jobs=1
        ),
        "GBT": GradientBoostingClassifier(
            n_estimators=100, max_depth=3, learning_rate=0.1, random_state=seed
        ),
        "SVM_RBF": SVC(kernel="rbf", C=10, probability=True, random_state=seed),
    }
    scores = {}
    for name, model in models.items():
        model.fit(x_train, y_train)
        scores[name] = np.asarray(model.predict_proba(x_test)[:, 1], dtype=np.float64)
    property_model = make_pipeline(StandardScaler(), LogisticRegression(C=1, max_iter=2000))
    property_model.fit(properties_train, y_train)
    scores["Property_LR"] = np.asarray(
        property_model.predict_proba(properties_test)[:, 1], dtype=np.float64
    )
    active = x_train[y_train == 1]
    intersection = x_test @ active.T
    union = x_test.sum(axis=1)[:, None] + active.sum(axis=1)[None, :] - intersection
    scores["Nearest_active"] = np.max(
        np.divide(intersection, union, out=np.zeros_like(intersection), where=union != 0), axis=1
    )
    scores["Equal_mean"] = consensus_scores(
        {name: scores[name] for name in ("RF", "GBT", "SVM_RBF", "Nearest_active")}
    )
    return scores


def out_of_fold(
    records: Sequence[Compound], *, split: str, folds: int, seed: int
) -> tuple[dict[str, FloatArray], list[dict[str, Any]]]:
    labels = np.asarray([record.label for record in records], dtype=np.int64)
    if len(np.unique(labels)) != 2 or (split != "series" and min(Counter(labels).values()) < folds):
        raise ValueError("Each class must contain at least the requested fold count")
    features = np.stack([smiles_to_fp(record.smiles) for record in records])
    descriptors = [compute_props(record.smiles) for record in records]
    properties = np.array([[row[name] for name in PROPERTY_NAMES] for row in descriptors])
    groups = [record.scaffold for record in records]
    if split == "scaffold":
        splits = scaffold_splits(labels, groups, folds, seed)
    elif split == "series":
        splits = series_splits(records, seed)
    elif split == "molecule":
        splits = list(
            StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed).split(features, labels)
        )
    else:
        raise ValueError("Split must be molecule, scaffold or series")
    predictions: dict[str, FloatArray] = {}
    assignments = []
    for fold, (train, test) in enumerate(splits):
        scores = fit_scores(
            features[train],
            labels[train],
            features[test],
            properties[train],
            properties[test],
            seed,
        )
        for name, values in scores.items():
            predictions.setdefault(name, np.full(len(labels), np.nan))[test] = values
        for index in test:
            assignments.append(
                {
                    "id": records[index].identifier,
                    "fold": fold,
                    "label": int(labels[index]),
                    "scaffold": groups[index],
                }
            )
    if any(not np.isfinite(values).all() for values in predictions.values()):
        raise ValueError("Out-of-fold predictions do not cover every record")
    return predictions, sorted(assignments, key=lambda row: row["id"])


def fold_statistics(
    records: Sequence[Compound],
    assignments: Sequence[Mapping[str, Any]],
    predictions: Mapping[str, FloatArray],
) -> dict[str, Any]:
    by_id = {record.identifier: i for i, record in enumerate(records)}
    labels = np.asarray([record.label for record in records], dtype=np.int64)
    output: dict[str, Any] = {}
    for fold in sorted({row["fold"] for row in assignments}):
        indices = np.array([by_id[row["id"]] for row in assignments if row["fold"] == fold])
        if len(np.unique(labels[indices])) != 2:
            output[str(fold)] = {
                "n": len(indices),
                "positives": int(labels[indices].sum()),
                "auc": None,
                "reason": "Test fold contains only one class",
            }
        else:
            output[str(fold)] = {
                name: metrics(labels[indices], scores[indices])
                for name, scores in predictions.items()
            }
    return output


def candidate_audit(
    records: Sequence[Compound], candidates: Sequence[Compound]
) -> list[dict[str, Any]]:
    fingerprints = [smiles_to_bitvect(record.smiles) for record in records]
    params = FilterCatalogParams()
    params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS)
    catalog = FilterCatalog(params)
    output = []
    for candidate in candidates:
        props = compute_props(candidate.smiles)
        similarity = np.asarray(
            BulkTanimotoSimilarity(smiles_to_bitvect(candidate.smiles), fingerprints)
        )
        matches = [
            record.identifier
            for record in records
            if record.canonical_smiles == candidate.canonical_smiles
        ]
        differences = {}
        for name, value in props.items():
            if name in candidate.source_row:
                try:
                    supplied = float(candidate.source_row[name])
                except ValueError as exc:
                    raise ValueError(f"{candidate.identifier}: nonnumeric supplied {name}") from exc
                if not math.isfinite(supplied):
                    raise ValueError(f"{candidate.identifier}: nonfinite supplied {name}")
                differences[name] = {
                    "reported": supplied,
                    "computed": value,
                    "delta": value - supplied,
                }
        nearest = int(np.argmax(similarity))
        active_indices = [i for i, record in enumerate(records) if record.label == 1]
        nearest_active = (
            max(active_indices, key=lambda i: float(similarity[i])) if active_indices else None
        )
        output.append(
            {
                "id": candidate.identifier,
                "canonical_smiles": candidate.canonical_smiles,
                "training_identity_matches": matches,
                "nearest_training_id": records[nearest].identifier,
                "max_training_tanimoto": float(similarity[nearest]),
                "nearest_active_id": (
                    records[nearest_active].identifier if nearest_active is not None else None
                ),
                "max_active_tanimoto": (
                    float(similarity[nearest_active]) if nearest_active is not None else None
                ),
                "properties": props,
                "reported_property_comparison": differences,
                "pains_alerts": [
                    match.GetDescription()
                    for match in catalog.GetMatches(molecule(candidate.smiles))
                ],
                "ro5_violations": sum(
                    [props["mw"] > 500, props["clogp"] > 5, props["hbd"] > 5, props["hba"] > 10]
                ),
                "veber_pass": props["nrb"] <= 10 and props["tpsa"] <= 140,
                "status": (
                    "training_overlap_not_novel"
                    if matches
                    else "not_in_training_set_novelty_and_activity_unverified"
                ),
            }
        )
    return output


def input_manifest(paths: Sequence[Path], arguments: Mapping[str, Any]) -> dict[str, Any]:
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=ROOT,
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
        )
    except (OSError, subprocess.CalledProcessError):
        revision, dirty = "unavailable", None
    return {
        "warning": WARNING,
        "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in arguments.items()},
        "python": platform.python_version(),
        "platform": platform.platform(),
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("rdkit", "scikit-learn", "numpy", "scipy", "matplotlib")
        },
        "git_revision": revision,
        "git_dirty": dirty,
        "files": {
            str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths
        },
    }


def write_json(path: Path, value: Any) -> None:
    payload = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, prefix=".nudt5-", delete=False
    ) as handle:
        staging = Path(handle.name)
        try:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        except BaseException:
            staging.unlink(missing_ok=True)
            raise
    try:
        os.link(staging, path)
    finally:
        staging.unlink(missing_ok=True)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", nargs="?", choices=["audit", "benchmark"], default="audit")
    parser.add_argument("--compounds", type=Path, default=ROOT / "compounds.csv")
    parser.add_argument("--candidates", type=Path, default=ROOT / "final_hits.csv")
    parser.add_argument("--output", type=Path, default=Path.cwd() / "results" / "audit")
    parser.add_argument(
        "--allow-invalid",
        action="store_true",
        help="Explicitly quarantine invalid records in exploratory benchmarks",
    )
    parser.add_argument("--acknowledge-unverified-labels", action="store_true")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--permutations", type=int, default=0)
    parser.add_argument("--bootstrap-draws", type=int, default=1000)
    parser.add_argument("--repeat-seeds", type=int, default=1)
    args = parser.parse_args(argv)
    try:
        if args.output.exists() and (not args.output.is_dir() or any(args.output.iterdir())):
            raise ValueError(
                "Output must be a new or empty directory; previous runs are never overwritten"
            )
        if args.seed < 0 or args.seed >= 2**32 or args.folds < 2:
            raise ValueError("Seed must be in [0, 2**32) and folds at least two")
        if args.permutations < 0 or args.bootstrap_draws < 1 or args.repeat_seeds < 1:
            raise ValueError("Invalid diagnostic draw counts")
        if args.seed + args.repeat_seeds > 2**32:
            raise ValueError("Repeated seeds exceed the supported range")
        records, issues = read_compounds(args.compounds)
        candidates, candidate_issues = read_compounds(args.candidates, labelled=False)
        unique, duplicates = deduplicate(records)
        unique_candidates, candidate_duplicates = deduplicate(candidates)
        audit = {
            "warning": WARNING,
            "valid_records": len(records),
            "unique_records": len(unique),
            "label_counts": dict(Counter(str(record.label) for record in unique)),
            "source_counts": dict(
                Counter(record.source_row.get("source", "") for record in unique)
            ),
            "unique_scaffolds": len({record.scaffold for record in unique}),
            "invalid_records": issues,
            "duplicate_structures": duplicates,
            "candidate_issues": candidate_issues,
            "candidate_duplicates": candidate_duplicates,
            "candidates": candidate_audit(unique, unique_candidates),
            "identity_policy": (
                "RDKit canonical isomeric SMILES; no salt, tautomer or protonation merging"
            ),
            "scaffold_policy": (
                "RDKit Bemis-Murcko, no chirality; all acyclic molecules share ACYCLIC"
            ),
        }
        benchmark: dict[str, Any] = {}
        if args.command == "benchmark":
            if not args.acknowledge_unverified_labels:
                raise ValueError(
                    "Benchmark requires --acknowledge-unverified-labels; read the warning"
                )
            if (issues or candidate_issues) and not args.allow_invalid:
                raise ValueError(
                    f"Invalid records: {issues + candidate_issues}; "
                    "use audit or explicitly --allow-invalid"
                )
            labels = np.asarray([record.label for record in unique], dtype=np.int64)
            for split in ("molecule", "scaffold", "series"):
                scores, assignments = out_of_fold(
                    unique, split=split, folds=args.folds, seed=args.seed
                )
                benchmark[split] = {
                    "metrics": {name: metrics(labels, values) for name, values in scores.items()},
                    "conditional_resampling": {
                        name: auc_resampling(
                            labels,
                            values,
                            [r.scaffold for r in unique],
                            args.seed,
                            args.bootstrap_draws,
                        )
                        for name, values in scores.items()
                    },
                    "fold_metrics": fold_statistics(unique, assignments, scores),
                    "assignments": assignments,
                    "predictions": [
                        {
                            "id": record.identifier,
                            "label": record.label,
                            **{name: float(values[index]) for name, values in scores.items()},
                        }
                        for index, record in enumerate(unique)
                    ],
                }
            if args.permutations:
                benchmark["label_randomization"] = randomized_label_diagnostic(
                    unique,
                    {
                        name: float(values["auc"])
                        for name, values in benchmark["molecule"]["metrics"].items()
                    },
                    args.permutations,
                    args.folds,
                    args.seed,
                )
            sensitivity = []
            for seed in range(args.seed, args.seed + args.repeat_seeds):
                for split in ("molecule", "scaffold", "series"):
                    if seed == args.seed:
                        values = benchmark[split]["metrics"]
                    else:
                        predictions, _ = out_of_fold(
                            unique, split=split, folds=args.folds, seed=seed
                        )
                        values = {
                            name: metrics(labels, scores) for name, scores in predictions.items()
                        }
                    sensitivity.append({"seed": seed, "split": split, "metrics": values})
            benchmark["seed_sensitivity"] = sensitivity
        environment_files = [
            ROOT / name
            for name in ("requirements.txt", "requirements.lock")
            if (ROOT / name).is_file()
        ]
        manifest = input_manifest(
            [args.compounds, args.candidates, Path(__file__), *environment_files], vars(args)
        )
        args.output.mkdir(parents=True, exist_ok=True)
        write_json(args.output / "audit.json", audit)
        if benchmark:
            write_json(args.output / "benchmark.json", benchmark)
        write_json(args.output / "manifest.json", manifest)
        print(WARNING)
        print(
            f"Audited {len(unique)} unique valid records; "
            f"{len(issues)} invalid records retained in report."
        )
        print(f"Results: {args.output.resolve()}")
        return 0
    except (OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
