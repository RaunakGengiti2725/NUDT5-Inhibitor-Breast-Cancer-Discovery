"""Fixed-design diagnostics of property, split and uncertainty sensitivity.

All reported activity labels retain their original evidentiary limitations.
A control that fails to remove discrimination does not exclude confounding.
These analyses measure neither inhibition nor therapeutic effects.
"""

from __future__ import annotations

import argparse
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray
from pipeline import (
    PROPERTY_NAMES,
    ROOT,
    Compound,
    auc_resampling,
    compute_props,
    deduplicate,
    fit_scores,
    input_manifest,
    ranking_inputs,
    read_compounds,
    scaffold_splits,
    smiles_to_fp,
    write_json,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    auc,
    average_precision_score,
    matthews_corrcoef,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]

CONTROL_WARNING = (
    "Confound diagnostics on unverified source labels versus untested decoys. "
    "These controls bound alternative explanations of ranking performance. They are "
    "not measurements of NUDT5 inhibition, selectivity, novelty or efficacy."
)
SIMILARITY_EDGES = (0.0, 0.3, 0.5, 0.7, 0.85, 1.0001)
ALPHAS = (0.1, 0.2)


class SplitInfeasibleError(ValueError):
    """The prespecified split cannot retain a usable training set."""


def labels_of(records: Sequence[Compound]) -> IntArray:
    if any(record.label is None for record in records):
        raise ValueError("Confound controls require labelled records")
    return np.asarray([record.label for record in records], dtype=np.int64)


def property_matrix(records: Sequence[Compound]) -> FloatArray:
    rows = [compute_props(record.smiles) for record in records]
    return np.array([[row[name] for name in PROPERTY_NAMES] for row in rows], dtype=np.float64)


def standardize(matrix: FloatArray) -> FloatArray:
    if matrix.ndim != 2 or not len(matrix) or not np.isfinite(matrix).all():
        raise ValueError("A finite nonempty matrix is required")
    spread = matrix.std(axis=0)
    spread[spread == 0] = 1.0
    return cast(FloatArray, (matrix - matrix.mean(axis=0)) / spread)


def balance_report(scaled: FloatArray, labels: IntArray, indices: Sequence[int]) -> dict[str, Any]:
    """Group mean gap in original full-cohort SD units, not pooled within-class SD."""
    subset = np.asarray(indices, dtype=np.int64)
    positive = scaled[subset][labels[subset] == 1]
    negative = scaled[subset][labels[subset] == 0]
    gaps = {
        name: float(positive[:, column].mean() - negative[:, column].mean())
        for column, name in enumerate(PROPERTY_NAMES)
    }
    return {
        "mean_gap_in_full_cohort_sd": gaps,
        "max_absolute_difference": float(max(abs(value) for value in gaps.values())),
        "mean_absolute_difference": float(np.mean([abs(value) for value in gaps.values()])),
        "positives": int(len(positive)),
        "negatives": int(len(negative)),
    }


def match_property_pairs(
    records: Sequence[Compound], labels: IntArray, scaled: FloatArray
) -> list[dict[str, Any]]:
    """Greedy 1:1 nearest-neighbour matching of decoys to positives in property space.

    Positives are processed in identifier order and decoys are consumed without
    replacement, so the pairing is deterministic and contains no reused negative.
    """
    positives = sorted(np.flatnonzero(labels == 1).tolist(), key=lambda i: records[i].identifier)
    negatives = sorted(np.flatnonzero(labels == 0).tolist(), key=lambda i: records[i].identifier)
    used: set[int] = set()
    pairs: list[dict[str, Any]] = []
    for index in positives:
        candidates = [n for n in negatives if n not in used]
        if not candidates:
            break
        distances = [float(np.linalg.norm(scaled[index] - scaled[n])) for n in candidates]
        choice = candidates[int(np.argmin(distances))]
        used.add(choice)
        pairs.append(
            {
                "positive_id": records[index].identifier,
                "negative_id": records[choice].identifier,
                "positive_index": int(index),
                "negative_index": int(choice),
                "standardized_distance": float(min(distances)),
            }
        )
    return pairs


def univariate_descriptor_auc(labels: IntArray, properties: FloatArray) -> dict[str, Any]:
    """Can a single descriptor rank positives above decoys without any model?"""
    output: dict[str, Any] = {}
    for column, name in enumerate(PROPERTY_NAMES):
        values = properties[:, column]
        forward = float(roc_auc_score(labels, values))
        output[name] = {
            "auc_higher_is_positive": forward,
            "auc_best_direction": max(forward, 1.0 - forward),
            "direction": "higher" if forward >= 0.5 else "lower",
        }
    return output


def extended_metrics(labels: IntArray, scores: FloatArray) -> dict[str, float | int | None]:
    labels, scores = ranking_inputs(labels, scores)
    if np.any((scores < 0) | (scores > 1)):
        raise ValueError("Scores must lie in [0, 1]")
    calls = (scores >= 0.5).astype(np.int64)
    true_positive = int(np.sum((calls == 1) & (labels == 1)))
    false_positive = int(np.sum((calls == 1) & (labels == 0)))
    true_negative = int(np.sum((calls == 0) & (labels == 0)))
    false_negative = int(np.sum((calls == 0) & (labels == 1)))
    sensitivity = true_positive / max(true_positive + false_negative, 1)
    specificity = true_negative / max(true_negative + false_positive, 1)
    predicted_positive = true_positive + false_positive
    predicted_negative = true_negative + false_negative
    precision_curve, recall_curve, _ = precision_recall_curve(labels, scores)
    return {
        "n": int(len(labels)),
        "prevalence": float(labels.mean()),
        "auc": float(roc_auc_score(labels, scores)),
        "average_precision": float(average_precision_score(labels, scores)),
        "pr_auc_trapezoid": float(auc(recall_curve, precision_curve)),
        "f1_at_0_5": float(
            2 * true_positive / max(2 * true_positive + false_positive + false_negative, 1)
        ),
        "mcc_at_0_5": float(matthews_corrcoef(labels, calls)),
        "balanced_accuracy_at_0_5": float((sensitivity + specificity) / 2),
        "sensitivity_at_0_5": float(sensitivity),
        "specificity_at_0_5": float(specificity),
        "precision_at_0_5": float(true_positive / predicted_positive)
        if predicted_positive
        else None,
        "npv_at_0_5": float(true_negative / predicted_negative) if predicted_negative else None,
        "brier": float(np.mean((scores - labels) ** 2)),
    }


def calibration_bins(labels: IntArray, scores: FloatArray, bins: int = 5) -> dict[str, Any]:
    """Use emitted NumPy linspace float edges, left-closed; the last bin includes 1."""
    labels, scores = ranking_inputs(labels, scores)
    if (
        isinstance(bins, bool)
        or not isinstance(bins, int)
        or bins < 1
        or np.any((scores < 0) | (scores > 1))
    ):
        raise ValueError("Positive bins and bounded scores required")
    edges = np.linspace(0.0, 1.0, bins + 1)
    rows = []
    error = 0.0
    for lower, upper in zip(edges[:-1], edges[1:], strict=True):
        last = math.isclose(upper, 1.0)
        mask = (scores >= lower) & ((scores <= upper) if last else (scores < upper))
        if not mask.any():
            continue
        observed = float(labels[mask].mean())
        predicted = float(scores[mask].mean())
        error += (int(mask.sum()) / len(labels)) * abs(observed - predicted)
        rows.append(
            {
                "interval": [float(lower), float(upper)],
                "n": int(mask.sum()),
                "mean_predicted": predicted,
                "observed_positive_rate": observed,
            }
        )
    return {"bins": rows, "expected_calibration_error": float(error)}


def similarity_bins(
    labels: IntArray, scores: FloatArray, similarity: FloatArray
) -> list[dict[str, Any]]:
    """Fixed similarity intervals; one-class bins have undefined AUC, not zero."""
    labels, scores = ranking_inputs(labels, scores)
    if (
        similarity.shape != labels.shape
        or not np.isfinite(similarity).all()
        or np.any((similarity < 0) | (similarity > 1))
    ):
        raise ValueError("Aligned finite similarities in [0, 1] required")
    rows = []
    for lower, upper in zip(SIMILARITY_EDGES[:-1], SIMILARITY_EDGES[1:], strict=True):
        mask = (similarity >= lower) & (similarity < upper)
        if not mask.any():
            continue
        both = len(np.unique(labels[mask])) == 2
        rows.append(
            {
                "similarity_interval": [float(lower), min(float(upper), 1.0)],
                "n": int(mask.sum()),
                "positives": int(labels[mask].sum()),
                "auc": float(roc_auc_score(labels[mask], scores[mask])) if both else None,
                "mean_score_positives": (
                    float(scores[mask][labels[mask] == 1].mean())
                    if int(labels[mask].sum())
                    else None
                ),
                "mean_score_negatives": (
                    float(scores[mask][labels[mask] == 0].mean())
                    if int((labels[mask] == 0).sum())
                    else None
                ),
            }
        )
    return rows


def conformal_pvalues(reference: FloatArray, probability: float) -> float:
    if reference.ndim != 1 or not np.isfinite(reference).all() or not 0 <= probability <= 1:
        raise ValueError("Finite reference and bounded probability required")
    return float((np.count_nonzero(reference >= 1 - probability) + 1) / (len(reference) + 1))


def conformal_sets(
    features: FloatArray,
    labels: IntArray,
    groups: Sequence[str],
    splits: Sequence[tuple[IntArray, IntArray]],
    seed: int,
) -> dict[str, Any]:
    """Group-disjoint calibration stress test; no exchangeability guarantee is asserted."""
    rows: list[dict[str, Any]] = []
    assignments = []
    for fold, (train, test) in enumerate(splits):
        try:
            inner = scaffold_splits(labels[train], [groups[i] for i in train], 2, seed)
        except ValueError as error:
            return {"status": "infeasible", "reason": str(error), "outer_fold": fold}
        proper, calibrate = inner[0]
        proper_index, calibration_index = train[proper], train[calibrate]
        model = RandomForestClassifier(
            n_estimators=100, max_features="sqrt", random_state=seed, n_jobs=1
        )
        model.fit(features[proper_index], labels[proper_index])
        calibration = np.asarray(model.predict_proba(features[calibration_index]), dtype=np.float64)
        test_probability = np.asarray(model.predict_proba(features[test]), dtype=np.float64)
        reference = {
            label: 1 - calibration[labels[calibration_index] == label, label] for label in (0, 1)
        }
        assignments.append(
            {
                "outer_fold": fold,
                "proper_indices": proper_index.tolist(),
                "calibration_indices": calibration_index.tolist(),
                "test_indices": test.tolist(),
                "calibration_class_counts": {str(c): len(reference[c]) for c in (0, 1)},
                "minimum_pvalue": {str(c): 1 / (len(reference[c]) + 1) for c in (0, 1)},
            }
        )
        for position, index in enumerate(test):
            pvalues = [
                conformal_pvalues(reference[c], float(test_probability[position, c]))
                for c in (0, 1)
            ]
            rows.append(
                {
                    "index": int(index),
                    "label": int(labels[index]),
                    "fold": fold,
                    "pvalues": pvalues,
                    "sets": {
                        str(alpha): [c for c in (0, 1) if pvalues[c] > alpha] for alpha in ALPHAS
                    },
                }
            )
    levels = []
    for alpha in ALPHAS:
        sets = [r["sets"][str(alpha)] for r in rows]
        levels.append(
            {
                "alpha": alpha,
                "nominal_coverage": 1 - alpha,
                "n": len(rows),
                "empirical_coverage": float(
                    np.mean([r["label"] in v for r, v in zip(rows, sets, strict=True)])
                ),
                "classwise_coverage": {
                    str(c): float(
                        np.mean([c in r["sets"][str(alpha)] for r in rows if r["label"] == c])
                    )
                    for c in (0, 1)
                },
                "mean_set_size": float(np.mean([len(v) for v in sets])),
                "singleton_rate": float(np.mean([len(v) == 1 for v in sets])),
                "empty_rate": float(np.mean([not v for v in sets])),
            }
        )
    return {
        "status": "computed",
        "method": "RF, class-conditional split conformal with group-disjoint calibration",
        "assignments": assignments,
        "predictions": sorted(rows, key=lambda r: r["index"]),
        "levels": levels,
        "assumptions": (
            "Class-conditional exchangeability is not assured under group "
            "shift, related chemistry or unverified labels. Coverage here is "
            "empirical only; tiny calibration classes can force both-label "
            "sets. An absent calibration class yields p=1, not evidence for "
            "that class."
        ),
    }


def ablation_scores(
    fingerprints: FloatArray,
    properties: FloatArray,
    labels: IntArray,
    splits: Sequence[tuple[IntArray, IntArray]],
    seed: int,
) -> dict[str, Any]:
    """Fixed feature ablations; no search or selection of the best design."""

    def forest() -> RandomForestClassifier:
        return RandomForestClassifier(
            n_estimators=100, max_features="sqrt", random_state=seed, n_jobs=1
        )

    def logistic() -> Any:
        return make_pipeline(StandardScaler(), LogisticRegression(C=1, max_iter=2000))

    combined = np.hstack([fingerprints, properties])
    single = {name: properties[:, [column]] for column, name in enumerate(PROPERTY_NAMES)}
    designs: dict[str, tuple[FloatArray, str]] = {
        "ecfp4_rf": (fingerprints, "forest"),
        "properties_lr": (properties, "logistic"),
        "ecfp4_plus_properties_rf": (combined, "forest"),
    }
    for name, matrix in single.items():
        designs[f"{name}_only_lr"] = (matrix, "logistic")
    for column, name in enumerate(PROPERTY_NAMES):
        designs[f"without_{name}_lr"] = (np.delete(properties, column, axis=1), "logistic")
    output: dict[str, Any] = {}
    for name, (matrix, kind) in designs.items():
        predictions = np.full(len(labels), np.nan)
        for train, test in splits:
            model = forest() if kind == "forest" else logistic()
            model.fit(matrix[train], labels[train])
            predictions[test] = model.predict_proba(matrix[test])[:, 1]
        if not np.isfinite(predictions).all():
            raise ValueError("Ablation predictions do not cover every record")
        output[name] = {
            "metrics": extended_metrics(labels, predictions),
            "scores": predictions.tolist(),
        }
    return output


def tanimoto_matrix(left: FloatArray, right: FloatArray) -> FloatArray:
    intersection = left @ right.T
    union = left.sum(axis=1)[:, None] + right.sum(axis=1)[None, :] - intersection
    return cast(
        FloatArray,
        np.divide(intersection, union, out=np.zeros_like(intersection), where=union != 0),
    )


def similarity_components(features: FloatArray, threshold: float = 0.7) -> list[str]:
    if not 0 < threshold <= 1:
        raise ValueError("Similarity threshold must lie in (0, 1]")
    similarity = tanimoto_matrix(features, features)
    labels = [-1] * len(features)
    for root in range(len(features)):
        if labels[root] >= 0:
            continue
        labels[root] = root
        pending = [root]
        while pending:
            node = pending.pop()
            for neighbor in np.flatnonzero(similarity[node] >= threshold):
                if labels[neighbor] < 0:
                    labels[neighbor] = root
                    pending.append(int(neighbor))
    return [f"component_{label}" for label in labels]


def paired_auc_interval(
    labels: IntArray,
    left: FloatArray,
    right: FloatArray,
    groups: Sequence[str],
    seed: int,
    draws: int = 1000,
) -> dict[str, Any]:
    ranking_inputs(labels, left)
    ranking_inputs(labels, right)
    if len(groups) != len(labels) or draws < 1:
        raise ValueError("Aligned groups and positive draws required")
    group_array = np.asarray(groups)
    members = [np.flatnonzero(group_array == group) for group in np.unique(group_array)]
    rng = np.random.default_rng(seed)
    differences = []
    for _ in range(draws):
        indices = np.concatenate([members[i] for i in rng.integers(0, len(members), len(members))])
        if len(np.unique(labels[indices])) == 2:
            differences.append(
                float(
                    roc_auc_score(labels[indices], left[indices])
                    - roc_auc_score(labels[indices], right[indices])
                )
            )
    if not differences:
        raise ValueError("No two-class bootstrap draws")
    return {
        "auc_difference": float(roc_auc_score(labels, left) - roc_auc_score(labels, right)),
        "conditional_percentile_95": np.quantile(differences, [0.025, 0.975]).tolist(),
        "valid_draws": len(differences),
        "draws": draws,
        "model_refitted": False,
        "unit": "exact Murcko scaffold",
        "contrast": "Equal_mean minus Property_LR",
    }


def evaluate(
    records: Sequence[Compound],
    indices: Sequence[int],
    seed: int,
    *,
    cluster: bool = False,
    diagnostics: bool = True,
) -> dict[str, Any]:
    subset = [records[i] for i in indices]
    labels = labels_of(subset)
    fingerprints = np.stack([smiles_to_fp(record.smiles) for record in subset])
    properties = property_matrix(subset)
    scaffolds = [record.scaffold for record in subset]
    groups = similarity_components(fingerprints) if cluster else scaffolds
    try:
        splits = scaffold_splits(labels, groups, 5, seed)
    except ValueError as error:
        raise SplitInfeasibleError(str(error)) from error
    predictions: dict[str, FloatArray] = {}
    active_similarity = np.full(len(subset), np.nan)
    all_similarity = np.full(len(subset), np.nan)
    rows = []
    fold_reports = []
    for fold, (train, test) in enumerate(splits):
        scores = fit_scores(
            fingerprints[train],
            labels[train],
            fingerprints[test],
            properties[train],
            properties[test],
            seed,
        )
        fp_lr = LogisticRegression(C=1, max_iter=2000)
        fp_lr.fit(fingerprints[train], labels[train])
        scores["Fingerprint_LR"] = np.asarray(
            fp_lr.predict_proba(fingerprints[test])[:, 1], dtype=np.float64
        )
        knn = KNeighborsClassifier(n_neighbors=5, metric="precomputed", algorithm="brute")
        train_distance = 1 - tanimoto_matrix(fingerprints[train], fingerprints[train])
        test_distance = 1 - tanimoto_matrix(fingerprints[test], fingerprints[train])
        knn.fit(train_distance, labels[train])
        scores["Tanimoto_kNN"] = np.asarray(
            np.asarray(knn.predict_proba(test_distance))[:, 1], dtype=np.float64
        )
        scores["Constant_0_5"] = np.full(len(test), 0.5)
        scores["Train_prevalence"] = np.full(len(test), float(labels[train].mean()))
        for name, values in scores.items():
            predictions.setdefault(name, np.full(len(subset), np.nan))[test] = values
        active_similarity[test] = tanimoto_matrix(
            fingerprints[test], fingerprints[train][labels[train] == 1]
        ).max(axis=1)
        all_similarity[test] = tanimoto_matrix(fingerprints[test], fingerprints[train]).max(axis=1)
        spread = properties[train].std(axis=0)
        spread[spread == 0] = 1
        fold_reports.append(
            {
                "fold": fold,
                "train_indices": train.tolist(),
                "test_indices": test.tolist(),
                "train_positives": int(labels[train].sum()),
                "test_positives": int(labels[test].sum()),
                "train_scaffolds": len(set(scaffolds[i] for i in train)),
                "test_scaffolds": len(set(scaffolds[i] for i in test)),
                "scaffold_overlap": sorted(
                    set(scaffolds[i] for i in train) & set(scaffolds[i] for i in test)
                ),
                "group_overlap": sorted(
                    set(groups[i] for i in train) & set(groups[i] for i in test)
                ),
                "test_minus_train_property_mean_in_training_sd": dict(
                    zip(
                        PROPERTY_NAMES,
                        (
                            (properties[test].mean(axis=0) - properties[train].mean(axis=0))
                            / spread
                        ).tolist(),
                        strict=True,
                    )
                ),
                "metrics": {
                    name: extended_metrics(labels[test], values) for name, values in scores.items()
                }
                if len(np.unique(labels[test])) == 2
                else None,
            }
        )
        for index in test:
            rows.append(
                {
                    "id": subset[index].identifier,
                    "index": int(index),
                    "fold": fold,
                    "label": int(labels[index]),
                    "scaffold": scaffolds[index],
                    "group": groups[index],
                    "max_training_tanimoto": float(all_similarity[index]),
                    "max_active_tanimoto": float(active_similarity[index]),
                }
            )
    components = ("RF", "GBT", "SVM_RBF", "Nearest_active")
    for removed in components:
        predictions[f"Mean_without_{removed}"] = np.mean(
            [predictions[name] for name in components if name != removed], axis=0
        )
    if any(not np.isfinite(values).all() for values in predictions.values()):
        raise ValueError("Incomplete predictions")
    for row in rows:
        row["scores"] = {name: float(values[row["index"]]) for name, values in predictions.items()}
    result: dict[str, Any] = {
        "n": len(subset),
        "positives": int(labels.sum()),
        "seed": seed,
        "split": "similarity_components_0.70" if cluster else "exact_scaffold",
        "folds": fold_reports,
        "predictions": sorted(rows, key=lambda row: row["index"]),
        "methods": {name: extended_metrics(labels, values) for name, values in predictions.items()},
    }
    if diagnostics:
        result.update(
            {
                "calibration": {
                    name: calibration_bins(labels, values) for name, values in predictions.items()
                },
                "similarity_generalization": {
                    reference: {
                        name: similarity_bins(labels, values, similarity)
                        for name, values in predictions.items()
                    }
                    for reference, similarity in [
                        ("all_training", all_similarity),
                        ("training_positive", active_similarity),
                    ]
                },
                "ablations": ablation_scores(fingerprints, properties, labels, splits, seed),
                "conformal": conformal_sets(fingerprints, labels, groups, splits, seed),
                "conditional_intervals": {
                    name: auc_resampling(labels, predictions[name], scaffolds, seed)
                    for name in ("Property_LR", "Equal_mean")
                },
                "paired_auc_difference": paired_auc_interval(
                    labels, predictions["Equal_mean"], predictions["Property_LR"], scaffolds, seed
                ),
            }
        )
    return result


def run(
    compounds: Path,
    output: Path,
    *,
    seed: int,
    allow_invalid: bool,
    acknowledge_unverified_labels: bool = False,
    repeat_seeds: int = 5,
) -> dict[str, Any]:
    if not acknowledge_unverified_labels:
        raise ValueError("Controls require --acknowledge-unverified-labels")
    if repeat_seeds < 1:
        raise ValueError("Positive repeat count required")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError("Refusing nonempty or non-directory output")
    records, issues = read_compounds(compounds)
    if issues and not allow_invalid:
        raise ValueError(f"{len(issues)} invalid records require --allow-invalid")
    unique, duplicates = deduplicate(records)
    labels = labels_of(unique)
    properties = property_matrix(unique)
    scaled = standardize(properties)
    pairs = match_property_pairs(unique, labels, scaled)
    if not pairs:
        raise ValueError("Both classes required for matching")
    matched = sorted(
        [pair["positive_index"] for pair in pairs] + [pair["negative_index"] for pair in pairs]
    )
    everything = list(range(len(unique)))
    caliper_matches = {
        unique[i].identifier: [
            unique[j].identifier
            for j in np.flatnonzero(labels == 0)
            if np.max(np.abs(scaled[i] - scaled[j])) <= 0.5
        ]
        for i in np.flatnonzero(labels == 1)
    }
    designs = [
        ("full_valid_set", everything, False),
        ("nearest_property_subset", matched, False),
        ("similarity_component_split", everything, True),
    ]
    evaluations: dict[str, Any] = {}
    sensitivity = []
    for name, indices, cluster in designs:
        for current_seed in range(seed, seed + repeat_seeds):
            try:
                result = evaluate(
                    unique, indices, current_seed, cluster=cluster, diagnostics=current_seed == seed
                )
            except SplitInfeasibleError as error:
                result = {"status": "infeasible", "reason": str(error), "seed": current_seed}
            if current_seed == seed:
                evaluations[name] = result
            sensitivity.append(
                {"design": name, **result}
                if current_seed != seed
                else {
                    "design": name,
                    "seed": current_seed,
                    "methods": result.get("methods"),
                    "status": result.get("status", "computed"),
                }
            )
    report: dict[str, Any] = {
        "warning": CONTROL_WARNING,
        "dataset": {
            "valid_unique_records": len(unique),
            "positives": int(labels.sum()),
            "decoys": int((labels == 0).sum()),
            "issues": issues,
            "duplicate_groups": duplicates,
        },
        "univariate_descriptor_auc": univariate_descriptor_auc(labels, properties),
        "property_matching": {
            "design": (
                "ID-ordered nearest-neighbour matching without replacement on "
                "seven full-cohort z-scored descriptors. Exploratory cohort "
                "selection, not training-only preprocessing or causal "
                "adjustment."
            ),
            "pairs": pairs,
            "balance_full": balance_report(scaled, labels, everything),
            "balance_subset": balance_report(scaled, labels, matched),
            "all_descriptor_caliper": 0.5,
            "caliper_matches": caliper_matches,
        },
        "evaluations": evaluations,
        "seed_sensitivity": sensitivity,
        "interpretation_limits": (
            "No guarantee of property balance. Matching can retain severe "
            "residual confounding; standardization sees the whole cohort for "
            "cohort design only. Models scale features on training data only. "
            "Similarity/fusion scores are not activity probabilities. Pooled "
            "scores mix fold-trained models, and constant training-prior scores "
            "can expose fold prevalence artifacts. Fixed-score bootstrap "
            "excludes retraining. No new confirmatory tests."
        ),
    }
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "controls.json", report)
    write_json(
        output / "manifest.json",
        input_manifest(
            [
                compounds,
                Path(__file__),
                Path(__file__).with_name("pipeline.py"),
                *[
                    path
                    for path in (ROOT / "requirements.lock", ROOT / "research/extension_design.md")
                    if path.is_file()
                ],
            ],
            {
                "seed": seed,
                "repeat_seeds": repeat_seeds,
                "command": "controls",
                "allow_invalid": allow_invalid,
                "acknowledge_unverified_labels": acknowledge_unverified_labels,
            },
        ),
    )
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=CONTROL_WARNING)
    parser.add_argument("--compounds", type=Path, default=ROOT / "compounds.csv")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--repeat-seeds", type=int, default=5)
    parser.add_argument("--allow-invalid", action="store_true")
    parser.add_argument("--acknowledge-unverified-labels", action="store_true")
    arguments = parser.parse_args(argv)
    try:
        run(
            arguments.compounds,
            arguments.output,
            seed=arguments.seed,
            allow_invalid=arguments.allow_invalid,
            acknowledge_unverified_labels=arguments.acknowledge_unverified_labels,
            repeat_seeds=arguments.repeat_seeds,
        )
    except (ValueError, OSError) as error:
        print(f"error: {error}")
        return 2
    print(f"{CONTROL_WARNING}\nResults: {arguments.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
