"""Descriptive re-expressions of frozen scores; no fitting or new observations."""

from __future__ import annotations

import math
from collections import Counter
from typing import Any

import numpy as np
from sklearn.metrics import roc_auc_score


def auc_estimands(rows: list[dict[str, Any]], scores: list[float]) -> dict[str, Any]:
    if len(rows) != len(scores) or not rows:
        raise ValueError("Scores must align with nonempty prediction rows")
    labels = [r["label"] for r in rows]
    positives = sum(labels)
    pairs = positives * (len(rows) - positives)
    numerator, denominator = 0.0, 0
    for fold in sorted({r["fold"] for r in rows}):
        indices = [i for i, row in enumerate(rows) if row["fold"] == fold]
        y = [labels[i] for i in indices]
        n_pairs = sum(y) * (len(y) - sum(y))
        if n_pairs:
            numerator += float(roc_auc_score(y, [scores[i] for i in indices])) * n_pairs
            denominator += n_pairs
    return {
        "pooled_auc": float(roc_auc_score(labels, scores)) if pairs else None,
        "within_fold_auc": numerator / denominator if denominator else None,
        "pooled_pairs": pairs,
        "within_fold_pairs": denominator,
    }


def conformal_sparsity(conformal: dict[str, Any]) -> dict[str, Any]:
    assignments = conformal["assignments"]
    calibration = []
    for assignment in assignments:
        counts = assignment["calibration_class_counts"]
        minima = {str(c): 1 / (counts[str(c)] + 1) for c in (0, 1)}
        if any(
            not math.isclose(minima[k], assignment["minimum_pvalue"][k], abs_tol=1e-12)
            for k in minima
        ):
            raise ValueError("Conformal minimum p-values disagree with class counts")
        calibration.append(
            {"fold": assignment["outer_fold"], "counts": counts, "minimum_pvalue": minima}
        )
    levels = []
    for alpha in (0.1, 0.2):
        predictions = conformal["predictions"]
        sets = [[i for i, value in enumerate(r["pvalues"]) if value > alpha] for r in predictions]
        if any(s != r["sets"][str(alpha)] for s, r in zip(sets, predictions, strict=True)):
            raise ValueError("Conformal sets disagree with strict p > alpha rule")
        counts = Counter(map(len, sets))
        levels.append(
            {
                "alpha": alpha,
                "n": len(sets),
                "set_size_counts": {str(k): counts[k] for k in (0, 1, 2)},
                "forced_inclusion": {
                    str(c): sum(
                        len(a["test_indices"])
                        for a in assignments
                        if a["minimum_pvalue"][str(c)] > alpha
                    )
                    for c in (0, 1)
                },
                "both_forced": sum(
                    len(a["test_indices"])
                    for a in assignments
                    if all(p > alpha for p in a["minimum_pvalue"].values())
                ),
            }
        )
    return {"calibration": calibration, "levels": levels}


def group_resampling(
    rows: list[dict[str, Any]], group_key: str, *, draws: int = 1000, seed: int = 42
) -> dict[str, Any]:
    """Pairwise AUC permits ties; frozen scores are never refitted."""
    y = np.array([r["label"] for r in rows])
    scores = np.array([r["scores"]["Equal_mean"] for r in rows])
    groups = np.array([r[group_key] for r in rows])
    members = [np.flatnonzero(groups == g) for g in np.unique(groups)]
    if not members or draws < 1:
        raise ValueError("Resampling needs groups and positive draw count")
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(draws):
        indices = np.concatenate([members[i] for i in rng.integers(0, len(members), len(members))])
        p, n = scores[indices][y[indices] == 1], scores[indices][y[indices] == 0]
        if len(p) and len(n):
            values.append(
                float(((p[:, None] > n).sum() + 0.5 * (p[:, None] == n).sum()) / (len(p) * len(n)))
            )
    return {
        "group_key": group_key,
        "seed": seed,
        "draws": draws,
        "model_refitted": False,
        "valid_draws": len(values),
        "single_class_draws": draws - len(values),
        "percentile_range_95": np.quantile(values, [0.025, 0.975]).tolist() if values else None,
    }


def diagnostic_reexpressions(controls: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, evaluation in controls["evaluations"].items():
        rows = evaluation["predictions"]
        methods = {
            method: auc_estimands(rows, [r["scores"][method] for r in rows])
            for method in evaluation["methods"]
        }
        for method, ablation in evaluation["ablations"].items():
            methods[method] = auc_estimands(rows, ablation["scores"])
        result[name] = {
            "methods": methods,
            "conformal": conformal_sparsity(evaluation["conformal"]),
        }
    result["similarity_split_grouping_sensitivity"] = [
        group_resampling(controls["evaluations"]["similarity_component_split"]["predictions"], key)
        for key in ("scaffold", "group")
    ]
    return result
