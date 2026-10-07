"""Synthetic behavioral oracles, not biological validation or source-label evidence."""

from __future__ import annotations

import itertools
from decimal import Decimal, localcontext

import numpy as np
import pytest
from controls import (
    calibration_bins,
    conformal_pvalues,
    extended_metrics,
    paired_auc_interval,
    similarity_bins,
)
from pipeline import bedroc, consensus_scores
from transfer import potency


@pytest.mark.parametrize("scale", [5e-324, 1e-320, 1e-300, 1.0, 1e100, 1e308])
def test_consensus_scale_invariant_without_overflow(scale: float) -> None:
    with np.errstate(all="raise"):
        result = consensus_scores(
            {"a": [1.0, 0.0, 1.0, 0.1], "b": [0.0, 1.0, 1.0, 0.7]}, {"a": scale, "b": scale}
        )
    np.testing.assert_allclose(result, [0.5, 0.5, 1.0, 0.4], rtol=1e-15)


@pytest.mark.parametrize("alpha", [5e-324, 1e-320, 1e-300, 1e-14, 1e-8, 0.001, 0.01, 20.0, 1000.0])
def test_bedroc_independent_decimal_oracle(alpha: float) -> None:
    with localcontext() as ctx:
        ctx.prec = 400
        weights = [(-Decimal(str(alpha)) * Decimal(i) / 4).exp() for i in range(4)]
        for active_indices in itertools.combinations(range(4), 2):
            labels = [int(i in active_indices) for i in range(4)]
            oracle = (sum(weights[i] for i in active_indices) - sum(weights[-2:])) / (
                sum(weights[:2]) - sum(weights[-2:])
            )
            assert bedroc(labels, [4, 3, 2, 1], alpha) == pytest.approx(float(oracle), abs=2e-13)
    assert bedroc([1, 0, 1, 0], [1, 1, 1, 1], alpha) == pytest.approx(0.5, abs=2e-13)


def test_exact_half_threshold_specificity_and_conformal_tie() -> None:
    result = extended_metrics(np.array([0, 1]), np.array([0.5, 0.9]))
    assert result["specificity_at_0_5"] == 0
    assert conformal_pvalues(np.array([0.5]), 0.5) == 1
    result = extended_metrics(np.array([0, 0, 1]), np.array([0.1, 0.9, 0.9]))
    assert result["specificity_at_0_5"] == 0.5


def test_nonzero_paired_difference_and_interval_direction() -> None:
    labels = np.array([0, 1, 0, 1])
    left = np.array([0.1, 0.9, 0.2, 0.8])
    result = paired_auc_interval(labels, left, 1 - left, ["a", "b", "c", "d"], 42, 100)
    assert result["auc_difference"] == 1
    assert result["conditional_percentile_95"] == [1, 1]


def test_emitted_float_calibration_edge_contract() -> None:
    edge = np.linspace(0, 1, 6)[3]
    values = np.array([0.6, np.nextafter(edge, 0), edge, np.nextafter(edge, 1)])
    rows = calibration_bins(np.array([0, 1, 0, 1]), values)["bins"]
    assert edge > 0.6
    assert [r["n"] for r in rows] == [2, 2]
    assert rows[0]["interval"][1] == rows[1]["interval"][0] == edge
    ece = calibration_bins(np.array([0, 1]), np.array([0.9, 0.8]))["expected_calibration_error"]
    assert ece == pytest.approx(0.35)


@pytest.mark.parametrize("values", [[-0.1, 0.5], [0.5, 1.01], [[0.2, 0.3]], [float("nan"), 0.5]])
def test_invalid_similarity_cannot_disappear(values: list[float] | list[list[float]]) -> None:
    with pytest.raises(ValueError):
        similarity_bins(np.array([0, 1]), np.array([0.2, 0.7]), np.asarray(values))


@pytest.mark.parametrize(
    "value", ["2 ± garbage", "2 ±", "2 ± -1", "2 ± nan", "2 ± inf", "2 ± 1 ± 1"]
)
def test_invalid_uncertainty_refused(value: str) -> None:
    with pytest.raises(ValueError):
        potency(value)


def test_mean_only_potency_contract() -> None:
    assert potency("2 ± 0.3") == ("numeric_ic50_uM", 2.0)


def test_nonperfect_bootstrap_quantiles_by_pair_counting() -> None:
    from pipeline import auc_resampling

    labels = np.array([0, 1, 0, 1, 0, 1])
    scores = np.array([0.1, 0.9, 0.7, 0.6, 0.4, 0.2])
    rng = np.random.default_rng(42)
    values = []
    for _ in range(300):
        indices = rng.integers(0, 6, 6)
        positive = [scores[i] for i in indices if labels[i]]
        negative = [scores[i] for i in indices if not labels[i]]
        if positive and negative:
            values.append(
                sum(float(p > n) + 0.5 * (p == n) for p in positive for n in negative)
                / (len(positive) * len(negative))
            )
    values.sort()

    def quantile(q: float) -> float:
        rank = (len(values) - 1) * q
        lower = int(rank)
        return float(values[lower] + (rank - lower) * (values[lower + 1] - values[lower]))

    result = auc_resampling(labels, scores, list("abcdef"), 42, 300)
    assert result["conditional_auc_percentile_95"] == pytest.approx(
        [quantile(0.025), quantile(0.975)]
    )
    assert quantile(0.025) != quantile(0.05)


def test_ceil_enrichment_cutoff_is_not_enlarged() -> None:
    from pipeline import enrichment_factor

    assert enrichment_factor([1, 0, 0], [3, 2, 1], 0.01) == 3


def test_descriptor_inclusive_exclusive_boundaries(monkeypatch: pytest.MonkeyPatch) -> None:
    import pipeline

    records, _ = pipeline.read_compounds(pipeline.ROOT / "compounds.csv")
    properties = {
        "mw": 500.0,
        "clogp": 5.0,
        "hbd": 5.0,
        "hba": 10.0,
        "nrb": 10.0,
        "tpsa": 140.0,
        "fsp3": 0.5,
    }
    monkeypatch.setattr(pipeline, "compute_props", lambda _: properties)
    result = pipeline.candidate_audit(records[:1], records[:1])[0]
    assert result["ro5_violations"] == 0
    assert result["veber_pass"] is True


def test_molecule_seed_and_fold_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    from typing import Any

    import pipeline
    from sklearn.model_selection import StratifiedKFold

    records, _ = pipeline.read_compounds(pipeline.ROOT / "compounds.csv")

    def fixed_scores(*args: Any) -> dict[str, np.ndarray[Any, np.dtype[np.float64]]]:
        return {"fixture": np.full(len(args[2]), 0.5)}

    monkeypatch.setattr(pipeline, "fit_scores", fixed_scores)
    _, assignments = pipeline.out_of_fold(records, split="molecule", folds=5, seed=42)
    labels = np.asarray([r.label for r in records], dtype=np.int64)
    expected = {}
    for fold, (_, test) in enumerate(
        StratifiedKFold(n_splits=5, shuffle=True, random_state=42).split(
            np.zeros((len(records), 1)), labels
        )
    ):
        expected.update({records[i].identifier: fold for i in test})
    assert {r["id"]: r["fold"] for r in assignments} == expected


def test_permutation_auc_uses_the_permuted_labels(monkeypatch: pytest.MonkeyPatch) -> None:
    from typing import Any

    import pipeline

    records, _ = pipeline.read_compounds(pipeline.ROOT / "compounds.csv")
    seen = []

    def scores_for_labels(permuted: Any, **kwargs: Any) -> Any:
        seen.append([r.label for r in permuted])
        return {"fixture": np.asarray(seen[-1], dtype=float)}, []

    monkeypatch.setattr(pipeline, "out_of_fold", scores_for_labels)
    result = pipeline.randomized_label_diagnostic(records, {"fixture": 0.7}, 2, 5, 42)
    assert result["methods"]["fixture"]["null_auc"] == [1, 1]
    assert seen[0] != [r.label for r in records]


@pytest.mark.parametrize("alpha", [5e-324, 1e-320, 1e-300, 1e-14, 1e-8, 0.001, 0.01, 20.0, 1000.0])
def test_bedroc_mixed_ties_decimal_oracle(alpha: float) -> None:
    rng = np.random.default_rng(271828)
    with localcontext() as ctx:
        ctx.prec = 400
        for panel in range(29):
            n = 2 + panel % 11
            labels = rng.integers(0, 2, n).tolist()
            labels[:2] = [0, 1]
            scores = rng.integers(0, 4, n).tolist()
            order = sorted(range(n), key=lambda i: -scores[i])
            expected = [Decimal(0)] * n
            for score in set(scores):
                ranks = [rank for rank, i in enumerate(order) if scores[i] == score]
                fraction = Decimal(sum(labels[order[rank]] for rank in ranks)) / len(ranks)
                for rank in ranks:
                    expected[rank] = fraction
            weights = [(-Decimal.from_float(alpha) * rank / n).exp() for rank in range(n)]
            active = sum(labels)
            best, worst = sum(weights[:active]), sum(weights[-active:])
            oracle = (sum(w * y for w, y in zip(weights, expected, strict=True)) - worst) / (
                best - worst
            )
            assert bedroc(labels, scores, alpha) == pytest.approx(float(oracle), abs=2e-12)


@pytest.mark.parametrize("scale", [5e-324, 1e-320, 1e-300, 1.0, 1e308])
def test_fractional_consensus_matches_representable_weight_oracle(scale: float) -> None:
    for left in [0.0, 0.1, 0.3, 0.5, 0.9, 1.0]:
        score = {"a": [left, 1.0], "b": [0.7, 1.0]}
        with localcontext() as ctx:
            ctx.prec = 400
            coefficient = Decimal.from_float(scale)
            oracle = float(
                (Decimal.from_float(left) * coefficient + Decimal.from_float(0.7) * coefficient)
                / (2 * coefficient)
            )
        with np.errstate(all="raise"):
            actual = consensus_scores(score, {"a": scale, "b": scale})
        np.testing.assert_allclose(actual, [oracle, 1.0], rtol=1e-15, atol=1e-16)


@pytest.mark.parametrize(
    "bad",
    [
        "T\u034fB\u034fD",
        "M\u034fISSING",
        "T\u0300B\u0300D",
        "PLACE\u034fHOLDER text",
        "see \u034f<\u034f!\u034f-\u034f- hidden",
        "T\u00a0B\u00a0D",
    ],
)
def test_mark_hidden_placeholders_refused(bad: str) -> None:
    from author_release import valid_statement

    assert not valid_statement(bad)


@pytest.mark.parametrize(
    "good",
    [
        "Jos\u00e9 Mu\u00f1oz, Institut f\u00fcr Chemie, Berlin.",
        "Supported by grant 123; the funder had no role in analysis.",
    ],
)
def test_accented_author_text_still_accepted(good: str) -> None:
    from author_release import valid_statement

    assert valid_statement(good)
