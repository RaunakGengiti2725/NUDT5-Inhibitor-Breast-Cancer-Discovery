"""Render computed control/transfer results to vector figures and machine-readable tables."""

from __future__ import annotations

import csv
import importlib
import json
from pathlib import Path
from typing import Any

from path_b_diagnostics import diagnostic_reexpressions

METHODS = ["Property_LR", "RF", "SVM_RBF", "GBT", "Nearest_active", "Equal_mean"]
LABELS = ["Property LR", "RF", "RBF-SVM", "GBT", "Nearest active", "Equal mean"]
BLUE, ORANGE, GREEN = "#0072B2", "#D55E00", "#009E73"


def save_figure(fig: Any, output: Path, name: str) -> Path:
    for extension in ("png", "pdf", "svg"):
        path = output / f"{name}.{extension}"
        fig.savefig(path, dpi=240, bbox_inches="tight")
        if extension == "svg":
            path.write_text(
                "\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n"
            )
    return output / f"{name}.png"


def source_cutoff_caption(challenge: dict[str, Any]) -> str:
    descriptions = []
    for cutoff in (1.0, 10.0):
        row = challenge["threshold_sensitivity_uM"][str(cutoff)]
        description = f"At {cutoff:g} uM, {row['positives']} of {row['n']} are threshold-positive; "
        if row["metrics"] is None:
            description += "ROC-AUC is unavailable."
        else:
            description += (
                "ROC-AUC: "
                + ", ".join(
                    f"{name} {row['metrics'][name]['auc']:.4f}"
                    for name in ("Equal_mean", "Nearest_active", "Property_LR")
                )
                + "."
            )
        descriptions.append(description)
    return " ".join(descriptions)


def build_figures(results: Path, output: Path) -> list[tuple[Path, str]]:
    controls = json.loads((results / "controls.json").read_text())
    transfer = json.loads((results / "transfer.json").read_text())
    mpl = importlib.import_module("matplotlib")
    mpl.use("Agg")
    plt = importlib.import_module("matplotlib.pyplot")
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "svg.fonttype": "none",
        }
    )
    output.mkdir(parents=True, exist_ok=True)
    full = controls["evaluations"]["full_valid_set"]
    captions = []
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), layout="constrained")
    properties = ["mw", "clogp", "tpsa", "hbd", "hba", "nrb", "fsp3"]
    matching = controls["property_matching"]
    for index, key in enumerate(("balance_full", "balance_subset")):
        axes[0].barh(
            [i + (index - 0.5) * 0.34 for i in range(7)],
            [matching[key]["mean_gap_in_full_cohort_sd"][name] for name in properties],
            height=0.34,
            color=[BLUE, ORANGE][index],
            label=["Full cohort", "Nearest-property subset"][index],
        )
    axes[0].axvline(0, color="#777777", linewidth=1)
    axes[0].set_yticks(range(7), properties)
    axes[0].set_xlabel("Positive minus decoy mean / full-cohort SD")
    axes[0].set_title("A  Measure balance; do not assume it")
    axes[0].legend(fontsize=8, frameon=False, loc="upper left")
    axes[1].barh(
        range(7),
        [full["ablations"][f"{name}_only_lr"]["metrics"]["auc"] for name in properties],
        color=BLUE,
    )
    axes[1].set_yticks(range(7), properties)
    axes[1].set_xlim(0, 1.05)
    axes[1].axvline(0.5, linestyle=":", color="#777777")
    axes[1].set_xlabel("Single-descriptor OOF ROC-AUC")
    axes[1].set_title("B  Exact-scaffold logistic baselines")
    captions.append(
        (
            save_figure(fig, output, "property_controls"),
            (
                "Figure 2. Property-balance and fixed single-descriptor controls. A: signed "
                "differences in original full-cohort SD units, not pooled within-class "
                "standardized differences. Nearest matching uses the full cohort for exploratory "
                "subset design and does not guarantee balance. B: seed-42 exact-scaffold OOF "
                "logistic models, training-only scaling; all seven descriptors are shown."
            ),
        )
    )
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), layout="constrained")
    axes[0].plot([0, 1], [0, 1], linestyle=":", color="#888888")
    for method, color in [("Property_LR", BLUE), ("Equal_mean", ORANGE), ("RF", GREEN)]:
        bins = full["calibration"][method]["bins"]
        axes[0].plot(
            [r["mean_predicted"] for r in bins],
            [r["observed_positive_rate"] for r in bins],
            marker="o",
            color=color,
            label=method,
        )
    axes[0].set(
        xlim=(0, 1),
        ylim=(0, 1.05),
        xlabel="Mean OOF score within occupied bin",
        ylabel="Observed positive-label fraction",
        title="A  Score-bin source-label frequencies",
    )
    axes[0].legend(fontsize=8, frameon=False, loc="upper left")
    groups = full["similarity_generalization"]["all_training"]["Equal_mean"]
    names = [f"{r['similarity_interval'][0]:.2f}–{r['similarity_interval'][1]:.2f}" for r in groups]
    axes[1].bar(
        range(len(groups)), [r["positives"] for r in groups], label="Positive labels", color=ORANGE
    )
    axes[1].bar(
        range(len(groups)),
        [r["n"] - r["positives"] for r in groups],
        bottom=[r["positives"] for r in groups],
        label="Untested decoys",
        color=BLUE,
    )
    axes[1].set_xticks(range(len(groups)), names, rotation=25, ha="right")
    axes[1].set(
        xlabel="Maximum Tanimoto to outer training set",
        ylabel="Unique molecules",
        title="B  Similarity-bin class support",
    )
    axes[1].legend(fontsize=8, frameon=False)
    captions.append(
        (
            save_figure(fig, output, "reliability_domain"),
            (
                "Figure 3. Score-bin source-label frequencies and similarity-bin counts "
                "under exact-scaffold "
                "splitting. Five fixed equal-width score bins; unoccupied bins are omitted, not "
                "imputed. Curves describe source labels, not verified activity probabilities. B "
                "shows class support in fixed similarity intervals; one-class-bin AUC is "
                "undefined. Per-bin metrics and nearest-positive similarity are supplied in "
                "controls.json."
            ),
        )
    )
    plt.close(fig)

    measured = transfer["measured_source_challenge"]["threshold_sensitivity_uM"]["50.0"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), layout="constrained")
    for index, (values, label, color) in enumerate(
        [
            (full["methods"], "Repository scaffold OOF", BLUE),
            (measured["metrics"], "Measured-source challenge", ORANGE),
        ]
    ):
        if values is None:
            axes[0].text(
                0.98,
                0.83,
                f"Source ROC-AUC unavailable\nn={measured['n']}; "
                f"{measured['positives']} positive, "
                f"{measured['n'] - measured['positives']} negative",
                transform=axes[0].transAxes,
                ha="right",
                va="top",
                fontsize=8,
            )
            continue
        axes[0].bar(
            [i + (index - 0.5) * 0.35 for i in range(6)],
            [values[m]["auc"] for m in METHODS],
            width=0.35,
            label=label,
            color=color,
        )
    axes[0].set_xticks(range(6), LABELS, rotation=35, ha="right")
    axes[0].set(ylim=(0, 1.25), ylabel="ROC-AUC", title="A  Change the evidence source")
    axes[0].axhline(0.5, linestyle=":", color="#777777")
    axes[0].legend(fontsize=8, frameon=False, loc="upper left")
    pair = transfer["probe_challenge"]["MRK_pair_contrast"]
    for index, (key, label, color) in enumerate(
        [("MRK952", "MRK-952", BLUE), ("MRK952_NC", "MRK-952-NC", ORANGE)]
    ):
        axes[1].bar(
            [i + (index - 0.5) * 0.35 for i in range(6)],
            [pair[m][key] for m in METHODS],
            width=0.35,
            label=label,
            color=color,
        )
    axes[1].set_xticks(range(6), LABELS, rotation=35, ha="right")
    axes[1].set(
        ylim=(0, 1.05),
        ylabel="Frozen-model score (not potency)",
        title="B  One known strong/weak probe pair",
    )
    axes[1].legend(fontsize=8, frameon=False)
    captions.append(
        (
            save_figure(fig, output, "source_transfer"),
            (
                "Figure 4. Retrospective source-transfer diagnostics of the frozen implementation. "
                "No tuning in this regeneration; earlier outcome exposure is not excluded. A: "
                f"Repository OOF discrimination (n={full['methods']['Property_LR']['n']}) versus "
                f"the measured-source challenge after eligibility/identity exclusions "
                f"(n={measured['n']}; {measured['positives']} threshold-positive and "
                f"{measured['n'] - measured['positives']} threshold-negative at IC50 <50 uM). "
                f"{source_cutoff_caption(transfer['measured_source_challenge'])} "
                "Source ROC-AUC is unavailable unless both classes are present. "
                "Different sample sizes, labels and fitting regimes preclude a controlled "
                "performance-drop estimate. B: scores for one externally sourced, already-known "
                "strong/weak pair; neither is a new inhibitor and NC is not inactive. No "
                "prospective validation, potency calibration or statistical generalization is "
                "established."
            ),
        )
    )
    plt.close(fig)
    return captions


def write_tables(results: Path, output: Path) -> Path:
    controls = json.loads((results / "controls.json").read_text())
    transfer = json.loads((results / "transfer.json").read_text())
    rows: list[dict[str, Any]] = []
    cutoffs: list[dict[str, Any]] = []

    def add_source_rows(prefix: str, challenge: dict[str, Any]) -> None:
        for cutoff, evaluation in challenge["threshold_sensitivity_uM"].items():
            design = f"{prefix}_below_{cutoff}_uM"
            metrics = evaluation["metrics"]
            cutoffs.append(
                {
                    "design": design,
                    "n": evaluation["n"],
                    "positives": evaluation["positives"],
                    "negatives": evaluation["n"] - evaluation["positives"],
                    "status": "available"
                    if metrics is not None
                    else "infeasible: both classes required",
                }
            )
            for method, values in (metrics or {}).items():
                rows.append({"design": design, "method": method, **values})

    for design, evaluation in controls["evaluations"].items():
        for method, metrics in evaluation.get("methods", {}).items():
            rows.append({"design": design, "method": method, **metrics})
    add_source_rows("measured_source", transfer["measured_source_challenge"])
    for split, evaluation in transfer["reference_sensitivity"]["evaluations"].items():
        for method, metrics in evaluation["metrics"].items():
            rows.append({"design": f"authenticated_reference_{split}", "method": method, **metrics})
    add_source_rows(
        "authenticated_reference_source",
        transfer["reference_sensitivity"]["measured_source_challenge"],
    )
    for design, evaluation in controls["evaluations"].items():
        for method, result in evaluation.get("ablations", {}).items():
            rows.append({"design": f"ablation_{design}", "method": method, **result["metrics"]})
    with (output / "all_metrics.csv").open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    with (output / "source_cutoff_status.csv").open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(cutoffs[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(cutoffs)
    candidates = [
        {
            "id": row["id"],
            "canonical_smiles": row["canonical_smiles"],
            "inchikey": row["inchikey"],
            "equal_mean_score_rank": row["equal_mean_score_rank"],
            "training_dissimilarity_rank": row["training_dissimilarity_rank"],
            "experimental_role": row["experimental_role"],
            "max_training_tanimoto": row["max_training_tanimoto"],
            "max_active_tanimoto": row["max_active_tanimoto"],
            **row["scores"],
        }
        for row in transfer["candidates"]
    ]
    with (output / "candidate_axes.csv").open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(candidates[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(candidates)
    lines = [
        "# Automatically generated supplementary results",
        "",
        (
            "These are exploratory diagnostics. AP and trapezoidal PR-AUC are distinct. Threshold "
            "0.5 is arbitrary. No measured activity probabilities or prospective error rates are "
            "claimed. MCC follows sklearn's zero convention when its denominator is zero; "
            "this is not an estimated correlation in a constant-prediction row."
        ),
        "",
        "| Design | Method | n | ROC-AUC | AP | Brier | MCC |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['design']} | {row['method']} | {row['n']} | {row['auc']:.4f} | "
            f"{row['average_precision']:.4f} | {row['brier']:.4f} | {row['mcc_at_0_5']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Source cutoff feasibility",
            "",
            "No score is imputed when a cutoff lacks both classes; undefined metrics are omitted "
            "from all_metrics.csv, not treated as zero.",
            "",
            "| Design | n | Positive | Negative | Status |",
            "|---|---:|---:|---:|---|",
        ]
    )
    for cutoff in cutoffs:
        lines.append(
            f"| {cutoff['design']} | {cutoff['n']} | {cutoff['positives']} | "
            f"{cutoff['negatives']} | {cutoff['status']} |"
        )
    lines.extend(
        [
            "",
            "## All fixed-seed runs",
            "",
            "| Design | Seed | Property LR AUC | Equal mean AUC |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in controls["seed_sensitivity"]:
        if row.get("methods"):
            lines.append(
                f"| {row['design']} | {row['seed']} | "
                f"{row['methods']['Property_LR']['auc']:.4f} | "
                f"{row['methods']['Equal_mean']['auc']:.4f} |"
            )
        else:
            lines.append(f"| {row['design']} | {row['seed']} | Infeasible | Infeasible |")
    lines.extend(
        [
            "",
            "## Sparse-calibration prediction-set diagnostics",
            "",
            (
                "No coverage guarantee under scaffold shift. Classwise coverage, class counts, "
                "p-values and all prediction sets are in controls.json."
            ),
            "",
            "| Cohort | 1 minus alpha (reference only) | Observed source-label inclusion "
            "| Mean set size | Singleton rate |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for design, evaluation in controls["evaluations"].items():
        for level in evaluation.get("conformal", {}).get("levels", []):
            lines.append(
                f"| {design} | {level['nominal_coverage']:.2f} | "
                f"{level['empirical_coverage']:.3f} | {level['mean_set_size']:.3f} | "
                f"{level['singleton_rate']:.3f} |"
            )
    diagnostics = diagnostic_reexpressions(controls)
    for design in controls["evaluations"]:
        result = diagnostics[design]
        lines += [
            "",
            f"### {design}: calibration sparsity",
            "",
            "| Outer fold | Class 0 n | Class 1 n | Class 0 p-min | Class 1 p-min |",
            "|---|---:|---:|---:|---:|",
        ]
        for row in result["conformal"]["calibration"]:
            lines.append(
                f"| {row['fold']} | {row['counts']['0']} | {row['counts']['1']} | "
                f"{row['minimum_pvalue']['0']:.4f} | {row['minimum_pvalue']['1']:.4f} |"
            )
        lines += [
            "",
            "| Alpha | Empty | Singleton | Both labels | Forced class 0 "
            "| Forced class 1 | Both forced |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
        for row in result["conformal"]["levels"]:
            counts, forced = row["set_size_counts"], row["forced_inclusion"]
            lines.append(
                f"| {row['alpha']} | {counts['0']} | {counts['1']} | {counts['2']} | "
                f"{forced['0']} | {forced['1']} | {row['both_forced']} |"
            )
        lines += [
            "",
            f"### {design}: AUC estimands",
            "",
            "| Method | Pooled | Within fold | Pooled pairs | Within-fold pairs |",
            "|---|---:|---:|---:|---:|",
        ]
        for name, row in result["methods"].items():
            pooled = "Undefined" if row["pooled_auc"] is None else f"{row['pooled_auc']:.4f}"
            within = (
                "Undefined" if row["within_fold_auc"] is None else f"{row['within_fold_auc']:.4f}"
            )
            lines.append(
                f"| {name} | {pooled} | {within} | "
                f"{row['pooled_pairs']} | {row['within_fold_pairs']} |"
            )
    path = output / "supplementary_results.md"
    path.write_text("\n".join(lines) + "\n")
    return path
