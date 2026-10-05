"""Render recorded selectivity results without censoring substitution or model fitting."""

from __future__ import annotations

import argparse
import importlib
import io
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from selectivity import (
    SCENARIOS,
    json_bytes,
    publish,
    read_json,
    require,
    run_manifest,
    sha256,
    validate_result,
)

BLUE, ORANGE, GREY = "#0072B2", "#D55E00", "#666666"
FIGURE_NAMES = ("selectivity_historical", "selectivity_reference_sensitivity")


def render(result: dict[str, Any]) -> dict[str, bytes]:
    validate_result(result)
    mpl = importlib.import_module("matplotlib")
    mpl.use("Agg")
    plt = importlib.import_module("matplotlib.pyplot")
    payloads: dict[str, bytes] = {}
    rows = {r["source_compound"]: r for r in result["rows"]}
    settings = {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "svg.fonttype": "none",
        "svg.hashsalt": "nudt5-selectivity-1",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.compression": 6,
    }
    with mpl.rc_context(settings):
        for scenario, name in zip(SCENARIOS, FIGURE_NAMES, strict=True):
            scores = [
                r
                for r in result["score_rows"]
                if r["scenario"] == scenario and r["paired_diagnostic_eligible"]
            ]
            counts = result["summary"]["scenarios"][scenario]["ratio_status_counts"]
            point_n = counts.get("point", 0)
            bound_n = counts.get("upper_bound", 0) + counts.get("lower_bound", 0)
            unknown_n = len(scores) - point_n - bound_n
            fig, (axis, strip) = plt.subplots(
                1, 2, figsize=(10.5, 6.5), sharey=True, gridspec_kw={"width_ratios": [3, 1]}
            )
            try:
                fig.subplots_adjust(left=0.1, right=0.97, bottom=0.35, top=0.79, wspace=0.08)
                label = (
                    "Historical original graphs"
                    if scenario == SCENARIOS[0]
                    else "Stored authenticated-reference sensitivity"
                )
                fig.suptitle(
                    f"{label}\nReported paired IC50 ratios versus frozen Equal_mean",
                    y=0.97,
                    fontsize=14,
                )
                fig.text(
                    0.1,
                    0.84,
                    f"n={len(scores)} nonoverlap paired compounds: {point_n} point ratios, "
                    f"{bound_n} strict bound(s), {unknown_n} without a finite ratio bound",
                    fontsize=10,
                )
                finite = [
                    rows[r["source_compound"]]["ratio"][key]
                    for r in scores
                    for key in ("log10_point", "log10_bound")
                    if rows[r["source_compound"]]["ratio"][key] is not None
                ]
                left, right = min([0.0, *finite]) - 0.5, max([0.0, *finite]) + 0.5
                width = right - left
                axis.set(
                    xlim=(left, right),
                    ylim=(-0.04, 1.09),
                    xlabel="log10 R (dimensionless); R = IC50(NUDT14) / IC50(NUDT5)",
                    ylabel="Frozen Equal_mean (uncalibrated label score)",
                )
                axis.axvline(0, color=GREY, linestyle=":", linewidth=1)
                axis.grid(axis="y", alpha=0.2)
                axis.text(
                    0.99,
                    0.98,
                    ">0: lower NUDT5 reported mean",
                    ha="right",
                    va="top",
                    transform=axis.transAxes,
                    fontsize=9,
                    color=GREY,
                )
                strip.set(xlim=(0, 1), xticks=[], xlabel="No finite ratio bound")
                strip.spines["left"].set_linestyle(":")
                strip.spines["left"].set_color(GREY)
                strip.tick_params(axis="y", left=False)
                if not scores:
                    axis.text(0.5, 0.5, "No eligible pairs", ha="center", transform=axis.transAxes)
                    strip.text(0.5, 0.5, "None", ha="center", transform=strip.transAxes)
                elif not unknown_n:
                    strip.text(0.5, 0.5, "None", ha="center", transform=strip.transAxes, color=GREY)
                unknown = sorted(
                    [
                        s
                        for s in scores
                        if rows[s["source_compound"]]["ratio"]["status"] == "double_censored"
                    ],
                    key=lambda s: s["Equal_mean"],
                )
                label_positions: dict[str, float] = {}
                last = -0.08
                for item in unknown:
                    last = max(item["Equal_mean"], last + 0.13)
                    label_positions[item["source_compound"]] = last
                overflow = max([0.0, *[y - 0.95 for y in label_positions.values()]])
                label_positions = {k: y - overflow for k, y in label_positions.items()}
                for score in scores:
                    compound = score["source_compound"]
                    value = rows[compound]["ratio"]
                    y = score["Equal_mean"]
                    if value["status"] == "point":
                        x = value["log10_point"]
                        axis.scatter([x], [y], marker="o", color=BLUE, s=52, zorder=3)
                        axis.annotate(compound, (x, y), xytext=(7, 6), textcoords="offset points")
                    elif value["status"] in ("upper_bound", "lower_bound"):
                        x = value["log10_bound"]
                        upper = value["status"] == "upper_bound"
                        axis.scatter(
                            [x],
                            [y],
                            marker="<" if upper else ">",
                            s=72,
                            edgecolors=ORANGE,
                            facecolors="none",
                            linewidths=1.5,
                            zorder=3,
                        )
                        axis.annotate(
                            "",
                            (x + (-1 if upper else 1) * width * 0.12, y),
                            (x, y),
                            arrowprops={"arrowstyle": "->", "color": ORANGE, "lw": 1.4},
                        )
                        axis.annotate(
                            f"{compound}: {value['comparator']}{value['log10_bound']:.4g}",
                            (x, y),
                            xytext=(0, -18),
                            textcoords="offset points",
                            color=ORANGE,
                        )
                    else:
                        strip.scatter([0.22], [y], marker="x", color=GREY, s=55)
                        strip.annotate(
                            f"{compound}: both >50 µM",
                            (0.22, y),
                            xytext=(0.33, label_positions[compound]),
                            textcoords="data",
                            fontsize=9,
                            arrowprops={"arrowstyle": "-", "color": GREY, "lw": 0.7},
                        )
                fig.text(
                    0.1,
                    0.18,
                    "Filled circles: point ratios of reported means. Open triangles/arrows: strict "
                    "censoring-derived bounds, not CIs.\n"
                    "Crosses: both targets >50 µM; any positive R "
                    "remains possible. Untested and overlapping rows remain in the complete table.",
                    fontsize=9,
                    linespacing=1.5,
                )
                fig.text(
                    0.1,
                    0.09,
                    "Source mean ± SD retained in tables; ratio uncertainty is not estimated "
                    "without "
                    "raw paired replicates.\nReaction times differ (20/60 min); "
                    "control normalization "
                    "and replication ambiguities remain unresolved.\nSingle previously inspected "
                    "publication, related chemistry; not fresh external or prospective validation.",
                    fontsize=9,
                    linespacing=1.5,
                )
                if result["warning"].startswith("SOFTWARE TESTS ONLY"):
                    fig.text(0.1, 0.025, result["warning"], fontsize=8, color=ORANGE)
                for extension in ("png", "pdf", "svg"):
                    buffer = io.BytesIO()
                    metadata: dict[str, Any] = (
                        {"CreationDate": None, "ModDate": None}
                        if extension == "pdf"
                        else {"Date": None}
                        if extension == "svg"
                        else {"Software": "selectivity"}
                    )
                    fig.savefig(buffer, format=extension, dpi=200, metadata=metadata)
                    data = buffer.getvalue()
                    if extension == "svg":
                        data = (
                            "\n".join(s.rstrip() for s in data.decode().splitlines()) + "\n"
                        ).encode()
                    payloads[f"{name}.{extension}"] = data
            finally:
                plt.close(fig)
    return payloads


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Recorded selectivity.json")
    parser.add_argument(
        "--manifest", type=Path, required=True, help="Matching selectivity-manifest.json"
    )
    parser.add_argument("--output", type=Path, required=True, help="New or empty directory")
    args = parser.parse_args(argv)
    try:
        result = read_json(args.input)
        manifest = read_json(args.manifest)
        entry = manifest["artifacts"]["selectivity.json"]
        require(
            sha256(args.input) == entry["sha256"]
            and args.input.stat().st_size == entry["size_bytes"],
            "Recorded result hash mismatch",
        )
        payloads = render(result)
        figure_manifest = run_manifest(
            [
                args.input,
                args.manifest,
                Path(__file__),
                Path(__file__).parent / "scripts/selectivity.py",
            ],
            vars(args),
            payloads,
        )
        payloads["selectivity_figures_manifest.json"] = json_bytes(figure_manifest)
        publish(args.output, payloads, "selectivity_figures_manifest.json")
    except (ValueError, OSError, KeyError, TypeError, OverflowError) as exc:
        print(f"Selectivity figures refused: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
