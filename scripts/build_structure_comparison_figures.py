"""Render accessible tables and an observed-distance map from a recorded structure result."""

from __future__ import annotations

import argparse
import csv
import importlib
import io
import sys
import textwrap
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from selectivity import publish
from structure_comparison import RADII, THRESHOLDS, WARNING, digest, json_bytes, read_json, require

FIGURE = "observed_proximity_map"
TOP_LEVEL = [
    "schema_version",
    "contract_id",
    "origin",
    "warning",
    "provenance",
    "structures",
    "sites",
    "residue_proximity",
    "atom_pairs_within_5A",
    "excluded_atoms",
    "nonprotein_inventory",
    "missingness",
    "refusals",
    "limitations",
]
CAVEAT = (
    "Retained deposited heavy atoms only; minimum distance is descriptive geometry, not binding "
    "energy, affinity, hydrogen-bond proof, selectivity or causality. Missing/zero-occupancy atoms "
    "are not imputed, so values cannot exclude closer absent atoms. Coordinate uncertainty is not "
    "propagated. The four crystal sites are not independent samples; NUDT5 and NUDT14 residue "
    "axes are target-specific and are not homology-aligned."
)


def validate(result: Any) -> dict[str, Any]:
    require(isinstance(result, dict) and set(TOP_LEVEL) <= set(result), "Invalid result schema")
    require(result["schema_version"] == 1 and result["warning"] == WARNING, "Unsupported result")
    require(result["contract_id"] == "W0O-8RIY-8OTV-observed-proximity-v1", "Wrong contract")
    require(result["radii_A"] == list(RADII), "Unexpected radii")
    sites = {s["site_identity"]["site_id"] for s in result["sites"]}
    require(sites and {r["site_id"] for r in result["residue_proximity"]} == sites, "Site rows")
    for row in result["residue_proximity"]:
        distance = row["observed_min_distance_A"]
        flags = [row[field] for field in THRESHOLDS]
        if distance is None:
            require(all(flag is None for flag in flags) and row["reasons"], "Null row needs reason")
        else:
            require(flags == [distance <= r for r in RADII], "Threshold inconsistent with distance")
    return dict(result)


def residue_label(row: dict[str, Any]) -> str:
    i = row["residue_identity"]
    alt = "" if row["protein_conformer_id"] == "." else f" alt {row['protein_conformer_id']}"
    insertion = "" if i["insertion_code_raw"] in (".", "?") else i["insertion_code_raw"]
    return (
        f"chain {i['label_asym_id']}/{i['auth_asym_id']} {i['label_comp_id'].title()}"
        f"{i['auth_seq_id']}{insertion} (label {i['label_seq_id']}){alt}"
    )


def atom_label(a: dict[str, Any]) -> str:
    residue = f"{a['label_asym_id']}:{a['label_comp_id']}{a['auth_seq_id']}"
    return f"{residue}:{a['label_atom_id']}:{a['label_alt_id']}#{a['atom_site_id']}"


def csv_bytes(header: list[str], rows: list[list[Any]]) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(header)
    writer.writerows(["" if value is None else value for value in row] for row in rows)
    return buffer.getvalue().encode()


def text(value: Any) -> Any:
    return (
        repr(value)
        if isinstance(value, float)
        else str(value).lower()
        if isinstance(value, bool)
        else value
    )


def tables(result: dict[str, Any]) -> dict[str, bytes]:
    target = {s["pdb_id"]: s["target"] for s in result["structures"]}
    residue_header = [
        "site_id",
        "target",
        "pdb_id",
        "model_id",
        "ligand_conformer_id",
        "label_asym_id",
        "auth_asym_id",
        "label_seq_id",
        "auth_seq_id",
        "insertion_code_raw",
        "label_comp_id",
        "protein_conformer_id",
        "observed_min_distance_A",
        *THRESHOLDS,
        "geometry_status",
        "reasons",
        "fractional_occupancy",
        "conditional_local_conformer",
        "declared_missing_atoms",
        "excluded_atoms",
        "minimum_witness_pairs",
        "complete_residue_distance_A",
        "complete_residue_distance_status",
    ]
    residues = []
    for r in result["residue_proximity"]:
        i, missing = r["residue_identity"], r["missing_or_excluded_atoms"]
        residues.append(
            [
                r["site_id"],
                target[i["pdb_id"]],
                i["pdb_id"],
                i["model_id"],
                r["ligand_conformer_id"],
                i["label_asym_id"],
                i["auth_asym_id"],
                i["label_seq_id"],
                i["auth_seq_id"],
                i["insertion_code_raw"],
                i["label_comp_id"],
                r["protein_conformer_id"],
                text(r["observed_min_distance_A"]),
                *[text(r[f]) for f in THRESHOLDS],
                r["geometry_status"],
                ";".join(r["reasons"]),
                text(r["fractional_occupancy"]),
                text(r["conditional_local_conformer"]),
                ";".join(
                    f"{m['label_atom_id']}:{m['label_alt_id']}:flag{m['occupancy_flag']}"
                    for m in missing["declared_missing_atoms"]
                )
                or ("whole_residue" if missing["declared_missing_residues"] else ""),
                ";".join(atom_label(a) for a in missing["excluded_atoms"]),
                ";".join(
                    f"{atom_label(p['ligand_atom'])}|{atom_label(p['protein_atom'])}"
                    for p in r["minimum_witness_pairs"]
                ),
                None,
                r["complete_residue_distance_status"],
            ]
        )
    pairs = [
        [
            p["site_id"],
            p["ligand_conformer_id"],
            p["protein_conformer_id"],
            atom_label(p["ligand_atom"]),
            atom_label(p["protein_atom"]),
            p["protein_atom"]["auth_asym_id"],
            p["protein_atom"]["auth_seq_id"],
            text(p["distance_A"]),
            text(p["ligand_atom"]["occupancy"]),
            text(p["protein_atom"]["occupancy"]),
            text(p["fractional_occupancy"]),
        ]
        for p in result["atom_pairs_within_5A"]
    ]
    sensitivity = []
    for site in result["sites"]:
        name = site["site_identity"]["site_id"]
        rows = [r for r in result["residue_proximity"] if r["site_id"] == name]
        for field, radius in zip(THRESHOLDS, RADII, strict=True):
            hit = [r for r in rows if r[field] is True]
            shared = [r for r in hit if r["protein_conformer_id"] == "."]
            alternate = [r for r in hit if r["protein_conformer_id"] != "."]
            sensitivity.append(
                [
                    name,
                    site["site_identity"]["target"],
                    radius,
                    len(shared),
                    ";".join(residue_label(r) for r in shared),
                    ";".join(residue_label(r) for r in alternate),
                    sum(r["geometry_status"] == "partial_observed" for r in hit),
                    sum(r["fractional_occupancy"] for r in hit),
                    sum(r["observed_min_distance_A"] is None for r in rows),
                    "No pooled count across alternate conformers; "
                    "false/absent means no retained pair.",
                ]
            )
    return {
        "residue_proximity.csv": csv_bytes(residue_header, residues),
        "atom_pairs_within_5A.csv": csv_bytes(
            [
                "site_id",
                "ligand_conformer_id",
                "protein_conformer_id",
                "ligand_atom",
                "protein_atom",
                "protein_auth_asym_id",
                "protein_auth_seq_id",
                "distance_A",
                "ligand_occupancy",
                "protein_occupancy",
                "fractional_occupancy",
            ],
            pairs,
        ),
        "radius_sensitivity.csv": csv_bytes(
            [
                "site_id",
                "target",
                "radius_A",
                "shared_conformer_rows_within",
                "shared_conformer_residues",
                "alternate_conformer_rows_within_listed_separately",
                "partial_observed_rows_within",
                "fractional_occupancy_rows_within",
                "refused_or_unobserved_rows_at_site",
                "interpretation_limit",
            ],
            sensitivity,
        ),
    }


def notes(result: dict[str, Any]) -> list[str]:
    lines = []
    for site in sorted(result["sites"], key=lambda s: s["site_identity"]["site_id"]):
        name = site["site_identity"]["site_id"]
        rows = [r for r in result["residue_proximity"] if r["site_id"] == name]
        null = sum(r["observed_min_distance_A"] is None for r in rows)
        hidden = [
            f"{residue_label(r)} {r['observed_min_distance_A']:.2f} Å"
            for r in rows
            if r["protein_conformer_id"] != "." and r["within_5_0A"] is False
        ]
        alternates = f"; alternates >5.0 Å: {', '.join(hidden)}" if hidden else ""
        lines.append(
            f"{name}: {len(rows)} residue-conformer rows, {null} null/refused "
            f"(not plotted; not no-contact){alternates}"
        )
    return lines


def render(result: dict[str, Any]) -> dict[str, bytes]:
    mpl = importlib.import_module("matplotlib")
    mpl.use("Agg")
    plt = importlib.import_module("matplotlib.pyplot")
    colors = importlib.import_module("matplotlib.colors")
    payloads: dict[str, bytes] = {}
    settings = {
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "svg.fonttype": "none",
        "svg.hashsalt": "nudt5-structure-1",
        "pdf.compression": 6,
    }
    panels = []
    for structure in sorted(result["structures"], key=lambda s: s["target"]):
        sites = sorted(
            s["site_identity"]["site_id"]
            for s in result["sites"]
            if s["site_identity"]["pdb_id"] == structure["pdb_id"]
        )
        rows = [r for r in result["residue_proximity"] if r["site_id"] in sites]
        keys = sorted(
            {
                (
                    r["residue_identity"]["label_asym_id"],
                    int(r["residue_identity"]["label_seq_id"]),
                    r["protein_conformer_id"],
                )
                for r in rows
                if r["within_5_0A"] is True
            }
        )
        lookup = {
            (
                r["site_id"],
                r["residue_identity"]["label_asym_id"],
                int(r["residue_identity"]["label_seq_id"]),
                r["protein_conformer_id"],
            ): r
            for r in rows
        }
        panels.append((structure, sites, keys, lookup))
    with mpl.rc_context(settings):
        heights = [len(p[2]) + 3 for p in panels]
        fig, axes = plt.subplots(
            1, len(panels), figsize=(14, 0.32 * max(heights) + 4.2), squeeze=False
        )
        try:
            fig.subplots_adjust(left=0.2, right=0.9, top=0.86, bottom=0.17, wspace=0.95)
            cmap = plt.get_cmap("viridis").copy()
            norm = colors.Normalize(vmin=2.5, vmax=5.0)
            image = None
            for axis, (structure, sites, keys, lookup) in zip(axes[0], panels, strict=True):
                matrix = np.full((len(keys), len(sites)), np.nan)
                for y, key in enumerate(keys):
                    for x, site in enumerate(sites):
                        row = lookup[(site, *key)]
                        value = row["observed_min_distance_A"]
                        label = "refused" if value is None else f"{value:.2f}"
                        if value is not None:
                            matrix[y, x] = value
                            label += "".join(
                                mark
                                for mark, flag in (
                                    ("†", row["geometry_status"] == "partial_observed"),
                                    ("*", row["fractional_occupancy"]),
                                )
                                if flag
                            )
                        shade = value is not None and value < 3.7
                        axis.text(
                            x,
                            y,
                            label,
                            ha="center",
                            va="center",
                            fontsize=8,
                            color="white" if shade else "black",
                        )
                image = axis.imshow(
                    np.where(matrix <= 5.0, matrix, np.nan), cmap=cmap, norm=norm, aspect="auto"
                )
                for y, x in zip(*np.where(~(matrix <= 5.0)), strict=True):
                    axis.add_patch(
                        plt.Rectangle(
                            (x - 0.5, y - 0.5), 1, 1, facecolor="#dddddd", edgecolor="white"
                        )
                    )
                axis.set_xticks(
                    range(len(sites)), [s.replace(":model1", "\nmodel 1") for s in sites]
                )
                axis.set_yticks(
                    range(len(keys)), [residue_label(lookup[(sites[0], *k)]) for k in keys]
                )
                axis.set_title(
                    f"{structure['target']} (PDB {structure['pdb_id']}, assembly 1)\n"
                    "rows with any retained pair ≤5.0 Å",
                    fontsize=10,
                )
                axis.set_xlabel("Deposited W0O site (label asym:auth chain:auth seq)")
                axis.tick_params(length=0)
            colorbar = fig.colorbar(
                image, ax=axes[0].tolist(), fraction=0.025, pad=0.02, ticks=(2.5, 3.0, *RADII)
            )
            colorbar.set_label("Minimum observed heavy-atom distance (Å)")
            for radius in RADII:
                colorbar.ax.axhline(radius, color="black", linewidth=0.8)
            fig.suptitle(
                "W0O/compound 9: per-site observed ligand–residue minimum distances",
                fontsize=13,
                y=0.97,
            )
            fig.text(
                0.03,
                0.915,
                (
                    "Inclusive radii 3.5, 4.0 (primary), 4.5, 5.0 Å. Grey cells: >5.0 Å. "
                    "† partial residue (declared missing or zero-occupancy atom); "
                    "* positive fractional occupancy, unweighted; alt = separate local conformer."
                ),
                fontsize=8.5,
                wrap=True,
            )
            fig.text(
                0.03,
                0.015,
                "\n".join([*notes(result), *textwrap.wrap(CAVEAT, 175)]),
                fontsize=8.5,
                va="bottom",
            )
            for extension in ("png", "pdf", "svg"):
                buffer = io.BytesIO()
                metadata: dict[str, Any] = (
                    {"CreationDate": None, "ModDate": None}
                    if extension == "pdf"
                    else {"Date": None}
                    if extension == "svg"
                    else {"Software": "structure_comparison"}
                )
                fig.savefig(buffer, format=extension, dpi=200, metadata=metadata)
                data = buffer.getvalue()
                if extension == "svg":
                    data = (
                        "\n".join(s.rstrip() for s in data.decode().splitlines()) + "\n"
                    ).encode()
                payloads[f"{FIGURE}.{extension}"] = data
        finally:
            plt.close(fig)
    return payloads


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path, required=True, help="Recorded observed_proximity.json"
    )
    parser.add_argument("--input-sha256", required=True, help="Expected recorded result hash")
    parser.add_argument("--output", type=Path, required=True, help="New or empty directory")
    args = parser.parse_args(argv)
    try:
        data = args.input.read_bytes()
        require(digest(data) == args.input_sha256, "Recorded result hash mismatch")
        result = validate(read_json(args.input))
        payloads = {**tables(result), **render(result)}
        manifest = {
            "warning": WARNING,
            "input": str(args.input.resolve()),
            "input_sha256": digest(data),
            "code_sha256": digest(Path(__file__).read_bytes()),
            "artifacts": {
                name: {"sha256": digest(value), "size_bytes": len(value)}
                for name, value in sorted(payloads.items())
            },
        }
        payloads["derived_manifest.json"] = json_bytes(manifest)
        publish(args.output, payloads, "derived_manifest.json")
    except (ValueError, OSError, KeyError, TypeError) as exc:
        print(f"Structure figures refused: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
