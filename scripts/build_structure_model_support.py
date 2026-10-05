"""Fixed offline model-support extraction; not independent density validation or refinement."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import importlib.metadata
import io
import math
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import gemmi
import numpy as np
from selectivity import publish
from structure_comparison import (
    ValidationError,
    category,
    checked_path,
    json_bytes,
    read_json,
    require,
)

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = "research/structure_comparison/model_support"
MANIFEST_SHA256 = "c9f6db8ef6fabfa5bfd2a6d412738d6ba1730757dea8ec7d2b77107700de844b"
NOTES_SHA256 = "770e37126c2c8f168329ba61aadc24e66ae78561355db90abf5b051b18eb91b3"
PDBS = ("8RIY", "8OTV")
SITES = {
    "8RIY": {("C", "AAA", "301"), ("D", "BBB", "301")},
    "8OTV": {("C", "A", "301"), ("F", "B", "302")},
}
ANCHORS = {"8RIY": {"28", "46", "47", "51"}, "8OTV": {"17", "34", "35", "107"}}
METRICS = ("rscc", "rsr", "rsrz", "avgoccu", "NatomsEDS")
ROW_FIELDS = (
    "site_id",
    "residue_identity",
    "ligand_conformer_id",
    "protein_conformer_id",
    "geometry_status",
    "observed_min_distance_A",
    "fractional_occupancy",
    "conditional_local_conformer",
    "retained_ligand_atom_count",
    "retained_protein_atom_count",
    "missing_or_excluded_atoms",
    "missing_ligand_heavy_atoms",
    "reasons",
    "within_3_5A",
    "within_4_0A",
    "within_4_5A",
    "within_5_0A",
)
LIMITS = [
    "Retrospective public-data assessment; no new measurement or independent validation.",
    "Report-derived RSCC/RSR are not recomputed here and are not per-atom support scores.",
    "PDBe maps are precomputed model-dependent maps, not omit maps or new experimental data.",
    "Map/report software dates need not describe the same map calculation or weighting.",
    "Fixed slices and trilinear atom sampling are limited direct map inspection, not a full "
    "three-dimensional crystallographic model validation or rerefinement.",
    "Interpolation does not increase map resolution. Sample values are not probabilities, "
    "RSCC, RSR, coordinate errors, interaction energies or occupancy estimates.",
    "No inferential test across crystal copies; sites and chains are not independent n.",
    "No proximity-to-affinity, energy, causality, homology or selectivity inference.",
    "B factors are retained metadata, never converted to coordinate uncertainties.",
    "Unobserved/zero-occupancy atoms are not imputed. Null is not no-contact or zero density.",
    "No atom-directed redesign, hydrogen-bond proof or refutation of source experiments.",
]


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def blank(value: str) -> str:
    return "" if value in ("", " ", ".", "?") else value


def numeric(raw: str, field: str) -> float:
    try:
        result = float(raw)
    except ValueError as exc:
        raise ValidationError(f"Non-numeric {field}: {raw}") from exc
    require(math.isfinite(result), f"Nonfinite {field}")
    return result


def metric_values(attributes: dict[str, str]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for field in METRICS:
        raw = attributes.get(field)
        if raw is None or blank(raw) == "":
            result[field] = {"value": None, "status": "not_reported", "raw": raw}
            continue
        value = numeric(raw, field)
        if field == "rscc":
            require(-1 <= value <= 1, "RSCC out of range")
        elif field in ("rsr", "avgoccu"):
            require(0 <= value <= 1, f"{field} out of range")
        elif field == "NatomsEDS":
            require(value >= 0 and value.is_integer(), "Invalid NatomsEDS")
        result[field] = {"value": value, "status": "reported", "raw": raw}
    return result


def report_key(attributes: dict[str, str]) -> tuple[str, ...]:
    return tuple(attributes[k] for k in ("model", "said", "seq", "resname")) + (
        blank(attributes["altcode"]),
    )


def parse_report(data: bytes, pdb: str) -> dict[str, Any]:
    require(b"<!DOCTYPE" not in data and b"<!ENTITY" not in data, "XML declarations refused")
    root = ET.fromstring(data)
    require(root.tag == "wwPDB-validation-information", "Not a wwPDB report")
    entries = root.findall("Entry")
    require(len(entries) == 1 and entries[0].get("pdbid") == pdb, "Wrong report entry")
    entry = dict(entries[0].attrib)
    require(bool(entry.get("XMLcreationDate")), "Missing report date")
    records = {}
    for node in root.findall("ModelledSubgroup"):
        attrs = dict(node.attrib)
        require(
            all(
                k in attrs
                for k in (
                    "model",
                    "said",
                    "seq",
                    "resname",
                    "altcode",
                    "chain",
                    "resnum",
                    "icode",
                    "ent",
                )
            ),
            "Incomplete report identity",
        )
        key = report_key(attrs)
        # Nonpolymer groups can share a label chain/seq marker; auth number disambiguates.
        key += (attrs["resnum"], blank(attrs["icode"]))
        require(key not in records, "Duplicate report subgroup")
        records[key] = {
            "attributes_raw": attrs,
            "metrics": metric_values(attrs),
            "outliers_raw": [{"type": x.tag, "attributes": dict(x.attrib)} for x in node],
        }
    require(bool(records), "Empty validation report")
    return {"entry": entry, "schema_raw": dict(root.attrib), "records": records}


def lookup_report(report: dict[str, Any], identity: dict[str, Any], alt: str) -> dict[str, Any]:
    key = (
        identity["model_id"],
        identity["label_asym_id"],
        identity["label_seq_id"],
        identity["label_comp_id"],
        blank(alt),
        identity["auth_seq_id"],
        blank(identity.get("atom_insertion_code_raw", identity["insertion_code_raw"])),
    )
    record = report["records"].get(key)
    if record is None:
        return {
            "status": "no_exact_report_record",
            "metrics": metric_values({}),
            "attributes_raw": None,
            "outliers_raw": None,
        }
    attrs = record["attributes_raw"]
    require(attrs["chain"] == identity["auth_asym_id"], "Report auth/label chain mismatch")
    require(attrs["ent"] == identity["label_entity_id"], "Report entity mismatch")
    return {"status": "exact_record", **record}


def identity_from_atom(row: dict[str, str], pdb: str) -> dict[str, str]:
    result = {
        k: row[k]
        for k in (
            "label_asym_id",
            "auth_asym_id",
            "label_seq_id",
            "auth_seq_id",
            "label_comp_id",
            "label_entity_id",
        )
    }
    result.update(
        pdb_id=pdb, model_id=row["pdbx_PDB_model_num"], insertion_code_raw=row["pdbx_PDB_ins_code"]
    )
    return result


def cell_and_group(block: gemmi.cif.Block) -> tuple[tuple[float, ...], int]:
    cell = category(block, "_cell.")[0]
    values = tuple(
        numeric(cell[x], x)
        for x in ("length_a", "length_b", "length_c", "angle_alpha", "angle_beta", "angle_gamma")
    )
    group = gemmi.find_spacegroup_by_name(category(block, "_symmetry.")[0]["space_group_name_H-M"])
    require(group is not None, "Unknown space group")
    assert group is not None
    return values, group.number


def check_cell(actual: Sequence[float], expected: Sequence[float]) -> None:
    require(len(actual) == len(expected) == 6, "Six unit-cell values required")
    require(
        all(
            math.isclose(a, b, rel_tol=0, abs_tol=0.002)
            for a, b in zip(actual, expected, strict=True)
        ),
        "Unit-cell mismatch",
    )


def inspect_sf(data: bytes, pdb: str, coordinate: gemmi.cif.Block) -> dict[str, Any]:
    doc = gemmi.cif.read_string(data.decode())
    require(len(doc) == 1, "Single structure-factor block required")
    block = doc.sole_block()
    require(block.find_value("_entry.id").upper() == pdb, "Structure-factor entry mismatch")
    cell, group = cell_and_group(block)
    expected_cell, expected_group = cell_and_group(coordinate)
    check_cell(cell, expected_cell)
    require(group == expected_group, "Structure-factor space-group mismatch")
    rows = category(block, "_refln.")
    require(bool(rows), "No reflection records")
    columns = set(rows[0])
    require({"index_h", "index_k", "index_l"} <= columns, "Missing Miller indices")
    keys = [(r["index_h"], r["index_k"], r["index_l"]) for r in rows]
    require(len(keys) == len(set(keys)), "Duplicate Miller indices in fixed source")
    inventory = {}
    for col in sorted(columns):
        present = [r[col] for r in rows if blank(r[col]) != ""]
        if col != "status":
            for v in present:
                numeric(v, col)
        inventory[col] = {"present": len(present), "missing": len(rows) - len(present)}
    pairs = {}
    for amplitude, phase in (("pdbx_FWT", "pdbx_PHWT"), ("pdbx_DELFWT", "pdbx_DELPHWT")):
        paired = sum(
            blank(r.get(amplitude, "")) != "" and blank(r.get(phase, "")) != "" for r in rows
        )
        pairs[amplitude + "/" + phase] = {
            "paired_rows": paired,
            "status": "available" if paired else "unavailable",
        }
    return {
        "block": block.name,
        "cell": cell,
        "spacegroup_number": group,
        "reflection_rows": len(rows),
        "columns": inventory,
        "coefficient_pairs": pairs,
        "interpretation": "Column inventory only; no coefficients recomputed, Fourier "
        "synthesis, reprocessing or rerefinement performed.",
    }


def load_map(path: Path, coordinate: gemmi.cif.Block) -> tuple[gemmi.FloatGrid, dict[str, Any]]:
    density = gemmi.read_ccp4_map(str(path))
    require(density.header_i32(4) == 2, "Float32 CCP4 map required")
    require(all(density.header_i32(i) == 0 for i in (5, 6, 7)), "Map start offset refused")
    require(all(density.header_float(i) == 0 for i in (50, 51, 52)), "Map origin refused")
    expected_cell, expected_group = cell_and_group(coordinate)
    check_cell(density.grid.unit_cell.parameters, expected_cell)
    require(
        density.grid.spacegroup is not None and density.grid.spacegroup.number == expected_group,
        "Map space-group mismatch",
    )
    require(
        tuple(density.grid.shape) == tuple(density.header_i32(i) for i in (8, 9, 10)),
        "Partial or reordered map refused by fixed inspection policy",
    )
    require([density.header_i32(i) for i in (17, 18, 19)] == [1, 2, 3], "Map axes refused")
    density.setup(float("nan"))
    array = np.asarray(density.grid.array, dtype=np.float64)
    require(bool(np.isfinite(array).all()), "Nonfinite/incomplete map")
    mean, sd = float(array.mean()), float(array.std())
    require(sd > 0, "Constant map")
    return density.grid, {
        "shape": list(density.grid.shape),
        "cell": list(density.grid.unit_cell.parameters),
        "spacegroup_number": expected_group,
        "axis_order": "XYZ",
        "origin": [0, 0, 0],
        "full_cell_mean": mean,
        "full_cell_population_sd": sd,
        "header_mean": density.header_float(22),
        "header_rms": density.header_float(55),
        "labels_raw": [density.header_str(57 + 20 * i, 80) for i in range(density.header_i32(56))],
        "generation_version": "not specified in the supplied CCP4 header",
        "method": "Gemmi 0.7.3 read_ccp4_map/setup; complete unit-cell map; no Fourier synthesis",
    }


def sample_atom(
    row: dict[str, str], maps: dict[str, tuple[gemmi.FloatGrid, dict[str, Any]]]
) -> dict[str, Any]:
    occupancy = numeric(row["occupancy"], "occupancy")
    require(0 <= occupancy <= 1, "Invalid occupancy")
    if occupancy == 0 or row["type_symbol"] in ("H", "D"):
        return {"status": "not_sampled_zero_occupancy_or_hydrogen", "maps": None}
    xyz = [numeric(row["Cartn_" + axis], "coordinate") for axis in "xyz"]
    result = {}
    for name, (grid, metadata) in maps.items():
        value = float(grid.interpolate_value(gemmi.Position(*xyz)))
        require(math.isfinite(value), "Nonfinite map sample")
        result[name] = {
            "interpolated_value": value,
            "standardized_value": (value - metadata["full_cell_mean"])
            / metadata["full_cell_population_sd"],
        }
    return {"status": "trilinear_positive_occupancy_heavy_atom", "maps": result}


def fixed_local(row: dict[str, Any]) -> bool:
    identity = row["residue_identity"]
    distance = row["observed_min_distance_A"]
    return (
        (distance is not None and distance <= 5.0)
        or identity["auth_seq_id"] in ANCHORS[identity["pdb_id"]]
        or (
            identity["pdb_id"] == "8RIY"
            and identity["auth_seq_id"] not in ("?", ".")
            and 45 <= int(identity["auth_seq_id"]) <= 55
        )
    )


def load_sources(repository: Path) -> tuple[dict[str, Any], dict[str, Path], dict[str, Any]]:
    require(repository.is_absolute(), "Absolute repository required")
    require(
        not any(x.is_symlink() for x in (repository, *repository.parents)),
        "Repository symlink refused",
    )
    manifest_path = checked_path(repository, PACKAGE + "/source_manifest.json")
    require(sha(manifest_path.read_bytes()) == MANIFEST_SHA256, "Fixed source manifest mismatch")
    manifest = read_json(manifest_path)
    sources = {}
    for item in manifest["baseline_inputs"]:
        path = checked_path(repository, item["path"])
        require(sha(path.read_bytes()) == item["sha256"], "Baseline input hash mismatch")
    for item in manifest["retrievals"]:
        if not item["redistributed"]:
            continue
        require(item.get("status") == 200, "Unsuccessful source retrieval")
        path = checked_path(repository, item["path"])
        raw = path.read_bytes()
        require(
            sha(raw) == item["stored_sha256"] and len(raw) == item["stored_bytes"],
            "Stored source hash/size mismatch",
        )
        downloaded = gzip.decompress(raw) if item["file"].endswith(".ccp4") else raw
        require(
            sha(downloaded) == item["sha256"] and len(downloaded) == item["bytes"],
            "Downloaded source hash/size mismatch",
        )
        require(item["file"] not in sources, "Duplicate source filename")
        sources[item["file"]] = path
    notes_path = checked_path(repository, PACKAGE + "/source_notes.json")
    require(sha(notes_path.read_bytes()) == NOTES_SHA256, "Source notes hash mismatch")
    notes = read_json(notes_path)
    article = ET.parse(repository / "research/structure_comparison/sources/PMC11089510.xml")
    for claim in notes["original_claims"]:
        xpath = claim["xpath"].replace("/article/", "./")
        if xpath.startswith("//"):
            xpath = "." + xpath
        nodes = article.findall(xpath)
        require(len(nodes) == 1, "Nonunique source-quote locator")
        text = "".join(nodes[0].itertext())
        require(text[claim["start"] : claim["end"]] == claim["quote"], "Source quotation mismatch")
    return manifest, sources, notes


def build(repository: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    require(importlib.metadata.version("gemmi") == "0.7.3", "Pinned Gemmi required")
    manifest, sources, notes = load_sources(repository)
    base = repository / "research/structure_comparison"
    geometry_path = base / "results/observed_proximity.json"
    geometry_sha256 = sha(geometry_path.read_bytes())
    require(
        geometry_sha256 in {x["sha256"] for x in manifest["baseline_inputs"]},
        "Geometry result is not a hashed baseline input",
    )
    geometry = read_json(geometry_path)
    structures = {}
    reports = {}
    atoms = {}
    maps = {}
    for pdb in PDBS:
        coordinate = gemmi.cif.read_file(str(base / "sources" / f"{pdb}.cif")).sole_block()
        require(coordinate.find_value("_entry.id") == pdb, "Coordinate entry mismatch")
        atoms[pdb] = category(coordinate, "_atom_site.")
        reports[pdb] = parse_report(
            gzip.decompress(sources[pdb.lower() + "_validation.xml.gz"].read_bytes()), pdb
        )
        maps[pdb] = {
            kind: load_map(sources[pdb.lower() + suffix + ".ccp4"], coordinate)
            for kind, suffix in (("EDS", ""), ("difference", "_diff"))
        }
        api = read_json(sources[pdb.lower() + "-files.json"])[pdb.lower()]
        advertised = {r["url"] for r in api["map"]["downloads"]}
        for suffix in ("", "_diff"):
            entry = next(
                r for r in manifest["retrievals"] if r["file"] == pdb.lower() + suffix + ".ccp4"
            )
            require(entry["url"] in advertised, "Map not advertised by archived PDBe API")
        structures[pdb] = {
            "report_entry_raw": reports[pdb]["entry"],
            "report_schema_raw": reports[pdb]["schema_raw"],
            "coordinate_revision_history": category(coordinate, "_pdbx_audit_revision_history."),
            "structure_factors": inspect_sf(
                gzip.decompress(sources[pdb.lower() + "-sf.cif.gz"].read_bytes()), pdb, coordinate
            ),
            "maps": {k: m for k, (_, m) in maps[pdb].items()},
        }
    site_records = []
    for index, site in enumerate(geometry["sites"]):
        identity = site["site_identity"]
        pdb = identity["pdb_id"]
        validation = lookup_report(reports[pdb], identity, ".")
        require(validation["status"] == "exact_record", "Missing required ligand report record")
        site_records.append(
            {
                "geometry_site_index": index,
                "site_identity": identity,
                "observed_geometry_status": site["observed_geometry_status"],
                "complete_site_geometry_status": site["complete_site_geometry_status"],
                "fractional_occupancy": site["fractional_occupancy"],
                "report": validation,
            }
        )
    for pdb in PDBS:
        actual = {
            (
                s["site_identity"]["label_asym_id"],
                s["site_identity"]["auth_asym_id"],
                s["site_identity"]["auth_seq_id"],
            )
            for s in site_records
            if s["site_identity"]["pdb_id"] == pdb
        }
        require(actual == SITES[pdb], "Incomplete four-site mapping")
        reported = {
            (
                r["attributes_raw"]["said"],
                r["attributes_raw"]["chain"],
                r["attributes_raw"]["resnum"],
            )
            for r in reports[pdb]["records"].values()
            if r["attributes_raw"]["resname"] == "W0O"
        }
        require(reported == SITES[pdb], "Report ligand inventory differs")
    residue_records = []
    local_reports: dict[str, dict[str, Any]] = {}
    local_members: dict[tuple[str, str, str], set[str]] = {}
    for index, row in enumerate(geometry["residue_proximity"]):
        identity = row["residue_identity"]
        validation = lookup_report(
            reports[identity["pdb_id"]], identity, row["protein_conformer_id"]
        )
        local = fixed_local(row)
        report_key_text = ":".join(
            (
                identity["pdb_id"],
                identity["label_asym_id"],
                identity["label_seq_id"],
                identity["auth_seq_id"],
                row["protein_conformer_id"],
            )
        )
        residue_records.append(
            {
                "geometry_row_index": index,
                **{k: row[k] for k in ROW_FIELDS},
                "minimum_witness_pairs": row["minimum_witness_pairs"] if local else None,
                "local_scope": local,
                "report_key": report_key_text,
                "report": {
                    "status": validation["status"],
                    "metrics": validation["metrics"],
                    "outlier_count": None
                    if validation["outliers_raw"] is None
                    else len(validation["outliers_raw"]),
                },
            }
        )
        if local:
            local_reports[report_key_text] = validation
            key = (identity["pdb_id"], identity["label_asym_id"], identity["label_seq_id"])
            local_members.setdefault(key, set()).add(row["site_id"])
    sampled_atoms = []
    for pdb in PDBS:
        for atom in atoms[pdb]:
            key = (pdb, atom["label_asym_id"], atom["label_seq_id"])
            if atom["label_comp_id"] != "W0O" and key not in local_members:
                continue
            identity = identity_from_atom(atom, pdb)
            sampled_atoms.append(
                {
                    "pdb_id": pdb,
                    "source_atom_row": atom,
                    "site_memberships": sorted(local_members.get(key, set())),
                    "sampling": sample_atom(atom, maps[pdb]),
                }
            )
    return {
        "schema_version": 1,
        "scope": "Fixed retrospective model-support assessment v1",
        "source_manifest_sha256": MANIFEST_SHA256,
        "source_notes_sha256": NOTES_SHA256,
        "baseline_revision": manifest["baseline_revision"],
        "geometry_result_sha256": geometry_sha256,
        "policy": {
            "selection": "All four W0O sites; all original residue-conformer rows preserved; "
            "atom sampling for residues with any retained distance <=5 A, both-chain "
            "published anchors (8RIY 28/46/47/51; 8OTV 17/34/35/107), and both-chain "
            "8RIY author residues 45–55. Alternates retained, not averaged.",
            "chronology": "Fixed retrospective policy after inspecting source reports; "
            "not preregistered.",
            "map_values": "Trilinear interpolation at positive-occupancy heavy-atom coordinates; "
            "standardized=(value-full-cell mean)/full-cell population SD. No weighting "
            "by occupancy; raw values retained. No significance or pass/fail threshold.",
            "slices": "Three Cartesian centroid slices (XY/XZ/YZ) per W0O site and per 8RIY Arg51 "
            "residue; 0.2 A plotting step, +0.7/+1.0 EDS contours, +/-3 difference contours. "
            "Only atoms within 0.75 A of a plane are marked. No bonds guessed.",
            "report_metrics": "RSCC, RSR, RSRZ, avgoccu and NatomsEDS parsed when present. Other "
            "raw fields (including EDIAm/OPIA) retained but not interpreted.",
            "comparison": "Target-specific numbering; label/auth identifiers checked separately.",
        },
        "structures": structures,
        "sites": site_records,
        "residues": residue_records,
        "local_residue_reports": dict(sorted(local_reports.items())),
        "local_atoms": sampled_atoms,
        "source_claims": notes["original_claims"],
        "original_geometry_refusals": geometry["refusals"],
        "original_geometry_missingness": geometry["missingness"],
        "limits": LIMITS,
    }, maps


def csv_bytes(rows: list[dict[str, Any]]) -> bytes:
    require(bool(rows), "Empty output table")
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode()


def tables(result: dict[str, Any]) -> dict[str, bytes]:
    sites = []
    for site in result["sites"]:
        ident = site["site_identity"]
        sites.append(
            {
                **ident,
                **{k: v["raw"] for k, v in site["report"]["metrics"].items()},
                "report_status": site["report"]["status"],
                "map_availability": "both_PDBe_maps_retrieved_and_sampled",
                "independent_density_validation": False,
                "rerefinement": False,
            }
        )
    residues = []
    for record in result["residues"]:
        row = record
        residues.append(
            {
                "site_id": row["site_id"],
                **row["residue_identity"],
                "ligand_alt": row["ligand_conformer_id"],
                "protein_alt": row["protein_conformer_id"],
                "geometry_status": row["geometry_status"],
                "observed_min_distance_A": row["observed_min_distance_A"],
                "local_scope": record["local_scope"],
                "report_status": record["report"]["status"],
                **{k: v["raw"] for k, v in record["report"]["metrics"].items()},
            }
        )
    atoms = []
    for record in result["local_atoms"]:
        sample = record["sampling"]
        atoms.append(
            {
                "pdb_id": record["pdb_id"],
                **record["source_atom_row"],
                "sampling_status": sample["status"],
                **{
                    f"{name}_{field}": sample["maps"][name][field] if sample["maps"] else None
                    for name in ("EDS", "difference")
                    for field in ("interpolated_value", "standardized_value")
                },
            }
        )
    return {
        "site_availability.csv": csv_bytes(sites),
        "residue_availability.csv": csv_bytes(residues),
        "local_atom_samples.csv": csv_bytes(atoms),
    }


def figures(result: dict[str, Any], maps: dict[str, Any]) -> dict[str, bytes]:
    import matplotlib

    matplotlib.use("Agg")
    with matplotlib.rc_context({**matplotlib.rcParamsDefault, "backend": "Agg"}):
        return slice_sheets(result, maps)


def slice_sheets(result: dict[str, Any], maps: dict[str, Any]) -> dict[str, bytes]:
    from matplotlib import pyplot as plt

    groups: dict[str, list[dict[str, Any]]] = {}
    for record in result["local_atoms"]:
        row, pdb = record["source_atom_row"], record["pdb_id"]
        if row["label_comp_id"] == "W0O" or (
            pdb == "8RIY" and row["label_comp_id"] == "ARG" and row["auth_seq_id"] == "51"
        ):
            name = f"{pdb}_{row['label_asym_id']}_{row['auth_asym_id']}_{row['auth_seq_id']}"
            groups.setdefault(name, []).append(record)
    output = {}
    for name, records in sorted(groups.items()):
        pdb = records[0]["pdb_id"]
        xyz = np.array(
            [[float(r["source_atom_row"]["Cartn_" + a]) for a in "xyz"] for r in records]
        )
        positive = np.array([float(r["source_atom_row"]["occupancy"]) > 0 for r in records])
        center = xyz[positive].mean(axis=0)
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        for axis, (u, v, w) in zip(axes, ((0, 1, 2), (0, 2, 1), (1, 2, 0)), strict=True):
            width = max(float(np.ptp(xyz[:, u])), float(np.ptp(xyz[:, v]))) / 2 + 3
            offsets = np.arange(-width, width + 0.1, 0.2)
            uu, vv = np.meshgrid(center[u] + offsets, center[v] + offsets)
            positions = np.tile(center, (uu.size, 1))
            positions[:, u], positions[:, v] = uu.ravel(), vv.ravel()
            for kind, levels, colors in (
                ("EDS", [0.7, 1.0], ["#86b8de", "#246495"]),
                ("difference", [-3.0, 3.0], ["#a33182", "#18823d"]),
            ):
                grid, metadata = maps[pdb][kind]
                values = grid.interpolate_position_array(positions, order=1).reshape(uu.shape)
                values = (values - metadata["full_cell_mean"]) / metadata["full_cell_population_sd"]
                for level, color in zip(levels, colors, strict=True):
                    if float(values.min()) <= level <= float(values.max()):
                        axis.contour(uu, vv, values, levels=[level], colors=[color], linewidths=0.8)
            for i, record in enumerate(records):
                if abs(xyz[i, w] - center[w]) > 0.75:
                    continue
                atom = record["source_atom_row"]
                axis.scatter(
                    xyz[i, u],
                    xyz[i, v],
                    c="black" if positive[i] else "red",
                    marker="o" if positive[i] else "x",
                    s=16,
                )
                axis.annotate(atom["label_atom_id"], (xyz[i, u], xyz[i, v]), fontsize=7)
            axis.set(
                xlabel="XYZ"[u] + " (A)",
                ylabel="XYZ"[v] + " (A)",
                title=f"{'XYZ'[w]} = {center[w]:.2f} A",
            )
            axis.set_aspect("equal")
        fig.suptitle(name + " | fixed centroid slices, not a full 3D validation", fontsize=12)
        fig.text(
            0.02,
            0.01,
            "Blue: EDS +0.7/+1 SD; purple/green: difference -3/+3 SD. "
            "Atoms marked only within 0.75 A of slice; red x: zero occupancy.\n"
            "Direct PDBe maps; interpolation adds no resolution. No inferred bonds or energies.",
            fontsize=9,
        )
        fig.tight_layout(rect=(0, 0.09, 1, 0.94))
        buffer = io.BytesIO()
        fig.savefig(buffer, format="png", dpi=130, metadata={"Software": "fixed model-support v1"})
        plt.close(fig)
        output[name + "_slices.png"] = buffer.getvalue()
    require(len(output) == 6, "Four ligand and two Arg51 slice sheets required")
    return output


def summary(result: dict[str, Any]) -> bytes:
    lines = [
        "# Public local-model support: extraction summary",
        "",
        "This is report extraction and limited direct map inspection, not independent density "
        "validation, rerefinement or new experimental validation.",
        "",
        "|Site (label / auth)|RSCC|RSR|Report|Maps|",
        "|---|---:|---:|---|---|",
    ]
    for site in result["sites"]:
        i = site["site_identity"]
        m = site["report"]["metrics"]
        lines.append(
            f"|{i['pdb_id']} {i['label_asym_id']} / {i['auth_asym_id']} {i['auth_seq_id']}|"
            f"{m['rscc']['raw']}|{m['rsr']['raw']}|exact identity match|EDS + difference|"
        )
    status = Counter(r["geometry_status"] for r in result["residues"])
    lines += [
        "",
        f"Original residue-conformer records retained: {dict(status)}.",
        "These are data records, not independent biological observations.",
        "",
        "## Limits",
        "",
    ]
    lines += ["- " + x for x in result["limits"]]
    return ("\n".join(lines) + "\n").encode()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, default=ROOT)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        require(
            not any(p.is_symlink() for p in (args.output, *args.output.parents)),
            "Output symlink refused",
        )
        require(not args.output.exists(), "Output must be a new directory")
        result, maps = build(args.repository)
        payloads = {
            "model_support.json": json_bytes(result),
            "summary.md": summary(result),
            **tables(result),
            **figures(result, maps),
        }
        completion = {
            "scope": result["scope"],
            "source_manifest_sha256": MANIFEST_SHA256,
            "source_notes_sha256": NOTES_SHA256,
            "script_sha256": sha(Path(__file__).read_bytes()),
            "versions": {
                p: importlib.metadata.version(p) for p in ("gemmi", "numpy", "matplotlib")
            },
            "outputs": {name: sha(data) for name, data in sorted(payloads.items())},
            "meaning": "Software extraction completed; not biological or independent "
            "density validation",
        }
        payloads["completion.json"] = json_bytes(completion)
        publish(args.output, payloads, "completion.json")
    except (ValueError, KeyError, OSError, RuntimeError, ET.ParseError) as exc:
        print(f"Refused: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
