"""Fixed-contract, observed-only W0O proximity; no implicit mmCIF disorder selection."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import re
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import gemmi
import numpy as np

RADII = (3.5, 4.0, 4.5, 5.0)
THRESHOLDS = tuple(f"within_{r:.1f}A".replace(".", "_") for r in RADII)
CONTRACT_SHA256 = "a22694d68a98876d577499a430927da6f30a92a9bd3f5505ba4e111d18fdf57b"
WARNING = (
    "Retrospective deposited-coordinate proximity only. Missing coordinates are not no-contact "
    "evidence. No inferred interaction, affinity, energy, selectivity, causality or clinical "
    "claim; crystal copies are not independent samples. No density inspection or coordinate "
    "uncertainty "
    "propagation. NUDT5 Arg51 and NUDT14 Leu107 are not asserted homologous."
)
Row = dict[str, str]


class ValidationError(ValueError):
    """Invalid input or unsupported fixed contract; no output is published."""


def require(condition: object, message: str) -> None:
    if not condition:
        raise ValidationError(message)


def json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_json(path: Path) -> Any:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            require(key not in result, f"Duplicate JSON key: {key}")
            result[key] = value
        return result

    def constant(token: str) -> Any:
        raise ValidationError(f"Nonfinite JSON: {token}")

    return json.loads(path.read_bytes(), object_pairs_hook=pairs, parse_constant=constant)


def exact_keys(value: Any, keys: str, context: str) -> None:
    require(
        isinstance(value, dict) and set(value) == set(keys.split()), f"Invalid {context} schema"
    )


def checked_path(repository: Path, relative: str) -> Path:
    require(isinstance(relative, str), "Input path must be a string")
    path = Path(relative)
    require(not path.is_absolute() and path.parts and ".." not in path.parts, "Unsafe input path")
    result = repository
    for part in path.parts:
        result /= part
        require(not result.is_symlink(), f"Input symlink refused: {relative}")
    require(result.is_file(), f"Missing input: {relative}")
    require(result.resolve().is_relative_to(repository), "Input escaped repository")
    return result


def category(block: gemmi.cif.Block, name: str, *, raw: bool = False) -> list[Row]:
    columns = block.get_mmcif_category(name, raw=True)
    if not columns:
        return []
    rows = []
    for i in range(len(next(iter(columns.values())))):
        row = {k: v[i] for k, v in columns.items()}
        rows.append(row if raw else {k: decode(v) for k, v in row.items()})
    return rows


def decode(value: str) -> str:
    return value if value in (".", "?") else gemmi.cif.as_string(value)


def parse_cif(path: Path) -> gemmi.cif.Block:
    try:
        doc = gemmi.cif.read_file(str(path), check_level=2)
        doc.check_for_missing_values()
        doc.check_for_duplicates()
        require(len(doc) == 1, "Exactly one CIF data block required")
        return doc.sole_block()
    except (RuntimeError, ValueError) as exc:
        raise ValidationError(f"Invalid CIF {path.name}: {exc}") from exc


def number(value: str, field: str) -> float:
    try:
        result = float(value)
    except ValueError as exc:
        raise ValidationError(f"Invalid numeric {field}: {value}") from exc
    require(math.isfinite(result), f"Nonfinite {field}")
    return result


def positive_integer(value: str, field: str) -> int:
    require(bool(re.fullmatch(r"[1-9][0-9]*", value)), f"Invalid {field}: {value}")
    return int(value)


def normalized_insertion(value: str) -> str:
    return "." if value == "?" else value


@dataclass(frozen=True, order=True)
class AtomIdentity:
    pdb_id: str
    assembly_id: str
    operation_id: str
    model_id: str
    label_entity_id: str
    label_asym_id: str
    auth_asym_id: str
    label_seq_id: str
    auth_seq_id: str
    insertion_code_raw: str
    label_comp_id: str
    auth_comp_id: str
    label_atom_id: str
    auth_atom_id: str
    label_alt_id: str
    type_symbol: str
    atom_site_id: str


@dataclass(frozen=True)
class Atom:
    identity: AtomIdentity
    xyz: tuple[float, float, float]
    occupancy: float
    b_factor: float | None
    protein: bool
    raw: Row

    @property
    def heavy(self) -> bool:
        return self.identity.type_symbol.upper() not in ("H", "D")

    @property
    def eligible(self) -> bool:
        return self.heavy and self.occupancy > 0

    @property
    def fractional(self) -> bool:
        return 0 < self.occupancy < 1

    def record(self, *, original: bool = False) -> dict[str, Any]:
        result = {
            **asdict(self.identity),
            "xyz_A": self.xyz,
            "occupancy": self.occupancy,
            "B_iso_or_equiv": self.b_factor,
            "fractional_occupancy": self.fractional,
            "is_protein": self.protein,
        }
        return {**result, "original_row": self.raw} if original else result

    def exclusion_reasons(self) -> list[str]:
        return (
            (["nonprotein_receptor_exclusion"] if not self.protein else [])
            + (["hydrogen_or_deuterium"] if not self.heavy else [])
            + (["zero_occupancy"] if self.occupancy == 0 else [])
        )


@dataclass(frozen=True)
class Residue:
    identity: Row
    atoms: tuple[Atom, ...]
    missing_atoms: tuple[Row, ...]
    missing_residues: tuple[Row, ...]
    scheme: Row


@dataclass(frozen=True)
class Conformer:
    label: str
    atoms: tuple[Atom, ...]
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class Structure:
    pdb_id: str
    target: str
    block: gemmi.cif.Block
    atoms: tuple[Atom, ...]
    residues: tuple[Residue, ...]
    model_ids: tuple[str, ...]
    protein_chains: tuple[str, ...]
    assembly_supported: bool
    missing_atoms: tuple[Row, ...]
    missing_residues: tuple[Row, ...]


def conformers(atoms: Sequence[Atom]) -> tuple[Conformer, ...]:
    labels = {a.identity.label_alt_id for a in atoms}
    alternatives = sorted(labels - {"."})
    result = []
    for label in alternatives or ["."]:
        local = tuple(
            sorted(
                (a for a in atoms if a.identity.label_alt_id in (".", label)),
                key=lambda a: a.identity,
            )
        )
        reasons = []
        if "?" in labels:
            reasons.append("unknown_altloc")
        names = [a.identity.label_atom_id for a in local]
        if len(set(names)) != len(names):
            reasons.append("ambiguous_shared_and_alternate_atom")
        result.append(Conformer(label, local, tuple(reasons)))
    return tuple(result)


def residue_key(row: Mapping[str, str], *, scheme: bool = False) -> tuple[str, str, str]:
    return (
        row["asym_id" if scheme else "label_asym_id"],
        row["seq_id" if scheme else "label_seq_id"],
        normalized_insertion(row["pdb_ins_code" if scheme else "PDB_ins_code"]),
    )


def assembly_supported(block: gemmi.cif.Block, chains: set[str]) -> bool:
    assemblies = category(block, "_pdbx_struct_assembly.")
    gens = category(block, "_pdbx_struct_assembly_gen.")
    ops = category(block, "_pdbx_struct_oper_list.")
    if len(assemblies) != 1 or assemblies[0]["id"] != "1" or len(gens) != 1 or len(ops) != 1:
        return False
    gen, op = gens[0], ops[0]
    return (
        gen["assembly_id"] == "1"
        and gen["oper_expression"] == op["id"] == "1"
        and set(gen["asym_id_list"].split(",")) == chains
        and all(
            number(op[f"matrix[{i}][{j}]"], "assembly matrix") == float(i == j)
            for i in (1, 2, 3)
            for j in (1, 2, 3)
        )
        and all(number(op[f"vector[{i}]"], "assembly vector") == 0.0 for i in (1, 2, 3))
    )


def parse_structure(path: Path, expected: Mapping[str, Any], ccd: Mapping[str, str]) -> Structure:
    block = parse_cif(path)
    entries = category(block, "_entry.")
    require(len(entries) == 1 and entries[0]["id"] == expected["pdb_id"], "Entry identity mismatch")
    entities = {r["id"]: r for r in category(block, "_entity.")}
    asym_rows = category(block, "_struct_asym.")
    asym = {r["id"]: r["entity_id"] for r in asym_rows}
    require(len(asym) == len(asym_rows), "Duplicate asym identity")
    polymers = {r["entity_id"]: r for r in category(block, "_entity_poly.")}
    protein_entities = {e for e, r in polymers.items() if r["type"] == "polypeptide(L)"}
    require(
        all(e in entities and entities[e]["type"] == "polymer" for e in polymers),
        "Entity/polymer mismatch",
    )
    protein_chains = {chain for chain, entity in asym.items() if entity in protein_entities}
    require(
        protein_chains == set(expected["protein_label_asym_ids"]) and len(protein_chains) == 2,
        "Both declared protein chains required",
    )
    atoms: list[Atom] = []
    ids: set[str] = set()
    identities: set[tuple[str, ...]] = set()
    fields = (
        "id type_symbol label_atom_id label_alt_id label_comp_id label_asym_id "
        "label_entity_id label_seq_id pdbx_PDB_ins_code Cartn_x Cartn_y Cartn_z occupancy "
        "B_iso_or_equiv auth_seq_id auth_comp_id auth_asym_id auth_atom_id pdbx_PDB_model_num"
    )
    for raw in category(block, "_atom_site.", raw=True):
        require(set(fields.split()) <= raw.keys(), "Missing atom_site fields")
        r = {k: decode(v) for k, v in raw.items()}
        require(r["id"] not in ids, "Duplicate atom_site.id")
        ids.add(r["id"])
        for field in (
            "id",
            "type_symbol",
            "label_atom_id",
            "label_comp_id",
            "label_asym_id",
            "label_entity_id",
            "auth_seq_id",
            "auth_comp_id",
            "auth_asym_id",
            "auth_atom_id",
        ):
            require(r[field] not in ("", ".", "?"), f"Missing atom identity {field}")
        require(gemmi.Element(r["type_symbol"]).atomic_number > 0, "Unknown element")
        model = r["pdbx_PDB_model_num"]
        positive_integer(model, "model")
        require(
            r["label_asym_id"] in asym and asym[r["label_asym_id"]] == r["label_entity_id"],
            "Atom entity/chain mismatch",
        )
        protein = r["label_entity_id"] in protein_entities
        if protein:
            positive_integer(r["label_seq_id"], "protein label residue")
        identity = AtomIdentity(
            expected["pdb_id"],
            "1",
            "1",
            model,
            r["label_entity_id"],
            r["label_asym_id"],
            r["auth_asym_id"],
            r["label_seq_id"],
            r["auth_seq_id"],
            r["pdbx_PDB_ins_code"],
            r["label_comp_id"],
            r["auth_comp_id"],
            r["label_atom_id"],
            r["auth_atom_id"],
            r["label_alt_id"],
            r["type_symbol"],
            r["id"],
        )
        unique = (
            model,
            r["label_asym_id"],
            r["label_seq_id"],
            r["auth_seq_id"],
            normalized_insertion(r["pdbx_PDB_ins_code"]),
            r["label_comp_id"],
            r["label_atom_id"],
            r["label_alt_id"],
        )
        require(unique not in identities, "Duplicate complete atom identity")
        identities.add(unique)
        occupancy = number(r["occupancy"], "occupancy")
        require(0 <= occupancy <= 1, "Occupancy outside [0,1]")
        b = None if r["B_iso_or_equiv"] in (".", "?") else number(r["B_iso_or_equiv"], "B factor")
        require(b is None or b >= 0, "Negative B factor")
        atom = Atom(
            identity,
            (number(r["Cartn_x"], "x"), number(r["Cartn_y"], "y"), number(r["Cartn_z"], "z")),
            occupancy,
            b,
            protein,
            raw,
        )
        if identity.label_comp_id == "W0O" and atom.heavy:
            require(
                ccd.get(identity.label_atom_id) == identity.type_symbol,
                "W0O atom name/element mismatch with CCD",
            )
            require(not protein, "W0O classified as protein")
        atoms.append(atom)
    require(atoms, "No atom_site rows")
    models = {a.identity.model_id for a in atoms}
    require(models == set(expected["expected_model_ids"]), "Unexpected/missing model")
    scheme = category(block, "_pdbx_poly_seq_scheme.")
    scheme_by_key = {
        residue_key(r, scheme=True): r for r in scheme if r["asym_id"] in protein_chains
    }
    require(len(scheme_by_key) == len(scheme), "Duplicate or nonprotein polymer scheme")
    sequence_rows = category(block, "_entity_poly_seq.")
    sequence = {(r["entity_id"], r["num"]): r["mon_id"] for r in sequence_rows}
    require(len(sequence) == len(sequence_rows), "Ambiguous polymer sequence")
    for r in scheme:
        require(asym.get(r["asym_id"]) == r["entity_id"], "Scheme entity/chain mismatch")
        require(sequence.get((r["entity_id"], r["seq_id"])) == r["mon_id"], "Sequence mismatch")
        positive_integer(r["seq_id"], "scheme residue")
    for chain in protein_chains:
        require(
            {r["seq_id"] for r in scheme if r["asym_id"] == chain}
            == {n for e, n in sequence if e == asym[chain]},
            "Incomplete polymer scheme",
        )
    grouped: dict[tuple[str, str, str, str], list[Atom]] = defaultdict(list)
    for a in atoms:
        if a.protein:
            i = a.identity
            key = (i.label_asym_id, i.label_seq_id, normalized_insertion(i.insertion_code_raw))
            require(key in scheme_by_key, "Protein atom absent from polymer scheme")
            r = scheme_by_key[key]
            require(
                i.label_comp_id == r["mon_id"]
                and i.auth_comp_id == r["mon_id"]
                and i.auth_asym_id == r["pdb_strand_id"]
                and i.auth_seq_id == r["auth_seq_num"],
                "Protein author/label identity mismatch",
            )
            grouped[(i.model_id, *key)].append(a)
    missing_atoms = category(block, "_pdbx_unobs_or_zero_occ_atoms.")
    missing_residues = category(block, "_pdbx_unobs_or_zero_occ_residues.")
    for is_atom, missing in ((True, missing_atoms), (False, missing_residues)):
        missing_ids: set[str] = set()
        for r in missing:
            require(r["id"] not in missing_ids, "Duplicate missingness id")
            missing_ids.add(r["id"])
            key = residue_key(r)
            require(
                key in scheme_by_key and r["PDB_model_num"] in models and r["polymer_flag"] == "Y",
                "Missingness cannot join residue/model",
            )
            s = scheme_by_key[key]
            require(
                r["label_comp_id"] == s["mon_id"] == r["auth_comp_id"]
                and r["auth_asym_id"] == s["pdb_strand_id"]
                and r["auth_seq_id"] == s["pdb_seq_num"],
                "Missingness identity mismatch",
            )
            matched = grouped[(r["PDB_model_num"], *key)]
            if is_atom:
                matched = [
                    a
                    for a in matched
                    if a.identity.label_atom_id == r["label_atom_id"]
                    and (r["label_alt_id"] == "?" or a.identity.label_alt_id == r["label_alt_id"])
                ]
                require(
                    all(a.identity.auth_atom_id == r["auth_atom_id"] for a in matched),
                    "Missingness atom author mismatch",
                )
            require(r["occupancy_flag"] in ("0", "1"), "Unknown missingness occupancy flag")
            require(
                (not matched)
                if r["occupancy_flag"] == "1"
                else (bool(matched) and all(a.occupancy == 0 for a in matched)),
                "Contradictory missingness occupancy",
            )
    residues: list[Residue] = []
    for model in sorted(models, key=int):
        for key, s in sorted(scheme_by_key.items(), key=lambda p: (p[0][0], int(p[0][1]), p[0][2])):
            local = tuple(sorted(grouped[(model, *key)], key=lambda a: a.identity))
            ma = tuple(
                sorted(
                    (
                        r
                        for r in missing_atoms
                        if r["PDB_model_num"] == model and residue_key(r) == key
                    ),
                    key=lambda r: tuple(sorted(r.items())),
                )
            )
            mr = tuple(
                sorted(
                    (
                        r
                        for r in missing_residues
                        if r["PDB_model_num"] == model and residue_key(r) == key
                    ),
                    key=lambda r: tuple(sorted(r.items())),
                )
            )
            require(local or mr, "Undeclared absent protein residue")
            identity_row = {
                "pdb_id": expected["pdb_id"],
                "target": expected["target"],
                "assembly_id": "1",
                "operation_id": "1",
                "model_id": model,
                "label_entity_id": s["entity_id"],
                "label_asym_id": key[0],
                "label_seq_id": key[1],
                "label_comp_id": s["mon_id"],
                "auth_asym_id": s["pdb_strand_id"],
                "auth_seq_id": s["auth_seq_num"],
                "pdb_seq_num": s["pdb_seq_num"],
                "insertion_code_raw": s["pdb_ins_code"],
                "atom_insertion_code_raw": local[0].identity.insertion_code_raw if local else "?",
            }
            residues.append(Residue(identity_row, local, ma, mr, s))
        require(
            all(
                any(a.identity.model_id == model and a.identity.label_asym_id == c for a in atoms)
                for c in protein_chains
            ),
            "Absent protein chain in model",
        )
    return Structure(
        expected["pdb_id"],
        expected["target"],
        block,
        tuple(sorted(atoms, key=lambda a: a.identity)),
        tuple(residues),
        tuple(sorted(models, key=int)),
        tuple(sorted(protein_chains)),
        assembly_supported(block, set(asym)),
        tuple(missing_atoms),
        tuple(missing_residues),
    )


def pair_record(ligand: Atom, protein: Atom, distance: float) -> dict[str, Any]:
    return {
        "ligand_atom": ligand.record(),
        "protein_atom": protein.record(),
        "distance_A": distance,
        "fractional_occupancy": ligand.fractional or protein.fractional,
    }


def proximity(
    site_id: str,
    ligand: Conformer,
    protein: Conformer,
    residue: Residue,
    ccd: Mapping[str, str],
    *,
    supported_assembly: bool = True,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    la = [a for a in ligand.atoms if a.eligible]
    pa = [a for a in protein.atoms if a.eligible]
    reasons = list(ligand.reasons + protein.reasons)
    missing_ligand = sorted(set(ccd) - {a.identity.label_atom_id for a in la})
    if missing_ligand or len(la) != len(ccd):
        reasons.append("incomplete_ligand_heavy_atom_set")
    if ligand.label != "." and protein.label != ".":
        reasons.append("unsupported_ligand_protein_altloc_compatibility")
    if not supported_assembly:
        reasons.append("unsupported_assembly_policy")
    if not pa:
        reasons.append("unobserved_residue" if not residue.atoms else "no_eligible_protein_atoms")
    missing = [r for r in residue.missing_atoms if r["label_alt_id"] in ("?", ".", protein.label)]
    excluded = [a.record() for a in protein.atoms if not a.eligible]
    partial = bool(
        missing
        or residue.missing_residues
        or any(a.occupancy == 0 and a.heavy for a in protein.atoms)
    )
    row: dict[str, Any] = {
        "site_id": site_id,
        "ligand_conformer_id": ligand.label,
        "residue_identity": residue.identity,
        "protein_conformer_id": protein.label,
        "observed_min_distance_A": None,
        "minimum_witness_pairs": [],
        **dict.fromkeys(THRESHOLDS),
        "geometry_status": "refused" if reasons else "partial_observed" if partial else "observed",
        "missing_or_excluded_atoms": {
            "declared_missing_atoms": missing,
            "declared_missing_residues": list(residue.missing_residues),
            "excluded_atoms": excluded,
        },
        "fractional_occupancy": any(a.fractional for a in la + pa),
        "conditional_local_conformer": ligand.label != "." or protein.label != ".",
        "complete_residue_distance_A": None,
        "complete_residue_distance_status": "not_estimated",
        "reasons": sorted(set(reasons)),
        "missing_ligand_heavy_atoms": missing_ligand,
        "retained_ligand_atom_count": len(la),
        "retained_protein_atom_count": len(pa),
    }
    if reasons:
        return row, []
    with np.errstate(all="raise"):
        delta = (
            np.asarray([a.xyz for a in la], dtype=np.float64)[:, None, :]
            - np.asarray([a.xyz for a in pa], dtype=np.float64)[None, :, :]
        )
        distances = np.sqrt(np.sum(delta * delta, axis=2, dtype=np.float64))
    require(bool(np.isfinite(distances).all()), "Nonfinite distance arithmetic")
    minimum = float(distances.min())
    row["observed_min_distance_A"] = minimum
    row.update({field: minimum <= radius for field, radius in zip(THRESHOLDS, RADII, strict=True)})
    pairs = []
    for i, ligand_atom in enumerate(la):
        for j, p in enumerate(pa):
            d = float(distances[i, j])
            if d == minimum:
                row["minimum_witness_pairs"].append(pair_record(ligand_atom, p, d))
            if d <= 5.0:
                pairs.append(
                    {
                        "site_id": site_id,
                        "ligand_conformer_id": ligand.label,
                        "protein_conformer_id": protein.label,
                        **pair_record(ligand_atom, p, d),
                    }
                )
    return row, pairs


def site_id(atom: Atom) -> str:
    i = atom.identity
    return f"{i.pdb_id}:model{i.model_id}:{i.label_asym_id}:{i.auth_asym_id}:{i.auth_seq_id}"


def analyze_structure(structure: Structure, ccd: Mapping[str, str]) -> dict[str, Any]:
    sites: dict[str, list[Atom]] = defaultdict(list)
    for atom in structure.atoms:
        if atom.identity.label_comp_id == "W0O":
            sites[site_id(atom)].append(atom)
    require(sites, "No W0O sites")
    result: dict[str, Any] = {
        k: []
        for k in (
            "sites",
            "residue_proximity",
            "atom_pairs_within_5A",
            "excluded_atoms",
            "nonprotein_inventory",
            "refusals",
        )
    }
    for atom in structure.atoms:
        reasons = atom.exclusion_reasons()
        if reasons:
            result["excluded_atoms"].append({**atom.record(original=True), "reasons": reasons})
        if not atom.protein:
            result["nonprotein_inventory"].append(atom.record(original=True))
    for name, atoms in sorted(sites.items()):
        first = atoms[0].identity
        local_residues = [r for r in structure.residues if r.identity["model_id"] == first.model_id]
        lconfs = conformers(atoms)
        complete_reasons = []
        if any(r.missing_atoms or r.missing_residues for r in local_residues):
            complete_reasons.append("deposited_dimer_has_missing_residues_or_atoms")
        if any(a.occupancy == 0 for r in local_residues for a in r.atoms):
            complete_reasons.append("zero_occupancy_atoms_excluded")
        if any(a.identity.label_alt_id != "." for r in local_residues for a in r.atoms):
            complete_reasons.append("local_alternatives_not_joint_complete_geometry")
        complete_reasons.append("complete_geometry_not_estimated_in_observed_only_contract")
        rows, pairs = [], []
        for lc in lconfs:
            for residue in local_residues:
                for pc in conformers(residue.atoms):
                    row, near = proximity(
                        name, lc, pc, residue, ccd, supported_assembly=structure.assembly_supported
                    )
                    rows.append(row)
                    pairs.extend(near)
        site: dict[str, Any] = {
            "site_identity": {
                "site_id": name,
                "pdb_id": structure.pdb_id,
                "target": structure.target,
                **{
                    k: v
                    for k, v in asdict(first).items()
                    if k
                    not in (
                        "label_atom_id",
                        "auth_atom_id",
                        "label_alt_id",
                        "type_symbol",
                        "atom_site_id",
                    )
                },
            },
            "all_deposited_ligand_atoms": [a.record() for a in atoms],
            "ligand_conformers": [
                {
                    "label": c.label,
                    "atom_site_ids": [a.identity.atom_site_id for a in c.atoms],
                    "reasons": list(c.reasons),
                }
                for c in lconfs
            ],
            "observed_geometry_status": "eligible_with_row_refusals"
            if any(r["observed_min_distance_A"] is not None for r in rows)
            else "refused",
            "complete_site_geometry_status": "refused",
            "complete_site_refusal_reason": complete_reasons,
            "receptor_members": [
                {
                    "label_asym_id": c,
                    "auth_asym_ids": sorted(
                        {
                            r.identity["auth_asym_id"]
                            for r in local_residues
                            if r.identity["label_asym_id"] == c
                        }
                    ),
                }
                for c in structure.protein_chains
            ],
            "fractional_occupancy": any(r["fractional_occupancy"] for r in rows),
            "warnings": [
                WARNING,
                "All distances concern retained atoms, not complete residues/pockets.",
            ],
            "refusals": [
                {
                    "residue_identity": r["residue_identity"],
                    "ligand_conformer_id": r["ligand_conformer_id"],
                    "protein_conformer_id": r["protein_conformer_id"],
                    "reasons": r["reasons"],
                }
                for r in rows
                if r["reasons"]
            ],
            "radius_contact_sets": {
                str(radius): [
                    {
                        "residue_identity": r["residue_identity"],
                        "ligand_conformer_id": r["ligand_conformer_id"],
                        "protein_conformer_id": r["protein_conformer_id"],
                        "geometry_status": r["geometry_status"],
                        "fractional_occupancy": r["fractional_occupancy"],
                    }
                    for r in rows
                    if r[field] is True
                ]
                for field, radius in zip(THRESHOLDS, RADII, strict=True)
            },
            "count_policy": (
                "No pooled contact count over mutually incompatible conformers or "
                "inferential statistics across sites."
            ),
        }
        result["sites"].append(site)
        result["residue_proximity"].extend(rows)
        result["atom_pairs_within_5A"].extend(pairs)
        result["refusals"].extend({"site_id": name, **r} for r in site["refusals"])
    result["missingness"] = {
        "pdb_id": structure.pdb_id,
        "atoms": sorted(structure.missing_atoms, key=lambda r: tuple(sorted(r.items()))),
        "residues": sorted(structure.missing_residues, key=lambda r: tuple(sorted(r.items()))),
    }
    return result


def canonical_rows(rows: list[Row]) -> list[str]:
    return sorted(json.dumps(r, sort_keys=True) for r in rows)


def load_package(
    repository: Path, manifest_path: Path, contract_path: Path
) -> tuple[list[Structure], dict[str, str], dict[str, Any]]:
    repository = repository.resolve(strict=True)
    for path in (manifest_path, contract_path):
        checked_path(repository, str(path.absolute().relative_to(repository)))
    manifest = read_json(manifest_path)
    exact_keys(
        manifest,
        "created_at_utc expected_entries input_revision inputs intended_use origin path_base "
        "schema_version source_change_policy",
        "manifest",
    )
    require(
        type(manifest["schema_version"]) is int and manifest["schema_version"] == 1,
        "Manifest version",
    )
    require(
        manifest["origin"] == "published_deposited_coordinates"
        and manifest["intended_use"] == "retrospective_descriptive_geometry_only",
        "Manifest scope",
    )
    require(
        isinstance(manifest["input_revision"], str)
        and re.fullmatch(r"[0-9a-f]{40}", manifest["input_revision"]),
        "Input revision",
    )
    require(
        isinstance(manifest["inputs"], list) and isinstance(manifest["expected_entries"], list),
        "Manifest arrays required",
    )
    by_role: dict[str, list[Path]] = defaultdict(list)
    hashes: dict[str, str] = {}
    allowed_roles = {
        "coordinates",
        "CCD_identity_and_atom_names",
        "CCD_descriptors",
        "ledger_cross_reference",
        "curated_metadata_expectations",
        "all_sites_expectations",
        "published_assignments_not_computed_contacts",
        "unchanged_existing_repository_provenance",
        "normative_fixed_contract",
        "implementation_acceptance_specification",
        "runtime_dependencies",
        "historical_input_manifest",
    }
    for record in manifest["inputs"]:
        exact_keys(record, "path role sha256", "input")
        require(record["role"] in allowed_roles, "Unknown input role")
        require(
            isinstance(record["sha256"], str) and re.fullmatch(r"[0-9a-f]{64}", record["sha256"]),
            "Invalid SHA256",
        )
        path = checked_path(repository, record["path"])
        require(str(path) not in hashes, "Duplicate input path")
        actual = digest(path.read_bytes())
        require(actual == record["sha256"], f"Input hash mismatch: {record['path']}")
        hashes[str(path)] = actual
        by_role[record["role"]].append(path)

    def one(role: str) -> Path:
        require(len(by_role[role]) == 1, f"Exactly one {role} input required")
        return by_role[role][0]

    require(
        one("normative_fixed_contract") == contract_path.absolute(),
        "Contract path not manifest-bound",
    )
    contract = read_json(contract_path)
    require(
        digest(
            json.dumps(contract, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        )
        == CONTRACT_SHA256,
        "Unsupported or modified contract",
    )
    for role in (
        "CCD_descriptors",
        "published_assignments_not_computed_contacts",
        "implementation_acceptance_specification",
    ):
        one(role)
    require(
        len(by_role["coordinates"]) == 2 and len(manifest["expected_entries"]) == 2,
        "Exactly two structures required",
    )
    ccd_block = parse_cif(one("CCD_identity_and_atom_names"))
    ccd_rows = category(ccd_block, "_chem_comp_atom.")
    require(ccd_rows and all(r["comp_id"] == "W0O" for r in ccd_rows), "CCD component identity")
    require(len({r["atom_id"] for r in ccd_rows}) == len(ccd_rows), "Duplicate CCD atoms")
    ccd = {r["atom_id"]: r["type_symbol"] for r in ccd_rows if r["type_symbol"] not in ("H", "D")}
    ligand = read_json(one("ledger_cross_reference"))
    require(
        ccd == ligand["expected_heavy_atoms"]
        and len(ccd) == ligand["heavy_atom_count"] == 30
        and ligand["ccd_component_id"] == "W0O",
        "CCD/ledger identity mismatch",
    )
    ledger_path = checked_path(repository, ligand["existing_ledger_path"])
    require(
        str(ledger_path) in hashes and hashes[str(ledger_path)] == ligand["existing_ledger_sha256"],
        "Ligand ledger not hash bound",
    )
    metadata = read_json(one("curated_metadata_expectations"))["structures"]
    sites = read_json(one("all_sites_expectations"))["sites"]
    metadata_by_id = {m["pdb_id"]: m for m in metadata}
    require(
        len(metadata_by_id) == len(metadata) == 2 and set(metadata_by_id) == {"8RIY", "8OTV"},
        "Metadata entries",
    )
    structures = []
    used_paths: set[Path] = set()
    used_entries: set[str] = set()
    for entry in sorted(manifest["expected_entries"], key=lambda e: e["pdb_id"]):
        exact_keys(
            entry,
            "assembly_id coordinate_path expected_model_ids expected_site_ids operation_ids "
            "pdb_id protein_label_asym_ids target",
            "expected entry",
        )
        pdb = entry["pdb_id"]
        require(pdb in metadata_by_id and pdb not in used_entries, "Unknown/duplicate entry")
        used_entries.add(pdb)
        require(
            entry["target"]
            == {"8RIY": "NUDT5", "8OTV": "NUDT14"}[pdb]
            == metadata_by_id[pdb]["target"],
            "Target mismatch",
        )
        require(
            entry["assembly_id"] == "1"
            and entry["operation_ids"] == ["1"]
            and entry["protein_label_asym_ids"] == ["A", "B"],
            "Fixed assembly identity",
        )
        for field in ("expected_model_ids", "expected_site_ids"):
            require(
                isinstance(entry[field], list)
                and entry[field]
                and all(isinstance(v, str) for v in entry[field])
                and len(set(entry[field])) == len(entry[field]),
                f"Invalid {field}",
            )
        path = checked_path(repository, entry["coordinate_path"])
        require(
            path in by_role["coordinates"] and path not in used_paths, "Malformed source mapping"
        )
        used_paths.add(path)
        structure = parse_structure(path, entry, ccd)
        m = metadata_by_id[pdb]
        require(len(structure.atoms) == m["atom_site_record_count"], "Atom inventory mismatch")
        require(list(structure.model_ids) == m["model_ids"], "Metadata model mismatch")
        for name, expected_rows in m["raw_mmcif_categories"].items():
            require(
                canonical_rows(category(structure.block, name, raw=True))
                == canonical_rows(expected_rows),
                f"Metadata mismatch: {pdb} {name}",
            )
        polymers = category(structure.block, "_entity_poly.")
        require(
            any(
                re.sub(r"\s+", "", r["pdbx_seq_one_letter_code_can"])
                == m["deposited_construct_sequence"]
                for r in polymers
            ),
            "Target sequence mismatch",
        )
        actual_sites: dict[str, list[Atom]] = defaultdict(list)
        for a in structure.atoms:
            if a.identity.label_comp_id == "W0O":
                actual_sites[site_id(a)].append(a)
        expected_sites = {s["site_id"]: s for s in sites if s["pdb_id"] == pdb}
        require(
            set(actual_sites) == set(entry["expected_site_ids"]) == set(expected_sites),
            "Unexpected/missing ligand site",
        )
        for name, aa in actual_sites.items():
            s = expected_sites[name]
            require(
                {a.identity.atom_site_id for a in aa} == set(s["atom_site_ids"]),
                "Site atom ID inventory mismatch",
            )
            for a in aa:
                for key in (
                    "pdb_id",
                    "model_id",
                    "label_entity_id",
                    "label_asym_id",
                    "auth_asym_id",
                    "label_seq_id",
                    "auth_seq_id",
                    "insertion_code_raw",
                    "label_comp_id",
                    "auth_comp_id",
                ):
                    require(getattr(a.identity, key) == s[key], f"Site identity mismatch: {key}")
        structures.append(structure)
    return (
        structures,
        ccd,
        {
            "input_hashes": hashes,
            "manifest_sha256": digest(manifest_path.read_bytes()),
            "contract_sha256": digest(contract_path.read_bytes()),
            "contract_id": contract["contract_id"],
        },
    )


def verified_source_inventory(repository: Path) -> dict[str, str]:
    """Check a portable snapshot without inventing Git history or authenticating its author."""
    inventory_path = repository / "SHA256SUMS.json"
    require(not inventory_path.is_symlink(), "Source inventory symlink refused")
    inventory = read_json(inventory_path)
    require(isinstance(inventory, dict) and bool(inventory), "Invalid source inventory")
    verified: dict[str, str] = {}
    for name, expected in inventory.items():
        require(isinstance(name, str) and bool(name), "Invalid source inventory path")
        relative = Path(name)
        require(
            not relative.is_absolute() and ".." not in relative.parts,
            "Unsafe source inventory path",
        )
        path = repository / relative
        require(
            not any(p.is_symlink() for p in (path, *path.parents)),
            "Source inventory symlink refused",
        )
        require(path.is_file(), f"Missing source inventory file: {name}")
        require(
            isinstance(expected, str) and digest(path.read_bytes()) == expected,
            f"Source inventory mismatch: {name}",
        )
        verified[name] = expected
    return verified


def git_state(repository: Path) -> dict[str, Any]:
    if not (repository / ".git").exists() and (repository / "SHA256SUMS.json").exists():
        verified_source_inventory(repository)
        return {
            "revision": None,
            "worktree_porcelain": None,
            "availability": "source_archive_without_git_history",
            "source_inventory_sha256": digest((repository / "SHA256SUMS.json").read_bytes()),
        }

    def run(*args: str) -> str:
        result = subprocess.run(
            ["git", "-C", str(repository), *args], capture_output=True, text=True, check=False
        )
        require(result.returncode == 0, "Cannot record Git provenance")
        return result.stdout.strip()

    return {
        "revision": run("rev-parse", "HEAD"),
        "worktree_porcelain": run("status", "--porcelain", "--untracked-files=all"),
    }


def build_result(
    repository: Path, manifest: Path, contract: Path, command: list[str]
) -> dict[str, Any]:
    started = datetime.now(UTC).isoformat()
    structures, ccd, provenance = load_package(repository, manifest, contract)
    result: dict[str, Any] = {
        "schema_version": 1,
        "contract_id": provenance["contract_id"],
        "origin": "retrospective_deposited_coordinates",
        "warning": WARNING,
        "primary_radius_A": 4.0,
        "radii_A": list(RADII),
        "provenance": {
            **provenance,
            "started_at_utc": started,
            "command": command,
            "repository": str(repository.resolve()),
            "manifest": str(manifest.resolve()),
            "contract": str(contract.resolve()),
            "code_path": str(Path(__file__).resolve()),
            "code_sha256": digest(Path(__file__).read_bytes()),
            "python": sys.version,
            "dependencies": {name: importlib.metadata.version(name) for name in ("gemmi", "numpy")},
            "git": git_state(repository),
        },
        "structures": [],
        "limitations": [
            WARNING,
            "All complete-residue distances are not_estimated.",
            "Unflagged observed residues are not independently validated complete models.",
            "Positive fractional occupancy is unweighted. B factors are not confidence intervals.",
            "Local alternatives do not assert correlated conformations across residues.",
        ],
    }
    for key in (
        "sites",
        "residue_proximity",
        "atom_pairs_within_5A",
        "excluded_atoms",
        "nonprotein_inventory",
        "missingness",
        "refusals",
    ):
        result[key] = []
    for s in structures:
        analysis = analyze_structure(s, ccd)
        for key, value in analysis.items():
            result[key].extend([value] if key == "missingness" else value)
        result["structures"].append(
            {
                "pdb_id": s.pdb_id,
                "target": s.target,
                "models": s.model_ids,
                "protein_chains": s.protein_chains,
                "assembly_supported": s.assembly_supported,
                "atom_count": len(s.atoms),
                "polymer_residue_model_count": len(s.residues),
                "occupancy_counts": dict(
                    sorted(Counter(a.raw["occupancy"] for a in s.atoms).items())
                ),
            }
        )
    result["provenance"]["finished_at_utc"] = datetime.now(UTC).isoformat()
    return result


def publish_json(output: Path, value: Any) -> None:
    payload = json_bytes(value)
    require(not os.path.lexists(output), "Existing output refused")
    require(
        output.parent.is_dir() and not output.parent.is_symlink(),
        "Existing nonsymlink output parent required",
    )
    staging: Path | None = None
    linked = False
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=output.parent, prefix=".structure-", delete=False
        ) as handle:
            staging = Path(handle.name)
            os.fchmod(handle.fileno(), 0o644)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(staging, output)
        linked = True
        fd = os.open(output.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    except BaseException:
        if (
            linked
            and staging is not None
            and output.exists()
            and os.path.samestat(output.stat(), staging.stat())
        ):
            output.unlink()
        raise
    finally:
        if staging is not None:
            staging.unlink(missing_ok=True)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument(
        "--output", type=Path, required=True, help="New JSON file; parent must exist"
    )
    args = parser.parse_args(argv)
    command = list(sys.argv if argv is None else ["nudt5-structure", *argv])
    try:
        require(args.repository.is_absolute(), "Repository must be absolute")
        repository = args.repository.resolve(strict=True)
        manifest = (
            args.input_manifest
            if args.input_manifest.is_absolute()
            else repository / args.input_manifest
        )
        contract = args.contract if args.contract.is_absolute() else repository / args.contract
        output = args.output.absolute()
        require(not os.path.lexists(output), "Existing output refused")
        result = build_result(repository, manifest, contract, command)
        publish_json(output, result)
    except (OSError, ValueError, RuntimeError, KeyError, TypeError, FloatingPointError) as exc:
        print(f"structure comparison refused: {exc}", file=sys.stderr)
        return 2
    print(f"Wrote observed-only result: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
