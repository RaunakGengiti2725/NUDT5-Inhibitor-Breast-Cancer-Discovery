"""Reproduce bounded identity checks only; never fit or score a model."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
from importlib.metadata import version
from pathlib import Path
from typing import Any

import gemmi
from build_pubchem_audit import build_audit, validated_snapshots
from pipeline import candidate_audit, deduplicate, molecule, read_compounds, write_json
from rdkit import rdBase
from rdkit.DataStructs.cDataStructs import TanimotoSimilarity
from transfer import identity, identity_matches, substitute_references

BASE = "40b9b0708d888a015abe5043bb273c3c6ee601ae"
EXPECTED = {
    "compounds.csv": "d628a0fa1926c1e44c9b2a79adf92b444959e80153cf9019ef5e56b4747c3576",
    "final_hits.csv": "38c8d5266ce127b1fd5babde0c895190d0b382e3c9e3dae7581e320766745100",
}
INPUTS = [
    *EXPECTED,
    "scripts/scripts/pipeline.py",
    "scripts/scripts/transfer.py",
    "scripts/scripts/controls.py",
    "requirements.lock",
    "requirements-structure.lock",
    "research/source_assays.csv",
    "research/reference_structures.csv",
    "research/reference_sources/9CH.cif.gz",
    "research/reference_sources/958.cif.gz",
    "research/structure_comparison/sources/W0O.cif",
    "research/selectivity/sources/jm4c00072_si_002.csv.gz",
    "research/external/observed_database_rows.csv",
    "research/external/nudt5_measured_ledger.csv",
    "scripts/build_pubchem_audit.py",
    *[
        f"research/external/pubchem/{name}.json"
        for name in (
            "query_manifest",
            "description_manifest",
            "concise_manifest",
            "cid_properties_manifest",
            "geneid-aids",
            "accession-aids",
            "assay-descriptions",
            "concise-activities",
            "cid-properties",
        )
    ],
]


def csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def graph(identifier: str, smiles: str, locator: str) -> dict[str, Any]:
    return {
        "id": identifier,
        "source_smiles": smiles,
        "locator": locator,
        **identity(smiles),
        "heavy_atoms": molecule(smiles).GetNumHeavyAtoms(),
    }


def ccd(root: Path, code: str, path: str) -> dict[str, Any]:
    raw = (root / path).read_bytes()
    text = gzip.decompress(raw) if path.endswith(".gz") else raw
    block = gemmi.cif.read_string(text.decode()).sole_block()
    descriptors = [
        dict(zip(("type", "program", "descriptor"), map(gemmi.cif.as_string, row), strict=True))
        for row in block.find("_pdbx_chem_comp_descriptor.", ["type", "program", "descriptor"])
    ]
    canonical = [r for r in descriptors if r["type"] == "SMILES_CANONICAL"]
    keys = {identity(r["descriptor"])["canonical_smiles"] for r in canonical}
    if len(keys) != 1:
        raise ValueError(f"{code}: CCD canonical descriptors disagree")
    out = graph(code, canonical[0]["descriptor"], path + "#_pdbx_chem_comp_descriptor")
    out.update(
        ccd_descriptors=descriptors,
        ccd_name=gemmi.cif.as_string(block.find_value("_chem_comp.name")),
        ccd_formula=gemmi.cif.as_string(block.find_value("_chem_comp.formula")),
        ccd_synonyms=gemmi.cif.as_string(block.find_value("_chem_comp.pdbx_synonyms")),
        source_url=f"https://files.rcsb.org/ligands/download/{code}.cif",
        decoded_sha256=hashlib.sha256(text).hexdigest(),
    )
    if out["formula"] != out["ccd_formula"].replace(" ", ""):
        raise ValueError(f"{code}: formula mismatch")
    if not any(r["type"] == "InChIKey" and r["descriptor"] == out["inchikey"] for r in descriptors):
        raise ValueError(f"{code}: InChIKey mismatch")
    return out


def build(root: Path) -> dict[str, Any]:
    from pipeline import smiles_to_bitvect

    hashes = {p: hashlib.sha256((root / p).read_bytes()).hexdigest() for p in INPUTS}
    for path, expected in EXPECTED.items():
        if hashes[path] != expected:
            raise ValueError(f"Original input hash mismatch: {path}")
    records, issues = read_compounds(root / "compounds.csv")
    candidates, candidate_issues = read_compounds(root / "final_hits.csv", labelled=False)
    records, duplicates = deduplicate(records)
    candidates, candidate_duplicates = deduplicate(candidates)
    training = [graph(r.identifier, r.smiles, f"compounds.csv#id={r.identifier}") for r in records]
    refs = [
        ccd(root, "9CH", "research/reference_sources/9CH.cif.gz"),
        ccd(root, "958", "research/reference_sources/958.cif.gz"),
        ccd(root, "W0O", "research/structure_comparison/sources/W0O.cif"),
    ]
    for ref, name, compound, pdbs, doi, locator in [
        (
            refs[0],
            "TH5427",
            "Page 28; Balikci 6",
            ["5NWH"],
            "10.1038/s41467-017-02293-7",
            "Page Sec37; reference_structures.csv",
        ),
        (
            refs[1],
            "TH1713",
            "Page 2",
            ["5NQR"],
            "10.1038/s41467-017-02293-7",
            "Page Sec37; reference_structures.csv",
        ),
        (
            refs[2],
            "compound 9",
            "Balikci 9",
            ["8RIY", "8OTV"],
            "10.1021/acs.jmedchem.4c00072",
            "Balikci sec4.1.2; fig3/fig4",
        ),
    ]:
        ref.update(
            authenticated_name=name,
            published_compound=compound,
            pdb_accessions=pdbs,
            primary_doi=doi,
            identity_locator=locator,
        )
    source_path = "research/selectivity/sources/jm4c00072_si_002.csv.gz"
    source_bytes = gzip.decompress((root / source_path).read_bytes())
    source_rows = list(csv.DictReader(io.StringIO(source_bytes.decode("latin-1"))))
    derived = {r["source_compound"]: r for r in csv_rows(root / "research/source_assays.csv")}
    if set(derived) != {r["Compound"] for r in source_rows}:
        raise ValueError("Source and curated compound inventories differ")
    sources = []
    for row in source_rows:
        cur = derived[row["Compound"]]
        comparisons = {
            "SMILES": "source_smiles",
            "NUDT5 IC50 (µM)": "nudt5_ic50_uM_as_reported",
            "NUDT14 IC50  (µM)": "nudt14_ic50_uM_as_reported",
        }
        if any(row[a].strip() != cur[b].strip() for a, b in comparisons.items()):
            raise ValueError("Curated source cells disagree with original supplement")
        sources.append(
            {
                **graph(
                    row["Compound"], row["SMILES"], source_path + "#Compound=" + row["Compound"]
                ),
                "source_doi": cur["source_doi"],
                "source_url": "https://doi.org/10.1021/acs.jmedchem.4c00072",
                "source_cells": row,
            }
        )
    pools = {"original_training": training, "balikci_23": sources, "authenticated_ccd": refs}
    for pool, path in [
        ("archived_database_records", "research/external/observed_database_rows.csv"),
        ("archived_measured_ledger", "research/external/nudt5_measured_ledger.csv"),
    ]:
        pools[pool] = [
            {
                **graph(f"{i}:{r['record_id']}", r["source_smiles"], f"{path}#data_row={i}"),
                "source_record": r,
            }
            for i, r in enumerate(csv_rows(root / path), start=1)
        ]
    pubchem_dir = root / "research/external/pubchem"
    snapshots = validated_snapshots(pubchem_dir)
    pubchem_audit = build_audit(pubchem_dir, root / "research/external/observed_database_rows.csv")
    pools["archived_pubchem_cids"] = []
    for row in snapshots["cid-properties.json"]["PropertyTable"]["Properties"]:
        info = graph(
            str(row["CID"]),
            row["SMILES"],
            f"research/external/pubchem/cid-properties.json#CID={row['CID']}",
        )
        if info["inchikey"] != row["InChIKey"] or info["formula"] != row["MolecularFormula"]:
            raise ValueError("PubChem graph, InChIKey and formula disagree")
        info["source_url"] = f"https://pubchem.ncbi.nlm.nih.gov/compound/{row['CID']}"
        info["source_properties"] = row
        pools["archived_pubchem_cids"].append(info)
    candidate_rows = []
    th5427 = next(r for r in refs if r["id"] == "9CH")
    for r, audit in zip(candidates, candidate_audit(records, candidates), strict=True):
        info = graph(r.identifier, r.smiles, f"final_hits.csv#id={r.identifier}")
        candidate_rows.append(
            {
                **info,
                "matches": {name: identity_matches(info, pool) for name, pool in pools.items()},
                "authenticated_TH5427_tanimoto": TanimotoSimilarity(
                    smiles_to_bitvect(r.smiles), smiles_to_bitvect(th5427["source_smiles"])
                ),
                "original_candidate_audit": audit,
            }
        )
    corrected, changes = substitute_references(records, root / "research/reference_structures.csv")
    corrected_graphs = [
        graph(r.identifier, r.smiles, "reference_sensitivity_only") for r in corrected
    ]
    source_overlaps = [
        {
            "source_compound": r["id"],
            "original_training": identity_matches(r, training),
            "authenticated_reference_sensitivity": identity_matches(r, corrected_graphs),
        }
        for r in sources
    ]
    return {
        "schema_version": 1,
        "base_revision": BASE,
        "rdkit_version": rdBase.rdkitVersion,
        "gemmi_version": version("gemmi"),
        "input_sha256": hashes,
        "parent_supplied_attachment_sha256": EXPECTED,
        "parent_attachment_hashes_match": True,
        "original_supplement_decoded_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "policy": {
            "exact": "RDKit canonical isomeric SMILES, without salt/protomer/tautomer merging",
            "parents": "transfer.identity neutral fragment parent and canonical parent tautomer",
            "parent_limit": (
                "Screening equivalence only; not physical sample identity or assay identity"
            ),
            "similarity": "Morgan radius 2, 2048 bits, no chirality; descriptive not activity",
            "scope": "Existing repository inputs only; no new data cohort, training or scoring",
            "non_overlap": "No match in these finite inputs does not establish chemical novelty",
            "database_limit": (
                "Multiple records and indexes may describe the same source experiment"
            ),
        },
        "invalid_training": issues,
        "invalid_candidates": candidate_issues,
        "training_duplicates": duplicates,
        "candidate_duplicates": candidate_duplicates,
        "pools": pools,
        "candidates": candidate_rows,
        "candidate_overlap_sets": {
            name: {
                level: [r["id"] for r in candidate_rows if r["matches"][name][level]]
                for level in (
                    "canonical_smiles",
                    "neutral_fragment_parent",
                    "canonical_parent_tautomer",
                )
            }
            for name in pools
        },
        "pubchem_linkage": pubchem_audit,
        "source_to_training": source_overlaps,
        "reference_substitutions_identity_only": changes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    write_json(args.output, build(args.repository.resolve()))
    print(json.dumps({"output": str(args.output), "mode": "identity_only_no_fitting"}))


if __name__ == "__main__":
    main()
