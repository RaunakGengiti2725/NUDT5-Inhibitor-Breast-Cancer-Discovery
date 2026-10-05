"""Reproduce a bounded PubChem provenance cross-check from hashed offline snapshots."""

from __future__ import annotations

import argparse
import hashlib
import json
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from pipeline import ROOT, write_json
from transfer import read_source_csv

MANIFESTS = (
    "query_manifest.json",
    "description_manifest.json",
    "concise_manifest.json",
    "cid_properties_manifest.json",
)


def validated_snapshots(directory: Path) -> dict[str, Any]:
    snapshots: dict[str, Any] = {}
    for name in MANIFESTS:
        manifest = json.loads((directory / name).read_text())
        for record in manifest if isinstance(manifest, list) else [manifest]:
            filename = record["file"]
            if Path(filename).name != filename or filename in snapshots:
                raise ValueError("Unique local snapshot filenames required")
            raw = (directory / filename).read_bytes()
            if (
                record["status"] != 200
                or len(raw) != record["size"]
                or hashlib.sha256(raw).hexdigest() != record["sha256"]
            ):
                raise ValueError(f"Snapshot provenance mismatch: {filename}")
            snapshots[filename] = json.loads(raw)
    return snapshots


def micromolar(value: str, units: str) -> Decimal | None:
    if not value:
        return None
    if units not in ("nM", "uM"):
        raise ValueError(f"Unsupported concentration unit: {units}")
    try:
        number = Decimal(value)
    except InvalidOperation as exc:
        raise ValueError("Positive finite concentration required") from exc
    if not number.is_finite() or number <= 0:
        raise ValueError("Positive finite concentration required")
    return number / (1000 if units == "nM" else 1)


def build_audit(directory: Path, ledger: Path) -> dict[str, Any]:
    snapshots = validated_snapshots(directory)
    gene = snapshots["geneid-aids.json"]["IdentifierList"]["AID"]
    protein = snapshots["accession-aids.json"]["IdentifierList"]["AID"]
    if len(gene) != len(set(gene)) or len(protein) != len(set(protein)):
        raise ValueError("Unique assay identifiers required within each target query")
    containers = snapshots["assay-descriptions.json"]["PC_AssayContainer"]
    descriptions = {c["assay"]["descr"]["aid"]["id"]: c["assay"]["descr"] for c in containers}
    if len(descriptions) != len(containers) or set(descriptions) != set(gene) | set(protein):
        raise ValueError("Unique descriptions must cover the target-query union exactly")
    property_rows = snapshots["cid-properties.json"]["PropertyTable"]["Properties"]
    properties = {str(row["CID"]): row for row in property_rows}
    if len(properties) != len(property_rows):
        raise ValueError("Unique PubChem compound identifiers required")
    known = read_source_csv(ledger, ("database", "record_id", "inchikey", "standard_endpoint"))
    table = snapshots["concise-activities.json"]["Table"]
    columns = table["Columns"]["Column"]
    if len(set(columns)) != len(columns):
        raise ValueError("Unique concise-table columns required")
    matches = []
    seen = set()
    for row in table["Row"]:
        if len(row["Cell"]) != len(columns):
            raise ValueError("Complete concise-table rows required")
        source = dict(zip(columns, row["Cell"], strict=True))
        signature = tuple(source[column] for column in columns)
        if signature in seen:
            raise ValueError("Duplicate concise activity row")
        seen.add(signature)
        description = descriptions[int(source["AID"])]
        origin = description["aid_source"]["db"]
        if origin["name"] != "ChEMBL":
            raise ValueError("This cross-check requires a ChEMBL source assay identifier")
        assay_id = origin["source_id"]["str"]
        key = properties[source["CID"]]["InChIKey"]
        value = micromolar(source["Activity Value [uM]"], "uM")
        candidates = [
            item
            for item in known
            if item["database"] == "ChEMBL"
            and item["assay_id"] == assay_id
            and item["inchikey"] == key
            and item["standard_endpoint"] == source["Activity Name"]
            and micromolar(item["value"], item["units"]) == value
        ]
        if len(candidates) != 1:
            raise ValueError(
                f"One-to-one source match required for AID {source['AID']}, CID {source['CID']}"
            )
        match = candidates[0]
        matches.append(
            {
                "aid": int(source["AID"]),
                "cid": int(source["CID"]),
                "inchikey": key,
                "chembl_assay_id": assay_id,
                "chembl_record_id": match["record_id"],
                "endpoint": source["Activity Name"],
                "pubchem_concise_value_uM": source["Activity Value [uM]"],
                "ledger_relation_not_provided_by_concise": match["relation"],
                "ledger_publication_doi": match["publication_doi"],
            }
        )
    catalog = []
    for aid, description in sorted(descriptions.items()):
        origin = description["aid_source"]["db"]
        catalog.append(
            {
                "aid": aid,
                "name": description["name"],
                "source": origin["name"],
                "source_assay_id": origin["source_id"]["str"],
                "gene_query": aid in gene,
                "accession_query": aid in protein,
                "url": f"https://pubchem.ncbi.nlm.nih.gov/bioassay/{aid}",
            }
        )
    return {
        "scope": (
            "Offline provenance cross-check; not a new biological evaluation or exhaustive search"
        ),
        "warning": (
            "Concise API omits relation symbols; "
            "joined ledger relations are not inferred from numeric values"
        ),
        "ledger_sha256": hashlib.sha256(ledger.read_bytes()).hexdigest(),
        "counts": {
            "gene_query_aids": len(gene),
            "accession_query_aids": len(protein),
            "union_aids": len(descriptions),
            "concise_aids": len({m["aid"] for m in matches}),
            "concise_rows_matched": len(matches),
            "distinct_cids_matched": len({m["cid"] for m in matches}),
            "ledger_strict_greater_than_rows": sum(
                m["ledger_relation_not_provided_by_concise"] == ">" for m in matches
            ),
            "missing_numeric_values": sum(not m["pubchem_concise_value_uM"] for m in matches),
        },
        "assay_catalog": catalog,
        "activity_matches": matches,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshots", type=Path, default=ROOT / "research/external/pubchem")
    parser.add_argument(
        "--ledger", type=Path, default=ROOT / "research/external/observed_database_rows.csv"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    write_json(args.output, build_audit(args.snapshots, args.ledger))


if __name__ == "__main__":
    main()
