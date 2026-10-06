"""SOFTWARE TESTS ONLY: generated fixtures are never biological evidence."""

from __future__ import annotations

import copy
import json
import math
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from typing import Any

import gemmi
import numpy as np
import pytest
import structure_comparison as mod
from pipeline import ROOT

PACKAGE = ROOT / "research/structure_comparison"
MANIFEST = PACKAGE / "runtime_input_manifest.json"
CONTRACT = PACKAGE / "geometry_contract.json"
CCD: dict[str, str] = json.loads((PACKAGE / "ligand_identity.json").read_text())[
    "expected_heavy_atoms"
]


def atom(
    name: str,
    xyz: tuple[float, float, float],
    *,
    occupancy: float = 1,
    alt: str = ".",
    element: str = "C",
    atom_id: str | None = None,
    protein: bool = True,
) -> mod.Atom:
    i = mod.AtomIdentity(
        "TEST",
        "1",
        "1",
        "1",
        "1",
        "A",
        "AAA",
        "52",
        "51",
        "A",
        "ALA",
        "ALA",
        name,
        name,
        alt,
        element,
        atom_id or name,
    )
    return mod.Atom(i, xyz, occupancy, None, protein, {})


def measure(
    protein: list[mod.Atom],
    ligand: list[mod.Atom] | None = None,
    missing: tuple[mod.Row, ...] = (),
    expected: dict[str, str] | None = None,
    supported: bool = True,
) -> list[tuple[dict[str, Any], list[dict[str, Any]]]]:
    ligand = [atom("L1", (0, 0, 0), protein=False)] if ligand is None else ligand
    residue = mod.Residue({"label_seq_id": "52"}, tuple(protein), missing, (), {})
    expected = (
        {a.identity.label_atom_id: a.identity.type_symbol for a in ligand}
        if expected is None
        else expected
    )
    return [
        mod.proximity("SOFTWARE ONLY", lc, pc, residue, expected, supported_assembly=supported)
        for lc in mod.conformers(ligand)
        for pc in mod.conformers(protein)
    ]


def test_T01_known_345() -> None:
    row, pairs = measure([atom("P1", (3, 4, 0))])[0]
    assert row["observed_min_distance_A"] == 5
    assert [row[f] for f in mod.THRESHOLDS] == [False, False, False, True]
    assert len(row["minimum_witness_pairs"]) == len(pairs) == 1


@pytest.mark.parametrize(
    "distance,expected",
    [
        (3.5, [True] * 4),
        (4, [False, True, True, True]),
        (4.5, [False, False, True, True]),
        (5, [False, False, False, True]),
        (5.000001, [False] * 4),
    ],
)
def test_T02_exact_boundaries(distance: float, expected: list[bool]) -> None:
    row, _ = measure([atom("P", (distance, 0, 0))])[0]
    assert row["observed_min_distance_A"] == distance
    assert [row[f] for f in mod.THRESHOLDS] == expected


def test_T03_all_ties() -> None:
    row, pairs = measure(
        [atom("P1", (3, 0, 0)), atom("P2", (3, 2, 0))],
        [atom("L1", (0, 0, 0)), atom("L2", (0, 2, 0))],
    )[0]
    assert row["observed_min_distance_A"] == 3
    assert [
        (p["ligand_atom"]["atom_site_id"], p["protein_atom"]["atom_site_id"])
        for p in row["minimum_witness_pairs"]
    ] == [("L1", "P1"), ("L2", "P2")]
    assert len(pairs) == 4


def test_T04_element_not_name_and_zero_exclusion() -> None:
    aa = [
        atom("C", (0.5, 0, 0), element="H"),
        atom("D", (0.6, 0, 0), element="D"),
        atom("Czero", (1, 0, 0), occupancy=0),
        atom("HEAVY", (6, 0, 0)),
    ]
    row, pairs = measure(aa)[0]
    assert row["observed_min_distance_A"] == 6 and row["within_4_0A"] is False
    assert row["geometry_status"] == "partial_observed"
    assert row["complete_residue_distance_A"] is None and pairs == []
    assert {a["atom_site_id"] for a in row["missing_or_excluded_atoms"]["excluded_atoms"]} == {
        "C",
        "D",
        "Czero",
    }


def test_T05_fractional_is_unweighted() -> None:
    row, pairs = measure([atom("P", (3, 0, 0), occupancy=0.5)])[0]
    assert row["observed_min_distance_A"] == 3 and row["fractional_occupancy"]
    assert pairs[0]["fractional_occupancy"]


def test_T06_separate_alternates_share_only_dot() -> None:
    aa = [
        atom("N", (6, 0, 0)),
        atom("C", (3, 0, 0), alt="A", occupancy=0.5, atom_id="CA"),
        atom("C", (8, 0, 0), alt="B", occupancy=0.5, atom_id="CB"),
    ]
    rows = [r for r, _ in measure(aa)]
    assert [r["protein_conformer_id"] for r in rows] == ["A", "B"]
    assert [r["observed_min_distance_A"] for r in rows] == [3, 6]
    assert [r["within_4_0A"] for r in rows] == [True, False]
    assert all(r["conditional_local_conformer"] for r in rows)


def test_T07_no_inferred_A_A_compatibility() -> None:
    aa = [atom("C", (3, 0, 0), alt=a, atom_id=a) for a in "AB"]
    ll = [atom("L1", (0, 0, 0), alt=a, atom_id=a) for a in "AB"]
    records = measure(aa, ll)
    assert len(records) == 4
    for row, pairs in records:
        assert row["observed_min_distance_A"] is None and not pairs
        assert "unsupported_ligand_protein_altloc_compatibility" in row["reasons"]


def test_T08_unknown_alternate_refused_locally() -> None:
    assert measure([atom("P", (3, 0, 0), alt="?")])[0][0]["reasons"] == ["unknown_altloc"]
    assert measure([atom("P", (3, 0, 0))])[0][0]["observed_min_distance_A"] == 3


def test_T09_ambiguous_shared_atom_refused() -> None:
    row, _ = measure([atom("C", (3, 0, 0)), atom("C", (5, 0, 0), alt="A", atom_id="alternate")])[0]
    assert row["reasons"] == ["ambiguous_shared_and_alternate_atom"]
    assert row["observed_min_distance_A"] is None


def test_T10_empty_and_zero_only_are_null() -> None:
    for aa, reason in [
        ([], "unobserved_residue"),
        ([atom("C", (1, 0, 0), occupancy=0)], "no_eligible_protein_atoms"),
    ]:
        row, _ = measure(aa)[0]
        assert row["reasons"] == [reason]
        assert row["observed_min_distance_A"] is None
        assert all(row[f] is None for f in mod.THRESHOLDS)


def test_T11_partial_observed_not_absence() -> None:
    row, _ = measure(
        [atom("P", (6, 0, 0))], missing=({"label_alt_id": "?", "label_atom_id": "CG"},)
    )[0]
    assert row["geometry_status"] == "partial_observed" and row["observed_min_distance_A"] == 6
    assert row["complete_residue_distance_A"] is None
    assert row["complete_residue_distance_status"] == "not_estimated"


@pytest.mark.parametrize("mode", ["removed", "zero"])
def test_T12_incomplete_real_CCD_set_refused(mode: str) -> None:
    ll = [atom(name, (0, 0, 0), element=e, protein=False) for name, e in CCD.items()]
    if mode == "removed":
        ll.pop()
    else:
        ll[-1] = replace(ll[-1], occupancy=0)
    row, pairs = measure([atom("P", (3, 0, 0))], ll, expected=CCD)[0]
    assert "incomplete_ligand_heavy_atom_set" in row["reasons"]
    assert len(row["missing_ligand_heavy_atoms"]) == 1 and not pairs
    assert row["observed_min_distance_A"] is None


def set_category(block: gemmi.cif.Block, name: str, rows: list[mod.Row]) -> None:
    if rows:
        block.set_mmcif_category(name, {key: [r[key] for r in rows] for key in rows[0]}, raw=True)
    else:
        block.find_mmcif_category(name).erase()


def simulated_cif(path: Path) -> tuple[gemmi.cif.Document, dict[str, Any]]:
    doc = gemmi.cif.Document()
    b = doc.add_new_block("SOFTWARE_ONLY")
    data: dict[str, list[mod.Row]] = {
        "_entry.": [{"id": "TEST"}],
        "_entity.": [{"id": "1", "type": "polymer"}, {"id": "2", "type": "non-polymer"}],
        "_entity_poly.": [{"entity_id": "1", "type": "'polypeptide(L)'"}],
        "_struct_asym.": [{"id": c, "entity_id": "2" if c == "C" else "1"} for c in "ABC"],
        "_entity_poly_seq.": [{"entity_id": "1", "num": "52", "mon_id": "ALA"}],
        "_pdbx_struct_assembly.": [{"id": "1"}],
        "_pdbx_struct_assembly_gen.": [
            {"assembly_id": "1", "oper_expression": "1", "asym_id_list": "A,B,C"}
        ],
        "_pdbx_struct_oper_list.": [
            {
                "id": "1",
                **{f"matrix[{i}][{j}]": str(int(i == j)) for i in (1, 2, 3) for j in (1, 2, 3)},
                **{f"vector[{i}]": "0" for i in (1, 2, 3)},
            }
        ],
        "_pdbx_poly_seq_scheme.": [
            {
                "asym_id": c,
                "entity_id": "1",
                "seq_id": "52",
                "mon_id": "ALA",
                "pdb_strand_id": c * 3,
                "pdb_ins_code": "A",
                "auth_seq_num": "51",
                "pdb_seq_num": "51",
            }
            for c in "AB"
        ],
    }
    rows: list[dict[str, str]] = []
    for chain, names, x in [("A", {"CA": "C"}, 8), ("B", {"CA": "C"}, 3), ("C", CCD, 0)]:
        for name, element in names.items():
            rows.append(
                {
                    "id": str(len(rows) + 1),
                    "group_PDB": "ATOM" if chain == "C" else "HETATM",
                    "type_symbol": element,
                    "label_atom_id": name,
                    "label_alt_id": ".",
                    "label_comp_id": "W0O" if chain == "C" else "ALA",
                    "label_asym_id": chain,
                    "label_entity_id": "2" if chain == "C" else "1",
                    "label_seq_id": "." if chain == "C" else "52",
                    "pdbx_PDB_ins_code": "?" if chain == "C" else "A",
                    "Cartn_x": str(x),
                    "Cartn_y": "0",
                    "Cartn_z": "0",
                    "occupancy": "1",
                    "B_iso_or_equiv": "?",
                    "auth_seq_id": "301" if chain == "C" else "51",
                    "auth_comp_id": "W0O" if chain == "C" else "ALA",
                    "auth_asym_id": "AAA" if chain == "C" else chain * 3,
                    "auth_atom_id": name,
                    "pdbx_PDB_model_num": "1",
                }
            )
    data["_atom_site."] = rows
    for cat, rr in data.items():
        set_category(b, cat, rr)
    doc.write_file(str(path))
    return doc, {
        "pdb_id": "TEST",
        "target": "SOFTWARE TESTS ONLY",
        "protein_label_asym_ids": ["A", "B"],
        "expected_model_ids": ["1"],
    }


@pytest.mark.parametrize(
    "mode", ["duplicate_id", "duplicate_identity", "duplicate_tag", "ragged", "blocks", "quote"]
)
def test_T13_parser_rejection(tmp_path: Path, mode: str) -> None:
    path = tmp_path / "software.cif"
    doc, expected = simulated_cif(path)
    rows = mod.category(doc.sole_block(), "_atom_site.", raw=True)
    if mode.startswith("duplicate_") and mode != "duplicate_tag":
        duplicate = dict(rows[0])
        duplicate["id"] = rows[0]["id"] if mode == "duplicate_id" else "999"
        set_category(doc.sole_block(), "_atom_site.", [*rows, duplicate])
        doc.write_file(str(path))
    else:
        suffix = {
            "duplicate_tag": "\n_entry.id AGAIN\n",
            "ragged": "\nloop_\n_t.a\n_t.b\n1\n",
            "blocks": "\ndata_other\n_entry.id OTHER\n",
            "quote": "\n_bad.x 'unterminated\n",
        }[mode]
        path.write_text(path.read_text() + suffix)
    with pytest.raises(mod.ValidationError):
        mod.parse_structure(path, expected, CCD)


@pytest.mark.parametrize(
    "field,value",
    [("Cartn_x", v) for v in ["NaN", "Inf", "text", "?"]]
    + [("occupancy", v) for v in ["NaN", "?", "-0.1", "1.1"]]
    + [("type_symbol", "?"), ("type_symbol", "Xx"), ("B_iso_or_equiv", "-1")],
)
def test_T14_bad_numbers_or_elements(tmp_path: Path, field: str, value: str) -> None:
    path = tmp_path / "software.cif"
    doc, expected = simulated_cif(path)
    rows = mod.category(doc.sole_block(), "_atom_site.", raw=True)
    rows[0][field] = value
    set_category(doc.sole_block(), "_atom_site.", rows)
    doc.write_file(str(path))
    with pytest.raises(mod.ValidationError):
        mod.parse_structure(path, expected, CCD)


def test_T14_ligand_element_and_nonpolymer_dot(tmp_path: Path) -> None:
    path = tmp_path / "software.cif"
    doc, expected = simulated_cif(path)
    assert len(mod.parse_structure(path, expected, CCD).atoms) == 32
    rows = mod.category(doc.sole_block(), "_atom_site.", raw=True)
    rows[2]["type_symbol"] = "O"
    set_category(doc.sole_block(), "_atom_site.", rows)
    doc.write_file(str(path))
    with pytest.raises(mod.ValidationError, match="W0O"):
        mod.parse_structure(path, expected, CCD)


@pytest.fixture
def package_copy(tmp_path: Path) -> tuple[Path, Path, Path]:
    manifest = mod.read_json(MANIFEST)
    for record in manifest["inputs"]:
        dst = tmp_path / record["path"]
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes((ROOT / record["path"]).read_bytes())
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_bytes(mod.json_bytes(manifest))
    return tmp_path, manifest_path, tmp_path / CONTRACT.relative_to(ROOT)


def rebind(manifest: Path, changed: Path) -> None:
    data = mod.read_json(manifest)
    for record in data["inputs"]:
        if record["path"] == str(changed.relative_to(manifest.parent)):
            record["sha256"] = mod.digest(changed.read_bytes())
    manifest.write_bytes(mod.json_bytes(data))


@pytest.mark.parametrize(
    "mode",
    [
        "unknown",
        "duplicate_key",
        "nan",
        "bool_radius",
        "sha",
        "traversal",
        "symlink",
        "wrong_target",
        "models",
        "site",
        "mapping",
        "absent_source",
        "sequence",
        "unknown_role",
        "extra_entry",
        "schema",
    ],
)
def test_T15_package_fail_closed(package_copy: tuple[Path, Path, Path], mode: str) -> None:
    root, manifest, contract = package_copy
    data = mod.read_json(manifest)
    if mode == "unknown":
        data["unknown"] = 1
    elif mode == "duplicate_key":
        manifest.write_text(
            manifest.read_text().replace(
                '"schema_version": 1', '"schema_version": 1, "schema_version": 1'
            )
        )
    elif mode == "nan":
        manifest.write_text(
            manifest.read_text().replace('"schema_version": 1', '"schema_version": NaN')
        )
    elif mode == "bool_radius":
        c = mod.read_json(contract)
        c["estimand"]["primary_radius_A"] = True
        contract.write_bytes(mod.json_bytes(c))
        rebind(manifest, contract)
    elif mode == "sha":
        data["inputs"][0]["sha256"] = "0" * 64
    elif mode == "traversal":
        data["inputs"][0]["path"] = "../bad.cif"
    elif mode == "symlink":
        p = root / data["inputs"][0]["path"]
        p.unlink()
        p.symlink_to(ROOT / data["inputs"][0]["path"])
    elif mode == "wrong_target":
        data["expected_entries"][0]["target"] = "NUDT14"
    elif mode == "models":
        data["expected_entries"][0]["expected_model_ids"] = ["1", "2"]
    elif mode == "site":
        data["expected_entries"][0]["expected_site_ids"].pop()
    elif mode == "mapping":
        data["expected_entries"][0]["coordinate_path"] = data["expected_entries"][1][
            "coordinate_path"
        ]
    elif mode == "absent_source":
        (root / data["inputs"][0]["path"]).unlink()
    elif mode == "sequence":
        p = root / data["inputs"][0]["path"]
        doc = gemmi.cif.read_file(str(p))
        rows = mod.category(doc.sole_block(), "_entity_poly_seq.", raw=True)
        rows[0]["mon_id"] = "GLY"
        set_category(doc.sole_block(), "_entity_poly_seq.", rows)
        doc.write_file(str(p))
        rebind(manifest, p)
    elif mode == "unknown_role":
        data["inputs"][0]["role"] = "bad"
    elif mode == "extra_entry":
        data["expected_entries"].append(data["expected_entries"][0])
    elif mode == "schema":
        data["schema_version"] = True
    if mode not in ("duplicate_key", "nan", "bool_radius", "sequence"):
        manifest.write_bytes(mod.json_bytes(data))
    output = root / "output.json"
    assert (
        mod.main(
            [
                "--repository",
                str(root),
                "--input-manifest",
                str(manifest),
                "--contract",
                str(contract),
                "--output",
                str(output),
            ]
        )
        == 2
    )
    assert not output.exists() and not list(root.glob(".structure-*"))


def test_T16_identity_dimer_no_monomer_fallback(tmp_path: Path) -> None:
    path = tmp_path / "software.cif"
    _, expected = simulated_cif(path)
    s = mod.parse_structure(path, expected, CCD)
    assert s.assembly_supported
    result = mod.analyze_structure(s, CCD)
    assert [r["observed_min_distance_A"] for r in result["residue_proximity"]] == [8, 3]
    assert [r["within_4_0A"] for r in result["residue_proximity"]] == [False, True]
    assert len(result["sites"]) == 1 and len(result["sites"][0]["receptor_members"]) == 2


def test_T17_nonidentity_refuses_all_rows(tmp_path: Path) -> None:
    path = tmp_path / "software.cif"
    doc, expected = simulated_cif(path)
    rows = mod.category(doc.sole_block(), "_pdbx_struct_oper_list.", raw=True)
    rows[0]["vector[1]"] = "1"
    set_category(doc.sole_block(), "_pdbx_struct_oper_list.", rows)
    doc.write_file(str(path))
    result = mod.analyze_structure(mod.parse_structure(path, expected, CCD), CCD)
    assert result["sites"][0]["observed_geometry_status"] == "refused"
    assert all(r["reasons"] == ["unsupported_assembly_policy"] for r in result["residue_proximity"])


def test_T18_classify_by_entity_not_keyword(tmp_path: Path) -> None:
    path = tmp_path / "software.cif"
    doc, expected = simulated_cif(path)
    rows = mod.category(doc.sole_block(), "_atom_site.", raw=True)
    for r in rows[:2]:
        r["Cartn_x"] = "7"
    for name, element in [("HOH", "O"), ("MG", "MG"), ("DMS", "C")]:
        r = dict(rows[2])
        r.update(
            id=str(len(rows) + 1),
            label_comp_id=name,
            auth_comp_id=name,
            label_atom_id=element,
            auth_atom_id=element,
            auth_seq_id=str(len(rows) + 1),
            type_symbol=element,
            Cartn_x="1",
        )
        rows.append(r)
    set_category(doc.sole_block(), "_atom_site.", rows)
    doc.write_file(str(path))
    result = mod.analyze_structure(mod.parse_structure(path, expected, CCD), CCD)
    assert all(r["observed_min_distance_A"] == 7 for r in result["residue_proximity"])
    assert {a["label_comp_id"] for a in result["nonprotein_inventory"]} == {
        "HOH",
        "MG",
        "DMS",
        "W0O",
    }


def test_T19_models_insertion_author_label_separate(tmp_path: Path) -> None:
    path = tmp_path / "software.cif"
    doc, expected = simulated_cif(path)
    b = doc.sole_block()
    rows = mod.category(b, "_atom_site.", raw=True)
    new = dict(rows[0])
    new.update(id="99", pdbx_PDB_ins_code="B", Cartn_x="9")
    rows.append(new)
    schema = mod.category(b, "_pdbx_poly_seq_scheme.", raw=True)
    new_s = dict(schema[0])
    new_s["pdb_ins_code"] = "B"
    schema.append(new_s)
    set_category(b, "_pdbx_poly_seq_scheme.", schema)
    rows += [
        {
            **r,
            "id": str(int(r["id"]) + 100),
            "pdbx_PDB_model_num": "2",
            "Cartn_x": str(float(r["Cartn_x"]) + 1),
        }
        for r in rows
    ]
    set_category(b, "_atom_site.", rows)
    doc.write_file(str(path))
    expected["expected_model_ids"] = ["1", "2"]
    s = mod.parse_structure(path, expected, CCD)
    result = mod.analyze_structure(s, CCD)
    assert len(s.residues) == 6 and len(result["sites"]) == 2
    assert {r.identity["insertion_code_raw"] for r in s.residues} == {"A", "B"}
    assert all(
        r.identity["auth_seq_id"] == "51" and r.identity["label_seq_id"] == "52" for r in s.residues
    )
    for r in result["residue_proximity"]:
        assert f"model{r['residue_identity']['model_id']}" in r["site_id"]
        assert all(
            p["ligand_atom"]["model_id"] == p["protein_atom"]["model_id"]
            for p in r["minimum_witness_pairs"]
        )


def test_T20_order_determinism(tmp_path: Path) -> None:
    path = tmp_path / "software.cif"
    doc, expected = simulated_cif(path)
    first = mod.analyze_structure(mod.parse_structure(path, expected, CCD), CCD)
    for cat in ["_atom_site.", "_pdbx_poly_seq_scheme."]:
        set_category(
            doc.sole_block(), cat, list(reversed(mod.category(doc.sole_block(), cat, raw=True)))
        )
    doc.write_file(str(path))
    assert first == mod.analyze_structure(mod.parse_structure(path, expected, CCD), CCD)
    assert mod.json_bytes({"z": 1, "a": 2}) == mod.json_bytes({"a": 2, "z": 1})


@pytest.mark.parametrize(
    "mode", ["file", "empty", "directory", "symlink", "dangling", "absent_parent"]
)
def test_T21_existing_outputs_refused(tmp_path: Path, mode: str) -> None:
    out = tmp_path / "result"
    if mode == "file":
        out.write_bytes(b"keep")
    elif mode == "empty":
        out.touch()
    elif mode == "directory":
        out.mkdir()
    elif mode in ("symlink", "dangling"):
        out.symlink_to(tmp_path if mode == "symlink" else tmp_path / "absent")
    else:
        out = out / "child"
    with pytest.raises((mod.ValidationError, OSError)):
        mod.publish_json(out, {"test": True})
    if mode == "file":
        assert out.read_bytes() == b"keep"
    if mode in ("symlink", "dangling"):
        assert out.is_symlink()
    assert not list(tmp_path.glob(".structure-*"))


def test_T21_concurrent_publication_and_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out = tmp_path / "result.json"

    def run() -> bool:
        try:
            mod.publish_json(out, {"test": True})
            return True
        except (OSError, mod.ValidationError):
            return False

    with ThreadPoolExecutor(max_workers=2) as pool:
        assert sorted(pool.map(lambda _: run(), range(2))) == [False, True]
    assert mod.read_json(out) == {"test": True}
    out.unlink()

    def failure(*args: Any, **kwargs: Any) -> None:
        raise OSError("simulated write/link failure")

    with monkeypatch.context() as mp:
        mp.setattr(os, "link", failure)
        with pytest.raises(OSError):
            mod.publish_json(out, {"test": True})
    with pytest.raises(ValueError):
        mod.publish_json(out, {"bad": math.nan})
    assert not out.exists() and not list(tmp_path.glob(".structure-*"))


@pytest.fixture(scope="module")
def real_structures() -> tuple[list[mod.Structure], dict[str, str], dict[str, Any]]:
    return mod.load_package(ROOT, MANIFEST, CONTRACT)


@pytest.fixture(scope="module")
def result() -> dict[str, Any]:
    return mod.build_result(
        ROOT, MANIFEST, CONTRACT, ["SOFTWARE VERIFICATION: real deposited inputs"]
    )


def test_T22_real_inventory(
    real_structures: tuple[list[mod.Structure], dict[str, str], dict[str, Any]],
) -> None:
    ss, ccd, _ = real_structures
    assert ccd == CCD
    assert {s.pdb_id: len(s.atoms) for s in ss} == {"8RIY": 3015, "8OTV": 3501}
    assert {s.pdb_id: len(s.residues) for s in ss} == {"8RIY": 418, "8OTV": 446}
    assert {s.pdb_id: len(s.missing_residues) for s in ss} == {"8RIY": 31, "8OTV": 23}
    assert {s.pdb_id: len(s.missing_atoms) for s in ss} == {"8RIY": 84, "8OTV": 31}
    assert sum(a.identity.label_comp_id == "W0O" for s in ss for a in s.atoms) == 120
    assert sum(a.occupancy == 0 for s in ss for a in s.atoms) == 1
    assert all(a.heavy for s in ss for a in s.atoms)
    s = next(s for s in ss if s.pdb_id == "8OTV")
    r = next(
        r
        for r in s.residues
        if r.identity["auth_asym_id"] == "B" and r.identity["auth_seq_id"] == "47"
    )
    assert [(c.label, len(c.atoms)) for c in mod.conformers(r.atoms)] == [("A", 8), ("B", 8)]


def test_T23_actual_output_acceptance(result: dict[str, Any]) -> None:
    rows = result["residue_proximity"]
    assert len(rows) == 1730 and len(result["sites"]) == 4
    assert all(s["complete_site_geometry_status"] == "refused" for s in result["sites"])
    assert len(result["excluded_atoms"]) == 375
    for s in result["sites"]:
        local = [r for r in rows if r["site_id"] == s["site_identity"]["site_id"]]
        assert len(local) == (447 if s["site_identity"]["pdb_id"] == "8OTV" else 418)
        assert {r["residue_identity"]["label_asym_id"] for r in local} == {"A", "B"}
    for r in rows:
        assert r["complete_residue_distance_A"] is None
        if r["observed_min_distance_A"] is not None:
            assert all(type(r[f]) is bool for f in mod.THRESHOLDS)
            assert [r[f] for f in mod.THRESHOLDS] == sorted(r[f] for f in mod.THRESHOLDS)
        else:
            assert all(r[f] is None for f in mod.THRESHOLDS)
        if r["missing_or_excluded_atoms"]["declared_missing_residues"]:
            assert r["observed_min_distance_A"] is None
        for p in r["minimum_witness_pairs"]:
            assert (p["protein_atom"]["pdb_id"], p["protein_atom"]["atom_site_id"]) != (
                "8RIY",
                "314",
            )
    assert all(
        (p["protein_atom"]["pdb_id"], p["protein_atom"]["atom_site_id"]) != ("8RIY", "314")
        for p in result["atom_pairs_within_5A"]
    )


@pytest.mark.parametrize("kind", ["atom", "residue", "join"])
def test_T24_contradictory_metadata(tmp_path: Path, kind: str) -> None:
    path = tmp_path / "software.cif"
    doc, expected = simulated_cif(path)
    row = {
        "id": "1",
        "PDB_model_num": "1",
        "polymer_flag": "Y",
        "occupancy_flag": "0" if kind == "atom" else "1",
        "auth_asym_id": "AAA",
        "auth_comp_id": "ALA",
        "auth_seq_id": "51",
        "PDB_ins_code": "A",
        "label_asym_id": "A",
        "label_comp_id": "ALA",
        "label_seq_id": "52",
    }
    if kind == "atom":
        row.update(auth_atom_id="CA", label_alt_id="?", label_atom_id="CA")
    if kind == "join":
        row["label_asym_id"] = "Q"
    cat = (
        "_pdbx_unobs_or_zero_occ_atoms." if kind == "atom" else "_pdbx_unobs_or_zero_occ_residues."
    )
    set_category(doc.sole_block(), cat, [row])
    doc.write_file(str(path))
    with pytest.raises(mod.ValidationError):
        mod.parse_structure(path, expected, CCD)


def test_T25_unspecified_missing_alt_flags_all() -> None:
    aa = [atom("C", (3, 0, 0), alt=a, atom_id=a) for a in "AB"]
    rr = measure(aa, missing=({"label_alt_id": "?", "label_atom_id": "N"},))
    assert len(rr) == 2 and all(r["geometry_status"] == "partial_observed" for r, _ in rr)
    assert all(len(r["missing_or_excluded_atoms"]["declared_missing_atoms"]) == 1 for r, _ in rr)


def transform(a: mod.Atom, rotation: Any, translation: Any) -> mod.Atom:
    xyz = rotation @ np.array(a.xyz) + translation
    return replace(a, xyz=(float(xyz[0]), float(xyz[1]), float(xyz[2])))


def test_rotation_translation_invariance() -> None:
    ll = [atom("L1", (1, 2, 3)), atom("L2", (-1, 2, 3))]
    pp = [atom("P1", (5, 6, 7)), atom("P2", (5, 6, 8))]
    base = measure(pp, ll)[0][0]
    theta = 0.713
    rotation = np.array(
        [[math.cos(theta), -math.sin(theta), 0], [math.sin(theta), math.cos(theta), 0], [0, 0, 1]]
    )
    for r, t in [
        (np.eye(3), np.array([12, -31, 8])),
        (rotation, np.zeros(3)),
        (rotation, np.array([12, -31, 8])),
    ]:
        row = measure([transform(a, r, t) for a in pp], [transform(a, r, t) for a in ll])[0][0]
        assert row["observed_min_distance_A"] == pytest.approx(
            base["observed_min_distance_A"], abs=1e-12
        )
        assert [row[f] for f in mod.THRESHOLDS] == [base[f] for f in mod.THRESHOLDS]


def test_nonfinite_arithmetic_refused() -> None:
    with pytest.raises(FloatingPointError):
        measure([atom("P", (1e308, 1e308, 1e308))])


def test_independent_actual_oracle_every_eligible_residue(result: dict[str, Any]) -> None:
    for pdb in ("8RIY", "8OTV"):
        structure = gemmi.read_structure(
            str(PACKAGE / f"sources/{pdb}.cif"), merge_chain_parts=False
        )
        model = structure[0]
        for row in result["residue_proximity"]:
            ri = row["residue_identity"]
            if ri["pdb_id"] != pdb or row["observed_min_distance_A"] is None:
                continue
            chain = row["site_id"].split(":")[2]
            ligand = [
                a
                for ch in model
                for res in ch
                if res.name == "W0O" and res.subchain == chain
                for a in res
            ]
            protein = [
                a
                for ch in model
                for res in ch
                if res.subchain == ri["label_asym_id"] and res.label_seq == int(ri["label_seq_id"])
                for a in res
                if a.occ > 0
                and a.element.name not in ("H", "D")
                and (a.altloc == "\0" or a.altloc == row["protein_conformer_id"])
            ]
            distances = [math.dist(a.pos.tolist(), b.pos.tolist()) for a in ligand for b in protein]
            assert min(distances) == pytest.approx(row["observed_min_distance_A"], abs=1e-12)
            assert [row[f] for f in mod.THRESHOLDS] == [
                min(distances) <= radius for radius in mod.RADII
            ]


def test_numeric_determinism_and_actual_provenance(result: dict[str, Any]) -> None:
    again = mod.build_result(ROOT, MANIFEST, CONTRACT, ["rerun"])
    assert {k: v for k, v in result.items() if k != "provenance"} == {
        k: v for k, v in again.items() if k != "provenance"
    }
    assert result["provenance"]["code_sha256"] == mod.digest(Path(mod.__file__).read_bytes())
    assert result["provenance"]["manifest_sha256"] == mod.digest(MANIFEST.read_bytes())


def test_changed_source_bytes_and_absent_chain_or_ligand(
    package_copy: tuple[Path, Path, Path],
) -> None:
    root, manifest, contract = package_copy
    data = mod.read_json(manifest)
    path = root / data["inputs"][0]["path"]
    path.write_bytes(path.read_bytes() + b"\n# changed source\n")
    with pytest.raises(mod.ValidationError, match="hash mismatch"):
        mod.load_package(root, manifest, contract)


@pytest.mark.parametrize("remove", ["ligand", "protein"])
def test_absent_site_or_chain(tmp_path: Path, remove: str) -> None:
    path = tmp_path / "software.cif"
    doc, expected = simulated_cif(path)
    rows = mod.category(doc.sole_block(), "_atom_site.", raw=True)
    rows = [r for r in rows if r["label_asym_id"] != ("C" if remove == "ligand" else "B")]
    set_category(doc.sole_block(), "_atom_site.", rows)
    doc.write_file(str(path))
    with pytest.raises(mod.ValidationError):
        mod.analyze_structure(mod.parse_structure(path, expected, CCD), CCD)


def test_copy_mutation_does_not_change_source() -> None:
    data = mod.read_json(MANIFEST)
    copied = copy.deepcopy(data)
    copied["origin"] = "not original"
    assert mod.read_json(MANIFEST) == data


@pytest.mark.parametrize("mutation", ["none", "changed", "missing", "traversal", "symlink"])
def test_portable_source_inventory_never_invents_git_history(tmp_path: Path, mutation: str) -> None:
    source = tmp_path / "evidence.txt"
    source.write_bytes(b"software fixture")
    inventory = {"evidence.txt": mod.digest(source.read_bytes())}
    if mutation == "changed":
        source.write_bytes(b"changed")
    elif mutation == "missing":
        source.unlink()
    elif mutation == "traversal":
        inventory = {"../outside": "0" * 64}
    elif mutation == "symlink":
        source.unlink()
        source.symlink_to(tmp_path / "elsewhere")
    (tmp_path / "SHA256SUMS.json").write_text(json.dumps(inventory))
    if mutation == "none":
        provenance = mod.git_state(tmp_path)
        assert provenance["revision"] is None
        assert provenance["worktree_porcelain"] is None
        assert provenance["availability"] == "source_archive_without_git_history"
    else:
        with pytest.raises(mod.ValidationError):
            mod.git_state(tmp_path)
