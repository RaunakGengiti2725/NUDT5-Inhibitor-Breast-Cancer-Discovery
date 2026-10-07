"""SOFTWARE TESTS ONLY: fixtures and checks here are not biological or density evidence."""

from __future__ import annotations

import gzip
import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

import build_structure_model_support as mod
import gemmi
import numpy as np
import pytest
from pipeline import ROOT
from structure_comparison import ValidationError

PACKAGE = ROOT / mod.PACKAGE
RESULTS = PACKAGE / "results"


@pytest.fixture(scope="module")
def built() -> tuple[dict[str, Any], dict[str, Any]]:
    return mod.build(ROOT)


def report_xml(groups: str, pdb: str = "TEST") -> bytes:
    return (
        f'<wwPDB-validation-information><Entry pdbid="{pdb}" XMLcreationDate="d"/>'
        f"{groups}</wwPDB-validation-information>"
    ).encode()


def group(**extra: str) -> str:
    attrs = {
        "model": "1",
        "said": "C",
        "seq": ".",
        "resname": "W0O",
        "altcode": " ",
        "chain": "A",
        "resnum": "301",
        "icode": " ",
        "ent": "2",
        "rscc": "0.9",
        "rsr": "0.1",
    } | extra
    return "<ModelledSubgroup " + " ".join(f'{k}="{v}"' for k, v in attrs.items()) + "/>"


IDENTITY = {
    "pdb_id": "TEST",
    "model_id": "1",
    "label_asym_id": "C",
    "auth_asym_id": "A",
    "label_seq_id": ".",
    "auth_seq_id": "301",
    "label_comp_id": "W0O",
    "label_entity_id": "2",
    "insertion_code_raw": "?",
}


def test_all_four_sites_map_exactly_with_label_and_auth_chains(built: Any) -> None:
    result, _ = built
    sites = {
        (
            s["site_identity"]["pdb_id"],
            s["site_identity"]["label_asym_id"],
            s["site_identity"]["auth_asym_id"],
            s["site_identity"]["auth_seq_id"],
        ): s
        for s in result["sites"]
    }
    assert set(sites) == {("8RIY", "C", "AAA", "301"), ("8RIY", "D", "BBB", "301")} | {
        ("8OTV", "C", "A", "301"),
        ("8OTV", "F", "B", "302"),
    }
    expected = {"8RIY": ("0.931", "0.940"), "8OTV": ("0.952", "0.928")}
    for pdb, values in expected.items():
        observed = sorted(
            (s["site_identity"]["label_asym_id"], s["report"]["metrics"]["rscc"]["raw"])
            for k, s in sites.items()
            if k[0] == pdb
        )
        assert tuple(v for _, v in observed) == values
    for site in sites.values():
        assert site["report"]["status"] == "exact_record"
        assert site["report"]["attributes_raw"]["chain"] == site["site_identity"]["auth_asym_id"]
        assert site["report"]["metrics"]["NatomsEDS"]["value"] == 30


def test_all_original_rows_preserved_and_missingness_not_reassigned(built: Any) -> None:
    result, _ = built
    original = json.loads(
        (ROOT / "research/structure_comparison/results/observed_proximity.json").read_text()
    )
    rows = original["residue_proximity"]
    assert len(result["residues"]) == len(rows) == 1730
    for new, old in zip(result["residues"], rows, strict=True):
        for key in mod.ROW_FIELDS:
            assert new[key] == old[key]
    assert Counter(r["geometry_status"] for r in result["residues"]) == {
        "observed": 1564,
        "partial_observed": 58,
        "refused": 108,
    }
    for row in result["residues"]:
        if row["geometry_status"] == "refused":
            assert row["report"]["status"] == "no_exact_report_record"
            assert row["observed_min_distance_A"] is None
    assert result["original_geometry_missingness"] == original["missingness"]
    assert result["original_geometry_refusals"] == original["refusals"]


def test_arg51_numbering_occupancy_and_zero_occupancy_atom(built: Any) -> None:
    result, _ = built
    arg = [
        a
        for a in result["local_atoms"]
        if a["pdb_id"] == "8RIY" and a["source_atom_row"]["auth_seq_id"] == "51"
    ]
    assert {
        (a["source_atom_row"]["label_asym_id"], a["source_atom_row"]["auth_asym_id"]) for a in arg
    } == {("A", "AAA"), ("B", "BBB")}
    assert {a["source_atom_row"]["label_seq_id"] for a in arg} == {"52"}
    assert {a["source_atom_row"]["label_comp_id"] for a in arg} == {"ARG"}
    cz = [a for a in arg if a["source_atom_row"]["label_atom_id"] == "CZ"]
    by_chain = {a["source_atom_row"]["auth_asym_id"]: a for a in cz}
    assert by_chain["AAA"]["source_atom_row"]["occupancy"] == "0.000"
    assert by_chain["AAA"]["sampling"] == {
        "status": "not_sampled_zero_occupancy_or_hydrogen",
        "maps": None,
    }
    assert by_chain["BBB"]["sampling"]["maps"] is not None
    cd = next(
        a
        for a in arg
        if a["source_atom_row"]["auth_asym_id"] == "BBB"
        and a["source_atom_row"]["label_atom_id"] == "CD"
    )
    assert cd["source_atom_row"]["occupancy"] == "0.780"
    report = result["local_residue_reports"]["8RIY:A:52:51:."]
    assert report["attributes_raw"]["chain"] == "AAA"
    assert {x["type"] for x in report["outliers_raw"]} >= {"bond-outlier", "angle-outlier"}


def test_target_numbering_not_silently_mapped(built: Any) -> None:
    result, _ = built
    names = {
        (r["residue_identity"]["pdb_id"], r["residue_identity"]["auth_seq_id"]): r[
            "residue_identity"
        ]["label_comp_id"]
        for r in result["residues"]
    }
    assert names[("8OTV", "51")] == "SER"
    assert names[("8OTV", "107")] == "LEU"
    assert names[("8RIY", "51")] == "ARG"


def test_alternates_retained_not_averaged(built: Any) -> None:
    result, _ = built
    altered = [r for r in result["residues"] if r["protein_conformer_id"] != "."]
    assert altered
    alt_atoms = [a for a in result["local_atoms"] if a["source_atom_row"]["label_alt_id"] != "."]
    for atom in alt_atoms:
        assert atom["sampling"]["status"] != "averaged"


def test_w0o_atom_inventory_and_sampling(built: Any) -> None:
    result, _ = built
    ligand = Counter(
        (a["pdb_id"], a["source_atom_row"]["label_asym_id"])
        for a in result["local_atoms"]
        if a["source_atom_row"]["label_comp_id"] == "W0O"
    )
    assert ligand == {("8RIY", "C"): 30, ("8RIY", "D"): 30, ("8OTV", "C"): 30, ("8OTV", "F"): 30}
    for atom in result["local_atoms"]:
        if atom["sampling"]["maps"] is not None:
            assert set(atom["sampling"]["maps"]) == {"EDS", "difference"}


def test_structure_factor_and_map_inventory(built: Any) -> None:
    result, _ = built
    for pdb, rows in (("8RIY", 26413), ("8OTV", 46127)):
        sf = result["structures"][pdb]["structure_factors"]
        assert sf["reflection_rows"] == rows
        assert {"F_meas_au", "pdbx_FWT", "pdbx_PHWT", "pdbx_DELFWT", "pdbx_DELPHWT"} <= set(
            sf["columns"]
        )
        assert set(result["structures"][pdb]["maps"]) == {"EDS", "difference"}


def test_report_parser_fail_closed() -> None:
    parsed = mod.parse_report(report_xml(group()), "TEST")
    assert mod.lookup_report(parsed, IDENTITY, ".")["status"] == "exact_record"
    missing = mod.lookup_report(parsed, IDENTITY | {"label_asym_id": "Z"}, ".")
    assert missing["status"] == "no_exact_report_record"
    assert missing["metrics"]["rscc"]["value"] is None
    with pytest.raises(ValidationError, match="chain mismatch"):
        mod.lookup_report(parsed, IDENTITY | {"auth_asym_id": "B"}, ".")
    with pytest.raises(ValidationError, match="Duplicate"):
        mod.parse_report(report_xml(group() + group()), "TEST")
    with pytest.raises(ValidationError, match="Wrong report"):
        mod.parse_report(report_xml(group()), "OTHER")
    with pytest.raises(ValidationError, match="Nonfinite"):
        mod.parse_report(report_xml(group(rscc="nan")), "TEST")
    with pytest.raises(ValidationError, match="out of range"):
        mod.parse_report(report_xml(group(rsr="1.5")), "TEST")
    with pytest.raises(ValidationError, match="declarations"):
        mod.parse_report(b"<!DOCTYPE x>" + report_xml(group()), "TEST")
    assert (
        mod.parse_report(report_xml(group(rscc="")), "TEST")["records"][
            ("1", "C", ".", "W0O", "", "301", "")
        ]["metrics"]["rscc"]["status"]
        == "not_reported"
    )


COORDINATE = """data_t
_entry.id TEST
_cell.length_a 10
_cell.length_b 11
_cell.length_c 12
_cell.angle_alpha 90
_cell.angle_beta 90
_cell.angle_gamma 90
_symmetry.space_group_name_H-M 'P 1'
"""


def sf_text(cell_a: str = "10", h2: str = "2", extra: str = "") -> bytes:
    return (
        COORDINATE.replace("length_a 10", "length_a " + cell_a)
        + "loop_\n_refln.index_h\n_refln.index_k\n_refln.index_l\n_refln.pdbx_FWT\n"
        + f"_refln.pdbx_PHWT\n1 0 0 1.0 0.0\n{h2} 0 0 2.0 90.0\n{extra}"
    ).encode()


def test_structure_factor_validation() -> None:
    block = gemmi.cif.read_string(COORDINATE).sole_block()
    result = mod.inspect_sf(sf_text(), "TEST", block)
    assert result["coefficient_pairs"]["pdbx_FWT/pdbx_PHWT"]["paired_rows"] == 2
    assert result["coefficient_pairs"]["pdbx_DELFWT/pdbx_DELPHWT"]["status"] == "unavailable"
    with pytest.raises(ValidationError, match="Unit-cell"):
        mod.inspect_sf(sf_text(cell_a="10.1"), "TEST", block)
    with pytest.raises(ValidationError, match="Duplicate Miller"):
        mod.inspect_sf(sf_text(h2="1"), "TEST", block)
    with pytest.raises(ValidationError, match="Structure-factor entry"):
        mod.inspect_sf(sf_text(), "OTHER", block)
    with pytest.raises(ValidationError, match="Non-numeric"):
        mod.inspect_sf(sf_text(extra="3 0 0 x 0.0\n"), "TEST", block)


def write_map(
    path: Path, values: np.ndarray[Any, Any], cell: tuple[float, ...], group: str
) -> None:
    density = gemmi.Ccp4Map()
    density.grid = gemmi.FloatGrid(values.astype(np.float32))
    density.grid.unit_cell = gemmi.UnitCell(*cell)
    density.grid.spacegroup = gemmi.find_spacegroup_by_name(group)
    density.update_ccp4_header()
    density.write_ccp4_map(str(path))


def test_map_validation(tmp_path: Path) -> None:
    block = gemmi.cif.read_string(COORDINATE).sole_block()
    values = np.random.default_rng(0).normal(size=(10, 10, 12))
    cell = (10.0, 11.0, 12.0, 90.0, 90.0, 90.0)
    good = tmp_path / "good.ccp4"
    write_map(good, values, cell, "P 1")
    grid, metadata = mod.load_map(good, block)
    assert metadata["shape"] == [10, 10, 12]
    assert metadata["full_cell_population_sd"] > 0
    wrong_cell = tmp_path / "cell.ccp4"
    write_map(wrong_cell, values, (10.5, 11.0, 12.0, 90.0, 90.0, 90.0), "P 1")
    with pytest.raises(ValidationError, match="Unit-cell"):
        mod.load_map(wrong_cell, block)
    wrong_group = tmp_path / "group.ccp4"
    write_map(wrong_group, values, cell, "P 2 2 2")
    with pytest.raises(ValidationError, match="space-group"):
        mod.load_map(wrong_group, block)
    constant = tmp_path / "constant.ccp4"
    write_map(constant, np.ones((10, 10, 12)), cell, "P 1")
    with pytest.raises(ValidationError, match="Constant"):
        mod.load_map(constant, block)
    row = {"occupancy": "0", "type_symbol": "C", "Cartn_x": "0", "Cartn_y": "0", "Cartn_z": "0"}
    assert mod.sample_atom(row, {"EDS": (grid, metadata)})["maps"] is None
    with pytest.raises(ValidationError, match="Nonfinite"):
        mod.sample_atom(row | {"occupancy": "1", "Cartn_x": "inf"}, {"EDS": (grid, metadata)})
    with pytest.raises(ValidationError, match="Invalid occupancy"):
        mod.sample_atom(row | {"occupancy": "1.2"}, {"EDS": (grid, metadata)})


def copied_repository(tmp_path: Path) -> Path:
    repository = tmp_path / "repo"
    manifest = json.loads((PACKAGE / "source_manifest.json").read_text())
    for item in manifest["baseline_inputs"]:
        target = repository / item["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / item["path"], target)
    shutil.copytree(PACKAGE, repository / mod.PACKAGE, ignore=shutil.ignore_patterns("results"))
    return repository


def test_source_hashes_fail_closed(tmp_path: Path) -> None:
    repository = copied_repository(tmp_path)
    mod.load_sources(repository)
    source = repository / mod.PACKAGE / "sources/8riy_validation.xml.gz"
    source.write_bytes(gzip.compress(gzip.decompress(source.read_bytes()) + b" "))
    with pytest.raises(ValidationError, match="Stored source hash"):
        mod.load_sources(repository)
    source.unlink()
    with pytest.raises(ValidationError, match="Missing input"):
        mod.load_sources(repository)


def test_baseline_and_manifest_tamper_refused(tmp_path: Path) -> None:
    repository = copied_repository(tmp_path)
    geometry = repository / "research/structure_comparison/results/observed_proximity.json"
    geometry.write_bytes(geometry.read_bytes() + b"\n")
    with pytest.raises(ValidationError, match="Baseline input hash"):
        mod.load_sources(repository)
    manifest = repository / mod.PACKAGE / "source_manifest.json"
    manifest.write_bytes(manifest.read_bytes() + b"\n")
    with pytest.raises(ValidationError, match="Fixed source manifest"):
        mod.load_sources(repository)


def test_existing_geometry_inputs_unchanged() -> None:
    manifest = json.loads((PACKAGE / "source_manifest.json").read_text())
    assert manifest["baseline_revision"] == "40b9b0708d888a015abe5043bb273c3c6ee601ae"
    for item in manifest["baseline_inputs"]:
        assert hashlib.sha256((ROOT / item["path"]).read_bytes()).hexdigest() == item["sha256"]


def test_main_refuses_existing_output(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    occupied = tmp_path / "out"
    occupied.mkdir()
    assert mod.main(["--output", str(occupied)]) == 2
    assert "new directory" in capsys.readouterr().err
    link = tmp_path / "link"
    link.symlink_to(tmp_path / "absent")
    assert mod.main(["--output", str(link)]) == 2
    assert not occupied.exists() or not any(occupied.iterdir())


def test_committed_results_match_current_code() -> None:
    completion = json.loads((RESULTS / "completion.json").read_text())
    script = Path(mod.__file__).read_bytes()
    assert completion["script_sha256"] == hashlib.sha256(script).hexdigest()
    assert completion["source_manifest_sha256"] == mod.MANIFEST_SHA256
    assert set(completion["outputs"]) | {"completion.json"} == {p.name for p in RESULTS.iterdir()}
    for name, digest in completion["outputs"].items():
        assert hashlib.sha256((RESULTS / name).read_bytes()).hexdigest() == digest


def test_regeneration_is_byte_identical(tmp_path: Path) -> None:
    output = tmp_path / "out"
    assert mod.main(["--output", str(output)]) == 0
    for path in RESULTS.iterdir():
        assert (output / path.name).read_bytes() == path.read_bytes(), path.name
