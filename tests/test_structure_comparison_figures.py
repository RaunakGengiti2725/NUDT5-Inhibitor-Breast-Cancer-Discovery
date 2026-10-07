"""SOFTWARE-ONLY rendering/table tests; not biological validation."""

from __future__ import annotations

import copy
import csv
import io
from pathlib import Path
from typing import Any

import build_structure_comparison_figures as figures
import pytest
import structure_comparison as mod
from PIL import Image
from pipeline import ROOT
from test_structure_comparison import CONTRACT, MANIFEST


@pytest.fixture(scope="module")
def result() -> dict[str, Any]:
    return mod.build_result(ROOT, MANIFEST, CONTRACT, ["SOFTWARE VERIFICATION"])


def test_tables_are_complete_and_lossless(result: dict[str, Any]) -> None:
    tables = figures.tables(figures.validate(result))
    assert tables == figures.tables(result)
    residues = list(csv.DictReader(io.StringIO(tables["residue_proximity.csv"].decode())))
    assert len(residues) == len(result["residue_proximity"]) == 1730
    for row, source in zip(residues, result["residue_proximity"], strict=True):
        distance = source["observed_min_distance_A"]
        assert row["observed_min_distance_A"] == ("" if distance is None else repr(distance))
        if distance is not None:
            assert float(row["observed_min_distance_A"]) == distance
        assert row["complete_residue_distance_A"] == ""
    pairs = list(csv.DictReader(io.StringIO(tables["atom_pairs_within_5A.csv"].decode())))
    assert len(pairs) == len(result["atom_pairs_within_5A"])
    assert all(float(p["distance_A"]) <= 5.0 for p in pairs)
    sensitivity = list(csv.DictReader(io.StringIO(tables["radius_sensitivity.csv"].decode())))
    assert len(sensitivity) == 16
    assert all("No pooled count" in r["interpretation_limit"] for r in sensitivity)


def test_figures_are_deterministic_labelled_and_complete(result: dict[str, Any]) -> None:
    first = figures.render(result)
    assert first == figures.render(result)
    assert set(first) == {f"{figures.FIGURE}.{e}" for e in ("png", "svg", "pdf")}
    assert Image.open(io.BytesIO(first[f"{figures.FIGURE}.png"])).format == "PNG"
    assert first[f"{figures.FIGURE}.pdf"].startswith(b"%PDF-")
    svg = first[f"{figures.FIGURE}.svg"].decode()
    for label in (
        "NUDT5",
        "NUDT14",
        "8RIY",
        "8OTV",
        "Å",
        "4.0 (primary)",
        "not homology",
        "not binding energy",
        "Arg51",
        "Leu107",
        "alt A",
        "null/refused",
    ):
        assert label in svg
    for site in result["sites"]:
        assert site["site_identity"]["site_id"].split(":model1")[1] in svg


@pytest.mark.parametrize("mode", ["threshold", "warning", "missing", "null"])
def test_invalid_results_refused(result: dict[str, Any], mode: str) -> None:
    bad = copy.deepcopy(result)
    row = next(r for r in bad["residue_proximity"] if r["observed_min_distance_A"] is not None)
    if mode == "threshold":
        row["within_4_0A"] = not row["within_4_0A"]
    elif mode == "warning":
        bad["warning"] = "changed"
    elif mode == "missing":
        del bad["refusals"]
    else:
        row["observed_min_distance_A"] = None
    with pytest.raises((ValueError, KeyError)):
        figures.validate(bad)


def test_cli_hash_and_overwrite_refusal(tmp_path: Path, result: dict[str, Any]) -> None:
    source = tmp_path / "result.json"
    source.write_bytes(mod.json_bytes(result))
    output = tmp_path / "derived"
    args = ["--input", str(source), "--output", str(output), "--input-sha256"]
    assert figures.main([*args, "0" * 64]) == 2 and not output.exists()
    assert figures.main([*args, mod.digest(source.read_bytes())]) == 0
    manifest = mod.read_json(output / "derived_manifest.json")
    assert manifest["input_sha256"] == mod.digest(source.read_bytes())
    assert manifest["source_validation"]["input_hashes_relative"] == figures.source_hashes(
        result["provenance"]
    )
    assert (
        manifest["source_validation"]["manifest_sha256"] == result["provenance"]["manifest_sha256"]
    )
    for name, entry in manifest["artifacts"].items():
        assert mod.digest((output / name).read_bytes()) == entry["sha256"]
    before = {p.name: p.read_bytes() for p in output.iterdir()}
    assert figures.main([*args, mod.digest(source.read_bytes())]) == 2
    assert before == {p.name: p.read_bytes() for p in output.iterdir()}


@pytest.mark.parametrize(
    "mode",
    [
        "nan",
        "negative",
        "string",
        "bool",
        "duplicate",
        "witness",
        "occupancy",
        "coordinates",
        "missingness",
        "pair_nan",
        "site",
        "unestimated",
    ],
)
def test_malformed_geometry_refuses_before_render(result: dict[str, Any], mode: str) -> None:
    bad = copy.deepcopy(result)
    row = next(r for r in bad["residue_proximity"] if r["observed_min_distance_A"] is not None)
    if mode in ("nan", "negative", "string", "bool"):
        row["observed_min_distance_A"] = {
            "nan": float("nan"),
            "negative": -1,
            "string": "3.0",
            "bool": True,
        }[mode]
    elif mode == "duplicate":
        bad["residue_proximity"].append(row)
    elif mode == "witness":
        row["minimum_witness_pairs"] = []
    elif mode == "occupancy":
        row["minimum_witness_pairs"][0]["protein_atom"]["occupancy"] = 0
    elif mode == "coordinates":
        atom = row["minimum_witness_pairs"][0]["protein_atom"]
        atom["xyz_A"] = [x + 100 for x in atom["xyz_A"]]
    elif mode == "missingness":
        del row["missing_or_excluded_atoms"]
    elif mode == "pair_nan":
        bad["atom_pairs_within_5A"][0]["distance_A"] = float("nan")
    elif mode == "site":
        row["site_id"] = "invented"
    else:
        row["complete_residue_distance_A"] = 3.0
    with pytest.raises(ValueError):
        figures.validate(bad)


@pytest.mark.parametrize("target", ["NUDT5", "NUDT14"])
def test_readable_target_panels_retain_both_sites_and_all_rows(
    result: dict[str, Any], target: str
) -> None:
    files = figures.render(result, target=target)
    assert files == figures.render(result, target=target)
    svg = files[f"{figures.FIGURE}_{target}.svg"].decode()
    rows = [
        r
        for r in result["residue_proximity"]
        if r["residue_identity"]["pdb_id"] == ("8RIY" if target == "NUDT5" else "8OTV")
    ]
    for row in rows:
        if row["within_5_0A"]:
            assert figures.residue_label(row) in svg
    assert "4.0 (primary)" in svg and "null/refused" in svg
    import xml.etree.ElementTree as ET

    visible = " ".join(" ".join(ET.fromstring(svg).itertext()).split())
    assert figures.CAVEAT in visible
    assert len({r["site_id"] for r in rows}) == 2


PUBLICATION_DEFECTS = (
    "drop_site",
    "drop_structure",
    "drop_residue_and_adjust_count",
    "drop_alternate",
    "empty_pairs",
    "drop_pair",
    "duplicate_pair",
    "swap_targets",
    "witness_ligand_component",
    "witness_protein_identity",
    "witness_other_residue",
    "witness_coherent_coordinates",
    "duplicate_witness",
    "pair_other_site",
    "pair_conformer",
    "pair_fractional_flag",
    "site_inventory",
    "site_receptor",
    "site_radius_summary",
    "missingness",
    "excluded_atoms",
    "nonprotein_inventory",
    "refusals",
    "limitations",
    "source_hashes",
    "manifest_hash",
    "contract_hash",
    "source_path_escape",
    "extra_field",
)


def software_corrupted_result(result: dict[str, Any], defect: str) -> dict[str, Any]:
    bad = copy.deepcopy(result)
    if defect in ("drop_site", "drop_structure"):
        removed = {
            s["site_identity"]["site_id"]
            for s in bad["sites"]
            if s["site_identity"]["site_id"] == "8OTV:model1:C:A:301"
            or (defect == "drop_structure" and s["site_identity"]["pdb_id"] == "8OTV")
        }
        bad["sites"] = [s for s in bad["sites"] if s["site_identity"]["site_id"] not in removed]
        for key in ("residue_proximity", "atom_pairs_within_5A"):
            bad[key] = [r for r in bad[key] if r["site_id"] not in removed]
        if defect == "drop_structure":
            bad["structures"] = [s for s in bad["structures"] if s["pdb_id"] != "8OTV"]
    elif defect == "drop_residue_and_adjust_count":
        bad["residue_proximity"] = [
            r
            for r in bad["residue_proximity"]
            if (r["residue_identity"]["label_asym_id"], r["residue_identity"]["label_seq_id"])
            != ("A", "1")
        ]
        for structure in bad["structures"]:
            structure["polymer_residue_model_count"] -= 1
    elif defect == "drop_alternate":
        row = next(r for r in bad["residue_proximity"] if r["protein_conformer_id"] == "A")
        bad["residue_proximity"].remove(row)
    elif defect in ("empty_pairs", "drop_pair", "duplicate_pair"):
        pairs = bad["atom_pairs_within_5A"]
        if defect == "empty_pairs":
            pairs.clear()
        elif defect == "drop_pair":
            pairs.pop()
        else:
            pairs.append(copy.deepcopy(pairs[0]))
    elif defect == "swap_targets":
        for structure in bad["structures"]:
            structure["target"] = {"NUDT5": "NUDT14", "NUDT14": "NUDT5"}[structure["target"]]
        for site in bad["sites"]:
            i = site["site_identity"]
            i["target"] = {"NUDT5": "NUDT14", "NUDT14": "NUDT5"}[i["target"]]
    elif defect.startswith("witness_") or defect == "duplicate_witness":
        rows = [r for r in bad["residue_proximity"] if r["minimum_witness_pairs"]]
        if defect == "witness_other_residue":
            rows[0]["minimum_witness_pairs"] = copy.deepcopy(rows[1]["minimum_witness_pairs"])
            rows[0]["observed_min_distance_A"] = rows[1]["observed_min_distance_A"]
            for field in mod.THRESHOLDS:
                rows[0][field] = rows[1][field]
        elif defect == "witness_coherent_coordinates":
            for atom in rows[0]["minimum_witness_pairs"][0].values():
                if isinstance(atom, dict) and "xyz_A" in atom:
                    atom["xyz_A"] = [x + 100 for x in atom["xyz_A"]]
        elif defect == "duplicate_witness":
            rows[0]["minimum_witness_pairs"].append(
                copy.deepcopy(rows[0]["minimum_witness_pairs"][0])
            )
        else:
            for row in rows:
                for pair in row["minimum_witness_pairs"]:
                    if defect == "witness_ligand_component":
                        pair["ligand_atom"]["label_comp_id"] = "NOT_W0O"
                    else:
                        pair["protein_atom"].update(
                            label_asym_id="WRONG_CHAIN", label_comp_id="XXX", is_protein=False
                        )
    elif defect.startswith("pair_"):
        pair = bad["atom_pairs_within_5A"][0]
        if defect == "pair_other_site":
            pair["site_id"] = "8RIY:model1:C:AAA:301"
        elif defect == "pair_conformer":
            pair["protein_conformer_id"] = "B"
        else:
            pair["fractional_occupancy"] = not pair["fractional_occupancy"]
    elif defect.startswith("site_"):
        site = bad["sites"][0]
        key = {
            "site_inventory": "all_deposited_ligand_atoms",
            "site_receptor": "receptor_members",
            "site_radius_summary": "radius_contact_sets",
        }[defect]
        site[key] = []
    elif defect in ("source_hashes", "manifest_hash", "contract_hash", "source_path_escape"):
        p = bad["provenance"]
        if defect == "source_hashes":
            p["input_hashes"][next(iter(p["input_hashes"]))] = "0" * 64
        elif defect == "source_path_escape":
            key = next(iter(p["input_hashes"]))
            p["input_hashes"][str(Path(p["repository"]) / ".." / "outside")] = p[
                "input_hashes"
            ].pop(key)
        else:
            p[defect.replace("_hash", "_sha256")] = "0" * 64
    elif defect == "extra_field":
        bad["undeclared"] = "SOFTWARE TEST ONLY"
    else:
        assert defect in (
            "missingness",
            "excluded_atoms",
            "nonprotein_inventory",
            "refusals",
            "limitations",
        )
        bad[defect] = []
    assert mod.json_bytes(bad) != mod.json_bytes(result)
    return bad


@pytest.mark.parametrize("defect", PUBLICATION_DEFECTS)
def test_self_consistent_corruptions_never_publish(
    result: dict[str, Any], tmp_path: Path, defect: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    import build_research_documents as documents

    def forbidden(*args: Any, **kwargs: Any) -> None:
        pytest.fail("Invalid SOFTWARE TEST ONLY evidence reached rendering/staging")

    monkeypatch.setattr(figures, "render", forbidden)
    monkeypatch.setattr(documents, "_build", forbidden)
    bad = software_corrupted_result(result, defect)
    source, manifest = tmp_path / "SOFTWARE_TEST_ONLY.json", tmp_path / "manifest.json"
    source.write_bytes(mod.json_bytes(bad))
    sha = mod.digest(source.read_bytes())
    manifest.write_bytes(mod.json_bytes({"input_sha256": sha}))
    output = tmp_path / "figures"
    assert (
        figures.main(
            [
                "--input",
                str(source),
                "--input-sha256",
                sha,
                "--repository",
                str(ROOT),
                "--output",
                str(output),
            ]
        )
        == 2
    )
    assert not output.exists()
    for required in (True, False):
        output = tmp_path / f"documents-{required}"
        with pytest.raises(ValueError):
            documents.build(
                ROOT / "research/manuscript.md",
                ROOT / "research/results",
                output,
                structure_input=source,
                structure_manifest=manifest,
                require_structure=required,
                repository=ROOT,
            )
        assert not output.exists()
    assert not list(tmp_path.glob(".nudt5-*"))
    assert source.read_bytes() == mod.json_bytes(bad)


@pytest.mark.parametrize("collection", ["minimum_witness_pairs", "atom_pairs_within_5A"])
@pytest.mark.parametrize("role", ["ligand_atom", "protein_atom"])
@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("pdb_id", "XXXX"),
        ("model_id", "2"),
        ("assembly_id", "2"),
        ("operation_id", "2"),
        ("atom_site_id", "-1"),
        ("label_entity_id", "99"),
        ("label_asym_id", "WRONG"),
        ("auth_asym_id", "WRONG"),
        ("label_seq_id", "999"),
        ("auth_seq_id", "999"),
        ("insertion_code_raw", "X"),
        ("label_comp_id", "XXX"),
        ("auth_comp_id", "XXX"),
        ("label_atom_id", "FAKE"),
        ("auth_atom_id", "FAKE"),
        ("label_alt_id", "Z"),
        ("type_symbol", "H"),
        ("occupancy", 0.5),
        ("B_iso_or_equiv", 999.0),
        ("fractional_occupancy", True),
        ("is_protein", None),
    ],
)
def test_every_deposited_atom_field_is_source_bound(
    result: dict[str, Any], collection: str, role: str, field: str, value: Any
) -> None:
    bad = copy.deepcopy(result)
    if collection == "minimum_witness_pairs":
        pairs = next(r[collection] for r in bad["residue_proximity"] if r[collection])
    else:
        pairs = bad[collection]
    atom = pairs[0][role]
    if field == "is_protein":
        value = not atom[field]
    assert atom[field] != value
    atom[field] = value
    with pytest.raises(ValueError, match="Source-package mismatch"):
        figures.validate(bad)


def test_historical_runtime_provenance_is_portable_not_rewritten(
    result: dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    portable = copy.deepcopy(result)
    p = portable["provenance"]
    old = Path(p["repository"])
    p["repository"] = "/historical/machine/evidence"
    p["input_hashes"] = {
        str(Path(p["repository"]) / Path(name).relative_to(old)): sha
        for name, sha in p["input_hashes"].items()
    }
    p["git"] = {"revision": "a" * 40, "worktree_porcelain": "historical runtime state"}
    p["command"] = ["SOFTWARE TEST ONLY historical command"]
    before = mod.json_bytes(portable)
    monkeypatch.chdir(tmp_path)
    assert figures.validate(portable, repository=ROOT) == portable
    assert mod.json_bytes(portable) == before


@pytest.mark.parametrize("failure", ["missing", "corrupted", "symlink", "relative"])
def test_publication_requires_trusted_source_package(
    result: dict[str, Any], tmp_path: Path, failure: str
) -> None:
    import shutil

    repository = tmp_path / "trusted"
    if failure == "relative":
        repository = Path("relative")
    elif failure == "symlink":
        repository.symlink_to(ROOT, target_is_directory=True)
    else:
        repository.mkdir()
        for path in (MANIFEST, CONTRACT):
            target = repository / path.relative_to(ROOT)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
        if failure == "corrupted":
            for entry in mod.read_json(MANIFEST)["inputs"]:
                target = repository / entry["path"]
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(ROOT / entry["path"], target)
            with (repository / "research/structure_comparison/sources/8OTV.cif").open("ab") as out:
                out.write(b"\n# SOFTWARE TEST ONLY changed source\n")
    with pytest.raises(ValueError):
        figures.validate(result, repository=repository)


@pytest.mark.parametrize("target", ["NUDT5", "NUDT14"])
def test_document_panel_legend_clears_axis_title(
    result: dict[str, Any], target: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    from matplotlib.figure import Figure

    original = Figure.savefig

    def inspect(figure: Any, *args: Any, **kwargs: Any) -> Any:
        figure.canvas.draw()
        renderer = figure.canvas.get_renderer()
        legend = figure.texts[1].get_window_extent(renderer)
        title = figure.axes[0].title.get_window_extent(renderer)
        assert legend.y0 > title.y1
        return original(figure, *args, **kwargs)

    monkeypatch.setattr(Figure, "savefig", inspect)
    assert figures.render(result, target=target, document_layout=True)
