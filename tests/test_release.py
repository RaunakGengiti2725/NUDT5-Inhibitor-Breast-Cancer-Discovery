from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from build_release_manifest import build_manifest
from pipeline import ROOT


def test_inventory_hashes_inputs_and_refuses_overwrite(tmp_path: Path) -> None:
    source = tmp_path / "compounds.csv"
    source.write_text("original input")
    generated = tmp_path / "scripts" / "package.egg-info"
    generated.mkdir(parents=True)
    (generated / "PKG-INFO").write_text("Build metadata is not an evidence input")
    output = tmp_path / "inventory.json"
    result = build_manifest(tmp_path, output)
    assert len(result["files"]) == 1
    assert result["files"][0]["sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert result["files"][0]["size_bytes"] == source.stat().st_size
    with pytest.raises(FileExistsError):
        build_manifest(tmp_path, output)


def test_claim_ledger_uses_requested_status_vocabulary() -> None:
    import csv

    allowed = {
        "VERIFIED",
        "PARTIALLY VERIFIED",
        "UNVERIFIED",
        "UNSUPPORTED",
        "CONTRADICTED",
        "REQUIRES DATA",
        "REQUIRES EXPERIMENT",
    }
    rows = list(csv.DictReader((ROOT / "research/claim_ledger.csv").open()))
    assert len(rows) >= 24
    assert {r["verification_status"] for r in rows} <= allowed


def test_source_auc_table_matches_recorded_metrics(tmp_path: Path) -> None:
    import csv

    from build_extension_figures import write_tables

    transfer = json.loads((ROOT / "research/results/transfer.json").read_text())
    write_tables(ROOT / "research/results", tmp_path)
    rows = list(csv.DictReader((tmp_path / "all_metrics.csv").open()))
    for cutoff, evaluation in transfer["measured_source_challenge"][
        "threshold_sensitivity_uM"
    ].items():
        for method, metrics in (evaluation["metrics"] or {}).items():
            row = next(
                r
                for r in rows
                if r["design"] == f"measured_source_below_{cutoff}_uM" and r["method"] == method
            )
            assert float(row["auc"]) == metrics["auc"]


def test_compressed_authenticated_sources_preserve_original_hashes() -> None:
    import csv
    import gzip

    for row in csv.DictReader((ROOT / "research/reference_structures.csv").open()):
        name = row["source_url"].rsplit("/", 1)[-1]
        content = gzip.decompress((ROOT / "research/reference_sources" / f"{name}.gz").read_bytes())
        assert hashlib.sha256(content).hexdigest() == row["source_sha256"]


def test_paired_table_matches_all_eligible_recorded_endpoints_and_scores() -> None:
    from build_research_documents import read_blocks

    result = json.loads((ROOT / "research/results/selectivity.json").read_text())
    text = (ROOT / "research/manuscript.md").read_text()
    table = text.split("<!-- path-b:paired:start -->")[1].split("<!-- path-b:paired:end -->")[0]
    cells = read_blocks(table)[0][1][1:]
    rows = {r["source_compound"]: r for r in result["rows"] if r["source_pair_has_both_endpoints"]}
    assert [c[0] for c in cells] == list(rows)
    assert len(cells) == 8
    for name, nudt5, nudt14, ratio in cells:
        row = rows[name]
        for target, printed in (("NUDT5", nudt5), ("NUDT14", nudt14)):
            endpoint = row["endpoints"][target]
            if endpoint["status"] == "right_censored":
                assert printed == f">{endpoint['bound']:g}"
            else:
                assert printed == endpoint["table1_text"]
                mean, sd = map(float, printed.split("±"))
                assert mean == endpoint["reported_mean"]
                assert sd == endpoint["reported_sd"]
        value = row["ratio"]
        if value["status"] == "double_censored":
            assert ratio == "No finite bound"
            assert value["point"] is None and value["bound"] is None
        elif value["status"] == "upper_bound":
            assert ratio == f"<{value['bound']:.3g}"
        else:
            assert ratio == f"{value['point']:.3g}"


def test_new_cli_registration_includes_both_modules() -> None:
    import tomllib

    config = tomllib.loads((ROOT / "pyproject.toml").read_text())
    for name in ("selectivity", "assay"):
        assert config["project"]["scripts"][f"nudt5-{name}"] == f"{name}:main"
        assert name in config["tool"]["setuptools"]["py-modules"]


def test_portable_inventory_excludes_canonical_previous_inventory(tmp_path: Path) -> None:
    research = tmp_path / "research"
    research.mkdir()
    (research / "release_manifest.json").write_text('{"previous": true}')
    (research / "evidence.md").write_text("Evidence boundaries")
    first = build_manifest(tmp_path, tmp_path / "inventory-a.json")
    second = build_manifest(tmp_path, tmp_path / "inventory-b.json")
    assert first == second
    assert [row["filename"] for row in first["files"]] == ["research/evidence.md"]


def test_handoff_source_quotes_match_archived_primary_xml() -> None:
    import xml.etree.ElementTree as ET

    base = ROOT / "research/structure_comparison"
    ledger = json.loads((base / "handoff_sources.json").read_text())
    source = (base / ledger["source_path"]).read_bytes()
    assert hashlib.sha256(source).hexdigest() == ledger["source_sha256"]
    tree = ET.fromstring(source)
    assert len(ledger["excerpts"]) == 5
    for excerpt in ledger["excerpts"]:
        node = tree.find("." + excerpt["xpath"])
        assert node is not None
        assert "".join(node.itertext()) == excerpt["text_exact_itertext"]
        assert excerpt["limit"] and excerpt["claim_category"] == "reported_primary_methods"


def test_handoff_arg51_and_manuscript_track_recorded_geometry() -> None:
    from collections import Counter

    base = ROOT / "research/structure_comparison"
    result = json.loads((base / "results/observed_proximity.json").read_text())
    rows = result["residue_proximity"]
    assert Counter(r["geometry_status"] for r in rows) == {
        "observed": 1564,
        "partial_observed": 58,
        "refused": 108,
    }
    assert len(result["atom_pairs_within_5A"]) == 1278
    text = (base / "lab_handoff.md").read_text()
    for site, atom_id, name, occupancy in (("C", "306", "N", 1.0), ("D", "1775", "CD", 0.78)):
        row = next(
            r
            for r in rows
            if r["site_id"].startswith(f"8RIY:model1:{site}:")
            and r["residue_identity"]["auth_seq_id"] == "51"
            and r["within_4_0A"]
        )
        atom = row["minimum_witness_pairs"][0]["protein_atom"]
        assert (atom["atom_site_id"], atom["label_atom_id"], atom["occupancy"]) == (
            atom_id,
            name,
            occupancy,
        )
        assert f"{row['observed_min_distance_A']:.3f}" in text
        assert f"#{atom_id}" in text
    paper = (ROOT / "research/manuscript.md").read_text()
    abstract = paper.split("## Abstract")[1].split("## 1.")[0]
    assert len(abstract.split()) <= 245
    assert "three questions" not in paper
    assert paper.index("### 3.1 All-site") < paper.index("### 3.2 Paired")
    assert paper.index("### 3.2 Paired") < paper.index("### 3.3 Source-label")
    assert "guanidinium" in paper and "historical" in paper.lower()


def test_handoff_controls_remain_unmeasured_and_bounded() -> None:
    import csv

    base = ROOT / "research/structure_comparison"
    rows = list(csv.DictReader((base / "hypotheses_controls.csv").open()))
    assert len(rows) == 5 and len({r["id"] for r in rows}) == 5
    assert {r["measurement_status"] for r in rows} == {"unmeasured"}
    for row in rows:
        assert all(
            row[k]
            for k in (
                "controls",
                "falsifying_or_challenging_outcome_if_qualified",
                "inconclusive_conditions",
                "unresolved_gates",
                "source_refs",
            )
        )
    text = (base / "lab_handoff.md").read_text()
    assert "../assay/PROTOCOL.md" in text and "UNMATCHED" in text
    assert "not physical-experiment-ready" in text


def test_release_inventory_includes_structural_dependency_locks(tmp_path: Path) -> None:
    report = build_manifest(ROOT, tmp_path / "release.json")
    rows = {row["filename"]: row for row in report["files"]}
    assert "requirements-structure.lock" in rows and "requirements-structure.txt" in rows
    assert (
        "proximity only"
        in rows["research/structure_comparison/results/observed_proximity.json"][
            "reproducibility_status"
        ]
    )


def test_manuscript_defines_requested_abbreviations_and_uses_micromolar_symbol() -> None:
    import re

    text = (ROOT / "research/manuscript.md").read_text()
    abstract, body = text.split("## Abstract")[1].split("## 1.", 1)
    assert len(abstract.split()) < 250
    definitions = {
        "NUDT5": "Nudix hydrolase 5 (NUDT5)",
        "RSCC": "real-space correlation coefficient (RSCC)",
        "RSR": "real-space R (RSR)",
        "IC50": "half-maximal inhibitory concentration (IC50)",
    }
    for abbreviation, definition in definitions.items():
        assert definition in abstract
        assert abstract.index(abbreviation) == abstract.index(definition) + definition.index(
            abbreviation
        )
    definitions = {
        "ROC-AUC": "area under the receiver operating characteristic curve (ROC-AUC)",
        "TPSA": "topological polar surface area (TPSA)",
        "HBD": "hydrogen-bond donor count (HBD)",
        "HBA": "hydrogen-bond acceptor count (HBA)",
        "Fsp3": "fraction of sp3-hybridized carbon atoms (Fsp3)",
        "ADPr": "adenosine diphosphate ribose (ADPr)",
        "WT": "wild-type (WT)",
        "SPR": "surface plasmon resonance (SPR)",
    }
    for abbreviation, definition in definitions.items():
        assert definition in body
        assert body.index(abbreviation) == body.index(definition) + definition.index(abbreviation)
    assert not re.search(r"\buM\b", text)
