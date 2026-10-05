import json
from pathlib import Path
from zipfile import ZipFile

import pytest
from build_research_documents import ROOT, build, plain, read_blocks


def test_markdown_blocks_preserve_headings_paragraphs_and_table() -> None:
    result = read_blocks("# Title\n\nFirst line\ncontinued.\n\n| A | B |\n|---|---:|\n| 1 | 2 |\n")
    assert result == [
        ("heading", (1, "Title")),
        ("paragraph", "First line continued."),
        ("table", [["A", "B"], ["1", "2"]]),
    ]
    assert plain("**bold** and `code`") == "bold and code"
    assert plain("fractional (*) and partial (†)") == "fractional (*) and partial (†)"


def test_build_pdf_and_word_from_recorded_results(tmp_path: Path) -> None:
    manuscript = tmp_path / "paper.md"
    manuscript.write_text(
        "# Evidence review\n\nNo clinical efficacy claimed.\n\n"
        "| Metric | Value |\n|---|---|\n| AUC | 0.5 |\n"
    )
    output = tmp_path / "documents"
    pdf, word = build(manuscript, ROOT / "research/results", output)
    assert pdf.read_bytes().startswith(b"%PDF")
    with ZipFile(word) as archive:
        xml = archive.read("word/document.xml").decode()
        assert "No clinical efficacy claimed." in xml
        assert "Evidence review" in xml
        assert any(path.startswith("word/media/") for path in archive.namelist())
    assert (output / "diagnostics.png").stat().st_size > 10000
    for name in ("property_controls", "reliability_domain", "source_transfer"):
        assert (output / f"{name}.png").stat().st_size > 5000
        assert (output / f"{name}.svg").stat().st_size > 1000
    assert (output / "all_metrics.csv").stat().st_size > 1000
    assert (output / "candidate_axes.csv").stat().st_size > 500
    assert "arbitrary" in (output / "supplementary_results.md").read_text()
    with pytest.raises(ValueError, match="new or empty"):
        build(manuscript, ROOT / "research/results", output)


def test_empty_manuscript_and_missing_results_fail(tmp_path: Path) -> None:
    manuscript = tmp_path / "empty.md"
    manuscript.write_text("")
    with pytest.raises(ValueError, match="empty"):
        build(manuscript, ROOT / "research/results", tmp_path / "out")
    manuscript.write_text("# Title")
    with pytest.raises(ValueError, match="benchmark.json"):
        build(manuscript, tmp_path, tmp_path / "out")
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize(
    ("cutoff", "n", "positives"),
    [(c, 10, p) for c in ("1.0", "10.0", "50.0") for p in (0, 10)] + [("50.0", 0, 0)],
)
def test_one_class_cutoffs_remain_explicit_in_documents(
    tmp_path: Path, cutoff: str, n: int, positives: int
) -> None:
    import csv
    import json
    import shutil

    results = tmp_path / "results"
    results.mkdir()
    for name in ("benchmark.json", "controls.json"):
        shutil.copyfile(ROOT / "research/results" / name, results / name)
    transfer = json.loads((ROOT / "research/results/transfer.json").read_text())
    for source in (transfer, transfer["reference_sensitivity"]):
        evaluations = source["measured_source_challenge"]["threshold_sensitivity_uM"]
        for key in evaluations if n == 0 else [cutoff]:
            evaluations[key] = {"n": n, "positives": positives, "metrics": None}
    (results / "transfer.json").write_text(json.dumps(transfer))
    manuscript = tmp_path / "paper.md"
    manuscript.write_text("# One-class diagnostic\n\nUndefined ROC-AUC is not zero.\n")
    output = tmp_path / "documents"
    pdf, word = build(manuscript, results, output)
    assert pdf.read_bytes().startswith(b"%PDF")
    assert word.is_file()
    status = list(csv.DictReader((output / "source_cutoff_status.csv").open()))
    affected = [row for row in status if row["design"].endswith(f"_{cutoff}_uM")]
    assert len(affected) == 2
    for row in affected:
        assert row["status"].startswith("infeasible")
        assert int(row["n"]) == n
        assert int(row["positives"]) == positives
        assert int(row["negatives"]) == n - positives
    metrics = list(csv.DictReader((output / "all_metrics.csv").open()))
    assert not any(row["design"] in {r["design"] for r in affected} for row in metrics)
    assert "infeasible: both classes required" in (output / "supplementary_results.md").read_text()
    if cutoff == "50.0":
        assert "Source ROC-AUC unavailable" in (output / "source_transfer.svg").read_text()


def test_missing_selectivity_is_explicit_or_required(tmp_path: Path) -> None:
    import json
    import shutil

    results = tmp_path / "results"
    results.mkdir()
    for name in ("benchmark.json", "controls.json", "transfer.json"):
        shutil.copyfile(ROOT / "research/results" / name, results / name)
    manuscript = tmp_path / "paper.md"
    manuscript.write_text("# Legacy diagnostic\n")
    required = tmp_path / "required"
    with pytest.raises(ValueError, match="selectivity.json"):
        build(manuscript, results, required, require_selectivity=True)
    assert not required.exists()
    _, word = build(manuscript, results, tmp_path / "legacy")
    with ZipFile(word) as archive:
        assert "Paired-target results unavailable" in archive.read("word/document.xml").decode()
    record = json.loads((word.parent / "documents-manifest.json").read_text())
    assert record["selectivity_status"] == "unavailable"
    assert not list(word.parent.glob("selectivity_*.png"))


@pytest.mark.parametrize("names", [[], ["2"], ["1"]])
def test_software_only_empty_and_single_pair_documents(tmp_path: Path, names: list[str]) -> None:
    import hashlib
    import shutil

    import selectivity
    from test_selectivity import software_subset

    results = tmp_path / "results"
    results.mkdir()
    for name in ("benchmark.json", "controls.json", "transfer.json"):
        shutil.copyfile(ROOT / "research/results" / name, results / name)
    result = software_subset(
        selectivity.read_json(ROOT / "research/results/selectivity.json"), names
    )
    content = selectivity.json_bytes(result)
    (results / "selectivity.json").write_bytes(content)
    (results / "selectivity-manifest.json").write_bytes(
        selectivity.json_bytes(
            {
                "artifacts": {
                    "selectivity.json": {
                        "sha256": hashlib.sha256(content).hexdigest(),
                        "size_bytes": len(content),
                    }
                }
            }
        )
    )
    manuscript = tmp_path / "software.md"
    manuscript.write_text("# SOFTWARE TESTS ONLY\n\nNot biological Results.\n")
    pdf, word = build(manuscript, results, tmp_path / "documents", require_selectivity=True)
    assert pdf.read_bytes().startswith(b"%PDF")
    with ZipFile(word) as archive:
        text = archive.read("word/document.xml").decode()
        assert "SOFTWARE TESTS ONLY" in text
        assert "No eligible pairs" in text if names != ["1"] else "n=1 nonoverlap" in text
        assert len([p for p in archive.namelist() if p.startswith("word/media/")]) == 6
    assert "SOFTWARE TESTS ONLY" in (word.parent / "selectivity.md").read_text()
    for scenario in ("historical", "reference_sensitivity"):
        for extension in ("png", "pdf", "svg"):
            assert (word.parent / f"selectivity_{scenario}.{extension}").is_file()


@pytest.mark.parametrize("defect", ["missing_result", "missing_manifest", "stale"])
def test_incomplete_or_stale_selectivity_refuses_all_documents(tmp_path: Path, defect: str) -> None:
    import shutil

    results = tmp_path / "results"
    results.mkdir()
    for name in (
        "benchmark.json",
        "controls.json",
        "transfer.json",
        "selectivity.json",
        "selectivity-manifest.json",
    ):
        shutil.copyfile(ROOT / "research/results" / name, results / name)
    if defect == "stale":
        with (results / "selectivity.json").open("a") as stream:
            stream.write(" ")
    else:
        (
            results
            / ("selectivity.json" if defect == "missing_result" else "selectivity-manifest.json")
        ).unlink()
    output = tmp_path / "documents"
    with pytest.raises(ValueError, match="required|hash mismatch"):
        build(ROOT / "research/manuscript.md", results, output)
    assert not output.exists()


def test_render_failure_does_not_publish_partial_documents(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import build_research_documents as documents

    def fail(*args: object) -> None:
        raise RuntimeError("Rendering failed")

    monkeypatch.setattr(documents, "diagnostic_figure", fail)
    output = tmp_path / "documents"
    output.mkdir()
    with pytest.raises(RuntimeError, match="Rendering failed"):
        build(ROOT / "research/manuscript.md", ROOT / "research/results", output)
    assert list(output.iterdir()) == []
    assert not list(tmp_path.glob(".nudt5-documents-*"))


STRUCTURE = ROOT / "research/structure_comparison"
RECORDED = STRUCTURE / "results/observed_proximity.json"
RECORDED_MANIFEST = STRUCTURE / "results/derived/derived_manifest.json"


def test_structural_documents_and_supplement_are_recorded(tmp_path: Path) -> None:
    import hashlib

    manuscript = tmp_path / "paper.md"
    manuscript.write_text("# SOFTWARE INTEGRATION TEST\n\nNo new biological result.\n")
    pdf, word = build(
        manuscript,
        ROOT / "research/results",
        tmp_path / "output",
        structure_input=RECORDED,
        structure_manifest=RECORDED_MANIFEST,
        require_structure=True,
    )
    assert pdf.read_bytes().startswith(b"%PDF")
    with ZipFile(word) as archive:
        text = archive.read("word/document.xml").decode()
        assert "Figure 7A" in text and "Figure 7B" in text
        assert "binding measurements" in text
        assert len([p for p in archive.namelist() if p.startswith("word/media/")]) == 8
    assert (
        "Observed-coordinate supplement" in (word.parent / "supplementary_results.md").read_text()
    )
    for name in ("lab_handoff.md", "hypotheses_controls.csv", "handoff_sources.json"):
        assert (word.parent / name).read_bytes() == (STRUCTURE / name).read_bytes()
    for name in (
        "residue_proximity.csv",
        "atom_pairs_within_5A.csv",
        "radius_sensitivity.csv",
        "observed_proximity_map.png",
        "observed_proximity_map.svg",
        "observed_proximity_map.pdf",
    ):
        assert (word.parent / name).read_bytes() == (
            STRUCTURE / "results/derived" / name
        ).read_bytes()
    record = json.loads((word.parent / "documents-manifest.json").read_text())
    assert record["structure_status"] == "recorded"
    required_inputs = [
        STRUCTURE / "runtime_input_manifest.json",
        STRUCTURE / "geometry_contract.json",
    ]
    required_inputs.extend(
        ROOT / entry["path"]
        for entry in json.loads((STRUCTURE / "runtime_input_manifest.json").read_text())["inputs"]
    )
    for path in required_inputs:
        assert record["files"][str(path)] == hashlib.sha256(path.read_bytes()).hexdigest()
    for name, entry in record["artifacts"].items():
        assert entry["sha256"] == hashlib.sha256((word.parent / name).read_bytes()).hexdigest()


@pytest.mark.parametrize(
    "defect",
    [
        "missing_input",
        "missing_manifest",
        "both_missing",
        "stale",
        "invalid_json",
        "empty_object",
        "empty_rows",
        "empty_sites",
        "empty_structures",
        "manifest_array",
        "manifest_missing_hash",
        "truncated_inventory",
        "symlink",
    ],
)
def test_bad_structural_evidence_never_publishes(tmp_path: Path, defect: str) -> None:
    import hashlib

    source, manifest = tmp_path / "structure.json", tmp_path / "manifest.json"
    data = RECORDED.read_bytes()
    if defect == "invalid_json":
        data = b"{broken"
    elif defect == "empty_object":
        data = b"{}"
    elif defect.startswith("empty_") or defect == "truncated_inventory":
        result = json.loads(data)
        if defect == "truncated_inventory":
            result["residue_proximity"].pop()
        else:
            key = {
                "empty_rows": "residue_proximity",
                "empty_sites": "sites",
                "empty_structures": "structures",
            }[defect]
            result[key] = []
        data = json.dumps(result).encode()
    source.write_bytes(data)
    manifest.write_text(json.dumps({"input_sha256": hashlib.sha256(data).hexdigest()}))
    if defect in ("missing_input", "both_missing", "symlink"):
        source.unlink()
        if defect == "symlink":
            source.symlink_to(RECORDED)
    if defect in ("missing_manifest", "both_missing"):
        manifest.unlink()
    if defect == "stale":
        source.write_bytes(data + b" ")
    if defect.startswith("manifest_"):
        manifest.write_text("[]" if defect == "manifest_array" else "{}")
    output = tmp_path / "output"
    # Legacy opt-in must never excuse malformed or partially present input.
    with pytest.raises(ValueError):
        build(
            ROOT / "research/manuscript.md",
            ROOT / "research/results",
            output,
            structure_input=source,
            structure_manifest=manifest,
            require_structure=defect == "both_missing",
        )
    assert not output.exists()
    assert not list(tmp_path.glob(".nudt5-documents-*"))


def test_absent_structural_evidence_is_explicit(tmp_path: Path) -> None:
    manuscript = tmp_path / "legacy.md"
    manuscript.write_text("# Legacy document\n")
    _, word = build(
        manuscript,
        ROOT / "research/results",
        tmp_path / "legacy-output",
        structure_input=tmp_path / "absent.json",
        structure_manifest=tmp_path / "absent-manifest.json",
    )
    with ZipFile(word) as archive:
        assert "Structural results unavailable" in archive.read("word/document.xml").decode()
    assert not (word.parent / "observed_proximity_map.png").exists()
    assert not (word.parent / "lab_handoff.md").exists()
    assert (
        json.loads((word.parent / "documents-manifest.json").read_text())["structure_status"]
        == "unavailable"
    )


def test_failed_structural_render_does_not_publish(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import build_structure_comparison_figures as figures

    def fail(*args: object) -> None:
        raise RuntimeError("Structural renderer failed")

    monkeypatch.setattr(figures, "render", fail)
    output = tmp_path / "output"
    with pytest.raises(RuntimeError, match="Structural renderer failed"):
        build(
            ROOT / "research/manuscript.md",
            ROOT / "research/results",
            output,
            structure_input=RECORDED,
            structure_manifest=RECORDED_MANIFEST,
        )
    assert not output.exists()
    assert not list(tmp_path.glob(".nudt5-documents-*"))


@pytest.mark.parametrize(
    "defect",
    [
        "drop_site",
        "witness_ligand_component",
        "witness_protein_identity",
        "empty_pairs",
        "swap_targets",
    ],
)
@pytest.mark.parametrize("allow_missing", [False, True])
def test_document_cli_rejects_review_bypasses_outside_checkout(
    tmp_path: Path, defect: str, allow_missing: bool
) -> None:
    import subprocess
    import sys

    import structure_comparison as mod
    from test_structure_comparison_figures import software_corrupted_result

    source, manifest = tmp_path / "SOFTWARE_TEST_ONLY.json", tmp_path / "manifest.json"
    bad = software_corrupted_result(mod.read_json(RECORDED), defect)
    source.write_bytes(mod.json_bytes(bad))
    manifest.write_bytes(mod.json_bytes({"input_sha256": mod.digest(source.read_bytes())}))
    output = tmp_path / "unpublished"
    output.mkdir()
    command = [
        sys.executable,
        str(ROOT / "scripts/build_research_documents.py"),
        "--structure-input",
        str(source),
        "--structure-manifest",
        str(manifest),
        "--repository",
        str(ROOT),
        "--output",
        str(output),
    ]
    if allow_missing:
        command.append("--allow-missing-structure")
    run = subprocess.run(command, cwd=tmp_path, capture_output=True, text=True, check=False)
    assert run.returncode != 0
    assert "Source-package mismatch" in run.stderr
    assert list(output.iterdir()) == []
    assert not list(tmp_path.glob(".nudt5-*"))
