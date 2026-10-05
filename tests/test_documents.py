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
