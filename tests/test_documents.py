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
