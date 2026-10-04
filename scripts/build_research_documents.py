"""Render the authored Markdown revision and recorded diagnostics into PDF/DOCX."""

from __future__ import annotations

import argparse
import html
import importlib
import json
import re
from pathlib import Path
from typing import Any

from docx import Document
from docx.shared import Inches, Pt

ROOT = Path(__file__).resolve().parents[1]


def read_blocks(text: str) -> list[tuple[str, Any]]:
    lines = text.splitlines()
    blocks: list[tuple[str, Any]] = []
    paragraph: list[str] = []
    index = 0
    while index < len(lines):
        line = lines[index].strip()
        if not line:
            if paragraph:
                blocks.append(("paragraph", " ".join(paragraph)))
                paragraph = []
        elif line.startswith("#") or line.startswith("|"):
            if paragraph:
                blocks.append(("paragraph", " ".join(paragraph)))
                paragraph = []
            if line.startswith("#"):
                blocks.append(
                    ("heading", (len(line) - len(line.lstrip("#")), line.lstrip("#").strip()))
                )
            else:
                rows = []
                while index < len(lines) and lines[index].strip().startswith("|"):
                    cells = [cell.strip() for cell in lines[index].strip().strip("|").split("|")]
                    if not all(re.fullmatch(r":?-+:?", cell) for cell in cells):
                        rows.append(cells)
                    index += 1
                blocks.append(("table", rows))
                continue
        else:
            paragraph.append(line)
        index += 1
    if paragraph:
        blocks.append(("paragraph", " ".join(paragraph)))
    return blocks


def plain(text: str) -> str:
    return text.replace("**", "").replace("`", "").replace("*", "")


def diagnostic_figure(results: Path, output: Path) -> Path:
    mpl = importlib.import_module("matplotlib")
    mpl.use("Agg")
    plt = importlib.import_module("matplotlib.pyplot")
    data = json.loads((results / "benchmark.json").read_text())
    methods = ["Property_LR", "SVM_RBF", "RF", "Equal_mean", "Nearest_active", "GBT"]
    labels = ["Property LR", "RBF-SVM", "RF", "Equal mean", "Nearest active", "GBT"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.7), layout="constrained")
    for index, split in enumerate(["molecule", "scaffold", "series"]):
        points = [i + (index - 1) * 0.24 for i in range(len(methods))]
        axes[0].bar(
            points,
            [data[split]["metrics"][method]["auc"] for method in methods],
            width=0.24,
            label=split,
            color=["#b2becd", "#527893", "#bf755e"][index],
        )
    axes[0].set_xticks(range(len(labels)), labels, rotation=40, ha="right")
    axes[0].set_ylim(0, 1.05)
    axes[0].axhline(0.5, color="#888888", linestyle=":", linewidth=1)
    axes[0].set_ylabel("Pooled out-of-fold ROC-AUC")
    axes[0].set_title("A  Seed 42; small, unverified-label dataset", fontsize=10)
    axes[0].legend(frameon=False, fontsize=8)
    for method, colour in [("Property_LR", "#527893"), ("Equal_mean", "#bf755e")]:
        seed_rows = [row for row in data["seed_sensitivity"] if row["split"] == "series"]
        axes[1].plot(
            [row["seed"] for row in seed_rows],
            [row["metrics"][method]["auc"] for row in seed_rows],
            marker="o",
            color=colour,
            label=method,
        )
    axes[1].set_ylim(0, 1.05)
    axes[1].axhline(0.5, color="#888888", linestyle=":", linewidth=1)
    axes[1].set_xlabel("Seed; no best-seed selection")
    axes[1].set_ylabel("Pooled series-holdout ROC-AUC")
    axes[1].set_title("B  Sensitivity to decoy partitions", fontsize=10)
    axes[1].set_xticks(sorted({row["seed"] for row in data["seed_sensitivity"]}))
    axes[1].legend(frameon=False, fontsize=8, loc="center right")
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    path = output / "diagnostics.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def build(manuscript: Path, results: Path, output: Path) -> tuple[Path, Path]:
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError("Document output must be a new or empty directory")
    blocks = read_blocks(manuscript.read_text())
    if not blocks:
        raise ValueError("Manuscript is empty")
    if not (results / "benchmark.json").is_file():
        raise ValueError("Recorded benchmark.json is required; results are never fabricated")
    if not (results / "controls.json").is_file() or not (results / "transfer.json").is_file():
        raise ValueError("Recorded controls.json and transfer.json are required")
    output.mkdir(parents=True, exist_ok=True)
    figure = diagnostic_figure(results, output)
    extension_figures = importlib.import_module("build_extension_figures")
    extension = extension_figures.build_figures(results, output)
    extension_figures.write_tables(results, output)
    flow = importlib.import_module("reportlab.platypus")
    styles_module = importlib.import_module("reportlab.lib.styles")
    colours = importlib.import_module("reportlab.lib.colors")
    pdfmetrics = importlib.import_module("reportlab.pdfbase.pdfmetrics")
    fonts = importlib.import_module("reportlab.pdfbase.ttfonts")
    font_manager = importlib.import_module("matplotlib.font_manager")
    pdfmetrics.registerFont(fonts.TTFont("Research", font_manager.findfont("DejaVu Sans")))
    pdfmetrics.registerFont(
        fonts.TTFont(
            "ResearchBold",
            font_manager.findfont(font_manager.FontProperties(family="DejaVu Sans", weight="bold")),
        )
    )
    styles = styles_module.getSampleStyleSheet()
    for name in ["BodyText", "Normal", "Title", "Heading1", "Heading2", "Heading3"]:
        styles[name].fontName = (
            "ResearchBold" if name.startswith("Heading") or name == "Title" else "Research"
        )
        styles[name].fontSize = 9 if name in ["BodyText", "Normal"] else styles[name].fontSize
        styles[name].leading = 13 if name in ["BodyText", "Normal"] else styles[name].leading
        styles[name].spaceAfter = 7
    table_style = styles_module.ParagraphStyle(
        "TableBody", parent=styles["BodyText"], fontSize=7, leading=10
    )
    doc = Document()
    doc.sections[0].left_margin = Inches(1)
    doc.sections[0].right_margin = Inches(1)
    doc.styles["Normal"].font.name = "Calibri"
    doc.styles["Normal"].font.size = Pt(10)
    story = []
    for kind, value in blocks:
        if kind == "heading":
            level, text = value
            doc.add_heading(plain(text), level=min(level - 1, 3))
            story.append(
                flow.Paragraph(
                    html.escape(plain(text)),
                    styles["Title" if level == 1 else f"Heading{min(level - 1, 3)}"],
                )
            )
        elif kind == "paragraph":
            doc.add_paragraph(plain(value))
            story.append(flow.Paragraph(html.escape(plain(value)), styles["BodyText"]))
        elif kind == "table":
            table = doc.add_table(rows=1, cols=len(value[0]))
            table.style = "Light Shading Accent 1"
            for index, cell in enumerate(value[0]):
                table.rows[0].cells[index].text = plain(cell)
            for row in value[1:]:
                for cell, text in zip(table.add_row().cells, row, strict=True):
                    cell.text = plain(text)
            pdf_rows = [
                [flow.Paragraph(html.escape(plain(cell)), table_style) for cell in row]
                for row in value
            ]
            pdf_table = flow.Table(
                pdf_rows,
                colWidths=[168] + [312 / (len(value[0]) - 1)] * (len(value[0]) - 1),
                repeatRows=1,
            )
            pdf_table.setStyle(
                flow.TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, 0), colours.HexColor("#e7edf2")),
                        ("VALIGN", (0, 0), (-1, -1), "TOP"),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
                        ("LINEBELOW", (0, 0), (-1, 0), 0.5, colours.HexColor("#8798aa")),
                    ]
                )
            )
            story.extend([pdf_table, flow.Spacer(1, 10)])
    caption = (
        "Figure 1. Newly computed diagnostics, not historical or prospective validation. "
        "A: seed-42 pooled scores; series pooling mixes differently trained models. "
        "B: all five specified seeds. No interval denotes population uncertainty. "
        "Descriptor performance is compatible with confounding but does not establish "
        "its causal contribution."
    )
    doc.add_heading("Diagnostic figure", level=1)
    doc.add_picture(str(figure), width=Inches(6.4))
    doc.add_paragraph(caption)
    story.extend(
        [
            flow.PageBreak(),
            flow.Paragraph("Diagnostic figure", styles["Heading1"]),
            flow.Image(str(figure), width=490, height=210),
            flow.Paragraph(caption, styles["BodyText"]),
        ]
    )
    for index, (path, figure_caption) in enumerate(extension, start=2):
        heading = f"Figure {index}"
        doc.add_heading(heading, level=1)
        doc.add_picture(str(path), width=Inches(6.4))
        doc.add_paragraph(figure_caption)
        image = flow.Image(str(path))
        image.drawHeight *= 490 / image.drawWidth
        image.drawWidth = 490
        story.extend(
            [
                flow.PageBreak(),
                flow.Paragraph(heading, styles["Heading1"]),
                image,
                flow.Paragraph(figure_caption, styles["BodyText"]),
            ]
        )
    pdf = output / "NUDT5_evidence_bounded_revision.pdf"
    word = output / "NUDT5_evidence_bounded_revision.docx"
    doc.save(str(word))
    template = flow.SimpleDocTemplate(
        str(pdf),
        pagesize=(612, 792),
        leftMargin=54,
        rightMargin=54,
        topMargin=48,
        bottomMargin=48,
        title="NUDT5 reproducibility and chemical-identity controls",
    )

    def footer(canvas: Any, document: Any) -> None:
        canvas.setFont("Research", 7)
        canvas.setFillColor(colours.HexColor("#64748b"))
        canvas.drawString(54, 28, "Author-review revision | No new biological experiments")
        canvas.drawRightString(558, 28, str(document.page))

    template.build(story, onFirstPage=footer, onLaterPages=footer)
    return pdf, word


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manuscript", type=Path, default=ROOT / "research/manuscript.md")
    parser.add_argument("--results", type=Path, default=ROOT / "research/results")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    for path in build(args.manuscript, args.results, args.output):
        print(path)


if __name__ == "__main__":
    main()
