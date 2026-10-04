"""Hash the reproducible release without self-referencing the manifest."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Any

from pipeline import ROOT, write_json

DIRECTORIES = ("scripts", "tests", "research", ".github")
FILES = (
    "README.md",
    "compounds.csv",
    "final_hits.csv",
    "pyproject.toml",
    "requirements.txt",
    "requirements-dev.txt",
    "requirements.lock",
    ".gitignore",
)


def build_manifest(root: Path, output: Path) -> dict[str, Any]:
    paths = {root / name for name in FILES if (root / name).is_file()}
    paths.update(
        path
        for name in DIRECTORIES
        for path in (root / name).rglob("*")
        if path.is_file()
        and "__pycache__" not in path.parts
        and not any(part.endswith(".egg-info") for part in path.relative_to(root).parts)
    )
    rows = []
    for path in sorted(paths):
        if path.resolve() == output.resolve():
            continue
        relative = path.relative_to(root).as_posix()
        if relative in ("compounds.csv", "final_hits.csv"):
            purpose, provenance = (
                "Immutable original input",
                "original Git release and author upload",
            )
            status = "preserved; biological labels not independently authenticated"
        elif relative.startswith("research/results/"):
            purpose, provenance = (
                "Recorded results or source/run provenance",
                "named run manifests and source ledgers",
            )
            status = "reproduced diagnostics where executable; source metadata is evidence only"
        elif relative.startswith("research/figures/"):
            purpose, provenance = (
                "Generated figure or table",
                "recorded JSON via figure/document builders",
            )
            status = "regenerable, not experimental validation"
        elif relative.startswith("research/"):
            purpose, provenance = (
                "Manuscript, ledger or evidence report",
                "source citations and review records within file",
            )
            status = "auditable; limitations and access gaps retained"
        else:
            purpose, provenance = (
                "Executable reproduction, tests or environment",
                "this revision; inspect Git history",
            )
            status = "covered by release quality checks; not proof of biological validity"
        rows.append(
            {
                "filename": relative,
                "purpose": purpose,
                "provenance": provenance,
                "size_bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "manuscript_dependency": "Methods/results, evidence traceability or reproduction",
                "reproducibility_status": status,
            }
        )
    report = {
        "hash_algorithm": "SHA-256",
        "scope": (
            "Repository research release; excludes this manifest, Git internals, "
            "environments and caches. Original attachments separately indexed "
            "in original_inventory.json."
        ),
        "files": rows,
    }
    write_json(output, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    build_manifest(args.root, args.output)


if __name__ == "__main__":
    main()
