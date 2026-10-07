"""Source-derived headline values plus an explicitly reviewed prose contract."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import build_path_b_documents as path_b

LOCK_SHA256 = "9a3d0f7f230ff73f136f7592302804e879ed045563c5ac29eefd8a9c97ed6eae"


def validate_prose(repository: Path, text: str, inputs: list[dict[str, Any]]) -> None:
    lock = repository / "research/submission/prose_lock.json"
    if lock.is_symlink() or hashlib.sha256(lock.read_bytes()).hexdigest() != LOCK_SHA256:
        raise ValueError("BMC prose lock is not trusted")
    for name, expected in json.loads(lock.read_text())["files"].items():
        path = repository / name
        if (
            any(p.is_symlink() for p in (path, *path.parents))
            or hashlib.sha256(path.read_bytes()).hexdigest() != expected
        ):
            raise ValueError(f"BMC quantitative/prose drift: {name}")
    structural, paired, support, controls = inputs
    witnesses = sorted(
        [
            r
            for r in structural["residue_proximity"]
            if r["residue_identity"]["pdb_id"] == "8RIY"
            and r["residue_identity"]["auth_seq_id"] == "51"
            and r["within_5_0A"]
        ],
        key=lambda r: r["residue_identity"]["auth_asym_id"],
    )
    if len(witnesses) != 2:
        raise ValueError("Expected two local Arg51 witnesses")
    rscc = [s["report"]["metrics"]["rscc"]["value"] for s in support["sites"]]
    ratio = next(r["ratio"]["point"] for r in paired["rows"] if r["source_compound"] == "9")
    rows = {r["method"]: r for r in path_b.diagnostic_controls(controls)}
    if f"{rows['tpsa_only_lr']['auc']:.4f}" != f"{rows['hba_only_lr']['auc']:.4f}":
        raise ValueError("Descriptor equality statement no longer holds")
    values = {
        "aaa_distance": f"{witnesses[0]['observed_min_distance_A']:.3f}",
        "bbb_distance": f"{witnesses[1]['observed_min_distance_A']:.3f}",
        "rscc_range": f"{min(rscc):.3f}–{max(rscc):.3f}",
        "compound9_ratio": f"{ratio:.3f}",
        "descriptor_auc": f"{rows['tpsa_only_lr']['auc']:.4f}",
        "consensus_auc": f"{rows['Equal_mean']['auc']:.4f}",
    }
    expected_text = (repository / "research/submission/manuscript.template.md").read_text()
    for name, value in values.items():
        expected_text = expected_text.replace("{{" + name + "}}", value)
    if "{{" in expected_text:
        raise ValueError("Unresolved BMC numerical token")
    blocks = path_b.quantitative_blocks(*inputs)
    expected_text = path_b.synchronize(
        expected_text, {k: blocks[k] for k in ("sites", "paired", "controls")}, check=False
    )
    if text != expected_text:
        raise ValueError("BMC quantitative/prose drift outside protected tables")
