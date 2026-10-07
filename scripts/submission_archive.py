"""Explicit candidate-public inventory; private provenance is never globbed into a supplement."""

from __future__ import annotations

import hashlib
import io
import re
import stat
from pathlib import Path, PurePosixPath
from typing import Any
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

from author_release import read_record, valid_statement
from selectivity import json_bytes
from structure_comparison import git_state, verified_source_inventory

PUBLIC_LOCK_SHA256 = "354e2cf26b923bc21dd278f9de85de86c3f824056383a4b177352c842f4863eb"

ALLOWLIST = Path("research/submission/public_archive.json")
RIGHTS = Path("research/submission/rights_review.json")


def safe_name(name: str) -> bool:
    return bool(name) and not (
        "\\" in name
        or ":" in name
        or any(ord(c) < 32 or c in '<>"|?*' for c in name)
        or name.startswith("/")
        or any(part in ("", ".", "..") for part in name.split("/"))
        or str(PurePosixPath(name)) != name
        or any(
            part.endswith((".", " "))
            or re.fullmatch(r"(?:CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])(?:\..*)?", part, re.I)
            for part in name.split("/")
        )
    )


def source_bytes(repository: Path, name: str) -> bytes:
    if not safe_name(name):
        raise ValueError(f"Unsafe archive path: {name!r}")
    path = repository / name
    current = path
    while current != repository:
        if current.is_symlink():
            raise ValueError(f"Symlink refused in source archive: {name}")
        current = current.parent
    metadata = path.stat()
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
        raise ValueError(f"Only ordinary single-link source files permitted: {name}")
    return path.read_bytes()


def zip_payload(files: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    portable_names: set[str] = set()
    with ZipFile(buffer, "w") as archive:
        for name, payload in sorted(files.items()):
            if not safe_name(name) or name.casefold() in portable_names:
                raise ValueError(f"Unsafe archive path: {name!r}")
            portable_names.add(name.casefold())
            info = ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, payload)
    return buffer.getvalue()


def public_inventory(repository: Path) -> dict[str, Any]:
    if hashlib.sha256(source_bytes(repository, str(ALLOWLIST))).hexdigest() != PUBLIC_LOCK_SHA256:
        raise ValueError("Public source allowlist is not trusted")
    inventory = read_record(repository / ALLOWLIST)
    if inventory.get("schema_version") != 1 or not isinstance(inventory.get("files"), dict):
        raise ValueError("Invalid public archive allowlist")
    for name, entry in inventory["files"].items():
        data = source_bytes(repository, name)
        if entry.get("sha256") is not None and hashlib.sha256(data).hexdigest() != entry["sha256"]:
            raise ValueError(f"Pinned public source mismatch: {name}")
    if not (repository / ".git").exists():
        verified = verified_source_inventory(repository)
        if set(verified) != set(inventory["files"]):
            raise ValueError("Source archive inventory membership mismatch")
        # Runtime caches/installed environments are not inputs and never join the inventory.
        actual = {
            p.relative_to(repository).as_posix()
            for p in repository.rglob("*")
            if p.is_file()
            and not any(
                part
                in {
                    ".venv",
                    "venv",
                    "__pycache__",
                    ".pytest_cache",
                    ".mypy_cache",
                    ".ruff_cache",
                    "build",
                    "dist",
                }
                or part.endswith(".egg-info")
                for part in p.relative_to(repository).parts
            )
        }
        if actual != set(verified) | {"SHA256SUMS.json"}:
            raise ValueError("Unlisted file in source archive")
    return inventory


def distribution_gaps(repository: Path) -> list[str]:
    inventory = public_inventory(repository)
    ledger = read_record(repository / RIGHTS)
    gaps = []
    classes = {row["rights_class"] for row in inventory["files"].values()}
    if set(ledger.get("classes", {})) != classes:
        gaps.append("rights: every allowlisted source class needs an explicit review")
    for name in sorted(classes):
        row = ledger.get("classes", {}).get(name, {})
        if row.get("author_cleared") is not True or any(
            not valid_statement(row.get(key))
            for key in ("source", "holder", "license_version", "reuse_basis", "attribution")
        ):
            gaps.append(f"rights: {name} not cleared")
    if not valid_statement(ledger.get("project_code_license")):
        gaps.append("code license: rights holder decision required")
    items = ledger.get("items", {})
    if set(items) != set(inventory["files"]):
        gaps.append("rights: item-level membership incomplete")
    if any(
        item.get("author_cleared") is not True
        or not valid_statement(item.get("decision_and_source"))
        for item in items.values()
    ):
        gaps.append("rights: item-level decisions remain unapproved")
    access = ledger.get("anonymous_access_checked")
    if access is not True:
        gaps.append("code access: author must verify anonymous current and archived access")
    return gaps


def code_access(repository: Path) -> dict[str, Any]:
    state = git_state(repository)
    revision = state.get("revision")
    base = "https://github.com/RaunakGengiti2725/NUDT5-Inhibitor-Breast-Cancer-Discovery"
    return {
        "project": "nudt5-evidence-audit",
        "current": base,
        "immutable_revision": revision,
        "archived_version": f"{base}/archive/{revision}.zip" if revision else None,
        "archive_identifier": f"Git commit {revision}"
        if revision
        else "SHA256SUMS.json source inventory",
        "language": "Python 3.12",
        "system": "Linux verified; other platforms not verified",
        "requirements": "requirements.lock and requirements-structure.lock, hash-locked",
        "license": read_record(repository / RIGHTS).get("project_code_license"),
        "source_state": state,
        "qualification": (
            "Links are version locations, not proof of access, preservation or permission. "
            "Dirty worktree bytes differ from the named commit."
        ),
    }


def source_archives(repository: Path, generated: Path) -> tuple[bytes, bytes]:
    inventory = public_inventory(repository)
    groups: list[dict[str, bytes]] = [{}, {}]
    hashes = {}
    for name in sorted(inventory["files"]):
        payload = source_bytes(repository, name)
        hashes[name] = hashlib.sha256(payload).hexdigest()
        group = int(name.endswith((".ccp4.gz", "-sf.cif.gz")))
        groups[group][f"source/{name}"] = payload
    groups[0]["source/SHA256SUMS.json"] = json_bytes(hashes)
    for name in inventory["generated_files"]:
        payload = source_bytes(generated, name)
        groups[0][f"generated/path_b/{name}"] = payload
    groups[0]["REPRODUCE.txt"] = (
        b"AUTHOR-REVIEW ONLY: redistribution and author release are not cleared.\n"
        b"Extract both Additional_file_2.zip and Additional_file_3.zip together.\n"
        b"For the full audit rebuild also extract Private_audit_provenance.zip beside source/.\n"
        b"That private companion is NOT a journal additional file and must not be uploaded.\n"
        b"The original review-hash gate remains mandatory; no missing private review is bypassed.\n"
        b"From source/: install the README hash-locked Python 3.12 environment, then run\n"
        b".venv/bin/python scripts/build_bmc_submission.py --output ../reproduced-note\n"
        b"No fitting, download, submission or public-deposit modification occurs.\n"
        b"SHA256SUMS.json describes this exact source inventory across both ZIPs.\n"
        b"Historical manifests describe their original runs, not this checkout.\n"
        b"Hashes verify these artifacts; PDF/DOCX/SVG metadata need not regenerate byte-for-byte.\n"
        b"A rewritable inventory proves consistency, not author authenticity.\n"
        b"Private author/reviewer files, policy captures and AI reviews are excluded.\n"
        b"Absent author records are replaced only by an unapproved blank record, never approval.\n"
    )
    return zip_payload(groups[0]), zip_payload(groups[1])
