"""Check archived evidence integrity and exact source locators, not biological truth."""

from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path
from xml.etree import ElementTree

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).parent


def test_archived_source_hashes_and_nguyen_attempt_boundary() -> None:
    sources = json.loads((HERE / "source_register.json").read_text())["sources"]
    for source in sources:
        if source.get("local_path"):
            raw = (ROOT / source["local_path"]).read_bytes()
            assert hashlib.sha256(raw).hexdigest() == source["stored_sha256"]
            if source.get("decoded_sha256"):
                assert hashlib.sha256(gzip.decompress(raw)).hexdigest() == source["decoded_sha256"]
    attempts = [s for s in sources if s["id"] == "nguyen_fulltext"]
    assert len(attempts) == 1
    assert attempts[0]["status"] == 403


def test_exact_claim_excerpts_resolve_in_retained_sources() -> None:
    sources = {
        s["id"]: s for s in json.loads((HERE / "source_register.json").read_text())["sources"]
    }
    claims = json.loads((HERE / "source_claims.json").read_text())["claims"]
    assert len({c["id"] for c in claims}) == len(claims)
    for claim in claims:
        source = sources[claim["source_id"]]
        raw = (ROOT / source["local_path"]).read_bytes()
        if source["local_path"].endswith(".gz"):
            raw = gzip.decompress(raw)
        if claim["locator"].startswith("/"):
            root = ElementTree.fromstring(raw)
            locator = claim["locator"].replace("/article/", "./", 1)
            if locator.startswith("//"):
                locator = "." + locator
            nodes = root.findall(locator)
            assert len(nodes) == 1
            text = " ".join("".join(nodes[0].itertext()).split())
        elif claim["source_id"] == "nguyen_metadata":
            text = json.loads(raw)["resultList"]["result"][0]["abstractText"]
        else:
            text = raw.decode()
        assert claim["quote"] in text


def test_journal_requirements_are_not_certified_from_an_inaccessible_guide() -> None:
    checks = json.loads((HERE / "journal_checks.json").read_text())
    assert checks["guide_access_status"] == 403
    assert checks["compliance_certified"] is False
    assert {c["topic"] for c in checks["checks"]} == {
        "article_fit",
        "structure_and_word_limits",
        "declarations",
        "data_and_code",
        "figures",
        "AI_disclosure",
        "submission",
    }
    assert all(c["unchecked_requirements"] for c in checks["checks"])
