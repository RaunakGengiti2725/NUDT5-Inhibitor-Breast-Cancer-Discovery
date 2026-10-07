"""Validate recorded author text, not author truth, eligibility or permission."""

from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path
from typing import Any

STATEMENTS = (
    "final_byline_and_addresses",
    "ethics_approval_and_consent",
    "consent_for_publication",
    "competing_interests",
    "funding",
    "author_contributions",
    "acknowledgements",
    "ai_assistance_disclosure",
    "prior_publication_and_submission_history",
    "label_and_decoy_provenance_disposition",
    "rights_permissions_and_code_license",
)
CONFIRMATIONS = (
    "contributor_history_resolved",
    "all_authors_approve_and_accept_accountability",
    "no_concurrent_submission",
    "all_retained_material_reuse_cleared",
    "source_provenance_disposition_approved",
    "human_review_of_ai_assisted_work_complete",
    "journal_fees_and_license_terms_reviewed",
)


def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def read_record(path: Path) -> dict[str, Any]:
    record = json.loads(path.read_text(), object_pairs_hook=unique_object)
    if not isinstance(record, dict):
        raise ValueError("Author record must be an object")
    return record


def blank_record() -> dict[str, Any]:
    return {
        "status": "author_answers_required",
        "statements": dict.fromkeys(STATEMENTS),
        "confirmations": dict.fromkeys(CONFIRMATIONS, False),
    }


def valid_statement(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    normalized = unicodedata.normalize("NFKC", value)
    if any(unicodedata.category(c).startswith("C") for c in normalized):
        return False
    visible = "".join(
        c
        for c in normalized
        if not unicodedata.category(c).startswith(("M", "Z"))
        and c not in "\u115f\u1160\u2800\u3164\uffa0"
    )
    if not any(c.isalnum() for c in visible):
        return False
    if re.search(r"\b(?:TODO|TBD|MISSING|INSERT|PLACEHOLDER|REPLACE(?:MENT)?)\b", normalized, re.I):
        return False
    if re.search(r"\[\s*OWNER|\\(?:g<|[0-9])|\$\{|\{\{|<!--", normalized, re.I):
        return False
    # Author fields are literal single paragraphs, not Markdown or document structure.
    return not normalized.lstrip().startswith(("#", "|")) and not any(
        c in normalized for c in ("\n", "\r", "\u2028", "\u2029")
    )


def release_gaps(record: dict[str, Any]) -> list[str]:
    if not isinstance(record, dict) or set(record) != {"status", "statements", "confirmations"}:
        raise ValueError("Invalid author record schema")
    statements, confirmations = record["statements"], record["confirmations"]
    if not isinstance(statements, dict) or not isinstance(confirmations, dict):
        raise ValueError("Author statements and confirmations must be objects")
    if set(statements) != set(STATEMENTS) or set(confirmations) != set(CONFIRMATIONS):
        raise ValueError("Author record has missing or unknown fields")
    if record["status"] not in ("author_confirmed", "author_answers_required"):
        raise ValueError("Invalid author record status")
    gaps = (
        [] if record["status"] == "author_confirmed" else ["status: author confirmation required"]
    )
    gaps.extend(name for name in STATEMENTS if not valid_statement(statements[name]))
    gaps.extend(name for name in CONFIRMATIONS if confirmations[name] is not True)
    return gaps
