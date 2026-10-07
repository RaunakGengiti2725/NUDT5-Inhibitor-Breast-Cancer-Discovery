# Verification record

Input commit `94eb1dde461eef82fb7a903f5cfd2eefb1deaf1e`. These are checks actually run during curation; they are evidence of package integrity, not of any geometry result.

## Evidence checks (all passed)

- All JSON files parse without duplicate keys or nonfinite constants.
- All 15 remote archives exactly match retrieval hashes/lengths; every source has URL and UTC retrieval time.
- All 14 input-manifest hashes including contract/acceptance cases agree with current bytes.
- Both coordinate SHA256 values match supplied preflight.
- All curated atom/missingness/polymer/category rows round-trip exactly to strict Gemmi parsing; 4 site inventories and official identity ASU assemblies cross-check against RCSB APIs.
- Exact real inventory assertions T22 confirmed; only 8RIY atom314 has occupancy zero; only 8OTV B47 has A/B alternatives.
- Bounded W0O exact graph/descriptor/ledger cross-reference verified; no other compound audit or model rerun.
- All 8 exact primary excerpts reproduce from archived XML; fresh XML bytes match existing source archive.
- Contract fixed radii and full residue/conformer coverage are consistent with inventories; all 25 acceptance cases have distinct IDs. Geometry implementation tests remain unexecuted by design.
- All 151 previously tracked files are byte-identical to BASE, stronger than protected-CSV/results-only check.

## Root repository checks

- `ruff_check`: All checks passed!
- `ruff_format_check`: 20 files already formatted
- `mypy_no_incremental`: Success: no issues found in 20 source files
- `pytest`: 392 passed, 16 warnings in 60.16s
- `compileall_src_tests`: ok
- `build_sdist_wheel`: nudt5_evidence_audit-0.1.0 sdist and wheel built
- `pip_audit_root`: No known third-party vulnerabilities; local project nudt5-evidence-audit itself is not on PyPI and cannot be audited (expected).
- `pip_audit_isolated_gemmi_0_7_3`: {"dependencies": [{"name": "gemmi", "version": "0.7.3", "vulns": []}], "fixes": []}

## Not performed

- No ligand-protein distances or contact maps computed
- No electron density, structure factors or refinement inspected
- No docking, MD, MM-GBSA, model rerun or threshold tuning
- No acceptance case executed against an implementation (none exists)
- No root dependency pin changed
