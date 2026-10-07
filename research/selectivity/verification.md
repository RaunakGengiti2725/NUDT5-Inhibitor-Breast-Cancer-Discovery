# Verification record

Verified in Python 3.12.15, pinned RDKit 2025.09.6, starting at `4f92a9742270a155f470aad5f06369bd76ed3092`. This is software/source verification, **not biological qualification or fresh external validation**. No new source-model scores, synthetic dose-response curves or physical experiments were generated. Existing regression tests may fit their own software-test models; those are not new research outputs.

## Environment and project checks

Executed from repository root:

```sh
uv venv .venv --python 3.12
uv pip sync --python .venv/bin/python --require-hashes requirements.lock
uv pip install --python .venv/bin/python --no-deps -e .
.venv/bin/ruff check .
.venv/bin/ruff format --check .
.venv/bin/mypy
.venv/bin/python -m pytest
.venv/bin/python -m build
.venv/bin/python -m compileall -q scripts tests
.venv/bin/pip-audit
```

Results: Ruff passed; 12 Python files already formatted; strict mypy passed on 12 files; **87 tests passed**, with 16 upstream Matplotlib/Pyparsing deprecation warnings; sdist and wheel built; byte-compilation passed. The dependency audit found no known third-party vulnerabilities; it skipped the local package, which is not on PyPI. No pre-commit configuration, AGENTS.md or custom repository hook was present. No dependency or code change was needed. Build artifacts, caches and editable-install metadata are ignored and not committed.

## Current curation checks

A local one-shot standard-library assertion harness, executed with:

```sh
.venv/bin/python /home/ubuntu/selectivity-work/validate_selectivity.py
```

passed these checks:

- All current JSON files parse without duplicate object keys or nonfinite constants. The harness implements and checks all validation keywords used by the closed Draft-07 release schema (not a general JSON Schema implementation).
- Canonical dataset passes that schema; six negative mutations fail (point for double censoring, fake censored mean, nonstrict bound, unknown field, wrong endpoint, missing target).
- 23 unique IDs/canonical graphs, 46 target cells; ratio arithmetic, comparators, foreign keys, long projection and complete count identities agree.
- All seven artificial arithmetic vectors execute against the reference function extracted from `analysis_contract.md`; all are explicitly **SOFTWARE TESTS ONLY**.
- 46 unique compound/scenario joins; **276 score cells exactly equal** their frozen JSON values; every within-six method rank and tie is checked.
- All four archived primary artifacts match both stored and decompressed SHA-256; all nine recorded repository inputs equal their starting-commit bytes.
- The full curation recomputed all23 source identities and both sets of45 training keys with the pinned identity helpers, checked candidate aliases, and compared every recomputed source identity field and overlap list to both stored score scenarios. No fitting/scoring function was called by curation.
- Every changed/untracked repository file is under `research/selectivity/`.

The one-shot curation/validation scripts were kept outside the repository: this scoped handoff is **data and executable design**, not a new implementation or CI test command. The schema, semantic acceptance requirements, reference arithmetic function and software-only test vectors are committed so the next implementer can build durable regression coverage. The future CLI's parser/write-failure tests in the contract are requirements, not falsely reported completed tests of a nonexistent CLI.

## Source checks

The two-stage dynamic workflow (`wfr-0225de3594f2447397d548a22043f292`) completed independent primary verification followed by a review that consumed those findings. Its output is preserved in `independent_verification.json`. This session also visually read Table1's value/scaffold images and inspected the XML sections and author CSV. All16 paired Table1 assay cells agree; all23 CSV rows agree numerically/status-wise with the original ledger after two trailing-space trims. Exact primary snippets, downloads, access timestamps and hashes accompany this handoff. No primary access blocker was encountered.

The known ACT-18 invalid-graph warning and RDKit undefined-stereo warnings occurred during identity checking. ACT-18 remains excluded/quarantined as before; no source graph/stereochemistry was guessed or repaired. Exact source graphs preserve stereochemical specification and the identity records expose unassigned centers; normalized parent/tautomer keys are exclusion flags only.

## Deliberate nonclaims

No selectivity classifier, AUC, inferential p-value, fitted ratio uncertainty, unseen external validation, new biochemical measurement, clinical efficacy, chemical novelty or experiment-ready qualification. Original ledgers and historical numerical JSON remain immutable. Source control/replication/SD ambiguities are retained, not “fixed” for favorable framing.
