# Commands and verification

Baseline `40b9b0708d888a015abe5043bb273c3c6ee601ae`. Commands below were executed from `/home/ubuntu/repos/NUDT5-Inhibitor-Breast-Cancer-Discovery` in the repository hash-locked Python 3.12 environment; Gemmi 0.7.3 is supplied by the additive structure lock. Scratch holds all build/environment outputs.

## Final checks

### lint

```sh
/home/ubuntu/path_b_source_scratch/venv/bin/ruff check --no-cache research/path_b/source_checks
```

Exit 0. Full output is retained in `verification.json`.

### format

```sh
/home/ubuntu/path_b_source_scratch/venv/bin/ruff format --check research/path_b/source_checks
```

Exit 0. Full output is retained in `verification.json`.

### typecheck

```sh
PYTHONDONTWRITEBYTECODE=1 MYPYPATH=scripts/scripts:scripts /home/ubuntu/path_b_source_scratch/venv/bin/mypy --cache-dir /home/ubuntu/path_b_source_scratch/mypy-cache research/path_b/source_checks
```

Exit 0. Full output is retained in `verification.json`.

### identity_reproduction

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=scripts/scripts:scripts /home/ubuntu/path_b_source_scratch/venv/bin/python research/path_b/source_checks/regenerate_identity.py --repository /home/ubuntu/repos/NUDT5-Inhibitor-Breast-Cancer-Discovery --output /home/ubuntu/path_b_source_scratch/identity_summary_final.json && cmp /home/ubuntu/path_b_source_scratch/identity_summary_final.json research/path_b/source_checks/identity_summary.json
```

Exit 0. Full output is retained in `verification.json`.

### original_audit_reproduction

```sh
PYTHONDONTWRITEBYTECODE=1 /home/ubuntu/path_b_source_scratch/venv/bin/python scripts/scripts/pipeline.py audit --output /home/ubuntu/path_b_source_scratch/original_audit_final && cmp /home/ubuntu/path_b_source_scratch/original_audit_final/audit.json research/results/audit.json
```

Exit 0. Full output is retained in `verification.json`.

### pubchem_reproduction

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=scripts/scripts:scripts /home/ubuntu/path_b_source_scratch/venv/bin/python scripts/build_pubchem_audit.py --snapshots research/external/pubchem --ledger research/external/observed_database_rows.csv --output /home/ubuntu/path_b_source_scratch/pubchem_audit_final.json && cmp /home/ubuntu/path_b_source_scratch/pubchem_audit_final.json research/external/pubchem/audit.json
```

Exit 0. Full output is retained in `verification.json`.

### relevant_tests

```sh
PYTHONDONTWRITEBYTECODE=1 /home/ubuntu/path_b_source_scratch/venv/bin/python -m pytest -p no:cacheprovider -q research/path_b/source_checks tests/test_pubchem_audit.py tests/test_pipeline.py::test_source_dataset_counts_invalid_and_overlap tests/test_pipeline.py::test_explicit_validation_and_canonical_identity tests/test_pipeline.py::test_schema_and_empty_input tests/test_pipeline.py::test_invalid_molecules_raise tests/test_pipeline.py::test_fingerprint_and_properties tests/test_pipeline.py::test_audit_cli_preserves_sources_and_refuses_overwrite tests/test_pipeline.py::test_cli_from_other_working_directory_and_import_has_no_io tests/test_pipeline.py::test_candidate_audit_distinguishes_active_and_all_training_reference tests/test_transfer.py::test_identity_preserves_stereo_and_flags_parent_equivalence tests/test_transfer.py::test_invalid_identity_fails tests/test_transfer.py::test_reference_substitution_is_separate_and_preserves_unaffected_records tests/test_transfer.py::test_missing_source_columns_and_unmatched_reference_fail
```

Exit 0. Full output is retained in `verification.json`.

### build

```sh
mkdir /home/ubuntu/path_b_source_scratch/build_checkout && git archive 40b9b0708d888a015abe5043bb273c3c6ee601ae | tar -x -C /home/ubuntu/path_b_source_scratch/build_checkout && mkdir -p /home/ubuntu/path_b_source_scratch/build_checkout/research/path_b && cp -R research/path_b/source_checks /home/ubuntu/path_b_source_scratch/build_checkout/research/path_b/ && /home/ubuntu/path_b_source_scratch/venv/bin/python -m build --no-isolation --outdir /home/ubuntu/path_b_source_scratch/dist /home/ubuntu/path_b_source_scratch/build_checkout
```

Exit 0. Full output is retained in `verification.json`.

### tracked_scope_and_whitespace

```sh
git diff --exit-code && git diff --check
```

Exit 0. Full output is retained in `verification.json`.

## Interpretation and boundaries

These are software/provenance checks, not biological validation. No full model-fitting suite was run because new training was prohibited. Relevant pipeline identity/audit, transfer identity, PubChem provenance, source-witness and no-fitting regression tests were selected. The build is of the unchanged Python distribution from BASE in a scratch export, with scoped handoff files copied in. No environment, wheel, generated PDF/DOCX or cache belongs in this commit.

## Setup and resolved diagnostics

The environment was installed from repository `requirements.lock` and additive `requirements-structure.lock`. Ruff line-length errors and mypy import/version-attribute issues were fixed before final checks. Both original input hash-failure paths are tested. A temporary source-excerpt assertion exposed two Marques abstract elements; the precise first abstract is now selected and tested.

PubChem was not assumed to be an exact graph match: regeneration showed NC5-02/CID 22346757 matches only after tautomer canonicalization despite their matching standard InChIKey. The test explicitly preserves this distinction.

The initial Page supplementary archive request timed out. The official publisher-linked PDF subsequently downloaded and was hashed. `pdftotext` was unavailable, so no internal PDF page/procedure inspection is claimed and the locked analysis environment was not modified to add a parser. Nguyen received one full-text attempt only (403). The official JMGM author guide also returned 403; no access bypass was attempted.

No active pre-commit configuration or active local hook was found; only Git sample hooks. Normal commit and push run with hooks enabled. No PR, merge or force push is authorized.
