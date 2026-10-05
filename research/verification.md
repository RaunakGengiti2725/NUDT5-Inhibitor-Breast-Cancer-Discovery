# Release verification

Checked on 4 October 2026 with Python 3.12.15 and the hash-pinned environment.

| Command | Observed result |
|---|---|
| `.venv/bin/ruff check .` | Passed |
| `.venv/bin/ruff format --check .` | 12 files formatted |
| `.venv/bin/mypy --no-incremental` | No issues in 12 source files |
| `.venv/bin/python -m pytest -q` | 87 passed; 16 upstream Matplotlib/pyparsing deprecation warnings; none skipped |
| `.venv/bin/python -m compileall -q scripts tests` | Passed |
| `.venv/bin/python -m build` | sdist and wheel built |
| `.venv/bin/pip-audit` | No known vulnerabilities in installed third-party dependencies; local unpublished project cannot be audited against PyPI |
| `git diff --cached --check` | Passed after lossless transport/format normalization |

All three complete analyses were rerun from their declared inputs. Benchmark (99 label permutations and five seeds), controls and transfer JSON were byte-identical to the preceding results. The source-input newline normalization was separately checked for equal parsed CSV cells and identical transfer results. Installed-wheel audit and transfer commands with explicit inputs also matched source outputs byte-for-byte. The unchanged original CSVs were checked against Git. Compressed PDB reference files are tested against the original uncompressed SHA-256 hashes.

The 16-page generated PDF was inspected for page-boundary overflow and its abstract/source-comparison table visually checked. Plots were visually checked, and all source-threshold table entries are regression-tested against recorded JSON. Independent computational/scientific review and source limitations are preserved separately. These are software/evidence checks, not experimental validation, a prospective test or human peer review.

PR-review regression coverage includes seven source-cutoff cases: all-negative/all-positive at each 1/10/50 µM cutoff, plus an empty eligible source set. Original and authenticated-reference outputs retain class counts and explicit infeasibility, with no fabricated AUC values; PDF/DOCX generation succeeds. Historical run manifests are intentionally retained, not rewritten to claim post-commit execution.

## Paired-target, future-assay and PubChem integration — 5 October 2026

The historical verification and manifests above remain execution-time records.
The integrated tree was checked with Python 3.12.15 and the same hash-locked
third-party dependencies; no dependency was added.

| Command | Observed integration result |
|---|---|
| `.venv/bin/ruff check .` | Passed |
| `.venv/bin/ruff format --check .` | All 20 Python files formatted |
| `.venv/bin/mypy --no-incremental` | No issues in 20 source files |
| `.venv/bin/python -m pytest -q` | 392 passed; none skipped; 16 upstream Matplotlib/pyparsing deprecation warnings |
| `.venv/bin/python -m compileall -q scripts tests` | Passed |
| `.venv/bin/python -m build` | sdist and wheel built |
| `.venv/bin/pip-audit` | No known third-party vulnerabilities; local unpublished package skipped |
| `git diff --check` and `git diff --cached --check` | Passed |

### Real-data and document reproduction

These commands used new output paths and never replaced historical run manifests:

```sh
.venv/bin/nudt5-selectivity \
  --source research/selectivity/paired_evidence.json \
  --predictions research/results/transfer.json --repository . \
  --output results/integration-paired
.venv/bin/python scripts/build_selectivity_figures.py \
  --input results/integration-paired/selectivity.json \
  --manifest results/integration-paired/selectivity-manifest.json \
  --output results/integration-paired-figures
.venv/bin/python scripts/build_pubchem_audit.py \
  --snapshots research/external/pubchem \
  --ledger research/external/observed_database_rows.csv \
  --output results/integration-pubchem-audit.json
cmp results/integration-pubchem-audit.json research/external/pubchem/audit.json
.venv/bin/python scripts/build_research_documents.py \
  --output results/integration-documents
.venv/bin/python scripts/build_release_manifest.py \
  --output results/integration-release-manifest.json
```

- Numerical selectivity JSON, three CSVs, Markdown and all six PNG/PDF/SVG files
  matched the recorded bytes. New runtime manifests retain their actual execution
  revision and dirty state; they are not asserted byte-identical.
- The ledger preserves 23 source graphs and 46 endpoint cells. Eight rows have both
  endpoints; excluding overlaps 10/11 leaves six: three point ratios, one strict
  upper bound and two double-censored non-estimable ratios. Table 3 is regression
  checked against every eligible endpoint, censoring label and both frozen scores.
- PubChem offline reproduction matched all 26 concise rows/16 CIDs to the existing
  ledger. Its 21 descriptions comprise 12 RNAi and nine ChEMBL-deposited protein
  records; six ledger relations are `>` and two values are missing. The recorded
  AID 2070371 ADP/1 h wording was compared directly with archived primary sec4.3
  (10 µM ADPr, 20/60 min target reaction, separate 60 min detection); neither snapshot
  was repaired. Linkage does not verify protocol equivalence or add validation.
- PDF/DOCX generation succeeded, including both scenarios and full CSV supplements.
  All 27 document artifact hashes matched the completion manifest. Both standalone
  paired figures and the PDF abstract, Table 3 and paired-figure layout were visually
  inspected; this is not a claim of human peer review or a complete visual check of
  every page. The PDF has 20 pages. DOCX ZIP/XML checks retain censoring symbols,
  reported values and software-only warnings.
- Document regressions cover missing/stale selectivity, explicit legacy absence,
  zero/one eligible-pair SOFTWARE TESTS ONLY fixtures and atomic failure cleanup.
  No missing value is silently turned into a zero.
- The portable inventory excludes its canonical previous copy even when generated
  elsewhere, preventing recursive stale-inventory references. Its hashes are checked
  after the final tracked-file edits. Generated PDF/DOCX, wheel, caches and synthetic
  assay inputs/reports remain outside the tracked release.

### Installed-wheel check (not editable-import behavior)

An isolated environment outside the checkout was created with `uv venv --python 3.12`,
synced from `requirements.lock` with `uv pip sync --require-hashes`, then installed
`dist/nudt5_evidence_audit-0.1.0-py3-none-any.whl` with `uv pip install --no-deps`.
All five installed `nudt5-* --help` commands succeeded. Imports of `assay`,
`selectivity` and `pipeline` resolved inside that wheel environment, not the checkout.
From an external working directory with `PYTHONPATH` unset:

```sh
"$WHEEL_ENV/bin/nudt5-selectivity" \
  --source "$REPO/research/selectivity/paired_evidence.json" \
  --predictions "$REPO/research/results/transfer.json" \
  --repository "$REPO" --output "$NEW_SELECTIVITY_OUTPUT"
"$WHEEL_ENV/bin/nudt5-assay" \
  --manifest "$SOFTWARE_FIXTURE/SOFTWARE_TEST_ONLY.manifest.json" \
  --observations "$SOFTWARE_FIXTURE/SOFTWARE_TEST_ONLY.observations.csv" \
  --repository "$REPO" --output "$NEW_SOFTWARE_REPORT"
```

`WHEEL_ENV`, `REPO`, `SOFTWARE_FIXTURE` and output variables denote absolute paths.
The selectivity numerical/table bytes matched committed outputs. The assay fixture
was generated solely by `tests/test_assay.py::software_fixture` outside the repository;
its output retained `SOFTWARE TESTS ONLY`, and its writer hash matched the actual
installed `pipeline.py`. An attempted second write refused and preserved existing
bytes. This exercises software capability only, not raw-data authenticity,
normalization arithmetic, blinding, biological independence or laboratory qualification.

### Preservation and scope

Fourteen files (`compounds.csv`, `final_hits.csv`, `research/source_assays.csv` and
all eleven pre-existing `research/results/` files) were byte-compared with base
`4f92a9742270a155f470aad5f06369bd76ed3092`. All matched, including historical
manifests. All four archived paired-source compressed snapshots and all ten PubChem
snapshot/manifest/audit data files matched the supplied contribution commits.
No models, thresholds, seeds or source exclusions were optimized. No physical assay,
independent validation, therapeutic discovery, PR creation or PR merge was performed.
These are local release checks; remote CI was not observed. The parent owns PR #1.
