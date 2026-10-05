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

## Observed-coordinate handoff and manuscript integration — 5 October 2026

Integration starts at `ca06233acc3190a876495155224864a99d625fd5` on a separate branch.
No diagnostic model was rerun for new results. Published compound-9 dual binding and
Arg51 reasoning remain attributed to Balıkçı et al.; the integration adds a bounded
H2 handoff, five unmeasured hypothesis/control rows, five exact primary-Methods
excerpts, and document-generation safeguards, not biological observations.

| Command / check | Observed result |
|---|---|
| `.venv/bin/ruff check .` | Passed |
| `.venv/bin/ruff format --check .` | 24 Python files already formatted |
| `.venv/bin/mypy --no-incremental` | No issues in 24 source files |
| `.venv/bin/python -m pytest -q` | **510 passed**, none skipped; 16 upstream Matplotlib/pyparsing deprecation warnings; final run 99.99 s |
| `.venv/bin/python -m compileall -q scripts tests` | Passed |
| `.venv/bin/python -m build --outdir <external-directory>` | sdist and wheel built |
| `.venv/bin/pip-audit` | No known third-party vulnerabilities; unpublished local project cannot be audited against PyPI |
| `git diff --check` | Passed |
| Installed-wheel `nudt5-structure`, outside checkout with explicit repository/manifest/contract paths | All 1,730 residue rows, 1,278 pairs and other non-provenance fields identical to recorded output |
| Fresh Python 3.12.15 wheel environment | Original hash lock, then Gemmi 0.7.3 hash lock, then `--no-deps` wheel installation succeeded; the inherited CI install sequence includes the necessary separate structural lock |
| Standalone figure/table reproduction from installed-wheel output | All six payloads byte-identical to the recorded three CSVs and PNG/SVG/PDF; new derived manifest correctly hashes the new runtime-provenance-bearing input |
| Document generation to an external directory | **43 files**: 42 hashed payloads plus completion-last manifest; paired and structure statuses both `recorded` |
| Document input and output hashes | All matched the final generating inputs and payloads |

The 34 integration regressions added to the incoming 476 cover missing, malformed,
empty, stale, symlinked and truncated structural input; invalid geometry/witnesses;
explicit legacy absence; failed-render nonpublication; complete supplement transport;
deterministic target panels retaining both sites and every display-eligible row;
source quotations against archived XML bytes; handoff uncertainty/control fields;
manuscript numerical anchors and structure; and structural dependency inventory.
The independent-source and geometry-engine tests from the incoming branch remain intact.

The complete two-panel map was too small at portrait manuscript width. Figure 7A–B now
uses separate target pages, without dropping displayed rows or altering cutoffs; the
complete vector map and all-residue CSV remain in the supplement. Explicit figure page
breaks in DOCX and heading-following rules in PDF prevent orphaned displays/headings.
The final **24-page PDF** was rasterized and the new Methods, Results, proposed-H2 and
figure pages inspected at ordinary reading size. DOCX was independently rendered with
LibreOffice (**25 pages**) and its structural Results and both target figure pages inspected.
The abstract is 225 words and the conclusion-category synthesis follows every Results
section. These are editorial/rendering checks, not independent scientific peer review.

All **16 protected legacy files** (the two original CSVs, source-assay CSV and all 13
`research/results/*` files) are byte-identical to the incoming commit. The structural
results, all four dependency input/lock files, archived sources and historical execution
manifests are unchanged. The historical structural report is retained as an exact byte
prefix with a dated supersession pointer appended. The portable release inventory is
regenerated after these final source/verification changes; it is not an edited historical
run manifest. External builds, screenshots, environments and wheels are not committed.

Remaining science: no density/structure-factor review or coordinate-uncertainty propagation;
partial-residue minima do not estimate complete residues; crystal sites are not independent
samples. Nearby Arg51 minima involve backbone N and fractional-occupancy CD, neither a
minimum guanidinium witness. Qualified protein preparations, direct/orthogonal binding,
interference/NUDT14 controls, pilot precision and governance gates remain unresolved.
No physical-experiment-ready qualification, completed prospective lock, affinity/selectivity
estimate or clinical inference is supplied. Remote CI is not asserted by these local checks.
