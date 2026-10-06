# Bounded Path B repair verification report

## Scope and verdict

One repair round from writer revision `22830642f7d33ca3319872b5ba4c513baff50154`, on
`devin/1791248176-path-b-bounded-repair`. All 37 numbered objections from three independent **AI**
reviews are answered in `response.md`; the original reviews are unchanged and hash-checked.
The package remains author-review-only. The central JMGM contribution objection is unresolved.
`unresolved_objections.json` lists every unresolved or partly unresolved objection and ownership.

## Verification

| Command | Observed result |
| --- | --- |
| `.venv/bin/ruff check .` | Passed |
| `.venv/bin/ruff format --check .` | Passed, 33 files |
| `.venv/bin/mypy --no-incremental` | Passed, 30 source files |
| `.venv/bin/python -m pytest` | 687 passed; no failures or skips; 16 dependency deprecation warnings |
| `.venv/bin/python -m pytest research/path_b/source_checks` | 11 passed; no failures or skips |
| `.venv/bin/python -m compileall -q scripts tests` | Passed |
| `.venv/bin/python -m build` | Source distribution and wheel built successfully |
| `.venv/bin/pip-audit` | No known vulnerabilities in audited packages; local `nudt5-evidence-audit` is not on PyPI and was explicitly unaudited |

Exact command results and timings are in `quality-suite.json`; external delivery includes the logs.
The first suite run identified an old six-decimal display expectation. The test now checks the
required at-most-three-significant-digit strings exactly; endpoint/source checks were not weakened.
Two regression tests also ensure cutoff captions derive their counts/AUCs from the supplied data and
do not invent metrics for an empty cohort. No active pre-commit hook was present. The staged diff is checked with
`git -c core.whitespace=cr-at-eol diff --cached --check` so preserved CSV CRLF endings are
recognized rather than rewritten.

The document tests prohibit model fitting and verify source locks, score alignment, archive hashes,
quantitative-block synchronization, artifact hashes, atomic refusal, retained provenance and DOCX
figure/table structure. New tests cover pooled versus pair-weighted within-fold AUC, ties and empty
inputs, sparse calibration and corruption, fixed-score group resampling, report fields, functional-
group radii, ratio presentation, review-response completeness and historical withdrawal metadata.

Initial headless document inspection covered the 13-page manuscript and 30-page diagnostic PDF:
no extracted text blocks fell outside a page; the main pages and key calibration tables were also
visually inspected. DOCX structure had four main tables/four images and fifteen supplement tables/
ten images. The final fresh build is made from the clean repair commit; its page counts, hashes and
layout checks are recorded in the external verification package, not assumed from this initial build.
No claim of an independent Word-application rendering inspection is made.

## Source preservation and release

`source-preservation.json` records exact byte comparisons: all 35 CSVs present at immutable baseline
`40b9b0708d888a015abe5043bb273c3c6ee601ae` are unchanged, including CRLF line endings, and 285
untouched writer-revision files match byte-for-byte. Only the explicitly named manuscript, builder
and regression-test files were excluded from that latter comparison. Historical manifests, frozen
science outputs, original inputs and all three reviews were not restamped or rewritten.
`final_hits.csv` retains its original hash; separate metadata marks each of its ten rows a withdrawn
historical assertion. Withdrawal does not prove the historical work never happened.

Generated PDF/DOCX, figures, caches, environments, wheels and scratch files remain outside Git. The
final delivery includes a source snapshot of the exact repair commit, generated documents/tables/
figures, completion-last document manifest and external verification records. The delivery manifest
records the exact commit without introducing a self-referential commit hash in this source report.
No PR, merge, force push or change to main/existing PR branches is authorized by this round.
A repo-scoped environment blueprint was proposed for user approval, not silently applied.

## Scientific limits

- Source-label separation, including perfect raw-MW AUC, does not establish target recognition.
  The universal claim that consensus loses in every retained design is false: the unique-molecule
  split and tight-cutoff source challenge are explicitly retained as numerical exceptions.
- Crystal copies are nonindependent. Positive-occupancy atom distances and fixed precomputed-map
  slices do not establish contact energy, affinity, mechanism, homology or selectivity. NUDT14
  Leu107 is not mapped to NUDT5 Arg51. The AAA zero-occupancy CZ is absent from the slices, not absent
  from the deposited model.
- EDIAm/OPIA definitions were retrieved and all six report rows displayed; the primary EDIA method
  article remained inaccessible and the map-generation version is unrecorded.
- Unequal 20/60-minute assays, shared TH5427 normalization, unclear replication and absent raw
  covariance preclude an inferred ratio interval. The compound-9 result is descriptive discordance.
- No new library/decoys, docking, MD, deep-learning stack, biological experiment, prospective
  validation, external contact, public submission or deposit change occurred. Tests exercise code
  including test-fixture fitting; no new scientific model-fitting analysis was run.
- Tests, hashes, audit counts, package size and AI-review recommendations carry no biological
  validation weight. Author-owned facts, laboratory gates and JMGM significance remain open.
