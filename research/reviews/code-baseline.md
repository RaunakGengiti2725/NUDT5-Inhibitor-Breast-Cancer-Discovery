# Independent review of the revised NUDT5 snapshot

**Verdict:** No high/critical findings, benchmark numerical discrepancies, or actual train/test leakage found in this bounded review. **Two medium-severity CSV validation defects remain**, demonstrated below. Neither affects the supplied CSVs or changes the reproduced results.

Reviewed only the attached `review-package.zip`, extracted outside the repository at `/home/ubuntu/review`. No baseline checkout, external literature research, source/data edits, commits, PRs, submissions, or child sessions were used. All 25 original archive files remain byte-identical after verification.

## Remaining findings, severity-ranked

### 1. Medium — Duplicate CSV headers silently replace labels or structures

**Location:** `scripts/scripts/pipeline.py:72–87`, `read_compounds`.

The schema check verifies that required column names exist, but not that they are unique. `csv.DictReader` keeps the last value for a repeated header. Thus this CSV silently reverses both labels, returns an empty issues list, and succeeds through the audit CLI:

```csv
id,smiles,label,label
X,CCO,0,1
Y,CCC,1,0
```

Observed records: `X: label=1`, `Y: label=0`; `issues=[]`; audit exit status **0**. The same defect with repeated `smiles` headers silently replaces molecular identities. Neither the original conflicting value nor a validation error reaches `source_row`/the audit issues.

**Smallest fix:** Before iterating records, reject duplicate header names, including optional metadata fields. Add regression cases for repeated `label`, `smiles`, and `series`. Treat an ambiguous schema as a file-level error, not a row that can be silently selected with `--allow-invalid`.

### 2. Medium — An unterminated quoted field can swallow subsequent records without an issue

**Location:** `scripts/scripts/pipeline.py:72–87`, CSV reader construction/iteration.

The default non-strict CSV parser accepts an unclosed quote at EOF. For this input, record B disappears into A's notes:

```csv
id,smiles,label,notes
A,CCO,1,"unclosed
B,CCC,0,note
```

Observed: only A is returned, with `notes='unclosed\nB,CCC,0,note'`; `issues=[]`; audit exit status **0**, reporting one valid record and zero invalid records. This can silently alter sample size/class balance in a larger input.

**Smallest fix:** Use `csv.DictReader(handle, strict=True)` and translate `csv.Error` into a clear file-level `ValueError` handled by `main`. Strict mode was independently verified to reject this fixture with `unexpected end of data`. Add a regression asserting nonzero CLI exit and no successful audit output. Ordinary properly quoted multiline fields should remain supported.

### Shared reproduction

From the extracted snapshot, save either fixture exactly as shown to a new file outside the snapshot, then run:

```sh
.venv/bin/python scripts/scripts/pipeline.py audit \
  --compounds /absolute/path/to/fixture.csv \
  --candidates "$PWD/final_hits.csv" \
  --output /absolute/path/to/new-empty-output
```

Both currently return 0. Inspect `audit.json`: neither identifies the CSV defect. No biological inference is needed to reproduce either issue.

## Verification results

Python **3.12.15**, isolated uv environment. Commands run from the extracted snapshot:

```sh
uv venv -p 3.12 .venv
uv pip install -p .venv/bin/python --require-hashes -r requirements.lock
.venv/bin/ruff check .
.venv/bin/ruff format --check .
.venv/bin/mypy
.venv/bin/python -m pytest -q
.venv/bin/python -m build --outdir /home/ubuntu/review-out/dist
.venv/bin/python -m compileall -q scripts tests
.venv/bin/pip-audit -r requirements.lock --require-hashes --disable-pip
.venv/bin/python scripts/scripts/pipeline.py benchmark \
  --allow-invalid --acknowledge-unverified-labels \
  --permutations 99 --repeat-seeds 5 \
  --output /home/ubuntu/review-out/repro
```

- Lint, format, strict mypy and compile checks passed. **46 tests passed**, with **16 visible upstream Matplotlib/pyparsing deprecation warnings**; none were disabled.
- Wheel and sdist built successfully. Dependency audit reported no known vulnerabilities.
- Full benchmark completed in approximately **92 seconds**.
- Regenerated **benchmark.json and audit.json are byte-identical** to their supplied recorded counterparts. Recursive comparison found zero key/length/value differences, including all **2,713 benchmark floating-point values**; maximum absolute difference **0.0**.
- All five manifest file hashes match: pipeline, both CSVs, requirements and lockfile. Pipeline SHA-256:

```text
7e80649f3ce9aaef62b9a87afabbb79fd8cbb24ea206b5c57c43c1776761eeb4
```

Python/dependency versions match the recorded manifest. Parent absolute input paths are expected; local paths/output, platform and unavailable Git metadata reflect extraction outside a checkout, not provenance errors.

## Statistical and status checks

- Independently recomputed pooled AUC by positive–negative pair counting, AP by tied-score thresholds, EF by fractional boundary-tie allocation, and threshold statistics for all **18 model/split combinations**. They agree with recorded values.
- Independently enumerated partial-tie BEDROC expectations for **62 binary label vectors × 36 tie orderings**; all agree. Existing exhaustive untied RDKit-reference tests also pass.
- Checked exact-identity train/test exclusion, complete one-time test coverage, scaffold disjointness, positive-series exclusion and both training classes. Scaffold/series partition checks passed across seeds **42–46**. Feature scaling, nearest-active references and SVM probability fitting use training data only; fixed ensemble averaging does not normalize against held-out data.
- Independently reproduced a nondegenerate fixed-score scaffold-bootstrap interval and all six plus-one permutation counts. Permutations rebuild label-dependent stratification and refit the molecule-split models.
- Fixed-OOF bootstrap intervals omit retraining uncertainty and use exact-scaffold clusters, not independent biological series. Cross-fold pooled AUC and arbitrary threshold 0.5 limitations are already acknowledged; they are **not new defects**.
- Candidate statuses are `training_overlap_not_novel` or `not_in_training_set_novelty_and_activity_unverified`. The warning and manuscript do not promote diagnostic scores, property compliance, or candidate status to biological validation.

**Bottom line:** The supplied revised benchmark is reproducible and its checked statistical calculations are correct. Fix the two CSV fail-closed gaps; no further concrete code/statistical correctness issue was established within this review's scope.
