# Independent second review — extension-review-v1

**Verdict:** No high-severity defect or numerical/conformal-indexing error found. Four medium-severity findings remain: two source-input validation gaps and two result-interpretation/reporting problems. The validation gaps require altered inputs; **the supplied source data and computed results reproduce correctly**.

Reviewed the new attached uncommitted snapshot, not the old checkout. All **78 archived files remain unchanged**. No repository edits, commits, PRs, external submissions, or child sessions. The already-planned introductory claim revision, unique regioisomer graph IDs, and authenticated-TH5427 sensitivity are not counted as new findings.

## Medium findings

### M1. Duplicate/conflicting measured-source identities are counted as independent examples

**Location:** `scripts/scripts/transfer.py:128–156` (`measured_challenge`); source CSV validation at lines 69–78 checks syntax, not row/identity uniqueness.

**Reproduction:** Copy `research/source_assays.csv` outside the snapshot and append compound 4's row unchanged. Calling `measured_challenge(training, copied_csv, 42)` succeeds: the eligible set becomes **11 rows / 6 positives**, with compound `4` twice. Alternatively append compound 1 under `source_compound=new-alias`, changing its NUDT5 result to `not active`: the same molecular graph is accepted with contradictory labels. RF AUC becomes **0.8833333333**, versus 1.0 originally; Property_LR becomes **0.75**, versus 0.8.

**Impact:** Accidental ledger duplication or alias merging can reweight the reported challenge or introduce contradictory outcomes without a warning. This is separate from the known probe display-name collision.

**Smallest fix:** Validate unique source-compound IDs and canonical identities for this fixed single-source/endpoint challenge before scoring. Reject duplicates/conflicts unless an explicit replicate-handling rule is introduced; do not silently count them as new molecules.

### M2. Censoring metadata can contradict the labels silently produced

**Location:** `transfer.py:81–89, 133–156`.

`potency("not active")` always means `inactive_gt50_uM`; `inactive_definition` is retained but never validated.

**Reproduction:** In a copied assay ledger, change the five inactive rows' `inactive_definition` to `IC50 >10 µM; not tested is distinct from inactive`, leaving the `not active` tokens unchanged. The challenge still succeeds, emits `inactive_gt50_uM` for all five, and treats them as negatives at the **50 µM** cutoff. A lower bound of >10 does not establish IC50 ≥50.

**Impact:** An edited/differently censored source can receive unsupported threshold labels. The supplied ledger consistently specifies >50 µM, so its current classification is correct.

**Smallest fix:** For this bounded implementation, validate the fixed endpoint/unit/censoring contract and reject incompatible definitions. A generic censor parser is unnecessary. If generalized later, classify only when a censoring interval establishes the threshold label.

### M3. The similarity-component split is not a stricter version of scaffold holdout

**Location:** `controls.py:455–464`; `research/manuscript.md:116`.

The implementation correctly enforces the **Tanimoto ≥0.70 component** boundary, but replacing scaffold groups with similarity components does not preserve scaffold isolation.

**Reproduction from supplied results:** Read `controls.json → evaluations → similarity_component_split → folds`. Four of the five seed-42 folds have nonempty `scaffold_overlap`. For example, fold 3 trains on **ACT-20** and tests **ACT-19** with the same Murcko scaffold; fold 1 trains on **ACT-17** while testing same-scaffold **ACT-15/ACT-16**. Independent checks confirm cross-fold similarities are below 0.70, so this is **not leakage against the implemented component policy**.

**Impact:** The manuscript's “stricter”/“harder” characterization and inference that improved AUC is inconsistent with dominant analogue memorization do not follow from this non-nested comparison.

**Smallest fix:** Describe an **alternative, non-nested similarity-boundary stress test**, disclose scaffold overlap, and remove that causal inference. No split redesign is required.

### M4. Two manuscript conclusions contradict the actual transfer scores

**Location:** `research/manuscript.md:7, 135, 139–143`; numerical source is `transfer.json`.

- The statement that fingerprint/similarity methods rank *every* measured inhibitor above *every* inactive is false for **GBT**. At 50 µM its AUC is **0.94**: positive compound **14** ties inactive compounds **12, 13, 15**, all at score **0.15304550305907036**. Table 2 and the generated figure correctly show the non-perfect GBT value.
- The heading “not ordered by consensus” and statement that consensus “did not pass” the pair ordering test conflict with `MRK_pair_contrast.Equal_mean`: **0.13084072864990803 − 0.12394161916468878 = +0.006899109485219254**. Equal fusion orders the pair correctly; one correct pair provides no meaningful validation, but that is not failed ordering.

**Smallest fix:** Name the four perfectly separating methods explicitly, disclose GBT's ties, and say the consensus ordering is correct but uninformative for validation. Keep the small-N caveats.

## Checks and independent tests

- Isolated uv environment, **Python 3.12.15**, installed `requirements.lock` with `--require-hashes`.
- `ruff check .`, `ruff format --check .`, and strict `mypy`: **pass**.
- `python -m pytest -q`: **70 passed**, **16 visible upstream warnings**; none suppressed.
- Re-ran controls with `--allow-invalid --acknowledge-unverified-labels --repeat-seeds 5`, and transfer with both acknowledgement flags, into new sibling output folders. **Both JSON reports are byte-identical** to the supplied records. Every controls/transfer manifest input hash matches the snapshot.
- Regenerated all three figures in PNG/PDF/SVG and supplementary tables. **All three PNGs and all three textual/CSV outputs are byte-identical** to the supplied artifacts.
- Independently checked 42 model/design combinations: pair-counted AUC, threshold-based AP/trapezoidal PR-AUC, Brier/confusion metrics, calibration-bin accounting; independently refit all 15 proper-training RFs and reconstructed class-conditional calibration indexing, p-values, sets, and group disjointness. All agree.
- Verified matching without replacement and zero simultaneous 0.5-SD caliper matches; all five requested seeds reproduce feasible component fits, with both training classes retained. One-class test folds correctly have undefined fold metrics.
- Verified 7 numeric / 5 censored inactive / 11 untested source rows, exclusion of training matches 10/11, 10 eligible molecules, fixed 1/10/50 µM labels, and unchanged trained-model protocol. Probe endpoint rows stay separate; no endpoint pooling is used to fit models.

### Added regression tests

The accompanying `test_extension_regressions.py` contains three desired fail-closed assertions for M1/M2. **All three fail on this snapshot**, specifically because no `ValueError` is raised; these are additional review tests, not failures in the original 70-test suite.

Run outside the snapshot, using its pinned environment:

```sh
SNAPSHOT_ROOT=/absolute/path/to/extracted-snapshot \
PYTHONPATH=/absolute/path/to/extracted-snapshot/scripts/scripts \
/path/to/pinned-env/bin/python -m pytest -q /path/to/test_extension_regressions.py
```

## Limitations, not new defects

Full-cohort standardization selects the exploratory matched subset; prediction preprocessing is training-only. The subset remains unbalanced and is not an independent causal adjustment. Zero caliper matches establishes failure of that specified criterion, not impossibility of every conceivable cohort design. Fixed-score bootstraps omit model-refitting uncertainty; pooled scores mix fold-trained models; threshold 0.5 is arbitrary; tiny conformal calibration classes give coarse p-values and often both-label sets. Those are disclosed design limitations, not numerical errors or biological validation.
