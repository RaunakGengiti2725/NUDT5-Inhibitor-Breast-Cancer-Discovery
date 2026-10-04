# NUDT5 chemical-data and reproducibility audit

**These records do not establish a new NUDT5 inhibitor or a breast-cancer treatment.**
This revision preserves the original CSVs and replaces an unreliable screening demonstration with explicit chemical-data audits and exploratory label diagnostics.

The supplied dataset contains 46 records (20 labelled positives, 26 untested decoys). One positive, ACT-18, is not a valid RDKit molecule. NC5-02 is identical to training record ACT-19. Two structures have authenticated source-paper assay mappings; most source labels remain unverified (see the evidence ledger). A negative decoy label does **not** mean experimentally demonstrated inactivity.

## Install (Python 3.11+, verified on Python 3.12)

```sh
uv venv .venv --python 3.12
uv pip sync --python .venv/bin/python --require-hashes requirements.lock
uv pip install --python .venv/bin/python --no-deps -e .
```

`requirements.lock` pins and hashes the complete development/document-generation environment. `requirements.txt` contains pinned analysis dependencies; `requirements-dev.txt` is the lockfile input. No commercial docking software or credentials are needed for the implemented audit. No docking or MM-GBSA implementation is included.

## Audit the raw data

```sh
.venv/bin/python scripts/scripts/pipeline.py audit --output results/audit
```

This writes machine-readable `audit.json` and `manifest.json`. It reports invalid records, canonical identity duplicates, candidate/training overlap, molecular properties, PAINS alerts and nearest-training similarities. Input files are never edited. A previous nonempty output directory is never overwritten; use a new run directory. Outputs use full numerical precision; round only for presentation.

The installed `nudt5-audit` command accepts the same arguments. When using an installed wheel rather than a checkout, pass explicit `--compounds /path/compounds.csv --candidates /path/final_hits.csv` inputs. CSVs are not bundled into the wheel.

## Exploratory diagnostics (not validated activity prediction)

```sh
.venv/bin/python scripts/scripts/pipeline.py benchmark \
  --allow-invalid --acknowledge-unverified-labels \
  --permutations 99 --repeat-seeds 5 --output results/diagnostic
```

Both flags are deliberate: the benchmark explicitly quarantines invalid rows and uses **unverified source labels**. It is a dataset-bias diagnostic, not a clinical or chemical-activity probability model. The defaults are seed 42 and five folds; the example examines seeds 42–46 without selecting the best result.

- Canonical isomeric-SMILES duplicates are collapsed with recorded representatives. Conflicting labels fail closed. No salts, tautomers, protomers or stereoisomers are silently merged.
- ECFP4 means RDKit Morgan radius 2, 2,048 bits, default bond-type handling, no chirality. Exact Bemis–Murcko scaffold splits exclude scaffold sharing across train/test. This is not equivalent to holding out a medicinal-chemistry series.
- Three evaluations are reported: unique-molecule stratification; exact-scaffold grouped stratification; and labelled-positive-series holdout with disjoint decoy partitions. Series evaluation requires at least two nonempty positive `series` values. The latter is extremely small here and not a reliable target-wide generalization estimate.
- RF (100 trees), sklearn GBT (100 trees, depth 3), RBF-SVM (C=10), a training-only nearest-active Tanimoto baseline, and seven-descriptor logistic regression are evaluated. Descriptor scaling is fitted inside each training fold. Models are fixed, not tuned using the test folds. SVM internal probability fitting uses training data only; its outputs are not validated biochemical-activity probabilities.
- `Equal_mean` is a fixed arithmetic mean of RF, GBT, SVM and nearest-active scores. There is no min–max scaling using held-out candidates. It is **not** the manuscript's unimplemented MTH1 transfer-weighted loss or a validated TWCS method.
- BEDROC20 uses correctly normalized finite-list exponential ranks. Exactly tied scores receive expected within-tie contributions. EF uses `ceil(fraction * N)` and reports the actual cutoff count. With 45 molecules, nominal EF1% means one molecule; nominal EF5% means three.
- Conditional bootstrap ranges resample exact-scaffold groups from **fixed out-of-fold predictions**. They do not refit models, do not include training uncertainty and are not population or clinical confidence intervals. Single-class bootstrap draws and test folds are explicitly recorded.
- Optional molecule-level label permutations refit the full molecule-split diagnostic and report `(exceedances + 1)/(B + 1)`. At B=99 the minimum is 0.01. Chemical series are not assumed biologically exchangeable; a small diagnostic p-value cannot rule out dataset bias or validate labels.

No candidate efficacy ranking is produced from these unverified labels. Candidate scores in the original CSV are retained as historical assertions, not promoted to regenerated or validated results. PAINS absence and rule-of-five compliance do not establish safety, solubility, selectivity or activity.

## Verification

```sh
.venv/bin/ruff check .
.venv/bin/ruff format --check .
.venv/bin/mypy
.venv/bin/python -m pytest
.venv/bin/python -m build
.venv/bin/python -m compileall -q scripts tests
.venv/bin/pip-audit
```

Tests include exhaustive small-ranking comparison against RDKit BEDROC, ties, finite permutation resolution, invalid inputs, duplicate identity/conflicting labels, scaffold/series isolation, deterministic evaluation and CLI execution from another working directory. Scientific Python type stubs cover external library interfaces; project code remains under strict type checking.

## What is and is not deposited

The original archive is [Zenodo 10.5281/zenodo.19374517](https://doi.org/10.5281/zenodo.19374517), linked by [IEEE DataPort 10.21227/cbef-k354](https://doi.org/10.21227/cbef-k354). These are data/software deposits, not evidence of journal peer review. The downloaded Zenodo archive corresponds to baseline Git commit `8c2a1b6990df1e140e15ab5a0eabea69b70eee14` and contains the five original repository files. The supplied attachment CSVs are byte-identical to those files. DataPort lists two subscription-restricted PDFs; their contents have not been inspected in this revision.

The 347 MTH1 proxy records, independent 520-decoy collection, 18,412-compound library, docking poses/logs, MM-GBSA results and assay-level activity ledger are **not available in the inspected repository, supplied CSVs or Zenodo archive**. This does not prove they never existed. They cannot be reconstructed or claimed to have been reproduced from the released files.

## Layout

- `compounds.csv`, `final_hits.csv`: unchanged historical source records.
- `scripts/scripts/pipeline.py`: audit/diagnostic CLI; no import-time writes.
- `tests/`: regression and integration tests.
- `research/`: evidence-bounded manuscript, claim ledger, audit, publication strategy and deposited-artifact correction draft.

Funding, affiliations, competing interests, contributions and original-study approvals require author confirmation. AI-assisted research/code/editorial work performed in this revision must be disclosed according to the selected venue. No journal submission, public deposit modification, release or merge is authorized by this revision.

## Regenerate the author-review manuscript

```sh
.venv/bin/python scripts/build_research_documents.py --output results/documents
```

This produces a PDF, editable DOCX and a data-derived diagnostic figure from `research/manuscript.md` and the recorded benchmark. These files are author-review drafts, not submission-ready declarations. Choose a new/empty output directory. No generated research image represents an experiment. The textual method settings and seed labels describe the supplied seed-42/five-seed diagnostic release; if rerunning a different protocol, update the manuscript and figure annotations accordingly.

## Fixed-design controls, measured-source challenge and probe pair

```sh
python scripts/scripts/controls.py --allow-invalid --acknowledge-unverified-labels --output <new-dir>
python scripts/scripts/transfer.py --allow-invalid --acknowledge-unverified-labels --output <new-dir>
```

`controls.py` runs the design frozen in `research/extension_design.md`: trivial baselines, single-descriptor and leave-one-out ablations, nearest-property matching with a 0.5 SD caliper, Tanimoto >= 0.70 component splitting, similarity-binned metrics, calibration, group-disjoint split-conformal diagnostics and fixed-score bootstrap intervals. `transfer.py` re-scores previously inspected measured-source compounds and the MRK-952/MRK-952-NC probe pair with frozen models, excluding training/parent overlaps and untested compounds, and keeps endpoint families separate. Both refuse to overwrite a non-empty output directory and require explicit invalid-record and unverified-label acknowledgement. Neither performs an assay, and neither output is external or prospective validation.

The transfer output also contains a **separate authenticated-reference sensitivity**: only ACT-01/ACT-02 graphs are replaced using the source-linked TH5427/TH1713 structures in `research/reference_structures.csv`; all six methods, three split schemes and candidate/source/probe scores are rerun. Original CSVs and original-graph results are retained. This sensitivity is not a new blinded experiment or a best-performing replacement dataset.

New evidence files: `research/conclusion_categories.csv`, `research/reviewer_simulation.md`, `research/original_inventory.json`, `research/external/`, `research/structural/` and `research/study/`. Source-derived reports retain their authorship and access limitations; they are not proof of new laboratory experiments.

To regenerate figures, supplementary tables and the PDF/DOCX from the recorded results:

```sh
.venv/bin/python scripts/build_research_documents.py --output results/documents
.venv/bin/python scripts/build_release_manifest.py --output results/release-manifest.json
```

Both destinations must be new (the document builder also accepts an empty directory). The inventory contains relative filenames, purpose, provenance, size, SHA-256 and evidence/reproduction status. The run manifests identify exact analysis inputs and source hashes; the release inventory covers the broader dossier. The supplied manuscript's quantitative source-comparison table is checked against recorded JSON by the test suite.

The final local verification passed 80 tests, Ruff, formatting, strict mypy, byte-compilation and package build. The dependency audit found no known advisories in the pinned third-party environment; it cannot certify this local project. Tests do not establish biological validity. Independent-review reports and their reconciliation are in `research/reviews/` and `research/reviewer_simulation.md`.
