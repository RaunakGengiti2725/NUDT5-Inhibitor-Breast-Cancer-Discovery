# Deposited NUDT5/NUDT14 structural pharmacology reanalysis

## Current Path B author-review package

The current manuscript leads with all-site 8RIY/8OTV W0O geometry and bounded published catalytic measurements. This is a descriptive/methodological reanalysis, not a new inhibitor, energetic mechanism, selectivity predictor or biological experiment. Official report extraction and fixed PDBe map sampling/slices are limited model-support inspection, not independent density validation. Crystal copies are nonindependent. The exact author sheet and current closure register are in `research/path_b/`; author declarations and laboratory gates remain unresolved.

The historical 18,412-compound library must not be reconstructed. NC5-02 is ACT-19/known compound 11. Original-label performance is diagnostic only: unmatched, unassayed presumed negatives and largely unauthenticated positive labels do not establish target recognition. The old audit/manuscript is preserved verbatim under `research/path_b/audit_archive/`, with immutable baseline/hash and correction provenance. Original audit routes are historical; current closure state is separate.

After the locked installation below, build the explicit **Path B** profile into a fresh external directory:

```sh
.venv/bin/python scripts/build_research_documents.py --profile path-b \
  --output "$HOME/nudt5-path-b-documents"
```

This validates the committed manuscript's abstract/quantitative blocks against frozen inputs, independently recomputes the same-split control AUCs from stored predictions, reconstructs all structural scientific fields and regenerates the model-support extraction. Changed or self-consistent-but-untrusted evidence fails closed; a numerical input change requires a reviewed versioned lock and manuscript update. No fitting occurs. Figures 1–3 are structural; diagnostic figures are S1 onward in a separate PDF/DOCX. Table 1 gives all W0O report mappings, Table 2 all eight paired compounds including training overlaps, and Table 3 all comparable same-split controls including seven single descriptors, descriptor LR, nearest neighbours, constants/prevalence and current consensus. Full score pointers, CSVs, figures and completion-last manifest accompany the documents. Generated binaries stay outside Git.

The default `legacy` profile preserves the former builder behavior for historical/custom manuscripts. To reproduce the archived manuscript layout, pass `--profile legacy --manuscript research/path_b/audit_archive/baseline_manuscript.md`. Use Path B for the current manuscript; do not append the legacy diagnostic-first figure ordering to it.

Reproduce numerical evidence without fitting models (each output must be new):

```sh
.venv/bin/python scripts/build_structure_model_support.py \
  --output "$HOME/nudt5-path-b-model-support"
.venv/bin/nudt5-selectivity --source research/selectivity/paired_evidence.json \
  --predictions research/results/transfer.json --repository . \
  --output "$HOME/nudt5-path-b-paired"
PYTHONPATH=scripts/scripts:scripts .venv/bin/python \
  research/path_b/source_checks/regenerate_identity.py --repository "$PWD" \
  --output "$HOME/nudt5-path-b-identity.json"
```

The geometry command sequence below accepts a fresh result and separate derived manifest; add `--profile path-b` to its document command. Compare scientific fields and normalized source hashes, not new runtime timestamps/paths/Git state. Historical manifests are never restamped. Run `.venv/bin/python -m pytest tests research/path_b/source_checks` to include both the full repository and upstream source-witness tests. Path B regression tests cover ordering, quantitative drift, source lock, corruption, no-fitting and atomic refusal. Dependency auditing may explicitly skip the local non-PyPI package; that is not an audited package claim.


**These records do not establish a new NUDT5 inhibitor or a breast-cancer treatment.**
This revision preserves the original CSVs and replaces an unreliable screening demonstration with explicit chemical-data audits and exploratory label diagnostics.

The supplied dataset contains 46 records (20 labelled positives, 26 untested decoys). One positive, ACT-18, is not a valid RDKit molecule. NC5-02 is identical to training record ACT-19. Two structures have authenticated source-paper assay mappings; most source labels remain unverified (see the evidence ledger). A negative decoy label does **not** mean experimentally demonstrated inactivity.

## Install (Python 3.11+, verified on Python 3.12)

```sh
uv venv .venv --python 3.12
uv pip sync --python .venv/bin/python --require-hashes requirements.lock
uv pip install --python .venv/bin/python --require-hashes -r requirements-structure.lock
uv pip install --python .venv/bin/python --no-deps -e .
```

`requirements.lock` pins and hashes the original development/document-generation environment; `requirements-structure.lock` adds Gemmi 0.7.3 for the deposited-coordinate CLI and structural document inputs, without changing existing pins. `requirements.txt` contains pinned analysis dependencies; `requirements-dev.txt` is the lockfile input. No commercial docking software or credentials are needed for the implemented audit. No docking or MM-GBSA implementation is included.

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

This produces a PDF, editable DOCX, figures and accessible supplementary tables from `research/manuscript.md` and the recorded diagnostic, paired-target and structural evidence. These files are author-review drafts, not submission-ready declarations. Choose a new/empty output directory. No generated research image represents an experiment. The textual method settings and seed labels describe the supplied seed-42/five-seed diagnostic release; if rerunning a different protocol, update the manuscript and figure annotations accordingly.

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

The baseline verification passed 87 tests; the integrated release adds paired-target, assay, provenance and document regressions. See [the verification ledger](research/verification.md) for current command results. The dependency audit found no known advisories in the pinned third-party environment; it cannot certify this local project. Tests do not establish biological validity. Independent-review reports and their reconciliation are in `research/reviews/` and `research/reviewer_simulation.md`.

### Numerical reproduction versus run provenance

The committed run manifests preserve their real execution-time revision, dirty-worktree state, paths and command arguments. A fresh run on another checkout should produce its own manifest; those runtime records are **not** expected to be byte-identical. The reported byte-identical reruns refer to numerical audit/benchmark/control/transfer JSON, not runtime metadata. Compare source/input content hashes and settings, and use the separate relative-path release inventory for the portable bundle. Do not rewrite historical provenance to resemble a post-commit run.

Source cutoffs lacking both classes remain explicit in `source_cutoff_status.csv` and supplementary feasibility tables. Undefined ROC-AUC is never displayed as zero; the source-comparison figure annotates an unavailable 50 µM comparison instead of fabricating bars.

## Descriptive paired-target analysis (not a selectivity predictor)

```sh
.venv/bin/nudt5-selectivity \
  --source research/selectivity/paired_evidence.json \
  --predictions research/results/transfer.json --repository . \
  --output results/paired-analysis
.venv/bin/python scripts/build_selectivity_figures.py \
  --input results/paired-analysis/selectivity.json \
  --manifest results/paired-analysis/selectivity-manifest.json \
  --output results/paired-figures
```

All 23 source graphs/46 endpoint cells are preserved. Eight source rows have paired endpoints;
six remain after overlap exclusion, with three point ratios, one strict upper bound and two
double-censored ratios with no finite bound. R is reported mean IC50(NUDT14)/IC50(NUDT5), not an
affinity constant. Highest Equal_mean accompanies known dual compound 9 (R=0.600) in both frozen
scenarios. Related chemistry from one already-inspected publication is **not fresh validation**.
No fitting, score selection, ratio confidence interval or selectivity classifier is introduced.
Source SDs, reaction-time differences and unresolved control/replication wording remain visible.
See [the complete paired evidence](research/selectivity/analysis/selectivity.md).

The analysis writes numerical JSON, complete pharmacology/endpoint/score CSVs, Markdown and an
execution-time manifest. The figure command writes PNG/PDF/SVG for **both** scenarios. All
commands refuse replacement. The document command above now includes both figures and full
paired tables, publishing `documents-manifest.json` last after successful staging. It requires
recorded paired results and a matching manifest by default. For a legacy results directory only,
pass `--allow-missing-selectivity`; absence is explicitly annotated, never converted to zero.
An empty eligible cohort renders an explicit empty state, not fictitious observations.

For a wheel installation, provide absolute paths and `--repository /path/to/evidence-checkout`.
The wheel packages both new modules, not research CSVs, source snapshots or contract files.
The figures/documents remain checkout-only builders. Reinstall the editable package after updating
entry points: `uv pip install --python .venv/bin/python --no-deps -e .`.

## Future assay inputs: software capability only

```sh
.venv/bin/nudt5-assay --help
.venv/bin/nudt5-assay --repository /path/to/evidence-checkout \
  --manifest /path/to/future-input/manifest.json \
  --observations /path/to/future-input/observations.csv \
  --output /existing/output-directory/new-report.json
.venv/bin/python -m pytest -q tests/test_assay.py
```

These are argument examples, **not delivered measured inputs**. Read the
[protocol and unresolved prerequisites](research/assay/PROTOCOL.md) before use. The explicit
repository supplies `research/assay/contract.json`, `manifest.schema.json` and `requirements.lock`;
the output parent must exist. Strict package checks precede descriptive bounded curve fitting.
Technical means do not inflate submitted biological n. Relative midpoint and absolute
control-normalized 50% crossing are separate; unsupported estimates remain null with reasons.

Synthetic curves are **SOFTWARE TESTS ONLY**, never biological Results or raw measured ledgers.
No physical experiment, qualification, registration, prospective lock, target validation or
therapeutic discovery has occurred. Hashes and declared metadata do not authenticate raw-source
semantics, normalization, blinding or independent preparations. All laboratory gates remain open.

## Bounded PubChem provenance cross-check

[Archived queries and offline audit](research/external/pubchem/README.md) cover 21 assay
descriptions (12 RNAi, 9 ChEMBL-deposited protein assays). All 26 concise rows across 16 CIDs map
to existing ChEMBL ledger records, not independent validation. Six ledger values are censored
and two missing; the concise API omits relation symbols. No evaluation rows or model results change.

```sh
PYTHONPATH=scripts/scripts .venv/bin/python scripts/build_pubchem_audit.py \
  --snapshots research/external/pubchem \
  --ledger research/external/observed_database_rows.csv \
  --output results/pubchem-audit.json
cmp results/pubchem-audit.json research/external/pubchem/audit.json
```

Use an existing output parent and a new filename. This bounded search is not exhaustive.


## Published structures: observed geometry and a bounded lab handoff

Read [the lab handoff](research/structure_comparison/lab_handoff.md) and
[the hypothesis/control/falsification table](research/structure_comparison/hypotheses_controls.csv).
This is an all-site calculation on the published compound-9 complexes 8RIY/8OTV, not a new
inhibitor, binding observation, affinity estimate or test of the published Arg51 rationale.
The nearby 8RIY Arg51 minima are 3.756 Å to backbone N (partial residue, zero-occupancy CZ
excluded) and 3.251 Å to side-chain CD (occupancy 0.78). The calculation does not establish
uniform side-chain dependence. The handoff prioritizes qualified WT/R51A/R51K direct-binding
falsification with TH5427 and compound 9 as **unmatched comparators**. Physical prerequisites,
pilot precision and prospective locking remain unresolved; `nudt5-assay` is not KD software.

From the repository root, after the installation above, reproduce **only geometry and documents**
without refitting any model. Choose a new external directory; `mkdir` deliberately fails if reused:

```sh
REPO="$(pwd)"
OUT="$HOME/nudt5-structure-reproduction"
mkdir "$OUT"
.venv/bin/nudt5-structure \
  --repository "$REPO" \
  --input-manifest "$REPO/research/structure_comparison/runtime_input_manifest.json" \
  --contract "$REPO/research/structure_comparison/geometry_contract.json" \
  --output "$OUT/observed_proximity.json"
SHA="$(sha256sum "$OUT/observed_proximity.json" | cut -d ' ' -f 1)"
.venv/bin/python scripts/build_structure_comparison_figures.py \
  --input "$OUT/observed_proximity.json" --input-sha256 "$SHA" \
  --repository "$REPO" \
  --output "$OUT/derived"
.venv/bin/python scripts/build_research_documents.py \
  --structure-input "$OUT/observed_proximity.json" \
  --repository "$REPO" \
  --structure-manifest "$OUT/derived/derived_manifest.json" \
  --output "$OUT/documents"
.venv/bin/python scripts/build_release_manifest.py --output "$OUT/release-manifest.json"
```

The structural result has 1,730 residue-conformer rows (1,564 observed, 58 partial, 108
null/refused) and 1,278 atom pairs within 5 Å. All four sites and both receptor chains remain;
crystal copies are not independent n. Fixed inclusive radii are 3.5/4.0/4.5/5.0 Å. Missing
atoms are not imputed; complete-residue distances and coordinate uncertainty are not estimated.
The figure command produces three CSVs, PNG/SVG/PDF and a derived manifest. Documents include
the structural map as readable separate-target Figure 7A–B pages, the complete vector map,
complete accessible tables, the handoff/control/source files
and a completion-last manifest. No output overwrites previous evidence.

The document CLI defaults to the committed structural result plus its hash-bearing derived
manifest. Missing, malformed, hash-mismatched or empty structural evidence fails closed before
publication. Both builders reconstruct the complete fixed-contract result from the trusted
checkout's archived source package and require exact scientific-field agreement, including all
sites, conformers, residue/atom identities, pair inventories, occupancy, missingness and target
labels. A caller-supplied result hash or self-consistent derived manifest is not evidence of that
agreement. `--repository` selects the absolute trusted evidence checkout (default: this checkout,
not the caller's working directory); it is never inferred from untrusted result provenance.
Source fingerprints must match after relative-path normalization, while historical runtime
paths, commands, timestamps and Git state are preserved, not rewritten. This is reproduction by
the existing coordinate engine, not an independent scientific validation. Changed source packages
need a separate versioned contract; publication is not a generic renderer for arbitrary results.
`--allow-missing-structure` is a legacy-only opt-in for **absent** inputs; it labels
unavailability and cannot excuse malformed/partial inputs. The separate `--allow-missing-selectivity`
option retains the same legacy boundary for paired-target evidence.

For a wheel installed outside the checkout, run `nudt5-structure` with the same absolute
repository/manifest/contract/output paths. The wheel packages the engine, not source evidence;
figures/documents are checkout-only builders. Compare numerical content excluding the newly
recorded provenance, not runtime timestamps/paths. The original structure-package README and
manifests describe the earlier curation stage; [the run record](research/structure_comparison/results/VERIFICATION.md)
and this section describe the implemented stage without rewriting historical provenance.
