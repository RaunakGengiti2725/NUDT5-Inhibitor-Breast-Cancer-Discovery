# Findings, verification and unresolved evidence

Baseline audited: `8c2a1b6990df1e140e15ab5a0eabea69b70eee14`. Six independent separate-VM audits (chemical provenance, statistics, engineering, biological evidence, novelty, publication policy) were run and then adjudicated against directly re-executed checks. Raw input CSVs were never edited. No deposit, submission or merge occurred.

These are implementation, provenance and interpretation defects. They are **not** a finding of misconduct, and absence of an artifact in the released files does not prove it never existed.

## Critical findings

| # | Finding | Location | Evidence | Fix in this revision |
|---|---|---|---|---|
| 1 | Loader silently dropped an unparsable structure, so the documented 20-active dataset is 19 valid | baseline `pipeline.py` `load_compounds`; `compounds.csv:19` | ACT-18 fails RDKit kekulization (atoms 5, 21, 28, 29, 30) | Invalid rows are reported with line, identifier and parser error, require `--allow-invalid`, and are excluded from scoring instead of disappearing |
| 2 | 26 unique decoys were tiled 20x to 520 rows, so every held-out negative had an identical training molecule | baseline `(decoys * 20)[:520]` | Exact-identity negative leakage; the manuscript also claims 1,400 decoys | Canonical-identity deduplication; replication removed; conflicting duplicate labels fail closed |
| 3 | BEDROC was misnormalized | baseline `bedroc` | Perfect ranking returned 0.00173, not 1.0 | Finite-list formulation with expected within-tie contributions, tested against RDKit `CalcBEDROC` over 1,004 ranking patterns |
| 4 | Reported p < 0.001 is unattainable from the stated procedure | manuscript P21, P357-P368; baseline permutation block | Code ran 5 permutations; the plus-one minimum for B=30 is 1/31 ~ 0.032 | Optional diagnostic reports `(exceedances+1)/(B+1)` and its floor; the significance claim is withdrawn |
| 5 | "Leave-scaffold-out" validation split by row position | baseline LSO block; manuscript Table 5 | After the parse failure only ACT-20 was held out, not 3 of 20 | `StratifiedGroupKFold` on exact Murcko scaffolds with asserted train/test disjointness, plus a labelled-series holdout |
| 6 | Named reference chemistry does not match the primary sources | `compounds.csv:2-3`; manuscript Table 9 | ACT-01 is C19H18Cl2N8O3; Page 2018 compound 28 / PDB 9CH (5NWH) TH5427 is C20H20Cl2N8O3. ACT-02 is not the 8-dimethylamino TH1713 of PDB 958/5NQR | Raw rows preserved and the identity conflict documented; the 29 nM potency is no longer attached to an unauthenticated analogue; PDB 958's conflicting TH5427 synonym is recorded, not silently resolved |
| 7 | A "novel candidate" is a training molecule with published activity | `final_hits.csv:5` = `compounds.csv:20` | NC5-02 = ACT-19 = Balikci 2024 compound 11, NUDT5 IC50 2.04 +/- 0.240 uM and NUDT14 IC50 0.519 +/- 0.084 uM (n = 2 biological replicates); ACT-20 = compound 10, 0.487 +/- 0.010 / 0.263 +/- 0.031 | Reported as a rediscovered, already-measured, NUDT14-cross-reactive compound; removed from any novelty count; the blanket "all molecules are untested" statement is corrected |
| 8 | Near-perfect accuracy was interpreted as prospective predictive ability | manuscript P19-P23, Table 4 | After deduplication a seven-descriptor logistic regression reaches 0.980 AUC on exact-scaffold splits and 0.998 on series holdout, above every fingerprint model; decoy and active property medians differ widely | Reported as dataset-bias diagnostics with property-only and nearest-neighbour baselines, three split types and explicit confounding discussion |
| 9 | Principal reported results have no accessible artifacts | manuscript P17, P229-P233, P399-P401, Table 8 | The 347 MTH1 proxy records, transfer weighting, 18,412-compound library, cascade attrition, docking poses/logs and MM-GBSA outputs are absent from the repository, the supplied CSVs and the 9,365-byte Zenodo archive | Those numerical results are withdrawn rather than restated; the author-supplied evidence needed to restore them is listed below |
| 10 | Public deposit metadata repeats the unsupported claims | IEEE DataPort `10.21227/cbef-k354`; Zenodo `10.5281/zenodo.19374517` | Archive MD5 `72e28aa847c97099a376017036c578b2` holds only the five baseline files; the supplied attachment CSVs are byte-identical | Author-approval correction draft prepared in `deposit_correction.md`; nothing was posted or modified |

## Major findings

| # | Finding | Evidence | Fix |
|---|---|---|---|
| 11 | The inspected released code does not implement the described transfer weighting | Baseline consensus is an unweighted mean of four scores; no proxy weighting or transfer loss exists. Benchmark and candidate code also differ, one min-max normalized and one not | Renamed `Equal_mean` with its exact definition; no transfer-learning or calibration claim |
| 12 | Model specifications conflict between manuscript and code | The manuscript states RF-500, XGBoost and a Tanimoto-kernel SVM; the code implements RF-100, scikit-learn GBT and RBF-SVM. Tables 5-7 report mutually incompatible values for the same protocols | Only the implemented, versioned models are reported, with seeds and fold assignments recorded |
| 13 | Applicability-domain tiers contradict their own definition | NC5-02 is stored as "Extrapolation" with a recomputed maximum similarity of 1.0; NC5-04 (0.197) and NC5-06 (0.234) are stored as "Moderate"; only 3 of 10 reach the stated 0.25 cutoff against valid actives | Nearest training identifiers and full-precision similarities are published instead of confidence tiers |
| 14 | Documented descriptor values are RDKit-version dependent | Four candidate HBA cells differ under RDKit 2023.09.6 but all 70 basic descriptor cells reproduce under pinned 2025.09.6; both runs are in `results/descriptors-rdkit-*.json` | Version-pinned, hash-recorded descriptor regeneration; the discrepancy is reported as version dependence, not a data-entry error |
| 15 | Diversity and similarity statistics do not reproduce | The mean of the 45 unordered candidate pairs is 0.1546, not the stated 0.12; NC5-01's similarity to authenticated TH5427 is 0.386, not 0.53 | Recomputed values with stated fingerprint settings and reference identity |
| 16 | The pipeline could not run from a clean checkout and wrote on import | The data path resolved to the nonexistent `scripts/data/compounds.csv`; importing the module created directories | Repository-relative resolution, explicit path arguments, no import-time side effects, refusal to overwrite a nonempty output directory |
| 17 | No tests, pins, build, typing or provenance | The baseline collected 0 tests and had no `pyproject.toml`; dependencies were unpinned | 46 regression tests, hash-verified lockfile, strict typing, buildable package, read-only GitHub CI, and manifests recording Git revision, input hashes and environment |
| 18 | Citation and disclosure defects | The Nguyen Science paper is 2025, DOI `10.1126/science.adv4257` (one audit unit wrongly disputed its authorship; Crossref adjudicated it, and the distinct Wu paper `10.1126/science.adx9717` is not a substitute). The Marques work is the 2025 preprint `10.1101/2025.03.16.643557`, subsequently published as Nature Communications 17:8192 (2026), `10.1038/s41467-026-74489-9`; these are related versions, not independent replications. Reference placeholders remain at P620-P621, and no funding, competing-interest, contribution, affiliation or AI statements exist | Corrected citation metadata; an author-confirmation register replaces invented declarations |
| 19 | The claimed absence of in vivo NUDT5 inhibitor work is contradicted | Published TNBC xenograft work reports tumour-growth effects together with deaths in 4 of 10 treated animals | Both the efficacy observation and the tolerability caveat are reported, with cause of death stated as not established |
| 20 | "First study" and venue-fit claims are unsupported | A bounded search cannot establish priority; the inspected NeurIPS 2026 deadlines (4 and 6 May) have passed, and Cell Press author-policy pages returned HTTP 403 | Priority claim removed; `publication_strategy.md` states conditional fit and unverified journal requirements |

## What the corrected diagnostics actually show

Three evaluations of 45 unique valid molecules (19 positive labels, 26 decoys), seed 42 (five folds for molecule/scaffold splits; two for series holdout). Seeds 42–46 are a separate sensitivity analysis:

| Method | Unique-molecule AUC | Exact-scaffold AUC | Positive-series holdout AUC |
|---|---:|---:|---:|
| Seven-descriptor logistic regression | 0.996 | 0.980 | 0.998 |
| RBF-SVM on ECFP4 | 1.000 | 0.960 | 0.812 |
| Random forest on ECFP4 | 1.000 | 0.941 | 0.639 |
| `Equal_mean` fusion | 0.998 | 0.931 | 0.563 |
| Nearest-active Tanimoto | 1.000 | 0.913 | 0.329 |
| Gradient-boosted trees | 0.929 | 0.843 | 0.291 |

The descriptor baseline performs numerically best in the scaffold and series diagnostics, not in molecule splitting. Its strong discrimination is compatible with property/source confounding; it does not prove that no target-specific information exists or quantify the causal share of dataset bias. Equal_mean's series-fold AUCs are 0.654 and 0.828 versus pooled 0.563: training-composition and score shifts make pooled comparisons fragile. At an arbitrary 0.5 threshold, fingerprint/similarity methods have sensitivity 0/19, which is not a calibration-independent conclusion. The property model also has sensitivity 0/2 in the smaller series fold despite perfect within-fold ranking. Ninety-nine molecule-label permutations yield unadjusted p = 0.01 for all six methods, including Property_LR; no multiplicity-adjusted or biological significance claim follows. Complete fold results and five-seed ranges are released.

## Fixed-design extension results

The design was frozen in `research/extension_design.md` before execution; no seed, threshold, similarity cutoff or model was chosen after seeing a result.

- Single-descriptor exact-scaffold AUC: TPSA 0.996, HBA 0.996, MW 0.988 versus 0.980 for all seven descriptors. Leave-one-out stays 0.968-0.984.
- Zero decoy/positive pairs satisfy a 0.5 SD caliper on all seven descriptors. Uncalipered nearest matching moves the mean standardized gap only from 1.49 to 1.41 SD and raises descriptor AUC to 0.994. Matching is reported as exploratory cohort design, never as causal adjustment.
- Constant 0.5 scores AUC 0.500; a fold-prevalence-only predictor scores pooled 0.246 while constant within folds, illustrating pooled cross-fold ranking artifacts rather than estimating their contribution to any fitted model.
- Tanimoto >= 0.70 component splitting gives AUC 1.000 for four of the original six methods and also for fingerprint logistic regression; cLogP alone gives pooled AUC 0.093. This partition is not nested within exact-scaffold grouping and does not isolate a causal mechanism.
- Split-conformal under scaffold grouping: coverage 0.978 at nominal 0.90 with mean set size 1.78; 0.844 at nominal 0.80 with 2.2% empty sets. Exchangeability is not assured under group shift, so no coverage guarantee is claimed.
- Measured-source challenge (10 non-overlapping Balikci compounds, 5 measured IC50 0.270-21.2 uM versus 5 reported inactive): RF, RBF-SVM, nearest-active and fusion reach AUC 1.000, GBT 0.940, descriptor logistic regression 0.800 and 0.563 at a 1 uM cutoff. The repository-label ordering reverses. Retrospective, single source, n = 10, maximum training Tanimoto 0.16-0.66 with ACT-20 usually nearest: retrospective single-source ranking with analogue relationships, not independent prospective validation.
- MRK-952 versus MRK-952-NC: correctly ordered by 3 of 6 methods, exact tie for GBT, reversed for descriptor logistic regression and random forest. MW, cLogP and authenticated-TH5427 similarity also favour MRK-952. Under symmetric random strict ordering either direction has probability 0.5, but this is not an inferential p-value for the observed model outputs/ties. MRK-952-NC is weakly inhibitory and is not used as a decoy; its assay species/construct are unstated and its stereochemistry is assumed by the source.
- Every conclusion is now categorized as Demonstrated, Strongly supported, Suggestive, Hypothesized, Unknown or Refuted in manuscript section 4.8.

## Verification

Run in a locked Python 3.12.15 environment:

```text
ruff check .                            All checks passed!
ruff format --check .                   12 files already formatted
mypy --no-incremental                   Success: no issues found in 12 source files
python -m pytest                        80 passed
python -m compileall -q scripts tests   clean
python -m build                         sdist and wheel built
pip-audit                               No known vulnerabilities in installed third-party dependencies
```

The test run emits upstream Matplotlib/pyparsing deprecation warnings; none are suppressed and none are failed or skipped tests. The local project has no public vulnerability-database entry; the dependency-audit verdict applies to third-party packages, not a guarantee about project security. Installed-wheel execution was separately checked with explicit input paths and exactly reproduced the source audit.

`pip-audit` initially reported PYSEC-2026-3001 (pymupdf, used only for inspecting the supplied manuscript outside the project environment), PYSEC-2026-1845 (pytest) and PYSEC-2026-3447 (setuptools); all three were resolved by upgrading rather than by suppressing the audit.

## Author-supplied evidence still required

1. The authoritative ACT-18 structure, from its source record rather than reconstruction.
2. Per-compound source identifiers, assay type, measured values, replicate counts and the activity threshold for every positive label, especially the 15 unverified Page-2018-labelled rows.
3. Provenance for the 26 decoys and any measured-inactivity data; otherwise they stay labelled presumed-negative.
4. The 347 MTH1 proxy records with identifiers, and the executable weighting procedure.
5. The 18,412-member library with versioned identifiers and stage-level outputs.
6. Receptor preparation, ligand poses, software versions, settings, scores and logs for docking and MM-GBSA.
7. Full-precision component scores and model artifacts behind the published candidate scores.
8. Affiliations, author contributions, funding, competing interests, licensing, approvals applicable to any original experiments, and the actual extent of AI assistance.
9. Confirmation of the intended citations and resolution of the P620-P621 placeholders.

## Findings deliberately not "fixed"

- Raw `compounds.csv` and `final_hits.csv` are unchanged, including the invalid row, the misidentified references and the stored scores. Correcting the primary record is an author decision with provenance consequences.
- Authenticated reference structures were not substituted into the original-graph primary diagnostics. A separate, explicitly labelled sensitivity replaces ACT-01/ACT-02 with source-authenticated graphs and recomputes all six models, three splits and source/probe/candidate scores; neither raw CSV is rewritten.
- No historical table was recomputed into a more favourable number; withdrawn results stay withdrawn pending artifacts.
- No public deposit or submission was touched.

## Audit scope and source-ledger supplement

Paragraph references count the supplied DOCX's 720 direct Word body paragraphs; printed Appendix Table 11 is XML table 13, not a caption mismatch. The DOCX SHA-256 is `11ef96824597f47daa1e6152caea8dc87cb3ebd6d78fd41be3baad14d31cca35`. The corrupted PDF could not be reviewed; DOCX text/table checks do not constitute a full inspection of all embedded figures. Six model-based review scopes are not six independent laboratories or human peer review.

`source_assays.csv` is a new provenance ledger of all 23 structures from Balikci et al.'s author-supplied supplementary CSV, with seven numeric NUDT5 IC50 values, five reported inactives and eleven not tested. It preserves reported values, source compound numbers, uncertainty and exact repository identity matches. Source URL, original-member hash, encoding and transformation are in `results/source-assay-provenance.json`. These records are not injected into the training set, and no new assay was performed.
