# Chemical-identity and evaluation-source sensitivity in low-data NUDT5 inhibitor prioritization

**Evidence-bounded revision for author review, not a submission-ready discovery claim.** Author list, affiliations and declarations require confirmation. No new biological experiments were performed. Published activity for a rediscovered structure is distinguished from new experimental validation.

## Abstract

High classification scores in a small molecular dataset need not imply target-selective inhibition. We audited a released NUDT5 screen through chemical-identity checks, fixed diagnostic baselines, published biochemical measurements and deposited-coordinate comparisons. The dataset contained 45 valid structures: 19 positive-labelled molecules and 26 unassayed decoys. After deduplication, descriptor-only logistic regression reached exact-scaffold out-of-fold ROC-AUC 0.980, and TPSA alone reached 0.996. Excluding training overlaps and untested entries from a published biochemical dataset left ten measured compounds. Four of six frozen methods ranked all five compounds with IC50 <50 µM above five reported inactives; descriptor logistic regression reached AUC 0.800 and fell to 0.5625 at a 1 µM cutoff. These exposed, single-source comparisons show evaluation-source sensitivity, not prospective performance. Chemical verification identified misassigned reference graphs and a candidate identical to a training molecule and a published dual NUDT5/14 inhibitor. Among six nonoverlap paired-target compounds, the highest fusion score accompanied reported dual activity, with a NUDT14/NUDT5 mean IC50 ratio of 0.600. An all-site audit of the published compound-9 complexes found nearby Arg51 minima involving backbone N at one site and fractional-occupancy side-chain CD at the other. Residue proximity therefore supplies no uniform guanidinium-interaction or energetic-dependence evidence. We release traceable calculations, a qualified-mutant direct-binding falsification design, and separate software for future hydrolysis inputs. The contribution is an auditable evaluation case study, not a new inhibitor, selectivity predictor or experimental validation.

## 1. Introduction

Nudix hydrolase 5 (NUDT5) has a mechanistic literature describing reported hormone-signalling effects of chemical inhibition in breast-cancer cells, a noncatalytic role in purine de novo synthesis, and degradation-based studies linking NUDT5 loss to thiopurine toxicity. That literature motivates computational hit-finding [1,4,5]. It does not license treating a small in-house screen as evidence of therapeutic promise.

Our central argument is that high discrimination of curated positives from unassayed decoys cannot, by itself, support a claim of target-selective inhibition. We first establish what the release can reproduce and whether chemical identities and evaluation splits support its claims. Fixed property and similarity controls then test alternative explanations for high scores; published measurements expose dependence on the evaluation source. Paired NUDT5/NUDT14 inhibition and all-site deposited geometry address the further gap between an activity-label score and a selectivity mechanism. The resulting experimental handoff asks what could falsify differential binding dependence, rather than treating a plausible structural story as validation.

## 2. Materials

### 2.1 Inspected artifacts

The audited revision is Git commit `8c2a1b6990df1e140e15ab5a0eabea69b70eee14`, containing five tracked files: a README, an unpinned requirements file, `compounds.csv`, `final_hits.csv` and `scripts/scripts/pipeline.py`. The linked Zenodo archive (`10.5281/zenodo.19374517`, 9,365 bytes, MD5 `72e28aa847c97099a376017036c578b2`), referenced from IEEE DataPort `10.21227/cbef-k354`, contains the same five files. Two CSVs supplied directly by the authors are byte-identical to the repository copies. Two DataPort PDFs are subscription-restricted and were not inspected.

`compounds.csv` holds 46 records: 20 rows labelled positive (18 annotated to Page et al. 2018, 2 to Balikci et al. 2024) and 26 decoys. `final_hits.csv` holds 10 candidate structures with stored scores and descriptors.

### 2.2 Artifacts required by the original claims but absent

The 347 MTH1 proxy actives, the independent 520- or 1,400-member decoy collection, the 18,412-compound screening library, stage-level cascade outputs, receptor preparation, docking poses, docking and MM-GBSA logs and scores, and any assay-level activity ledger are not present in the repository, the supplied CSVs or the Zenodo archive. Their absence from the release does not prove they never existed; it does mean the dependent numbers cannot be verified, reproduced or restated here.

An additional, separately curated ledger preserves all 23 structures in Balikci et al.'s supplementary CSV: seven have numeric NUDT5 IC50 values, five are reported inactive (IC50 >50 µM), and eleven were not tested. It records assay context, source compound IDs and exact matches to the original CSVs. These source records are not added to the diagnostic training set; mixed potency thresholds or untested intermediates must not be converted silently into binary labels [2].

### 2.3 Environment

Historical diagnostic results retain the Python and dependency versions in their execution-time manifests; they are not retroactively assigned the integration environment. Reproduction uses Python 3.12 with hash-locked RDKit 2025.09.6, scikit-learn 1.7.2, NumPy 2.2.6 and SciPy 1.15.3. The separate structural run records Python 3.12.15 and Gemmi 0.7.3, installed from an additional hash lock without changing the original pins. Each run records its actual Git revision, worktree state, input/source hashes, platform and command arguments.

## 3. Methods

### 3.1 Record validation and identity

Every row is parsed with RDKit default sanitization. Failures are retained and reported with line number, identifier and parser error rather than skipped; benchmarking the remaining records requires an explicit `--allow-invalid` flag. Invalid structures themselves are never scored. Valid records receive canonical isomeric SMILES and an exact Bemis-Murcko scaffold. Records are grouped by canonical identity; duplicates collapse to a recorded representative and conflicting labels raise an error. This is exact graph identity under one pinned RDKit version - not salt, tautomer, protomer or stereochemical equivalence.

Candidates are audited against training records for exact identity, nearest neighbour, maximum Tanimoto similarity, descriptor agreement, PAINS alerts and rule-of-five/Veber compliance. Original inputs are never rewritten.

### 3.2 Representations and models

ECFP4 means RDKit Morgan fingerprints, radius 2, 2,048 bits, default bond typing, no chirality. The descriptor vector is molecular weight, cLogP, TPSA, HBD, HBA, rotatable bonds and Fsp3, standardized inside each training fold.

Six scoring methods are evaluated: random forest (100 trees), scikit-learn gradient-boosted trees (100 estimators, depth 3) and an RBF SVM (C = 10) on fingerprints; a logistic regression on the seven descriptors; maximum Tanimoto similarity to training actives only; and `Equal_mean`, the fixed arithmetic mean of the first three plus the similarity baseline. `Equal_mean` is the operation the original code actually performs. We do not call it transfer learning, consensus calibration or Transferability-Weighted Consensus Scoring, because no proxy weighting or transfer loss exists in the released code. Hyperparameters are fixed, not tuned against test folds.

### 3.3 Three complementary split schemes

1. **Unique-molecule**: stratified five-fold over deduplicated molecules. Removes the replication artifact but allows close analogues across folds.
2. **Exact-scaffold**: `StratifiedGroupKFold` on exact Murcko scaffolds, with asserted train/test scaffold disjointness. Analogues sharing a broad SAR series can still cross folds.
3. **Positive-series holdout**: each labelled positive series is held out entirely with a disjoint random partition of decoys. This withholds a broader grouping than an exact Murcko scaffold. The supplied series labels are descriptive, not independently validated chemical taxonomies. With two positive series, this evaluation has two folds, not five.

### 3.4 Metrics and uncertainty

ROC-AUC, average precision, enrichment factor and BEDROC20 are computed on pooled out-of-fold scores. EF uses `ceil(fraction * N)` and reports the realized cutoff count, which at N = 45 means one molecule for nominal EF1%. BEDROC uses the normalized finite-list exponential formulation; exactly tied scores receive expected within-tie contributions so that input order cannot change a result. The implementation is tested against RDKit `CalcBEDROC` across 1,004 ranking patterns.

Reported bootstrap intervals use **conditional cluster resampling** of exact-scaffold groups from fixed out-of-fold predictions. Separately, seeds 42–46 assess sensitivity to repeated fitting and splitting. The fixed-prediction bootstrap does not refit models or include training-set uncertainty. The seed analysis does refit models but is not a population confidence interval; neither supports clinical inference. Single-class draws and single-class test folds are counted and reported.

The label-randomization diagnostic refits the entire molecule-split procedure on each of 99 permuted label vectors and reports `(exceedances + 1)/(B + 1)`, whose floor is 0.01. Because correlated chemical series are not biologically exchangeable, we treat this as a sanity check on the fitting procedure, not as evidence of biological signal.

### 3.5 Fixed-design control extension

Before running the extension, we recorded the design locally in `extension_design.md`; this is not an external preregistration. Fixed seeds 42–46 and unchanged model hyperparameters are retained, with no best-run selection. Additional baselines are constant 0.5, fold training prevalence, fingerprint logistic regression (C = 1), and five-neighbour classification using precomputed Tanimoto distance and uniform weights. We fit seven single-descriptor and seven leave-one-descriptor-out logistic models, fingerprint-plus-properties random forests, and four leave-one-component-out fusion variants. Predictive standardization is learned within training folds only.

A separate exploratory cohort-design sensitivity standardizes seven descriptors across the full cohort, processes positives in ID order and greedily chooses the nearest unused decoy in Euclidean property distance. It reports balance in full-cohort SD units and the existence of matches within 0.5 SD on every descriptor; it does not claim unbiased or independent adjustment. Another evaluation separates connected components of a fixed Morgan-Tanimoto >=0.70 graph with grouped folds. This partition is different from, not necessarily harder than, exact-Murcko grouping. Prespecified infeasible splits are reported, not repaired by outcome-driven threshold changes.

We retain pooled and per-fold ROC-AUC, AP and trapezoidal PR-AUC separately; sensitivity, specificity, precision, NPV, MCC, F1 and balanced accuracy at an arbitrary score threshold 0.5; Brier score and five equal-width reliability bins. Nearest-training and nearest-positive similarities use bins [0,0.3), [0.3,0.5), [0.5,0.7), [0.7,0.85), [0.85,1]; one-class AUC is undefined. Paired 1,000-draw exact-scaffold bootstraps resample fixed OOF predictions and omit retraining uncertainty.

For uncertainty diagnostics, each outer training fold is divided into group-disjoint proper-training and calibration sets using the first of two inner grouped folds. A fixed 100-tree RF fits only proper-training records. Class-conditional nonconformity is 1 minus the score assigned to the observed label; a candidate label's p-value is (1 + count of calibration nonconformities at least as large)/(1 + class calibration count). Sets retain labels with p > alpha for alpha 0.1/0.2. Class counts, finite p-value resolution, assignments, coverage and efficiency are released. Group shift and label uncertainty mean exchangeability and future coverage are not assured.

### 3.6 Retrospective source scoring and chemical identity

We fit the original six methods on all 45 valid structures and score the previously inspected Balikci ledger [2], excluding exact, neutral-fragment-parent or canonical-parent-tautomer training overlaps, and all untested structures. For each of the fixed 1, 10 and 50 µM cutoffs, numeric IC50 below cutoff defines the threshold-positive group; values above a lower cutoff are threshold negatives, not inactive compounds. The source's five reported inactives remain censored >50 µM, never exact 50 µM observations. All cutoffs are retained without selection; no external set tunes the model.

A bounded ChEMBL/BindingDB and primary-source search yielded an additional six unique graphs with eight measurement rows. Database mirrors are not independent experiments. IC50, the SGC curve-label EC50 discrepancy, SPR KD, lysate Kinobead apparent Kd and cell-viability endpoints are not pooled. The SGC/MSD MRK pair is a known, exposed one-pair diagnostic [10]; species/construct and control stereochemistry uncertainties remain visible. Standard InChI/InChIKey and neutral-parent/tautomer flags supplement exact identity checks without editing training graphs or inferring equal bioactivity. Original source structures and endpoint records are retained alongside predictions. A subsequent two-index PubChem provenance cross-check (`research/external/pubchem/`) found 21 assay descriptions (12 RNAi screens, 9 ChEMBL-deposited protein assays). All 26 concise rows across 16 CIDs linked to existing ledger records, not new independent evidence. Six ledger values are censored and two missing; the concise API omits relation symbols, so its numbers are not treated as exact measurements.

### 3.7 Paired-target evidence and future-input software

The paired ledger retains 23 Balikci source graphs and all 46 NUDT5/NUDT14 endpoint cells [2, Table 1 and supplementary CSV]. We compute R = reported mean catalytic IC50(NUDT14)/reported mean catalytic IC50(NUDT5), and log10 R, only where identified. A strict censored denominator supplies an upper bound; double censoring identifies neither a finite bound nor direction. Untested is not inactive. Published target-level SDs are retained, but raw paired replicates/covariance are unavailable, so ratio uncertainty is not estimated. Exact/parent/tautomer training overlaps exclude model diagnostics only. All six previously frozen scores and both graph scenarios are joined without refitting, thresholds, AUC or inferential testing.

Separately, the future-input assay analyzer validates declared hashes/metadata, planned wells, units and technical-replicate aggregation before a bounded descriptive four-parameter fit. It distinguishes the relative fitted midpoint from the absolute control-normalized 50% crossing, refusing unsupported curves rather than inventing potency or confidence intervals. Synthetic curves are SOFTWARE TESTS ONLY, never measured-source or biological Results. Software checks cannot authenticate raw vendor values, normalization, actual blinding, independent preparations or laboratory qualification; these remain human responsibilities under the protocol in `research/assay/`.

### 3.8 Fixed, all-site observed-coordinate comparison

The archived 8RIY/NUDT5 and 8OTV/NUDT14 mmCIFs contain the same published ligand, compound 9/W0O [2, Figures 3–4]. Gemmi 0.7.3 parses raw categories with label/author identities, occupancy and alternate/missing tokens retained. Hash-checked inputs define model 1, assembly 1 (identity over the deposited dimer), both protein chains and every W0O site. For each local residue conformer we compute the minimum Euclidean distance between retained ligand and protein heavy atoms, preserving atom witnesses and every pair within 5.0 Å. Hydrogen/deuterium and zero-occupancy atoms are excluded; positive fractional occupancy is flagged, not weighted. Alternatives are not merged into joint states. Missing residues receive nulls; partial residues report only observed minima, upper bounds on their unknown complete-residue minima. Complete-residue distances are not estimated.

Inclusive radii 3.5/4.0/4.5/5.0 Å are fixed descriptive conventions (4.0 Å primary), not affinity thresholds. No density, structure factors, refinement, missing-loop reconstruction or coordinate-uncertainty propagation is performed. Crystal sites are not independent samples; no site-pooled inference, homology mapping or interaction-energy assignment is made. The immutable structural input/run records and full tables accompany Figure 7. The lab handoff translates these bounded calculations and the published protein/SPR/catalytic Methods [2, sections 4.2, 4.3, 4.8] into the earlier study's H2 falsification, not new experimental evidence.

## 4. Results

### 4.1 The released evaluation contains identity leakage and metric defects

The released code expands 26 unique decoy structures into 520 rows by twenty-fold replication before cross-validation. Every replicated negative in a test fold therefore has an identical molecule in training, violating independence of chemical identities. The magnitude of inflation is not isolated here; strong property-based separation remains after replication is removed, and the accessible code does not authenticate every historical manuscript number.

Two metric implementations are independently wrong. The released BEDROC omits the finite-list normalization: a perfect ranking of 19 actives among 539 molecules returns 0.00173, against 1.0 for the standard formula and RDKit's reference implementation, so the deposit's BEDROC20 = 0.999 cannot have come from that code path. The released permutation loop runs five iterations, not the 30 the manuscript reports, and uses an uncorrected ratio; the smallest attainable one-sided empirical p-value is 1/31 ~ 0.032 for B = 30 and 1/6 ~ 0.167 for B = 5. The claim p < 0.001 is therefore not supportable as an empirical permutation result. The described leave-scaffold-out validation also splits by row position rather than scaffold identity, which after the parse failure holds out a single positive compound rather than three.

### 4.2 Strong property-only discrimination challenges target-specific interpretation

Table 1. Pooled out-of-fold ROC-AUC over 45 unique valid molecules (19 positive labels, 26 decoys), **seed 42**. These are new diagnostics, not reproduced historical results. Five-seed sensitivity analyses (42–46) are reported separately below.

| Method | Unique-molecule | Exact-scaffold | Positive-series holdout |
|---|---:|---:|---:|
| Descriptor logistic regression (7 features) | 0.996 | 0.980 | 0.998 |
| RBF-SVM, ECFP4 | 1.000 | 0.960 | 0.812 |
| Random forest, ECFP4 | 1.000 | 0.941 | 0.639 |
| `Equal_mean` fusion | 0.998 | 0.931 | 0.563 |
| Nearest-active Tanimoto | 1.000 | 0.913 | 0.329 |
| Gradient-boosted trees, ECFP4 | 0.929 | 0.843 | 0.291 |

The descriptor model approaches the fingerprint methods under molecule splitting and exceeds them numerically under exact-scaffold and series grouping. It uses structure-derived properties but neither fingerprints nor protein features. Its performance demonstrates that high discrimination can be achieved without a target-specific representation; it does not establish the proportion of any other model's performance caused by property bias. No statistical superiority claim is made.

Series-holdout pooled scores require particular caution. The two models see different positive-series composition and class proportions during training. For Equal_mean, AUC is 0.654 within the fold containing 2 positive labels and 13 decoys, and 0.828 in the fold containing 17 positive labels and 13 decoys, whereas pooled AUC is 0.563. Cross-fold score shifts therefore materially affect the pooled result. The property model has within-fold AUC 1.000 in both series folds. Full fold assignments and predictions are supplied so that pooled rankings are not mistaken for a single prospectively deployed model.

At the arbitrary score threshold 0.5, all five fingerprint/similarity/fusion methods have 0/19 sensitivity in this series split. This is threshold-specific, not proof that they contain no ranking signal or that the effect is unrelated to calibration. Property_LR has sensitivity 17/19, including 0/2 in the smaller fold despite that fold's perfect rank discrimination. Scores are not validated biological probabilities.

Across seeds 42–46, scaffold-split AUC ranges from 0.9798 to 0.9858 for Property_LR and 0.9211 to 0.9372 for Equal_mean. Under series holdout the ranges are 0.9899–0.9980 and 0.4372–0.7267, respectively. These finite-seed ranges measure this procedure's sensitivity, not population uncertainty. Conditional fixed-prediction scaffold-bootstrap intervals for seed-42 scaffold AUC are [0.9082, 1.0000] and [0.6849, 1.0000], respectively. Such intervals omit retraining uncertainty and cannot support a precise prospective-performance claim.

All six unadjusted permutation p-values are 0.01, the minimum at B = 99. They are diagnostic label-association results, not independent confirmatory tests; no multiplicity-adjusted significance claim is made. Unrestricted permutations also do not preserve chemical-series dependence. The strong property-baseline result remains an alternative explanation that this randomization does not eliminate [6].

Independent matching checks reinforce concern about the comparison set. A prior structure audit reported that only 9 of 26 decoys match any valid active under the manuscript's stated joint molecular-weight, cLogP and Tanimoto criteria, and that none matches ACT-01 through ACT-17. The original text does not state the numerical tolerances, similarity inequality or boundary convention, the released pipeline does not implement that joint check, and no per-decoy matching record is supplied. That count is therefore carried forward as an unreproduced prior audit result, not a verified output of this package; the tolerances are not reconstructed by guesswork. Decoys remain presumed negatives rather than measured inactives. The 19/45 prevalence also differs from the replicated benchmark's 19/539, so enrichment and precision measures are not directly comparable across those datasets.

### 4.3 Chemical identity corrections

ACT-18 is the only unparsable record, failing kekulization at atoms 5, 21, 28, 29 and 30; its authoritative structure must come from a source record, not reconstruction. The structure the deposit labels TH5427 is C19H18Cl2N8O3, whereas Page et al.'s compound 28, PDB ligand 9CH and structure 5NWH give C20H20Cl2N8O3, a difference of one N-methyl substituent (net C1H2, about 14 Da); the published 29 nM potency and the 32-heavy-atom ligand-efficiency calculation therefore attach to the wrong graph, and NC5-01's similarity to the authenticated reference is 0.386 rather than 0.53. The record annotated "TH1713 mono-Cl" is also not the 8-dimethylamino TH1713 of PDB 958/5NQR; we note that the 958 component file itself carries a conflicting TH5427 synonym and resolve identities by structure, formula, source compound number and accession rather than synonym strings.

Candidate NC5-02 is canonically identical to training record ACT-19, which is Balikci et al. compound 11 with measured NUDT5 IC50 2.04 +/- 0.240 uM and NUDT14 IC50 0.519 +/- 0.084 uM (n = 2 biological replicates); ACT-20 is compound 10 at 0.487 +/- 0.010 and 0.263 +/- 0.031 uM. Both are more potent against NUDT14 than NUDT5 under those conditions. So one "novel candidate" is a rediscovered, already-measured, cross-reactive compound, two positive labels are authenticated, and the other 17 valid positive labels lack authenticated assay-level assignments: ACT-01/02 have conflicting named identities, while ACT-03 through ACT-17 have only unverified source/series annotations.

Of the ten candidates, only 3 reach the deposit's own 0.25 applicability-domain cutoff against valid actives, and the stored tiers contradict recomputed values - NC5-02 is stored as "Extrapolation" with a recomputed maximum similarity of 1.0. Mean pairwise candidate similarity is 0.1546, not the stated 0.12. Four candidate HBA values differ between RDKit 2023.09.6 and 2025.09.6 while all 70 basic descriptor cells reproduce under the pinned version, so those cells reflect version dependence rather than transcription error. All ten candidates pass the stated PAINS and rule-of-five filters; nine pass Veber. None of that constitutes evidence of activity, selectivity, solubility or safety.

### 4.4 Reproducibility repair

The released pipeline could not run from a clean checkout (its data path resolved to a nonexistent directory), created directories on import, silently discarded invalid structures, and shipped with no tests, pins, type checking, build configuration or provenance records. The revision resolves paths relative to the repository, accepts explicit input paths, refuses to overwrite a nonempty output directory, requires explicit acknowledgement before scoring unverified labels, emits machine-readable audit/benchmark/manifest JSON, and ships a regression suite covering invalid input, duplicate and conflicting labels, metric boundaries and ties, BEDROC equivalence against RDKit, permutation resolution, scaffold and series disjointness, determinism, candidate overlap and CLI behaviour from an arbitrary working directory. Lint, strict type checking, build, byte-compilation and dependency audit all pass.

### 4.5 Single-property, matching and neighbourhood controls

Under exact-scaffold out-of-fold evaluation (seed 42), single-descriptor logistic models reach ROC-AUC 0.996 for TPSA alone, 0.996 for HBA alone and 0.988 for molecular weight alone, against 0.980 for all seven descriptors together. Leave-one-descriptor-out models stay between 0.968 and 0.984. Individual structure-derived scalar descriptors therefore discriminate the positive/decoy labels numerically as well as or better than the seed-42 fingerprint models, including ECFP4 random forests (0.941). This does not identify what any other model learns.

Prespecified nearest-property 1:1 matching does not repair this. Using a 0.5 standard-deviation caliper on all seven descriptors simultaneously, **no decoy matches any positive**: zero admissible pairs exist. Deterministic nearest-neighbour matching without a caliper selects 19 decoys but reduces the mean absolute standardized gap only from 1.49 to 1.41 full-cohort standard deviations, and the matched-subset descriptor AUC rises to 0.994. These records cannot supply a comparison satisfying this particular fixed caliper; that is not a proof that every possible balancing method is infeasible. The residual imbalance is why nearest matching is not presented as a causal adjustment.

Two further fixed-design controls behave as stated in advance. A constant 0.5 predictor yields AUC 0.500. A predictor emitting only each fold's training prevalence yields pooled AUC 0.246 while being constant within every fold: pooling out-of-fold scores across folds with different class composition by itself injects fold-identity information. Pooled values in Table 1 and elsewhere must be read with that in mind; per-fold results are released.

Splitting by connected components of the Morgan-Tanimoto >=0.70 graph gives AUC 1.000 for descriptor logistic regression, fingerprint logistic regression, RBF-SVM, random forest and nearest-active similarity, and 0.982 for equal fusion. This grouping is not nested within exact-scaffold grouping: four of five component folds share Murcko scaffolds across training/test while enforcing the <0.70 cross-boundary similarity rule. Different folds and training distributions can explain different pooled AUCs. High scores persist under this particular neighbourhood separation, but do not isolate a causal learning mechanism. The cLogP-only model gives pooled AUC 0.093; we retain this reversal rather than choosing a more favourable split, and do not interpret it alone as evidence of a particular confound.

Group-disjoint split-conformal diagnostics give empirical coverage 0.978 at nominal 0.90 with mean prediction-set size 1.78 of 2 possible labels, and coverage 0.844 at nominal 0.80 with 86.7% singletons and 2.2% empty sets. Sets containing both labels have little decision value. Exchangeability is not assured under scaffold grouping, so these are diagnostics, not coverage guarantees, and the labels they cover are repository labels of uncertain assay provenance.

### 4.6 Method rankings change in a measured-source challenge

The diagnostics above all use repository labels. We therefore re-scored, with the frozen seed-42 models and no tuning, all 23 structures from Balikci et al.'s author-supplied supplementary CSV. Two (compounds 10 and 11) are excluded as exact training/parent-structure overlaps and eleven as never tested against NUDT5, leaving 10 compounds with measured outcomes: five with numeric IC50 values (0.270-21.2 uM) and five reported inactive above the source's tested range.

Table 2. ROC-AUC against measured NUDT5 outcomes for 10 previously inspected, non-overlapping compounds, frozen models, no tuning. The repository column is the exact-scaffold out-of-fold result from Table 1.

| Method | Repository scaffold AUC; n = 45 | Source <1 µM; 2/8 positive/negative | Source <10 µM; 2/8 positive/negative | Source <50 µM; 5/5 positive/negative |
|---|---:|---:|---:|---:|
| Descriptor logistic regression | 0.980 | 0.5625 | 0.5625 | 0.800 |
| Random forest, ECFP4 | 0.941 | 1.000 | 1.000 | 1.000 |
| RBF-SVM, ECFP4 | 0.960 | 1.000 | 1.000 | 1.000 |
| Nearest-active Tanimoto | 0.913 | 1.000 | 1.000 | 1.000 |
| Equal_mean fusion | 0.931 | 1.000 | 1.000 | 1.000 |
| Gradient-boosted trees, ECFP4 | 0.843 | 0.938 | 0.938 | 0.940 |

The ordering reverses numerically for these particular comparisons. The descriptor baseline that leads the six original methods on repository scaffold labels ranks last on measured-source labels. At 1 µM it reaches AUC 0.5625 while four structure/similarity methods reach 1.000. The 10 µM cutoff gives the same labels and results as 1 µM, because no retained numeric value lies between them; neither analysis is omitted. Changing cohort, fitting regime and label definition together is not a causal experiment. The observations suggest that evaluation-source choice matters; they do not establish a general superiority or identify decoy construction as the unique cause.

It is not external validation, and we do not report an AUC confidence interval, a p-value or a performance-drop estimate for it. The constraints are explicit: n = 10 from a single publication whose contents were inspected during this audit, so the evaluation is retrospective rather than blind; the labels mix numeric IC50 values with censored inactives; maximum Tanimoto similarity to training is 0.164-0.660 with ACT-20 often nearest; the source includes clinical inhibitors and related SAR compounds, so analogue relationships and a common source remain possible explanations rather than evidence of broad out-of-domain utility; at 50 µM the five-versus-five comparison changes from AUC 1.00 to 0.96 for one strictly inverted pair (to 0.9375 at 1/10 µM with 2×8 pairs). Ties instead contribute half credit. Classification uses published mean IC50 without propagating assay uncertainty: source compound 1 is 0.837 ± 0.329 µM (SD, n=2), so its mean-based 1 µM classification is not an uncertainty-resolved boundary. GBT ties positive compound 14 with reported-inactive compounds 12, 13 and 15 at the 50 µM cutoff, hence its AUC 0.940 rather than perfect ordering.

### 4.7 A distant published probe pair gives discordant model orderings

MRK-952 and MRK-952-NC are an SGC/MSD producer-dossier probe pair [10], reported at approximately 85 nM and 10 uM biochemically, with SPR K_D 0.031 +/- 0.018 uM and 1.380 +/- 0.380 uM (n = 2; the source does not specify the ± uncertainty type). Both lie far outside this dataset's chemistry: maximum training Tanimoto 0.168 and 0.181, with no exact, salt-parent or canonical-tautomer match to any repository structure. Scored once with the frozen models, three of six methods order the pair correctly (nearest-active +0.026, RBF-SVM +0.022, equal fusion +0.007), gradient-boosted trees tie exactly, and descriptor logistic regression and random forest order it backwards (-0.043 and -0.020). Molecular weight (+33 Da), cLogP (+1.26) and similarity to the authenticated TH5427 structure (+0.032) all favour MRK-952, so even the correct orderings are reproducible by trivial descriptors and carry no method-specific credit.

For a symmetric random ordering of two distinguishable scores, either strict order has probability 0.5; this is not a calculated inferential p-value and does not cover model ties. The three concordant orderings include Equal_mean. This single already-known pair is a descriptive challenge, not a consensus failure or proof of generalization. These classifiers were not trained as potency regressors. MRK-952-NC is a weak inhibitor, not an inactive compound, and we do not use it as a decoy; the source does not state assay species or construct and explicitly assumes the negative-control stereochemistry; its curve figure labels fitted values EC50 while the dossier text reports AMP-Glo IC50. Endpoint families (IC50, EC50, SPR K_D, Kinobead apparent K_d, cell viability, qualitative calls) are kept separate throughout and never pooled.

### 4.8 Authenticated-reference substitution sensitivity

To measure the impact of the two verified reference-identity errors rather than merely flag them, we separately replace ACT-01/ACT-02 with authenticated TH5427/TH1713 graphs [1]. The original CSVs, graphs and primary diagnostic outputs remain unchanged. Fixed labels, record order, seed 42 and hyperparameters are retained; scaffolds and folds are recomputed. This is a secondary correction sensitivity, not a newly independent cohort or an outcome-selected replacement. All predictions are supplied in `transfer.json`.

| Method | Molecule AUC | Exact-scaffold AUC | Positive-series AUC |
|---|---:|---:|---:|
| Property_LR | 0.994 | 0.980 | 0.996 |
| RF | 1.000 | 0.924 | 0.642 |
| SVM_RBF | 1.000 | 0.970 | 0.745 |
| GBT | 0.930 | 0.846 | 0.289 |
| Nearest_active | 1.000 | 0.915 | 0.350 |
| Equal_mean | 0.998 | 0.931 | 0.585 |

The descriptor advantage in scaffold and series comparisons persists, as does the source-challenge pattern at 50 µM. At 1/10 µM the RF AUC becomes 0.969 rather than 1.000; the other five methods retain their original source AUCs. Three MRK pair orderings remain concordant, two reverse the reported potency ordering and GBT ties. Both favourable and unfavourable changes are retained. Reference correction is necessary chemical bookkeeping; it does not authenticate other analogue labels or add biological validation.

### 4.9 A high activity-label score does not establish paired-target selectivity

Eight source compounds have both target endpoints. Excluding training-overlap compounds 10 and 11 leaves six: three point ratios, one strict upper bound and two double-censored ratios with no finite bound. Table 3 reports every eligible compound; the complete supplement retains all 23 graphs, source roles, target-level SDs and exclusions. Figures 5–6 show both frozen scenarios separately.

Table 3. Same-paper retrospective paired evidence. IC50 in µM; R is dimensionless. Scores are uncalibrated Equal_mean, not potency/selectivity probabilities. Source SDs (Table 1, n = 2 reported biological replicates) are in the full pharmacology table; no ratio uncertainty is estimated.

| Compound | NUDT5 IC50 | NUDT14 IC50 | R / strict bound | Historical score | Reference sensitivity score |
|---|---:|---:|---|---:|---:|
| 1 | 0.837 | 0.990 | 1.182796 | 0.684990 | 0.703881 |
| 9 | 0.270 | 0.162 | 0.600000 | 0.811292 | 0.829005 |
| 12 | >50 | >50 | Not estimable | 0.254912 | 0.343391 |
| 13 | >50 | 3.72 | <0.074400 | 0.253611 | 0.340683 |
| 14 | 13.8 | 1.64 | 0.118841 | 0.325585 | 0.408507 |
| 15 | >50 | >50 | Not estimable | 0.236553 | 0.318446 |

Compound 9 has the highest Equal_mean in both scenarios despite its published dual activity and lower reported NUDT14 mean IC50. This is descriptive coexistence, not a novel biological finding or a selectivity-classifier test. These six related compounds come from one already-inspected paper; no generalization, significance or prospective validation claim follows. NUDT5/NUDT14 reaction times differ (20/60 min). The shared TH5427 zero-activity normalization and technical/biological replication wording remain unresolved comparability limitations [2, Catalytic Assays; Table 1 footnotes; Figure 1 caption]. R is neither an affinity ratio nor a clinical selectivity measure.

### 4.10 Observed proximity narrows, but does not test, the Arg51 rationale

Across four W0O sites, 1,730 residue-conformer rows comprise 1,564 observed, 58 partial and 108 null/refused rows; 1,278 retained atom pairs lie within 5.0 Å. At the primary 4.0 Å radius the shared-conformer row counts are 9/12 for 8RIY sites C/D and 11/8 for 8OTV sites C/F. These are coordinate-inventory counts, not independent biological n or pocket-completeness measures.

For 8RIY site C (label C, author AAA:301), the nearby Arg51 (label A/auth AAA, label residue 52/auth 51) minimum is 3.756 Å from W0O C20 to **backbone N**. This row is partial: Arg51 CZ has zero occupancy and is excluded. At site D (label D, author BBB:301), the Arg51 B/BBB minimum is 3.251 Å from W0O C18 to side-chain CD, whose occupancy is 0.78. Thus both nearby rows meet 4.0 Å, but only the latter meets 3.5 Å; neither minimum is a guanidinium-group witness. A residue flag cannot establish a guanidinium interaction or energetic dependence; this does not exclude other atom pairs. The opposite-chain Arg51 minima, also retained, are 14.329/14.296 Å for C/D. In 8OTV the nearby Leu107 minima are 3.609 Å (B/B at C/A:301) and 3.781 Å (A/A at F/B:302). Leu107 is reported by its own numbering, not as an Arg51 homologue.

Figure 7A–B displays separate target panels containing all residue/conformer rows with any retained pair within 5.0 Å, with both sites, units, fixed radii, partial/fractional markers and explicit missingness. The complete two-panel vector map remains in the supplement. The complete tables preserve nulls and the separate Leu47 alternatives outside that plotting range. These calculations add auditable proximity and atom identities to published structures [2], not a new binding observation. The different witnesses and occupancy caveats do not refute the authors' assignments, but cannot establish an energetic Arg51 explanation, compare TH5427 geometry or justify atom-directed redesign.

### 4.11 Status of every major conclusion

| Conclusion | Category |
|---|---|
| Twenty-fold decoy replication placed identical structures on both sides of cross-validation | Demonstrated |
| Released BEDROC normalization, permutation estimator and "leave-scaffold-out" split are incorrect as implemented | Demonstrated |
| `p < 0.001` is unattainable from the reported permutation count | Demonstrated |
| NC5-02 is canonically identical to ACT-19 and to Balikci compound 11, a published dual NUDT5/NUDT14 inhibitor | Demonstrated |
| The structure labelled TH5427 differs from the authenticated compound by one N-methyl group | Demonstrated |
| Repository labels are recoverable from a single trivial descriptor (TPSA alone, AUC 0.996) | Demonstrated |
| No positive/decoy pair satisfies the fixed seven-descriptor 0.5 full-cohort-SD caliper | Demonstrated |
| The proxy dataset, full library, docking and MM-GBSA artifacts are absent from the inspected release | Demonstrated |
| Property/source composition remains a viable alternative to interpreting high repository AUC as target recognition | Strongly supported |
| Measured-source scoring suggests method rankings depend on the evaluation cohort and label definition | Suggestive |
| Highest paired-cohort Equal_mean accompanies compound 9 dual activity, R = 0.600 | Demonstrated descriptive coexistence [2]; not selectivity validation |
| Published Arg51 rationale for ligand discrimination | Source hypothesis [1,2]; causal energetic dependence untested |
| All-site proximity preserves partial/null rows and different nearby Arg51 atom witnesses | Demonstrated coordinate calculation, not an energetic test |
| Greater TH5427 than compound-9 dependence on qualified R51A/R51K | Hypothesized; unmatched comparators, no new binding measurements |
| Physical qualification and pilot-based precision for that comparison | Unknown; no completed prospective lock |
| Published E112Q/Y74E and degradation studies distinguish catalytic and noncatalytic functions in their reported systems | Demonstrated in cited systems [4,5], not here |
| Transfer of that separation to the proposed same-line hormone/purine comparison | Hypothesized |
| Activity/selectivity of the remaining unmeasured candidates and disease-relevant effects of this prioritization | Unknown |
| Whether the historical 18,412-compound cascade was executed as described | Unknown |
| The released AUC alone establishes prospective utility or NUDT5-specific learning | Refuted |
| NC5-02 is a novel compound | Refuted |
| An MTH1-to-NUDT5 transfer-learning procedure was executed in the released code | Refuted |

These categories distinguish computation, interpretation and proposals; model agreement does not upgrade a biological claim.

## 5. Discussion

This case illustrates why independence, comparison-set construction and chemical provenance must accompany aggregate metrics. Repeated molecular identities violate the intended validation boundary. Deduplicating them is necessary but not sufficient: nearly perfect property-only discrimination remains, and a nontrivial AUC cannot on its own identify a NUDT5-specific mechanism. We cannot determine whether the fingerprint models primarily use properties, analogue relationships, true activity-related features or some combination without a stronger independent benchmark.

Descriptor and nearest-neighbour baselines, explicit chemical grouping and source-level identity checks are practical controls, not a claim of algorithmic novelty. Their implementations use documented toolkit APIs [7,8]. The contribution here is their executable application to the inspected release, including failure records and adversarial metric tests. The fixed equal-weight mean does not establish a novel transfer-learning mechanism. A real transfer study would require the source-task dataset, executed weighting, task-aware ablations, training-only tuning and an independent target-task evaluation. Neither an impressive score nor a permutation result supplies those missing experiments.

The reference discrepancy also demonstrates why a compound name must not substitute for a chemical identity. A missing N-methyl group changes the graph to which potency and similarity are being attributed. In addition, RDKit-version-dependent HBA counts show that not every numerical disagreement is a transcription error. The two relevant descriptor versions are recorded without editing the source CSV.

Biologically, Page et al. provide a rationale for studying NUDT5-dependent hormone signalling [1], while Balikci et al. establish activity and NUDT14 cross-reactivity for the rediscovered compounds [2]. Qian et al. report TNBC xenograft effects together with deaths in four of ten treated animals; the inspected text does not establish the cause of death [3]. Nguyen et al. identify a nonenzymatic PPAT-related function, and Marques et al. distinguish protein loss from catalytic inhibition in the 6-thioguanine response [4,5]. These results concern different perturbations and biological contexts. They cannot be combined into a universal claim that catalytic NUDT5 inhibition treats ER-positive breast cancer.

### 5.1 What would change the answer

The immediate discriminating experiment is the earlier H2 direct-binding comparison, now bounded by the all-site geometry. Compare authentic TH5427 and compound 9 as **unmatched comparators** on WT NUDT5 and separately qualified R51A/R51K. For each mutant M, the proposed contrast is ln[KD(M,TH5427)/KD(WT,TH5427)] minus ln[KD(M,9)/KD(WT,9)]. Sequence/lot identity, folding, oligomerization, stability, active fraction and retained ADPr-turnover controls precede interpretation; neither mutant is a validated resistance allele. SPR surface/solvent/transport checks, orthogonal binding and functional/interference controls must agree. IC50 is not substituted for KD. WT NUDT14 comparisons are required separately; no reciprocal-homolog mutation is supported.

A resolved reversed contrast, or equivalence to no differential effect under an independently justified precision rule, would challenge greater TH5427 dependence. Equal penalties do not support differential recognition. Failed protein qualification or unresolved binding is inconclusive, not a biological falsifier. Even positive contrasts would establish only comparator-specific dependence, not a causal atom or a new ligand's selectivity. The finite materials, readout, independent-pilot/precision and governance gaps are explicit in `research/structure_comparison/lab_handoff.md` and its hypothesis/control/falsification table; no sample size, variance or effect margin is invented.

Cellular attribution is a later gate: an additional compound-9 effect at matched measured NUDT5 engagement would require qualified NUDT14 perturbation/WT rescue, abundance, reporter and off-target controls, including the known BTK confound [2]. Persistence after adequate NUDT14 depletion would challenge that attribution; floor effects or failed rescue are inconclusive. The broader catalytic-versus-protein-loss programme remains in the earlier study [4,5], not a completed part of this audit. The blinded assay contract supports only normalized NUDT5 ADPr-hydrolysis inputs, not direct-binding, NUDT14 or cellular analysis. Physical qualification and a prospective lock remain unresolved.

## 6. Limitations

This is a single-dataset, small-N analysis. We did not have the original library, proxy set, docking artifacts or assay ledger, so we cannot say what the original pipeline would have produced with them, and we cannot exclude the possibility that an unreleased procedure generated the published scores. Our series holdout has two groups and is noisy. Our conditional resampling ranges ignore training-set uncertainty. Identity checks use one pinned RDKit version and exact graph matching. The source audit authenticated assay provenance for 2 of 20 raw positive labels; the other labels require curation. The novelty of the candidate structures was not established by exhaustive search. No biological experiment was performed; the paired-target analysis reuses published inhibition measurements and adds no newly measured selectivity, cellular engagement, toxicity or efficacy. All assay qualification and prospective-lock prerequisites remain unresolved. The structural map contains incomplete, occupancy-qualified deposited models under different crystallization conditions, not density-validated contacts or binding energetics. The repaired implementation passes the stated regression checks, but is not guaranteed error-free or a validated activity predictor; its own numbers inherit every limitation of the labels it is given.

## 7. Data, code and provenance

Corrected code, tests, lockfile, machine-readable results, claim-level evidence ledger (`claim_ledger.csv`), findings table (`audit.md`), publication assessment (`publication_strategy.md`) and the draft author-approval deposit correction (`deposit_correction.md`) accompany this manuscript. Original CSVs are preserved unmodified. The existing public deposits have not been altered; their metadata repeats claims this work withdraws and needs an author-approved, dated correction.

The reproducible dossier includes: `research/extension_design.md` freezes the control design before execution; `research/results/controls.json` and `research/results/transfer.json` hold assignments, predictions, metrics, similarity bins, calibration bins, ablations, conformal sets and bootstrap summaries with deterministic seeds; `research/external/`, `research/structural/` and `research/study/` hold the retrieved public-assay ledgers, structural metadata and roadmap with retrieval provenance and hashes; `research/original_inventory.json` records the preserved baseline artifacts with sizes and SHA-256 hashes. Figures and supplementary tables are regenerated from those JSON files by `scripts/build_extension_figures.py`, which fails rather than invent values when a required result file is missing. The paired release adds `research/results/selectivity.json`, its original execution-time manifest, full endpoint/score tables and PNG/PDF/SVG figures. The document CLI requires paired results by default; an explicit legacy option annotates their absence. Empty eligible cohorts receive an explicit empty-state figure, never invented zero ratios. `research/assay/` contains the future-input protocol and software-only test specification, not measured outcomes. `research/structure_comparison/` adds the archived inputs, fixed contract, observed-coordinate results, lab handoff and exact source quotations. The document builder hash-checks structural inputs and rejects missing, malformed or empty structural results by default; an explicit legacy option labels absence. Historical investigation and execution manifests retain their original scope and bytes.

## 8. Author responsibilities before any submission

Author list, affiliations, contributions, correspondence, funding, competing interests, licensing, applicable approvals for any original experiments, the final citation choices (including the Nguyen 2025 Science paper `10.1126/science.adv4257` and the Marques preprint `10.1101/2025.03.16.643557` and its 2026 Nature Communications version `10.1038/s41467-026-74489-9`), resolution of the P620-P621 reference placeholders, and an accurate disclosure of AI assistance in auditing, coding, analysis and drafting. These must be stated by the authors; they are not supplied here, and this document does not assert that human expert review has already occurred.

## 9. References

1. Page BDG et al. Targeted NUDT5 inhibitors block hormone signaling in breast cancer cells. *Nature Communications* **9**, 250 (2018). https://doi.org/10.1038/s41467-017-02293-7. Identity evidence: Figure 2, crystallography Methods, supplementary synthesis entry 28 (viewer pages 30–31); PDB 5NWH/9CH and 5NQR/958. https://www.rcsb.org/ligand/9CH ; https://www.rcsb.org/ligand/958 .
2. Balikci E et al. Unexpected Noncovalent Off-Target Activity of Clinical BTK Inhibitors Leads to Discovery of a Dual NUDT5/14 Antagonist. *Journal of Medicinal Chemistry* **67**, 7245–7259 (2024). https://doi.org/10.1021/acs.jmedchem.4c00072. Table 1 (all eight paired rows and footnotes a/b), supplementary author CSV `jm4c00072_si_002.csv` (23 rows), Catalytic Assays (sec4.3), Binding Affinity Determination (sec4.8), Protein Expression and Purification (sec4.2), crystallization/refinement (sec4.9–4.10), and Figures 1, 3–5. PDB 8RIY https://www.rcsb.org/structure/8RIY and 8OTV https://www.rcsb.org/structure/8OTV; W0O identities, exact Methods quotations and retrieval/source hashes are retained in `research/structure_comparison/`. IC50 uncertainties are reported SDs, not confidence intervals; replication/normalization ambiguities are retained. Exact inspected sections, retrieval times and source-byte hashes are in `research/selectivity/provenance.json` and `research/assay/references.json`.
3. Qian J et al. The novel phosphatase NUDT5 is a critical regulator of triple-negative breast cancer growth. *Breast Cancer Research* **26**, 23 (2024). https://doi.org/10.1186/s13058-024-01778-w. Xenograft Methods/Results and discussion; funding correction https://doi.org/10.1186/s13058-024-01814-9 does not remove the mortality observation.
4. Nguyen T-A et al. A non-enzymatic role of Nudix hydrolase 5 in repressing purine de novo synthesis. *Science* **390**, 1143–1150 (2025). https://doi.org/10.1126/science.adv4257. Figures 4–6 and corresponding Results; bibliography verified against Crossref and the primary author manuscript. This is not the distinct Wu paper with DOI `science.adx9717`.
5. Marques A-SMC et al. Targeted Protein Degradation of NUDT5 Dissociates Catalytic Inhibition from Protein Loss in 6-Thioguanine Response. *Nature Communications* **17**, 8192 (2026). https://doi.org/10.1038/s41467-026-74489-9. Figures 1 and 4 and degradation/rescue Results. Related 2025 preprint: https://doi.org/10.1101/2025.03.16.643557; the preprint and article are not independent replications.
6. Phipson B, Smyth GK. Permutation P-values Should Never Be Zero: Calculating Exact P-values When Permutations Are Randomly Drawn. *Statistical Applications in Genetics and Molecular Biology* **9**, Article 39 (2010). https://doi.org/10.2202/1544-6115.1585. Bibliographic metadata/abstract inspected; the finite-p-value arithmetic here is directly derived and unit-tested.
7. RDKit. `rdkit.ML.Scoring.Scoring.CalcBEDROC` and Morgan fingerprint APIs. https://www.rdkit.org/docs/source/rdkit.ML.Scoring.Scoring.html ; https://www.rdkit.org/docs/source/rdkit.Chem.rdFingerprintGenerator.html . The installed pinned implementation is used as the numerical BEDROC reference; reference agreement is not a biological validation.
8. Scikit-learn. `StratifiedGroupKFold`. https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedGroupKFold.html . The supplied manifest fixes the implementation at version 1.7.2; group identities and assignments are released.
9. Original data/software deposits: IEEE DataPort https://doi.org/10.21227/cbef-k354 and Zenodo https://doi.org/10.5281/zenodo.19374517 . These are deposits, not evidence of peer-reviewed inhibitor discovery. Archive and input hashes are supplied in the accompanying provenance records.

10. Structural Genomics Consortium / MSD. MRK-952 chemical-probe dossier and linked experimental information. https://www.thesgc.org/chemical-probes/mrk-952 . Linked source figures: https://www.thesgc.org/sites/default/files/inline-images/download_17.png and https://www.thesgc.org/sites/default/files/inline-images/download%20%281%29_13.png . Retrieved during this audit; the supplied source ledger preserves URLs, endpoint discrepancies and assay/species/construct limitations. This is a public probe dossier, not an independent experiment performed here.
