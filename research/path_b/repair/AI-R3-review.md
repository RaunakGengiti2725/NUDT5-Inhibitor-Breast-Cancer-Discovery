# AI adversarial review of the Path B revision

**Label: AI review, not human peer review.** Produced by an automated reviewer with read-only access to the supplied revision. It is not a journal decision and no human expert signed it.

**Revision reviewed:** commit `22830642f7d33ca3319872b5ba4c513baff50154` on branch `devin/1791243628-path-b-manuscript` (declared baseline `40b9b0708d888a015abe5043bb273c3c6ee601ae`).
**Review package SHA-256:** `63a8c1f1fde68b51f0544129dda997ad3a7618b0c1043428b7a1206aed91f517`.
**Working tree during review:** clean, `git status --porcelain` empty, no tracked file edited, no model refitted.

## Recommendation

**Reject** for the conditional target, the Journal of Molecular Graphics and Modelling specialist venue.

This is a contribution judgment, not an allegation of fabrication. The decisive quantities reproduce exactly from the frozen inputs, and the manuscript is unusually disciplined about its own boundaries. The problem is that the central new result is a recalculation of atom-level proximity inside deposited structures whose source publication already reports the same Arg51 contact, with no independent placement validation, omit or polder map, rerefinement, energy calculation, new measurement or validated predictive method. The journal's own scope text, retained in `research/path_b/source_checks/sources/jmgm_scope_excerpts.txt`, states that "Routine applications of standard modelling approaches, providing only very limited new scientific insight, will not meet our criteria for publication." On the evidence actually supplied, this revision falls on the wrong side of that sentence.

A reproducibility, data or methods-audit venue could reasonably judge the same package differently. That is an editorial decision and the package does not establish it: the revision's own source register records that the journal author guide returned an access failure, so article category, limits and editorial fit remain uncertified.

## What I verified independently

All checks were run in an external directory from frozen repository inputs, with no fitting and no writes inside the repository. Input hashes are recorded in `recomputed_statistics.json`.

- Package integrity: 136 of 136 release-inventory checksums, and all 294 tracked files identical to the repository-source archive.
- Source fidelity: all 46 paired endpoint cells in `research/selectivity/paired_evidence.json` match the author CSV byte for byte, including the two "not active" censored rows per compound.
- All eight paired ratio states and every finite ratio, including compound 9 at 0.162/0.270 = 0.600 and the strict bound 3.72/50 = 0.0744.
- Pooled AUCs, fold-level AUCs, five-bin descriptive ECE, Brier scores, all 17 stored ablation AUCs and the raw four-score equal-weight mean, which agrees with the stored `Equal_mean` to within 1e-14.
- Fixed-score scaffold bootstrap quantiles and the paired AUC-difference interval, reproduced to the stored values.
- Conformal prediction sets, per-fold calibration class counts and minimum p-values, recomputed from stored p-values.
- Arg51 minima recomputed directly from `8RIY.cif` with an independent script: auth chain AAA, N to W0O C20, 3.7556 Å at occupancy 1.000; auth chain BBB, CD to W0O C18, 3.2513 Å at occupancy 0.780. Both match the manuscript.

So the arithmetic is sound. Everything below concerns inference, estimand choice, display and significance.

## The single objection most likely to kill the paper

**Objection 1.** The incremental contribution is not demonstrated. Everything else is repairable; this is not.

## Numbered objections

### 1. The new scientific contribution is not demonstrated (critical, significance)

Locations: `research/manuscript.md:13-17`, `:49-61`, `:124-130`, `:144-146`; `research/path_b/diagnostic_supplement.md:61`; `research/path_b/laboratory_specification.md` (conditional, unexecuted).

The source publication already states the result the structural section qualifies. Its Figure 3 caption, retained in the authenticated XML at `research/selectivity/sources/balikci.xml.gz`, reads: "An additional hydrophobic interaction with R51 in chain B and a hydrogen bond with the main chain of E47 in chain A can be observed." The same sentence appears for compound 1 in the Figure 1 caption. The source also deposited 8RIY and 8OTV. This revision recomputes distances, exposes occupancy and partial or null residues, and keeps both chains and all four W0O sites. That is a cleaner description of a published model. It does not show the source placement is wrong, does not supply independent density evidence, and the manuscript itself concedes at `:130` that "Distance recalculation alone does not establish a scientific novelty threshold."

Classification: **nothing currently available.** No rewrite closes this. Closing it needs a pre-specified, independently validated structural analysis across a justified set of complexes, or an executed orthogonal experiment or energy analysis. Neither exists in the revision and neither may be invented.

Venue consequence: rejection at a specialist modelling venue even after perfect copyediting. The honest alternatives are a reproducibility or data-audit venue with a reframed paper, or execution of the conditional laboratory specification before resubmission.

### 2. "Counterexample" claims more than a ratio of two reported means can identify (major)

Locations: `research/manuscript.md:93`, Methods `:35-37`, Table 2 `:76-93`; `scripts/scripts/selectivity.py:290-333`.

Compound 9's R = 0.600 is correct, and the manuscript does say R is a ratio of reported catalytic means rather than affinity. But the score being contradicted is trained on unverified source labels against unmatched, unassayed decoys, and is not a target-selectivity estimator at all. The paired assays are not protocol-matched: the source methods section states NUDT5 reactions ran 20 minutes and NUDT14 reactions one hour, both normalized to 500 nM TH5427 as the zero-activity control. There are no raw paired replicates, no covariance and no ratio uncertainty. What the data support is a descriptive discordance between an uncalibrated label score and one protocol-bound ratio, not a counterexample to a selectivity prediction the method never makes.

Classification: **existing evidence**, wording repair. Replace "counterexample" with a descriptive discordance under the two reported protocols, and state that it challenges only the invalid reading of the score. Do not manufacture a ratio interval from summary SDs.

### 3. Six-decimal ratios imply precision the reported summaries cannot carry (major)

Locations: `research/manuscript.md:35-37`, `:76-93`, `:136`; `research/selectivity/paired_evidence.json`; `research/selectivity/assay_conditions.json`.

Table 2 prints mean ± SD beside ratios at six decimals. The source reports n = 2 independent biological replicates with triplicate technical sets, which the revision preserves honestly, and it explicitly supplies no ratio CI. The residual problem is reader-facing: 0.540041 looks like a measured quantity when it is an exact arithmetic function of two rounded means.

Classification: **existing evidence** for the display repair; **nothing currently available** for any inferential upgrade. Round to source-appropriate precision or label the values as computed from displayed source means, and note that the displayed precision is arithmetic, not measurement precision.

### 4. The conformal diagnostic is near-vacuous in the similarity split (major)

Locations: `research/path_b/diagnostic_supplement.md:33-41`; generated `documents/diagnostic_supplement.md:257-268`; `scripts/scripts/controls.py:244-330`.

Recomputed from the stored outputs: in `similarity_component_split` the class-1 calibration count is 2 in every one of the five outer folds and the class-1 minimum p-value is 1/3, so class 1 enters every prediction set at both α = 0.1 and α = 0.2. Reported coverage is 0.978 and 0.956, while 43 of 45 and 42 of 45 sets contain both labels. In `full_valid_set` at α = 0.1, 22 of 45 molecules are in folds where both classes are force-included by the minimum p-value alone. The code is correct and warns that there is no shifted-population guarantee; the issue is that the published coverage column reads as reassurance for a rule with almost no discriminatory content for the positive class.

Classification: **existing evidence**, display repair, no refitting. Put per-class calibration counts, minimum p-values and set-size frequencies next to each cohort, and label the similarity split a sparse-calibration stress test.

### 5. Pooled cross-fold AUC and within-fold discrimination are different estimands (major)

Locations: `research/manuscript.md:97-120`; `research/path_b/diagnostic_supplement.md:33`; `scripts/scripts/controls.py:444-588`.

On `full_valid_set`, recomputation gives Equal_mean pooled 0.9312 versus within-test-fold pair-weighted 0.9655, and Property_LR 0.9798 versus 1.0000, over 87 within-fold comparable pairs against 494 pooled pairs. The fold-prevalence control is the clean demonstration: pooled 0.2460, within-fold exactly 0.5000 in all five folds, and pooled 0.1569 in the similarity split. The manuscript does define pooled AUC and does warn at `:120` that fold-constant scores can fall below chance, but Table 3 still reads as a single out-of-sample ranking number.

Classification: **existing evidence**, display repair. Report both estimands side by side with the pair counts and say why the pooled one is retained.

### 6. The bootstrap is conditional on fixed scores and on one grouping choice (major)

Locations: `research/manuscript.md:43`, `:120`; `research/path_b/diagnostic_supplement.md:37`; `scripts/scripts/controls.py:334-364`, `:406-441`.

The 1,000 draws resample scaffold groups from frozen out-of-fold predictions and never refit; the stored output says so with `model_refitted: false`. Stored 95% conditional ranges on `full_valid_set` are Equal_mean [0.6849, 1.0000], Property_LR [0.9082, 1.0000] and the paired difference [-0.2464, 0]. Two further points: these are not uncertainty for the fitted workflow or for any future compound population, and the grouping choice matters. Resampling the stored Tanimoto components instead of exact scaffolds in the similarity split moves the Equal_mean range from [0.9339, 1.0000] to [0.9286, 1.0000] and produces 17 single-class draws, which the exact-scaffold version never encounters. The headline range is therefore conditional on both the frozen scores and the grouping definition.

Classification: **existing evidence** for labelling and for disclosing the grouping sensitivity; **nothing currently available** for workflow-level uncertainty, which would need refitting inside resamples and a defensible target population.

### 7. Panel breadth creates selection risk even without a formal test (moderate)

Locations: `research/manuscript.md:97-120`; `research/path_b/diagnostic_supplement.md:65-67` with the generated annex; `scripts/build_path_b_documents.py:322-345`.

The package evaluates six methods, 17 ablations, three cohorts, five seeds, three source cutoffs, reference sensitivity, calibration, two conformal levels and a transfer challenge. It correctly calls these diagnostics and claims no winner. Still, two of the single-descriptor controls reach 0.9960 pooled AUC, above the consensus score, which the manuscript concedes at `:120`; and prominence in tables and figures is an outcome-facing choice across a broad panel.

Classification: **existing evidence**, framing repair. Add a compact inventory of every cohort, method, seed and cutoff evaluated, and qualify any superlative as within this stored panel.

### 8. Figure S1B attributes joint seed randomness to decoy partitioning alone (minor to moderate, actual mislabel)

Locations: generated `documents/diagnostics.png` panel B, titled "Sensitivity to decoy partitions", produced at `scripts/build_research_documents.py:102-115`; `scripts/scripts/pipeline.py:410-492`.

The same seed drives `series_splits` decoy allocation and the model fitting and split generation inside `fit_scores` and `out_of_fold`. The plotted seed 42–46 curve therefore mixes both sources. The caption's "no best-seed selection" is good practice; the panel title is simply wrong about what varies.

Classification: **existing evidence**. Rename to seed sensitivity of the series-holdout diagnostic, or hold fitting randomness fixed and vary only decoy allocation.

### 9. The stated undefined-denominator policy contradicts the generated table (minor, actual inconsistency)

Locations: `research/path_b/diagnostic_supplement.md:35` ("A zero denominator remains undefined rather than becoming zero"); generated `documents/diagnostic_supplement.md:76`, `:90`, `:104` and `:117`; `scripts/scripts/controls.py:144-176`.

`sklearn.metrics.matthews_corrcoef` returns 0.0 for constant predictions, and the annex duly publishes MCC = 0.0000 for every `Constant_0_5` row and for `Train_prevalence` in two cohorts. Nothing scientific turns on it, and the same table correctly reports `precision_at_0_5` as null elsewhere, which makes the MCC convention look deliberate when it is inherited.

Classification: **existing evidence**. Either report those MCC cells as undefined, or state that the implementation follows sklearn's zero convention, then regenerate the annex.

## What I decided not to object to

- The equal-weight mean is implemented as described, is not the historical normalized TWCS, and the manuscript keeps the two separate.
- Censoring handling is strict and correct: "not active" is kept as a bound, double-censored pairs yield no ratio, and no inactive potency is invented.
- Descriptive ECE is genuinely implemented in `controls.py:179-202`. Any earlier claim that it is absent is wrong. Bin-count sensitivity is modest, Equal_mean moving 0.0627 at five bins to 0.0673 at ten.
- The grouped conformal implementation matches its stated empirical-only scope; objection 4 is about sparse calibration and display, not a hidden guarantee.
- The structural extraction preserves both chains, all four W0O sites, occupancy, alternates and partial or null rows, and states that crystal copies are nonindependent. My independent distances confirm it. That does not make it energy or refute the source.
- Figure S4's source challenge is correctly flagged as not controlled validation. With only ten eligible endpoints and five threshold-positives at 50 µM, the 1.000 AUC for four of six methods carries almost no information; the figure says as much.
- Tests, hashes, page-identity checks and build success are software and provenance evidence. I did not treat any of them as scientific validation, and neither does the manuscript.
- Access failures for the Nguyen full text and the journal guide are recorded as failures. I did not infer any source contradiction from them.

## Unanswerable objections and their venue consequences

1. No independent biological or energetic result, and no demonstrated general method, is available. Consequence: the specialist-venue contribution threshold stays unmet however well the paper is written.
2. Raw paired replicates, covariance, full nesting and protocol-matched comparability are unavailable. Consequence: no ratio interval, no selectivity claim and no harmonized cross-target comparison can be certified; the ratios must stay descriptive.
3. The journal author guide was inaccessible and author-owned declarations, including Nikhil Srinivasan's retained contribution and authorship, remain open. Consequence: submission compliance cannot be assessed here, and these gates are author decisions, not review findings.

## Closing

The revision is a trustworthy audit artifact and a clear improvement on the narrative it replaces. Objections 2 through 9 are all repairable from material already in the package, and none of them would change my recommendation if fixed, because none of them is the reason for it. The reason is objection 1: the paper tells the reader, accurately, that it has recalculated distances in someone else's deposited structures. For this venue that is not enough, and no amount of further polishing will make it enough.
