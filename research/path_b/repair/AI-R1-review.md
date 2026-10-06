# Adversarial review of the Path B structural reanalysis revision

**This is an AI review, not human peer review.** It was produced by an automated reviewer with repository and public-source access. It is not an editorial decision, not a certification of journal compliance, and not a substitute for qualified human referees or a crystallographic expert.

## 0. What was reviewed

| Item | Value |
|---|---|
| Repository | `RaunakGengiti2725/NUDT5-Inhibitor-Breast-Cancer-Discovery` |
| Reviewed commit | `22830642f7d33ca3319872b5ba4c513baff50154` |
| Reviewed tree | `da847ec8c141598974bd81f9daffd37f96d7d0b9` |
| Branch named by the request | `devin/1791243628-path-b-manuscript` |
| Immutable baseline cited by the revision | `40b9b0708d888a015abe5043bb273c3c6ee601ae` |
| Review package SHA256 | `63a8c1f1fde68b51f0544129dda997ad3a7618b0c1043428b7a1206aed91f517` |
| Release manifest artifacts verified | 136 of 136 hashes matched |
| Manuscript / supplement rendered | 11 pages / 22 pages, read as PDF text and page images |

The review was read-only. No tracked file was modified, nothing was committed, pushed, merged, or submitted anywhere. Checks I ran myself, in an external pinned environment built from the repository locks:

- Recomputed every Table 3 ROC-AUC from the stored out-of-fold scores in `research/results/controls.json`.
- Recomputed `Equal_mean` from its four components (maximum absolute deviation 1.1e-16), and the descriptive ECE (0.0627, identical to stored).
- Recomputed raw-graph properties, class overlap, canonical identity and reference-graph formulae from `compounds.csv`, `final_hits.csv`, `research/source_assays.csv`, `research/reference_structures.csv`.
- Recomputed source-challenge eligibility, censoring and cutoff AUCs from `research/results/transfer.json`; paired rows and ratios from `research/results/selectivity.json`.
- Recomputed the deposited W0O site inventory and the Arg51 minima directly from the archived `8RIY.cif` / `8OTV.cif` coordinates with Gemmi, independently of the project's own extraction code.
- Read the official wwPDB validation XML for both entries directly.
- Reproduced `identity_summary.json` byte-for-byte with the repository's own driver, and ran 32 source-check and document tests (all passed).

**Every number I checked in the manuscript and supplement reproduced.** I found no arithmetic error, no misreported structural value, and no fabricated source quotation. The objections below are therefore almost entirely about *what the work establishes* and *whether it is enough for this venue*, not about whether its numbers are right. That distinction is maintained explicitly in Section 3.

## 1. Recommendation

**Reject** for the conditional *Journal of Molecular Graphics and Modelling* specialist target.

This is not a recommendation to reject because the paper is dishonest or wrong. It is unusually careful, and its disclosure discipline is better than much of the literature it cites. It is a recommendation to reject because, after every claim has been correctly narrowed, what remains is a recalculation and reorganization of already-published deposited coordinates and already-published measurements, plus a set of diagnostics whose own results show they establish nothing about the target. The official JMGM scope states that routine applications of existing methods with little new scientific insight do not meet its publication criteria. By the authors' own correct admission (`research/manuscript.md:17`, `:130`, `:144`), the contribution is descriptive and methodological and "distance recalculation alone does not establish a scientific novelty threshold." I agree with that sentence, and it decides the venue question against the paper.

The honest content would fit a data-descriptor, reproducibility note, or short technical-comment format better than a modelling research article.

## 2. The single objection most likely to kill the paper

**O1. There is no demonstrated new scientific insight — only a more careful description of a published result.** (significance risk; nothing currently available)

The published source already reports compound 9 as a dual NUDT5/NUDT14 antagonist, deposits both complexes (8RIY, 8OTV), reports the paired catalytic IC50 values, and states the chain-B R51 hydrophobic-interaction interpretation in its own figure caption, which I read directly in the archived source (`research/structure_comparison/sources/PMC11089510.xml`, Figure 3 caption: "An additional hydrophobic interaction with R51 in chain B"). The reviewed manuscript adds: all four W0O sites instead of the representative ones, both receptor chains, explicit atom witnesses, occupancy and alternate-conformer bookkeeping, extracted report RSCC/RSR, and fixed slices of precomputed PDBe maps.

Each of those is correct and verifiable, and none of them changes a scientific conclusion. The manuscript states this itself: the observations "qualify the published chain-specific account without testing or refuting its energetic interpretation" (`research/manuscript.md:8`), and "An energetic claim would require additional evidence" (`:15`). The strongest candidate for a new finding — that the nearest retained Arg51 witnesses are a backbone N at 3.756 Å in chain AAA and a fractional-occupancy side-chain CD at 3.251 Å in BBB, with neither being a guanidinium atom (`:59`) — is a true and well-evidenced observation that the manuscript then correctly refuses to convert into any claim about binding. A reviewer is left with a corrected granularity of description and no altered understanding of the system.

Nothing currently available answers this. The diagnostics cannot supply it (O2–O9). The structural work cannot supply it without omit/polder maps, rerefinement, or expert 3-D review, all of which the manuscript lists as not performed (`:134`, G-S4 in the gap register). The discriminating experiment that would make the Arg51 observation consequential is specified but explicitly unexecuted and unauthorized (`:128`). **Venue consequence:** for JMGM this is dispositive, because insufficient new insight is an explicit scope exclusion rather than a fixable presentation defect. No rewrite of this dataset resolves it; only new evidence would, and new evidence is out of scope for this revision.

## 3. Objections

Classification per objection: **category** is `actual error`, `disclosure asymmetry` (accurate but unevenly presented), or `significance risk`; **answerability** is `existing evidence`, `specified new analysis`, or `nothing currently available`.

### 3a. Target recognition and the decoy problem

**O2. The label task is solvable by molecular weight alone, which makes the entire diagnostic panel a property detector.** (significance risk; existing evidence)
Location: `research/manuscript.md:95–120` (Table 3), `research/path_b/diagnostic_supplement.md:11`, `research/results/controls.json`.
My independent recomputation from `compounds.csv`: across the 45 valid unique records, source-positive molecular weights span 303.3–491.3 and decoy weights span 206.3–291.7. **The two ranges do not overlap.** Raw-descriptor separation is therefore perfect (MW AUC 1.000; HBA and TPSA 0.998 each). The manuscript reports out-of-fold single-descriptor logistic AUCs of 0.9879 for MW and 0.9960 for HBA and TPSA, and says only that descriptor controls "discriminate these labels at least as well numerically as the current consensus" and that this "is compatible with dataset composition effects but does not identify a unique cause" (`:120`). That wording is too weak for what the data show: the classes are trivially linearly separable on a single trivial descriptor, so no result computed on this cohort — consensus, fingerprint, similarity, or conformal — can carry any information about NUDT5 recognition. The paper is already committed to this conclusion ("High label AUC cannot establish target recognition", `:120`); it should state the separability fact that forces it.

**O3. The written decoy-matching criteria are not satisfied by most of the decoys, and the manuscript stops short of saying so.** (disclosure asymmetry; existing evidence)
Location: `research/path_b/diagnostic_supplement.md:11`, `compounds.csv`.
The supplement correctly corrects the historical false denial and records that the historical DOCX did state joint criteria of |ΔMW| < 50, |ΔcLogP| < 1.0 and Tanimoto < 0.3. I applied exactly those criteria myself over the retained valid graphs: only **9 of 26 decoys** satisfy them against any valid source-positive, and the only positives ever matched are ACT-19 and ACT-20. So the stated design is not merely unverified for provenance — it is contradicted by the retained data for roughly two-thirds of the decoy set. The supplement says the original code "lacks an executed joint matching check and per-decoy matching provenance", which is accurate but reads as an absence of evidence when the available evidence is actively negative.

**O4. The exploratory property-matched subset does not repair the imbalance and should not be presented as a balance control.** (significance risk; existing evidence)
Location: `research/path_b/diagnostic_supplement.md:43`, supplement Figure S2, supplementary results table (`nearest_property_subset`).
On that subset Property_LR rises to 0.9945 while Equal_mean is 0.9280. The supplement concedes that "Matching did not remove the diagnosed imbalance". A reader skimming the figure panel titled "Measure balance; do not assume it" may still take the subset as a bias control; it is a demonstration that the bias survives matching.

**O5. Exact-scaffold grouping leaves heavy analogue leakage, so the headline split is not the generalization probe its name implies.** (significance risk; existing evidence)
Location: `research/path_b/diagnostic_supplement.md:25`, `scripts/scripts/pipeline.py` out-of-fold grouping, `research/results/controls.json` fold assignments.
Recomputing nearest cross-fold training-positive similarity from the stored fold assignments: **16 of 19 source-positives have Tanimoto ≥ 0.70 to a positive in another fold** under the paper's own radius-2/2048-bit representation. Folds are also severely unbalanced (fold 0 holds 11 of 19 positives; remaining folds hold 1–3). Within-fold AUC is 1.000 for Property_LR in all five folds and for Equal_mean in four of five. The supplement does disclose that exact-scaffold grouping "prevents exact scaffold sharing, not all medicinal-chemistry-series overlap"; the quantitative severity belongs in the main text beside Table 3, because it explains why every method saturates.

**O6. The one split that breaks series structure collapses the consensus, which argues against any methodological contribution.** (significance risk; existing evidence)
Location: `research/results/benchmark.json` (`series`), supplement Figure S1A, `research/path_b/diagnostic_supplement.md:25`.
Under the positive-series holdout I recompute Equal_mean 0.5628 and Nearest_active 0.3289 — at or below chance — while Property_LR stays at 0.9980. The consensus method that the historical work promoted therefore has no demonstrated value over a seven-descriptor logistic baseline anywhere in this package, and loses to it badly as soon as analogue structure is removed. The supplement notes the two-fold design is noisy and partition-sensitive, which is fair, but the direction of the result is consistent across all designs and is unflattering.

### 3b. Compound identity, overlap and retrospective exposure

**O7. The "retrospective counterexample" rests on a single compound whose score is a near-neighbour artifact.** (significance risk; existing evidence)
Location: `research/manuscript.md:93`, `research/results/transfer.json`, `research/results/selectivity.json`.
I confirm compound 9 has the top stored Equal_mean among the six eligible non-overlapping paired compounds in both scenarios, and that its reported catalytic ratio is 0.600 — the manuscript's arithmetic is exact. But compound 9's nearest retained training compound is ACT-20 at Tanimoto 0.660, and ACT-20 is itself source compound 10 from the same publication; the eligible cohort is six compounds with three finite ratios. The counterexample is correctly labelled as not a calibrated selectivity test, yet it is doing visible rhetorical work in Section 3.2 while resting on one analogue of a training compound.

**O8. The source challenge is an exposed same-paper analogue set whose perfect scores are driven by similarity to training actives.** (significance risk; existing evidence)
Location: `research/manuscript.md:74–93`, `research/path_b/diagnostic_supplement.md:47–49`, `scripts/scripts/transfer.py:136–196`, supplement Figure S4A.
Recomputed eligibility: of 23 scored source structures, 11 are excluded as untested in that dataset and 2 (compounds 10 and 11) as training identity/parent overlaps, leaving n = 10. At the 1 µM and 10 µM cutoffs only **2** compounds are threshold-positive; at 50 µM, 5 are. Equal_mean, Nearest_active, RF and SVM_RBF all score AUC 1.000 at every cutoff, while Property_LR falls to 0.5625 at the tighter cutoffs. That pattern — nearest-neighbour similarity at a perfect 1.000 and properties at chance — is the signature of same-series proximity, not target recognition. The supplement already says this is "an exposed single-publication challenge with related chemistry, not a blinded holdout"; Figure S4's caption "stress tests without tuning" nonetheless oversells it, since freedom from tuning in the frozen code cannot be verified for the analytical history that selected this source.

**O9. Compound identity and overlap conclusions are correct, and they remove the discovery narrative rather than supporting one.** (no error; existing evidence)
Location: `research/path_b/diagnostic_supplement.md:13–15`, `research/path_b/source_checks/identity_summary.json`.
Independent canonicalization confirms NC5-02 = ACT-19 = Balikci compound 11 (C17H13N5O), that this is the only candidate overlapping the training pool, that ACT-01's graph (C19H18Cl2N8O3, 32 heavy atoms) is not authenticated TH5427 (C20H20Cl2N8O3, 33), and that ACT-02 (C19H19ClN8O3) is not authenticated TH1713 (C19H21N7O3). ACT-18 remains unparseable and was not repaired. I reproduced `identity_summary.json` byte-for-byte. This is diligent work; it is also the reason no discovery claim survives, and the repository still ships `final_hits.csv` with historical candidate rows that a careless reader could mistake for results.

### 3c. Reference sensitivity and score meaning

**O10. Identical AUCs understate real per-compound instability under the reference substitution.** (disclosure asymmetry; existing evidence)
Location: `research/path_b/diagnostic_supplement.md:15`, `research/results/transfer.json` (`reference_sensitivity`), `scripts/scripts/transfer.py:293–358`.
Substituting only the two misidentified reference graphs leaves the scaffold-split Equal_mean AUC numerically unchanged (0.93117 in both). But recomputing from the stored predictions, **13 compounds change fold assignment**, ACT-02's Equal_mean moves by −0.207, and one record (DEC-19) crosses the 0.5 threshold. Reporting ordering preservation and unchanged AUC, without these movements, makes the pipeline look more robust to its own corrected inputs than it is. Fully answerable from data already in the repository.

**O11. Pooled Table 3 mixes separately fitted fold models and is presented without the uncertainty the package already computed.** (disclosure asymmetry; existing evidence)
Location: `research/manuscript.md:95–120`, `research/results/controls.json` (`paired_auc_difference`).
The stored paired conditional interval for Equal_mean minus Property_LR is [−0.2464, 0.0000] with a point difference of −0.0486 over 1000 draws without refitting. The main text asserts the qualitative comparison but omits the interval, whose upper limit is exactly zero. The supplement properly warns these are not population confidence intervals. The main-text claim is weaker than the supplement's own numbers suggest, in the paper's favour, and should carry the interval.

**O12. Descriptive ECE and grouped conformal output are reported in a way that can still read as calibration and coverage.** (disclosure asymmetry; existing evidence)
Location: `research/path_b/diagnostic_supplement.md:35`, `:41`, supplement conformal table, `scripts/scripts/controls.py:179–202`.
I verified that `calibration_bins` computes a descriptive bin-weighted ECE only (0.0627 for Equal_mean, reproduced exactly) and that no fused-score calibration model is trained. The supplement states both facts. The residual risk is presentational: a reliability diagram and a table of "empirical coverage" against "nominal coverage" invite exactly the inference the text disclaims, on 45 molecules with unverified labels.

### 3d. Structural reanalysis and model support

**O13. The abstract foregrounds the favourable report metrics while the two fields that speak most directly to per-atom ligand support are set aside as undefined — and they are unfavourable for the NUDT5 entry carrying the Arg51 argument.** (disclosure asymmetry; specified new analysis)
Location: `research/manuscript.md:8` (abstract), `:29–31` (Methods), supplement model-support table, `research/structure_comparison/model_support/README.md:93–99`, `:172`.
From the official validation XML I read for the four W0O sites: 8RIY AAA 301 has EDIAm 0.412 and OPIA 16.67; 8RIY BBB 301 has EDIAm 0.410 and OPIA 26.67; 8OTV A 301 has 0.804 / 83.33; 8OTV B 302 has 0.728 / 60.00. The Arg51 residues themselves carry EDIAm 0.199 / OPIA 18.18 (AAA) and 0.413 / 45.45 (BBB), with residue RSCC 0.894 and 0.901. The abstract and the supplement's model-support table report only RSCC 0.928–0.952 and RSR 0.076–0.097, and the manuscript declines to interpret EDIAm/OPIA "because source definitions were not recorded" (`:31`). The values do appear, unreduced, in the repository README — so this is a presentation asymmetry, not concealment. But interpreting two whole-ligand metrics from a report row while declining to interpret two per-atom metrics from the same row, when the latter are the less flattering ones and bear on exactly the local placement question the paper is about, is not defensible on a definitional technicality. **Specified new analysis:** retrieve and cite the published EDIA/EDIAm and OPIA definitions from the primary methods literature, then report all four fields for all four sites and both Arg51 residues in the manuscript, or report none of the report-derived metrics and rely only on coordinates. No new computation on the structures is required.

**O14. Chain-specific structural claims are accurate, and I could not break them.** (no error; existing evidence)
Location: `research/manuscript.md:8`, `:49–72`, `:59`.
Independently from the archived coordinates with Gemmi: four W0O sites with 30 positive-occupancy heavy atoms each; 8RIY AAA Arg51 N to W0O C20 at 3.7556 Å with occupancy 1.00; 8RIY BBB Arg51 CD to C18 at 3.2513 Å with occupancy 0.78; cross-chain minima both above 14 Å. AAA Arg51 CZ is deposited at occupancy 0.000. The official report lists, for AAA Arg51 only, three bond outliers (CZ–NH1 Z = 45.44, CZ–NH2 Z = 29.26, NE–CZ Z = −31.12), three angle outliers, a side-chain plane outlier and eight clashes; BBB Arg51 carries no outlier child elements. Every one of these matches the manuscript, including the claim that neither witness is a guanidinium atom. The 1730 residue-conformer rows and 1278 atom pairs within 5 Å also reproduce.

**O15. The site-selection rule is retrospective and unregistered, which the paper concedes but which still limits the model-support section.** (significance risk; nothing currently available)
Location: `research/manuscript.md:29`, `research/path_b/diagnostic_supplement.md:29`.
The rule "was set after inspecting reports and was not preregistered", and `extension_design.md` is explicitly not external preregistration. Code inspection cannot certify the absence of earlier exploration. There is no artifact that can retire this, and I do not treat its absence as evidence of misconduct.

**O16. The map work is bounded to the point where it cannot support the section's framing.** (significance risk; specified new analysis)
Location: `research/manuscript.md:31`, `:134`, supplement Figures S7–S10, gap item G-S4.
Precomputed model-dependent PDBe maps, fixed slices, trilinear sampling, no omit or polder maps, no rerefinement, no expert 3-D review, no independent placement validation, and no recorded map-generation version. The disclosures are complete and correct. The consequence is that "limited model-support inspection" is the most that can be said, and it is not a modelling contribution. **Specified new analysis** that would make it one: omit or polder map computation at the four W0O sites plus local rerefinement and independent placement assessment, with the EDIAm/OPIA interpretation of O13. That is a defined crystallographic reanalysis of public data; it is not the Path A rebuild and not a laboratory experiment.

### 3e. Endpoint semantics and pharmacology

**O17. The paired-ratio section is correct and correctly limited; its scientific yield is close to zero.** (significance risk; nothing currently available)
Location: `research/manuscript.md:33–37`, `:74–93`, `:136`.
All eight paired rows and the 0.600 ratio reproduce exactly from `research/results/selectivity.json`, and the source conditions I read in the archived XML confirm 20-minute NUDT5 and 60-minute NUDT14 protocols, shared 500 nM TH5427 normalization, and the unresolved "triplicate sets" versus "two independent biological replicates" wording. Because raw replicate data are unavailable, no ratio confidence interval, covariance or inferential comparison can exist. What remains is a correctly annotated restatement of a published table. Only the authors' raw replicate data or new measurements could change this, and neither is available.

**O18. Endpoint separation is handled well.** (no error; existing evidence)
Location: `research/path_b/diagnostic_supplement.md:53`, `research/path_b/source_checks/source_claims.json`.
Catalytic IC50, SPR KD (approximately 250 nM NUDT5 and 400 nM NUDT14 for compound 9, which I verified in the source figure captions), cellular engagement EC50, Kinobead apparent affinity and viability are kept distinct, and the NUDT14 Leu107 environment is explicitly not mapped onto NUDT5 Arg51. I found no endpoint conflation.

### 3f. Provenance claims that are not scientific evidence

**O19. Software and provenance verification is extensive and proves nothing biological; the paper says so, and the review package's framing still risks the inference.** (disclosure asymmetry; existing evidence)
Location: revision summary `verified` list, `research/manuscript.md:45`, `research/path_b/diagnostic_supplement.md:59`.
I reproduced `identity_summary.json` byte-for-byte, verified all 136 release hashes and the package SHA256, and ran 32 source-check and document tests, all passing. The manuscript correctly states that tests "verify computational behavior, not the validity of source labels or biological hypotheses." I record the passing checks as software evidence only and give them no weight in the recommendation.

**O20. Unresolved author-owned and journal-specific items make the revision non-submittable independently of its science.** (significance risk; nothing currently available)
Location: `research/manuscript.md:3`, `:140`, `research/path_b/diagnostic_supplement.md:63`.
Affiliation, funding, COI, CRediT, approvals, rights, prior-version and submission history, acknowledgments, reviewer conflicts, complete AI disclosure, and Nikhil Srinivasan's retained contributions and authorship all remain open and are correctly marked author-owned. The JMGM author guide returned 403 under bounded access, so no numerical limit, mandatory section or checklist compliance is certified. I treat none of this as a scientific defect, and the inaccessibility of the guide and of the Nguyen full text is not evidence against any publication.

## 4. Actual errors versus publication-significance risk

**Actual errors found: none.** Every manuscript and supplement quantity I recomputed — Table 3 AUCs, Equal_mean composition, ECE, the four site RSCC/RSR values, the two Arg51 witnesses and their occupancies, the 1730/1564/58/108 row counts, the 1278 atom pairs, all eight paired IC50 rows and ratios, source-challenge eligibility and censoring, identity and reference-graph mismatches — matched. Source quotations resolve in the archived primary XML. The historical-DOCX decoy-tolerance correction is right, the ECE erratum is right, and the known-compound identity is right.

**Disclosure asymmetries (accurate but unevenly presented): O3, O10, O11, O12, O13, O19.** These are fixable from evidence already in the repository, except for the EDIAm/OPIA definitions in O13, which require a bounded literature step. None of them involves a false statement.

**Significance risk: O1, O2, O4, O5, O6, O7, O8, O15, O16, O17, O20.** These decide the recommendation. The paper's own diagnostics are the strongest evidence against its publication significance: a label task separable by molecular weight alone, a consensus method that never beats a descriptor baseline and falls to chance when series structure is removed, and a ten-compound same-paper challenge whose perfect scores track nearest-neighbour similarity.

## 5. Venue consequences for objections that cannot be answered

- **O1 (killer), O15, O17:** nothing currently available. For JMGM these are terminal, because insufficient new insight and unregistered retrospective design are scope and quality exclusions rather than presentation defects. A data-descriptor or technical-note venue that rewards reproducible re-extraction of public structural data would not be blocked by them.
- **O16:** answerable only by the specified crystallographic reanalysis. Without it the structural section cannot be presented as a modelling contribution at this venue.
- **O13:** answerable by a bounded literature step plus symmetric reporting. It does not rescue the paper, but leaving it unaddressed converts an honest asymmetry into a reviewer-visible selective-reporting concern, which is the kind of finding that turns a reject into a reject with prejudice.
- **O20:** author-owned and journal-owned. Unresolved, it prevents submission regardless of the scientific verdict.
- **O2, O5, O6, O8:** answerable with existing evidence in the sense that the facts can be stated plainly, but stating them plainly strengthens the case against publication. That is the correct outcome: the dataset cannot support target-recognition claims, and no presentation choice changes that.

## 6. What the authors should not be asked for

I do not request a Path A rebuild, a new library, a replacement decoy cohort, docking, molecular dynamics, deep learning, or any biological experiment as a condition of fixing this paper. The historical 18,412-compound library is unavailable and must not be reconstructed. Missing artifacts and failed access to the Nguyen full text and the JMGM author guide are not evidence that the underlying work or publications do not exist, and I have not treated them as such. The crystal copies are not independent replicates and I have not asked for statistics across them.

## 7. Reviewer limitations

This is an automated review. I did not perform expert three-dimensional map inspection, rerefinement, or any independent crystallographic placement assessment, so my agreement with the manuscript's structural numbers is agreement about coordinates and report fields, not about model correctness. I did not re-derive EDIAm/OPIA definitions from primary methods literature. I did not access the JMGM author guide or the Nguyen full text. I ran a 32-test subset rather than the full suite, and I treat all test results as software evidence only. No new model was fitted at any point in this review. My judgement of editorial fit is an inference from the official published scope, not an editorial decision.
