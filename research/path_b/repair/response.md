# Point-by-point response to three independent AI reviews

These are AI reviews, not human peer review, and no reviewer recommendation here is evidence of
scientific validity. Each review and its objection file is preserved verbatim in this directory with
a SHA-256 manifest (`reviews_manifest.json`). Independent recomputations supplied with the reviews
were rerun from the frozen repository inputs before any repair; the JSON and CSV outputs in this
directory are those reruns. No model was refitted, no library or decoy set was rebuilt, no docking,
molecular dynamics or deep-learning stack was run, and no experiment was performed.

Three categories are used below. **Repaired** means a defect in our reporting, display or code was
fixed with existing evidence. **Confirmed, no repair needed** means the reviewer found no error.
**Unresolved** means the objection stands and cannot be answered with the evidence available.

The central objection is unresolved. All three reviews independently conclude that the package
describes a published compound and its deposited structures more completely without demonstrating a
new scientific result. We agree, state it in the manuscript Discussion and Conclusion, and do not
mark it resolved.

## Review 1 (AI-R1-review.md, AI-R1-objections.json)

| ID | Disposition | Response and evidence | Location |
|---|---|---|---|
| O1 | **Unresolved** | No new energetic, biological or methodological result exists in this package. The Discussion now says so explicitly and states that the evidence does not meet the incremental-insight threshold of the previously proposed specialist modelling venue. No wording change can supply the missing contribution. | `research/manuscript.md` Discussion; `research/path_b/current_gaps.md` G-R2 |
| O2 | **Repaired** | Stated plainly in the main text: positive and decoy molecular-weight ranges do not overlap (303.325–491.339 against 206.285–291.734); raw MW alone gives AUC 1.000, raw HBA and TPSA 0.998. The weaker "compatible with dataset composition effects" wording is gone. | `research/manuscript.md` 3.3; `research/path_b/diagnostic_supplement.md` S1 |
| O3 | **Repaired** | We now report that only 9 of 26 decoys satisfy the three written historical tolerances jointly, and that the only positives ever matched are ACT-19 and ACT-20, instead of reporting only the absence of an executed check. | `research/manuscript.md` 3.3; supplement S1 |
| O4 | **Repaired** | The nearest-property subset is labelled a demonstration that the imbalance survives matching, with Property_LR 0.9945 against Equal_mean 0.9280. It is no longer presentable as a balance control. | supplement S3 |
| O5 | **Repaired** | Cross-fold analogue leakage (16 of 19 positives with Tanimoto >=0.70 to a positive in another fold) and fold imbalance (fold 0 holds 11 of 19) are reported beside Table 3, with within-fold AUC and pair counts. | `research/manuscript.md` 3.3; supplement S2; generated annex |
| O6 | **Repaired with a correction to the objection** | The series-holdout collapse is reported: Equal_mean 0.5628, Nearest_active 0.3289, Property_LR 0.9980. We do not adopt the requested sentence that the consensus never outperforms the descriptor baseline in any retained design: on the separate ten-endpoint published-source cohort Equal_mean reaches 1.000 against Property_LR 0.5625 at the 1 and 10 µM cutoffs. The claim is also false for the unique-molecule repository-label split: Equal_mean 0.9980 versus Property_LR 0.9960 (authenticated-reference scenario 0.9980 versus 0.9939). We therefore name only the seed-42 scaffold, series, nearest-property-subset and component-split comparisons where the claim holds, rather than asserting a universal loss. That cohort is an exposed same-paper analogue set with two threshold-positives and establishes no generalization. | `research/manuscript.md` 3.3; supplement S2, S4 |
| O7 | **Repaired** | The compound-9 result is now a descriptive discordance, with the eligible cohort size (six compounds, three finite ratios) and the nearest training neighbour ACT-20 at Tanimoto 0.660 reported in the same paragraph. | `research/manuscript.md` 3.2 |
| O8 | **Repaired** | The Figure S4 caption no longer says "stress tests without tuning"; it describes retrospective diagnostics of the frozen implementation, notes that earlier outcome exposure is not excluded, and reports the two-positive cohort composition with the Property_LR 0.5625 contrast. | `scripts/build_extension_figures.py`; supplement S4 |
| O9 | **Confirmed; requested action taken** | The identity conclusions stand. The ten `final_hits.csv` rows are now marked explicitly as withdrawn historical assertions through a non-destructive record; the original file is retained byte for byte (SHA-256 `38c8d526…6745100`). | `research/path_b/withdrawn_historical_assertions.json`; supplement S4 |
| O10 | **Repaired** | The unchanged scaffold-split AUC is now reported beside the per-compound instability it hides: 13 compounds change fold, ACT-02's Equal_mean changes by -0.207 and DEC-19 crosses the 0.5 threshold. | supplement S4 |
| O11 | **Repaired** | The stored paired Equal_mean-minus-Property_LR range [-0.2464, 0.0000] around a point difference of -0.0486 is now in the main text with its non-population caveat, and it does not separate the two scores. | `research/manuscript.md` 3.3; supplement S3 |
| O12 | **Repaired** | "Reliability" is replaced by descriptive score-bin wording in the figure title, caption and supplement, and the conformal table column is relabelled observed source-label inclusion rather than empirical coverage against nominal. | `scripts/build_extension_figures.py`; supplement S3 |
| O13 | **Repaired** | EDIA, EDIAm and OPIA definitions were retrieved from the EDIAscorer documentation and the wwPDB dictionary, with retrieval records, hashes and archived pages. All four report fields are now reported for all four W0O sites and both Arg51 residues in Table 1, in the generated abstract and in Limitations: W0O EDIAm 0.410–0.804 with OPIA 16.67–83.33%, Arg51 EDIAm 0.199/OPIA 18.18% (AAA) and 0.413/45.45% (BBB). The EDIA method article returned 403, which is recorded rather than worked around. | `research/manuscript.md` 2.2, Table 1, Limitations; `definitions_sources.json` |
| O14 | **Confirmed, no repair needed** | The chain-specific claims reproduced independently. Retained unchanged. | — |
| O15 | **Unresolved, already disclosed** | The site-selection rule was set after inspecting reports and is not preregistered; `extension_design.md` is not external registration. Retained as stated. | Methods 2.2; supplement S2 |
| O16 | **Unresolved** | No omit or polder map, local rerefinement, expert three-dimensional review or independent placement assessment was performed. These validation computations were not authorized or attempted in this round; tool availability is not asserted, so this is routed as open structural validation rather than silently closed. | `current_gaps.md` G-R3, G-M7/G-S4 |
| O17 | **Confirmed; framing repaired** | The arithmetic stands. The ratio section is now explicitly descriptive and the displayed precision is labelled arithmetic on displayed means. | `research/manuscript.md` 3.2, Table 2 caption |
| O18 | **Confirmed, no repair needed** | Endpoint families remain separate; NUDT14 Leu107 is not mapped onto NUDT5 Arg51, and the manuscript now says so where the source conjecture is discussed. | `research/manuscript.md` 3.1 |
| O19 | **Confirmed** | Tests, hashes and audit counts continue to carry no evidentiary weight for biology. The gap register row for software verification says so. | `current_gaps.md` G-SW1 |
| O20 | **Unresolved, author-owned** | Affiliation, funding, COI, CRediT, approvals, rights, prior-version history, reviewer conflicts, complete AI disclosure and Nikhil Srinivasan's retained contribution and authorship remain open. No declaration was drafted. The journal-specific author guide remains inaccessible. | `author_requests.md`; `current_gaps.md` |

## Review 2, structural (AI-R2-review.md, AI-R2-objections.json)

| ID | Disposition | Response and evidence | Location |
|---|---|---|---|
| AI-S01 | **Unresolved** | Same as Review 1 O1. Added granularity is not incremental scientific insight. Stated in the Discussion. | `research/manuscript.md` Discussion |
| AI-S02 | **Repaired** | The published compound-9 hydrophobic R51 account is now separated from the published hydrogen-bond and TH5427-selectivity conjecture, which rests on ADP-ribose and TH5427 evidence and is not tested here. Leu107 is explicitly not mapped onto Arg51. | `research/manuscript.md` 3.1 |
| AI-S03 | **Repaired** | Functional-group minima are generated from the frozen retained pairs and reported for both chains: AAA backbone N–C20 3.7556 Å, aliphatic side chain CD–C18 4.0011 Å, guanidinium NE–C18 4.7617 Å; BBB backbone N–C20 3.9465 Å, aliphatic side chain CD–C18 3.2513 Å, guanidinium NE–C18 4.6064 Å. Radius sensitivity is explicit: the AAA side-chain witness appears only beyond the 4.0 Å primary radius. The text states that a larger minimum excludes no interaction and that the guanidinium minima are limited by the zero-occupancy AAA CZ. | `scripts/build_path_b_documents.py`; `research/manuscript.md` 3.1 |
| AI-S04 | **Unresolved, disclosure strengthened** | Precomputed maps and whole-ligand report scores cannot establish local-model significance. The absence of expert full-3D assessment, omit/polder maps, rerefinement and placement validation is restated in Limitations, and the unfavourable per-atom fields are now reported rather than set aside. | Limitations; `current_gaps.md` G-R3 |
| AI-S05 | **Repaired** | The Figure 3A caption now records that AAA CZ, at occupancy 0, lies outside all three 0.75 Å slabs and therefore appears in no panel, and that it is excluded from distances rather than absent from the deposited model. The main text repeats the limitation. | `scripts/build_path_b_documents.py`; `research/manuscript.md` 3.1 |
| AI-S06 | **Unresolved** | The catalytic ratio cannot bridge structure and cross-target function. No ratio interval was computed or invented; endpoint SDs were not converted into a selectivity interval. | `research/manuscript.md` 2.3, 3.2 |
| AI-S07 | **Partly repaired** | The main-text legacy-score interpretation is condensed to two paragraphs, while the full named panel, calibration details and design-specific disclosures stay in the supplement and generated annex. This repairs presentation only; it does not convert the score audit into a structural modelling contribution. The full audit trail is preserved. | `research/manuscript.md` 3.3; supplement S2–S4 |
| AI-S08 | **Unresolved, author-owned** | Same as Review 1 O20. | `author_requests.md` |

## Review 3 (AI-R3-review.md, AI-R3-objections.json)

| ID | Disposition | Response and evidence | Location |
|---|---|---|---|
| O1 | **Unresolved** | Same as Review 1 O1. | `research/manuscript.md` Discussion |
| O2 | **Repaired** | "Retrospective counterexample" is replaced by descriptive discordance under the two reported protocols, limited to the invalid reading of a source-label score as selectivity. No ratio uncertainty is derived from summary SDs. | `research/manuscript.md` 3.2 |
| O3 | **Repaired** | Six-decimal ratios are gone; R is displayed to at most three significant digits and the caption states that the precision is arithmetic on displayed source means, not measurement precision. | `scripts/build_path_b_documents.py`; Table 2 caption |
| O4 | **Repaired** | Per-class calibration counts, minimum p-values and set-size frequencies, including forced-inclusion counts, are generated per cohort. The similarity split is labelled a sparse-calibration stress test: class-1 calibration count is 2 in all five outer folds, so the smallest class-1 p-value is 1/3 and class 1 is force-included at both alphas; 43 of 45 and 42 of 45 sets then contain both labels, and 22 of 45 full-valid-set molecules sit in folds where both classes are forced at alpha 0.1. | `scripts/path_b_diagnostics.py`; `scripts/build_extension_figures.py`; supplement S3 |
| O5 | **Repaired** | Pooled and within-fold ROC-AUC are reported side by side with pair counts (494 pooled, 87 within-fold) in main Table 3 and for every method and ablation in the annex, with the reason pooled values are retained. | `research/manuscript.md` Table 3; generated annex |
| O6 | **Repaired** | Every range is labelled a fixed-score group-resampling percentile range with no refitting, and the grouping dependence is disclosed: component resampling gives Equal_mean [0.9286, 1.0000] with 17 single-class draws against [0.9339, 1.0000] with none under scaffold resampling. Superiority language is removed. | supplement S3; `scripts/path_b_diagnostics.py` |
| O7 | **Repaired** | A named panel inventory distinguishes six benchmark/source-challenge methods from 14 controls scores, 17 ablations, three controls designs, five seeds, three cutoffs, reference sensitivity, score/similarity bins, two conformal levels and one probe pair, superlatives are qualified as within that stored panel, and the two single-descriptor controls at 0.9960 are named. | `research/manuscript.md` 3.3; supplement S4 |
| O8 | **Repaired** | The panel is renamed "Series-holdout seed sensitivity"; the seed drives fitting and split generation as well as decoy allocation, so decoy allocation is not isolated. | `scripts/build_research_documents.py` |
| O9 | **Repaired** | One policy is chosen and the annex regenerated: Matthews correlation follows scikit-learn's zero convention when its denominator is zero, stated as a convention rather than an estimated correlation; precision/NPV are null for absent predicted classes, and one-class AUC is undefined rather than zero. | supplement S3; `scripts/build_extension_figures.py` |

## Objections we cannot answer at all

- **No new scientific insight** (R1 O1, R2 AI-S01, R3 O1/U1). Unresolved. It is not answerable by
  arithmetic, test counts, hashes, document regeneration or a favourable re-review.
- **No structural validation of ligand placement** (R1 O16, R2 AI-S04). Unresolved; routed to the
  laboratory or a separately authorized structural round.
- **No raw paired replicate data, covariance or protocol-matched comparability** (R1 O17, R2 AI-S06,
  R3 unanswerable U2). Unresolved; ratios stay descriptive.
- **Author-owned declarations and journal compliance** (R1 O20, R2 AI-S08, R3 unanswerable U3).
  Unresolved and author-owned; nothing was invented.

## Evidence index for the numerical responses

Paths below are relative to this directory. `recomputed_evidence.json` supports R1 O2–O11 and source-cohort/identity context; `recomputed_statistics.json` supports R3 O3–O9 (AUC estimands, calibration counts and panel inventory). `component_bootstrap_sensitivity.json` records the independent component resampling. `independent_geometry.json`, `independent_context.json`, `independent_support.json`, `arg51_atom_distances.csv` and `arg51_group_minima.csv` support AI-S02–S05 and R1 O13–O14. `source_cell_checks.json` and `definitions_sources.json` preserve source-cell and definition witnesses. Live generators recompute descriptive re-expressions without fitting; generated `diagnostic_reexpressions.json` and `arg51_functional_group_minima.json` are included in the completion-last document manifest.
