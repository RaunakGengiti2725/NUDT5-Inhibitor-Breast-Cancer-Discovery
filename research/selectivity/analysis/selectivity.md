# Retrospective paired NUDT5/NUDT14 evidence

Previously inspected single-publication retrospective evidence; not fresh external or prospective validation. No models fitted, new experiments, selectivity classifier, significance tests, ratio confidence intervals or biological qualification.

All 23 source graphs and 46 endpoint cells are retained; 8 source rows have both endpoints.

R = reported mean IC50(NUDT14) / reported mean IC50(NUDT5), dimensionless. Positive log10 R favors lower NUDT5 reported IC50, not validated selectivity.

## Complete source pharmacology

IC50 in µM; ± denotes published SD, not ratio uncertainty.

| Compound | NUDT5 IC50 (µM) | NUDT14 IC50 (µM) | R / bound | Historical inclusion or exclusion |
|---|---|---|---|---|
| 1 | 0.837 ± 0.329 | 0.990 ± 0.110 | 1.1828 | paired; no training identity overlap |
| 2 | >50 (strict) | untested | missing endpoint | missing_target_endpoint |
| 3 | >50 (strict) | untested | missing endpoint | missing_target_endpoint |
| 4 | 13.9 ± 0.62 | untested | missing endpoint | missing_target_endpoint |
| 5 | 21.2 ± 1.02 | untested | missing endpoint | missing_target_endpoint |
| 6 | untested | untested | missing endpoint | missing_target_endpoint |
| 7 | untested | untested | missing endpoint | missing_target_endpoint |
| 8 | untested | untested | missing endpoint | missing_target_endpoint |
| 9 | 0.270 ± 0.027 | 0.162 ± 0.005 | 0.6 | paired; no training identity overlap |
| 10 | 0.487 ± 0.010 | 0.263 ± 0.031 | 0.540041 | training_exact_parent_or_tautomer_overlap |
| 11 | 2.04 ± 0.240 | 0.519 ± 0.084 | 0.254412 | training_exact_parent_or_tautomer_overlap |
| 12 | >50 (strict) | >50 (strict) | not estimable: both >50 | paired; no training identity overlap |
| 13 | >50 (strict) | 3.72 ± 0.190 | <0.0744 (strict) | paired; no training identity overlap |
| 14 | 13.8 ± 0.900 | 1.64 ± 0.140 | 0.118841 | paired; no training identity overlap |
| 15 | >50 (strict) | >50 (strict) | not estimable: both >50 | paired; no training identity overlap |
| 16 | untested | untested | missing endpoint | missing_target_endpoint |
| 17a | untested | untested | missing endpoint | missing_target_endpoint |
| 17b | untested | untested | missing endpoint | missing_target_endpoint |
| 18a | untested | untested | missing endpoint | missing_target_endpoint |
| 18b | untested | untested | missing endpoint | missing_target_endpoint |
| 18c | untested | untested | missing endpoint | missing_target_endpoint |
| 19 | untested | untested | missing endpoint | missing_target_endpoint |
| N-Boc-protected 7 | untested | untested | missing endpoint | missing_target_endpoint |

## historical_original_graphs

n=6 nonoverlap paired compounds; 3 point ratios. Uncalibrated scores (within-cohort shared-minimum rank); no ratio ranking.

| Compound | R / bound | RF | GBT | SVM_RBF | Nearest_active | Property_LR | Equal_mean |
|---|---|---|---|---|---|---|---|
| 9 | 0.6 | 0.69 (1/6) | 0.999975 (1/6) | 0.894815 (1/6) | 0.660377 (1/6) | 0.853829 (3/6) | 0.811292 (1/6) |
| 1 | 1.1828 | 0.54 (2/6) | 0.999975 (1/6) | 0.707449 (2/6) | 0.492537 (2/6) | 0.957093 (1/6) | 0.68499 (2/6) |
| 14 | 0.118841 | 0.39 (3/6) | 0.153046 (3/6) | 0.412072 (3/6) | 0.347222 (3/6) | 0.928141 (2/6) | 0.325585 (3/6) |
| 12 | not estimable: both >50 | 0.28 (4/6) | 0.153046 (3/6) | 0.316333 (4/6) | 0.27027 (4/6) | 0.691137 (5/6) | 0.254912 (4/6) |
| 13 | <0.0744 (strict) | 0.28 (4/6) | 0.153046 (3/6) | 0.311128 (5/6) | 0.27027 (4/6) | 0.691137 (5/6) | 0.253611 (5/6) |
| 15 | not estimable: both >50 | 0.26 (6/6) | 0.153046 (3/6) | 0.283165 (6/6) | 0.25 (6/6) | 0.697845 (4/6) | 0.236553 (6/6) |

The largest displayed Equal_mean is 0.811292 for compound 9, with R 0.6. This coexistence is descriptive, not predictive validation.

## stored_authenticated_reference_sensitivity

n=6 nonoverlap paired compounds; 3 point ratios. Uncalibrated scores (within-cohort shared-minimum rank); no ratio ranking.

| Compound | R / bound | RF | GBT | SVM_RBF | Nearest_active | Property_LR | Equal_mean |
|---|---|---|---|---|---|---|---|
| 9 | 0.6 | 0.74 (1/6) | 0.999975 (1/6) | 0.915668 (1/6) | 0.660377 (1/6) | 0.868497 (3/6) | 0.829005 (1/6) |
| 1 | 1.1828 | 0.57 (2/6) | 0.999975 (1/6) | 0.753011 (2/6) | 0.492537 (2/6) | 0.963285 (1/6) | 0.703881 (2/6) |
| 14 | 0.118841 | 0.4 (3/6) | 0.41952 (3/6) | 0.467284 (3/6) | 0.347222 (3/6) | 0.935503 (2/6) | 0.408507 (3/6) |
| 12 | not estimable: both >50 | 0.31 (4/6) | 0.41952 (3/6) | 0.373775 (4/6) | 0.27027 (4/6) | 0.740752 (5/6) | 0.343391 (4/6) |
| 13 | <0.0744 (strict) | 0.31 (4/6) | 0.41952 (3/6) | 0.362942 (5/6) | 0.27027 (4/6) | 0.740752 (5/6) | 0.340683 (5/6) |
| 15 | not estimable: both >50 | 0.27 (6/6) | 0.41952 (3/6) | 0.334263 (6/6) | 0.25 (6/6) | 0.749928 (4/6) | 0.318446 (6/6) |

The largest displayed Equal_mean is 0.829005 for compound 9, with R 0.6. This coexistence is descriptive, not predictive validation.

## Evidence limits

- Previously inspected single-publication retrospective evidence; not fresh external or prospective validation. No models fitted, new experiments, selectivity classifier, significance tests, ratio confidence intervals or biological qualification.
- R is reported mean catalytic IC50(NUDT14)/reported mean catalytic IC50(NUDT5), dimensionless; R>1 favors lower reported NUDT5 IC50 under these assay conditions, not an affinity constant, clinical selectivity or a validated biological class.
- Source mean ± SD remains visible at each target. Raw paired replicates and covariance are unavailable; ratio uncertainty is not estimated. Table 1 reports two biological replicates; Methods triplicate sets and Figure 1 technical triplicates are not extra independent biological replicates.
- NUDT5 and NUDT14 reaction times differ (20 versus 60 minutes). The shared TH5427 normalization wording has no clear NUDT14-specific exception. Comparability remains unresolved; this is not proof of assay failure.
- Untested is not inactive. >50 uM is a strict bound, never an exact mean of 50. Double censoring permits any positive R and identifies no finite bound or direction.
- Exact/neutral-parent/parent-tautomer training matches exclude only model diagnostics. All source rows, known controls and source roles remain visible. Related chemistry from one already-inspected publication is not an independent validation cohort.
- All six frozen methods and both recorded scenarios are retained separately. Scores are uncalibrated label diagnostics, not IC50 predictions or selectivity probabilities. There is no outcome-chosen high-score threshold and no pooling of scenarios.
- Source disagreements remain unresolved: Figure 1 versus CSV SD precision for 4/5, two trailing spaces, control/replication ambiguity, all-compounds-tested context and the reciprocal ratio orientation in the earlier structural report.

## Primary evidence inspected in the inherited curation

Balikci E et al. Unexpected Noncovalent Off-Target Activity of Clinical BTK Inhibitors Leads to Discovery of a Dual NUDT5/14 Antagonist. J Med Chem 67,7245-7259 (2024). DOI 10.1021/acs.jmedchem.4c00072; PMC11089510.

- balikci_xml: https://www.ebi.ac.uk/europepmc/webservices/rest/PMC11089510/fullTextXML; sections: tbl1/t1fn1, tbl1/t1fn2, sec4.2, sec4.3, fig1/caption, sec2/p[1], sec2/p[4], sec2/p[5], sec2/p[7]; retrieved 2026-10-05T03:49:46.449676+00:00; decompressed SHA-256 `0710da1ebedd4accde6e6cddf8cfe9a2d350c062890fbcfd03bf94bd6f316df0`.
- balikci_csv: https://www.ebi.ac.uk/europepmc/webservices/rest/PMC11089510/supplementaryFiles; sections: header and all 23 data rows (lines 2-24); retrieved 2026-10-05T03:50:32.517062+00:00; decompressed SHA-256 `17ef0e3770d49db8b7575a18fd1cdb4623fd83db0c928ba7ade75d951f7d7533`.
- table1_scaffold: https://www.ebi.ac.uk/europepmc/webservices/rest/PMC11089510/supplementaryFiles; sections: tbl1/fx1; scaffold only; retrieved 2026-10-05T03:50:32.517062+00:00; decompressed SHA-256 `dae7572f3d20892ce6dc86e15fb0a3f2722a9881510fcf92ff4ef64d42f218b5`.
- table1_values: https://www.ebi.ac.uk/europepmc/webservices/rest/PMC11089510/supplementaryFiles; sections: tbl1/fx2; all eight compounds and both endpoint columns; retrieved 2026-10-05T03:50:32.517062+00:00; decompressed SHA-256 `fdfc9e95b92fe83bf84fc870c9d7188b3f5d45ed9162dace5640c281ef8b437a`.
