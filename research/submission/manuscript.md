# Occupancy-aware reanalysis of deposited NUDT5 and NUDT14 inhibitor complexes

AUTHOR-REVIEW DRAFT. Proposed corresponding author: Raunak Gengiti, gengitir@gmail.com. Final byline, address and declarations require author confirmation. This document is not authorized for submission.

## Abstract

### Objective

This descriptive reanalysis asks what atom-specific inspection adds to published residue-proximity descriptions of compound 9 bound to NUDT5 and NUDT14. It also examines the interpretive limits of the source's catalytic measurements and a historical screening benchmark.

### Results

All four deposited ligand sites in 8RIY and 8OTV were retained. The nearest positive-occupancy Arg51 witnesses in the two NUDT5 sites were backbone N at 3.756 Å and side-chain CD at 3.251 Å. The first site contains a zero-occupancy CZ atom and report-listed geometry problems; the second witness has fractional occupancy. Whole-ligand real-space correlation coefficients of 0.928–0.952 coexist with less favourable atom-level density summaries. Compound 9's reported NUDT14/NUDT5 catalytic half-maximal inhibitory concentration ratio is 0.600, under different reaction durations and without recoverable ratio uncertainty. Simple descriptor controls expose source-label separation in the screening benchmark. These observations support site-specific description and caution against interpreting proximity or screening scores as binding energetics or selectivity. No new inhibitor, biological experiment or prospective validation is reported. The structures, maps and validation reports share experimental and model dependencies.

## Keywords

NUDT5; NUDT14; crystallographic occupancy; structural reanalysis; benchmark bias; reproducibility

## Introduction

Published compound-9 complexes provide a direct starting point for checking how residue-level descriptions relate to individual atoms and crystallographic copies [1,2]. Balikci and colleagues discussed the Arg51 environment in NUDT5 and compared inhibitor activity against NUDT14 [2]. Such observations motivate questions about molecular recognition, but coordinates alone do not measure an interaction's energetic contribution.

This study extends inspection of those deposited models rather than reporting independent structural experiments. It preserves both ligand sites in each asymmetric unit, distinguishes backbone and side-chain witnesses, and places proximity beside occupancy and model-support fields. Published catalytic endpoints from the same source supply protocol-bound context. A separate historical screening dataset supplies an additional control: can simple molecular descriptors separate its source labels? The objective is to identify which interpretations these records permit. The work does not evaluate breast-cancer treatment efficacy.

## Main text

### Sources and reproducible methods

The analysis uses the deposited 8RIY and 8OTV coordinate files, official validation reports, deposited structure factors and PDBe precomputed maps [3,4]. Inputs and recorded outputs have SHA-256 manifests. Python 3.12, Gemmi 0.7.3 and the pinned environment implement the analysis. Additional file 1 specifies every retained parameter and diagnostic design. Additional files 2 and 3 provide code, source snapshots, machine-readable tables and reproduction commands. No new model fitting occurs during document generation.

Both W0O sites and both receptor chains were retained for each entry. Euclidean heavy-atom distances include finite coordinates with positive occupancy. Fractional occupancies are retained without weighting. Alternate conformers are enumerated separately; zero-occupancy, absent and nonfinite coordinates do not contribute observed distances. The fixed radii are 3.5, 4.0, 4.5 and 5.0 Å, with 4.0 Å primary. Thresholds were not adjusted to obtain a contact. A partial-residue minimum is an upper bound on the unknown complete-residue minimum. Missing rows remain explicit. Crystal copies are nonindependent, and no cross-target residue homology is assumed.

Exact chain/residue identifiers link all four ligands to validation-report rows. Real-space correlation coefficient (RSCC), real-space R factor (RSR), electron-density fit summaries (EDIAm) and the percentage of atoms above the EDIA threshold (OPIA) are retained together. The dictionary identifies EDIAm below 0.8 and OPIA below 50% as reasons for closer inspection [5,6]. Fixed interpolated map slices provide limited model-dependent inspection. They are not independent ligand-placement validation.

For the single-source catalytic ledger [2], R is the reported NUDT14 IC50 divided by reported NUDT5 IC50. IC50 denotes half-maximal inhibitory concentration. Ratios use reported means; right-censored endpoints remain bounds. Both endpoints above 50 µM give no finite ratio bound. The source describes two independent biological replicates, but replication wording is unresolved. Raw replicate vectors and covariance are unavailable. NUDT5 and NUDT14 reaction durations differ, 20 versus 60 minutes, and share a TH5427 normalization anchor. No uncertainty interval, paired-replicate estimate or affinity ratio is inferred. This is single-source arithmetic, not a pooled analysis or meta-analysis.

The historical diagnostic dataset contains 46 rows: 20 positive source labels and 26 unassayed presumed-negative decoys. Invalid ACT-18 remains excluded, leaving 45 valid unique graphs and 19 positive labels. Fixed seed-42, five-fold exact-scaffold out-of-fold scores are reused. Morgan fingerprints use radius 2 and 2,048 bits. Equal_mean averages raw bounded scores from random forest, gradient boosting, RBF-SVM and nearest-active Tanimoto similarity with equal weights. Property_LR is separate. This differs from historical min–max normalization. It is neither learned transferability weighting nor calibrated biochemical risk. The full implementation, model settings, fold membership and score vectors accompany Additional file 2.

Table 3 reports pooled receiver-operating-characteristic area under the curve (ROC-AUC) and a pair-weighted within-fold ROC-AUC from the same predictions. The latter excludes comparisons between differently trained models. Constant, prevalence, descriptor, nearest-neighbour and single-descriptor controls remain visible. These are source-label diagnostics, not validation against measured inactive compounds.

### Site-specific observations

8RIY has resolution approximately 2.288 Å and no modeled Mg; 8OTV has resolution approximately 1.82 Å and one Mg. All four W0O sites map exactly to report rows (Table 1). In NUDT5 author chain AAA, the nearest retained Arg51 witness is backbone N at 3.756 Å; CZ has zero occupancy and is excluded. Official reports list local bond, angle and clash problems. In BBB, the witness is side-chain CD at 3.251 Å with occupancy 0.78, and no Arg51 outlier is listed in the inspected report. Neither witness is a guanidinium atom. Absence of a listed outlier does not establish that BBB is correct.

Whole-ligand RSCC spans 0.928–0.952 and RSR spans 0.076–0.097. EDIAm spans 0.410–0.804 and OPIA spans 16.67–83.33%. Arg51 EDIAm is 0.199 in AAA and 0.413 in BBB. These less favourable atom-level summaries limit interpretation of the nearest-atom difference. Leu47 alternate conformers in 8OTV remain separate. NUDT14 residue 51 is serine; no equivalence to NUDT5 Arg51 is asserted.

Table 1. Deposited ligand and Arg51 model-support summaries

Site identifiers are label chain / author chain / author residue number. The first four rows are W0O; the last two are Arg51. Report atoms are those included in source density analysis. RSCC, RSR, EDIAm and OPIA describe the same deposited models and do not independently validate ligand placement or interaction energy.

<!-- path-b:sites:start -->
| PDB | Target | Site label / auth / residue | RSCC | RSR | EDIAm | OPIA (%) | Report atoms |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 8OTV | NUDT14 | C / A / 301 | 0.952 | 0.076 | 0.804 | 83.330 | 30 |
| 8OTV | NUDT14 | F / B / 302 | 0.928 | 0.097 | 0.728 | 60.000 | 30 |
| 8RIY | NUDT5 | C / AAA / 301 | 0.931 | 0.094 | 0.412 | 16.670 | 30 |
| 8RIY | NUDT5 | D / BBB / 301 | 0.940 | 0.091 | 0.410 | 26.670 | 30 |
| 8RIY | NUDT5 | Arg51 A / AAA / 51 | 0.894 | 0.103 | 0.199 | 18.180 | 11 |
| 8RIY | NUDT5 | Arg51 B / BBB / 51 | 0.901 | 0.143 | 0.413 | 45.450 | 11 |
<!-- path-b:sites:end -->

### Catalytic endpoints and screening controls

Table 2 retains every paired-endpoint compound, including training-overlap compounds 10 and 11. Compound 9 has R = 0.600 and the highest stored Equal_mean among eligible nonoverlapping paired compounds in both frozen score scenarios. Only three of the six eligible compounds have finite ratios. Its nearest retained training neighbour is ACT-20, source compound 10, at Tanimoto similarity 0.660. This is descriptive discordance between a source-label score and source-reported catalytic ordering. It does not test general selectivity prediction.

Table 2. Reported catalytic endpoints and source-bound ratios

IC50 values are reported means ± SD or bounds in µM [2]. R = NUDT14/NUDT5 is dimensionless. Endpoint SDs are not ratio uncertainty. Unequal reaction durations, shared normalization and unresolved replicate pairing apply to every row. An upper bound is strict. Displayed ratio precision is arithmetic precision, not measurement precision.

<!-- path-b:paired:start -->
| Compound | NUDT5 IC50, µM (mean ± SD or bound) | NUDT14 IC50, µM (mean ± SD or bound) | R or strict bound |
| --- | --- | --- | --- |
| 1 | 0.837 ± 0.329 | 0.990 ± 0.110 | 1.18 |
| 9 | 0.270 ± 0.027 | 0.162 ± 0.005 | 0.6 |
| 10 | 0.487 ± 0.010 | 0.263 ± 0.031 | 0.54 |
| 11 | 2.04 ± 0.240 | 0.519 ± 0.084 | 0.254 |
| 12 | >50 | >50 | No finite bound |
| 13 | >50 | 3.72 ± 0.190 | <0.0744 |
| 14 | 13.8 ± 0.900 | 1.64 ± 0.140 | 0.119 |
| 15 | >50 | >50 | No finite bound |
<!-- path-b:paired:end -->

Table 3 shows that high original-label discrimination is available to simple descriptors. TPSA-only and HBA-only logistic regression both reach pooled ROC-AUC 0.9960, compared with 0.9312 for Equal_mean. No positive/decoy pair meets the fixed seven-descriptor, 0.5 full-cohort-standard-deviation caliper. Nine of 26 decoys meet all three historical written tolerances, and only ACT-19 and ACT-20 have any matched decoys in the available records. These findings expose benchmark construction; they establish no target-specific recognition. Prevalence-score pooled ROC-AUC differs from its within-fold value because fold prevalences order observations across fitted models.

Table 3. Same-split source-label controls

All rows reuse full-valid-set, seed-42, exact-scaffold five-fold predictions. The pooled estimand compares 494 positive/negative pairs; 87 occur within folds. Within-fold ROC-AUC weights concordance by valid pairs. Descriptor suffix only_lr denotes single-descriptor logistic regression. No row establishes measured inactivity, target-specific recognition or model superiority.

<!-- path-b:controls:start -->
| Control / method | Pooled ROC-AUC | Within-fold ROC-AUC | Pairs pooled / within |
| --- | --- | --- | --- |
| Constant_0_5 | 0.5000 | 0.5000 | 494 / 87 |
| Train_prevalence | 0.2460 | 0.5000 | 494 / 87 |
| Property_LR | 0.9798 | 1.0000 | 494 / 87 |
| Nearest_active | 0.9130 | 0.9655 | 494 / 87 |
| Tanimoto_kNN | 0.9332 | 0.9425 | 494 / 87 |
| RF | 0.9413 | 0.9713 | 494 / 87 |
| GBT | 0.8431 | 0.9368 | 494 / 87 |
| SVM_RBF | 0.9595 | 0.9770 | 494 / 87 |
| Equal_mean | 0.9312 | 0.9655 | 494 / 87 |
| clogp_only_lr | 0.7895 | 0.8161 | 494 / 87 |
| fsp3_only_lr | 0.7510 | 0.8161 | 494 / 87 |
| hba_only_lr | 0.9960 | 1.0000 | 494 / 87 |
| hbd_only_lr | 0.7176 | 0.7874 | 494 / 87 |
| mw_only_lr | 0.9879 | 1.0000 | 494 / 87 |
| nrb_only_lr | 0.8704 | 0.9655 | 494 / 87 |
| tpsa_only_lr | 0.9960 | 1.0000 | 494 / 87 |
<!-- path-b:controls:end -->

### Interpretation

We observed different nearest Arg51 atom identities across the retained NUDT5 sites, subject to local model limitations. This qualifies residue-level shorthand without refuting the published chain-B account or demonstrating an energetic role. We observed source-label separation by simple descriptors and a protocol-bound catalytic ordering that does not follow an activity-score interpretation. These evidence types remain separate.

A future discrimination experiment would compare authenticated TH5427 and compound 9 with qualified NUDT5 wild-type, R51A and R51K, direct binding, orthogonal binding confirmation and functional readouts. NUDT14 would require a separate cross-target comparison. Variant folding, oligomerization, stability and active fraction, and compound solubility, aggregation and interference must be checked. Unqualified protein makes an outcome uninterpretable, not disconfirming. R51K is not assumed conservative. These experiments are unexecuted; no sample size, effect, schedule or venue outcome is inferred.

## Limitations

The models, reports and maps share experimental/model dependencies. Fixed slices lack omit/polder maps, rerefinement, expert three-dimensional review, coordinate-error propagation and independent ligand-placement validation. B factors are metadata, not positional uncertainty. Fractional occupancy, alternate conformers, partial residues, nulls and nonindependent copies constrain every distance. The model-support fields are not uniformly favourable. Proximity supplies no affinity, interaction energy, hydrogen bond, causality, homology or selectivity estimate and does not exclude other contacts.

Source catalytic summaries cannot resolve replicate pairing, covariance, kinetic comparability or ratio uncertainty. Source-method disagreements remain in the ledger. Graph matching and database records do not authenticate a physical sample or provide independent experimental replication. The bounded source check is not a systematic review or exhaustive identity or patent search.

Original source-label provenance is incomplete; decoys are unmatched and unassayed. ACT-18 is invalid, and historical TH5427 and TH1713 reference graphs are mismatched. NC5-02 is ACT-19 and published compound 11, so its discovery claim is withdrawn. The reported 18,412-compound library is unavailable and will not be reconstructed; every dependent screening result is removed. Missing artifacts do not establish that earlier work never occurred. Additional file 1 preserves calibration, conformal, resampling, chemical-series, multiplicity and applicability limitations. No new library, replacement decoys, docking, molecular dynamics, deep-learning study, biological experiment or prospective validation was performed. Software tests establish implementation behavior, not biological validity or journal acceptance.

## List of abbreviations

EDIA, electron density score for individual atoms; EDIAm, aggregate EDIA score; GBT, gradient-boosted trees; HBA, hydrogen-bond acceptors; HBD, hydrogen-bond donors; IC50, half-maximal inhibitory concentration; LR, logistic regression; MW, molecular weight; NUDT, Nudix hydrolase; OPIA, percentage of atoms with EDIA above 0.8; PDB, Protein Data Bank; RBF, radial basis function; RF, random forest; ROC-AUC, receiver-operating-characteristic area under the curve; RSCC, real-space correlation coefficient; RSR, real-space R factor; SD, standard deviation; SVM, support-vector machine; TPSA, topological polar surface area.

## References

[1] Page BDG, et al. Targeted NUDT5 inhibitors block hormone signaling in breast cancer cells. Nat Commun. 2018;9:250. https://doi.org/10.1038/s41467-017-02293-7. Funding correction: 2019;10:5050. https://doi.org/10.1038/s41467-019-12806-1.

[2] Balıkçı E, Marques AMC, Bauer LG, Seupel R, Bennett J, Raux B, et al. Unexpected Noncovalent Off-Target Activity of Clinical BTK Inhibitors Leads to Discovery of a Dual NUDT5/14 Antagonist. J Med Chem. 2024;67(9):7245. https://doi.org/10.1021/acs.jmedchem.4c00072.

[3] Protein Data Bank. NUDT5 compound-9 complex, entry 8RIY. https://www.rcsb.org/structure/8RIY. Accessed 6 October 2026. Source retrieval records, validation reports and hashes accompany Additional file 2.

[4] Protein Data Bank. NUDT14 compound-9 complex, entry 8OTV. https://www.rcsb.org/structure/8OTV. Accessed 6 October 2026. Source retrieval records, validation reports and hashes accompany Additional file 2.

[5] wwPDB. Validation dictionary: EDIAm. https://mmcif.wwpdb.org/dictionaries/mmcif_pdbx_vrpt.dic/Items/_pdbx_vrpt_model_instance_density.EDIAm.html. Accessed 6 October 2026.

[6] wwPDB. Validation dictionary: OPIA. https://mmcif.wwpdb.org/dictionaries/mmcif_pdbx_vrpt.dic/Items/_pdbx_vrpt_model_instance_density.OPIA.html. Accessed 6 October 2026.
