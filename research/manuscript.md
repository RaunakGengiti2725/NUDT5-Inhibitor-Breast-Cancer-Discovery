# Deposited NUDT5/NUDT14 inhibitor geometry and the limits of paired catalytic measurements

Author-review revision. Raunak Gengiti; correspondence: gengitir@gmail.com. The historical byline also names Nikhil Srinivasan. The final author list, retained contributions, affiliations and declarations require author confirmation. This is not a submission-ready byline or a declaration of sole authorship.

## Abstract

<!-- path-b:abstract:start -->
Deposited inhibitor complexes describe local geometry but cannot by themselves test energetic dependence. We reanalyse all 4 W0O sites in published Nudix hydrolase 5 (NUDT5) and NUDT14 structures 8RIY/8OTV, retaining both receptor chains, occupancy, alternate conformers and missingness. Official validation reports give ligand real-space correlation coefficient (RSCC) 0.928–0.952 and real-space R (RSR) 0.076–0.097. Fixed sampling and slices of precomputed PDBe maps provide limited model-support inspection, not independent density validation. The nearest retained 8RIY Arg51 witnesses differ: backbone N at 3.756 Å in auth chain AAA and side-chain CD at 3.251 Å in BBB. AAA has a zero-occupancy CZ and report-listed local geometry problems. BBB CD has fractional occupancy. These observations qualify the published chain-specific account without testing or refuting its energetic interpretation. Published compound-9 catalytic means give NUDT14/NUDT5 half-maximal inhibitory concentration (IC50) ratio 0.600; unequal reaction times, shared control normalization and unresolved replication wording prevent an affinity or general selectivity interpretation. Crystal copies are nonindependent. Source-label controls are retained only as diagnostics. The contribution is descriptive and methodological; no new inhibitor, biological experiment or prospective validation is reported.
<!-- path-b:abstract:end -->

## 1. Introduction

A residue near a bound inhibitor provides a structural description before it provides an explanation of binding. This distinction matters for NUDT5 and NUDT14, whose inhibition, binding and cellular roles are studied through different experimental readouts. Page et al. reported TH5427 and cellular effects in a hormone-signalling context [1]. Balikci et al. reported compound 9, paired catalytic measurements and its deposited complexes with NUDT5 and NUDT14 [2]. Those experiments already establish the published chemical and structural context. Recalculating deposited distances does not establish another inhibitor discovery.

The question here is narrower: what does an all-site account of the deposited compound-9 geometry add to the published Arg51 interpretation, and how far can that account be connected to the reported paired catalytic endpoints? The published discussion identifies the chain-B Arg51 environment in NUDT5 [2]. A chain-specific description is compatible with differences between deposited copies. An energetic claim would require additional evidence. We retain every deposited W0O site, both receptor chains and explicit atom witnesses, rather than reducing each target to a representative pose or one residue name.

The contribution is descriptive and methodological. It joins source-bound coordinate extraction, local model-support inspection and endpoint-aware reporting in a reproducible package. It does not refute the source experiments or test energetic dependence. Earlier screening claims are demoted to diagnostics of a small source-label dataset. The historical library is unavailable and will not be reconstructed. NC5-02 is the already published compound 11. No new biological experiment or prospective validation was performed.

## 2. Methods

### 2.1 Deposited-coordinate scope

We use the archived 8RIY and 8OTV coordinate records for compound 9, identified as W0O, with the fixed contract and recorded biological assembly in the repository [2,3]. Label and author chain/residue identifiers are retained separately. Positive-occupancy heavy atoms enter observed distances. Zero-occupancy, absent and nonfinite coordinates do not. Positive fractional occupancies are retained without occupancy weighting. Local alternate conformers are enumerated without pretending that local alternatives are globally phased states. Ligand and protein heavy-atom completeness are recorded separately.

For every ligand site and receptor residue-conformer combination, the extraction retains the observed minimum, all tied atom witnesses and status. The inclusive radii are 3.5, 4.0, 4.5 and 5.0 Å, with 4.0 Å primary; these were fixed for the audited extraction rather than tuned to a conclusion. Distances for a partial residue are upper bounds on its unknown complete-residue minimum. Null/refused rows do not indicate absence of a contact. Nonprotein atoms and excluded atoms remain inventoried. B factors are retained metadata; neither complete-residue distances nor coordinate uncertainty are estimated. The target-specific residue axes are not homology-aligned. Crystal copies are nonindependent descriptive records and are not tested statistically.

### 2.2 Bounded local model-support assessment

The added assessment maps official wwPDB validation XML to the exact ligand sites, inventories deposited structure-factor columns and uses the precomputed PDBe electron-density and difference maps [3]. Real-space correlation coefficient (RSCC) and real-space R (RSR) are extracted report summaries, not newly calculated density validation or per-atom support probabilities. All W0O sites and the local Arg51 environments are retained under a fixed retrospective selection rule. That rule was set after inspecting reports and was not preregistered.

Atom sampling uses fixed interpolation and standardized map values; fixed orthogonal slices provide limited direct inspection. Interpolation does not increase resolution. Standardized values have no pass/fail threshold and do not estimate occupancy, coordinate error, affinity or energy. The maps are model-dependent and were not recalculated as omit or polder maps. Map cells and space groups were checked against the coordinates. Report/map software dates need not describe the same map computation. EDIAm and OPIA fields are archived without interpretation because source definitions were not recorded. The CCP4 headers do not record map-generation version. No rerefinement, expert three-dimensional map review or independent ligand-placement validation was performed.

### 2.3 Published paired measurements

The paired ledger preserves Balikci's author CSV, Table 1 and protocol witnesses, including numeric means, reported standard deviations (SDs), strict right-censoring and untested endpoints [2]. R denotes the dimensionless ratio of the reported mean catalytic IC50 for NUDT14 to that for NUDT5. IC50 is half-maximal inhibitory concentration under the reported assay conditions. It is distinct from an equilibrium dissociation constant (KD), cellular engagement or viability. There is no conversion to KD or Ki. Untested and censored values are never replaced by arbitrary numeric values.

The reported adenosine diphosphate ribose (ADPr) hydrolysis protocols differ in duration: 20 minutes for NUDT5 and 60 minutes for NUDT14. Both use a vehicle anchor and the stated 500 nM TH5427 zero normalization. The inspected passages do not establish target-specific control adequacy, Km or a common initial-rate regime. The Methods' triplicate-sets wording, representative technical triplicates and Table 1's two independent biological replicates do not define a complete nesting or pairing structure. No replicate covariance, ratio confidence interval or inferential ratio comparison is available. Ratio bounds reflect censoring only.

### 2.4 Source-label diagnostics and reproducibility

The main-text diagnostic controls use one identical recorded design: all valid unique original-input graphs, exact-scaffold group-disjoint folds and seed 42. Seven single-descriptor logistic models, descriptor-only logistic regression, nearest-active and five-neighbour Tanimoto scores, constant/fold-prevalence controls, random forest (RF), gradient-boosted trees (GBT), radial-basis-function support-vector machine (RBF-SVM) and Equal_mean are shown together. Equal_mean is a raw equal-weight mean of RF, GBT, RBF-SVM and nearest-active scores, each weighted 0.25. It is a fixed equal-weight four-score consensus over Morgan-fingerprint classifiers with group-disjoint resampling diagnostics, without trained fused-score calibration or learned transfer weights. It differs from the historical normalized consensus. Property_LR is separate.

The seven descriptors are molecular weight (MW), calculated octanol/water partition coefficient (cLogP), topological polar surface area (TPSA), hydrogen-bond donor count (HBD), hydrogen-bond acceptor count (HBA), rotatable bonds (NRB) and fraction of sp3-hybridized carbon atoms (Fsp3). Single-property values are out-of-fold logistic scores, not raw-descriptor ranking chosen after inspecting outcomes. The area under the receiver operating characteristic curve (ROC-AUC) is recalculated from recorded scores. The supplement supplies identities, fold assignments, score pointers, resampling, expected calibration error (ECE), conformal and cutoff diagnostics. No models were newly fitted for this Path B rewrite.

The document profile checks frozen input hashes, reconstructs the trusted structural scientific fields and compares manuscript quantitative blocks with generated values before publishing. It stages a complete package and publishes its manifest last into a fresh directory. Historical manifests keep their original paths, timestamps and Git state; new runs have separate provenance. Software tests and hashes verify computational behavior, not the validity of source labels or biological hypotheses.

## 3. Results

### 3.1 All-site geometry and chain-specific Arg51 witnesses

Figures 1 and 2 show NUDT5 and NUDT14 separately. Each retains both W0O sites and both receptor chains. Residues appear when any retained pair is within the displayed range at either site; complete numerical tables also retain distant, partial and null rows. Neither target axis is a homology correspondence.

<!-- path-b:geometry:start -->
The fixed-contract extraction retains 1,730 residue-conformer rows: 1,564 observed, 58 partial and 108 refused/null. It retains 1,278 atom pairs within 5 Å. These are coordinate records, not independent observations. Refused rows are not no-contact results. Partial minima use retained atoms only.

8RIY auth chain AAA (label A): Arg51 N to W0O C20 is 3.756 Å; witness occupancy 1; residue status partial_observed. 8RIY auth chain BBB (label B): Arg51 CD to W0O C18 is 3.251 Å; witness occupancy 0.78; residue status observed.
<!-- path-b:geometry:end -->

The nearest witnesses differ in atom identity as well as distance. Neither is a guanidinium atom. In AAA, CZ has zero occupancy and is excluded from distances, while the residue remains partial. Official reports list severe local bond/angle outliers and clashes there. BBB has no report-listed Arg51 outlier in the inspected record; its nearest witness is a fractional-occupancy side-chain carbon. This does not establish that every aspect of BBB is correct. Figures 3A–B expose the fixed map slices for these local contexts; the supplement provides all W0O-site slices and sampling tables.

These findings qualify the source's chain-B account rather than contradicting it. A residue-level distance is not evidence that the same atom mediates binding in every copy. It also does not determine whether Arg51 contributes energy to either complex. In 8OTV residue 51 is SER; Leu107 is a NUDT14 residue number and is not mapped to NUDT5 Arg51. No hydrogen bond, causal atom, homology or selectivity conclusion follows from these minima.

Table 1. Exact W0O report mappings. Sites are identified by label chain, author chain and author residue number. Report atoms count ligand atoms included in the source EDS analysis. All sites contain the full retained ligand heavy-atom set. RSCC/RSR summarize agreement in the official report; they do not validate a ligand independently or quantify interaction energy.

<!-- path-b:sites:start -->
| PDB | Target | W0O label / auth / residue | RSCC | RSR | Report atoms |
| --- | --- | --- | --- | --- | --- |
| 8OTV | NUDT14 | C / A / 301 | 0.952 | 0.076 | 30 |
| 8OTV | NUDT14 | F / B / 302 | 0.928 | 0.097 | 30 |
| 8RIY | NUDT5 | C / AAA / 301 | 0.931 | 0.094 | 30 |
| 8RIY | NUDT5 | D / BBB / 301 | 0.940 | 0.091 | 30 |
<!-- path-b:sites:end -->

### 3.2 Paired catalytic measurements remain protocol-bound

Table 2 retains every paired-endpoint compound, including known training-overlap compounds 10 and 11. The latter are excluded only from the separate frozen-score challenge, not from the pharmacology ledger. Untested compounds remain in the full ledger. Numeric R values are ratios of reported means, not means of replicate ratios. An upper bound is strict; when both target values are right-censored, their ratio has no finite bound.

Table 2. Published catalytic IC50 means ± SD or source bounds, in µM [2]. R = NUDT14/NUDT5, dimensionless. SDs are source endpoint SDs from two reported independent biological replicates; no ratio uncertainty or inferred replicate pairing is supplied. Reaction durations, normalization and replication limitations in Methods apply to every row.

<!-- path-b:paired:start -->
| Compound | NUDT5 IC50, µM (mean ± SD or bound) | NUDT14 IC50, µM (mean ± SD or bound) | R or strict bound |
| --- | --- | --- | --- |
| 1 | 0.837 ± 0.329 | 0.990 ± 0.110 | 1.182796 |
| 9 | 0.270 ± 0.027 | 0.162 ± 0.005 | 0.600000 |
| 10 | 0.487 ± 0.010 | 0.263 ± 0.031 | 0.540041 |
| 11 | 2.04 ± 0.240 | 0.519 ± 0.084 | 0.254412 |
| 12 | >50 | >50 | No finite bound |
| 13 | >50 | 3.72 ± 0.190 | <0.074400 |
| 14 | 13.8 ± 0.900 | 1.64 ± 0.140 | 0.118841 |
| 15 | >50 | >50 | No finite bound |
<!-- path-b:paired:end -->

Compound 9 has the highest stored Equal_mean among the eligible nonoverlapping paired compounds in both stored scenarios, yet its reported catalytic ratio is below one. This is a retrospective counterexample to reading a high activity-label score as NUDT5-selective inhibition under these assays. It is not a calibrated selectivity test, an affinity comparison or a general assessment of either target. Both frozen score scenarios, all target SDs, exclusions and unbounded ratios remain in the supplement. NC5-02/ACT-19 is compound 11, so this record cannot support a new-discovery claim.

### 3.3 Source-label controls do not establish target recognition

Table 3. Identical full-valid-set exact-scaffold five-fold split, seed 42; 45 valid unique compounds, 19 positive source labels. Scores and indices are from research/results/controls.json, evaluations/full_valid_set. Every ROC-AUC is recomputed from those out-of-fold scores, with no new model fitting. Models using a single descriptor are named with the suffix only_lr. This compact table does not combine the source challenge, authenticated-reference sensitivity or other split designs.

<!-- path-b:controls:start -->
| Control / method | Pooled ROC-AUC |
| --- | --- |
| Constant_0_5 | 0.5000 |
| Train_prevalence | 0.2460 |
| Property_LR | 0.9798 |
| Nearest_active | 0.9130 |
| Tanimoto_kNN | 0.9332 |
| RF | 0.9413 |
| GBT | 0.8431 |
| SVM_RBF | 0.9595 |
| Equal_mean | 0.9312 |
| clogp_only_lr | 0.7895 |
| fsp3_only_lr | 0.7510 |
| hba_only_lr | 0.9960 |
| hbd_only_lr | 0.7176 |
| mw_only_lr | 0.9879 |
| nrb_only_lr | 0.8704 |
| tpsa_only_lr | 0.9960 |
<!-- path-b:controls:end -->

The descriptor controls discriminate these labels at least as well numerically as the current consensus on this partition. This is compatible with dataset composition effects but does not identify a unique cause or reveal exactly what a classifier learns. Positive labels are largely unauthenticated; decoys are unmatched, unassayed presumed negatives. High label AUC cannot establish target recognition. Fold-prevalence scores are constant within each fold but differ across folds, so their pooled AUC can fall below chance. Pooled scores from different fitted models are not a single validated biological ranking. Conformal coverage is empirical under unassured exchangeability, and descriptive ECE is not trained calibration.

## 4. Discussion

An all-site analysis changes the granularity of the structural statement. Arg51 proximity in the deposited NUDT5 model is chain- and atom-specific, with a partial local model in AAA and a fractional-occupancy side-chain witness in BBB. The report and map records make those distinctions inspectable. They do not test the energetic rationale proposed for compound 9 [2]. A favorable report statistic cannot turn a residue name into a demonstrated chemical interaction, and absence of a listed outlier cannot establish energetic correctness.

The paired measurements answer a different, protocol-specific question. They describe catalytic inhibition under each source assay; unequal durations and shared normalization limit their ratio. They do not connect a particular crystallographic atom to affinity or explain differences between cellular responses. NUDT5 protein loss, catalytic inhibition and nonenzymatic roles are different perturbations. The current Marques article distinguishes degradation and catalytic inhibition in a 6-thioguanine context [4]; Nguyen provides abstract-supported context for a nonenzymatic role [5]. Neither licenses an unmeasured mechanistic or breast-cancer claim here. Nguyen full-text/figure and Y74E-specific claims are not retained.

A proposed discriminating experiment would compare authenticated TH5427 and compound 9 against qualified human NUDT5 wild-type (WT), R51A and R51K, with surface plasmon resonance (SPR), orthogonal binding and direct functional readouts. These are unmatched ligands, so even a resolved differential response would be comparator-specific. Mutant folding, oligomeric state, active fraction, ADPr turnover, compound identity, aggregation and detector interference must be qualified first. R51K is not assumed conservative and neither variant is called a resistance allele. NUDT14 binding/function is a separate cross-target requirement. A thermal shift alone is not a KD measurement. The conditional laboratory specification defines supportive, refuting and inconclusive outcomes, orthogonal confirmation and pilot-based precision. Nothing has been executed, priced, scheduled or authorized for acquisition.

The practical output is therefore a bounded, reproducible description and an explicit separation of evidence types. Distance recalculation alone does not establish a scientific novelty threshold. Journal suitability remains an editorial question, not a conclusion of software validation. The unresolved author and laboratory gates do not become closed because the document is readable or the code passes tests.

## 5. Limitations

The deposited models, reports and maps share experimental/model dependencies. Fixed slices are limited inspection; there is no omit/polder map, rerefinement, expert 3D review, coordinate-error estimate or independent ligand-placement validation. Alternate conformers, fractional occupancy, partial residues, nulls and nonindependent crystal copies constrain every structural statement. The separate targets have different numbering and crystallographic contexts. There is no proximity-to-energy, affinity, causality, homology or selectivity inference.

Published catalytic means and SDs do not resolve replicate pairing, covariance, kinetic comparability or uncertainty of their ratios. Source-bound censoring and protocol disagreements remain explicit. Database indexing and graph normalization are not independent experimental evidence or proof of physical sample identity. The retained-source check is bounded, not a systematic literature or exhaustive correction search.

The small label dataset has unverified activity provenance, unmatched unassayed decoys, one invalid structure and misidentified reference graphs. The historical library and its dependent results are unavailable. Missing artifacts do not prove that earlier work never occurred. Screening, matching, calibration, resampling, conformal and applicability-domain limitations remain in the diagnostic supplement. No new library, decoy rebuild, docking, molecular dynamics, deep-learning study, biological experiment or prospective validation was undertaken.

Author-owned affiliation, funding, COI, CRediT, approvals, rights, prior-version/submission history, acknowledgments, reviewer conflicts and AI-assistance disclosure remain unresolved, including Nikhil Srinivasan's retained contribution and authorship. The exact author request sheet records what is needed and why. Laboratory capability, verified materials, independent-unit design and pilot variance are unknown. The journal-specific guide was inaccessible; topical scope does not certify submission compliance. No final declarations, cover letter or reviewer list are prepared.

## 6. Conclusion

Deposited compound-9 geometry supports an all-site, atom-specific description of the Arg51 environment, qualified by occupancy and local model limitations. Published paired catalytic measurements add protocol-bound context. Connecting either record to differential binding energy or cellular selectivity requires evidence that this reanalysis does not supply.

## Data and code availability for author review

The repository retains source snapshots, fixed contracts, complete geometry and pharmacology tables, source witnesses, historical results and manifests. The Path B build exports the main manuscript, separate diagnostic supplement, generated tables/figures, exact score pointers and a new completion-last manifest. The audit archive preserves the baseline manuscript and original claim dispositions with hashes; the current gap register records present closure state separately. These files are for author review and do not assert public-deposit updates, licensing clearance or submission readiness.

## References

[1] Page BDG et al. Targeted NUDT5 inhibitors block hormone signaling in breast cancer cells. Nature Communications 9, 250 (2018). DOI: 10.1038/s41467-017-02293-7. Funding acknowledgment correction: Nature Communications 10, 5050 (2019), DOI: 10.1038/s41467-019-12806-1. The official synthesis supplement was archived; its internal procedure pages were not certified by the bounded source check.

[2] Balikci E et al. Unexpected Noncovalent Off-Target Activity of Clinical BTK Inhibitors Leads to Discovery of a Dual NUDT5/14 Antagonist. Journal of Medicinal Chemistry 67(9), 7245 (2024). DOI: 10.1021/acs.jmedchem.4c00072. Primary full text, Table 1, Methods, Figures 3–4 and author CSV support the retained source-specific claims; exact witnesses are archived.

[3] Protein Data Bank entries 8RIY and 8OTV, with wwPDB validation reports, deposited structure factors and PDBe precomputed maps. https://www.rcsb.org/structure/8RIY and https://www.rcsb.org/structure/8OTV. Individual retrieval URLs, timestamps, versions where available and file hashes are retained in the structure-comparison source manifests. The PDBe validation-PDF copy was used after the RCSB endpoint returned 403.

[4] Marques et al. Targeted Protein Degradation of NUDT5 Dissociates Catalytic Inhibition from Protein Loss in 6-Thioguanine Response. Nature Communications 17, 8192 (2026). DOI: 10.1038/s41467-026-74489-9. This current article is distinct from the 2025 preprint, DOI: 10.1101/2025.03.16.643557.

[5] Nguyen et al. A non-enzymatic role of Nudix hydrolase 5 in repressing purine de novo synthesis. Science 390(6778), 1143–1150 (2025). DOI: 10.1126/science.adv4257. Metadata/abstract-supported context only; the full text remained inaccessible. Online and issue dates are distinct.
