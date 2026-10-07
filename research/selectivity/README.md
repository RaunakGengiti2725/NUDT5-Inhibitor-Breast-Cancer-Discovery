# Paired NUDT5/NUDT14 evidence: curated handoff

**Curation of previously inspected Balikci 2024 evidence, not new experiments or fresh external validation.** Only `research/selectivity/` is changed. Original ledgers, historical numerical JSON and PR #1 are untouched. No models were fitted or tuned by this curation. The two-stage read-only verification workflow is recorded in `independent_verification.json`.

## What the table answers

**High frozen NUDT5 label-diagnostic scores also occur for a published dual compound with a lower reported NUDT14 mean.** Among the six nonoverlap paired compounds, compound **9 ranks first** on historical `Equal_mean` (0.811292), while its reported IC50s are **0.270 ± 0.027 µM for NUDT5** and **0.162 ± 0.005 µM for NUDT14**: R=0.6. The paper calls it a dual inhibitor. This is descriptive coexistence, not validation of a selectivity classifier. Compound 1 ranks second and has R≈1.183; that uncertain point ratio near unity is **not a validated NUDT5-selective label**. Method-specific scores differ and are all shown below.

There are **23 source graphs and 46 endpoint cells**, but only **8 paired source rows**. Excluding exact/parent/tautomer training overlap leaves **6 paired compounds**, of which **3 have point ratios, 1 a strict upper bound and 2 no informative finite ratio bound**. These are compound counts, not raw paired-replicate counts. No AUC, significance test, ratio CI, model tuning or generalization claim is appropriate.

## Full source pharmacology accounting (all 23 rows)

IC50 units are **µM**. Numeric strings preserve source-reported mean ± SD; paired Table 1 values use n=2 biological replicates. Compounds 4/5 have uncertainty-reporting ambiguities below. Untested is not inactive. `>50` is a strict source-censored bound, not exact 50 or a numeric mean. R is **reported mean IC50(NUDT14)/reported mean IC50(NUDT5)**; R>1 favors lower reported NUDT5 IC50 under these particular assay conditions. Ratios have no units.

| Original compound ID | NUDT5 IC50 (µM) | NUDT14 IC50 (µM) | R or strict bound | Model diagnostic status |
|---|---:|---:|---|---|
| 1 | 0.837 ± 0.329 | 0.990 ± 0.110 | 1.1828 | eligible paired |
| 2 | >50 | untested | unavailable: untested target | missing_target_endpoint |
| 3 | >50 | untested | unavailable: untested target | missing_target_endpoint |
| 4 | 13.9 ± 0.62 | untested | unavailable: untested target | missing_target_endpoint |
| 5 | 21.2 ± 1.02 | untested | unavailable: untested target | missing_target_endpoint |
| 6 | untested | untested | unavailable: untested target | missing_target_endpoint |
| 7 | untested | untested | unavailable: untested target | missing_target_endpoint |
| 8 | untested | untested | unavailable: untested target | missing_target_endpoint |
| 9 | 0.270 ± 0.027 | 0.162 ± 0.005 | 0.6 | eligible paired |
| 10 | 0.487 ± 0.010 | 0.263 ± 0.031 | 0.540041 | training_exact_parent_or_tautomer_overlap |
| 11 | 2.04 ± 0.240 | 0.519 ± 0.084 | 0.254412 | training_exact_parent_or_tautomer_overlap |
| 12 | >50 | >50 | not estimable: both >50 | eligible paired |
| 13 | >50 | 3.72 ± 0.190 | <0.0744 (strict) | eligible paired |
| 14 | 13.8 ± 0.900 | 1.64 ± 0.140 | 0.118841 | eligible paired |
| 15 | >50 | >50 | not estimable: both >50 | eligible paired |
| 16 | untested | untested | unavailable: untested target | missing_target_endpoint |
| 17a | untested | untested | unavailable: untested target | missing_target_endpoint |
| 17b | untested | untested | unavailable: untested target | missing_target_endpoint |
| 18a | untested | untested | unavailable: untested target | missing_target_endpoint |
| 18b | untested | untested | unavailable: untested target | missing_target_endpoint |
| 18c | untested | untested | unavailable: untested target | missing_target_endpoint |
| 19 | untested | untested | unavailable: untested target | missing_target_endpoint |
| N-Boc-protected 7 | untested | untested | unavailable: untested target | missing_target_endpoint |

Both >50 values for 12 or 15 permit any positive R (and any real log10 R): **neither R=1 nor a finite one-sided bound is identified**. Compound 13 gives **R<0.0744**, log10 R<−1.128427, using its reported NUDT14 mean; this is a censoring-derived strict bound, not a confidence bound. The earlier structural report's >13.4 uses the reciprocal NUDT5/NUDT14 orientation. Compounds 10→ACT-20 and 11→ACT-19/NC5-02 remain in source pharmacology but are excluded from model diagnostics. No extra parent-only or tautomer-only training overlaps were found in the historical scenario.

## Separate frozen-score diagnostic (six nonoverlap paired rows)

Scores are copied from `research/results/transfer.json` only after exact graph and source-row verification. They are uncalibrated label-diagnostic scores, not IC50 predictions or selectivity probabilities. “High” means the displayed score and rank, not an outcome-chosen threshold. Rank is 1 plus the count of strictly greater scores in the six-row cohort; exact ties share the minimum rank. `Equal_mean` averages RF, GBT, SVM_RBF and Nearest_active, **not all six methods**. Point ratios near one are not classified.

| ID | R | log10 R | RF | GBT | SVM_RBF | Nearest_active | Property_LR | Equal_mean | Equal_mean rank / 6 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 9 | 0.6 | -0.221849 | 0.690000 | 0.999975 | 0.894815 | 0.660377 | 0.853829 | 0.811292 | 1 |
| 1 | 1.1828 | 0.0729097 | 0.540000 | 0.999975 | 0.707449 | 0.492537 | 0.957093 | 0.684990 | 2 |
| 14 | 0.118841 | -0.925035 | 0.390000 | 0.153046 | 0.412072 | 0.347222 | 0.928141 | 0.325585 | 3 |
| 12 | not estimable: both >50 | — | 0.280000 | 0.153046 | 0.316333 | 0.270270 | 0.691137 | 0.254912 | 4 |
| 13 | <0.0744 (strict) | <-1.12843 | 0.280000 | 0.153046 | 0.311128 | 0.270270 | 0.691137 | 0.253611 | 5 |
| 15 | not estimable: both >50 | — | 0.260000 | 0.153046 | 0.283165 | 0.250000 | 0.697845 | 0.236553 | 6 |

Compounds 1 and 9 tie on GBT at full stored precision. Compound 14 has a high Property_LR score despite a much lower reported NUDT14 mean, whereas 13 scores lower on the displayed methods. No favorable method is selected, and censored/untested observations do not become threshold-negative labels. Nonoverlap is not independence: this is related chemistry from the same already-exposed publication, trained on mostly unauthenticated labels and presumed-negative decoys.

### Existing authenticated-reference sensitivity, separately retained

These are the **already-recorded** ACT-01/ACT-02 graph-substitution scores, not a new rerun or a preferred replacement model. Compound 6 becomes a training overlap here but is untested for both IC50 endpoints; the same six paired compounds remain. R does not change. All 23 score rows per scenario, all six methods and tied ranks are retained in `frozen_scores.csv`.

| ID | RF | GBT | SVM_RBF | Nearest_active | Property_LR | Equal_mean | Equal_mean rank / 6 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 9 | 0.740000 | 0.999975 | 0.915668 | 0.660377 | 0.868497 | 0.829005 | 1 |
| 1 | 0.570000 | 0.999975 | 0.753011 | 0.492537 | 0.963285 | 0.703881 | 2 |
| 14 | 0.400000 | 0.419520 | 0.467284 | 0.347222 | 0.935503 | 0.408507 | 3 |
| 12 | 0.310000 | 0.419520 | 0.373775 | 0.270270 | 0.740752 | 0.343391 | 4 |
| 13 | 0.310000 | 0.419520 | 0.362942 | 0.270270 | 0.740752 | 0.340683 | 5 |
| 15 | 0.270000 | 0.419520 | 0.334263 | 0.250000 | 0.749928 | 0.318446 | 6 |

## Primary-source verification and noncomparability

Balikci et al., *Unexpected Noncovalent Off-Target Activity of Clinical BTK Inhibitors Leads to Discovery of a Dual NUDT5/14 Antagonist*, J Med Chem **67**, 7245–7259 (2024), [DOI](https://doi.org/10.1021/acs.jmedchem.4c00072), [PMC11089510](https://pmc.ncbi.nlm.nih.gov/articles/PMC11089510/).

- **Table 1**, XML `tbl1`, images `fx1/jm4c00072_0007.jpg` (scaffold) and `fx2/jm4c00072_0008.jpg` (values): all eight rows, both columns visually checked; all **16 cells agree** with author CSV and original ledger, including all means and SD digits. Footnote `t1fn1`: n=2 biological replicates. Footnote `t1fn2`: NA means IC50 **>50 µM**, not missing.
- **Author CSV** `jm4c00072_si_002.csv`: all 23 IDs, exact source SMILES and 46 assay cells checked. Numeric/status contents agree after trimming two trailing spaces. Original raw strings and exact bytes are archived.
- **Protein Expression and Purification**, `sec4.2`: NUDT5 NP_054861 residues 1–208; NUDT14 NP_803877 residues 1–222; pNIC28-Bsa4 N-terminal 6×His-TEV constructs expressed in E. coli (DE3); TEV treatment and reverse nickel purification. Do not substitute deposited crystallographic lengths or cell-assay reporter fusions for these constructs.
- **Catalytic Assays**, `sec4.3`: 1 nM enzyme, 10 µM ADPr, 2 µL in 1536-well plates, AMP-Glo luminescence, 1% DMSO. **NUDT5 20 min versus NUDT14 60 min** at room temperature. Buffer, stop/detection protocol and control wording are preserved in `assay_conditions.json`. IC50 ratios are not affinity constants, intrinsic kinetic selectivity or cellular/clinical selectivity; target-specific Km, linear progress and control qualification cannot be recovered from these passages.
- **Unresolved TH5427 control wording (D3):** the combined protocol defines 500 nM TH5427 as 0% activity without a NUDT14 exception, while Results describes TH5427 as NUDT5-selective. Do not silently repair the protocol or infer validated complete inhibition of NUDT14; this is an unresolved source-method concern, not proof of assay failure. The compound 6 “not tested” cells mean no catalytic IC50 estimate in this ledger, despite stated use as a control.
- **Figure 1B caption versus CSV (D1):** compound 4 SD 0.6 versus 0.62 µM; compound 5 SD 1.0 versus 1.02 µM. Means agree, and rounding is a plausible explanation, not an adjudicated correction. Neither compound has a NUDT14 value, so preserving both alternatives does not alter paired ratios or counts.
- **Replication (D6):** Table 1 reports IC50 mean/SD across n=2 biological replicates. Methods says “triplicate sets”; Figure 1 describes technical-triplicate plotted mean/SD in a representative biological experiment. No raw target-paired IC50s, covariance or complete aggregation procedure were available in these inspected materials. Do not use n=6 or manufacture ratio error bars/CIs from marginal SDs.
- **Coverage and lexical differences (D2/D4):** two CSV cells include trailing spaces removed by the historical ledger; its “parsed cell values unchanged” provenance claim is therefore not literally exact for whitespace. Results' “all compounds were tested” occurs in Table 1 SAR context and does not override 26 explicitly untested cells across the full CSV. Original files are unchanged.

Exact retrieval times, URLs, SHA-256 hashes, archive members and locations are in `provenance.json`. The XML and author CSV plus two Table 1 images are stored as **lossless gzip snapshots** in `sources/`, not screenshots or newly generated figures. Decompress them to inspect original bytes. Article attribution/license is retained (CC BY 4.0 in the inspected permissions). All requested primary artifacts were accessible; no broad source search, new plates or physical experiments occurred.

## Handoff files

| File | Role |
|---|---|
| `paired_evidence.json` + `paired_evidence.schema.json` | Canonical 23-row evidence with exact identity, target endpoints, exclusions and ratio state; strict JSON Schema |
| `target_evidence.csv` | 46 target-long rows with source raw text, mean/SD or strict bound, replicate/context and provenance |
| `frozen_scores.csv` | Separate 46 compound/scenario rows with all six frozen methods and within-six ranks |
| `source_verification.csv` | Cell-level cross-check, including explicit not-in-Table-1 status |
| `assay_conditions.json`, `primary_excerpts.json` | Target-specific protocol conditions and precisely located source text |
| `source_disagreements.json` | Attributed numeric, lexical, control, coverage and replicate ambiguities without source edits |
| `analysis_contract.json`, `analysis_contract.md` | Concrete executable-analysis design, parser/identity/join/censoring/output contracts |
| `contract_test_vectors.json` | **SOFTWARE TESTS ONLY**, artificial arithmetic cases; not measured data or synthetic biological curves |
| `summary_counts.json`, `provenance.json`, `independent_verification.json` | Denominators, immutable-input/source hashes, two-stage review record |

See `verification.md` for executed checks. This handoff supplies curated evidence and an implementation-ready design, **not a new analysis CLI, experiment-ready qualification, completed biological validation, or novelty/efficacy claim**.
