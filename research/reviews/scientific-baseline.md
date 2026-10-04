# Bounded skeptical review of the revised NUDT5 snapshot

**Verdict:** Substantially sound as an evidence-limited reproducibility case study, not a validated discovery paper. No new critical defect found. Two material reporting repairs and four minor corrections remain. The earlier claim that Property_LR is best under molecule splitting **is already fixed**; do not reinstate it.

Reviewed the attached archive, extracted outside any clone. Locations below are one-based lines in that snapshot; `research/` is omitted from document paths. No supplied file was modified. This is one bounded model-assisted review, not independent human peer review or another literature audit.

## Ranked findings and smallest repairs

### 1. Major — the decoy-matching result lacks its executable definition
**Location:** `manuscript.md:94`.

> “Only 9 of 26 decoys match any valid active under the stated joint molecular-weight, cLogP and Tanimoto criteria; none matches ACT-01 through ACT-17.”

The revised Methods do not state the numerical matching tolerances, similarity inequality or boundary convention. Neither supplied result JSON contains this matching result, and the released pipeline does not implement that joint matching check. Thus this particular headline cannot be independently checked from the supplied definition and artifacts. This is a reproducibility gap, **not evidence that 9/26 is wrong**.

**Smallest repair:** State the exact three conditions and link a per-decoy matching record or the actual check used. Until that is available, identify the number as a prior audit result whose calculation is not supplied, rather than a fully reproducible result of this package. Do not invent tolerances.

### 2. Major — two absolute formulations exceed the audit's own limits
**Locations:** `manuscript.md:15`; `audit.md:26`, finding 11.

> “it supplies the minimal diagnostic battery that makes that distinction.”
>
> “Transferability-Weighted Consensus Scoring was never implemented as described”

The first follows the distinction between a working screen and a biased benchmark, yet the Discussion correctly says the controls cannot isolate what features explain the models' performance. Neither sufficiency nor minimality is established. “Never” similarly extends beyond the inspected release, despite the explicit allowance for an unreleased historical procedure at `manuscript.md:122`.

**Smallest repairs:** “it supplies diagnostic controls that expose the limits of the released evaluation”; and “The inspected released code does not implement the described transfer weighting.” Retain the existing non-misconduct and missing-artifact caveats.

### 3. Minor — one duplicated AUC rounding error
**Locations:** `manuscript.md:82`; `audit.md:48`.

> “| Gradient-boosted trees, ECFP4 | 0.929 | 0.843 | 0.292 |”

`results/benchmark.json`, `series.metrics.GBT.auc` (line 4990), is **0.291497975708502**. Direct three-decimal rounding is **0.291**, not 0.292. Independent pairwise calculation from released predictions agrees with the JSON.

**Smallest repair:** Replace the series value in both tables with **0.291**. No substantive interpretation changes.

### 4. Minor — Methods conflate bootstrap intervals and seed ranges
**Locations:** `manuscript.md:59`, contrasted with `:90`.

> “Reported ranges are a **conditional cluster resampling** of exact-scaffold groups from fixed out-of-fold predictions.”

This blanket statement is not true of the separate five-seed ranges, which come from repeated fitting/splitting. The Results already distinguish the two correctly.

**Smallest repair:** Begin “Reported bootstrap intervals use conditional cluster resampling…” and add “Separately, seeds 42–46 assess sensitivity to repeated fitting and splitting.” Keep the non-population-uncertainty caveats for both.

### 5. Minor — citation scope is broader than the cited implementation documentation
**Location:** `manuscript.md:112`.

> “Descriptor and nearest-neighbour baselines, explicit chemical grouping and source-level identity checks are established safeguards rather than novel algorithms [7,8].”

Reference 7 identifies RDKit scoring/fingerprint APIs; reference 8 documents grouped cross-validation. These support implementation details, but not the entire methodological-history claim, particularly the status of descriptor baselines and source-level identity checks.

**Smallest repair without another literature search:** Recast as a description of the controls used here, place [7,8] beside the implementations they document, and [1,2] beside the concrete source-identity checks. Preserve the non-novelty conclusion without suggesting these API pages substantiate every safeguard.

### 6. Minor — citation status is stale in the findings table
**Location:** `audit.md:33`, finding 18.

> “The Marques work is preprint `10.1101/2025.03.16.643557`.”

The manuscript reference 5 (`:138`) and claim ledger (`:22`) correctly include the 2026 journal article. Crossref confirms *Nature Communications* 17, 8192, published 30 June 2026, DOI **10.1038/s41467-026-74489-9**.

**Smallest repair:** Say “2025 preprint, subsequently published in 2026…” and include the journal DOI. These are related versions, not independent replications. The author-confirmation wording at `manuscript.md:130` can identify the journal version too.

## Checks that passed — preserve these corrections

### Benchmark reconciliation
All **18 pooled AUCs** were independently recalculated from released positive–negative prediction pairs, including ties, and match the JSON. Correct seed-42 table:

| Method | Molecule | Exact-scaffold | Positive-series |
|---|---:|---:|---:|
| Property_LR | 0.996 | 0.980 | 0.998 |
| SVM_RBF | 1.000 | 0.960 | 0.812 |
| RF | 1.000 | 0.941 | 0.639 |
| Equal_mean | 0.998 | 0.931 | 0.563 |
| Nearest_active | 1.000 | 0.913 | 0.329 |
| GBT | 0.929 | 0.843 | **0.291** |

- Series has **two folds**, with **2 positives + 13 decoys** and **17 positives + 13 decoys**. Equal_mean within-fold AUCs **0.654/0.828** versus pooled **0.563** are correctly distinguished. Property_LR is **1.000 in each fold**; pooling is not a single-model prospective test.
- At arbitrary threshold **0.5**, all five non-property methods have **0/19** sensitivity. Property_LR has **17/19**, but **0/2** in the smaller fold. The calibration/ranking cautions are appropriate.
- Seeds **42–46**: scaffold Property_LR **0.9798–0.9858**, Equal_mean **0.9211–0.9372**; series **0.9899–0.9980** and **0.4372–0.7267**. Seed-42 scaffold conditional intervals **[0.9082, 1] / [0.6849, 1]** match JSON.
- All six methods have **99 null scores**, **p = 0.01** with the plus-one correction. The manuscript appropriately avoids biological-significance or multiplicity-adjusted claims. The enumerated BEDROC reference-test design contains **1,004 patterns**; its tests were not rerun here.

### Chemistry, counts and provenance
- **46 raw records = 20 positive labels + 26 decoys; 45 valid = 19 + 26.** ACT-18 is invalid. Only two valid positives have the authenticated source mappings here; **17 valid positive identities/assay assignments remain unverified**, not demonstrated inactive or fabricated.
- Independently verified **NC5-02 = ACT-19 = Balikci compound 11** by canonical graph matching. Source NUDT5/NUDT14 values are **2.04 ± 0.240 / 0.519 ± 0.084 µM**. ACT-20 = compound 10: **0.487 ± 0.010 / 0.263 ± 0.031 µM**. Table 1 specifies SD and **two independent biological replicates**.
- All **23** source IDs, source SMILES and both assay columns match the primary supplementary CSV, allowing whitespace trimming; its SHA-256 matches provenance. Counts are **7 numeric NUDT5 values, 5 inactive (>50 µM), 11 untested**. These are not additional training observations.
- ACT-01 is **C19H18Cl2N8O3** versus authenticated 9CH **C20H20Cl2N8O3**: **one missing N-methyl substitution**, net C1H2, not two. The ledger's reference to the authentic compound having two N-methyls is not a claim that two are missing. NC5-01/authenticated-reference similarity independently reproduces **0.385542 → 0.386**.
- Pinned RDKit **2025.09.6** reproduces all **70** stored descriptor cells at CSV precision. The supplied 2023 record differs in four HBA cells: version dependence, not established transcription error. Candidate mean pairwise similarity independently reproduces **0.154596** over **45 pairs**. Audit JSON supports **3/10** at similarity ≥0.25, all ten passing PAINS/RO5 and nine passing Veber; none establishes activity or safety.

## Citation/interpretation boundaries and remaining uncertainty

The strongest passages to retain are `manuscript.md:86–94,110–122`: pooled-score and threshold limitations, confounding as an alternative rather than proven cause, and separation of catalytic inhibition, protein loss and clinical efficacy. The publication strategy remains conditional and makes no acceptance or clinical promise. At `publication_strategy.md:68`, adding the Balikci citation would make “the compound-9 BT-474 result” self-contained; the primary text indeed reports no viability effect under that experiment. This is not a contradiction.

Primary checks were limited to [Balikci XML/Table 1 footnote and BT-474 context](https://www.ebi.ac.uk/europepmc/webservices/rest/PMC11089510/fullTextXML), its [supplementary CSV](https://www.ebi.ac.uk/europepmc/webservices/rest/PMC11089510/supplementaryFiles), [RCSB 9CH](https://www.rcsb.org/ligand/9CH), [Marques bibliographic metadata](https://api.crossref.org/works/10.1038/s41467-026-74489-9), and [group-split documentation](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedGroupKFold.html). RDKit's documentation URL returned HTTP 406; its citation-scope assessment is limited to the APIs identified in the bibliography.

I did not independently reread all biological papers, rerun the full benchmark/test suite, inspect the absent original DOCX/PDFs, or reconstruct historical docking, screening, decoy matching or baseline-code outputs. Those historical quantities are not established by the new benchmark JSON. The cause of reported animal deaths remains unspecified, and unverified labels remain unverified. No absence of artifacts is treated as evidence of nonexistence or misconduct.

**Integrity:** read-only Python JSON/CSV and pairwise-AUC checks; targeted chemistry checks using `uv run --no-project --with rdkit==2025.9.6` (NumPy 2.2.6); primary-source comparisons. All five manifest-listed input/code/lock hashes match. All **25 supplied files remain byte-identical to the ZIP**. No repository, deposit, journal or external collaboration state was changed.
