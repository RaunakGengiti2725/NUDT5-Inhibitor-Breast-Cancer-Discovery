# NUDT5 extension: bounded second scientific/editorial review

**Snapshot:** `extension-review-v1.zip`, 4 October 2026. Locations below refer to this snapshot. Read-only review of manuscript §§4.5–4.9, their controls/transfer outputs, and supporting source/structural/study reports. The parent’s already-listed corrections are not counted as new findings.

**Verdict:** The released score-level results reproduce; the remaining problems are chiefly reporting and inference boundaries. The extension moves coherently from label controls to measured-source challenge, a probe-pair comparison, evidence categories, and proposed experiments. Preserve that structure, but repair the following before circulating it as a scientific result.

## Ranked remaining repairs

### 1. Major — restore the 10 µM column and threshold-specific denominators

**Location/excerpt:** `research/manuscript.md:126–137`: Table 2 displays only “Measured IC50 < 50 uM” and “Measured IC50 < 1 uM”; the caveat describes “five-versus-five class counts.”

The JSON and supplementary output retain **all three prespecified thresholds**. At **1 and 10 µM**, there are **2 positives and 8 negatives**, not five versus five: three numeric inhibitors above those thresholds join the five censored negatives. At **50 µM**, there are five versus five. The identical 1/10 µM results reflect identical labels, not independent confirmation.

| Model(s) | IC50 <1 µM; 2/8 | IC50 <10 µM; 2/8 | IC50 <50 µM; 5/5 |
|---|---:|---:|---:|
| Property_LR | 0.5625 | 0.5625 | 0.8000 |
| GBT | 0.9375 | 0.9375 | 0.9400 |
| RF, SVM_RBF, Nearest_active, Equal_mean (each) | 1.0000 | 1.0000 | 1.0000 |

**Smallest repair:** Add the missing column and positive/negative counts; define labels as **published mean IC50 strictly below the cutoff versus measured values at/above it or censored IC50 >50 µM**. Do not call every threshold-negative compound inactive. Scope the “one misordering … 0.96” statement to the 5×5 comparison; the corresponding strict inversion at 2×8 gives 0.9375. State that assay uncertainty is not propagated: compound 1 is **0.837 ± 0.329 µM, mean ± SD, n=2**, so its mean-based 1 µM classification is not evidence of an uncertainty-resolved boundary (`source_assays.csv:2`).

### 2. Major — the perfect-ranking statement wrongly includes GBT

**Location/excerpt:** `manuscript.md:135`: “the fingerprint and similarity methods rank every measured inhibitor above every reported inactive.”

GBT is explicitly an ECFP4 method but scores **0.9375/0.9375/0.9400** at 1/10/50 µM. In the 50 µM comparison, compound 14 ties reported-inactive compounds 12, 13 and 15. Thus “every … above every” is false even without an inverted pair.

**Smallest repair:** Name **RF, RBF-SVM, nearest-active and Equal_mean** as the four perfect-ordering methods; retain GBT’s exceptions. Replace the §4.6 heading’s “collapses” with the numerical result or “ranks lower”: 0.800 at 50 µM is not a general collapse to chance. Qualify “numerically best on repository labels” as the original six-method exact-scaffold comparison, not every extension ablation or split.

### 3. Major — add direct citations where the new source and mutant claims appear

**Locations/excerpts:** `manuscript.md:141`, “reported at approximately 85 nM and 10 uM”; `:173`, “published E112Q … and Y74E … variants.” The nine-item bibliography has **no SGC/MSD dossier entry**, and §§4.7/4.9 have no adjacent source citations.

**Smallest repair:** Cite the [SGC/MSD producer dossier](https://www.thesgc.org/chemical-probes/mrk-952) and its [SPR figure](https://www.thesgc.org/sites/default/files/inline-images/download_17.png) / [biochemical curve figure](https://www.thesgc.org/sites/default/files/inline-images/download%20%281%29_13.png). Identify producer evidence rather than implying a dedicated peer-reviewed MRK assay publication. Say **AMP-Glo IC50 in dossier prose**, versus curve-labelled **EC50**, and retain the existing species/construct and assumed-stereochemistry caveats. For SPR, the ledger records **n=2 and an unspecified ± uncertainty type**; do not silently call it SD, SEM or a confidence interval.

Attach existing reference **[4], Nguyen Figures 4–5**, directly to the mutant properties; use **[5]** for the inhibitor-versus-protein-loss precedent, and **[1,2] plus the identified PDB entries** for the structural/hormone rationale. These are already available references, not a request for a new literature audit.

### 4. Major — restore the roadmap’s qualification gates; feasibility is unverified

**Locations/excerpts:** `manuscript.md:171–173`: “one decisive, affordable experiment”; “Y74E … catalysis retained”; “a negative result at any gate is informative.”

The supporting study is appropriately more conditional: `study/NUDT5_extension_report.md:14–18` says stocks, prices and local capabilities were not checked and calls H1 “substantial cellular work”; `:45–55,69–73` requires qualified constructs and both catalytic readouts. “One qualified ER-positive model” does not itself qualify each mutant or chemical perturbation.

**Smallest repair:** Call this a **gated separation-of-function program**, not a demonstrated affordable single experiment. Add a short qualification sentence covering near-endogenous expression/localization, folding/dimerization, PPAT binding, **ADP-ribose hydrolysis and PPi-dependent ATP production separately**, and compound-specific engagement/selectivity. A failed qualification or absent usable phenotype is **inconclusive for the biological hypothesis**, though useful for a stop decision. Preserve the excellent same-cell-model requirement and explicit statement that none of these experiments was performed here. Specify that **8RIY/8OTV** are the compound-9 cross-paralogue comparison; the five listed structures do not all contain one ligand.

### 5. Minor — a supporting audit count is wrong

**Location/excerpt:** `research/audit.md:59`: “AUC to 1.000 for five of six methods.”

Among the original six, **four** reach 1.000: Property_LR, RF, SVM_RBF and Nearest_active. Equal_mean is 0.981781 and GBT 0.932186. Including the added Fingerprint_LR gives **five of seven**, not five of six.

**Smallest repair:** Name those five methods, as the manuscript already does, or correct the denominator. This numerical repair is separate from the parent’s planned removal of the stronger split/mechanism interpretation.

### 6. Minor — the release does not contain “every … bootstrap draw”

**Location/excerpt:** `manuscript.md:195` makes that promise for the two JSON files. The bootstrap objects contain counts, interval summaries and limitations, not draw-by-draw scores or sampled indices; `scripts/scripts/controls.py:409–444` likewise returns summaries.

**Smallest repair:** Say **“bootstrap summaries and reproducible resampling specifications”**, unless individual draws are actually added. No need to regenerate results merely to repair this sentence.

## Preserve; propagate; remaining uncertainty

- **Verified:** 42 pooled control AUCs, 51 ablation AUCs and 18 threshold AUCs independently recomputed from released scores; conformal coverage/set-size summaries reproduced; all 60 exported metric rows agree with JSON; input hashes in both extension manifests match. All 23 transfer source records match the supplied assay ledger. The reported headline rounding is consistent.
- **Source identity:** Live SGC CSV InChIKeys/formulas match both scored MRK structures. The pair has no reported exact/parent/tautomer training overlap; its low similarities and separate biochemical/binding endpoints should remain visible. A descriptor/score ordering is not a potency estimate.
- **Strong passages to retain:** §4.6’s retrospective/single-source limitations; §4.7’s weak-inhibitor, species/construct and endpoint caveats; §4.9’s “coordinates establish geometry, not energetics”; the supporting roadmap’s explicit falsifiers and qualification gates. Keep database mirrors non-independent, Kinobead Kd apparent, and the 2017 **IC50 >100 µM** record censored and primary-value-unverified.
- **Propagate existing parent corrections:** especially `audit.md:59–62`, `external/NUDT5_external_evidence_report.md:66–70`, and `external/artifact_sources.json:47`, which retain stronger split/exchangeability/pair-falsification language. This is synchronization, not another independent scientific finding.
- **Authenticated TH5427 sensitivity:** not present as the parent’s proposed separate refitted sensitivity in this snapshot, so not reviewed. Preserve the original CSV/results; label any sensitivity’s changed reference identity, training composition and comparisons explicitly. Do not imply that this review validates that future output.

**Limits and integrity:** This was not a model refit, new biological experiment, exhaustive literature search or full raw-source re-audit. Targeted live SGC and Nguyen checks were possible; a later Europe PMC XML request returned HTTP 500, so detailed mutant qualification still rests partly on the supplied source-backed reports. No prospective or clinical inference follows. **All 78 supplied archive members remain byte-identical.** No repository/deposit edits, commits, pushes or PRs were made.
