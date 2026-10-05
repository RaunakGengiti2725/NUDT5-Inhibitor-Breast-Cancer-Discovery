# Executable-analysis design contract, version 1

This is a **retrospective, locally recorded implementation contract**, not a preregistration: source values and frozen scores were already inspected. Its normative machine counterpart is `analysis_contract.json`; canonical evidence is `paired_evidence.json`. This handoff does not implement a new CLI or alter project code. The next implementer can implement the specified analysis without recuration, training, label imputation or scientific choices hidden in plotting code.

## 1. Input boundary and output boundary

Use Python 3.12 and the existing hash-locked README environment. No additional dependency is needed for arithmetic, CSV/JSON parsing or assertions. RDKit identity helpers already exist in `scripts/scripts/transfer.py`; importing pure `identity`/`identity_matches` is allowed, calling fitting/scoring functions is not. The input hash set and actual source retrieval records are in `provenance.json`. Do not change historical hashes/manifests to claim this curation was their original run.

Suggested implementation interface (specification, **not an existing command**): an `analyze_selectivity(evidence_path, score_path, output_dir)` function returning a manifest plus all-row source table, all-row score accounting and separately eligible descriptive table. Optional figures must additionally write PNG/PDF/SVG. It should accept the curated files, not scrape mutable sources during numerical analysis. An intentional re-curation requires a new version and disagreement record.

Validate every input before writing. Reject nonempty output directories, existing files, duplicate JSON keys, nonfinite numbers, malformed CSV, missing required fields, duplicate IDs/graphs, conflicting source rows, ambiguous score joins or unknown states. Stage output on the destination filesystem, flush/fsync, publish without replacement using the existing `pipeline.write_json` hard-link pattern, and clean only owned staging files. A multi-file failure must not leave an apparently complete report; publish completion manifest last. Never overwrite originals or previous output. CSV conventions: UTF-8, LF records, newline='', exact header, quoted fields preserved, strict parsing, no ragged rows or duplicate/empty headers. Raw source whitespace is evidence: preserve it rather than normalizing the `author_csv_raw` field.

## 2. Schemas and null semantics

`paired_evidence.schema.json` is Draft-07 JSON Schema with closed objects, fixed 23-row release size, target enums and mutually exclusive endpoint/ratio states. It validates structure; the semantic/identity/hash/foreign-key checks below are additionally required. A new release with different row count requires a versioned schema, not deletion of missing rows.

Each row is keyed by **original string `source_compound`**, not an integer (17a/17b and N-Boc-protected 7 are valid IDs). `source_csv_line` is the 1-based source line including its header. `source_smiles` is byte-equivalent cell content from the author CSV. The identity object preserves canonical isomeric SMILES, InChI/InChIKey, formula, fragment count, formal charge, unassigned stereo count, neutral-parent and canonical-parent-tautomer keys, and scaffold. Exact isomeric graph is the join boundary; parent/tautomer transformations are only supplementary exclusion flags.

Both `endpoints.NUDT5` and `endpoints.NUDT14` must exist for every graph, even untested ones:

| State | Comparator | reported_mean | reported_sd | bound | bound_strict |
|---|---|---|---|---|---|
| numeric | = | finite >0 | finite >=0 | null | null |
| right_censored | > | null | null | 50 µM | true |
| untested | null | null | null | null | null |

`endpoint=IC50`, `endpoint_family=purified_enzyme_catalytic_inhibition`, and `unit=uM` are fixed here. Binding KD/Ki, cellular EC50, viability and thermal-shift values cannot enter these fields. Table 1's raw “NA” **means >50**, while absent `table1_text` means the row is not in that table. `reported_mean` is never an imputed limit. The ratio of displayed source means is not a fitted concentration-response curve. `biological_n_for_reported_sd=2` only for numeric Table 1 rows with explicit footnote support. For 4/5 it is null because caption/Methods aggregation is unresolved, not because no repeats occurred. Censored/untested rows have no reported SD to assign an n to. Protocol-level replication wording remains in `assay_conditions.json`.

JSON null becomes an empty CSV field, **never zero or the string NA**. Boolean fields in the current CSV projection are literal `True`/`False`; parse only those spellings and empty only for nullable booleans. Arrays in `target_evidence.csv` (`source_ids`, `source_locations`, `discrepancy_ids`) use semicolon-separated members; no current member contains a semicolon. The CSV is a flat projection; the JSON is canonical for typed arrays and nested endpoint identity.

- `source_verification.csv`: one compound/target row per cell; CSV line, raw source text, ledger text, optional Table 1 text, content-match boolean, and `table1_check=agrees|not_in_table1`. Absence is not disagreement.
- `assay_conditions.json`: two records keyed by `conditions_id`. Endpoint references resolve even for untested cells, but `conditions_application` then explicitly means context only, not an observed assay.
- `source_disagreements.json`: stable D1–D6 records with source locations, representations and unresolved interpretation. Preserve alternate 4/5 SDs rather than treating them as extra measurements.
- `frozen_scores.csv`: one original compound/scenario row, all six numeric scores, JSON pointer, identity verification, overlap/missingness exclusion reasons, paired eligibility, ratio status, nearest-training context, and six method-specific rank columns. Exactly 46 rows (23 per scenario). Rank blanks are deliberate for noneligible rows.

## 3. Ratio arithmetic and censoring

Define **R = reported mean IC50(NUDT14) / reported mean IC50(NUDT5)** and **L = log10(R)**. R>1/L>0 favors the lower NUDT5 reported mean, only under the source assay conditions. It is not an affinity constant, a clinical selectivity measure, a ratio of raw paired observations, a mean ratio or a confidence interval. Do not silently invert to the older structural report's orientation.

The following pure function is an executable reference for the arithmetic dispatch after strict endpoint validation. It is a specification, not an installed API. It returns `(status, comparator, value)` where value is a **point only for `point`**, a strict bound for upper/lower states, and null otherwise. Keep separate point/bound columns in output; never plot all three as exact values.

```python
def reported_ratio(nudt5, nudt14):
    s5, s14 = nudt5["status"], nudt14["status"]
    if "untested" in (s5, s14):
        return "missing_endpoint", None, None
    if s5 == s14 == "right_censored":
        return "double_censored", None, None
    if s5 == "right_censored":
        return "upper_bound", "<", nudt14["reported_mean"] / nudt5["bound"]
    if s14 == "right_censored":
        return "lower_bound", ">", nudt14["bound"] / nudt5["reported_mean"]
    return "point", "=", nudt14["reported_mean"] / nudt5["reported_mean"]
```

Use `math.log10(value)` on the positive point/bound and preserve its comparator. Both >50 permit any positive R and any real L: no informative finite bound. Any untested target takes precedence over censoring on the other. For compound 13, `3.72/(>50)` gives **R<0.0744**, not R=0.0744, not R>13.4 and not a CI. Carry the NUDT14 SD 0.190 µM in its source endpoint; do not use it to decorate the strict bound with fabricated confidence.

Do not propagate marginal SDs into ratio uncertainty: raw biological replicate fits, target pairing and covariance are unavailable. Do not infer paired observations from two numbers on the same compound row. Synthetic examples in `contract_test_vectors.json` are **SOFTWARE TESTS ONLY**, never read as source evidence, never inserted into measured ledgers or biological Results.

## 4. Identity, eligibility and frozen-score joins

1. Confirm unchanged author CSV/raw graph and pinned ledger strings. Check every source ID, not just numeric rows. Reject duplicate IDs and exact graphs; do not silently collapse aliases or average conflicting assays.
2. Recompute the original **45 valid unique** training identities with pinned RDKit. Keep invalid ACT-18 in quarantine accounting; no guessed repair. Compare canonical isomeric graph, neutral fragment parent and canonical parent tautomer separately; no salt/stereo changes to original source identity. Standard InChI, scaffold, name or fingerprint similarity alone is insufficient for a score join.
3. Exclude model diagnostics if **any** of those three training-match lists is nonempty, or either target is untested. Keep every row in source pharmacology and all-score accounting. Historical overlaps: 10→ACT-20 and 11→ACT-19. Candidate 11→NC5-02 is a separate alias, not a new observation. Parent/tautomer checks may erase details, so do not transfer activity labels between normalized structures.
4. Join stored rows from `/measured_source_challenge/rows` as primary. Require ID, source SMILES, exact canonical identity, every recomputed identity field and recorded raw source row to match. Retain RF, GBT, SVM_RBF, Nearest_active, Property_LR and Equal_mean without rescaling or tuning. `Equal_mean` is the existing four-method mean (excludes Property_LR). Do **not** call any scoring/fitting function.
5. Join `/reference_sensitivity/measured_source_challenge/rows` separately. Recompute exclusion keys using only the authenticated ACT-01/ACT-02 replacements already specified in `reference_structures.csv`; no label changes/refitting. Sensitivity adds overlap 6, whose endpoints are both untested; paired eligibility remains identical. Do not choose the favorable branch or count both scenarios as twice the N.
6. Verify all source/input hashes and disclose historical run provenance unchanged. If identity or score join fails, fail the promised score-table release; retain the pharmacology record and an explicit unavailable-score reason rather than zero, refitting or silent exclusion.

**Do not reuse the transfer challenge's 10 eligible IDs as the paired cohort.** That earlier function only required a NUDT5 observation; 2/3/4/5 lack NUDT14. Source: 5 point + 1 upper-bound + 2 double-censored + 15 missing rows. Nonoverlap paired: 1/9/12/13/14/15, n=6; informative ratio n=4; exact point ratio n=3. All Table 1 rows remain visible in the full source table, including 10/11. This does not become an independent sample merely because exact overlap was excluded.

## 5. Descriptive outputs, not a selectivity benchmark

Retain source order for the all23 pharmacology table. For the separate six-row diagnostic table use historical Equal_mean descending, and display all six methods, ratio/comparator/state and rank denominator. Method-specific rank is `1 + sum(other_score > score)` within the relevant six-row scenario, yielding shared minimum ranks for exact ties. Keep full stored precision to identify ties. No ratio rank for censored/missing values; do not treat compound13's bound as its true position. Values are full binary64 precision in JSON/CSV and rounded only in presentation.

The scientific question is whether a high existing NUDT5 label score can coexist with published dual/NUDT14-lower reported means. Show the values/ranks, not a post hoc “high” threshold or selective/nonselective label. Compound9 is a source-described dual inhibitor with R=0.6 and the top six-cohort Equal_mean; compound1 has R≈1.183. Neither R<1 nor R>1 alone establishes statistical or biologically meaningful selectivity. Report 14 and censored13 too, and retain unfavorable/inconclusive patterns. No selectivity classifier, AUC, AP, correlation significance, optimized thresholds, p-values or biological bootstraps.

No new figure is required because the tables answer the question. If a future plot is implemented, it must show **log10 R (dimensionless; >0 favors NUDT5)** against an explicitly uncalibrated score, labelled compound IDs, upper-bound arrows/open markers for13 and a separate no-informative-ratio strip for12/15. Keep overlaps and untested rows in an accompanying complete table. Label **n=6, 3 point ratios** and SD/censoring limitations; use colorblind-safe Okabe–Ito colors plus shapes/text, and publish downloadable PNG/PDF/SVG outputs. Do not add fabricated uncertainty bands or crop away inconvenient rows.

## 6. Acceptance tests and hard failure cases

Run all supplied SOFTWARE-ONLY vectors and validate finite arithmetic with `math.isclose(rel_tol=1e-12, abs_tol=1e-12)`. These cover orientation, both directions of strict censoring, double censoring, missing numerator/denominator and equality. Equality at R=1 is a numeric state, not a validated biological label. Also test every invalid case in that file against the parser/output validator; these are requirements for the future implementation, not claims that the new CLI exists.

Current release invariants: 23 unique source graphs, 46 cells; NUDT5 7 numeric/5 >50/11 untested; NUDT14 6 numeric/2 >50/15 untested; eight paired source rows; six nonoverlap paired; exactly three nonoverlap point ratios; complete all23 rows in both score scenarios; no favorable filtering. Round-trip quoted CSV, trailing spaces, nullable fields and booleans. Assert exact unchanged frozen scores, method names, graph/ID matches, tied ranks and source hashes. Mutated hash/graph, duplicate ID/graph, NaN/inf, mixed endpoint families, numeric values for untested rows, >= substituted for >, manufactured mean50, or double-censored R=1 must fail closed. Nonempty outputs refuse without changing any byte; interrupted writes do not leave a completed report.

Resolve no source ambiguity by preference for better metrics. D1/D2 retain caption/CSV SD/whitespace differences; D3 retains the TH5427 NUDT14 normalization uncertainty; D4 retains missingness; D5 makes ratio orientation explicit; D6 retains replication-level uncertainty. All16 paired Table1 cells agreed, so there is no invented paired-value discrepancy or alternative favorable ratio scenario to select.
