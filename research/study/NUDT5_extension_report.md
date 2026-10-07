# NUDT5: a separable-function experiment, not another hit-ranking claim
**Decision report · 4 October 2026 · Read-only, bounded primary-source investigation**

## Recommendation and novelty boundary

Prioritize a **context-specific separation-of-function experiment**: does hormone-induced nuclear ATP production require NUDT5 catalysis, whereas a purine-stress response requires its PPAT-binding function? Use the published **E112Q** catalytic-deficient and **Y74E** PPAT-binding-deficient comparators, after requalification in the intended system. Combine these with chemically distinct inhibitors and a NUDT14 attribution arm. The actionable advance would be a causal boundary between functions in the breast-cancer model—not discovery of NUDT5 inhibitors, dual NUDT5/14 binding, or noncatalytic NUDT5 biology.

A pivotal citation changes the novelty assessment. **Nguyen et al., Science (2025), already report NUDT5–PPAT-mediated repression of purine synthesis**, catalytic-independent phenotypes, and Y74E separation-of-function experiments [S7]. **Marques et al., Nature Communications (2026), already distinguish catalytic inhibition from protein loss in 6-thioguanine response**, using MRK-952, TH5427, degraders and genetic rescue [S6]. These studies share investigators and the degrader toolkit; they are not wholly independent replications. Calling the same general mechanism a new discovery is untenable. Its transfer to the specific hormone-response setting is a testable extension, not an established result or a guaranteed literature novelty.

The supplied 45 structures, 19 positives/26 **unassayed** decoys, NC5-02 = ACT-19 = compound 11, ACT-20 = compound 10, and known TH5427 graph error are accepted context, not re-audited. Balikci's already-curated 23 structures are **not new external data**.

## Ranked hypotheses

Ranks reflect scientific value, not the cheapest execution order. Feasibility assumes access to the stated platforms; stocks, prices and local capabilities were not checked.

| Rank | Falsifiable hypothesis | Value / feasibility | Decisive extension |
|---|---|---|---|
| **1 — H1** | NUDT5 catalysis and its PPAT-binding function have distinguishable requirements across acute hormone response and purine-stress response. | **Highest value; conditional, substantial cellular work.** Published mutants and controls make this more actionable than inventing new probes. | A validated E112Q/Y74E rescue contrast in T47D, triangulated with TH5427/MRK-952. NUDT14 perturbation diagnoses any extra effect of dual compound 9. |
| **2 — H2** | The Arg51 environment contributes differentially to ligand recognition, and can support NUDT5-over-NUDT14 discrimination. | **Moderate–high value; moderate feasibility** with purified proteins and direct binding. | Quantify ligand-by-mutation binding interactions, then test a newly locked matched chemical pair. Existing structures alone cannot establish the energetic contribution. |
| **3 — H3** | An unchanged scoring rule transfers its claimed activity preference to the chemically distinct MRK-952/NC pair. | **Limited scientific value; highest computational feasibility.** Useful as a cheap challenge, not a publication-sized benchmark. | One frozen, fully disclosed pair ordering with trivial baselines and applicability-domain reporting. No new model fitting. |

**Execution:** perform the inexpensive H3 diagnostic on a later, explicitly scoped execution pass; qualify H1/H2 reagents and assays before committing to a large cell matrix. H1 can proceed without new chemistry. H2 medicinal chemistry should follow, not precede, a valid binding result.

## What the primary evidence actually supports

- **SGC/MSD probe pair:** dossier prose reports MRK-952 IC50 **85 nM** versus NC **10 µM**. The curve image instead labels fits **EC50 0.0849/10.6 µM**. These are not four independent observations. SPR gives **KD 0.031 ± 0.018 / 1.380 ± 0.380 µM, n = 2**. Its panel marks NUDT14 “NB” and “also inactive in NUDT14 catalytic assay,” but the inspected figures do not define the quantitative inactivity cutoff. The NC is a **weak inhibitor**, with “stereochemistry assumed,” not an inert control. Assay species/construct remain unresolved in this dossier. Human-cell use in S6 does not identify the purified-enzyme construct in S1 [S1–S3].
- **Existing dual chemistry:** Balikci compound 9 inhibits NUDT5/14 and binds both in structures 8RIY/8OTV. NUDT5 NanoBRET EC50 is **1.08 µM**, while BTK EC50 is **0.377 µM**; NUDT14 CETSA was measured at **30 µM**, not as an occupancy curve. Its NUDT5 and NUDT14 catalytic assays used different reaction times. No cellular selectivity window follows automatically from these numbers [S4, Figs. 3–5; Methods].
- **Biological contexts differ:** Page's TH5427 results concern hormone-stimulated T47D nuclear ATP readout, transcription and proliferation—not generic cancer-cell killing [S5, Fig. 4]. Nguyen's scaffold/PPAT experiments concern purine metabolism, including MTHFD1-deficient settings; Marques's inhibitor/degrader comparison concerns thiopurine response [S6–S7]. Neither permits automatic extrapolation to the proposed hormone experiment.

### Evidence categories

“Demonstrated” below means **reported primary experimental observation within its tested conditions**, not replication by this investigation. The accompanying JSON records claim-level sources and limits.

| Category | Scoped conclusion |
|---|---|
| **Demonstrated** | Published probe-pair potency/binding differences; compound-9 dual binding and BTK engagement; the reported T47D hormone phenotype; reported E112Q/Y74E rescue and PPAT-binding observations. |
| **Strongly supported** | A noncatalytic NUDT5–PPAT contribution in the tested purine contexts, through chemical, genetic, interaction and biochemical evidence. Not a universal mechanism for every NUDT5 phenotype. |
| **Suggestive** | Arg51-dependent ligand discrimination from published structures; MRK-952 as an orthogonal NUDT14-sparing comparator at an as-yet-unqualified exposure. |
| **Hypothesized** | H1's context-specific double dissociation; H2's prospective selectivity gain; H3's uncomputed fixed-model ordering. |
| **Unknown** | Mutant behavior and engagement windows in the intended breast-cancer system; MRK enzyme species/construct; repository overlap; actual scores; reagent availability; experimental outcomes. |
| **Refuted** | Universal interchangeability of catalytic inhibition and protein removal; treating MRK-952-NC as enzymatically inactive; calling compound 9 NUDT5-specific; presenting the general PPAT/noncatalytic mechanism as newly discovered here. |

## Falsification matrix

A failed qualification is **inconclusive**, not biological falsification. A confidence interval that includes both relevant benefit and no effect is also inconclusive; lack of statistical significance is not equivalence.

| Hypothesis / test | Predicted result | Result that challenges or falsifies the scoped claim | Crucial validity control |
|---|---|---|---|
| **H1: hormone arm** | After NUDT5 depletion, WT and qualified Y74E restore hormone-induced nuclear ATP response; qualified E112Q does not. TH5427 and MRK-952 suppress the response without protein loss. | E112Q restores the response despite verified loss of the relevant catalytic reactions; or Y74E fails despite preserved catalytic activity, expression and localization. Either challenges the proposed clean separation. | Near-endogenous rescue; folding/dimerization/localization; hydrolysis **and ATP-producing** reaction competence; PPAT binding; reporter interference and general toxicity. |
| **H1: context arm** | In a demonstrated purine-stress-responsive setting, WT/E112Q restore sensitivity, whereas Y74E does not. | A validated opposite rescue pattern, or no context interaction within predeclared equivalence bounds, challenges context dependence. | Known HAP1 purine-response experiment is a **positive-control replication**, not novelty. A new same-line T47D comparison requires prior assay competence. No detectable phenotype means no interpretable comparison. |
| **H1: dual-compound attribution** | An additional compound-9 effect, beyond matched NUDT5 engagement, depends on NUDT14 and is restored by NUDT14 rescue. | Persistence after validated NUDT14 loss challenges that attribution; persistence after adequate dual depletion points toward another target/artifact. | Avoid floor effects; independent depletion reagents; WT rescue; BTK engagement/activity and a separately counterscreened BTK control. |
| **H2: differential binding** | TH5427 suffers a greater R51-variant affinity penalty than compound 9, without generalized protein damage. A later new ligand pair reproduces a prespecified target-selectivity change. | No differential penalty within adequate precision, reversed direction, or no selectivity shift in the prospective pair. | WT/R51K/R51A fold, dimer and basal function checks; direct KD rather than interpreting changed substrate kinetics as resistance. Mutants are not validated resistance alleles. |
| **H3: one pair** | Frozen scoring favors MRK-952 over NC, **if** the score actually claims potency/activity discrimination. | Reverse order contradicts that pair-specific prediction; a tie/abstention supplies no ordering evidence. This does not prove the model has no utility elsewhere. | Entire processing/score/direction fixed beforehand. A curated-positive-versus-unassayed classifier is not a calibrated potency model. |

For H1, compare contexts in the **same qualified cell model**; HAP1 alone is a known positive-control replication. If the arms use different cell lines, context and cell identity are confounded and the claim must narrow to two model-specific results.

For H2, report the interaction contrast **ln(KD(mutant)/KD(WT)) for TH5427 minus the same quantity for compound 9**, with uncertainty. Do not silently substitute IC50 for KD. Existing 9/10/11 remeasurement is a useful reproducibility arm, not a new holdout or proof that its N1 substituent contacts Arg51. The supplied structural report's Arg51 occupancy/local-density warning must be resolved before atom-level redesign; it was not rechecked here.

## No-tuning retrospective versus prospectively locked validation

**Retrospective, diagnostic only.** The MRK pair's outcomes have already been seen. Before any scores, timestamp the model/data/configuration hashes, corrected authentic TH5427 identity, molecular standardization and stereochemistry policy, score direction, seeds, failures/abstentions and baselines. No fitting, weight adjustment, pose cherry-picking, endpoint switching or selective reporting. Report exact graph overlap and nearest training neighbors against **all 45 actual repository structures**, not just the prior 14 retrieved Balikci graphs. The prior low-similarity figures are inherited and do not settle repository disjointness.

Score the pair once and report the raw values, ordering and applicability domain. Prespecify trivial similarity baselines and any descriptor-ranking direction. A correct order is one concordant pair; under a symmetric random-order null its probability is **1/2**. It cannot establish superiority over baselines, generalization, calibration, AUC, enrichment or precision. The assay-species/construct gap bars a strict species-confirmed human biochemical holdout claim. Keep IC50, binding KD, cellular engagement and viability endpoints separate. Do not label 26 unassayed decoys as experimental negatives.

**Prospective, new outcomes only.** Use independent pilot/QC work to qualify assays, exposure windows, rescue constructs and precision. Exclude pilot outcomes from confirmatory evidence. Then lock the complete eligible compound/construct list, blinded IDs, primary endpoints and contrasts, plate randomization, independent biological units, exclusions, sample size, analysis, uncertainty/equivalence rules and stopping rule **before collecting the validation outcomes**. Hold the scorer fixed before blinded assays. Known MRK/NC and 9/10/11 remain controls, even if remeasured; they do not become prospective discovery predictions. Newly chosen compounds with no inspected activity outcomes can constitute prospective predictive validation under this lock. Amendments restart a separately labeled validation, not an edited success criterion.

**Empirical go/no-go gates—no invented numerical pass marks:**

1. **Materials:** authentic identity, purity/solubility and resolved stereochemistry/constructs at the intended exposure. Unresolved identity stops mechanistic attribution.
2. **Assays:** validated product quantitation/progress curves for human NUDT5 and NUDT14, orthogonal binding, reporter-only and aggregation controls; measure PPi-dependent ATP production separately. Interference or an unqualified dynamic range stops biological interpretation.
3. **Separation of function:** WT/E112Q/Y74E must exhibit the intended catalytic/PPAT-binding separation with comparable expression and appropriate localization; otherwise H1 is not testable with that panel.
4. **Cellular window:** demonstrate NUDT5 engagement with protein abundance measured, separately assess NUDT14/BTK and reporter effects. Equal doses, NanoBRET EC50 and CETSA thermal shifts are **not interchangeable occupancy measures**. Without a discriminating window, do not attribute compound-9 biology to one target.
5. **Lock and precision:** derive sample size and equivalence margins from independent assay repeatability and the smallest mechanism-relevant effect, justified before validation. Do not invent a fold-change, viability percentage or arbitrary p-value gate here. If feasible precision cannot resolve the planned contrast, stop or narrow the question.
6. **Adjudication:** advance only when the prespecified contrast is resolved and orthogonal evidence agrees. A precise contradictory result rejects the scoped hypothesis; wide intervals or invalid controls require an inconclusive report, not a success claim. H2 chemistry and H1 efficacy claims must not outrun their preceding gates.

## Realistic venue fit

These are scope-based judgments, **not acceptance probabilities**. At present the deliverable is a research plan; no original experimental result or method validation was produced.

| Venue | Potential fit after execution | Likely rejection argument now / remaining burden |
|---|---|---|
| **ACS Chemical Biology** [V1] | H1 with a genuinely new causal mechanism/boundary, selective perturbations and qualified rescue. | Merely repeating the 2025/2026 scaffold story, or adding a cell line without mechanistic insight. Its guidelines explicitly caution against follow-up Articles without extensive new information. |
| **ACS Medicinal Chemistry Letters** [V2] | H2 with experimentally supported new selectivity/SAR, or a new mechanism supported by independent assay types. | Existing compounds and already-described pocket differences are incremental; docking or unqualified assay ratios are insufficient. Chemical characterization, orthogonal binding/function and credible selectivity would be needed. |
| **Journal of Chemical Information and Modeling** [V3] | Only a broader genuine methodological advance with adequate experimental validation. | H3 alone has one exposed pair, no new method and no estimable generalization. Official scope excludes straightforward single-target docking without adequate experimental validation. |

If wet-lab work is unavailable, retain H3 as an honest internal diagnostic/reproducibility result rather than enlarging its claim to fit a journal.

## Exact sources and investigation boundary

- **S1:** [SGC/MSD MRK-952 dossier](https://www.thesgc.org/chemical-probes/mrk-952), overview and assay data. Producer evidence, not a dedicated peer-reviewed assay paper.
- **S2:** [SGC SPR/selectivity figure](https://www.thesgc.org/sites/default/files/inline-images/download_17.png).
- **S3:** [SGC biochemical curves/stereochemistry figure](https://www.thesgc.org/sites/default/files/inline-images/download%20%281%29_13.png).
- **S4:** Balıkçı et al., 2024, [10.1021/acs.jmedchem.4c00072](https://doi.org/10.1021/acs.jmedchem.4c00072); [inspected XML](https://www.ebi.ac.uk/europepmc/webservices/rest/PMC11089510/fullTextXML), Results/Figs. 3–5 and catalytic Methods.
- **S5:** Page et al., 2018, [10.1038/s41467-017-02293-7](https://doi.org/10.1038/s41467-017-02293-7); [inspected XML](https://www.ebi.ac.uk/europepmc/webservices/rest/PMC5772648/fullTextXML), Fig. 4, hormone Results and ligand-contact assignments.
- **S6:** Marques et al., 2026, [10.1038/s41467-026-74489-9](https://doi.org/10.1038/s41467-026-74489-9); [inspected XML](https://www.ebi.ac.uk/europepmc/webservices/rest/PMC13462928/fullTextXML), Figs. 1, 3–4, Discussion and reference 45.
- **S7:** Nguyen et al., 2025, [10.1126/science.adv4257](https://doi.org/10.1126/science.adv4257); [inspected primary text](https://pmc.ncbi.nlm.nih.gov/articles/PMC7618541/), Figs. 4–6, Y74 mutagenesis Results and catalytic Methods. Structural PPAT model is predicted, not an experimentally solved complex.
- **V1/V2/V3:** Official [ACS Chemical Biology](https://researcher-resources.acs.org/publish/author_guidelines?coden=acbcct), [ACS Medicinal Chemistry Letters](https://researcher-resources.acs.org/publish/author_guidelines?coden=amclct), and [JCIM](https://researcher-resources.acs.org/publish/author_guidelines?coden=jcisd8) author guidelines, scope sections inspected.

**Actually inspected:** both supplied reports; selective pivotal passages from four primary papers, the SGC page/two figures; three official journal scopes. Scientific expansion stopped after the single S6→S7 citation hop. One PMID metadata lookup and three venue-scope web searches were performed. S7 XML returned HTTP 500; PMC HTML supplied accessible primary full text. The source register preserves all 12 directly downloaded HTTP responses/hashes, including the failure; venue-search queries are logged separately.

**Not searched or redone:** broad literature census, patents, PubChem, databases, vendors/availability, all later citations, repository contents or the 45-structure overlap, model reviews, raw experimental data reanalysis, new PDB/contact calculations, structure factors, supplementary-data re-audit, or the original nuclear-ATP paper cited by Page. No model scores, simulations or experiments were generated. No repository edits, commits, pushes, PRs, sessions, contacts or deposit changes occurred. No novelty, efficacy or acceptance guarantee follows from this bounded investigation.
