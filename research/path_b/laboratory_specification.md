# Conditional OWNER-LAB experiment specification

Baseline `40b9b0708d888a015abe5043bb273c3c6ee601ae`. Audited baseline manuscript SHA256 `5ee7257be5a09c9862c2ead2f98416e1e8d1b2fa875624ca009df5cb95934bbb`. Baseline provenance and executed commands: audit_archive/input_manifest.json; audit_archive/commands_run.md. **Conditional and unexecuted.** No experiment was performed, authorized, scheduled, quoted or contracted. No laboratory was contacted. Every outcome below is a prespecified possibility, not a result. Current Path B revision: the unsupported yield-frequency assertion has been removed. The exact original specification remains in the audit archive. This specification refines the existing plan in `research/structure_comparison/lab_handoff.md` and `hypotheses_controls.csv` (all rows `measurement_status=unmeasured`); it does not supersede them. Path B does not require this work, and the paper must keep it explicitly proposed.

I invent no sample size, no effect size, no equivalence margin, no variance, no cost, no timeline and no result. Where such a value is required, this specification names who must supply it and from what.

## 1. Question

Does TH5427 depend more than compound 9 on the Arg51 environment of human NUDT5? The motivation is the audited all-site geometry: the nearest retained Arg51 witness is a backbone nitrogen at one site and a fractional-occupancy side-chain CD at the other, and neither is a guanidinium atom. Geometry raises the question; it cannot answer it.

TH5427 and compound 9 are **unmatched comparators**, not a matched molecular pair. A positive result would establish comparator-specific differential dependence under the tested conditions only. It would not localize a causal atom, prove a hydrogen bond, establish selectivity or predict any new ligand.

## 2. Proteins and compounds

Human NUDT5 wild type, R51A and R51K, in one documented construct background; and wild-type human NUDT14 for the cross-target comparison. The source publication reports NUDT5 residues 1-208 and NUDT14 residues 1-222 as N-terminal His/TEV constructs with cleavage and size-exclusion purification; those are **reported settings, not settings validated for R51 variants**. No reciprocal homolog mutation is proposed, and neither mutant may be called a resistance allele. Do not assume R51K is functionally conservative.

Compounds must be authenticated TH5427 and compound 9 from a verified source: supplier, catalogue and lot recorded; identity confirmed by NMR or LC-MS and purity by an orthogonal method in the testing laboratory; salt form, counterion, stereochemistry, stock solvent, concentration verification, solubility and aggregation behaviour over the tested range documented. Known compounds remain known controls even when remeasured.

### Authenticated published compound sources

TH5427 is Page et al. compound 28, identified structurally by CCD 9CH and PDB 5NWH. The source handoff archives Page's official synthesis supplement and verifies its download/hash; its internal procedure pages were not text-inspected, so no step-level procedure or page number is certified here. Compound 9 is the Balikci et al. W0O ligand in 8RIY/8OTV; its synthesis and analytical characterization passages were inspected in the archived full text. The exact claim witnesses and source URLs are in source_checks/source_claims.json and source_checks/source_register.json. These sources establish published chemical identities and routes, not vendor stock, present author availability, future-lot purity or mutant-assay feasibility. No contact or acquisition is authorized. Any laboratory must retrieve the complete route, evaluate feasibility and authenticate each actual lot independently. The repository's mislabelled ACT-01/02 reference graphs are not procurement specifications.

## 3. Measurements and readouts

**Primary readout.** Direct equilibrium dissociation constant K_D for each of the four protein-compound combinations per mutant, by surface plasmon resonance. The source used amine-coupled CM5 at 25 C, 2% DMSO, 30 uL/min, 60 s association and 200 s dissociation; the laboratory must re-qualify every one of those settings for the variants rather than adopting them. Report each K_D with its own uncertainty and fit support. IC50 is never substituted for K_D and no Ki conversion is made. An unresolved or censored estimate is reported as unresolved, never inserted as a point value.

**Primary contrast.** For each separately qualified mutant M:

`delta_M = ln[K_D(M, TH5427) / K_D(WT, TH5427)] - ln[K_D(M, compound 9) / K_D(WT, compound 9)]`

Report delta_A and delta_K separately, with every underlying K_D. Do not select the better-behaved mutant and do not pool the two.

**Orthogonal binding, required.** An independently qualified second binding method, preferably solution phase, chosen by the laboratory after feasibility assessment. Candidates include isothermal titration calorimetry or a qualified solution-phase binding titration (for example MST or NMR); a thermal shift alone is not a direct affinity estimate; the laboratory selects on the basis of compound solubility, protein stability and dynamic range. Agreement in direction across qualified methods is required for the scoped inference. An instrument reporting a fit is not agreement.

**Function.** ADPr hydrolysis turnover for every protein preparation: quantify substrate depletion and AMP/product formation by a qualified direct chromatographic assay (LC-MS or HPLC with authenticated standards), alongside the coupled readout. Record calibrated product concentration over time, initial rates, response to substrate concentration and enzyme-load linearity, with recovery and detection limits established in the pilot. Keep ATP production separate from the hydrolysis readout. A shifted substrate response is not inhibitor resistance.

**Cross-target.** Wild-type NUDT14 binding and function for **both** compounds under interpretable conditions. NUDT14 sparing may not be inferred from the NUDT5 mutant panel. The published 20 minute NUDT5 and 60 minute NUDT14 protocols and the shared 500 nM TH5427 zero anchor are not universally qualified controls and must be re-justified.

## 4. Qualification criteria, all required before any interpretation

- Sequence verification and lot identity for every protein preparation, with independent preparations defined explicitly by the laboratory.
- Purity, folding and thermal stability, for example by circular dichroism or thermal profiling.
- Oligomeric state, for example by SEC-MALS, since NUDT5 functions as a dimer.
- Retained ADPr hydrolysis with direct product detection for every preparation, including each mutant.
- Active protein fraction and verified concentration.
- Compound solubility and absence of aggregation across the tested range, with a detergent or aggregation control.
- SPR reference and blank surfaces, solvent correction, vehicle and nonspecific-binding controls, plus surface density, mass-transport and rebinding checks and an explicit assessment of equilibrium versus kinetic fit adequacy.
- Interference and coupling controls for the functional readout: detector-only, no-enzyme, product-spike and, where luminescent detection is used, an inhibition-of-detection control.
- A documented pilot establishing independent-preparation variance, from which the laboratory and a statistician derive the independent unit, the precision target, the sample size, the equivalence margin, any multiplicity adjustment and the stopping rule. **None of these values exists yet and none is supplied here.** Technical wells are never biological replicates.
- A dated analysis lock, frozen after qualification and before acquiring validation data: fixed materials, panel, protocol, randomization, custodial blinded identifiers, exclusion rules, unblinding plan and analysis code with input hashes.

## 5. Prespecified interpretation

**Supports greater TH5427 dependence** only if, for a separately qualified mutant, delta_M is resolved and positive with its interval excluding no-difference under the pilot-justified precision rule, **and** the orthogonal method agrees in direction, **and** that mutant retains folding, oligomeric state and ADPr turnover.

**Refutes or challenges it** if delta_M is resolved and negative, or is statistically equivalent to no differential effect under the prespecified equivalence margin, for a qualified mutant. Equal penalties for both ligands are not differential recognition.

**Inconclusive/uninterpretable and reported as such** if the mutant is unfolded, altered in oligomeric state or has lost function; if binding is unresolved or censored for any required combination; if the contrast interval spans meaningful positive and nonpositive effects; if the orthogonal method lacks range or precision, or disagrees in direction; if aggregation, solubility or interference is uncontrolled. Discordant but individually qualified mutant results must remain separately reported and narrow the variant-specific interpretation rather than invalidate both automatically. A surface-dependent effect that disappears solution-phase points to assay dependence, not biology. An uninterpretable outcome is published as uninterpretable.

Even the strongest supporting outcome establishes comparator-specific differential dependence under the tested conditions. It does not establish a causal atom, a hydrogen bond, selectivity, a therapeutic effect or any property of a future ligand.

## 6. Cellular attribution: a separate later gate

Only after biochemical qualification, and only with separate authorization, ask whether an additional compound-9 cellular effect at **matched measured NUDT5 engagement** depends on NUDT14. That would require validated NUDT14 perturbation with wild-type rescue, independent perturbation reagents, target-abundance measurement, phenotype dynamic-range and floor-effect checks, reporter-interference controls and assessment of BTK and other off-target explanations. Equal nominal doses, NanoBRET EC50 values and CETSA shifts are not interchangeable occupancy measurements. Nothing here authorizes cell work or supports a breast-cancer claim.

## 7. Capability and qualitative cost drivers

Required capability: recombinant protein expression and purification with mutagenesis; protein-quality characterization (CD or thermal profiling, SEC-MALS); SPR with method development on variant surfaces; at least one orthogonal solution-phase binding method; a quantitative enzymatic assay with direct product detection; compound handling with identity and purity verification; and access to statistical design support for the pilot and lock.

Cost and schedule drivers, qualitative only, since no quote exists: the number of protein constructs and independent preparations; mutant expression yield and stability; SPR method development time on variant surfaces; which orthogonal method proves feasible; authenticated compound acquisition and purity verification; pilot size, which is unknown until variance is measured; and repeat cycles after any failed qualification gate. I will not estimate a figure or a timeline. A provider quote is required (gap G-M12).

## 8. Stop decisions

If materials or constructs cannot be verified, do not interpret target-specific comparisons. If a protein fails qualification, stop that mutant's inference and never substitute a favourable narrative for the failed preparation. If binding cannot be resolved, report feasibility failure and do not manufacture a K_D or design chemistry from incomplete geometry. If precision or governance is unresolved, do not acquire confirmatory data and do not claim a completed prospective lock.
