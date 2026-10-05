# A differential NUDT5/NUDT14 experiment, not another docking campaign
**Read-only primary-source investigation | 4 October 2026**

## Recommendation
Use the **same experimentally bound ligand, Balikci compound 9, in NUDT5 (8RIY) and NUDT14 (8OTV)** to design a differential-recognition experiment. Pair that experiment with selective-versus-dual chemical perturbation and genetic rescue in hormone-stimulated breast-cancer cells. The useful project advance is a **causal selectivity panel**: does a phenotype require NUDT5 catalysis, NUDT14 inhibition, or loss of the NUDT5 protein?

This is a proposed project direction, **not a claim that the pocket differences or compounds are newly discovered**. Balikci’s 23-structure CSV is already curated; none of it is offered as new external training data. Project identities NC5-02 = ACT-19 = compound 11 and ACT-20 = compound 10 are accepted from the brief, not re-audited.

## 1. Verified structural starting points
All five deposits are X-ray structures of **human proteins expressed in E. coli**, with zero deposited sequence mutations. Ligand and metal counts below refer to the **asymmetric unit**, not solution stoichiometry. Author-assigned residue numbers and exact missing ranges are in the metadata CSV.

| Structure | Experimental ligand | Resolution | Deposited construct / assembly | Other modeled components; unmodeled residues |
|---|---|---:|---|---|
| [5NWH](https://www.rcsb.org/structure/5NWH) | TH5427, **9CH**, two copies | 2.60 Å | NUDT5, 219-residue sequence; A/B dimer | Water; no modeled metal; 50 residues absent across two chains |
| [5NQR](https://www.rcsb.org/structure/5NQR) | TH1713, **958** | 2.20 Å | NUDT5, 219-residue sequence; A/B dimer | Water; no modeled metal; 48 absent residues |
| [8RDZ](https://www.rcsb.org/structure/8RDZ) | Ibrutinib / compound 1, **A1H14**, four copies | 2.02 Å | NUDT5, 219-residue sequence; **A/C and B/D dimers**, not a biological tetramer | Eight Mg ions, seven ethylene glycols, water; 115 absent residues across four chains |
| [8RIY](https://www.rcsb.org/structure/8RIY) | Compound 9, **W0O**, two copies | 2.288 Å | NUDT5 residues 1–208 + Ser0; dimer, author chains AAA/BBB (label A/B) | Water; no modeled metal; 31 absent residues |
| [8OTV](https://www.rcsb.org/structure/8OTV) | Compound 9, **W0O**, two copies | 1.82 Å | NUDT14 residues 1–222 + Ser0; A/B dimer | One Mg ion, one DMSO, water; 23 absent residues |

**Preparation caveats that matter to the design:**
- Page’s Methods specify crystallization of NUDT5 residues **1–210**, whereas 5NWH/5NQR deposit 219-residue sequences and model residues only through 208. Balikci specifies **1–208**, but 8RDZ also deposits a 219-residue sequence. These discrepancies are unresolved, not silently treated as full-length experimental constructs. 8RIY/8OTV agree with the stated protein regions plus the residual Ser tag.
- 5NWH lacks A162–163 internally; 8RDZ lacks portions of the 53–57 region, and 8RIY lacks BBB54–56. 8OTV lacks A83–85, A173–177 and B169–177, plus terminal residues. Do not build a contact or mutation argument on an absent loop. 8RIY also flags a zero-occupancy atom at AAA Arg51 CZ; examine local density before atom-level redesign.
- 5NQR has **three ligand residue records with occupancies 0.5, 0.5 and 1.0**, consistent with the paper’s alternate orientations and two bound molecules, not three full-occupancy sites. The dimer must be retained: both papers assign binding-site contributions to both subunits.
- The matched-ligand structures have **different crystal metal/solvent states**. No modeled Mg is not evidence of metal-independent catalysis. Balikci SI Table S2 also reports anisotropic-data completeness of 85.8% for 8RIY. Resolution alone does not establish contact certainty.

**Chemical identity anchor:** [CCD 9CH](https://www.rcsb.org/ligand/9CH) is TH5427, formula **C20H20Cl2N8O3**, InChIKey **QXCXMVYVUHVFLP-UHFFFAOYSA-N**, with a **1,3-dimethyl** xanthine core. Use that deposited graph and Page’s synthetic identity when correcting the known reference-graph problem. [CCD 958](https://www.rcsb.org/ligand/958) is TH1713, not TH5427. No repository graph was edited or compared here.

## 2. Published observations versus the proposed mechanism
**Published:** Page describes TH5427/TH1713 binding through the xanthine carbonyls and the NUDT5 Glu47/Arg51 environment, with Trp28/Trp46 stacking. Balikci describes compound 9 between NUDT5 Trp46/Trp28, versus NUDT14 Trp34/Tyr17; the aminopyrimidine interacts with Glu47 or Asp35, respectively. Its phenoxy region engages the Arg51 environment in NUDT5 versus Leu107 in NUDT14. Balikci explicitly discusses nonconservation of Arg51 as a possible explanation for TH5427 selectivity. These are **the authors’ assignments**, not newly calculated contacts. [P, B]

**Hypothesis H1:** ligand affinity can be made more dependent on the NUDT5 Arg51 environment than on the corresponding NUDT14 pocket. Test this first with existing ligands and structurally intact NUDT5 R51K/R51A variants. Compare the effect on **direct binding** of TH5427 versus compound 9. These are chemically distinct comparators, not a matched molecular pair. A larger TH5427 affinity penalty would support differential dependence; equal penalties, no differential effect, or mutant unfolding would respectively weaken the proposal or make it uninterpretable. Mutants are proposed tests, not validated resistance or catalytic-dead alleles.

**A low-cost SAR arm:** remeasure the already-published 9/10/11 N1-substituent series side by side. Removing compound 10’s N1 methyl to make 11 raises the reported NUDT5 IC50 about **4.19-fold**, but NUDT14 IC50 about **1.97-fold**. That motivates differential recognition as a question; there is **no inspected 10/11 cocrystal**, so the binding pose and cause cannot be inferred from 9. Only after these tests should a focused acceptor-bearing/acceptor-deleted ligand pair be designed in the experimentally supported Arg51-facing region.

## 3. Compound panel: roles, not new hits
Values are published mean **IC50, µM**, not new measurements. Balikci used different reaction times for NUDT5 (20 min) and NUDT14 (1 h), so cross-target ratios are provisional assay ratios, not affinity constants. [B, Table 1; existing source CSV]

| Compound | NUDT5 / NUDT14 | Recommended role and constraint |
|---|---|---|
| Authentic TH5427 | 0.029 / no published dose-response value verified here | Orthogonal, NUDT5-preferring comparator. Page reports **38% NUDT14 inhibition at 100 µM**, not a NUDT14 IC50. Retest at the chosen exposure. |
| 9 / W0O | 0.270 / 0.162 | Structural and dual-inhibition anchor. Not NUDT5-selective or BTK-free. |
| 10 / ACT-20 | 0.487 / 0.263 | N1-methyl member of the existing matched series. Cellular selectivity unverified. |
| 11 / NC5-02 / ACT-19 | 2.04 / 0.519 | Existing N1-H comparator, nominally about 3.93-fold NUDT14-biased. Not a new NUDT5-selective lead. |
| 13 | >50 / 3.72 | Provisional NUDT14-biased **biochemical** comparator, nominal ratio >13.4. Electrophile and cellular off-target/engagement questions remain. |
| 12 or 15 | >50 / >50 | Assayed-inactive scaffold controls, subject to exposure and interference checks; not universal cellular negatives. |
| Ibrutinib | 0.837 / 0.990 | Confounded BTK/NUDIX reference, not an attribution control. |

Balikci reports compound 9 NUDT5 NanoBRET EC50 **1.08 µM**, but BTK EC50 **0.377 µM**. NUDT14 HiBiT-CETSA stabilization was measured at **30 µM**, not as a cellular potency curve. The SI reports no TH5427-induced NUDT14 stabilization in that assay; a negative thermal shift is not proof of zero binding. Do not choose equal nominal doses and assume equal target coverage. Unassayed decoys are not biological negative controls.

## 4. Hypothesis-driven experimental sequence
**H2:** an acute hormone-response defect can arise from NUDT5 catalytic inhibition without NUDT5 protein loss; additional effects of dual compounds might instead involve NUDT14 or other targets. This is a testable proposal, not established by crystallography.

1. **Establish trustworthy biochemical separation.** Verify compound identity/purity/solubility; quantify ADPr-to-AMP conversion directly by LC-MS/HPLC for purified human NUDT5 and NUDT14, with orthogonal SPR binding. Establish linear progress curves and substrate dependence for each enzyme before comparing selectivity. Use matched free Mg, solvent, detergent/aggregation checks, no-enzyme blanks and product-spike detection controls. Do not assume TH5427 is a complete-inhibition control for NUDT14. Measure the NUDT5 PPi-dependent ATP-producing reaction separately; ADPr hydrolysis IC50 does not automatically establish ATP-synthesis inhibition.
2. **Falsify the pocket explanation.** Test WT versus proposed R51 variants only after fold, dimer and catalytic-baseline characterization. Prefer direct binding when mutation changes substrate kinetics. Reject mechanistic interpretation if the mutant is unstable or globally inactive; do not equate Arg51 mutation with selective ligand resistance.
3. **Define a cellular engagement window.** Titrate NUDT5 engagement and NUDT14 engagement separately, alongside protein abundance and reporter-only interference controls. Check BTK expression/activity; an orthogonal BTK inhibitor becomes a useful control only after both NUDIX counterscreens. Advance to attribution only if differential target coverage is actually observed.
4. **Test acute biology before viability.** Start with hormone-starved T47D cells ± R5020, reflecting Page’s model. Compare TH5427 and 9 at matched measured NUDT5 engagement. Measure early nuclear ATP dynamics and later hormone-responsive transcription, then EdU/cell counts. Include no-hormone, cytosolic ATP, direct reporter-inhibition and general toxicity controls. A bulk ATP measurement alone cannot establish a nuclear mechanism.
5. **Separate NUDT14 effects.** Cross vehicle/TH5427/9 with inducible NUDT5 depletion, NUDT14 depletion, and double depletion, using independent reagents and WT rescue. An extra compound-9 effect that depends on NUDT14 and returns with NUDT14 rescue would support its contribution; persistence after verified double depletion points to another target or artifact. Avoid floor effects: lack of additivity alone proves neither mechanism.
6. **Separate catalysis from protein loss.** Measure total and nuclear NUDT5/NUDT14 abundance at every acute and delayed endpoint. Compare NUDT5 depletion rescued by near-endogenous WT versus an independently validated, folded/dimeric, correctly localized catalytically inactive construct. WT-only rescue supports catalytic dependence; rescue by the inactive protein supports a noncatalytic protein function. Partial or failed rescue without construct validation is inconclusive. Inhibitor effects preceding detectable protein loss support—but do not alone prove—a catalytic mechanism.

Use independent biological repeats and confidence intervals; select concentrations from measured exposure/engagement rather than these literature numbers. The experimental-plan CSV gives controls and explicit falsifiers. Page’s DARTS assay measures protease protection after lysis, not drug-induced protein loss. No experiments, mutations, docking or MD were performed here.

## 5. Scope and limits
**Inspected:** Page and Balikci primary full texts; selected supplementary captions/methods, Balikci Table 1 image and SI Table S2; the existing Balikci CSV; five current mmCIF deposits, six assembly records, polymer metadata and four CCD identities. Missing-residue totals were independently cross-checked against RCSB API counts.

**Not searched/tested:** exhaustive later literature, patents, all NUDT14 deposits, repository data/code, clinical relevance, compound availability, structure factors/electron-density re-refinement, new contact distances, docking, MD or cellular assays. Wright’s earlier nuclear-ATP work was cited by Page, not independently investigated here. No literature-novelty, efficacy or acceptance guarantee is made.

Structure analysis cannot establish cellular selectivity, inhibition kinetics, occupancy at a biological dose, protein degradation, scaffolding functions, permeability, or breast-cancer efficacy. Page’s verified result is a **hormone-dependent T47D response**, not generic tumor killing. Balikci also found no significant protein-bound ADP-ribose change in its tested U2OS ARH3-KO setting; that does not settle hormone-driven nuclear ATP biology. [P, B]

### Exact primary sources
- **P:** Page et al., *Targeted NUDT5 inhibitors block hormone signaling in breast cancer cells*, Nature Communications 9, 250 (2018). DOI [10.1038/s41467-017-02293-7](https://doi.org/10.1038/s41467-017-02293-7). [Full text](https://pmc.ncbi.nlm.nih.gov/articles/PMC5772648/); [downloaded full-text XML](https://www.ebi.ac.uk/europepmc/webservices/rest/PMC5772648/fullTextXML); [supplementary package](https://www.ebi.ac.uk/europepmc/webservices/rest/PMC5772648/supplementaryFiles). Relevant: structural Results, TH5427 selectivity, Fig. 4, crystallography/selectivity Methods, SI Figs. 5 and 9, TH5427 synthesis.
- **B:** Balikci et al., *Unexpected Noncovalent Off-Target Activity of Clinical BTK Inhibitors Leads to Discovery of a Dual NUDT5/14 Antagonist*, Journal of Medicinal Chemistry 67, 7245–7259 (2024). DOI [10.1021/acs.jmedchem.4c00072](https://doi.org/10.1021/acs.jmedchem.4c00072). [Full text](https://pmc.ncbi.nlm.nih.gov/articles/PMC11089510/); [downloaded XML](https://www.ebi.ac.uk/europepmc/webservices/rest/PMC11089510/fullTextXML); [supplementary package](https://www.ebi.ac.uk/europepmc/webservices/rest/PMC11089510/supplementaryFiles). Relevant: Table 1, Figs. 1–5, catalytic/protein/crystallography Methods, SI S11–12 and S44, and jm4c00072_si_002.csv.
- Deposits are linked above. Exact mmCIF, entity, assembly and CCD endpoints, retrieval hashes and a claim-to-source ledger accompany this report.


## Integration pointer (5 October 2026; original investigation preserved above)

The later [all-site observed-coordinate run](../structure_comparison/results/VERIFICATION.md)
and [bounded lab handoff](../structure_comparison/lab_handoff.md) supersede the prospective
contact-calculation recommendation, not this report's historical scope. They retain all sites,
missingness and occupancy, including a backbone Arg51 minimum at one NUDT5 site. They do not
establish the published Arg51 rationale, energetics, selectivity or experiment readiness.
No earlier source inspection or runtime provenance is retroactively claimed.
