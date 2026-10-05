# Lab handoff: falsify differential Arg51 dependence before ligand redesign

**Decision aid, not physical-experiment-ready qualification or a completed prospective lock.**
No new samples, binding measurements or pilot outcomes are supplied. This narrows
[H2 in the earlier study](../study/NUDT5_extension_report.md#falsification-matrix), not its
cellular H1 programme. [hypotheses_controls.csv](hypotheses_controls.csv) is the machine-readable
hypothesis/control/falsification table. Every proposed outcome below is unmeasured.

## 1. What is known, and what the calculation adds

**Reported source observations.** Balıkçı et al. already reported compound 9 binding to NUDT5
and NUDT14, the 8RIY/8OTV structures, and the Arg51 rationale for ligand discrimination
([primary excerpts C02–C04](primary_excerpts.json); [paper, Figures 3–4](https://doi.org/10.1021/acs.jmedchem.4c00072)).
Their SPR estimates are approximately 250 and 400 nM, respectively, not our measurements or
new affinity estimates. Their ligand-region assignments do not establish Arg51/Leu107 homology.
TH5427 and compound 9 are **UNMATCHED comparators**, not a matched molecular pair.

**Newly calculated proximity.** The fixed observed-heavy-atom calculation uses all four W0O
sites and both deposited dimer chains, not a selected best site. Values below are rounded only
for display; identities and full precision are in [residue_proximity.csv](results/derived/residue_proximity.csv).
All sites are model 1, assembly 1. Chain names are label/auth; residue numbers are author
numbers (Arg51 label 52, Leu107 label 108). Witness atom IDs are mmCIF atom-site IDs.

| W0O site, label/auth:residue | Receptor residue | Minimum retained-atom witness | Distance, Å | Result-specific uncertainty |
|---|---|---|---:|---|
| 8RIY C/AAA:301 | A/AAA Arg51 | W0O C20 #2940 → backbone N #306 | 3.756 | Partial residue: CZ #314 has zero occupancy and is excluded. N occupancy 1; residue O is fractional. |
| 8RIY D/BBB:301 | B/BBB Arg51 | W0O C18 #2968 → side-chain CD #1775 | 3.251 | CD occupancy 0.78, unweighted; other Arg51 atoms also fractional. |
| 8OTV C/A:301 | B/B Leu107 | W0O C15 #3233 → CD2 #2419 | 3.609 | No deposited missingness flag for this row; not density validation. |
| 8OTV F/B:302 | A/A Leu107 | W0O C17 #3270 → CG #795 | 3.781 | No deposited missingness flag for this row; not density validation. |

The other-chain Arg51 minima are 14.329 Å at site C and 14.296 Å at site D, not omitted
counterexamples. At the fixed inclusive primary 4.0 Å radius, both nearby Arg51 rows qualify;
at 3.5 Å only the BBB row does. Sensitivity radii remain 3.5/4.0/4.5/5.0 Å; no cutoff was changed.
The AAA minimum is **backbone**, so a residue-level proximity flag cannot be read as uniform
side-chain dependence. Neither minimum is a guanidinium-group witness; residue-name proximity
is not guanidinium-interaction or energetic-dependence evidence. The BBB witness is carbon–carbon,
not hydrogen-bond proof. This does
not refute the authors' interaction assignments or exclude other atom pairs; it does not test
TH5427 geometry or establish the energetic premise of H2.

The full map retains 1,730 residue-conformer rows: 1,564 observed, 58 partial and 108 null/refused,
with 1,278 atom pairs within 5.0 Å. For partial rows the observed minimum is an upper bound on
the unknown complete-residue minimum, not an estimate of it. Missing residues are not no-contact.
8OTV B Leu47 alt A/B remain separate (all minima >5.0 Å). No density/structure factors were
inspected; coordinate uncertainty is not propagated. Crystal copies are not independent n,
and the distinct crystal buffers, solvent/metal inventories and resolutions preclude reading
between-site distances as affinity differences. See [the full run record](results/VERIFICATION.md).

## 2. Smallest informative experiment: qualified WT/R51 comparisons

**Proposed H2 test.** Measure direct binding of authentic TH5427 and compound 9 to human NUDT5
WT, R51A and R51K, using the same documented construct background and matched assay conditions.
Do not nominate either mutant as a resistance allele or assume R51K preserves WT function.
For each separately qualified mutant M, estimate the earlier H2 interaction contrast:

`ΔM = ln[KD(M,TH5427)/KD(WT,TH5427)] − ln[KD(M,9)/KD(WT,9)]`.

Report each underlying KD, its support/uncertainty and both contrasts; do not select the better
mutant. Positive ΔM is the proposed direction, not an observed result. No KD is inferred from
IC50, no Ki conversion is made, and an unresolved/censored binding estimate cannot be inserted
as a point value. A contrast spanning meaningful positive and nonpositive effects is inconclusive.
Actual independent-preparation variance, effect margins, precision and sample size remain unknown.

**Primary-method anchor, not an adopted SOP.** Balıkçı Experimental Section 4.2 used human
NUDT5 1–208 and NUDT14 1–222, N-terminal His/TEV constructs with cleavage and size-exclusion
purification. Section 4.8 used amine-coupled CM5 SPR at 25 °C, 2% DMSO, 30 µL/min, 60 s association
and 200 s dissociation. These are reported settings, not validated settings for R51 variants.
Exact quotations, source bytes and limitations are in [handoff_sources.json](handoff_sources.json).

**Controls that decide interpretability.** Before interpreting a binding difference, verify each
sequence/lot, purity, folding/stability and oligomeric state (for example CD or thermal profiling
plus SEC-MALS), and retained ADPr hydrolysis with direct product detection and progress/substrate
curves. A shifted substrate response is not inhibitor resistance. Keep ATP production separate
if later making an ATP-mechanism claim. Qualify active protein fraction, concentration, solubility
and aggregation over the tested range. For SPR use reference/blank surfaces, solvent correction,
vehicle and nonspecific-binding controls, and check surface density, mass transport, rebinding
and the adequacy of equilibrium/kinetic fits. Use an independently qualified orthogonal binding
method, preferably solution phase, chosen by the laboratory after feasibility checks; agreement
is required for the scoped inference, not assumed because an instrument reports a fit.

In parallel, measure functional ADPr turnover with orthogonal product quantitation and detector-only,
product-spike, no-enzyme and aggregation/detergent controls. The published AMP-Glo 20/60 min
NUDT5/NUDT14 protocols and shared TH5427 zero anchor are not universally qualified controls.
Measure WT NUDT14 binding/function for **both** compounds under interpretable conditions; do not
infer NUDT14 sparing from the NUDT5 mutant panel. No reciprocal-homolog mutation is proposed.

**Counterfactuals.** A precise near-zero contrast under a pilot-justified equivalence rule, or a
resolved negative contrast, challenges the proposed greater TH5427 dependence. Equal penalties
for both ligands are not differential recognition. A binding effect that disappears with an
orthogonal method points toward assay/surface dependence. Unfolding, altered oligomerization,
loss of retained function, unresolvable binding or interference makes the mechanism inconclusive,
not falsified or supported. Even positive contrasts for qualified proteins would establish only
comparator-specific differential dependence; they cannot localize a causal atom, prove a hydrogen
bond or predict a new ligand's selectivity.

## 3. Attribution is a later, separately gated question

Only after biochemical qualification, ask whether an additional compound-9 cellular effect at
**matched measured NUDT5 engagement** depends on NUDT14. Require independently validated NUDT14
perturbation and WT rescue, target/protein-abundance measurements, independent perturbation
reagents, reporter-interference controls, and assessment of BTK/other off-target explanations.
A separately counterscreened BTK comparator is not automatically selective. Loss of the extra
effect on NUDT14 depletion and return on rescue would support attribution in that context;
persistence after adequate depletion challenges it. Double depletion and floor-effect checks
help distinguish residual target effects from other targets/artifacts. Equal nominal doses,
NanoBRET EC50 and CETSA shifts are not interchangeable occupancy measurements. This gate does
not authorize cell work or establish a breast-cancer benefit.

## 4. Finite unresolved prerequisites and stop decisions

| Gate | Missing physical or design evidence | Decision if unresolved |
|---|---|---|
| G0 Materials | Lab owner/capacity; authentic compound lots, purity, salts, concentration and solubility; verified WT/R51A/R51K/NUDT14 constructs and independent-preparation identities | Do not interpret target-specific comparisons. |
| G1 Protein/function | Folding, oligomerization, stability, retained function and active fractions for every protein | Stop affected mutant inference; never replace a failed mutant with a favorable story. |
| G2 Readouts | Qualified direct/orthogonal binding ranges and models; functional/interference controls; NUDT14 comparison; local density review before any atom-directed redesign | If binding cannot be resolved, report feasibility failure. Do not manufacture KD or design new chemistry from incomplete geometry. |
| G3 Precision/governance | Independent pilot data, meaningful effect/equivalence margins, independent unit, sample size/precision, multiplicity/stopping rules; custodian/blind IDs, randomization, exclusions and unblinding plan | No confirmatory acquisition or completed prospective-lock claim. Freeze a separate dated plan only after qualification, before validation outcomes. |
| G4 Cellular attribution, conditional | Qualified engagement window, NUDT14 depletion/rescue, abundance, phenotype dynamic range and off-target controls | Remain at the purified-protein question; no cellular causal claim. |

[The existing blinded input contract](../assay/PROTOCOL.md) and
[unresolved handoff requirements](../assay/handoff_requirements.json) apply only to normalized
NUDT5 ADPr-hydrolysis inputs. `nudt5-assay` is not qualified direct-binding/KD, NUDT14, ATP or
cellular analysis software. Separate endpoint contracts and real qualification records are
required; no measurements or prospective lock are invented here. Known compounds remain known
controls even if remeasured. The immediate useful outcome is a resolved or explicitly inconclusive
H2 comparison, not a claim of inhibitor discovery.
