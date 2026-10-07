# Adversarial structural-specialty AI review

**AI review, not human peer review.** Read-only assessment of the exact rewritten Path B revision.

**Recommendation: REJECT for the conditional Journal of Molecular Graphics and Modelling (JMGM) primary-research target.**

**Single most likely paper-killing objection: AI-S01, insufficient incremental scientific insight beyond Balikci.** The independently checked distances are not the problem. The paper adds a careful description of already deposited models but does not establish a new structural mechanism, biological consequence or demonstrated methodological advance. Fixing presentation or declarations would not by itself justify changing this verdict to major revision.

## 1. Exact revision and review scope

- Repository: https://github.com/RaunakGengiti2725/NUDT5-Inhibitor-Breast-Cancer-Discovery
- Branch fetched: `devin/1791243628-path-b-manuscript`.
- Exact detached commit: [22830642f7d33ca3319872b5ba4c513baff50154](https://github.com/RaunakGengiti2725/NUDT5-Inhibitor-Breast-Cancer-Discovery/commit/22830642f7d33ca3319872b5ba4c513baff50154).
- Immutable starting revision: `40b9b0708d888a015abe5043bb273c3c6ee601ae`, verified as an ancestor.
- Supplied ZIP SHA256: `63a8c1f1fde68b51f0544129dda997ad3a7618b0c1043428b7a1206aed91f517`.
- All 136 release inventory entries matched recorded hashes/sizes; all 294 tracked files matched the supplied source archive. The checkout remained clean.

I inspected the manuscript and diagnostic supplement, generated DOCX text and supplied rendering sheets, structural source/code/data, both proximity figures, both Arg51 slice figures and all four W0O slice figures. I independently recomputed decisive quantities from raw mmCIF records and separately reproduced the existing model-support builder outside the repository. No tracked file was edited; no commit, push, PR, child session, workflow, external contact or submission was made.

References below identify **lines in the exact committed Markdown/code**, not invented PDF page numbers. JSON selectors identify data records. The companion JSON contains immutable source links and the same numbered objections.

## 2. What independently survived hostile checking

### All four ligand sites and every retained atom/site identity

| Entry / target | W0O label chain / author chain / author residue | Positive-occupancy heavy atoms | Report RSCC / RSR |
|---|---|---:|---|
| 8RIY / NUDT5 | C / AAA / 301 | 30 | 0.931 / 0.094 |
| 8RIY / NUDT5 | D / BBB / 301 | 30 | 0.940 / 0.091 |
| 8OTV / NUDT14 | C / A / 301 | 30 | 0.952 / 0.076 |
| 8OTV / NUDT14 | F / B / 302 | 30 | 0.928 / 0.097 |

All ligand heavy-atom name/element sets match archived W0O CCD exactly. All four sites have occupancy 1 and no ligand alternate label. All four archived CCD SMILES descriptors canonicalize to the source compound-9 graph under the pinned RDKit environment. This is graph/identifier agreement, not authentication of a physical lot, protonation state or purity.

Independent raw-category traversal with Gemmi 0.7.3 and NumPy, without importing the repository geometry engine, reproduced **1,730 residue-conformer rows: 1,564 observed, 58 partial and 108 refused/null**. Row identities, minima, statuses, witness pairs and all four radius flags agree. All **1,278 pairs within 5 Å** match, with maximum pair-distance difference **0.0 Å**. These are descriptive records, not independent sample sizes. Both deposited assembly-1 operators are identity operations; both receptor chains were retained for every site.

8RIY has no nonblank alternate labels. Its positive fractional Arg51 atoms remain eligible, while AAA CZ at occupancy zero is excluded. In 8OTV the local LEU author B47 / label B48 A/B alternatives remain separate; their four site-specific minima are all beyond 5 Å. No A/B averaging or fictitious global conformer pairing was found in the actual extraction. Null rows remain null, and partial minima are not complete-residue minima. A missing atom does not become a no-contact observation. B factors are metadata, not a propagated coordinate error.

### Decisive Arg51 functional-group minima

The following are **retained-coordinate minima**, not energetic contributions. Backbone = N, CA, C, O, OXT if present. Side chain = CB, CG, CD, NE, CZ, NH1, NH2 if present and positive-occupancy. Guanidinium subset = NE, CZ, NH1, NH2 under the same retention rule; it is a subset of side chain, not an independent group.

| 8RIY local environment | Backbone witness / Å | Side-chain witness / Å | Retained guanidinium witness / Å |
|---|---|---|---|
| W0O C/AAA301 – receptor A/AAA51 | N–C20 / 3.755561 | CD–C18 / 4.001089 | NE–C18 / 4.761740 |
| W0O D/BBB301 – receptor B/BBB51 | N–C20 / 3.946537 | CD–C18 / 3.251273 | NE–C18 / 4.606398 |

Witness occupancies are 1 except BBB CD = 0.78. AAA's guanidinium subset is incomplete because CZ occupancy is zero. The opposite-chain Arg51 residue minima are approximately 14.329 Å and 14.296 Å, not the quoted local witnesses. Six decimal places here expose a cutoff issue, **not experimental precision**.

| Site | Rows within 3.5 Å | 4.0 Å | 4.5 Å | 5.0 Å |
|---|---:|---:|---:|---:|
| 8RIY C/AAA301 | 7 | 9 | 14 | 15 |
| 8RIY D/BBB301 | 8 | 12 | 15 | 16 |
| 8OTV C/A301 | 4 | 11 | 13 | 15 |
| 8OTV F/B302 | 4 | 8 | 11 | 12 |

Counts include both receptor chains. No alternate rows qualify within these radii; the AAA Arg51 qualifying row at 4.0–5.0 Å is partial. These counts are not a target-comparison statistic or independent replication.

Different numbering matters. In 8OTV, author Leu107 is label residue 108. The nearest Leu107 witness for W0O C/A301 is **receptor B Leu107 CD2–ligand C15, 3.609412 Å**; for W0O F/B302 it is **receptor A Leu107 CG–ligand C17, 3.780542 Å**. Thus even the ligand's author-chain letter cannot be used to assume the closest receptor chain. The other Leu107 copies are approximately 17.270 and 18.593 Å away. No R51↔L107 sequence/structural homology mapping is established or needed for these coordinate facts.

### What the map checks establish, and what they do not

The repository model-support builder produced 12 files outside the repository, all byte-identical to the committed outputs. A second calculation using explicit periodic trilinear weights, not the repository sampling function, checked **636 positive-occupancy atom positions in each of the two map types** across the entries. The maximum absolute difference from archived standardized values is below **2.55 × 10⁻⁷** full-cell SD units, consistent with numerical precision. Full-cell means and SDs agree.

That confirms consistent interpolation and provenance, **not independent crystallographic validation**. RSCC/RSR were read from archived official reports, not recalculated. Maps are precomputed and model-dependent; fixed slices and atom samples cannot establish affinity, energy, causality or correct occupancy. I did not perform expert 3D review, omit/polder maps or rerefinement. The figures correctly disclose their limited scope, but have the specific visibility weakness in AI-S05.

For context, independent positive/negative pair counting reproduces all 14 stored full-valid-set method AUCs and the seven single-descriptor AUCs, and verifies Equal_mean's four-component arithmetic for all 45 records. Compound 9's reported-mean ratio is exactly 0.600. No models were fitted. These checks do not authenticate the labels, decoys or biological meaning of the scores.

## 3. Numbered objections

Every objection has exactly one requested evidence status. **Existing evidence** means the required factual material already exists. **Specified new analysis** names work not performed here. **Nothing currently available** means that the reviewed record cannot answer the objection; it does not claim that the answer is impossible or that missing historical work never occurred. Some objections are openly acknowledged limitations rather than undisclosed errors. Their honesty does not remove their effect on significance.

### 1. The added granularity does not establish enough incremental scientific insight for JMGM (AI-S01)

**Severity:** critical. **Type:** publication-significance risk, not a demonstrated numerical error.

**Evidence status: nothing currently available.**

**Exact locations:**
- [research/manuscript.md, lines 8-17](https://github.com/RaunakGengiti2725/NUDT5-Inhibitor-Breast-Cancer-Discovery/blob/22830642f7d33ca3319872b5ba4c513baff50154/research/manuscript.md#L8-L17)
- [research/manuscript.md, lines 124-144](https://github.com/RaunakGengiti2725/NUDT5-Inhibitor-Breast-Cancer-Discovery/blob/22830642f7d33ca3319872b5ba4c513baff50154/research/manuscript.md#L124-L144)
- [research/path_b/source_checks/sources/jmgm_scope_excerpts.txt, lines 1-15](https://github.com/RaunakGengiti2725/NUDT5-Inhibitor-Breast-Cancer-Discovery/blob/22830642f7d33ca3319872b5ba4c513baff50154/research/path_b/source_checks/sources/jmgm_scope_excerpts.txt#L1-L15)
- [research/structure_comparison/sources/PMC11089510.xml; fig[@id='fig3']/caption; Results paragraph discussing Figure 4D](https://github.com/RaunakGengiti2725/NUDT5-Inhibitor-Breast-Cancer-Discovery/blob/22830642f7d33ca3319872b5ba4c513baff50154/research/structure_comparison/sources/PMC11089510.xml)

**Excerpt:** “The contribution is descriptive and methodological.”

Balikci already reported compound 9, both target complexes, paired catalytic results and the chain-B hydrophobic Arg51 description. The new result is a reproducible census of deposited coordinates, missingness and report/map metadata. Showing that the closest atom is N in one copy and CD in another refines that account but does not establish a different binding mechanism, a new biological consequence or a demonstrated new analysis method. The manuscript itself says it neither tests nor refutes the energetic interpretation. The conjunction of accurate limitations, archived sources and many checks is valuable audit work; it is not itself the scientific advance required by a primary molecular-modelling research paper.

**Evidence:** The source Figure 3 caption explicitly describes a hydrophobic R51 contact in chain B for compound 9. The source discussion of Figure 4D already contrasts the compound-9 R51 and L107 environments and separately speculates about TH5427 selectivity. Independently reproduced nearest-atom values are compatible with this context. The official JMGM scope explicitly excludes routine standard modelling with only very limited new scientific insight.

**Smallest defensible response:** Identify a substantive, evidence-supported scientific question that this reanalysis actually resolves, beyond more complete reporting of known coordinates. Do not manufacture a refutation or new method. No such result is demonstrated in the reviewed package; prose revision alone cannot supply it. A tightly framed reproducibility/critical-commentary output is a different publication proposition, not an established JMGM article category or a promised acceptance route.

**Venue consequence:** Reject the present revision as a JMGM primary research article. Even complete repair of every presentation and administrative issue below would not on its own change this recommendation. A major-revision verdict would imply a credible incremental route to the missing contribution that this package does not presently establish.

### 2. Separate the compound-9 hydrophobic account from the TH5427 hydrogen-bond/selectivity conjecture (AI-S02)

**Severity:** major. **Type:** source-attribution imprecision and argument framing; not a fabricated source contradiction.

**Evidence status: existing evidence.**

**Exact locations:**
- [research/manuscript.md, lines 15-17](https://github.com/RaunakGengiti2725/NUDT5-Inhibitor-Breast-Cancer-Discovery/blob/22830642f7d33ca3319872b5ba4c513baff50154/research/manuscript.md#L15-L17)
- [research/manuscript.md, lines 59-61](https://github.com/RaunakGengiti2725/NUDT5-Inhibitor-Breast-Cancer-Discovery/blob/22830642f7d33ca3319872b5ba4c513baff50154/research/manuscript.md#L59-L61)
- [research/manuscript.md, lines 124-128](https://github.com/RaunakGengiti2725/NUDT5-Inhibitor-Breast-Cancer-Discovery/blob/22830642f7d33ca3319872b5ba4c513baff50154/research/manuscript.md#L124-L128)
- [research/structure_comparison/sources/PMC11089510.xml; fig[@id='fig3']/caption; Results paragraph beginning 'To explore the potential of 9'](https://github.com/RaunakGengiti2725/NUDT5-Inhibitor-Breast-Cancer-Discovery/blob/22830642f7d33ca3319872b5ba4c513baff50154/research/structure_comparison/sources/PMC11089510.xml)

**Excerpt:** “They do not test the energetic rationale proposed for compound 9 [2].”

This wording is too unspecific about which source proposition is being qualified. The inspected source describes a hydrophobic compound-9/R51 contact, and its explicit subsequent speculation about R51-mediated H-bonding/selectivity concerns ADP-ribose/TH5427, not a requirement that compound 9 contact a guanidinium atom in every copy. A carbon witness is compatible with the stated hydrophobic account. The revision correctly says it does not contradict the source; that restraint must govern the premise and prominence of the result, not merely its disclaimer.

**Evidence:** Source Figure 3: “An additional hydrophobic interaction with R51 in chain B”. In the following NUDT14 comparison paragraph, the source describes the phenoxy group engaging the hydrophobic side chain of R51, then discusses H-bond interactions shown for ADP-ribose and TH5427, ending with a proposed explanation for TH5427 selectivity. These are distinct propositions; the review does not allege that the source claimed a guanidinium-mediated compound-9 interaction.

**Smallest defensible response:** Name the exact source proposition. For example: Balikci depicts a hydrophobic compound-9/R51 contact and separately proposes R51-mediated H-bonding as one explanation for TH5427 selectivity; this study tests neither. Do not use the non-guanidinium minimum to imply a mechanistic correction of the first or a test of the second.

**Venue consequence:** Repairable by source-accurate wording. The repair makes the lack of mechanistic novelty clearer rather than curing AI-S01.

### 3. A residue minimum and a non-guanidinium witness are not a functional-group exclusion test (AI-S03)

**Severity:** major. **Type:** incomplete scientific presentation of a correct calculation; fixed-threshold interpretation risk.

**Evidence status: existing evidence.**

**Exact locations:**
- [research/manuscript.md, lines 23-25](https://github.com/RaunakGengiti2725/NUDT5-Inhibitor-Breast-Cancer-Discovery/blob/22830642f7d33ca3319872b5ba4c513baff50154/research/manuscript.md#L23-L25)
- [research/manuscript.md, lines 54-61](https://github.com/RaunakGengiti2725/NUDT5-Inhibitor-Breast-Cancer-Discovery/blob/22830642f7d33ca3319872b5ba4c513baff50154/research/manuscript.md#L54-L61)
- [research/structure_comparison/geometry_contract.json, lines 19-33](https://github.com/RaunakGengiti2725/NUDT5-Inhibitor-Breast-Cancer-Discovery/blob/22830642f7d33ca3319872b5ba4c513baff50154/research/structure_comparison/geometry_contract.json#L19-L33)
- [research/structure_comparison/results/observed_proximity.json; /atom_pairs_within_5A; /residue_proximity, 8RIY Arg51](https://github.com/RaunakGengiti2725/NUDT5-Inhibitor-Breast-Cancer-Discovery/blob/22830642f7d33ca3319872b5ba4c513baff50154/research/structure_comparison/results/observed_proximity.json)
- [research/structure_comparison/results/derived/radius_sensitivity.csv](https://github.com/RaunakGengiti2725/NUDT5-Inhibitor-Breast-Cancer-Discovery/blob/22830642f7d33ca3319872b5ba4c513baff50154/research/structure_comparison/results/derived/radius_sensitivity.csv)

**Excerpt:** “Neither is a guanidinium atom.”

The sentence is true, but the selection of one minimum conceals the closest competitors. In AAA the side-chain CD–C18 separation is 4.001089 Å, only 0.001089 Å beyond the primary 4.0 Å convention. The retained guanidinium-group NE–C18 minima are 4.761740 Å in AAA and 4.606398 Å in BBB: outside 4.5 Å but inside 5 Å. BBB also has a backbone N–C20 pair at 3.946537 Å. Thus nearest-atom identity is not synonymous with the only nearby atom, the absence of a side-chain environment, or functional-group noninvolvement. Existing sensitivity output is useful but this central example should be explained next to the claim.

**Evidence:** Independent raw-CIF calculation gives AAA backbone/side-chain/retained guanidinium minima 3.755561/4.001089/4.761740 Å and BBB 3.946537/3.251273/4.606398 Å. AAA CZ has occupancy zero and is excluded; its missing contribution prevents a complete-group conclusion. At 3.5 Å only the BBB residue minimum qualifies; at 4.0, 4.5 and 5.0 Å both qualify. These are exact deposited-coordinate conventions, not uncertainty or energy thresholds.

**Smallest defensible response:** Add the small atom-group table and explicit radius classifications supplied in this review, with group definitions, witnesses and occupancies. Preserve positive fractional atoms without inventing occupancy weights. State the radii as retrospective descriptive choices; do not treat the last decimal around 4.0 Å as a chemically resolved boundary or claim prospective preregistration.

**Venue consequence:** Repairable from already available coordinates and pair tables, without a new experiment or changed structure. It improves transparency but supplies no affinity, energy or new inhibitor claim.

### 4. Precomputed maps and whole-ligand report scores cannot establish the significance of the local copy difference (AI-S04)

**Severity:** major. **Type:** acknowledged model-support limitation and conditional evidence gap, not an incorrect distance or mandatory validation for as-deposited reporting.

**Evidence status: specified new analysis.**

**Exact locations:**
- [research/manuscript.md, lines 29-31](https://github.com/RaunakGengiti2725/NUDT5-Inhibitor-Breast-Cancer-Discovery/blob/22830642f7d33ca3319872b5ba4c513baff50154/research/manuscript.md#L29-L31)
- [research/manuscript.md, lines 59-71](https://github.com/RaunakGengiti2725/NUDT5-Inhibitor-Breast-Cancer-Discovery/blob/22830642f7d33ca3319872b5ba4c513baff50154/research/manuscript.md#L59-L71)
- [research/manuscript.md, lines 124-134](https://github.com/RaunakGengiti2725/NUDT5-Inhibitor-Breast-Cancer-Discovery/blob/22830642f7d33ca3319872b5ba4c513baff50154/research/manuscript.md#L124-L134)
- [research/path_b/diagnostic_supplement.md, lines 61-61](https://github.com/RaunakGengiti2725/NUDT5-Inhibitor-Breast-Cancer-Discovery/blob/22830642f7d33ca3319872b5ba4c513baff50154/research/path_b/diagnostic_supplement.md#L61-L61)
- [research/structure_comparison/model_support/results/model_support.json; /local_residue_reports; /sites; /structures; /policy](https://github.com/RaunakGengiti2725/NUDT5-Inhibitor-Breast-Cancer-Discovery/blob/22830642f7d33ca3319872b5ba4c513baff50154/research/structure_comparison/model_support/results/model_support.json)
- [scripts/build_structure_model_support.py, lines 268-323](https://github.com/RaunakGengiti2725/NUDT5-Inhibitor-Breast-Cancer-Discovery/blob/22830642f7d33ca3319872b5ba4c513baff50154/scripts/build_structure_model_support.py#L268-L323)

**Excerpt:** “No rerefinement, expert three-dimensional map review or independent ligand-placement validation was performed.”

This limitation is honestly disclosed and remains consequential. Ligand RSCC 0.928–0.952 is not an atom-specific validation of Arg51 N, CD or the guanidinium group. The maps are model-dependent; sampling a precomputed map at a model coordinate is not an independent test of that coordinate. Different report/map software dates, unknown map-generation version and non-equivalent map sampling across the entries prevent treating their standardized values as a calibrated cross-target support scale. Reproduction of the processing does not remove those dependencies.

**Evidence:** The archived 8RIY AAA Arg51 record has RSCC 0.894, RSR 0.103 and 15 listed outlier records (8 clash, 3 bond, 3 angle, 1 plane); BBB has RSCC 0.901, RSR 0.143 and no listed local Arg51 outlier. These are report records, not independent failures, and do not prove BBB correct or AAA incorrectly placed. All 12 regenerated model-support files match byte-for-byte; manual trilinear interpolation agrees within 2.55e-7 full-cell SD units. These checks verify implementation, not local crystallographic correctness.

**Smallest defensible response:** If the paper seeks stronger structural significance than as-deposited bookkeeping, perform and document expert full-3D assessment of all four ligand sites and both Arg51 environments using the deposited diffraction data, reports and surrounding model. Examine the relevant atoms jointly with the ligand, inspect appropriate maps, and where justified assess alternate/restraint/occupancy choices or explicitly versioned omit-map/rerefinement analyses. Report whether the descriptive contrast survives those justified choices. No such work was done in this review; even it would not establish interaction energy. It is not necessary merely to report the present coordinates with the existing limits.

**Venue consequence:** Without that analysis, the work must remain bounded to the deposited model. This is not a newly discovered hidden overclaim: the manuscript already accepts the restriction. Removing the restriction rhetorically would be unacceptable, and retaining it leaves the significance problem unresolved.

### 5. The highlighted zero-occupancy CZ is invisible in every central Figure 3A slice (AI-S05)

**Severity:** minor. **Type:** reproducible figure-communication defect, not evidence of a wrong model.

**Evidence status: existing evidence.**

**Exact locations:**
- [research/manuscript.md, lines 59-59](https://github.com/RaunakGengiti2725/NUDT5-Inhibitor-Breast-Cancer-Discovery/blob/22830642f7d33ca3319872b5ba4c513baff50154/research/manuscript.md#L59-L59)
- [scripts/build_structure_model_support.py, lines 625-681](https://github.com/RaunakGengiti2725/NUDT5-Inhibitor-Breast-Cancer-Discovery/blob/22830642f7d33ca3319872b5ba4c513baff50154/scripts/build_structure_model_support.py#L625-L681)
- [scripts/build_path_b_documents.py, lines 303-310](https://github.com/RaunakGengiti2725/NUDT5-Inhibitor-Breast-Cancer-Discovery/blob/22830642f7d33ca3319872b5ba4c513baff50154/scripts/build_path_b_documents.py#L303-L310)
- [research/structure_comparison/model_support/results/8RIY_A_AAA_51_slices.png; Figure 3A; all three panels](https://github.com/RaunakGengiti2725/NUDT5-Inhibitor-Breast-Cancer-Discovery/blob/22830642f7d33ca3319872b5ba4c513baff50154/research/structure_comparison/model_support/results/8RIY_A_AAA_51_slices.png)

**Excerpt:** “Figures 3A–B expose the fixed map slices for these local contexts”

The generated caption explains that zero-occupancy atoms would use red crosses, but the zero-occupancy AAA CZ falls outside all three 0.75 Å plotting slabs and no red CZ marker appears. Its absolute offsets are 0.9374 Å from the XY plane, 2.3610 Å from XZ and 1.8034 Å from YZ. Also, a projected marker can be as far as 0.75 Å off the sampled plane, so the contour directly under it need not be the density at that atom. The central visual evidence therefore cannot show the particular excluded atom emphasized in the abstract and Results.

**Evidence:** The filter is explicit at build_structure_model_support.py lines 657–659. The reviewed image and independent centroid calculation agree. This is not suppression of raw data: coordinates, sampling tables and the exclusion remain available. No absence of density is inferred from absence of the marker.

**Smallest defensible response:** Annotate that CZ is omitted by the slab filter, or add an explicitly labelled atom-centred view with ligand context and a small atom/report table. State any off-plane projection. Retain the warning that no rendered zero-occupancy coordinate enters distance calculations and that these maps do not validate it independently.

**Venue consequence:** A minor presentation repair from existing evidence. It cannot remedy AI-S01 or turn slices into independent validation.

### 6. The catalytic ratio cannot connect the structural description to a cross-target energetic result (AI-S06)

**Severity:** major. **Type:** acknowledged inferential ceiling and publication-significance risk, not an error in the reported ratio.

**Evidence status: nothing currently available.**

**Exact locations:**
- [research/manuscript.md, lines 35-37](https://github.com/RaunakGengiti2725/NUDT5-Inhibitor-Breast-Cancer-Discovery/blob/22830642f7d33ca3319872b5ba4c513baff50154/research/manuscript.md#L35-L37)
- [research/manuscript.md, lines 76-93](https://github.com/RaunakGengiti2725/NUDT5-Inhibitor-Breast-Cancer-Discovery/blob/22830642f7d33ca3319872b5ba4c513baff50154/research/manuscript.md#L76-L93)
- [research/manuscript.md, lines 126-136](https://github.com/RaunakGengiti2725/NUDT5-Inhibitor-Breast-Cancer-Discovery/blob/22830642f7d33ca3319872b5ba4c513baff50154/research/manuscript.md#L126-L136)
- [research/structure_comparison/sources/PMC11089510.xml; NUDT5/NUDT14 Activity Assay Methods paragraph; tbl1](https://github.com/RaunakGengiti2725/NUDT5-Inhibitor-Breast-Cancer-Discovery/blob/22830642f7d33ca3319872b5ba4c513baff50154/research/structure_comparison/sources/PMC11089510.xml)
- [research/selectivity/paired_evidence.json; /rows/8/endpoints; /rows/8/ratio](https://github.com/RaunakGengiti2725/NUDT5-Inhibitor-Breast-Cancer-Discovery/blob/22830642f7d33ca3319872b5ba4c513baff50154/research/selectivity/paired_evidence.json)

**Excerpt:** “No replicate covariance, ratio confidence interval or inferential ratio comparison is available.”

The arithmetic 0.162/0.270 = 0.600 is correct. It is not an independent confirmation of a structural hypothesis. The source describes 20-minute versus 1-hour reactions and a shared 500 nM TH5427 zero anchor; the package lacks raw paired replicate information and a fully resolved nesting/covariance structure. These limitations neither prove that the original experiments failed nor establish a general selectivity magnitude. The counterexample to interpreting a source-label score as selectivity is narrowly legitimate, but it does not supply the missing structural-to-function result.

**Evidence:** The source assay paragraph explicitly supplies both durations and the normalization anchors. The reviewed ledger preserves the two reported means, their endpoint SDs, two reported biological replicates and unavailable paired raw replicates. Catalytic IC50 is not KD; the source also reports direct-binding experiments that must not be conflated with this ratio.

**Smallest defensible response:** Keep the endpoint-specific contextual claim and no ratio CI. A stronger inference would require authenticated raw data and design/normalization information, and an appropriately comparable or independently qualified analysis/measurement design. These materials are not present; a conditional laboratory plan is not their substitute. Do not invent covariance, convert endpoint SDs into a selectivity interval, assume control failure, or perform an unauthorized experiment.

**Venue consequence:** No presently available calculation repairs the missing structural-to-functional bridge. This does not invalidate the descriptive table, but it cannot rescue JMGM significance. Treat the ratios as context or remove claims that require a stronger interpretation.

### 7. The extensive legacy-score story does not establish a structural modelling contribution (AI-S07)

**Severity:** minor. **Type:** editorial coherence and scope concern, not a demonstrated calibration/AUC error.

**Evidence status: existing evidence.**

**Exact locations:**
- [research/manuscript.md, lines 39-45](https://github.com/RaunakGengiti2725/NUDT5-Inhibitor-Breast-Cancer-Discovery/blob/22830642f7d33ca3319872b5ba4c513baff50154/research/manuscript.md#L39-L45)
- [research/manuscript.md, lines 93-120](https://github.com/RaunakGengiti2725/NUDT5-Inhibitor-Breast-Cancer-Discovery/blob/22830642f7d33ca3319872b5ba4c513baff50154/research/manuscript.md#L93-L120)
- [research/path_b/diagnostic_supplement.md, lines 17-43](https://github.com/RaunakGengiti2725/NUDT5-Inhibitor-Breast-Cancer-Discovery/blob/22830642f7d33ca3319872b5ba4c513baff50154/research/path_b/diagnostic_supplement.md#L17-L43)
- [research/path_b/diagnostic_supplement.md, lines 55-67](https://github.com/RaunakGengiti2725/NUDT5-Inhibitor-Breast-Cancer-Discovery/blob/22830642f7d33ca3319872b5ba4c513baff50154/research/path_b/diagnostic_supplement.md#L55-L67)

**Excerpt:** “Source-label controls do not establish target recognition”

The revision correctly bounds the scores to unauthenticated source labels versus unmatched unassayed presumed negatives. That correction should not be counted as biological evidence or as the novelty missing from the structural paper. The long diagnostic story makes this read partly as a forensic correction of a previous screening submission rather than a coherent new structural investigation. ECE is implemented; its presence is not trained fused-score calibration. Conformal outputs are empirical and do not guarantee shifted-population coverage.

**Evidence:** Main Section 2.4 and supplement Sections S2–S3 explicitly distinguish the raw four-score Equal_mean from Property_LR, historical normalized consensus and descriptive ECE. The supplement also explicitly withdraws missing-library claims and corrects the historical written decoy-tolerance denial. These are appropriate corrections, not new experimental validation.

**Smallest defensible response:** Retain the full audit and score evidence as provenance, but reduce the main-text detour to the necessary correction and why a selectivity reading is unwarranted. Make any retained structural manuscript stand on its structural result. Do not rebuild the historical library or replace the decoys to manufacture a new benchmark.

**Venue consequence:** Repairable by reorganizing existing material. A cleaner paper is not necessarily a sufficiently novel paper; AI-S01 remains.

### 8. The author-owned byline and declarations are unresolved (AI-S08)

**Severity:** critical administrative. **Type:** submission-readiness blocker, separate from scientific merit.

**Evidence status: nothing currently available.**

**Exact locations:**
- [research/manuscript.md, lines 3-3](https://github.com/RaunakGengiti2725/NUDT5-Inhibitor-Breast-Cancer-Discovery/blob/22830642f7d33ca3319872b5ba4c513baff50154/research/manuscript.md#L3-L3)
- [research/manuscript.md, lines 140-148](https://github.com/RaunakGengiti2725/NUDT5-Inhibitor-Breast-Cancer-Discovery/blob/22830642f7d33ca3319872b5ba4c513baff50154/research/manuscript.md#L140-L148)
- [research/path_b/diagnostic_supplement.md, lines 63-63](https://github.com/RaunakGengiti2725/NUDT5-Inhibitor-Breast-Cancer-Discovery/blob/22830642f7d33ca3319872b5ba4c513baff50154/research/path_b/diagnostic_supplement.md#L63-L63)
- [research/path_b/author_requests.md, lines 3-25](https://github.com/RaunakGengiti2725/NUDT5-Inhibitor-Breast-Cancer-Discovery/blob/22830642f7d33ca3319872b5ba4c513baff50154/research/path_b/author_requests.md#L3-L25)

**Excerpt:** “The final author list, retained contributions, affiliations and declarations require author confirmation.”

The package explicitly remains author-review-only. Retained qualifying contributions and authorship involving Nikhil Srinivasan, affiliations, funding, COI, approvals/accountability, rights, acknowledgments, prior versions/submissions, reviewer conflicts and complete AI disclosure cannot be certified from the supplied materials. This is not an allegation of misconduct or an invitation to infer declarations from public profiles. The author-guide access limitation also prevents certification of journal-specific compliance.

**Evidence:** The manuscript and exact author request sheet identify the unresolved items and expressly prohibit assumed sole authorship or blank/assumed declarations. Raunak Gengiti and the correspondence email are already confirmed and are not being requested again.

**Smallest defensible response:** Obtain the author-owned factual records and approvals through the existing request sheet; preserve unknowns as unknowns. Resolve the historical coauthor’s qualifying contribution explicitly. Verify the current journal-specific instructions through authorized access before any separately authorized submission. No declaration was drafted or supplied by this review.

**Venue consequence:** Do not submit this revision, regardless of scientific recommendation. Author confirmation can remove the administrative block but cannot remove AI-S01. Nothing in the presently supplied record settles these matters.

## 4. Error versus significance verdict

**No numerical error was found in the checked geometry, identity mapping, interpolation or frozen-score arithmetic.** I do not convert acknowledged limits into fictional hidden claims. AI-S02 is a source-attribution precision problem; AI-S05 is a directly reproduced figure-communication problem. AI-S03 concerns incomplete presentation of the decisive atom-group/cutoff context. AI-S01 is the scientific rejection reason. AI-S04 and AI-S06 explain why the added evidence cannot currently elevate the result. AI-S07 is an editorial scope concern. AI-S08 is a separate administrative block.

The strongest features to retain are the atom-resolved witnesses, explicit occupancy/missingness, inclusion of all four sites and both receptor chains, refusal to turn crystal copies into statistical n, separate target numbering, explicit distinction between catalytic IC50 and other endpoints, and the sentence that the source account is **qualified rather than contradicted**. Do not repair the paper by deleting those safeguards or overstating a source dispute.

The argument currently moves from a valid distinction between description and mechanism, through a correct coordinate/report inventory, to already bounded published endpoints and extensive legacy-score diagnostics, then back to the same distinction. What is missing is a substantive intervening scientific finding. This is why a cleaned-up version can remain unsuitable as JMGM primary research.

## 5. Consequences and repair order

1. Decide the publication proposition before further polish. If it remains a JMGM primary research article on the current evidence, my recommendation remains reject. Reproducibility is necessary but not sufficient under the inspected scope statement.
2. Use existing evidence to correct the source proposition, expose the functional-group/radius table, repair the CZ visualization and streamline the legacy-score material. These repairs improve accuracy and clarity; they do not create scientific novelty.
3. Only if separately authorized and scientifically justified, pursue the specified local-model analysis. It may assess robustness of a structural description; it still cannot measure energetic dependence. The conditional laboratory plan is not completed evidence and no experiment is authorized by this review.
4. Resolve author-owned declarations before any separately authorized submission. Do not silently remove a historical coauthor, invent a no-conflict statement or assume a journal permits the current disclosure/byline.

## 6. Verification trail and review limits

The accompanying evidence ZIP includes independent scripts, numerical JSON/CSV outputs, source excerpts, revision/package verification and byte-comparison results. All scratch calculations and report generation were outside the checkout. The three independent-check scripts pass Ruff formatting/lint and default mypy checks; these are software checks only. Core executed commands were:

```text
venv/bin/python recompute.py
venv/bin/python check_support.py
venv/bin/python check_context.py
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=scripts/scripts venv/bin/python   scripts/build_structure_model_support.py --output <new external output directory>
venv/bin/ruff check --no-cache <three independent-check scripts>
venv/bin/ruff format --check <three independent-check scripts>
venv/bin/mypy --no-incremental <three independent-check scripts>
git rev-parse HEAD
git status --porcelain
git merge-base --is-ancestor 40b9b0708d888a015abe5043bb273c3c6ee601ae HEAD
```

The scripts use the pinned repository lockfiles in an external Python environment. Their default review paths are explicit and can be adjusted when rerunning elsewhere. Independent geometry means a second implementation of the calculation over the same deposited data, not an independent structure determination. Shared Gemmi/NumPy dependencies remain a limitation of that cross-check.

The package's reported 689-test run was inspected as provenance, **not independently rerun or treated as biological validation**. I did not independently validate every diagnostic implementation, original training history, publication in the literature, source experiment or document layout. Nguyen full text remains uncertified; lack of access does not contradict it. The journal-specific author guide and author declarations remain unresolved. The official scope was inspected; a conditional recommendation is not a prediction of what an editor will decide.

**Final verdict:** the reviewed revision is substantially more careful than a discovery/target-recognition manuscript, and its principal checked calculations reproduce. On the currently supplied evidence, however, **reject for insufficient incremental scientific contribution to the conditional JMGM target**. This is an AI review, not human peer review.
