# Path B rewrite and execution brief

Baseline `40b9b0708d888a015abe5043bb273c3c6ee601ae`. Manuscript SHA256 `5ee7257be5a09c9862c2ead2f98416e1e8d1b2fa875624ca009df5cb95934bbb`. Scratch deliverable for a later parent-controlled rewrite stage. **No manuscript text, tracked file, deposit, branch or PR was changed, and no experiment, library, decoy, workflow or external contact was created.** This is an AI evidence audit, not human peer review, biological validation, journal-fit certification or any prediction of acceptance.

## 1. Audit-stage verdict

The completed deliverables are an audit handoff, not submission readiness. The upstream audits exhaustively assessed the immutable manuscript against code, data, deposited coordinates and cited sources, subject to their stated access limits. This consolidation independently checked every locator and quotation and selected decisive code/source evidence: 1041 audit units (874 exact inventory units plus 167 supplemental semantic and caption units), with 925 surviving as bounded statements, 112 requiring qualification and 4 whole rows requiring cuts. Of 38 material gaps, 27 remain PENDING.

What this means in practice: the existing evidence supports a careful, narrow methods-and-reanalysis paper. It does not support a discovery, target-recognition or selectivity-prediction paper, and no amount of rewriting can change that. The honest implemented method name is a **fixed equal-weight four-score consensus over Morgan-fingerprint classifiers with group-disjoint resampling diagnostics**. The code contains no TWCS, no TWCS++, no learned transferability, no min-max normalization of the fused components, no eight-component consensus and no target-weighted term.

## 2. The honest central claim

> The all-four-site analysis of deposited W0O complexes in 8RIY and 8OTV retains both receptor chains, occupancy and alternate-conformer limits. At the two 8RIY NUDT5 sites, the nearest retained nearby Arg51 atom differs: a backbone nitrogen at 3.756 A in one chain and a fractional-occupancy side-chain CD at 3.251 A in the other, with neither witness a guanidinium atom. Residue-level proximity therefore supplies no uniform guanidinium-interaction evidence, and this descriptive geometry establishes no binding energetics, no causal selectivity, no homology between the NUDT5 and NUDT14 residue axes and no refutation of the original authors' interpretation. The nearest-witness result neither excludes other retained pairs nor estimates distances to absent atoms.

Everything else in the paper is context for that claim or an explicit limitation of it. The screening diagnostics are valid diagnostic results but belong in a supplement, not as evidence of target recognition or as the primary Path B result.

## 3. Most severe outstanding Path B objection

A reviewer can reasonably hold that the incremental contribution is too small: the source publication already reported compound 9 binding both targets, deposited both structures and proposed the Arg51 rationale (Balikci et al., DOI 10.1021/acs.jmedchem.4c00072, Figures 3–4 and Sections 4.2/4.8–4.10). The existing reanalysis recalculates distances in those same deposited models and finds the nearest-atom identities are heterogeneous. This limits a reading that treats residue-level proximity as a uniform interaction; it does not correct or refute the source experiments. It is a descriptive refinement of an existing structural rationale, with no new measurement, no energetics, no density inspection (G-S4) and no experiment (G-D6). The defensible answer is honesty about scope plus the reusable auditable artifact: the fixed geometry contract, the full all-site ledger with missingness and occupancy handling, and the falsifiable experiment design. Whether that clears a specialist journal's novelty bar is a reviewer judgement I cannot make, and I will not predict it.

Second-order objections, in order: unresolved label provenance and declarations (G-D7, G-M8); inaccessible Nguyen full text behind a reproduced-inspection claim (G-A1); and the possibility that the prominent catalytic comparison R = 0.600 is dismissed as a ratio of reported means under unequal 20 and 60 minute protocols (G-P0-7).

## 4. Retain, demote, cut

**Retain as the core.** The all-site observed-coordinate analysis with its full ledger: 1,730 residue-conformer rows (1,564 observed, 58 partial, 108 null/refused), 1,278 retained pairs within 5.0 A, four sites, both deposited chains, model 1, assembly 1, primary 4.0 A radius with 3.5, 4.5 and 5.0 A sensitivity. Keep every interpretive prohibition from the geometry contract, the not_estimated status for complete-residue distances, and the statement that null rows are not no-contact.

**Retain as context.** Published paired catalytic endpoints with units, source SDs, censoring semantics, the unequal reaction times, the shared 500 nM TH5427 zero anchor and the unresolved replication wording. Source-level chemical identity findings, including NC5-02 = ACT-19 = published compound 11 and the mismatched reference graphs. Cited source biology with its own perturbation boundaries.

**Demote to a diagnostic supplement.** All AUC values, score rankings, the retrospective source-transfer challenge and property matching. If any performance figure stays in the main text, the single-descriptor, descriptor-LR, nearest-neighbour and constant/prevalence baselines must appear beside it, with D1 and D2 stated in the same place.

**Demote to explicitly proposed.** The Arg51 energetic rationale, the direct-binding comparison and any cellular attribution, each with its unresolved gates named.

**Cut.** The four whole-row cuts ['C-0100', 'C-0167', 'C-0327', 'C-0471'] and every fragment listed in the claim table: the version equivalence between current Equal_mean and the original normalized consensus; "original six methods" in both locations; the false denial of the historical decoy tolerances; the Y74E fragment; the unreproducible primary-author-manuscript inspection; and all claims dependent on the nonexistent 18,412-compound library.

**Hold for the author.** Every item in section 6. Sole authorship is not finalized.

## 5. Rewrite sequence for the later stage

1. Freeze a new manuscript hash before editing and reconcile against `unified_claim_table.csv` by claim ID.
2. Apply the four whole-row cuts and the listed fragment cuts first, since later sections cite them.
3. Rewrite the title, abstract and introduction around the section 2 central claim.
4. Restructure: structural reanalysis becomes the primary result; screening diagnostics move to a supplement; paired-target evidence becomes bounded context.
5. Rewrite Methods with the honest method name and the corrected ECE wording (diagnostic expected calibration error over five fixed equal-width bins is computed; no trained calibration of the fused score exists).
6. Carry every qualification next to the claim it qualifies, not into a distant limitations section.
7. Regenerate only authorized figures and tables into fresh empty output directories under a recorded revision; verify no stale artifact survives; keep historical run manifests historical.
8. Rerun lint, type checks and the full test suite, and state in the paper that they verify software behaviour only.
9. Obtain the section 6 declarations before any submission step.

Do not backfill calibration, a transfer term, missing experimental outputs or the absent library. Do not reconstruct or request the 18,412-compound library.

## 6. Author request list

These must come from you. I have not drafted, inferred or placeheld any of them, and I will not.

1. Affiliation for the author list, or the exact wording you want for independent-researcher status, plus the postal address a journal will require.
2. Funding: every grant, fellowship or institutional support that applies, or an explicit confirmation that none does.
3. Competing interests: financial and non-financial, including any relationship to NUDT5 or BTK inhibitor programmes, or an explicit confirmation of none.
4. Nikhil Srinivasan: his exact retained contribution to the current manuscript, dataset or code. The historical document lists him as a coauthor, which is not evidence that he contributed nothing. Decide authorship against the target journal's criteria, document the reasoning, and tell me whether he has been contacted and consents.
5. Any other contributor, including anyone who supplied data, structures or review, and their CRediT roles for every author.
6. Confirmation of who is corresponding author and the email to publish.
7. Complete prior-version history: every earlier manuscript version, preprint, deposit, dataset release and submission, with dates and current status, including the Zenodo and IEEE DataPort records and whether either has been submitted anywhere.
8. Whether any prior submission was rejected, withdrawn or is under review now, since simultaneous submission is prohibited.
9. Label provenance: the original assay records or traceable source evidence behind the positive activity labels, and the authoritative ACT-18 chemical structure if any record of it exists.
10. Decoy provenance: how the 26 decoys were generated, by what tool and version, and any record of per-decoy matching. If no record exists, say so and I will bound the claim.
11. Laboratory capability: whether you have access to any laboratory, an accountable experimental lead, and whether you want the conditional specification in section 7 pursued at all.
12. Reviewer conflicts: anyone who must be excluded, and anyone you wish to suggest.
13. Licensing and permissions: the licence for the code, data and manuscript, any reuse permission needed for figures or source tables, and any institutional or ethical approval that applies.
14. Acknowledgments text, and the exact AI-assistance disclosure wording you approve, covering auditing, coding, analysis and drafting assistance. Check the target journal's policy before finalising it.

## 7. Pointers

`owner_lab_experiment_specification.md` holds the conditional OWNER-LAB specification. `unsent_collaboration_inquiry.md` holds an UNSENT draft; it has not been sent and must not be sent without your instruction. `unified_claim_table.md` and `.csv` hold every claim disposition; `material_gap_register.md` and `.csv` hold the routed gaps; `reconciliation_log.md` records where this consolidation changed an upstream audit conclusion; `coverage_integrity_report.json` and `commands_run.md` record exact inputs, hashes and the commands actually run.
