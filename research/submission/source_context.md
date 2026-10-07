# Current source-context and access qualifications

AI repair note, not human peer review, biological validation or permission clearance.

## Validation-field attribution

The frozen Table 1 values are from wwPDB XML reports: 8RIY creation date 20 March 2026 and 8OTV 8 March 2026; validation pipeline 2.49, density-fitness 1.0.12 and percentiles 20250101.v01. Exact URLs, hashes and extraction keys remain in `research/structure_comparison/model_support/source_manifest.json` and the corresponding results. These XML fields are not claimed to appear in PDF summaries, and no density metric is independently recomputed here.

The consolidated R46 audit inspected dictionary v0.071 and reports that the legacy `pdbx_vrpt_model_instance_density` container is deprecated in favour of `pdbx_vrpt_model_instance_map_fitting`. That container change does not alter the frozen XML values or establish what future regenerated reports will contain. The original dictionary item pages remain locators for the definitions used. OPIA retains its operational percentage definition; an inconsistent upstream explanatory sentence is not copied as a claim.

Meyder et al., DOI https://doi.org/10.1021/acs.jcim.7b00391, is the EDIA method attribution. This repair rechecked its indexed abstract/metadata, not its publisher full text. Candidate references to benchmark bias, contact analysis and local validation are bounded prior-art positioning, not claims of global novelty, complete anticipation or patent clearance. Stachowski/Fischer 2026 was inspected at abstract level only. Rosenberg et al., https://doi.org/10.1038/s41597-024-03595-4, was re-inspected in full via Europe PMC (PMC11255211): the Contact calculation section describes all-altloc atom contacts below 5 Å, residue minima, exclusion of hydrogen/water, and an asymmetric-unit scope. It is explicitly cited as close prior art, not a claim of complete anticipation; no PDB-wide extension was run.

## Retrieval and historical annotation boundaries

Original retrieval timestamps for the retained 958 and 9CH snapshots are not recoverable from the inspected records; they remain unknown, not backfilled with this repair date. Their original bytes and hashes are retained. The file named `nguyen_fulltext.html.gz` is a recorded HTTP 403 access-attempt body, not an inspected full text. The source register already records that failure.

Historical descriptive names are not graph authorities. The supplied audit records these additional name/graph mismatches: DEC-05 “Benzoxazinone” has an N/N five-membered carbonyl heterocycle; DEC-08 “Quinazolinone” has a five-membered benzimidazolinone carbonyl ring; DEC-13 “Morpholine benzoate” has an amide; DEC-14 “Quinazoline aniline” has an additional aromatic N; NC5-08 “Lumazine” has a fused 5/6 rather than lumazine 6/6 ring system. Original names/SMILES and all CSV bytes are retained. No replacement identity, source-label truth or physical-sample identity is inferred. Central ACT-18 quarantine and ACT-19/NC5-02/compound-11 overlap remain unchanged.

## Conventions and provenance

Radius-set counts are ligand/residue-conformer rows, not unique residues or independent experiments. Alternate-conformer rows must not be pooled as independent samples. Positive fractional occupancy is included without weighting distances. Site-, residue- and witness-level fractional flags have different scopes; see the corrected model-support README.

Calibration keeps the emitted NumPy linspace floating-point boundaries: left-closed/right-open, with one included in the last bin. Exact decimal fifths are a different convention. No frozen ECE value was changed. Backend/thread/version provenance in each new submission manifest records the current environment only; historical backend dispatch remains unknown, and the existing RDKit compatibility outputs stay separately named.

The older release manifest and rendering snapshots describe earlier runs. Current scientific tables/figures are regenerated through their builders; old `research/figures` renderings are excluded from the current supporting archives. Archive checksums identify actual distributed bytes, not byte-for-byte regeneration of PDF/DOCX/SVG metadata.

The immutable selectivity provenance binds the original transfer code. That original source is preserved at `research/runtime_history/transfer-aa5febd.py.txt` with its original hash and size; the repaired live transfer source has a separate exact code-hash gate. Unknown live code or changed historical bytes still refuse. This is an explicit reviewed runtime transition, not a restamping of an old manifest.
