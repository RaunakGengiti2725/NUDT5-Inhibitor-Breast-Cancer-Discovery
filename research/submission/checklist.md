# BMC Research Notes: single-journal preparation checklist

## Decision and scope

Target: **BMC Research Notes, Research note**. This is a descriptive extension of a published structural analysis with additional computational controls. The official article-type criteria include these categories and do not require predicted impact. They explicitly exclude pooled analyses of selected publications, systematic reviews and meta-analyses. This package uses a single primary compound-9 source for catalytic arithmetic and does not pool studies. That distinction must remain in the manuscript. The editor still determines distinct contribution, scientific validity, originality and eligibility. Scope is not an acceptance promise. The earlier JMGM primary-research rejection risk is not represented as resolved by additional biological evidence.

Official guidance checked 6 October 2026:

- Research-note criteria/sections/limits: https://link.springer.com/journal/13104/submission-guidelines/research-note
- General formatting, supporting files, cover letter and reviewer identity: https://link.springer.com/journal/13104/submission-guidelines
- Publisher policies: https://link.springer.com/brands/bmc/editorial-policies
- Fee information is recorded in the archived official guide. Recheck at submission and acceptance.

Exact fetched URLs (including redirects), capture dates and hashes are in sources/manifest.json. The current AI policy requires disclosure and accountable human judgment. AI-generated critiques are internal assistance, not journal peer review.

| Requirement | Implementation / status | Route |
| --- | --- | --- |
| One venue and appropriate article category | Research Note; no simultaneous submission, no external contact. Editorial eligibility remains an editor's decision. | CLOSED-BY-WORK for preparation |
| Objective/Results abstract, ≤200 words | Build-time count and section check. No citations or projections in abstract. | CLOSED-BY-WORK |
| Introduction + Main text + Limitations, ≤2,000 words | Build-time conservative count includes table titles/legends (normally excluded); table cells counted separately. | CLOSED-BY-WORK |
| 3–10 keywords | Six fixed keywords, automatically checked. | CLOSED-BY-WORK |
| No more than three figures/tables | Three editable main tables, no main figures. Tables remain visible with simple baselines and unfavourable model-support values. | CLOSED-BY-WORK |
| Editable manuscript, double spacing, line/page numbers, no manual page breaks | DOCX with continuous line numbering, PAGE footer, double-spaced text and table paragraphs; proof PDF is for reading, not upload as main manuscript. | CLOSED-BY-WORK |
| Table titles ≤15 words; legends ≤300 words | Three short titles/legends, checked at build. Repeating headers and non-splitting rows; no colour coding. | CLOSED-BY-WORK |
| Supporting files in citation order, ≤20 MB each | Additional file 1: supplementary methods/diagnostics PDF. Additional file 2: code/data/results ZIP. Additional file 3: map/structure-factor ZIP. Build enforces size and records every payload hash. | CLOSED-BY-WORK |
| Additional tables/figures preserved | Source-generated supplementary document and original SVG/PDF/CSV numerical artifacts in Additional file 2. No image compression substitutes for source values. | CLOSED-BY-WORK |
| Main quantitative tables track recorded inputs | Existing Path B source locks and quantitative-block checks retained. No refit or new benchmark. | CLOSED-BY-WORK |
| Source references and bounded claims | Single-source endpoints, model dependencies, four ligand sites, atom identities, censoring and limitations retained. See claim_traceability.md. | CLOSED-BY-WORK |
| Unrecovered library and unsupported discovery claims | Removed as current results; historical records retained with withdrawal metadata. | CLOSED-BY-CUT |
| Every required declaration heading | Intentionally absent from the author-review draft until confirmed text exists; generated only from completed author statements. Draft must not be uploaded as a finished submission. | OWNER-USER |
| Affiliation/byline/author approval | Proposed corresponding author is known; retained contributor history remains unresolved. | OWNER-USER |
| Competing interests, funding, CRediT, acknowledgements and approvals | Exact requested fields in author_actions.md; no assumed “none” or “not applicable”. | OWNER-USER |
| Prior deposits, exclusive submission, source provenance and reuse permissions | Must be documented; code license and public archive permission are not invented. | OWNER-USER |
| AI disclosure and responsible human review | Observed AI work includes analyses, code, interpretation assistance and drafting, not only spelling. Full scope and human accountability require confirmation. | OWNER-USER |
| Cover letter | A 250-word scientific core is prepared; required approval, competing-interest and prior-publication statements are appended only after author confirmation. | OWNER-USER for final declarations |
| Optional reviewer suggestions | Four identities verified through institutions; no outreach. Private conflicts not certified. Omit suggestions if uncleared. | OWNER-USER |
| Fees/licensing and author-only submission | Author reviews the current terms and submits personally if they choose. No upload, payment or acceptance implied. | OWNER-USER |
| New mechanistic/binding/selectivity claim | Requires qualified experiments in the existing conditional lab specification; not a conclusion of this note. | OWNER-LAB |

## Upload order after author release

1. BMC_research_note.docx as the main manuscript. Do not upload the reading-proof PDF as its substitute.
2. Cover_letter.docx after required factual declarations have been supplied.
3. Additional_file_1.pdf: **Supplementary methods and diagnostic results**. Extended structural methods, retained diagnostics, maps, limitations and numerical annex.
4. Additional_file_2.zip: **Reproducibility code, source records and numerical outputs**. Versioned source tree, immutable historical inputs, source manifests and generated tables/figures. Historical manuscripts/audits inside the source archive are provenance records, not competing submissions or current efficacy claims.
5. Additional_file_3.zip: **Deposited map and structure-factor snapshots**. Restore into the same source tree using the documented root-relative paths. No raw biological replicates are implied.
6. Enter identity-verified reviewer suggestions only if conflict screening is complete. Do not upload internal author forms, policy captures, AI reviews or the checklist as manuscript supplements in isolation.

The portable archive also includes the reading proof, author action sheet and verification manifest for the author's use. Those internal files are not submission uploads. Rights review applies to the source archives as a whole, including retained historical and third-party records.
