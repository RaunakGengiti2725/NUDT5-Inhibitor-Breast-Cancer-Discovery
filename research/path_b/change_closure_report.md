# Path B implementation and closure report

Current-status note: this records an earlier Path B build, not the stress-repair commit. Historical commands and manifests remain unmodified. The current candidate is BMC Research Notes, not an approved submission.

## Verdict and scope

The rewritten manuscript is an author-review-only, descriptive structural pharmacology reanalysis. It gives an all-site, atom-specific account of deposited compound-9 geometry and reports paired catalytic measurements within their source protocols. The model-support inspection qualifies a chain-specific reading of the published Arg51 account; it neither refutes the source authors nor tests energetic dependence. Distance recalculation alone does not establish a sufficient scientific contribution for a particular journal. No biological experiment, prospective validation or new scientific model-fitting run was performed for this rewrite. Software tests exercise code, including fitting on test fixtures; they are not scientific validation.

The fixed baseline is `40b9b0708d888a015abe5043bb273c3c6ee601ae`. Work is isolated on `devin/1791243628-path-b-manuscript`. Upstream structure commit `b69494166d68ff5a1ad71052a201dcc14c777b6c` was cherry-picked as `58235064340e19a87ad248d11742135a5c1a2582`; source/provenance commit `fd16ba5712102d7ee8c5827ebae172d24cefe539` became `eb13c8a0755991723562a2fe3284d55a77298da6`. No integration conflict occurred. The delivered release manifest identifies the final implementation commit without a self-referential hash in this file. No PR, merge, force push, main/existing-PR-branch edit, submission, deposit change, purchase or external contact was made.

## Changes linked to evidence

| Change | Evidence and implementation | Boundary retained |
|---|---|---|
| Structure-first title, abstract, argument and results | research/manuscript.md; structure_comparison/results/observed_proximity.json; model_support/results/model_support.json | Four sites and both chains retained; copies are nonindependent; no geometry-to-affinity/energy/causality/homology/selectivity inference. |
| Chain-specific Arg51 account | Full atom witnesses and wwPDB report extraction: AAA backbone N, 3.756 Å; BBB side-chain CD, 3.251 Å, occupancy 0.78; AAA CZ occupancy zero | Partial and null/refused rows are not absence of contact. NUDT14 residue 51 is SER; Leu107 is target-specific numbering. |
| All-site report table | Four exact W0O report mappings, RSCC 0.928–0.952, RSR 0.076–0.097; 30/30 reported ligand atoms per site | Report metrics and fixed precomputed-map slices are not independent density or placement validation. |
| All eight paired catalytic rows | research/results/selectivity.json and primary-source endpoint ledger; Tables 2 and table_paired.csv | Training-overlap compounds 10/11 are excluded only from the score challenge. SDs, censoring, 20/60-minute protocols, shared 500 nM TH5427 normalization and unresolved replication wording remain. No KD ratio, ratio CI or generalized selectivity claim. |
| Same-split diagnostic controls | controls.json / evaluations / full_valid_set; exact_scaffold, seed 42; stored indices, folds and score paths; diagnostic_control_sources.json | All 16 displayed methods use the same 45-record, 19-positive partition. Seven single descriptors include TPSA. Descriptor LR, nearest neighbours, constant/prevalence, component models and raw four-score consensus are compared only descriptively. Unverified labels/unassayed unmatched decoys cannot establish target recognition. |
| Screening demoted | Diagnostic supplement S1–S6 plus generated numerical annex, ten supplementary figures, raw result JSON and historical manifests | Different cohorts, fitting regimes and endpoints are not silently pooled; no new fitting, library/decoy reconstruction, docking or MD. Compound 9 remains an exposed retrospective counterexample. |
| Source identity and bibliography corrected | source_checks/source_claims.json, source_register.json, bibliography_checks.json and identity_summary.json | NC5-02/ACT-19/compound 11 is known chemistry. Original reference graphs do not authenticate TH5427/TH1713. Nguyen full-text/Y74E claims are cut; access failure does not disprove a publication. |
| Explicit document profile | scripts/build_research_documents.py --profile path-b; build_path_b_documents.py; frozen_inputs.json | Legacy default remains available, including original standalone figure behavior. Path B cannot opt out of trusted structural/paired evidence. |

Quantitative manuscript blocks are generated and checked against frozen inputs. The builder recomputes displayed AUCs from stored predictions, verifies index/fold alignment, reconstructs the trusted structural scientific fields, regenerates model support and rejects changed or self-consistent-but-untrusted evidence. Publication is staged into a fresh directory with a completion manifest written last. No malformed data are replaced by convenient numbers.

## Audit decisions and actual closure

The exact audited baseline prose, all 1,041 claim units and original decisions are preserved outside the rewritten manuscript in audit_archive/. The ledger has 925 survives, 112 needs qualification and four must-be-cut decisions. These are audit counts, not scientific discoveries. archive_manifest.json verifies the preserved payload subset and records the full supplied ZIP hash and location; it does not claim that this subset contains every upstream script or partition report.

The four whole-row cuts are implemented: C-0100's historical/current consensus equivalence, C-0167's original-six-method claim, C-0327's false denial of written decoy tolerances and C-0471's six-original-method comparison. The supplement explicitly preserves the historical stated strict tolerances while distinguishing missing executable matching provenance. It records the current raw four-score mean, implemented descriptive ECE and absence of trained fused-score calibration. Grouped conformal results do not establish shifted-population coverage. Local design files do not prove preregistration.

current_gaps.md maps every original material-gap ID to an actual route, state and consequence, merging duplicates. G-M2/G-M4 are out of scope, not requests for additional Path A or transfer-learning studies. G-S4 is partly bounded by added report/map inspection, not fully closed. Author declarations, measured-label/decoy provenance and laboratory gates remain open. The historical 18,412-compound library is not requested or reconstructed, and no missing artifact is treated as proof that no historical work occurred.

The exact supplied author request sheet is retained byte-for-byte. Confirmed Raunak Gengiti/name and correspondence are not requested again. Nikhil Srinivasan's authorship and retained contributions remain author-owned. The current laboratory specification preserves authenticated published chemical-source context, WT/R51A/R51K qualification, orthogonal direct binding, product-based function, NUDT14, interference/aggregation controls, pilot-based precision and supportive/challenging/inconclusive outcomes. Only qualitative cost drivers are given. The unsupported yield-frequency assertion was removed. The original specification remains archived. No experiment is authorized or claimed to have occurred.

## Fresh numerical reproduction

numerical_reproduction.json records source/fresh paths and exact hashes. Model-support, paired-evidence, identity-summary, original-audit and PubChem-audit outputs reproduced byte-for-byte. Fresh geometry matched every scientific field, retaining 1,730 residue-conformer rows (1,564 observed, 58 partial, 108 refused/null) and 1,278 atom pairs within 5 Å. Its runtime provenance differs as expected. Historical manifests and source data retain their original bytes and timestamps.

Executed reproduction routes were the deposited-coordinate CLI plus derived figure manifest, build_structure_model_support.py, nudt5-selectivity with frozen transfer predictions, regenerate_identity.py in identity-only mode, nudt5-audit audit and build_pubchem_audit.py. README contains the supported commands. Fresh outputs and execution manifests accompany the delivered bundle; they do not restamp historical runs. There was no new scientific model-fitting run.

## Verification and document inspection

The final functional revision passed:

- `.venv/bin/ruff check .`
- `.venv/bin/ruff format --check .`
- `.venv/bin/mypy --no-incremental` (strict configuration, 28 checked source files)
- `.venv/bin/python -m pytest tests research/path_b/source_checks` (689 passed, zero failed/skipped; 16 dependency deprecation warnings)
- `.venv/bin/python -m compileall -q scripts tests`
- `.venv/bin/python -m build` (sdist and wheel, outside Git)
- `.venv/bin/pip-audit` (no known vulnerabilities in audited dependencies; local non-PyPI package explicitly unaudited)
- `git diff --check` for authored tracked edits; full baseline comparison used `git -c core.whitespace=blank-at-eol,blank-at-eof,space-before-tab,cr-at-eol diff --check 40b9b0708d888a015abe5043bb273c3c6ee601ae HEAD` to recognize preserved CRLF CSV line endings.

The initial unstaged-only diff check did not inspect new archive files. The expanded default baseline-to-commit check flags 3,851 lines in nine exact-byte historical/upstream CSVs. Every flagged line was checked to contain only CR before LF, not trailing spaces/tabs. The CRLF-aware check passes with the remaining whitespace checks enabled. No source bytes, repository Git configuration or merge policy were changed to silence this. diff_verification.json records the affected paths.

Regression coverage includes frozen-source drift and forged-lock rejection, malformed/truncated structural data, both-chain/all-site retention, numeric-block/abstract drift, same-split score alignment, no-fitting guards, exact audit archive hashes, atomic late-render failure, separate main/supplement figures, supplementary caption numbering, Word table header/row controls and non-overlapping structural figure headers. No active pre-commit hook or configured hook system was found. Remote CI was not observed and is not claimed.

Headless LibreOffice and Poppler were used, without UI testing. The main PDF and DOCX-to-PDF render each have 11 pages; the supplement has 22 PDF pages and 20 DOCX-rendered pages. Page raster/contact-sheet inspection and enlarged critical-page inspection found and repaired initial figure-header overlap, orphan captions and cramped Word columns. Final figures and full captions share pages; all text is within page bounds. Unchanged pages were pixel-compared across render iterations. Main Figures 1–3 are structural, with 3A/3B on separate pages; supplementary Figures S1–S10 stay in the supplement. Small diagram details also have separate vector/raster exports and full-precision CSVs. This is document QA, not journal-specific specification certification or independent expert review.

## Remaining limits and decisions

- Author affiliation, funding, COI, CRediT/contributions, authorship, all contributor approvals, rights, acknowledgments, applicable approvals, prior versions/submissions, reviewer conflicts and full AI-assistance disclosure remain unresolved. No final declarations, cover letter or reviewer list exists.
- The original label/decoy ledger and ACT-18 authoritative structure remain unresolved. Finite graph identity is neither exhaustive chemical novelty nor physical-lot identity.
- No expert 3-D map review, omit/polder map, rerefinement, independent ligand-placement validation, coordinate-error propagation, inferential crystal-copy statistics or energetic test was performed. EDIAm/OPIA definitions were subsequently retrieved and are now interpreted only as model-support summaries; the map-generation version remains unrecorded.
- Catalytic IC50, direct KD, engagement and viability remain distinct. Unequal protocols, common normalization and unresolved replicate hierarchy/covariance prevent ratio uncertainty or general selectivity inference.
- Qualified laboratory leadership/materials, pilot variance, independent-unit design, prospective lock, budget, schedule and execution approval remain open. The proposed laboratory panel may yield inconclusive results.
- Nguyen full text remains inaccessible under bounded checks. The historical JMGM guide failure is retained in source records; current BMC Research Notes guidance and limits are documented in research/submission/. General topical scope and readability do not establish publication fit, sufficient incremental contribution or submission readiness.

A verified environment setup was proposed through approval-gated Devin settings for future sessions. Acceptance was not observed; present verification does not depend on it. The review package is the deliverable, not an authorization to submit or publish.
