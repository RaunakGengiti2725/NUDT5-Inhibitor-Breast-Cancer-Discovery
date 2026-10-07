# Path B source and identity handoff

Current venue: BMC Research Notes candidate; see `research/submission/checklist.md`. JMGM access checks below are historical, not the current venue gate. No author or editorial clearance follows.

Checked 5 October 2026 UTC against immutable baseline `40b9b0708d888a015abe5043bb273c3c6ee601ae`. This is a scoped evidence handoff, not a manuscript rewrite, human peer review, journal-fit certification or biological validation. All branch changes are under this directory. No training, new cohort, screening, library/decoy rebuild, docking, MD, experiment, external contact, deposit change or PR is part of this work.

## Writer instructions first

1. Make the deposited-coordinate, all-four-site W0O reanalysis the core, with both chains, alternates, occupancy, partial/null residues and atom witnesses preserved. Do not turn residue proximity into binding energy, affinity, causality, homology, selectivity or a refutation of source experiments. Crystal copies are not independent biological replicates. This source check did not rerun geometry or inspect electron density.
2. State **NC5-02 = ACT-19 = published Balikci compound 11**, not a discovery. Only NC5-02 overlaps the finite comparison pools described below. PubChem is a **tautomer-parent match**, not an exact canonical-SMILES match. The other nine candidates have no match in these inputs; that does not establish novelty.
3. Separate authenticated TH5427, TH1713, compound 9 and compound 11. The original ACT-01/ACT-02 reference graphs are not their authenticated references. Preserve original CSV bytes. Do not reconstruct invalid ACT-18.
4. Keep catalytic IC50, direct-binding KD, cell engagement and viability as separate endpoints. Keep source SDs/censoring and the unequal 20/60-minute protocols beside any paired catalytic comparison. No ratio CI, covariance, Ki conversion or generalized selectivity is justified.
5. Nguyen supports **abstract-level non-enzymatic/scaffolding context only** here. Cut Y74E, figure-level and reproduced-full-text-inspection claims. One official full-text attempt returned 403; this does not disprove the publication.
6. Update the Marques current citation to its **2026 published article**, not only the 2025 preprint. Qian's 2024 and Page's 2019 corrections concern funding, not changed efficacy/structures. Preserve Qian's reported xenograft deaths without assigning their cause.
7. Use the completed consolidated audit for the rewrite. Preserve its ECE correction, raw-current versus normalized-historical consensus distinction, and correction of the false denial of historical decoy tolerances. Do not reopen read-only reviews or characterize their unit counts as discoveries.
8. Do not claim JMGM compliance. Its current official scope and publisher AI policy were inspected; the journal-specific author guide was inaccessible. Do not finalize authorship, declarations, a cover letter or submission materials from this handoff.

## 1. Original-input provenance and identity definition

Both repository files exactly match the parent-confirmed original user-attachment SHA256 values:

| Input | SHA256 |
|---|---|
| `compounds.csv` | `d628a0fa1926c1e44c9b2a79adf92b444959e80153cf9019ef5e56b4747c3576` |
| `final_hits.csv` | `38c8d5266ce127b1fd5babde0c895190d0b382e3c9e3dae7581e320766745100` |

The parent holds those attachments; their availability is **not a missing scientific datum**. This session compared repository bytes to the supplied hashes, rather than claiming independent access to those two attachments. The original audit was regenerated in scratch and is byte-identical to `research/results/audit.json` (SHA256 `f62ffa50147b3d8ba6d73fd141172b3ffc728fbadda5b2889d6c634ad0403654`). Historical run manifests were not rewritten.

[The identity summary](identity_summary.json) reuses `pipeline.candidate_audit`, `transfer.identity`, `transfer.identity_matches`, `transfer.substitute_references` and the existing offline PubChem provenance validator. Exact identity means RDKit canonical **isomeric** SMILES; parent and canonical-parent-tautomer comparisons are reported separately. Morgan radius 2 / 2048-bit similarity without chirality remains a descriptive comparison, not activity. Standard InChIKey agreement does not by itself imply equality under the stricter canonical-SMILES policy.

There are 46 original training rows, 45 parseable unique graphs, with ACT-18 invalid; all ten candidates parse and have no duplicate exact graphs. Original labels remain unauthenticated source labels versus unmatched, unassayed presumed negatives. Identity matches do not validate all source activity labels.

### All ten candidates, actual bounded overlap

`E/P/T` means exact, neutral-fragment-parent and canonical-parent-tautomer match. A dash means no match under any of those policies. Comparison inventories are 45 original training graphs, 23 Balikci supplement rows, 50 archived database rows, 16 archived PubChem CIDs, 8 measured-ledger rows and 3 authenticated CCD graphs. Rows/records are not independent experiments.

| Candidate | Original training | Balikci supplement | Archived ChEMBL/BindingDB | Archived PubChem | Measured ledger / authenticated CCD |
|---|---|---|---|---|---|
| NC5-01 | — | — | — | — | — |
| NC5-02 | ACT-19 (E/P/T) | compound 11 (E/P/T) | two records (E/P/T) | CID 22346757 (**T only**) | — |
| NC5-03 | — | — | — | — | — |
| NC5-04 | — | — | — | — | — |
| NC5-05 | — | — | — | — | — |
| NC5-06 | — | — | — | — | — |
| NC5-07 | — | — | — | — | — |
| NC5-08 | — | — | — | — | — |
| NC5-09 | — | — | — | — | — |
| NC5-10 | — | — | — | — | — |

NC5-02 links to ChEMBL molecule **CHEMBL1242204**, activity **25858862**, and archived BindingDB molecule **50636309**, record **Q9UKK9-response-row-7**. PubChem CID **22346757** has the same standard InChIKey but a different source tautomer: neither exact nor neutral-parent SMILES matches, while canonical-parent-tautomer identity does. The offline PubChem audit reproduced 26 linked activity rows over 16 CIDs and 21 assay descriptions; these are index/linkage counts, not independent new measurements. Its concise API omits relation symbols; retain censoring from the linked ledger. PubChem AID 2070371 protocol metadata conflicts with the primary Balikci method; record linkage does not resolve that protocol disagreement. No live PubChem search or new evaluation row was added.

Separately, the 23-row source-to-training overlap is **compound 10 = ACT-20** and **compound 11 = ACT-19**. Reference substitution, performed only as an identity sensitivity check, additionally matches **compound 6 = authenticated TH5427**. No model was fitted or rescored for that substitution.

### Authenticated graph/accession distinction

| Identity | Source/accession | Formula; heavy atoms | Standard InChIKey |
|---|---|---|---|
| TH5427 | Page 28; Balikci 6; CCD 9CH; [5NWH](https://www.rcsb.org/structure/5NWH) | C20H20Cl2N8O3; 33 | QXCXMVYVUHVFLP-UHFFFAOYSA-N |
| TH1713 | Page 2; CCD 958; [5NQR](https://www.rcsb.org/structure/5NQR) | C19H21N7O3; 29 | NHRNLJPTVPQENT-UHFFFAOYSA-N |
| Compound 9 | Balikci 9; CCD W0O; [8RIY](https://www.rcsb.org/structure/8RIY), [8OTV](https://www.rcsb.org/structure/8OTV) | C23H24N6O; 30 | KHLKLLMMPVSSQY-UHFFFAOYSA-N |
| NC5-02 / ACT-19 / compound 11 | Original CSV graphs and Balikci supplement compound 11 | C17H13N5O; 23 | YYVUOZULIDAKRN-UHFFFAOYSA-N |

Full source and canonical SMILES, decoded CCD descriptors, input hashes and accession links are in the machine-readable summary; none were invented. **958 lists the synonym `TH5427`, but its graph/formula and Page's 5NQR mapping identify TH1713.** Do not merge 958 with 9CH by that synonym. Compound 11 is not W0O and no deposited ligand accession is assigned to it here. NC5-01 similarity to the authenticated TH5427 graph is 32/83, approximately 0.386, under the stated fingerprint; the historical mislabeled reference is a different graph.

## 2. Retained-source and bibliographic checks

[Source register](source_register.json), [claim ledger](source_claims.json), and [bibliographic corrections](bibliography_checks.json) retain URLs, exact XML/metadata locators, excerpts, access outcomes and hashes. Fresh primary XML archives retain their original CC BY 4.0 permission blocks; gzip storage changes no decoded article bytes. Balikci uses the immutable repository archive. Article text/caption reading does not imply new figure-image, density or laboratory verification.

- **Page et al., Nature Communications 9, 250 (2018), [10.1038/s41467-017-02293-7](https://doi.org/10.1038/s41467-017-02293-7).** Source compound mappings and synthesis route were checked. Its 29 nM TH5427 and 70 nM TH1713 malachite-green IC50 values are assay-specific; the reported 2.1 µM intact-cell ITDRF-CETSA apparent EC50 is a different endpoint. [2019 correction](https://doi.org/10.1038/s41467-019-12806-1) adds ERC 695376 (T.H.) to acknowledgments. Do not import the source authors' funding into this manuscript.
- **Balikci et al., [10.1021/acs.jmedchem.4c00072](https://doi.org/10.1021/acs.jmedchem.4c00072).** Compound 9 is already published against both targets, with their deposited structures. Author-supplement cells give NUDT5 **0.270 ± 0.027 µM** and NUDT14 **0.162 ± 0.005 µM** catalytic IC50. Their mean ratio, NUDT14/NUDT5, is **0.600**, not a measured KD ratio, kinetic constant or generalized selectivity metric. Section 4.3 specifies AMP-Glo, 1 nM enzyme, 10 µM ADPr, 1% DMSO, **20 minutes versus 1 hour**, and common 500 nM TH5427 zero normalization. It says both “triplicate sets” and mean ± SD of “two independent biological replicates”; do not infer a hierarchy or replicate covariance from those phrases. Table 1 NA means IC50 >50 µM. Source SPR captions separately report approximately 250 and 400 nM KD for compound 9; do not exchange those with catalytic IC50.
- **Qian et al., Breast Cancer Research 26, 23 (2024), [10.1186/s13058-024-01778-w](https://doi.org/10.1186/s13058-024-01778-w).** Retain its report of four TH5427-treated xenograft mice found dead after day 7; cause is not established by this source check. [Correction 26, 53](https://doi.org/10.1186/s13058-024-01814-9) concerns incomplete funding information, not a claimed correction of that result.
- **Nguyen et al., Science 390(6778), 1143–1150 (2025), [10.1126/science.adv4257](https://doi.org/10.1126/science.adv4257).** Europe PMC metadata/abstract were accessible; first online date 6 November differs from December issue metadata. The sole allowed [official full-text attempt](https://pmc.ncbi.nlm.nih.gov/articles/PMC7618541/) returned HTTP 403. Abstract supports a non-enzymatic/scaffolding contribution involving PPAT and purine synthesis; no Y74E or figure inspection is authenticated here.
- **Marques et al., Nature Communications 17, 8192 (2026), [10.1038/s41467-026-74489-9](https://doi.org/10.1038/s41467-026-74489-9).** The published title is *Targeted Protein Degradation of NUDT5 Dissociates Catalytic Inhibition from Protein Loss in 6-Thioguanine Response*. This supersedes use of only the differently titled [2025 preprint](https://doi.org/10.1101/2025.03.16.643557) for current biological context; retain that preprint when documenting version history. Perturbation-specific non-enzymatic context does not establish breast-cancer efficacy or equivalence of inhibition and depletion. Crossref's preprint relation field is empty; a formal metadata relation is not claimed.

This is a bounded check of retained sources and corrections, not a systematic literature search, exhaustive correction/retraction search, or verification of every primary experiment in database records.

## 3. Published-source routes for the conditional laboratory specification

**TH5427:** Page's Methods explicitly directs chemical synthesis to its Supplementary Methods. The [official linked Supplementary Information PDF](https://media.springernature.com/original/springer-static/esm/art%3A10.1038%2Fs41467-017-02293-7/MediaObjects/41467_2017_2293_MOESM1_ESM.pdf) was downloaded successfully, SHA256 `490c6658a4089c122d0e69897c6ea5ebb9809fb350f2c72b7d16e784b76e1ea9`. It is source material, not a generated deliverable; the 29 MB file is retained in scratch, not committed. Internal PDF procedure pages were not text-inspected (no PDF text extractor available), so no page number, step-level reproduction or batch availability is certified. Main-text compound 28/TH5427 and 5NWH identity were independently read. Page also gives a reasonable-request materials/protocol route; that is not a promise of current supply.

**Compound 9:** Balikci section **4.1.2**, linked to Scheme 1/precursor chemistry, directly supplies the synthesis and characterization of 1-(1-methylpiperidin-4-yl)-3-(4-phenoxyphenyl)-1H-pyrazolo[3,4-d]pyrimidin-4-amine. The source reports 69% isolated yield, 96.3% HPLC purity, NMR and MS; these are source measurements, not this project's measurements. A qualified laboratory can evaluate this published route under separate authorization. No transfer agreement, current author stock or commercial supply has been established.

Both are **unmatched known comparators**, not a matched molecular pair or newly discovered candidates. Do not populate vendor/catalogue/stock/price/lot fields without actual supply evidence. No vendor or laboratory was contacted and nothing was purchased. Any future lot requires graph/salt/solvate and concentration authentication, identity and purity checks, solubility/aggregation assessment, and the protein/assay qualification gates already specified in the conditional OWNER-LAB document. Mutant feasibility, independent-preparation variance, orthogonal binding, sample size, cost and timetable remain laboratory-owned and unmeasured.

## 4. Historical JMGM check, not current BMC certification

The [official Elsevier journal scope](https://shop.elsevier.com/journals/journal-of-molecular-graphics-and-modelling/1093-3263) includes computational investigations of molecular structure, function and interactions. It emphasizes reproducibility and machine-readable supplementary data, and explicitly excludes routine applications with little new insight. **Inference, not editorial assurance:** Path B is broadly topical, but its incremental contribution over the published structures/interpretation is a serious unresolved fit risk. Tests and a complete provenance package do not settle novelty or acceptance.

The [journal-specific author guide](https://www.sciencedirect.com/journal/journal-of-molecular-graphics-and-modelling/publish/guide-for-authors) returned 403. [Journal checks](journal_checks.json) list the exact unchecked items: eligible article category; word/abstract/title/keyword/highlight/reference limits; section structure; title-page/anonymization and declaration forms; mandatory data/code/repository/license requirements; figure format/resolution/dimensions; graphical abstract; permission forms; submission portal fields; reviewer/exclusion requirements; prior-publication/exclusivity disclosures; fees and publication options. **No numerical limit, mandatory section or checklist compliance is certified from memory.**

The [current general Elsevier AI policy](https://www.elsevier.com/about/policies-and-standards/generative-ai-policies-for-journals) was accessible. It requires disclosure of substantive assistance, including tool, purpose and oversight; research-process use belongs in Methods and AI is not an author. The current policy distinguishes explanatory images, data visualizations and primary research images: do not repeat an outdated blanket claim that every AI-assisted figure is forbidden. Data visualizations must derive faithfully from underlying data through reproducible reported methods, with tool/version/developer disclosure. AI-assisted explanatory images require caption and general disclosure. Do not use AI to create or alter purported primary observed images not directly obtained in the research. Formal AI-based research methods are not prohibited by that policy, but must be described reproducibly in Methods. General-purpose generative-AI image tools are prohibited for graphical abstracts; AI-generated cover art requires prior editor/publisher permission. Author approval, complete assistance history and JMGM-specific placement remain unresolved; no final AI declaration is supplied.

## 5. Unresolved author, laboratory and access gates

Raunak Gengiti and correspondence `gengitir@gmail.com` are already confirmed; do not ask again. The supplied author-request sheet remains authoritative. Resolve affiliation/address; funding and material support; financial/nonfinancial COI; **Nikhil Srinivasan's retained qualifying contributions, authorship and consent**; other contributions/CRediT/approval/accountability; acknowledgments and permission; prior versions/deposits/submissions/exclusivity; original ACT label/decoy provenance; rights/licenses/approvals; private reviewer conflicts; and full author-approved AI-assistance history. Do not assume sole authorship or “none” declarations. The original CSV attachment availability is not among the gaps, and the unavailable historical 18,412 library must not be reconstructed or requested.

Laboratory access is optional for the narrow retrospective Path B paper. No qualified experimental lead, supply lot, variant qualification, pilot variance, independent sample design or execution authorization has been established here. These prevent experimental claims, not the present provenance handoff. Nguyen full text and the JMGM author guide remain access gaps; PDF procedure-level inspection is separately bounded above. Missing artifacts/access do not establish that no historical work or publication exists.

## 6. Reproduction and artifact map

Final verification: **40 relevant tests passed**; Ruff, formatting, strict mypy, three deterministic reproductions and the scratch package build passed. The full training/benchmark suite was intentionally not run under the no-new-fitting scope.

- [`regenerate_identity.py`](regenerate_identity.py): offline identity-only driver; imports existing identity/provenance functions.
- [`identity_summary.json`](identity_summary.json): every candidate/source graph, all E/P/T match sets, original nearest-neighbor audit, named CCD metadata, hashes and reproduced PubChem linkage.
- [`source_register.json`](source_register.json), [`source_claims.json`](source_claims.json): retrievals and exact source witnesses.
- [`bibliography_checks.json`](bibliography_checks.json), [`journal_checks.json`](journal_checks.json): bounded bibliographic repairs and checked/unchecked publisher requirements.
- [`audit_context.json`](audit_context.json): supplied audit hashes and reconciled constraints, attributing parent verification accurately.
- [`verification.json`](verification.json) and [`commands_run.md`](commands_run.md): exact local validation commands/results and intentional nonexecutions. Tests verify implementation/provenance only.

Reproduce in the repository's hash-locked Python 3.12 environment with Gemmi from `requirements-structure.lock`. Use an existing scratch parent and a **new output filename**:

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=scripts/scripts:scripts \
  /path/to/locked-venv/bin/python research/path_b/source_checks/regenerate_identity.py \
  --repository "$PWD" --output /existing/scratch/new-identity-summary.json
cmp /existing/scratch/new-identity-summary.json research/path_b/source_checks/identity_summary.json
PYTHONDONTWRITEBYTECODE=1 /path/to/locked-venv/bin/python -m pytest \
  -p no:cacheprovider -q research/path_b/source_checks
```

Expected RDKit diagnostics include the invalid ACT-18 kekulization warning and undefined-stereochemistry warnings on source identity standardization. They are retained, not hidden or repaired with guessed structures.
