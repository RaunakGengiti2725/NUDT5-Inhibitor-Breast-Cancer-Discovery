# Publication strategy and evidence gates

## Current verdict

The current project is **not ready for submission as a validated inhibitor-discovery paper**. A reproducibility-focused revision can truthfully report chemical-identity problems, benchmark artifacts and controls; it cannot supply missing discovery experiments. No venue acceptance, clinical benefit or novelty priority is guaranteed. A data/software DOI is not equivalent to peer-reviewed journal publication.

## Evidence-first routes

| Route | What the present revision supports | What must be demonstrated before a strong submission |
|---|---|---|
| Computational reproducibility case study | Transparent reconstruction of a small released dataset; tested controls; explicit limits | Establish relevance beyond this single demonstration, independent reproduction and a clear contribution beyond correcting implementation errors |
| NUDT5 chemical-probe research | Source-backed target rationale and explicit candidate hypotheses | Assay-level identity/potency provenance, chemically characterized material, orthogonal biochemical and binding evidence, selectivity and cellular target engagement; negative results retained |
| Broad ML/data-science method | A benchmark critique, not a new validated transfer algorithm | Multiple independently curated targets, leakage-resistant splits, strong property/similarity baselines, nested tuning, ablations, prospective or external evaluation, uncertainty and reusable data |
| Translational breast-cancer discovery | Motivation only; no new efficacy experiments in this work | Context-specific causal mechanism, relevant models, exposure/target engagement and tolerability evidence; approvals before any new animal/human work |

## Named venues: fit is conditional, not a prediction

### NeurIPS

NeurIPS 2026 main-track abstract and paper deadlines were May 4 and May 6; they have passed as of this audit (4 October 2026). Future-cycle dates and policies must be rechecked. The use-inspired contribution type is within the main track, not a shortcut around significance or originality. A mean of existing classifiers is not by itself a demonstrated methodological advance. This dataset is too small and provenance-limited to establish general transfer-learning claims.

Inspected official policy sources:
- https://neurips.cc/Conferences/2026/CallForPapers
- https://neurips.cc/Conferences/2026/MainTrackHandbook
- https://neurips.cc/Conferences/2026/ReviewerGuidelines
- https://neurips.cc/public/guides/PaperChecklist
- https://neurips.cc/public/EthicsGuidelines

Prepare the required checklist, anonymized submission artifacts, exact environments, statistical uncertainty, compute disclosure and licensing information. Important/nonstandard AI-agent contributions to methodology should be described accurately; this revision is not merely spell-checking. Public preprints can be permitted, but submission anonymity and current overlapping/dual-submission rules still apply. Do not submit the same paper indiscriminately to several archival venues.

### Cell Press Patterns

Potential fit would require a reusable data-science contribution with transparent, convincing validation, not just a NUDT5 screening application. Journal-specific author-policy pages returned HTTP 403 during the independent policy audit. Their indexed excerpts were inspected, but full current journal requirements, article type, format, fees and timing are **not certified here**. Confirm them before submission.

- https://www.cell.com/patterns/information-for-authors
- https://www.cell.com/patterns/information-for-authors/journal-policies
- https://www.elsevier.com/about/policies-and-standards/publishing-ethics
- https://www.elsevier.com/about/policies-and-standards/research-data
- https://www.elsevier.com/about/policies-and-standards/generative-ai-policies-for-journals

The inspected Elsevier parent policies require responsible human oversight, accurate reporting, appropriate data access and author disclosures. They cannot substitute for every journal-specific requirement. Scientific figures in this package are calculated from recorded data, not generative images of experiments.

### Nature and Nature Portfolio

Nature flagship requires outstanding scientific importance. The accessible evidence does not establish that level of biological or methodological contribution. Nature-family journals are not synonymous with Nature, and an appropriate specialist scope does not lower evidence requirements. A stronger research program should determine the venue, not the reverse.

- https://www.nature.com/nature/for-authors/editorial-criteria-and-processes
- https://www.nature.com/nature/for-authors/initial-submission
- https://www.nature.com/nature-portfolio/editorial-policies/reporting-standards
- https://www.nature.com/nature/editorial-policies/ai
- https://www.nature.com/nature/editorial-policies/preprints-conference-proceedings
- https://www.nature.com/ncomms/aims

Current reporting, data/code, ethics, authorship and AI-disclosure requirements must be satisfied. LLMs are not authors; the human authors remain responsible for claims and citations.

## What the extension changed about venue fit

The extension adds a concrete retrospective measured-source evaluation: the descriptor baseline leads the original six-method exact-scaffold comparison (AUC 0.980), but scores 0.800 on ten measured-source records and 0.5625 at the fixed 1/10 µM cutoffs. RF, RBF-SVM, nearest-active and equal fusion separate the five numeric inhibitors from five reported inactives at 50 µM; GBT does not (AUC 0.940, with ties). This is an observed difference in method ranking across cohorts, label definitions and fitting regimes, not a causal attribution to decoy construction or a prospective-performance estimate. The frozen design, failed fixed-caliper matching and released per-prediction artifacts make it a reproducible case study.

This strengthens the cheminformatics/reproducibility route (Journal of Cheminformatics, JCIM, PLOS Computational Biology, Patterns) and does not create an inhibitor-discovery or methods-novelty claim. n = 10 from one previously inspected publication, one probe pair ordered correctly by half the methods, and no prospective or blinded evaluation remain hard ceilings. A reviewer can still reasonably judge the scope too narrow; no venue fit or acceptance is predicted here.

## Author confirmation, not invented declarations

Before any submission, the authors must confirm:
1. Author order, contributions, correspondence and affiliations. DataPort lists Canyon Crest Academy, but a deposit is not consent to assume the correct affiliation for a revised submission.
2. Funding and competing interests. Absence of statements is not evidence that there were none.
3. Actual scope of any original experiments, prior approvals and source-data ownership. This computational audit itself performed no new human or animal work.
4. Complete data/code availability, original artifact provenance and reuse licensing. No license should be invented for the authors' work or third-party chemistry data.
5. Substantive AI assistance in auditing, coding, analysis, literature triage and drafting; tool identity and extent of human verification. Do not imply human review has already occurred when it has not.
6. Existing preprints, deposits and prior submissions, and an author-approved correction to inconsistent public metadata.

## The most consequential next scientific question

Can independently identity-verified chemistry separate **NUDT5 catalytic inhibition** from protein-loss/noncatalytic effects and produce a disease-relevant phenotype with a defensible selectivity window? Published target engagement is not sufficient evidence of anticancer efficacy. The compound-9 BT-474 viability result reported by Balikci et al. (`10.1021/acs.jmedchem.4c00072`), the TNBC xenograft mortality caveat and 2025–2026 PPAT/thiopurine findings make mechanistic discrimination more useful than another optimistic docking score.

Progress should be governed by evidence gates: identity and source mapping; assay-independent biochemical inhibition; direct binding/selectivity; cellular engagement in the intended model; orthogonal phenotype and causal controls; exposure/tolerability. A negative result at any gate is informative and should not be hidden. These are proposed research directions, not experiments claimed to have been completed or authorized protocols.
