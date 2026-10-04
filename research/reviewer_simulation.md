# Adversarial review and publication decision record

This is a model-assisted rejection simulation and synthesis of the six initial independent audit scopes, two implementation/manuscript reviews and the later public-assay/structural workflow. The nine perspectives below are not nine new independent experts, laboratories or journal decisions. Parent session: https://la-hacks-rttothemoon.devinenterprise.com/sessions/67a2d297f2b8476093efc2ab17ac2c9c . No acceptance or human peer review is claimed.

## Nine distinct questions

| Perspective | Strongest rejection argument | Available-data repair | What remains genuinely unresolved | Strongest defensible contribution |
|---|---|---|---|---|
| ML skeptic | Labels may encode source/property differences; AUC cannot establish target recognition | Seven scalar baselines; leave-one-out ablations; constant/prevalence baselines; exact-scaffold, series and 0.70-component partitions; complete OOF outputs | No large, verified cross-source training set or prospective cohort | An executable test of trivial-property alternatives |
| Medicinal chemist | Named reference graphs and a candidate novelty claim are incorrect | InChI/canonical/parent/tautomer ledger; authentic reference sensitivity kept separate; known compound 11 explicitly identified | No authenticated physical samples; uncertain analogue labels; patents not exhaustively searched | Traceable structure/assay linkage rather than nominal identifiers |
| Statistician | n=45 repository records and n=10 exposed source records cannot support general performance claims | Plus-one permutations; finite resolution; conditional scaffold resampling; fold assignments; arbitrary-threshold warnings; all seeds/cutoffs retained | Retraining uncertainty not covered by fixed-score intervals; source challenge not blinded; correlated chemistry | Transparent estimands and counterexamples to naive pooled interpretation |
| Structural biologist | Docking claims lack artifacts and cannot prove inhibition or selectivity | Structural metadata, missing-residue/metal/construct caveats; paired-ligand NUDT5/14 comparison and falsifiable binding experiment | No new contact-energy calculation, redocking, MD or causal mutation evidence | Existing structures define tests of differential recognition |
| Cancer biologist | Catalytic inhibition, protein loss and noncatalytic function are not equivalent interventions | Separate E112Q/Y74E hypotheses, chemical controls, NUDT14 attribution, same-model context comparison | No new target engagement, rescue, phenotype or exposure data | A qualified separation-of-function roadmap |
| Reproducibility expert | Original source labels and screening artifacts are missing | Immutable baseline/input hashes; strict parser; manifests; locked dependencies; tests; executable figures and challenge scripts | Cannot reconstruct the historical proxy/library/docking experiment | Independent reproduction of the new diagnostics, not the absent history |
| Journal editor | One corrected tiny dataset may be insufficiently general or consequential | Centre the paper on label-source dependence, identity and benchmark design rather than high AUC | General relevance requires additional independent datasets or experiments | Concrete reusable diagnostics with a complete negative/positive-result accounting |
| Hostile reviewer | New exposed-source AUC=1.0 could just replace one overclaim with another | Exclude known overlaps; report analogues, n=10, all 1/10/50 µM cutoffs and probe-pair disagreement; no validation or superiority claim | No causal decomposition of why source rankings differ | The pipeline reveals both favourable and unfavourable challenges |
| Supportive expert | A correction alone risks offering no next step | Integrate a source-level challenge and staged selectivity/mechanism programme with go/no-go gates | Experimental collaboration and author oversight still needed | A testable path from benchmark audit to qualified biochemical evidence |

## Venue-specific rejection simulation

- **Strong general ML venue:** no new learning algorithm, broad task suite, or established general result. Repairable now: correct evaluation and comparisons. Not repairable by rhetoric: breadth and methodological significance.
- **Bioinformatics/computational-biology venue:** unclear impact beyond one small collection and uncertain biochemical label provenance. A broader multi-target/source study or independent assay dataset would be more persuasive.
- **Cheminformatics journal:** a plausible case-study/reproducibility route, but an editor may still find limited novelty, dataset size or generality. Clear executable identity/provenance and property controls are the relevant contribution.
- **High-impact interdisciplinary journal:** no new experimentally supported mechanism, chemistry or widely demonstrated computational advance. More simulations alone would not close this gap.
- **Cancer/biology journal:** no new cellular causal evidence, selectivity window, exposure or mechanism. The proposed experiments must be performed and independently interpreted, not described as results.

## Decision boundaries

No new screening library was manufactured to imitate the missing historical cascade. No graph neural network was added to 45 uncertain labels merely for complexity. Temporal and publication-held-out validation are not asserted from unverified provenance and two sparse positive source groups. Source-label adversarial classification would largely restate the way this dataset was assembled; scalar descriptor and direct source-challenge results are more informative here. No claim of an exhaustive patent/database novelty search is made.

The next decision is which evidence-first project the authors want to pursue: an openly reproducible computational case study with broader independent datasets, or an experimentally led chemical-probe/separation-of-function study. The current package supports planning both; it does not replace either missing evidence base.

## Extension review reconciliation

Two independent sessions reproduced the original extension JSON/results and identified source-input validation and prose gaps. Reports preserved under `reviews/` apply only to their named snapshots, not subsequent reference-sensitivity outputs. Repairs include all 1/10/50 µM columns/counts; GBT ties; direct SGC/mutant citations; construct, engagement and separate catalytic-readout qualification gates; scoped caliper infeasibility; explicit scaffold overlap in the non-nested component split; descriptive rather than inferential one-pair claims; bootstrap summaries rather than nonexistent stored draws. Source CSV validation now rejects duplicate identifiers/graphs, conflicting alias labels and inconsistent inactivity censor bounds before scoring; review reproductions are regression tests.

Original external/structural/study reports remain source documents. Their assertion that one pair statistically falsifies potency discrimination is superseded by the manuscript's descriptive, tie-aware interpretation. Their strict-holdout exclusions remain in force.
