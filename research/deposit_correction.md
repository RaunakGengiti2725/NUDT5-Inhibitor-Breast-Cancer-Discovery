# Proposed correction to public dataset descriptions

**Draft for author approval only. Nothing has been posted to IEEE DataPort or Zenodo.**

## Replacement scope statement

This deposit contains a small computational NUDT5-related demonstration and ten proposed structures, not an experimentally validated inhibitor discovery dataset. The released compound CSV has 46 rows: 20 labelled positives and 26 untested decoys. RDKit parses 45 rows; ACT-18 requires source-structure clarification. NC5-02 is identical to training record ACT-19. Source labels have not been matched to complete assay-level provenance, and decoy labels do not establish experimentally confirmed inactivity.

The archived code repeats 26 decoys to create 520 rows before cross-validation. This repetition does not supply 520 independent compounds and allows negative-identity overlap across train/test folds. Its BEDROC implementation is incorrectly normalized. The manuscript's reported p < 0.001 cannot be supported by the stated 30-permutation empirical procedure. Historical candidate scores should not be treated as regenerated or validated activity probabilities.

The inspected release does not contain the claimed 347 MTH1 proxy records, complete 18,412-compound screening library, docking poses/logs or MM-GBSA outputs. Until their original artifacts are supplied and verified, claims depending on those artifacts remain unverified. Except for NC5-02's independently authenticated published biochemical activity, candidate activity remains unverified by this revision; none is established here as a therapeutic lead or treatment. NC5-02 also inhibits NUDT14 in the published assay and is not a new discovery.

A separate corrective revision provides chemical-data validation, duplicate-aware exploratory diagnostics, tests and an evidence ledger. Its newly computed statistics are not replacements for missing historical experiments and should be versioned separately from the original deposit.

## Record-preserving update procedure

1. Authors confirm the facts, missing-artifact status and exact revised-code commit.
2. Preserve the existing DOI/version and original files; do not silently replace or backdate them.
3. Add a dated correction and a new version with a clear relationship to the original deposit, following each repository's official update procedure.
4. Include the exact code revision, input hashes, environment lock and regenerated outputs in the new version.
5. Correct the abstract, file descriptions and any downstream manuscript/preprint text consistently.
6. Confirm authorship, affiliations, reuse licenses, funding, competing interests and actual AI use before publication. The audit does not authorize declarations on the authors' behalf.
