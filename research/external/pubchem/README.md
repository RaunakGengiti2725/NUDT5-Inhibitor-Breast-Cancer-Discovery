# Bounded PubChem BioAssay follow-up

**Evidence class:** verified database-provenance cross-check, not a new biological result.
**Retrieval:** 5 October 2026 UTC; exact timestamps, URLs, byte counts and SHA-256
hashes are preserved in the four request manifests. No models were retrained,
thresholds changed, or measured-source evaluation rows added.

This follow-up addresses the PubChem search gap explicitly left open in
`../NUDT5_external_evidence_report.md`; it does not retroactively change that
report's search history. Two official PUG REST target-index queries used human
NUDT5 Gene ID **11164** and protein accession **Q9UKK9**. They returned 13 and 9
assay IDs, respectively, with one shared ID: **21 unique assay descriptions**.
Description inspection identified 12 RNAi/genetic-perturbation screens and 9
ChEMBL-deposited protein-assay records. Genetic perturbation is not compound
inhibition evidence. The full descriptions remain available in the snapshot.

## Endpoint and independence triage

| PubChem AID | Deposited assay scope | Use in this package |
|---|---|---|
| 1440791 | 2017 dCTPase-paper NUDT5 biochemical counter-screen | Existing ChEMBL record; primary-source access limitations remain |
| 2070370 | Balikci 2024 qualitative pull-down | Same publication; not a quantitative inhibition endpoint |
| 2070371 | Balikci 2024 luminescence-based catalytic IC50 | Existing source, not an independent holdout |
| 2070372 | Balikci 2024 qualitative catalytic activity, tested up to 50 µM | Same publication; not converted to an exact IC50 |
| 2070373 | Balikci 2024 SPR Kd | Binding, kept separate from biochemical inhibition |
| 2070374 | Balikci 2024 cellular NanoBRET EC50 | Cellular engagement, not biochemical inhibition or viability |
| 2116012 | Non-kinase off-target review-derived IC50 | Secondary source; not treated as a new primary experiment |
| 2148899 | NVP-BHG712/regioisomer Kinobead Kd and ED50 | Existing lysate-binding records; not enzyme IC50 |
| 2200593 | EUbOPEN probe biochemical/affinity annotation | Existing ChEMBL annotation; no new independent experiment established |

Source assay identifiers and links for all 21 records are in `audit.json`.
The classifications above use the deposited descriptions, not fresh inspection
of every underlying experiment or raw plate. AID counts are not compound
counts, experiment counts, or independent-publication counts.

## Record-level cross-check

The seven quantitatively annotated AIDs yielded **26 concise activity rows**
covering **16 distinct PubChem CIDs**. Every row matched exactly one existing
ChEMBL row in `../observed_database_rows.csv` by deposited ChEMBL assay ID,
full InChIKey, endpoint, and reported value after nM-to-µM conversion; missing
values were kept missing. This is database linkage, not independent replication
or proof of physical sample identity. These records therefore do not enlarge
the project's independent biochemical evidence base.

**Important API limitation:** the concise response does not include relation
symbols. Six matched ledger records have `>` censoring, and two have no numeric
value. For example, PubChem's concise value `100` µM for AID 1440791 corresponds
to **>100 µM**, not an exact 100 µM IC50, in the existing ChEMBL ledger. Likewise,
a displayed 50 µM value is not automatically an exact measurement. `audit.json`
labels the relation explicitly as coming from the existing ledger; the original
snapshot is not rewritten or silently “corrected.” No activity labels were
inferred from these displayed numbers or from PubChem outcome categories.

## Offline reproduction

From the repository root with its documented hash-locked environment:

```sh
PYTHONPATH=scripts/scripts .venv/bin/python scripts/build_pubchem_audit.py \
  --snapshots research/external/pubchem \
  --ledger research/external/observed_database_rows.csv \
  --output /path/to/new/pubchem-audit.json
cmp /path/to/new/pubchem-audit.json research/external/pubchem/audit.json
```

Use an existing output parent directory and a new filename. The script checks
all downloaded payloads against their manifests, requires one-to-one source
matches, keeps endpoint types separate, and refuses output replacement. Tests
cover deterministic reproduction, censoring provenance, corrupted snapshots,
invalid concentrations/units, and ambiguous or changed source matches.

**Search boundary:** two target indexes are not an exhaustive PubChem search.
Unindexed assays, other accessions/organisms, keyword-only records and later
updates may be missed. No broad RNAi activity-table download was performed.
The two qualitatively annotated Balikci AIDs were inspected as metadata, not
converted into extra numeric activity rows. Live API responses may change;
reproduction uses the archived payloads, not current network responses.
