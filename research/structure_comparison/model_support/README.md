# 8RIY/8OTV local-model support (G-M7 / G-S4)

Retrospective assessment from public archive data only. It covers four things: (1) extracting
the official wwPDB validation report, (2) inventorying the deposited structure-factor columns,
(3) limited direct inspection of the precomputed PDBe maps (trilinear values at atom
coordinates plus fixed centroid slices), and (4) a comparison with the published claims. **It
is not independent density validation.** No rerefinement was done, no omit or new maps were
made, and no new experiment was run. No existing geometry result, contract, source snapshot,
or manuscript file was changed. The input hashes are checked against `source_manifest.json`.

Regenerate into a new directory, using the pinned `.venv` from `requirements*.lock`:

```sh
PYTHONPATH=scripts/scripts .venv/bin/python scripts/build_structure_model_support.py --output NEW_DIR
```

The committed `results/` directory is byte-identical to a fresh run, and `completion.json` is
linked last. Any of the following stops the build: a missing or tampered source; a mismatch
in PDB ID, cell, space group, or report chain/entity; a duplicate report row or Miller index; a
nonfinite or out-of-range value; a partial or reordered map; or a nonempty output directory.

## Sources

The full URLs, UTC retrieval times, HTTP headers, raw and stored SHA-256 hashes, sizes, and
licence basis are in `source_manifest.json`. Definitions and short attributed quotations are
in `source_notes.json`.

- **Redistributed in `sources/`:**
  - wwPDB validation XML (RCSB `validation_reports`).
  - Deposited `-sf.cif.gz` files (RCSB).
  - PDBe `8riy|8otv(.ccp4|_diff.ccp4)` maps, stored gzip-compressed with their downloaded hash
    verified.
  - PDBe file-API JSON.
  - Licence basis: PDB archive data are CC0 1.0; the PDBe material is used under the EMBL-EBI
    terms, with attribution.
- **Not redistributed:** the PDBe full-validation PDFs. Their SHA-256 hashes and software
  metadata are recorded:
  - wwPDB-VP 2.49, EDS 3.0, CCP4 9.0.010, Density-Fitness 1.0.12, MolProbity 4-5-2/Phenix 2.0,
    Mogul 2022.3.0, percentiles 20250101.v01.
  - Report dates: 8RIY 2026-03-20; 8OTV 2026-03-08.
- **Access limitations:**
  - The RCSB validation-PDF endpoint returned 403, so the official PDBe copy was used.
  - One guessed PDBe help URL returned 404 and is not used.
  - The wwPDB X-ray help page timed out.
  - The CCP4 map headers do not state a generation version. The map calculation date or
    weighting need not match the report pipeline.
- **Definitions:** RCSB states that RSR measures how well each residue matches the
  experimental data locally (lower is better), and that RSCC reflects agreement between
  coordinates and experimental density. These are model-dependent summary indicators. They are
  not interaction, energy, or occupancy evidence.

## Findings

### Whole-entry report fields

The values below are copied from the report as given:

- **8RIY:** resolution 2.29 Å; R/Rfree 0.2255/0.2948; DCC Rfree 0.3010; Fo/Fc correlation
  0.919; `DataCompleteness` 35.53; clashscore 7.24; RSRZ outliers 6.46%.
- **8OTV:** resolution 1.82 Å; R/Rfree 0.2061/0.2396; Fo/Fc correlation 0.963;
  `DataCompleteness` 98.04; clashscore 3.88; RSRZ outliers 8.75%.

Both structure-factor files contain `F_meas_au`, `pdbx_FWT/PHWT`, and `pdbx_DELFWT/DELPHWT`:

- 8RIY: 26,413 rows.
- 8OTV: 46,127 rows, of which 45,224 have paired difference coefficients.

The PDBe map cell and space group match the coordinates for both entries:

- 8RIY: P 61 2 2.
- 8OTV: P 21 21 21.

### W0O sites

All four W0O sites map exactly to their report rows (label and auth chain, auth number, and
entity). Each site has 30 modelled atoms and NatomsEDS 30.

| Site (label / auth) | Avg occ | RSCC | RSR | Mogul outliers | EDS SD at atoms min / median / max | diff SD at atoms min / max |
|---|---:|---:|---:|---|---:|---:|
| 8RIY C / AAA 301 | 1.000 | 0.931 | 0.094 | 3 angle, 5 torsion | 0.68 / 1.99 / 3.61 | −1.11 / 1.46 |
| 8RIY D / BBB 301 | 1.000 | 0.940 | 0.091 | 3 angle | 0.29 / 2.32 / 3.93 | −1.31 / 1.34 |
| 8OTV C / A 301 | 1.000 | 0.952 | 0.076 | 1 torsion | 0.53 / 2.70 / 4.41 | −1.17 / 1.63 |
| 8OTV F / B 302 | 1.000 | 0.928 | 0.097 | 1 angle, 2 torsion | 1.00 / 2.10 / 3.75 | −2.01 / 0.94 |

How to read the sampled map values:

- The "SD" columns are interpolated map values standardized by the full-cell mean and SD.
- They carry no pass/fail threshold, are not RSCC, and do not prove ligand placement.
- The lowest-valued atom is C1 at every site.
- No difference sample at a W0O atom position reaches ±3 SD. This statement is limited to
  those sampled positions.

The report also contains `EDIAm` and `OPIA`:

- 8RIY: 0.412/16.67 and 0.410/26.67.
- 8OTV: 0.804/83.33 and 0.728/60.00.

These values are retained unchanged but not interpreted, because their definitions were not
recorded in a source here.

### 8RIY Arg51 (label seq 52, auth 51)

- **Chain AAA:**
  - Report values: RSCC 0.894, RSR 0.103, RSRZ 0.065, avgoccu 0.890.
  - CZ is deposited at occupancy 0.000 and was not sampled.
  - The report lists:
    - CZ–NH1 1.959 Å (Z 45.44), CZ–NH2 1.710 Å (Z 29.26), and NE–CZ 0.984 Å (Z −31.12) bond
      outliers.
    - Three angle outliers.
    - Clashes on NE, NH1, and NH2.
  - Sampled EDS SD: NH2 0.16, NH1 1.32, NE 1.56, CD 1.74.
  - The nearest retained W0O witness is the backbone N, at 3.756 Å.
- **Chain BBB:**
  - Report values: RSCC 0.901, RSR 0.143, RSRZ 1.000, avgoccu 0.980. No outliers are listed.
  - CD occupancy is 0.78.
  - Sampled EDS SD: NH2 0.30, NH1 1.11, CD 1.27, NE 1.97, CZ 2.16.
  - The nearest witness is CD, at 3.251 Å.
- 8OTV residue 51 is SER. 8OTV Leu107 is NUDT14 numbering and is not mapped to NUDT5 Arg51.

### Coverage

- All 1,730 original residue-conformer rows are retained without change: 1,564 observed, 58
  partial, and 108 refused.
- The 108 refused rows have no exact report record. They are reported as unavailable, not
  reassigned.
- Fixed local scope: 637 atoms in 65 residue-conformer report records. These are:
  - Atoms within 5 Å.
  - The published anchors: NUDT5 28/46/47/51 and NUDT14 17/34/35/107.
  - 8RIY residues 45–55 in both chains.
- Alternates are kept separate and not averaged.
- Six fixed slice sheets cover the four ligands and both 8RIY Arg51 residues.

## Incremental contribution verdict

The following were already published by Balikci et al. (2024, J Med Chem, CC BY 4.0;
quotations B1–B5 in `source_notes.json`):

- The NUDT5 (8RIY) and NUDT14 (8OTV) compound-9 structures.
- The hydrophobic R51 interaction "in chain B".
- The NUDT14 L107 contrast.
- The statement that R51 "may" explain TH5427 selectivity.

This package adds a reproducible public-data audit:

- Exact all-site and both-chain identity mapping.
- Preserved occupancy, alternates, and missingness.
- Report-metric extraction.
- Proof that structure factors and maps are available.
- Limited direct map sampling.

It **qualifies** the published account. It does not refute it.

- The chain-B (BBB) Arg51 side chain is modelled without listed outliers and lies close to W0O
  through CD.
- The chain-A (AAA) Arg51 guanidinium is modelled with a zero-occupancy CZ and severe geometry
  outliers. A uniform two-chain Arg51 interaction should therefore not be stated.
- This analysis does not test energetics or selectivity.

Distance recalculation and report extraction are reproducibility work. They are not a
discovery, and they do not by themselves establish publishable novelty.

## Unresolved gaps

- **G-S4** is only partly closed. The maps are public and were inspected only through fixed
  sampling and slices. It remains open for:
  - Expert 3-D map review.
  - Omit or polder maps.
  - Rerefinement.
  - Any independent ligand-placement validation.
- **G-M7** is closed only as an assessment: the contribution is methodological and
  qualifying.
- `EDIAm`/`OPIA` definitions and the map-generation version remain unrecorded.

## Writer instructions

- **Main text:** at most one sentence. Report RSCC/RSR as "wwPDB validation-report metrics".
  Never write "density validated".
- **Arg51:** describe it per chain. Chain BBB: CD witness at 3.251 Å, occupancy 0.78. Chain
  AAA: backbone N witness at 3.756 Å; the guanidinium has a zero-occupancy CZ and
  report-listed geometry outliers. Attribute the hydrophobic-interaction interpretation to
  Balikci et al.
- **Supplement:** put the site table, the Arg51 records, the slice sheets, the source manifest,
  and the limitations here.
- **Required wording:**
  - The maps are precomputed PDBe maps, inspected only by fixed sampling and slices.
  - There was no rerefinement, no omit map, and no new experiment.
  - The two crystal copies per entry are not independent observations, and no inferential
    statistics are used.
  - B factors are metadata only.
  - The paper makes no claim about affinity, energy, selectivity, causality, or homology.
- **Do not imply novelty** for compound 9, W0O, R51, or L107.
- **Declarations** stay author-owned, as set out in the Path B author-request sheet: authorship
  (including Nikhil Srinivasan), affiliation, funding, COI, CRediT, prior versions, and AI
  disclosure. Do not invent them.
