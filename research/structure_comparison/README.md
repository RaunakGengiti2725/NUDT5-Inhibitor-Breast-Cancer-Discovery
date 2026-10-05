# Compound 9: bounded deposited-coordinate comparison contract

This package freezes the inputs and analysis rules for **8RIY (human NUDT5)** and
**8OTV (human NUDT14)**, both with **W0O / Balikci compound 9**. It contains source
snapshots and inventories, **not a distance-analysis implementation or newly measured biology**.
Both complexes and the Arg51 rationale were published in Balikci et al., *J. Med. Chem.*
67, 7245–7259 (2024), [DOI 10.1021/acs.jmedchem.4c00072](https://doi.org/10.1021/acs.jmedchem.4c00072),
[PMC11089510](https://pmc.ncbi.nlm.nih.gov/articles/PMC11089510/).

## Verified input facts, with their limits

| Deposit | Deposited protein, not an assumed lab construct | Resolution | Official assembly already in ASU | Every W0O site: model / label chain / author chain:residue |
|---|---|---:|---|---|
| 8RIY | NUDT5, Q9UKK9 residues 1–208 plus residual Ser0; 209 deposited sequence positions; human, E. coli expression | 2.288 Å | Assembly 1, operation 1 = identity; protein A/B = author AAA/BBB | 1 / C / AAA:301; 1 / D / BBB:301 |
| 8OTV | NUDT14, O95848 residues 1–222 plus residual Ser0; 223 positions; human, E. coli BL21(DE3) expression | 1.82 Å | Assembly 1, operation 1 = identity; protein A/B = author A/B | 1 / C / A:301; 1 / F / B:302 |

Both deposits are X-ray diffraction models and report zero sequence mutations. Exact deposited
sequences, reference alignments, tag discrepancies, matrices, assembly-member lists, species and
refinement metadata are in [structure_metadata.json](structure_metadata.json) and their raw sources.
“Complete assembly membership” does **not** mean complete atomic geometry. Apply neither a second
identity copy nor guessed crystal symmetry. Each site uses **both** protein chains.

[ligand_identity.json](ligand_identity.json) links W0O exactly to the existing compound-9 ledger
using InChI, InChIKey **KHLKLLMMPVSSQY-UHFFFAOYSA-N**, formula **C23H24N6O**, charge 0 and a bounded
canonical-isomeric-SMILES comparison with the existing pinned RDKit. All four sites contain the
same complete **30-heavy-atom CCD name/element set**, with ligand occupancy 1 and altloc `.`.
This is chemical bookkeeping, not purity, protonation, affinity or selectivity validation.
CCD ideal/model coordinates are never substituted for deposited coordinates.

## Missingness and conformers determine what can be calculated

- **8RIY Arg51:** model 1, label A/auth AAA, label residue 52/auth 51, atom-site 314, CZ has
  occupancy **0.000** and is excluded. Its deposited coordinate is retained only for traceability.
  Fractional atoms are AAA Arg51 O **0.770**, BBB Arg51 O **0.970**, CD **0.780**, NH2 **0.990**.
  Positive fractional atoms remain flagged, not occupancy-weighted or treated as certain.
- **8OTV Leu47:** label/auth B, label residue 48/auth 47 has two alternatives, **A and B**, each
  eight heavy atoms at occupancy **0.50**. Report two local-residue conformer rows; do not pick
  the first/highest-occupancy copy, combine their atoms or invent cross-residue correlations.
- **8RIY missing residues:** AAA 0–13; BBB 0–13 and 54–56. There are 31 absent residues, 83 declared
  unobserved atoms and the one zero-occupancy atom above. **8OTV:** A 83–85, 173–177, 220–222;
  B 0, 169–177, 221–222: 23 absent residues and 31 declared unobserved atoms. Exact rows, including
  residue names and label/auth identifiers, are in the two missingness CSVs.
- **All four sites refuse complete-pocket geometry.** They remain eligible for explicitly
  observed-coordinate maps. Missing residues get null distances, never zero/infinity; partially
  modeled residues get conditional observed minima and missing-atom reasons. No observed pair
  inside a radius does not prove absence of a contact involving missing atoms.
- 8RIY contains **33 waters and no modeled metal**; 8OTV contains **216 waters, one Mg and one
  DMSO**. The paper reports different crystallization buffers/temperatures. These differences
  do not establish metal-independent catalysis or explain affinity/selectivity.

No structure factors, electron-density maps, density fitting or refinement were inspected or
performed. Resolution, B factors and deposited occupancies are not a coordinate-confidence
interval. No model positions were rebuilt. See the [caveat ledger](caveats.csv), which binds
these limits to specific observations rather than a generic closing disclaimer.

## Implement the fixed contract, not an inferred interaction model

[geometry_contract.json](geometry_contract.json) is normative; [acceptance_cases.json](acceptance_cases.json)
specifies 25 small synthetic/parser/output acceptance cases, **not passed tests of an absent engine**.

1. Accept explicit repository, input-manifest, contract and output paths. Verify all hashes and
   exact identities. Parse complete mmCIF categories with a mature library; do not write a CIF
   parser or accept a high-level conversion that discards alternatives/models/author IDs.
2. Enumerate every ligand site and every residue on both dimer chains. Exclude H/D and
   occupancy-zero atoms; retain exclusions, nonprotein inventories and missingness.
3. For each eligible ligand/local-residue conformer, compute the minimum Euclidean distance
   between retained heavy atoms, in Å. Preserve minimum witness pairs and all pairs ≤5.0 Å.
   The **primary radius is 4.0 Å**; **3.5, 4.5, 5.0 Å** are fixed sensitivity conventions, all
   inclusive, with no rounding before comparisons. They are not chemical/affinity thresholds.
4. `.` atoms are shared with each explicitly labeled local conformer, except duplicate
   shared/alternate atom names are refused as ambiguous. `?` atom altloc is unknown, not `.`.
   If both a ligand and a protein residue have alternatives, this v1 refuses their paired
   geometry without an independently supported compatibility policy. The real W0O sites have
   no ligand alternatives, so both Leu47 conformers remain separately calculable.
5. Retain unobserved/refused residues as nulls with reasons. Expected residue/conformer rows are
   **418 per 8RIY site, 447 per 8OTV site, 1730 total**. This is inventory coverage, not biological n.
   Complete-residue distance is not estimated; primary output is observed distance only.
6. Write one finite JSON report atomically to a **new** path: same-directory staging, flush,
   fsync, non-overwriting hardlink, cleanup. Existing files/directories/symlinks and races must
   not overwrite prior results. Exit 0 means a report was written, possibly containing refusals;
   exit 2 means invalid input or failed output, not a silently truncated result.

Do not average crystal sites as independent samples, calculate CIs/p-values, assign hydrogen
bonds/energies, introduce models or tune thresholds. Do not superpose proteins or treat NUDT5
Arg51 and NUDT14 Leu107 as homologous residues. Their published ligand-region assignments are
reported as **author interpretations**, not new computed contacts or residue correspondence.

## Files and source provenance

- [retrieval_manifest.json](retrieval_manifest.json): exact retrieval time, URL, byte count and
  SHA-256 for every new remote source; [sources/](sources/) contains the actual unmodified bytes.
  Both mmCIF hashes match the parent's supplied preflight. Current bytes are not rewritten into
  an old manifest. The paper XML also matches the existing archived primary-text bytes.
- [input_manifest.json](input_manifest.json): explicit repository-relative, hash-bound inputs,
  all expected sites, model/assembly identities and unchanged compound-9 ledger links.
- [ligand_sites.json](ligand_sites.json) and [ligand_atoms.csv](ligand_atoms.csv): four full site
  identities and all 120 deposited W0O atom rows, including coordinates/occupancy/B factors.
- [flagged_atoms.csv](flagged_atoms.csv): all 21 zero/fractional/alternate records.
  [primary_exclusions.csv](primary_exclusions.csv): all 375 rows excluded from the **protein
  receptor pool**. W0O rows are still query atoms at their own site; other W0O sites are not protein.
  Neither deposit contains coordinate H/D rows, but the implementation must test their exclusion.
- [polymer_residue_inventory.csv](polymer_residue_inventory.csv): all 864 deposited sequence
  positions; [unobserved_or_zero_occupancy_atoms.csv](unobserved_or_zero_occupancy_atoms.csv) and
  [unobserved_or_zero_occupancy_residues.csv](unobserved_or_zero_occupancy_residues.csv) preserve
  every deposited missingness row. Absence of a missingness flag is not independent density QC.
- [primary_excerpts.json](primary_excerpts.json): exact XML text/XPath, source hash, attribution,
  claim categories and claim-specific limitations. Paper © 2024 The Authors, CC BY 4.0.
- [inspection_provenance.json](inspection_provenance.json): actual curation environment and
  transformations; [inspection-parser.lock](inspection-parser.lock) hashes the inspected Gemmi
  **0.7.3 CPython 3.12 Linux x86_64 wheel**, installed outside the root environment. Official
  upstream versioned API documentation/PyPI metadata are archived. Inspection audit found no
  known advisories, not a security certification. Implementer must reverify version/API/security
  and lock a runtime dependency if used; **no root dependency pins changed**.
- [verification.json](verification.json) and [verification.md](verification.md): actual evidence
  checks and root test/build/audit results. [artifact_manifest.json](artifact_manifest.json)
  binds this package's delivered files, excluding its own self-hash.

No distance results, software engine, experiments, laboratory qualification, new inhibitor,
selectivity proof, causal mechanism or clinical benefit is claimed. Old data, research/results,
manuscript and provenance files remain byte-for-byte unchanged from the incoming commit.
