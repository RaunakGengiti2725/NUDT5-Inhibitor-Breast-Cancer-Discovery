# Observed-coordinate W0O proximity: run record

Retrospective descriptive geometry from deposited coordinates (8RIY/NUDT5, 8OTV/NUDT14). Compound 9 binding to both proteins, these structures and the Arg51 rationale are published work; nothing here is a new biological observation, binding energy, affinity, hydrogen-bond assignment, selectivity result or validation. The four W0O sites are crystal copies, not independent samples, so no statistics are computed across them.

## Outputs

- `observed_proximity.json` (SHA-256 `c4ded0765a28a07fc52e7cebdaf52536d5d3b6fe0d884394e78c951a93c1fffd`): every residue/conformer row for both receptor chains at all four sites, atom witnesses, all ligand-protein heavy-atom pairs within 5.0 Å, exclusion ledger, non-protein inventory, missingness and refusals.
- `derived/residue_proximity.csv`, `derived/atom_pairs_within_5A.csv`, `derived/radius_sensitivity.csv`: flat tables with full-precision distances; blank distance means not observed or refused, never zero.
- `derived/observed_proximity_map.{png,svg,pdf}`: per-site minimum distances for every residue/conformer with a retained pair within 5.0 Å; null rows and alternates beyond 5.0 Å are enumerated in the figure footer.
- `run_provenance.json`: commands, code commit, dependency versions, lock and artifact hashes, counts and hand checks.

## Per-site results

| Site | Target | Rows | Observed | Partial | Refused | Residue rows ≤3.5 / 4.0 / 4.5 / 5.0 Å |
|---|---|---|---|---|---|---|
| `8OTV:model1:C:A:301` | NUDT14 | 447 | 417 | 7 | 23 | 4 / 11 / 13 / 15 |
| `8OTV:model1:F:B:302` | NUDT14 | 447 | 417 | 7 | 23 | 4 / 8 / 11 / 12 |
| `8RIY:model1:C:AAA:301` | NUDT5 | 418 | 365 | 22 | 31 | 7 / 9 / 14 / 15 |
| `8RIY:model1:D:BBB:301` | NUDT5 | 418 | 365 | 22 | 31 | 8 / 12 / 15 / 16 |

Counts are shared-conformer rows. No 8OTV Leu47 alternate (A/B) row is within 5.0 Å at either site, so no alternate rows enter these counts. Refused rows are unobserved residues: distance null, not no-contact. Partial rows (8RIY Arg51 chain AAA, where CZ is at zero occupancy, and residues with declared missing atoms) report the minimum over retained atoms only; because absent atoms could lie closer, this is an upper bound on the unknown complete-residue minimum, not a complete-residue distance; complete-residue distance is `not_estimated` everywhere.

At 4.0 Å, 8RIY chain BBB Arg51 is within the radius at its own site D (3.25 Å, fractional occupancy, unweighted), and chain AAA Arg51 is within the radius at site C (3.76 Å, partial: CZ at zero occupancy). This is proximity only. It does not test the published Arg51 hypothesis. NUDT14 Leu107 rows are listed by their own numbering and are not treated as homologous to Arg51.

## Independent checks

Five witness distances were recomputed from the raw mmCIF text with `awk` (no Gemmi, no NumPy):

| Site | Residue | Ligand atom | Protein atom | Engine (Å) | awk (Å) |
|---|---|---|---|---|---|
| `8RIY:model1:D:BBB:301` | ASP auth BBB/60 | C18 #2968 | OD2 #1818 occ 1.0 | 2.710402553127 | 2.710402553127 |
| `8RIY:model1:D:BBB:301` | ARG auth BBB/51 | C18 #2968 | CD #1775 occ 0.78 | 3.251272520107 | 3.251272520107 |
| `8RIY:model1:C:AAA:301` | ARG auth AAA/51 | C20 #2940 | N #306 occ 1.0 | 3.755561342862 | 3.755561342862 |
| `8OTV:model1:C:A:301` | ASP auth A/35 | N5 #3249 | O #273 occ 1.0 | 2.850305948490 | 2.850305948490 |
| `8OTV:model1:F:B:302` | LEU auth B/148 | N5 #3284 | CD2 #2737 occ 1.0 | 3.311839821006 | 3.311839821006 |

Every observed residue minimum (all 1,622 non-null rows) is also checked in `tests/test_structure_comparison.py` against Gemmi's high-level `Structure` model with `math.dist`, which is a separate code path from the engine's raw-category parser and NumPy arithmetic. Synthetic fixtures in the tests are software checks only and do not enter any evidence ledger.

## Dependency and provenance notes

- Gemmi 0.7.3 is pinned with hashes in `requirements-structure.lock`, installed after `requirements.lock`. The root `requirements.txt` and `requirements.lock` are byte-identical to the base commit, because the existing selectivity analysis fails closed if the recorded `requirements.lock` hash changes. CI installs the extra lock.
- `runtime_input_manifest.json` binds this run's inputs. The historical `input_manifest.json` is unchanged and bound as an input.
- Python and package versions in `run_provenance.json` apply to this structural run only. Earlier `research/results/*` files keep their own recorded runtimes.

## Limits

- Only deposited coordinates are used. No electron density, structure factors or refinement are inspected, and coordinate uncertainty is not propagated. Occupancy and B factors are not confidence intervals.
- Missing loops/atoms are not imputed. A closer contact by an absent atom cannot be excluded.
- Alternates are evaluated locally per residue, not as correlated whole-structure states. Paired ligand/protein alternates are refused (none occur here).
- Assembly 1 is the identity over the deposited dimer. No symmetry mates or lattice contacts are generated.
