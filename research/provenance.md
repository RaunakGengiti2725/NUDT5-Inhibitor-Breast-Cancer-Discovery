# Data provenance, chemical identity and evidence boundaries

## Source tiers

| Tier | Records and use | What is preserved | Limitation |
|---|---|---|---|
| Original release | 46 training rows, 10 candidate rows | Unchanged CSVs, baseline Git revision, hashes in original_inventory.json | One invalid training graph; 19 valid positive labels with incomplete assay mapping; 26 untested decoys |
| Balikci primary supplement | 23 structures: 7 numeric NUDT5 IC50, 5 reported inactive >50 µM, 11 untested | source_assays.csv; exact source URL/hash/transformation in results/source-assay-provenance.json | Previously inspected single publication, related chemistry; 2 numeric records overlap training |
| Public database queries | 29 ChEMBL activity and 21 BindingDB rows in a bounded retrieval | external/observed_database_rows.csv, overlap ledger, 41-file snapshot manifest with URLs/time/hashes | Mirrors of one experiment are not independent. Not an exhaustive search; inaccessible primary sources remain database-only evidence |
| Producer dossier | MRK-952 and MRK-952-NC biochemical and SPR measurements | external/nudt5_measured_ledger.csv and corresponding JSON, URLs and source caveats | Species/construct unstated; NC is a weak inhibitor with assumed stereochemistry; IC50 prose/EC50 curve discrepancy |
| Authentic reference structures | Page/PDB 9CH (TH5427) and 958 (TH1713) | reference_structures.csv plus exact downloaded CIFs in reference_sources/ | 958 has a conflicting TH5427 synonym; resolve by graph, source and accession, not synonym alone |
| Structural sources | 5NWH, 5NQR, 8RDZ, 8RIY, 8OTV | structural_metadata.csv, source register and evidence ledger | Construct discrepancies, missing loops, metal/solvent differences and local occupancy caveats; no new docking or energetic inference |

The 2017 NUDT5 IC50 >100 µM record is a censored database-supported counter-screen, not an exact 100 µM observation. Two Kinobead records are lysate apparent Kd, not purified-enzyme IC50. TH5427 database mirrors of Page 2018 are not new independent evidence. Cell-viability IC50 values were excluded from biochemical model labels. No Ki observations were found in the queried records; that is not evidence that none exist elsewhere.

## Chemical identity and normalization

The model uses original sanitized graphs and achiral radius-2/2048-bit Morgan fingerprints. Canonical **isomeric** SMILES and standard InChI/InChIKey identify structures, so stereo information is retained in provenance even though the model representation does not use chirality. Charge/fragment and unassigned-tetrahedral-centre fields make uncertainty visible. Exact graph, neutral fragment-parent and canonical parent-tautomer matches are separately reported. Parent/tautomer operations can remove details, including stereo affected by tautomerization, and are exclusion flags rather than proof of identical biological activity. No failed molecule is repaired by guessing; no normalization silently rewrites the CSVs.

NC5-02 is the exact graph of ACT-19 and Balikci compound 11: published NUDT5 and stronger NUDT14 inhibition are source facts, not a new discovery. Claims of exact-molecule novelty are removed. The remaining candidate structures have no newly established activity, selectivity, patent novelty or disease-specific benefit. PAINS and rule-of-five flags are screening alerts, not evidence of toxicity or experimental inactivity. Broad chemical novelty, patent priority, synthetic accessibility and compound availability have not been comprehensively established.

## Reproduction and derived artifacts

`pipeline.py` retains original-graph benchmark diagnostics. `controls.py` implements the locally recorded extension design. `transfer.py` scores exposed source compounds, retains each endpoint record and separately reruns all six original methods with two authenticated reference substitutions. That sensitivity does **not** authorize altering the original deposit and does not repair other unknown source labels. Every input and code file is hashed in the corresponding run manifest; absolute paths record the original run location and can be remapped to the checkout. Dirty-worktree flags honestly identify pre-commit runs; hashes are authoritative for their exact source bytes.

Derived figures/tables are built from recorded JSON, not hand-drawn numbers. `all_metrics.csv` contains all implemented cohorts/methods; `candidate_axes.csv` keeps score rank and training-dissimilarity rank separate from qualified experimental roles. The full transfer JSON preserves source values, relation/units, identity flags and predictions, including excluded/untested records. Raw external snapshots are preserved as a separately downloadable source archive referenced by external/artifact_sources.json rather than duplicating a roughly 29 MB archive in Git.

## Evidence still requiring authors or a laboratory

An authoritative ACT-18 graph; original positive-label assay mapping; decoy provenance/inactivity testing; historical proxy/library/docking outputs; chemical-material identity/purity; orthogonal biochemical and binding assays; NUDT14/NUDIX selectivity; context-qualified target engagement, rescue and phenotype experiments. Author declarations, reuse licensing and AI-assistance disclosure require explicit confirmation. No public deposit, journal submission, animal experiment or clinical intervention was performed here.

## Lossless repository transport

The two authenticated PDB chemical-component files are stored as deterministic gzip archives in `reference_sources/`; decompressing preserves their original bytes and the SHA-256 values in `reference_structures.csv`. Newly curated CSV copies use LF row terminators; parsed cell values are unchanged. The original `compounds.csv` and `final_hits.csv` remain byte-identical. Imported source attachments remain separate, unchanged evidence. Generated SVG trailing formatting whitespace is normalized without changing the plot.

## Paired-target and future-input integration (5 October 2026)

The selectivity extension retains 23 graphs and 46 endpoint cells from the already-inspected
Balikci publication. Eight rows have paired target endpoints; training-overlap compounds 10/11
remain in pharmacology but are excluded from score diagnostics, leaving six. Three ratios are
points, one is a strict upper bound and two are double-censored with no finite bound. Both frozen
score scenarios are retained without refitting. `selectivity/provenance.json` records source URLs,
retrieval times and uncompressed hashes. Integration re-inspected the stored Table 1 value image
(all 16 target cells), XML Table 1 footnotes, Catalytic Assays sec4.3 and Figure 1 caption; no new
source retrieval or independent measurement is implied. Conditions and source disagreements
remain in their separate ledgers. The reciprocal ratio orientation in the older structural
report is not silently rewritten; this analysis defines R = IC50(NUDT14)/IC50(NUDT5).

The bounded PubChem follow-up in `external/pubchem/README.md` links 26 concise rows/16 CIDs to
existing ChEMBL ledger rows. Its 21 assay descriptions include 12 genetic-perturbation screens,
not chemical inhibition experiments. Six matched values are censored and two missing; relation
symbols come from the existing ledger, not from the concise API. Raw snapshots/manifests and the
offline join audit preserve this distinction. The earlier search report remains a historical record.

`assay/PROTOCOL.md` and its primary-reference ledger describe future-input requirements. All
synthetic curves are SOFTWARE TESTS ONLY, generated in isolated test directories, never committed
as measured data or biological Results. The analyzer checks bytes and declared metadata, not
vendor semantics, normalization arithmetic, actual blinding, biological independence or scientific
qualification. All laboratory prerequisites and prospective-lock decisions remain unresolved.
Installed wheels accept an explicit evidence repository; the assay report hashes the actual installed
analyzer and writer plus the selected contract/schema/lockfile, not a guessed checkout code path.

Historical analysis and curation manifests remain their original execution-time records. New
local reruns have separate manifests with their real revision and dirty state. The portable
`release_manifest.json` inventories final content without pretending historical runs occurred
post-commit. Original CSVs, source assay ledger and historical numerical JSON remain byte-identical.
