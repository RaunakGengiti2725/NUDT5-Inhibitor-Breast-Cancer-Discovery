# Independent public NUDT5 biochemical evidence: bounded primary-source investigation

Retrieval window: 2026-10-04T22:18:31Z to 2026-10-04T22:23:03Z (UTC). ChEMBL_37 (released 2026-05-01), BindingDB REST, Europe PMC REST, RCSB, UniProt, SGC. RDKit 2025.03.6.
Read-only. No repository file was opened, no commit, PR, session, deposit or outreach was made, no model was trained, and no dataset selection was optimized.

## 1. The actionable new angle

The single most actionable item found is not a new bulk dataset. It is a **structurally disjoint, externally produced NUDT5 chemical-probe pair with a machine-readable structure file**: SGC/MSD **MRK-952** and its matched weak control **MRK-952-NC**.

Why it is actionable:

- It is **chemically disjoint from the Balikci/TH5427 chemistry already in hand**. Maximum Morgan(r=2, 2048-bit, chirality-aware) Tanimoto of MRK-952 to any retrieved Balikci ChEMBL graph is **0.187**; for MRK-952-NC, **0.157**. Neither graph appears in any ChEMBL or BindingDB row retrieved for the target.
- It carries **measured biochemical values with a built-in difficulty gradient**: MRK-952 NUDT5 IC50 85 nM (AMP-Glo, ADP-ribose substrate) versus MRK-952-NC IC50 10 µM, plus independent SPR KD 0.031 ± 0.018 µM versus 1.380 ± 0.380 µM (n = 2).
- Its structures are **exactly specified and verified**, not drawn from a figure. The SGC CSV export carries isomeric SMILES, InChIKey and molecular formula; we recomputed both from the SMILES with RDKit and both matched the published InChIKey and formula exactly.

This is a *diagnostic probe set*, not a training resource: two graphs with ~100-fold measured separation on the same target, invisible to the existing series. That is enough to test whether a scoring function discriminates by NUDT5 pharmacology or by proximity to the one chemotype it has already seen. It is not enough to estimate a general error rate.

**Three cautions must travel with it.** (i) The SGC page's prose says "IC50", but the accompanying biochemical figure labels its own fits **EC50** (MRK-952 0.0849 µM, 95% CI 0.0815–0.0883; NC 10.6 µM, 9.62–11.8); we record the prose values and the figure labels separately and never count them as two observations. (ii) The inspected dossier does **not state the assay species or construct**, so these records cannot enter a species-confirmed holdout without confirmation. (iii) The NC figure states "single isomer, stereochemistry assumed", and MRK-952-NC is a **weak inhibitor, not an inactive**; scoring it as a decoy would be wrong.

The 2026 Nature Communications NUDT5 degrader paper (PMC13462928) cites MRK-952 by this exact SGC URL and states the probe work is "to be published", which corroborates the probe's existence and use while confirming no dedicated peer-reviewed assay paper was available to us.

## 2. What the databases actually contain

Human NUDT5 is UniProt **Q9UKK9**, ChEMBL target **CHEMBL4105713** ("ADP-sugar pyrophosphatase", tax 9606). The ChEMBL free-text target search returns exactly one target, so there is no sibling-target ambiguity to resolve.

Complete, uncapped target queries returned **29 ChEMBL activity rows and 21 BindingDB affinity rows: 50 raw rows, but far fewer measurements.**

| | ChEMBL | BindingDB |
|---|---|---|
| IC50 | 13 | 10 |
| EC50 | 7 | 7 |
| Kd | 4 | 4 |
| ED50 | 2 | 0 |
| qualitative "Activity" | 3 | 0 |
| **Ki** | **0** | **0** |

After deduplicating by publication, **22 of 29 ChEMBL rows and 17 of 21 BindingDB rows are Balikci 2024 (10.1021/acs.jmedchem.4c00072)** — already curated, and explicitly out of scope as new data. What remains outside Balikci is small and mostly not purified-enzyme inhibition:

1. **10.1021/acs.jmedchem.6b01786** (J Med Chem 2017, dCTPase paper): one NUDT5 **IC50 > 100000 nM** counter-screen of a triazolothiadiazole. A genuine measured censored negative, indexed in both databases from the same single experiment.
2. **10.1002/cmdc.201800398** (ChemMedChem 2018, NVP-BHG712 regioisomers): two **Kinobead competitive pull-down apparent Kd** values (3616.84 and 7684.47 nM). These are lysate-competition binding numbers, not enzyme IC50, and ChEMBL re-expresses the same measurements as ED50 rows — stacking both would double-count.
3. **10.1016/j.ejmech.2024.116540**: a 2024 *review*, whose NUDT5 row restates Balikci. Not independent.
4. **EUbOPEN dataset CHEMBL5723035**: TH5427 IC50 29 nM, whose own provenance field points to the Page 2018 Nature Communications URL. Not an independent study.

So the honest count is: **zero Ki records, and zero records that are simultaneously new, human-species-confirmed, purified-enzyme, and primary-value-inspected.** Absence of such records in the repository was not evidence they exist publicly; they largely do not, at least within this bounded search.

## 3. Deduplication and identity

Deduplication was done on **recomputed canonical isomeric SMILES and InChIKey**, then on publication DOI — not on database row counts. BindingDB's own ligand page states its NUDT5 record was "Curated by ChEMBL"; the two databases are *mirrors* here, not corroborating sources. One asymmetry is worth recording: the BindingDB **REST** response preserves the `>` relation on the 2017 counter-screen, while the ligand **web page** renders it as "IC50: 1.00E+5nM" with no relation. Reading that page instead of the API would silently convert a censored inactive into an exact value.

On the known reference: PDB **5NWH** ligand **9CH** is `CN1c2c(n(c(n2)N3CCNCC3)Cc4nnc(o4)c5ccc(c(c5)Cl)Cl)C(=O)N(C1=O)C`, InChIKey `QXCXMVYVUHVFLP-UHFFFAOYSA-N`, matching the ChEMBL TH5427 graph exactly and confirming an **unsubstituted distal piperazine nitrogen**. The Page 2018 supplementary synthetic section independently names compound 28 (TH5427) as the **8-(piperazin-1-yl)** purinedione, while the adjacent compound 26 (TH5424) is the 4-ethylpiperazine. This is consistent with the already-identified N-methyl discrepancy in the reference graph and is reported here only as confirmation, not as a discovery.

## 4. Searched versus not searched

**Searched and retrieved in full:** ChEMBL target search, complete ChEMBL activity/assay/document sets for CHEMBL4105713, BindingDB `getLigandsByUniprots` for Q9UKK9 with no affinity cutoff, two BindingDB ligand pages, UniProt Q9UKK9, RCSB 5NWH entry/ligand/component, the SGC MRK-952 dossier with its CSV and four figure images, and a Europe PMC `TITLE_ABS:NUDT5 AND (INHIBITOR OR INHIBITORS)` query returning **58 hits, all 58 retrieved**.

**Inspected as primary full text:** Page 2018 (article + 45-page supplement), the 2026 degrader paper, the 2020 approved-drug repurposing paper, the 2022 coumarin paper, and the 2025 fluvoxamine record.

**Blocked (HTTP 403, publisher paywalls):** full text and supporting information for 10.1021/acs.jmedchem.6b01786 and 10.1002/cmdc.201800398, including the latter's `Table_1.xlsx` / `Table_2.xlsx` proteomics tables that would resolve the Kinobead measurements. Two PMC fetches returned a reCAPTCHA interstitial, worked around via Europe PMC XML or the publisher page. **These are access limits, not evidence of absence.**

**Deliberately not searched:** patents, PubChem BioAssay deposits, thesis repositories, non-English literature, any screening-library vendor data, and the remaining 533 of the 633 broad Europe PMC hits. Any of these may contain further records.

**Searched and rejected on the merits:** a large computational-only literature exists for NUDT5 (docking, MD, MM-GBSA, flavonoid and marine-natural-product studies). Where these report an "IC50", the inspected papers measure **MCF-7 cell viability by MTT**, not NUDT5 turnover — the 2020 repurposing paper (nomifensine, isoconazole and others), the 2022 coumarin paper (55.57 ± 0.7 µg/mL), and the 2025 fluvoxamine paper (53.86 ± 0.05 µM) are all cell-viability endpoints. Importing them as biochemical labels would be a category error and would quietly inflate any apparent "external validation". They are excluded.

## 5. A feasible retrospective evaluation design

The design must respect that only **6 unique graphs and 8 measurement rows** survived, and that **0** of them meet the strict bar. It is therefore a falsification test, not a benchmark.

**Prespecify, in writing, before computing any score** (otherwise the test is worthless at this sample size):

1. **Primary test — frozen ranking of a disjoint pair.** Freeze the existing model exactly as-is. Score MRK-952 and MRK-952-NC. The prespecified success criterion is a single bit: does the model rank MRK-952 above MRK-952-NC? Their measured separation is ~100-fold on IC50 and ~45-fold on SPR KD, so a model claiming potency discrimination should get this. With n = 1 pair the test can **falsify** a claim; passing it proves almost nothing (coin-flip probability 0.5). Report it as such.
2. **Secondary — censored-negative sanity check.** The 2017 triazolothiadiazole is a measured NUDT5 inactive at > 100 µM. Require only that it is not ranked above MRK-952. A `>` value supports a one-sided constraint, never a regression target.
3. **Endpoint hygiene.** Keep IC50, SPR KD, Kinobead apparent Kd, EC50 and cell-viability IC50 in **separate** tables; never pool or convert between them. Ki is empty — do not synthesize it.
4. **Novelty controls that must be reported alongside any result.** For every scored compound report (a) maximum Tanimoto to the training set, (b) whether the exact graph was ever previously inspected, and (c) nearest-neighbour identity. Already computed against the retrieved Balikci graphs; **it must be recomputed against the repository's own 45 structures, which were not inspected here.** A compound that is "externally sourced" but a near-neighbour of training chemistry is not an external test.
5. **Negative controls.** Rank the same compounds by molecular weight and by Tanimoto-to-TH5427 alone. If a trivial baseline reproduces the model's ordering, the model's ordering carries no additional evidence.
6. **Blocking conditions.** Do not report the probe-pair result as independent validation until the MRK-952 assay **species and construct** are confirmed from a primary source. Do not treat MRK-952-NC as an inactive decoy.

**What this design cannot do:** it cannot estimate AUC, enrichment, precision or calibration; it cannot establish generalization; and it cannot support any claim about acceptance, publication or prospective success. Any such claim would be unsupported by the evidence assembled here.

## 6. Reproducibility

Every HTTP request is logged in `snapshot_manifest.json` with exact URL, final URL, UTC timestamp, status, content type, byte count and SHA-256; all 41 requests are recorded, including the 9 non-200 responses (6 x HTTP 403 publisher paywalls, 2 x HTTP 500, and 1 x HTTP 202 reCAPTCHA interstitial). All 33 stored response bodies were re-hashed at ledger build time and matched. Re-running `build_ledger.py` regenerates every CSV/JSON from the stored bytes with no network access, and it asserts the structural identity checks rather than assuming them.
