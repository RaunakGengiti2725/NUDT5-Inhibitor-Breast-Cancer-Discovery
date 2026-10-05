# Public local-model support: extraction summary

This is report extraction and limited direct map inspection, not independent density validation, rerefinement or new experimental validation.

|Site (label / auth)|RSCC|RSR|Report|Maps|
|---|---:|---:|---|---|
|8OTV C / A 301|0.952|0.076|exact identity match|EDS + difference|
|8OTV F / B 302|0.928|0.097|exact identity match|EDS + difference|
|8RIY C / AAA 301|0.931|0.094|exact identity match|EDS + difference|
|8RIY D / BBB 301|0.940|0.091|exact identity match|EDS + difference|

Original residue-conformer records retained: {'observed': 1564, 'partial_observed': 58, 'refused': 108}.
These are data records, not independent biological observations.

## Limits

- Retrospective public-data assessment; no new measurement or independent validation.
- Report-derived RSCC/RSR are not recomputed here and are not per-atom support scores.
- PDBe maps are precomputed model-dependent maps, not omit maps or new experimental data.
- Map/report software dates need not describe the same map calculation or weighting.
- Fixed slices and trilinear atom sampling are limited direct map inspection, not a full three-dimensional crystallographic model validation or rerefinement.
- Interpolation does not increase map resolution. Sample values are not probabilities, RSCC, RSR, coordinate errors, interaction energies or occupancy estimates.
- No inferential test across crystal copies; sites and chains are not independent n.
- No proximity-to-affinity, energy, causality, homology or selectivity inference.
- B factors are retained metadata, never converted to coordinate uncertainties.
- Unobserved/zero-occupancy atoms are not imputed. Null is not no-contact or zero density.
- No atom-directed redesign, hydrogen-bond proof or refutation of source experiments.
