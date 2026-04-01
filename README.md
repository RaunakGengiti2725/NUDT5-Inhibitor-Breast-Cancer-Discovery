# Structure-Based and Machine Learning Discovery of NUDT5 Inhibitors

This repository contains code, data, and results for an in silico drug discovery pipeline targeting NUDT5 in ER+ breast cancer.

## Overview

We combined:
- Structure-based drug design
- Classical cheminformatics
- Supervised machine learning under low-data conditions

Pipeline components:
- Tanimoto similarity and ECFP4 fingerprints
- Glide SP/XP docking
- MM-GBSA rescoring
- Random forest classification

## Data

- Initial library: ~18,400 compounds
- Final hits: 10 structurally diverse candidates
- Drug-like properties:
  - Mean MW = 348 Da
  - Mean cLogP = 1.3
  - Mean TPSA = 107 Å²

## Repository Structure

- `data/` → compound libraries
- `results/` → final screened hits
- `scripts/` → pipeline code

## Reproducibility

All steps are documented and can be reproduced using the provided scripts and datasets.

## Citation

Preprint available on bioRxiv.
