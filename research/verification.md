# Release verification

Checked on 4 October 2026 with Python 3.12.15 and the hash-pinned environment.

| Command | Observed result |
|---|---|
| `.venv/bin/ruff check .` | Passed |
| `.venv/bin/ruff format --check .` | 12 files formatted |
| `.venv/bin/mypy --no-incremental` | No issues in 12 source files |
| `.venv/bin/python -m pytest -q` | 80 passed; 16 upstream Matplotlib/pyparsing deprecation warnings; none skipped |
| `.venv/bin/python -m compileall -q scripts tests` | Passed |
| `.venv/bin/python -m build` | sdist and wheel built |
| `.venv/bin/pip-audit` | No known vulnerabilities in installed third-party dependencies; local unpublished project cannot be audited against PyPI |
| `git diff --cached --check` | Passed after lossless transport/format normalization |

All three complete analyses were rerun from their declared inputs. Benchmark (99 label permutations and five seeds), controls and transfer JSON were byte-identical to the preceding results. The source-input newline normalization was separately checked for equal parsed CSV cells and identical transfer results. Installed-wheel audit and transfer commands with explicit inputs also matched source outputs byte-for-byte. The unchanged original CSVs were checked against Git. Compressed PDB reference files are tested against the original uncompressed SHA-256 hashes.

The 16-page generated PDF was inspected for page-boundary overflow and its abstract/source-comparison table visually checked. Plots were visually checked, and all source-threshold table entries are regression-tested against recorded JSON. Independent computational/scientific review and source limitations are preserved separately. These are software/evidence checks, not experimental validation, a prospective test or human peer review.
