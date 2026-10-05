# Executable paired-evidence analysis

This implements the verified curation and analysis contract at input commit
`fe9f2b01246597e6dd5172184733d43c4e368970`. The original curation, its inventory,
all source snapshots, original CSVs and historical numerical JSON remain unchanged.
The parent workflow owns integration, packaging, the main README/manuscript and PR #1.
No nested sessions or workflows were started by this implementation node.

## Run from the existing locked Python 3.12 environment

Use the repository README's `uv` installation commands. No new dependency is needed.
Both input paths are mandatory. The source's sibling provenance/schema/projection
files must accompany it. `--repository` remaps immutable input paths from the original
provenance to another checkout; it does not permit substituting training graphs.

```sh
.venv/bin/python scripts/scripts/selectivity.py \
  --source research/selectivity/paired_evidence.json \
  --predictions research/results/transfer.json \
  --repository . \
  --output /absolute/path/to/a/new-analysis-directory

.venv/bin/python scripts/build_selectivity_figures.py \
  --input /absolute/path/to/a/new-analysis-directory/selectivity.json \
  --manifest /absolute/path/to/a/new-analysis-directory/selectivity-manifest.json \
  --output /absolute/path/to/a/new-figure-directory
```

The CLI validates this **complete verified 23-row release**, not arbitrary new assay
measurements. Its supplied closed schema deliberately requires 23 source rows. The
arithmetic, document and figure functions are separately exercised on software-only
zero/one-eligible-pair fixtures, both one-sided bound directions and equality boundaries.
Those fixtures never enter biological ledgers or committed analysis outputs.

Run from any working directory using absolute script/input paths. The integration now registers the `nudt5-selectivity` entry point and bundles its
module in the wheel. Supply explicit input paths and `--repository` for installed-wheel
use; research evidence is not bundled. See the root verification ledger for integrated checks.

## Artifacts and interpretation

- `research/results/selectivity.json`: complete source pharmacology and the two frozen
  score scenarios, recomputed ratios, explicit inclusion/exclusion reasons, target-level
  SDs, conditions, unresolved source disagreements and original retrieval provenance.
- `research/results/selectivity-manifest.json`: actual command/revision/dirty status,
  dependencies, source/input/code hashes and hashes of every analysis output.
- `research/selectivity/analysis/target_endpoints.csv`: all 46 cells, preserving source
  text whitespace, numeric SDs, replicate definitions, strict bounds and nulls.
- `research/selectivity/analysis/pharmacology.csv`: all 23 source graphs, known-source
  roles, both endpoint states, ratio/log ratio and historical diagnostic exclusion.
- `research/selectivity/analysis/diagnostic_scores.csv`: all 46 compound/scenario rows,
  six unchanged methods, source JSON pointers, availability, ranks and rank denominators.
  Eligible rows sort by historical/frozen Equal_mean within each separate scenario;
  excluded rows remain present and have no model-diagnostic rank.
- `research/selectivity/analysis/selectivity.md`: generated scientific summary and tables,
  including the full source cohort and separate eligible diagnostic tables.
- `research/figures/selectivity_historical.{png,pdf,svg}` and
  `research/figures/selectivity_reference_sensitivity.{png,pdf,svg}`: point ratios,
  strict bound arrows/open markers and a distinct double-censored strip, generated only
  from recorded JSON. All six methods remain in the table, not only plotted Equal_mean.
- `research/figures/selectivity_figures_manifest.json`: recorded input and rendering hashes.

The run-directory basenames in the analysis manifest remain unchanged when those files
are transported to these repository locations. The figure CLI accepts the transported
JSON and matching manifest explicitly and verifies their content hash. Neither manifest
is silently rewritten to imply that the historical prediction run occurred at a new
revision.

R is reported mean IC50(NUDT14)/reported mean IC50(NUDT5). A positive log10 R favors
lower **reported** NUDT5 IC50 under the source conditions, not biological or clinical
selectivity. A strict upper/lower bound is not an uncertainty interval. Double censoring
identifies neither R=1 nor a finite bound. Untested is not inactive. Source SDs remain
visible; no ratio uncertainty is estimated without raw paired replicates.

The six nonoverlap paired compounds comprise three point ratios, one strict upper bound
and two double-censored non-estimable ratios. The full source tables retain 10/11 despite
training overlap and retain every untested row. Reference sensitivity adds an overlap
flag for 6 (both endpoints untested), not extra paired observations. No new training,
refitting, seed selection, threshold, significance test or selective cohort tuning occurs.

## Failure behavior and determinism

JSON duplicate keys, nonfinite numbers/exponent overflow, unknown release fields,
CSV duplicate/empty/wrong headers, ragged records, malformed reported numbers, wrong
target/endpoint/unit, numeric values for untested/censored endpoints, identity conflicts,
duplicate IDs/graphs, altered frozen methods/scores, incomplete joins and incompatible
input hashes cause refusal before artifact publication. A failed promised score join
does not become zero, a refit, a dropped pharmacology row or a partially released report;
the original complete pharmacology ledger remains intact.

Each destination must be new or empty. All bytes are prepared and validated first,
flushed and fsynced in same-filesystem staging, then linked without replacement. The
completion manifest is linked last; consumers must not regard a directory without that
manifest as a completed run. Handled write failures roll back only this writer's files
and clean staging. An uncatchable process/OS kill may leave a manifest-less incomplete
directory, which a rerun refuses rather than overwriting. No claim of a multi-file
filesystem transaction is made.

Numerical JSON/CSVs and figure bytes are deterministic for fixed inputs and locked
versions. Figure dates are removed and SVG IDs are deterministically salted. Run
manifests intentionally record the actual invocation/output path and Git dirty state,
so manifests from different commands need not be byte-identical.

## Scientific verification basis

The durable tests independently encode the Table 1 values read from archived
`jm4c00072_0008.jpg.gz` (`tbl1/fx2`, all eight rows and both IC50 columns), not an assumed
favorable model result. For example: 0.990/0.837 = 1.1827956989247312 for 1;
0.162/0.270 = 0.6 for 9; 3.72/50 gives the strict upper bound R<0.0744 for 13;
1.64/13.8 ≈ 0.11884058 for 14. All five source point ratios, including excluded
10/11, are tested. Source snapshots are hashed both compressed and decompressed.

Primary source: Balikci et al., *J Med Chem* 67, 7245–7259 (2024),
https://doi.org/10.1021/acs.jmedchem.4c00072. The inherited retrieval time, URL, exact
section/table and SHA-256 remain in `provenance.json` and generated results. The present
implementation reused those archived bytes; it did not claim a fresh source download.

The 20/60-minute assay-duration difference, unclear NUDT14-specific normalization,
biological/technical replication distinctions, SD precision differences and missing
paired raw replicates remain explicit. This is an auditable retrospective description,
not a selectivity benchmark, new experiment or experiment-ready biological qualification.

## Completed verification

The implementation was exercised with Python 3.12.15 and the existing hash-locked
requirements. The repository's 87 prior tests plus 118 selectivity tests pass (205 total).
The 16 full-suite warnings are upstream Matplotlib/pyparsing deprecations, not suppressed
checks. Commands run successfully:

```sh
.venv/bin/ruff check .
.venv/bin/ruff format --check .
.venv/bin/mypy
.venv/bin/python -m pytest -q
.venv/bin/python -m pytest -q tests/test_selectivity.py tests/test_selectivity_figures.py
.venv/bin/python -m build
.venv/bin/python -m compileall -q scripts tests
.venv/bin/pip-audit
```

The audit found no known third-party vulnerabilities and skipped the local package,
which is not on PyPI. Package build success is distinct from packaging the new module,
as noted above. No pre-commit hook configuration is present. Tests cover byte-for-byte
output determinism, stale hashes, source/score projection disagreement, strict units and
endpoint identity, nulls, negative/zero/nonfinite values, malformed numbers, duplicate keys
and headers, ragged CSV records, conflicting structures and IDs, both score scenarios,
rank ties, salt/stereochemistry identity, four-method Equal_mean preservation, nonempty
and symlink output refusal, simulated publication failure with rollback, complete source
retention, raw SDs, all censoring/equality cases and both bound-arrow directions.

### Compact claim/evidence ledger

| Output claim | Inspected evidence | Verdict / limit |
|---|---|---|
| Eight paired rows, all 16 primary IC50 cells | Balikci et al. (2024), DOI above, Table 1 value image `tbl1/fx2`, archived `jm4c00072_0008.jpg.gz`; author CSV lines 2–24 | Verified against the archived primary table and curated raw fields; missing tests are not imputed. |
| R for 9 is 0.6, while 13 has R<0.0744 | Table 1 9: 0.270±0.027 / 0.162±0.005 µM; 13: NA / 3.72±0.190 µM; `tbl1/t1fn2` defines NA | Verified by division in the specified NUDT14/NUDT5 orientation; 13 remains a strict upper bound. |
| Three point ratios, one upper bound, two non-estimable double-censored rows in the nonoverlap diagnostic | Curated exact/parent/tautomer identity joins, frozen source tables and recomputed full-cohort counts | Verified computationally; one already-inspected source, not independent prospective validation. |
| Source SD is not ratio uncertainty | Table 1 footnote `tbl1/t1fn1`, Methods `sec4.3`, Figure 1 caption; paired raw replicates/covariance unavailable | Reported SD retained; replication/aggregation ambiguities unresolved. No ratio CI estimated. |
| Assay comparability is limited | Catalytic Assays `sec4.3`, inherited `assay_conditions.json` and `source_disagreements.json` | Verified duration difference and ambiguous normalization wording; not evidence that the source assays failed. |
| Largest historical Equal_mean occurs for 9 | Immutable `research/results/transfer.json`, historical source panel, checked four-method arithmetic | Verified descriptive score ordering only; neither biochemical selectivity nor prospective prediction performance follows. |

These checks establish software/evidence consistency, not new biochemical experiments,
chemical novelty, therapeutic efficacy or clinical validation.
