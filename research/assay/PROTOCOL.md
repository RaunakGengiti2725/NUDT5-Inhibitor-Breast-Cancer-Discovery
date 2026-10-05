# Blinded NUDT5 dose-response contract, v1

**Future-input analysis contract and methods specification—not an experiment, qualified assay,
completed analysis lock, preregistration, independent validation or inhibitor discovery.**
This handoff contains specifications only. A dedicated subsequent stage must implement the analyzer
and executable tests; no production analyzer, generator or test suite is delivered here. All tool
behavior below is a normative requirement, not a claim of shipped functionality.

No new physical samples, plates or biological outcomes exist in this deliverable. Future synthetic
curves must be explicitly **SOFTWARE TESTS ONLY**. Do not put them in the measured-source ledger,
biological Results, or an efficacy/power argument. Previously inspected Balikci and MRK outcomes
remain retrospective even if remeasured. Original CSVs and numerical JSON remain untouched.

## 1. Scope selected from methods, not from outcomes

Analyze normalized **NUDT5 ADP-ribose hydrolysis percent activity** from a qualified purified-enzyme
assay. Input is not fluorescence, luminescence, absorbance, binding affinity, target engagement,
ATP synthesis, cellular ATP or viability. Those endpoints require separate contracts. Normalization
and detector specificity must already be documented; the fitter cannot establish them. The model
is empirical, not a mechanism-of-inhibition model.

A curve belongs to exactly one blinded material ID, independent-experiment ID, run and plate.
Different plates are fitted separately even within a run; a split-plate dilution series is
unsupported. Do not pool technical wells as independent experiments, or combine curves across
conditions. Replicate runs from the same experimental preparation retain the same independent
experiment ID. A submitted ID is a declaration, not software verification of independence.
Across-run meta-analysis and biological population estimates are out of scope.

### Primary-method anchors (published examples, not recommended new lab settings)

The exact inspected sections, download URLs/times, byte hashes and limitations are in
`references.json`. Failed HTTP-200 challenge pages are explicitly not source evidence.

| Source and inspected location | Relevant observed method | Consequence for this contract |
|---|---|---|
| Page et al. 2018, DOI 10.1038/s41467-017-02293-7; Methods **Small molecule screening campaign** (Sec24), **In vitro analysis of substrate hydrolysis** (Sec16), supplement **Enzymatic assay applied for screening and hit confirmation**, PDF pp. 22–24 | Malachite-green coupled phosphate detection. Campaign 1: 1.5 nM NUDT5, 50 µM ADPr, 5 U/mL calf intestinal alkaline phosphatase; campaign 2 uses different buffer/enzyme/substrate/coupler concentrations. Published inhibitor dilution range 100 µM–1.7 nM, threefold; duplicate measurements confirmed in two independent experiments. Vehicle and no-enzyme controls anchor inhibition. | Technical duplicates are not independent n. Campaigns and general substrate assay differ; do not silently mix them. Alkaline-phosphatase/coupling and optical interference require checks. |
| Page 2018, Sec16 HPLC paragraph; Sec31 **Biochemical conversion of ADPR to ATP by NUDT5** | HPLC checks substrate/AMP turnover; ATP-production experiment is separate. | Page is **not** cited here as an AMP-Glo protocol. HPLC turnover evidence does not prove every published potency was independently measured by HPLC. Hydrolysis inhibition is not ATP-production inhibition. |
| Balıkçı et al. 2024, DOI 10.1021/acs.jmedchem.4c00072; Experimental Section **Protein Expression and Purification** (sec4.2), **Catalytic Assays** (sec4.3); Table 1 title and footnotes a/b | NUDT5 construct residues 1–208 (NP_054861); AMP-Glo, 1 nM enzyme, 10 µM ADPr, 1% DMSO, 2 µL reactions, NUDT5 20 min at room temperature; NUDT14 1 h. Vehicle = 100%; 500 nM TH5427 = 0% activity in that publication. Triplicate sets; mean ± SD of two independent biological replicates as reported. Table footnote defines NA as IC50 >50 µM. | Coupled luminescence needs detector-only controls. Published TH5427 anchor is not a universal full-inhibition standard. Do not convert reported >50 µM into 50 µM or no activity at every exposure. Reaction-time differences prevent a thermodynamic selectivity claim. |
| GraphPad **Relative vs. absolute IC50**, relative/absolute definitions; **Incomplete dose-respone curves**, Prism 11 guide | Relative midpoint depends on fitted top/bottom; absolute crossing depends on control-defined scale. A converged fit with incomplete plateaus can be meaningless. | Free top/bottom; retain endpoint definitions; require sampled plateaus and bracketed midpoint; refuse unsupported estimates rather than silently fixing asymptotes. |
| Auld & Inglese, **Interferences with Luciferase Reporter Enzymes**, Assay Guidance Manual, 2018 chapter version; artifact definition, orthogonal/counterscreen roles, mitigation sections | Detection-only/product-spike counterscreens and independently interpretable orthogonal assays distinguish assay artifacts from target biology. | Negative interference screens alone do not validate target inhibition; changed signal alone does not establish it. Current NCBI pages were inaccessible behind challenges; author/publisher-attributed 2018 manual from an Ohio State mirror was inspected, not asserted identical to today's chapter. |

The papers do **not** fully specify free/fixed asymptotes, weighting, all dose grids, numerical
acceptance rules or whether each reported IC50 was relative or absolute. Our explicitly stated
4PL/equal-dose-weight analysis is a new conservative software contract, not a claim to reproduce
unstated publication fits. Eight doses, solver bounds and conditioning rules below are declared
software support restrictions; they are not laboratory thresholds extracted from these papers.

## 2. Input package and blinded identity

`manifest.schema.json` describes the JSON shape; the implementation must also enforce the stricter
cross-record rules below. JSON Schema alone is insufficient. `contract.json` is the machine-readable numerical/CSV contract. Paths are relative to the
manifest directory, cannot escape it (including symlinks), and every referenced artifact must
match its SHA-256. Duplicate JSON keys, nonfinite values and unknown manifest fields fail closed.

- `origin`: exactly `simulated` or `measured`, consistent across manifest, all artifacts and all
  observation rows. `simulated` requires `phase=software_test` and software-test-only QC/lock roles.
  Measured inputs require `pilot` or `validation`. The tool checks declarations and bytes, not
  forensic authenticity: a human custodian must prevent relabeling synthetic content as measured.
- `endpoint`: exactly `biochemical_ADPr_hydrolysis_percent_activity`; target name `NUDT5`, species,
  construct ID and hash-linked construct artifact. Unknown species/construct stays `null`, and
  affected curves abstain. Do not import a crystal construct or MRK dossier assumption as the lab
  construct. Sequence boundaries, tag/cleavage, mutations, lot and analytical verification belong
  in the construct/material record.
- `compounds`: opaque `BLD-` plus at least four uppercase alphanumerics, with `candidate`,
  `known_control` or (only simulated) `synthetic_test` role. No structures or chemical names go to
  the blinded analyst. Controls retain their control role, not novelty credit.
- A separate custodian retains the physical-material-to-ID key, lot/purity/solubility records,
  salts, isotopes and stereochemical certainty. Full isomeric/salt identity, neutral fragment
  parent and canonical tautomer are **different identity levels**. Do not merge them. The tool
  never strips salts, canonicalizes stereochemistry, converts mass concentration using guessed
  molecular weight, or equates two blinded samples.
- `plates` binds each plate to one run, independent experiment, condition artifact, normalization
  ID/document and QC document/status. A run cannot belong to different experiment IDs.
- `design` enumerates every planned curve's positive molar concentrations and technical replicate
  IDs. Expected records are their Cartesian product. Omitted wells/whole curves are errors: add
  explicit `absent`, `failed` or `untested` rows. A technical ID may recur at another concentration,
  but `(compound,plate,molar dose,technical ID)` is unique. Plate-well, observation-ID and
  `(raw-source byte hash, source row)` identities are also unique.
- `artifacts` registers `id,path,sha256,role,origin`. Roles: `construct`, `conditions`,
  `qualification`, `normalization`, `raw_controls`, `raw_observations`, `plate_qc`, `analysis_lock`.
  Immutable raw sources include acquisition/missingness records. `source_row` is a positive 1-based
  **record index under the documented raw-source export convention**, not assumed text line number.
  The tool verifies bytes/references, not arbitrary vendor-file semantics or normalization arithmetic.
  The custodian's conversion review must verify those links and the normalized values.

### CSV (strict UTF-8, optional BOM)

Header/order is exactly `contract.json:csv_columns`; duplicate/extra/ragged headers/rows and
whitespace-padded cells fail. All rows retain original CSV string values in the report.

| Fields | Meaning/rules |
|---|---|
| `observation_id,origin,blinded_compound_id,target` | Unique observation identity and explicit endpoint origin; target always NUDT5. |
| `independent_experiment_id,run_id,plate_id,well_id,technical_replicate_id` | Match plate registry; expected technical identities unique, no pseudoreplication. |
| `concentration,concentration_unit` | Finite positive concentration; units exactly M/mM/uM/nM/pM. Decimal conversion before floating point; underflow/overflow refused. Zero-dose controls are in the normalization/raw-control artifact, **not** log-dose curve points. Unicode aliases and mass units intentionally refused until explicitly converted in provenance. |
| `response_percent_activity` | Observed finite response or censor limit; not clipped to [0,100]. Numerical support is strictly between −1e6 and +1e6 percent; values outside are refused, never clipped. This is not a biological acceptance interval. |
| `observation_status,response_relation,status_reason` | `observed` uses `=` and empty reason. `censored` uses `<,<=,>,>=`, numeric limit, nonblank reason. `failed,absent,untested` use blank response, `not_applicable` relation, nonblank reason. Unknown reason is not silently fabricated. |
| `normalization_id,source_id,source_row` | Match the plate, registered raw source and unambiguous source record. |

### Hash-linked metadata documents

**Conditions JSON** has exactly: `endpoint,species,construct_id,enzyme_concentration_M,substrate,
substrate_concentration_M,reaction_minutes,preincubation_minutes,temperature_C,pH,buffer_description,
cosolvent,cosolvent_percent,detergent_description,readout,protocol_id`. Substrate is `ADP-ribose`;
enzyme/substrate/time are positive; preincubation and cosolvent are nonnegative; pH is within 0–14,
cosolvent ≤100%. Supply actual temperature, reagent/coupler concentrations, enzyme/substrate lots,
free-Mg considerations, incubation/quench timing, detection protocol and reaction-volume details
in the linked condition description/protocol. These are not licensed to default to published values.
If not known, keep the condition artifact ID `null` and abstain pending qualification.

**Normalization JSON** has exactly: `plate_id,normalization_id,formula,signal_unit,blank_mean,
vehicle_mean,blank_control_ids,vehicle_control_ids,controls_artifact_id,conversion_description`.
Formula is `100*(signal-blank)/(vehicle-blank)`, with a positive denominator, disjoint explicit
vehicle and blank control IDs, immutable raw controls and reviewed conversion. The `blank` means
qualified no-reaction/background reference for this target/readout, not fitted Bottom and not
an assumed full inhibitor effect. This deliberately does not automatically adopt Balikci's
500 nM TH5427 zero anchor. An independently qualified alternative zero convention would require
an explicit versioned contract change, not quietly treating an inhibitor well as no-enzyme blank.

**Qualification and plate QC documents** are hashed human-reviewed evidence, not numbers this
program can authenticate. Cover G0/G1: material/construct identity, linear substrate conversion
and detector range, progress curves, solvent tolerance, per-plate vehicle/blank dispersion and
separation, reference-control reproducibility, aggregation/solubility, coupler/reporter controls,
product-spike/compound-only/no-enzyme interference, and orthogonal direct-product confirmation.
Record criteria, outcomes, reviewer, deviations and applicability to every condition/lot. Do not
claim binding by itself proves catalysis inhibition. `normalization_reviewed=true` and `qc_status`
are attestations, not software-derived approvals.

## 3. Pilot-derived policy and refusal before fitting

There are **no default measured laboratory acceptance thresholds**. Independently qualify and
justify in `qualification.policy`:

- `min_response_span_pp`: minimal interpretable activity span in percentage points;
- `max_fit_rms_pp`: maximal residual RMS compatible with assay precision/model adequacy;
- `plateau_tolerance_pp`: acceptable deviation of end-dose means from fitted plateaus;
- `min_technical_replicates`: required observed wells per concentration, integer ≥1.

Thresholds need rationale/evidence in the qualification artifact and must be locked before
validation data. They are assay-fit acceptance criteria, **not** meaningful biological effect
margins or powers. Unresolved precision/meaningful margins remain unresolved. Broad pilot tolerances
can make diagnostics permissive: the program does not turn them into validated identifiability.

Pending target/conditions/normalization/QC/qualification metadata gives retained-input refusal.
For measured data, require `qualification.status=qualified` and plate QC `pass`. A qualified pilot
may get descriptive fits without a validation lock, always `phase=pilot`; no pilot observation
enters later confirmatory outcomes. Measured validation also requires an intact pre-acquisition lock.

**Any non-observed planned well causes the whole curve to abstain** in v1. Retain the reason,
relation/limit and available technical summaries; no imputation, limit substitution, selective
complete-dose fitting or automatic retry/exclusion. This intentionally sacrifices available fits
rather than introduce an unqualified missingness/censoring estimator. Invalid file structure or
provenance rejects the input package; it does not delete or correct the immutable sources.

## 4. Estimands and deterministic fitting algorithm

For each complete qualified curve, sort positive doses and compute arithmetic technical means
within concentration. Every concentration gets equal least-squares weight regardless of well
count. Technical SD uses n−1 only when n≥2 and is descriptive; it is **not** a confidence interval
or independent-experiment variance. Shared control normalization creates correlated errors;
this contract deliberately permits no inferential coverage claim.

Let x = log10(c in M). Fit

`f(x) = B + A / (1 + 10**(h*(x-m)))`, with `A>0`, `h>0`, and `T=B+A`.

Free parameters are `B, ln(A), m, ln(h)`. No fixed 0/100 asymptotes, clipping, robust downweighting,
residual-driven exclusions, post-hoc dose removal or model selection. Use SciPy `least_squares`,
TRF, linear loss, Jacobian scaling, max 4000 evaluations, ftol/xtol/gtol 1e−10. Start from the last
two means for B and first-two minus last-two means for A; use nine fixed combinations of midpoint
fractions .25/.5/.75 across log-dose range and h=.5/1/2. Choose lowest SSE deterministically.
These are optimization starts, not tuned random seeds or alternate scientific models.

Numerical support (not laboratory biology): B in [−1e6,+1e6], A in [1e−6,1e6], h in [.05,20],
m in [lowest log-dose−2,highest log-dose+2]. Exploration outside observed dose range is for solver
stability only; output never extrapolates potency. These bounds can refuse genuine but unsupported
curves; they must not be widened based on desired results.

Refuse all fitted parameters/midpoints if any of the following applies:

1. Fewer than eight distinct doses (four free parameters plus redundant range sampling), too few
   technical observations, or span below independent pilot policy.
2. Nonnegative covariance of log-dose and mean response (flat/inverted), unsupported numeric start,
   any solver failure/nonfinite result, or best fit at a numerical bound.
3. Zero Jacobian column or column-normalized Jacobian condition number >1e6/nonfinite.
4. Near-optimal fits (SSE difference ≤1e−8 + 1e−6×best SSE) disagree in midpoint by >.01 log10 M.
5. Residual RMS exceeds pilot limit; midpoint is outside the sampled range; fewer than two
   concentrations **and** two observed means on each side of the fitted midpoint in their respective
   x/y axes; or either of the lowest/highest two mean responses fails the plateau tolerance.

Failure gives explicit `reason`, `parameters=null`, and both concentration estimates `null`.
Conditioning and multistart checks are numerical warning screens, **not** proof of unique biological
potency or reliable precision. Partial/flat/ill-conditioned/inverted/undersampled states are not
converted to apparently precise values. Do not infer `IC50 > max dose` automatically from a flat
curve: failure of assay, interference, or an unattained plateau may be alternative explanations.

### Relative midpoint is not automatically absolute IC50

- `relative_midpoint_M = 10**m` gives `(T+B)/2`, halfway between the **fitted asymptotes**.
- `absolute_50_vehicle_M` solves **50% of the control-normalized vehicle activity**:
  `log10(C50) = m + log10((T-50)/(50-B))/h`.
- Only report the latter if `B < 50 < T`, observed dose means bracket 50, and the fitted crossing
  is strictly inside the tested dose range. Otherwise it is null with an explicit crossing state.
  Asymptotic/no sampled crossing is not a precise censor bound or universal non-inhibition claim.

A curve from 100% to 65% may have a supported relative midpoint at 82.5% activity and **no absolute
50% crossing**. A curve from 100% to 20% has its midpoint at 60%; its absolute crossing is at a
higher concentration. Both examples are algebra, not biological observations. “Apparent IC50” is
condition-specific to substrate/enzyme, construct, time, chemistry and free exposure. No Ki
conversion (including Cheng–Prusoff), binding inference, kinetics or clinical efficacy follows.

Output preserves full floating-point values for reproducibility, not claimed significant-figure
precision. No CI, p-value, between-experiment SD, efficacy ranking, power or sample-size estimate
is produced. Downstream presentations must label conditional point estimates and missing uncertainty.

## 5. Pilot-first blinded laboratory handoff and lock

1. **Custodian/materials, G0:** establish physical identities, purity, salts/stereochemistry and
   qualified concentration/exposure; resolve the actual species/construct, sequence, tag and lots.
   Assign opaque IDs, retain encrypted/unshared key and distinguish known controls. Analyst gets
   roles/opaque IDs only. Chemical identity and sample independence cannot be invented by software.
2. **Independent pilot, G1:** qualify assay/detection controls above and a dose range capturing
   plateaus without precipitation/aggregation/interference. Estimate technical and between-run
   variability from actual independent preparations. Choose policy, design and interpretable
   scope before validation; pilot exposure is documented, not called blinded validation.
3. **Feasibility decision:** if range, construct, QC or interpretation cannot be qualified, stop
   affected claims. No arbitrary efficacy, equivalence, superiority or sample-size margins are
   supplied here. Any future inferential study requires justified meaningful margins, pilot
   variance, independent-unit definition, multiplicity/stopping/precision rules and a separate
   analysis extension; this descriptive tool is not a substitute.
4. **G4 local lock:** freeze panel/roles, full custodial chemical/material identity, actual unit
   hierarchy, plate maps/randomization and known prior exposure, all doses/technical IDs, endpoint,
   normalization rule, protocol/conditions, policy, exclusions/missingness/solver failures and
   software/dependency bytes. Record UTC time **before first validation acquisition**, custodian,
   pilot exclusion rule and unblinding rule. No external registration/deposit is created here.
5. **Acquire blinded validation:** preserve immutable raw acquisition and control data, explicit
   absent/failed/censored rows, QC decisions made without compound identities, readout conversion
   trace and deviations. Hash exact files. QC failure is inconclusive, not biological falsification.
6. **Run/report/unblind:** run once under lock, retain every refusal and conditional per-plate
   result; verify provenance and only then allow custodial unblinding under the agreed rule.
   A post-lock change creates a new version and labeled exploratory analysis/new validation set,
   not an overwritten result or retroactively completed preregistration.

`lock` commits code, `contract.json` and repository `requirements.lock` hashes plus local UTC times.
The required analysis-lock JSON has exactly `locked_at_utc,plan_sha256,custodian,unblinding_rule,
independent_unit_definition,pilot_exclusion_rule,analysis_code_sha256,contract_sha256,
environment_lock_sha256`. Define a plan digest as SHA-256 of UTF-8, sorted-key, compact JSON (separators comma/colon,
ASCII escaping enabled, no nonfinite values) for the manifest's
schema/origin/phase/endpoint/target/compounds/design/qualification; construct and qualification-evidence
hashes; and plate→run/experiment/condition-hash/normalization-ID/formula mapping. This excludes future
raw observations and per-plate control/QC outcomes (not available before acquisition), and excludes
the lock itself (no circular hash). Outcome artifacts are separately verified on analysis input.
The lock artifact must reproduce the commitment and timestamp/code hashes. Text governance fields
must be nonblank. Human review must verify substantive plate randomization, custodial identity key,
pilot separation and protocol content; timestamps/hashes alone do not prove historical blinding.

## 6. Implementation handoff and acceptance

Implement separately using the repository's existing **Python 3.12 hash-locked environment** and
pinned NumPy 2.2.6 / SciPy 1.15.3; no new dependency is required by this design. Do not introduce a
second analyzer here. Suggested CLI arguments are `--manifest`, `--observations` and `--output`,
with the final entry-point name chosen by the implementation stage. Resolve code/contract paths
relative to the installed module or checkout, never the caller's current working directory.

Required order: parse and validate all bytes/metadata; reconcile every planned well; evaluate
qualification/lock prerequisites; aggregate technical replicates per curve; fit only eligible
curves; construct the retained-input report; publish atomically. Invalid input prevents fitting
and publication. Valid but unqualified/unsupported input produces a refusal report, not an error
silently converted into a potency. Implement the output states in `contract.json`.

Publish a single JSON result using the existing repository pattern: serialize with nonfinite
values forbidden, write a same-directory temporary file, flush and fsync it, atomically hardlink
to the new destination, and remove the staging file even on failure. Never replace existing
results or symlinks. The parent output directory must already exist. Exit 0 means **a report was
written**, possibly entirely refusals; exit 2 means invalid input/provenance or output failure.
Inspect per-curve states, not exit code alone. No figures or biological Results are required.

`stress_matrix.json` is the deterministic acceptance specification. Implement every case with
exact fixture mutations and independent assertions for preserved observations, grouping,
estimand identity and refusal states; do not just assert that a command ran. The fixture policy
is artificial software-test input, not measured laboratory defaults. Numeric equality assertions
use the declared tolerances; hashes bind exact input/provenance bytes, not cross-machine floating
point output. These tests do not establish statistical coverage, assay performance or error rates.

The downstream implementation must run both existing and new tests explicitly. The root
pytest/mypy configuration currently excludes `research/assay`; add its path when invoking checks
without changing root files in this handoff. Existing package builds do not package this directory.
The implementation stage must document its supported invocation rather than imply that the
current wheel provides an assay command. Record Ruff, strict type checks, full tests, compilation,
build and dependency-audit outcomes against the actual delivered implementation revision.

This contract chooses refusal over inferred censor bounds, residual-based exclusions or an
elaborate uncertainty model. G0/G1/G4 qualification and lock evidence remain human prerequisites.
It is ready for engineering implementation, **not** laboratory execution or biological validation.
