# Proposed architecture

This is a design map, not a description of implemented components. The current
fictional milestone implements the study specification, fixture validation,
episode cohort construction, and local CLI portions described below. Add crates
only as their behavior and boundaries become necessary. The initial target is a
local, reproducible research workflow; distributed execution and a service are
future options, not prerequisites.

## Workflow

```text
Study protocol + data dictionary + evidence + model assumptions
                       |
             validate and version inputs
                       |
       optional permitted retrospective data adapter
                       |
       define cohort, index, exposure and follow-up
                       |
       fit/calibrate model on designated development data
                       |
       scenario -> generator -> synthetic cohort + lineage
                       |                    |
                       +------> evaluator <-+ held-out data, if permitted
                                   |
                       metrics, diagnostics, limitations, run manifest
```

The blue team consumes the generated cohort and its declared contract. Some
checks also need real reference data, independent clinical review, or a known
ground truth. Reports must identify which checks could not be performed. Synthetic
generation does not require a detector in the loop; adversarial optimization is
an optional later capability and must not replace independent evaluation.

For flat categorical cohorts, `evaluate_flat_external` accepts caller-supplied
in-memory training, generated and reference rows and returns aggregate fidelity
and utility metrics. The caller must establish source compatibility, patient
independence and permissions; the API cannot verify those claims. See decision
0039.

Decision 0043 places a clinician-guided loop around clinical assessment. For
clinical claims, clinicians and study researchers author a per-study clinical-
coherence specification alongside the study protocol. Exploratory technical
comparisons do not require clinician sign-off or a reserved reference. Candidate generators,
including the existing statistical baselines and possible neural models, fit
only designated development data and sample in memory under source terms.
Clinicians review a permitted, blinded mixture of real and synthetic development
or designated review records and record severity-rated findings; the evaluator
combines permitted aggregate review findings with cohort checks across seeds.
Review artifacts can expose patient information: under aggregate-only terms,
patient-level rows, predictions and fitted models remain in memory and only
permitted aggregate findings and provenance may be stored. The final reserved
reference for a clinical claim is excluded from iterative review and model
selection. Once the clinical assessment method and protocol are frozen, its
separate reference passes intake gates before evaluation. These review and
coherence components are proposed, not implemented.
Preserve seed, software and environment metadata and the replay limitations
described below; a seed alone does not guarantee cross-platform reproducibility.

## Logical components

| Component | Responsibility | Boundary |
| --- | --- | --- |
| Study specification | Eligibility, index, variables, exposures, outcomes, observation window and analysis intent | Declarative, versioned; no hidden disease-specific defaults |
| Clinical-coherence specification | Clinician- and researcher-authored meanings, timing, missingness, plausible relationships, constraints and uncertainty for a study | Version before model selection; distinguishes hard integrity from reviewable plausibility |
| Data adapters | Import permitted data and map it to a canonical representation | Source-specific code and access controls remain outside core study logic |
| Cohort construction | Apply temporal eligibility and derive study variables | Records exclusions and reasons; prevents future-information leakage |
| Evidence registry | Track citations, extracted claims, populations, uncertainty and review | A source is not a parameter until an explicit mapping is approved |
| Model and scenario | Describe distributions, causal graph, mechanisms, interventions and assumptions | Separate observed associations from intervention semantics |
| Constraints | Check schema integrity and configurable clinical or anatomical rules | Hard and soft constraints have explicit scope, severity and rationale |
| Generator | Sample patients and trajectories using a specified model and random stream | Return lineage, failures and sampling diagnostics alongside records |
| Evaluator | Run fidelity, validity, utility, privacy and forensic checks | Each metric declares prerequisites, reference data, uncertainty and limitations |
| Clinical review | Blind review of permitted mixed real and synthetic records, including severity-rated concerns | Source permissions and blinding govern access and artifacts; separate from final reserved evaluation |
| Provenance and storage | Capture manifests, artifacts and lineage with access controls | Do not expose protected identifiers or hashes without permission |
| CLI | Validate specifications, execute runs and inspect reports | Thin orchestration over library interfaces |

The implemented fixture contract is YAML schema version `1` or `2`: events carry
patient and episode identifiers, event and recording days, one typed variable
value or an explicit missing reason, and provenance. Derived values distinguish
`Observed`, `Missing`, and `Absent`. Eligibility currently supports explicit
equality rules only; unsupported expressions are rejected. Index availability
for eligibility requires event and recording time at or before index and, when
declared, observation coverage. This applies to version 1 studies as well as
version 2 studies; absent coverage declarations remain unknown. Included
rows retain those eligibility values and their source event IDs separately from
the latest values available through the inclusive follow-up end. Version 2 can
also declare episode-specific observation periods and descriptive censoring
reasons; absent declarations retain unknown coverage. The summary's
derived-value counts refer to the latter, not the separate eligibility values.
Observed booleans use `true` or `false`, integers are signed 64-bit values,
and numbers are finite 64-bit floating point values; equality compares the
original strings, so numeric spellings are not normalized. These are fictional
observations and do not establish causal claims.

Study schema version 2 requires a role for each variable: `baseline`,
`index_context`, or `follow_up_outcome`. The index must be index context.
Baseline events must occur and be recorded by index; context events occur and
are recorded at index; follow-up outcomes occur from index through the inclusive
follow-up end. Declared observation limits apply to event and recording times.
Eligibility may use baseline or index context and uses the same selected value
as the cohort. Cohort rows map variables to source event IDs, and evaluation
includes role and selected source IDs. These artifacts may contain sensitive
identifiers and require appropriate handling. Study version 1 retains its
previous selection semantics and does not accept roles.

The fixture run produces a summary of source events, distinct patients and
episodes, index coverage, included and excluded episodes, exclusion reasons,
and derived value states for included rows. Episodes without an index are
explicit exclusions. Descriptive diagnostic records carry a numerator,
denominator, status and limitation. They inspect timing, source references,
provenance, missingness and episode distribution within the fixture; input
validation rejects invalid timestamps and missing provenance before a run.
They are not clinical, privacy, causal or external validation. The manifest
lists all artifacts and the CLI stages them before publishing the output.

The version 1 configured generator is separate from observed fixtures and study
protocols. It independently samples explicitly specified categorical outcomes
and their event and recording days, producing one patient and one episode per
sample. Its manifest records the configuration and random stream, and the staged
output includes the parsed study configuration used for cohort construction. This is an
illustrative assumption, not a fitted clinical or causal model.

Generator version 2 permits earlier variables as parents and requires a
complete categorical table over their possible values. Tables express configured
associations and do not establish temporal causation or treatment effects.

The version 1 evaluator constructs both supplied fixtures with the same study
and reports included and excluded episodes and categorical value frequencies.
Proportions use observed values as their denominator; joint proportions use
episodes where both selected variables are observed. Missing and absent values
are counted separately. Boolean domains include both values; other domains are
the union of observed strings, including numeric values treated as categories.
Joint tables require explicit, distinct string or boolean variables and cover
the cross product of marginal domains. A zero denominator gives null proportions
and differences. The caller designates the reference as held out; independence
and leakage cannot be verified. Optional categorical uncertainty is described
below. Optional continuous evaluation selects numeric study variables and
reports observed/missing/absent states, descriptive moments, interpolated
quantiles, empirical CDF distance, and patient bootstrap intervals when
uncertainty is configured. Integer inputs parse as signed 64-bit values before
conversion to floating point for metrics. These descriptive comparisons do not
establish clinical, privacy, causal, forensic or external validity. Evaluation has
its own manifest and report and does not change existing run schemas.

The patient split operation uses an explicit versioned configuration and a
stable seeded FNV-1a assignment with a SplitMix64 finalizer on patient identifiers. All events and episodes
for a patient stay together. Its manifest records requested and actual patient
proportions, patient and episode counts, and a fingerprint of the holdout
fixture. Evaluation may verify that fingerprint before declaring holdout
provenance; this check does not establish how development data was used, prevent
upstream leakage, or establish temporal, site, clinical, privacy, causal, or
external validity. The split is an internal holdout, not independent external
validation.

The optional temporal evaluation configuration declares named inclusive windows
relative to each index event. Timing distributions cover source events, including
excluded episodes; window state counts use indexed source episodes and the latest
event in each window recorded by its end. Episode event spans and the last
available follow-up event describe recorded events, not actual observation or
censoring duration. When declared, coverage counts and capped duration are
reported separately; out-of-period source events are retained but cannot supply
derived follow-up values. Invalid recording order is rejected before evaluation. The
report and manifest state denominators and unperformed checks; fingerprints are
identity checks, not proof of independence. Temporal similarity does not validate
clinical plausibility, privacy, causality or external applicability. The current
independent fitted generator does not model trajectories or repeated episodes.

## Conceptual contracts

Optional categorical uncertainty independently resamples included patients in
each cohort, retaining each sampled patient's episodes. Sorted patient IDs and
seeded SplitMix64 rejection sampling fix the draws; percentile bounds interpolate
between sorted replicate values. Original category domains and observed-value
denominators are retained. Any zero-denominator replicate makes its corresponding
interval unavailable. The intervals are conditional descriptive sampling
uncertainty, not evidence of independence, clinical, privacy, causal or external
validity. Temporal and longitudinal metrics are outside the interval calculation.

The optional version 1 evidence registry is manually curated and uses strict
records with stable IDs, source type, context, rationale, limitations, review
status and explicit parameter links. Manifests snapshot referenced records and
identify the registry by its canonical YAML fingerprint. Fit binds that
fingerprint into its generated configuration. Explicit split and synthetic
manifests allow later commands to verify fixture and registry identity. A
fingerprint is an identity check, not proof of independent review, clinical or
causal validity, privacy, or external validation. Build toolchain, target,
lockfile fingerprint, OS and architecture are informational; compiler flags,
runtime environment and floating-point behavior can still affect results.

The version 1 fit operation estimates independent empirical categorical
marginals from observed follow-up values among included development episodes.
Missing and absent states are recorded separately and excluded from
probabilities. It emits the existing generator configuration plus a provenance
manifest. This descriptive baseline has no causal, temporal, privacy, or
clinical-validity claim; its fingerprint does not prove leakage-free use.

The separate aggregate-only flat workflow optionally fits independent empirical
column marginals within each allocation group on development rows and samples
new combinations for holdout comparison. Its default remains whole-row
resampling. Marginal sampling preserves individual value and missing-state
frequencies in expectation, but does not model cross-variable dependencies or
joint missingness and can generate implausible combinations. No synthetic rows
are published; neither model establishes privacy or clinical validity.

The flat workflow also supports explicitly configured multi-parent dependencies
with categorical parents and categorical or numeric children, fitted on
development rows within each allocation group. Parent
variables must precede their children in the configuration. It compares each
configured categorical pair's joint raw-token distribution against the holdout;
numeric children receive per-parent-state quantile and missingness comparisons.
An optional ordered mixed forest learns categorical parents for numeric children
using development-derived quantile bins, then samples empirical numeric values
within the selected parent state. It does not learn numeric parents or infer
causal or temporal direction. See decision 0023.
An additional binned forest can learn numeric parents as well as numeric
children, using development-derived numeric bins for fitting and conditioning.
It samples original empirical values and reports binned pair diagnostics. See
decision 0025.
The regularized binned alternative penalizes the number of fitted conditional
states when selecting an edge; see decision 0027.
An optional greedy binned DAG can add a penalized second earlier parent,
with explicit failure or counted parent backoff on unsupported combinations.
It can miss pure interactions and remains a descriptive model; see decision 0033.
An additional opt-in binned model searches pairs jointly when a child has no
selected single parent, allowing some pure interactions; the broader search
raises overfitting and runtime concerns. See decision 0035.
A reusable in-memory binned-forest API fits one caller-partitioned group and
draws empirical rows without writing files; see decision 0030.
Its stratified wrapper fits each caller-partitioned group, including the greedy
and interaction DAGs, and samples explicit group counts using group-derived
random streams. Policy-aware sampling returns per-group backoff counts and
fails the entire call on group errors; see decisions 0031 and 0036.
An optional study-configured bounded numeric smoothing step can produce unseen
continuous values without changing missing states or fitted parent bins. It
does not infer appropriate clinical ranges or establish privacy; see decision
0028.
An optional empirical interpolation method for binned forests uses neighboring
training values in the selected parent and child states; policy-aware in-memory
sampling also returns both interpolation and sparse-backoff counts. Unsupported
full parent states remain unchanged. See decisions 0029 and 0038.
The flat workflow can additionally split its development patients into inner
training and validation partitions for candidate comparison. This does not seal
the ordinary outer-holdout report sections. See decision 0024.
Predeclared categorical joint families can be compared on inner validation
patients using total variation across raw-token combinations; these metrics
can compare models on the same variables independently of learned edges.
See decision 0034.
Optional study-declared binary utility tasks compare smoothed joint categorical
predictor states trained on synthetic and real inner-training rows against the same
inner validation patients, reporting aggregate Brier scores and counts. Timing
is declared by the study and cannot be verified from flat rows; unequal training
sizes, sparse joint states and repeated development splits limit interpretation.
See decisions 0032 and 0037.
Optional inner-validation exact-match counts make complete modeled-row copying
visible without emitting rows; they are not a privacy assessment. See decision
0026.
These include denominators and do not imply causal direction. It does
not estimate dependencies beyond the selected pairs or uncertainty for their
generated-versus-holdout distances. Multi-parent categorical families receive an additional
joint-distance report. If generated parent values have no joint development
support, the default is to fail without output. The opt-in backoff policy drops
parents in reverse dependency-list order until support exists and records how
often each level was used; this makes an explicit extrapolation assumption.

An ordered categorical forest may instead select each categorical variable's
strongest earlier categorical parent by empirical mutual information on the
development partition, separately per group. The specified order is not a
causal or temporal guarantee, and small or high-cardinality samples can overfit.
Numeric variables remain independent. The model records learned edges and
compares their joint distributions against the designated holdout; those results
do not establish privacy, clinical plausibility or external validity.

Fit version 2 estimates selected conditional categorical tables. Version 3
fits one outcome's value and index-relative timing bin from included episodes
with declared complete follow-up, optionally conditional on baseline or index
context. Generator version 3 samples a day uniformly within the selected bin
and can declare complete observation. Evaluation version 3 compares outcome
value and bin counts with coverage states reported separately. These are
descriptive assumptions; trajectories, censoring mechanisms, causal effects,
privacy, and clinical validity are not established. See decision 0009.

Evaluation version 4 compares recorded repeated-event counts and timing and
first-event ordering within a declared window. Complete, incomplete, and
unknown observation coverage are distinct, and source IDs and missing reasons
are retained. Zero observed events do not establish absence. These reports may
be sensitive and do not generate or fit trajectories; see decision 0010.

Contracts below are intended behavior, not stable APIs or a commitment to a
serialization format. Identifiers and schemas should be versioned when defined.

- **StudySpec:** population and eligibility rules; unit of analysis; index and
  observation windows; variable definitions, units and time availability;
  exposures, outcomes, censoring and estimands; missing-data policy.
- **DatasetDescriptor:** source and version, permitted uses, schema mapping,
  access classification, cohort selection and data-quality summary. Keep actual
  credentials and protected data outside descriptors and version control.
- **EvidenceRecord:** bibliographic identifier, population and context, claim,
  estimate and uncertainty when available, extraction/review status and limits.
- **ModelSpec:** graph or mechanism, parameters and uncertainty, evidence links,
  assumptions, supported population and validation status.
- **ScenarioSpec:** references to study and model versions, sampling strategy,
  optional interventions and constraints, random seed and requested outputs.
- **RunManifest:** identifiers and versions of inputs, code, dependencies and
  environment; RNG algorithm and seed; permitted fingerprints; timestamps,
  execution status and artifact references. Sensitive paths and hashes may need
  redaction or restricted storage.
- **EvaluationReport:** cohort counts and exclusions, test definitions,
  estimates and uncertainty, data splits, failures, unperformed tests and
  limitations. Results must not imply independent validation without it.

## Proposed repository layout

```text
janus/
  AGENTS.md
  README.md
  docs/                 # architecture, study semantics, roadmap, decisions
  crates/               # future Rust libraries, introduced incrementally
  apps/                 # future CLI or service, when justified
  examples/             # fictional studies and deterministic fixtures
  schemas/              # versioned interchange contracts, once designed
  tests/                # cross-component tests, once behavior exists
```

Possible future library boundaries include domain/study types, cohort assembly,
modeling and sampling, constraints, evaluation, provenance, and adapters. Start
with the smallest workspace that supports a vertical slice; split crates only
when ownership or dependency boundaries are clear. Arrow, Parquet and embedded DuckDB support the tabular boundary described below.
Relational services, Python interoperability, biomechanical solvers, and neural
detectors remain candidates to evaluate against concrete needs.

## Reproducibility and security

Use stable, explicit random streams and record RNG implementation and version.
Define whether parallelism, platform changes, and floating-point behavior affect
repeatability; test the guarantee actually offered. Validate all external input.
Separate restricted reference data, outputs, and logs from shareable examples.
Run manifests and reports may themselves reveal sensitive information. Dataset
terms determine permissible access, derivatives, retention, and publication.

## CSV / Parquet / DuckDB boundary

The planned human-facing import and export boundary is CSV with a versioned
mapping. Typed Parquet is the durable event dataset, preserving original value
spellings, missing reasons, provenance, and episode observation metadata.
DuckDB supplies fixed aggregate analytical queries; Rust cohort construction
remains authoritative for temporal selection and eligibility. YAML fixtures
remain supported. See decision 0015. The adapter has been verified with fictional fixtures;
real-data validation and governance remain future work.
