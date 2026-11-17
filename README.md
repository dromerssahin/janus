# Janus

Janus is a proposed research engine for study-agnostic synthetic clinical cohorts,
with an initial interest in neurosurgery and retrospective studies. A generator
(red team) will produce cohorts from explicit evidence, causal assumptions, and
constraints; an evaluator (blue team) will investigate fidelity, clinical
plausibility, privacy, forensic detectability, and the behavior of study-specific
scores and rules.

Janus aims for cohorts whose patient-variable combinations clinicians find
plausible, including unusual patients, missing observations and realistic
variation. Its governing [clinician-guided loop](docs/decisions/0043-clinician-guided-realism.md)
combines a study-specific coherence specification, candidate comparisons,
blind clinical review, cohort-level checks and a frozen reserved evaluation;
no single model or score establishes realism.

The core is intended to be Rust-native. Study protocols and specialty-specific
knowledge should be configurable extensions, not hard-coded assumptions. Real
patient-level data is not required to develop the initial software, but appropriate
held-out real data and clinical expertise will be needed to assess real-world
validity. Simulated interventions and counterfactuals are hypotheses conditional
on their assumptions, not clinical guidance.

## Status

An optional standalone experimental conditional neural generator is available
in [experimental/neural](experimental/neural/README.md). Its first exploratory
comparison favored the existing forests on reported marginal and pairwise
distances; see [decision 0044](docs/decisions/0044-experimental-neural-candidate.md).
The follow-up conditioning-policy diagnostic found no improvement from
empirical-frequency conditioning; see [decision 0045](docs/decisions/0045-neural-conditioning-diagnostic.md).
The default workspace does not require LibTorch. These technical results do not
establish clinical validity or privacy safety.

Optional categorical uncertainty uses independent patient-level percentile
bootstrap resampling, retaining every included episode for each sampled patient.
For example, substitute `examples/evaluation-uncertainty.yaml` for the evaluation
configuration in the held-out workflow. The report and manifest record the seed,
replicates, confidence level, method and limits. Intervals cover category
proportions and signed synthetic-minus-reference differences for marginals and
selected pairs. Categories come from the original cohorts; each replicate uses
its observed-value denominator. Any zero-denominator replicate makes the affected
interval unavailable, with the valid replicate count reported. Percentiles use
linear interpolation. Small cohorts may yield unstable intervals; these describe
sampling variability conditional on the inputs, not clinical, causal, privacy,
or external validity. Temporal and longitudinal metrics have no intervals.

The optional manually curated evidence registry records fictional assumptions,
their scope, limitations and explicit parameter links. It does not establish
clinical validity, causal effects, privacy or external validity. Registry
fingerprints identify serialized content, not independent review. Evidence
references require `--evidence-registry`; unreferenced legacy configurations
continue to work without one. Do not include restricted identifiers, credentials
or sensitive paths in a registry.

A fictional evidence-backed workflow (use fresh output paths):

```text
cargo run --offline -p janus-cli -- split examples/study-evidence.yaml examples/fixture.yaml examples/split-for-fit.yaml /tmp/janus-evidence-split --evidence-registry examples/evidence.yaml
cargo run --offline -p janus-cli -- fit examples/study-evidence.yaml /tmp/janus-evidence-split/development.yaml examples/fit.yaml /tmp/janus-evidence-fit --evidence-registry examples/evidence.yaml --split-manifest /tmp/janus-evidence-split/manifest.yaml
cargo run --offline -p janus-cli -- generate examples/study-evidence.yaml examples/generator-repeated-evidence.yaml /tmp/janus-evidence-generated --evidence-registry examples/evidence.yaml
cargo run --offline -p janus-cli -- evaluate examples/study-evidence.yaml /tmp/janus-evidence-generated/fixture.yaml /tmp/janus-evidence-split/holdout.yaml examples/evaluation-longitudinal.yaml /tmp/janus-evidence-eval --evidence-registry examples/evidence.yaml --split-manifest /tmp/janus-evidence-split/manifest.yaml --synthetic-manifest /tmp/janus-evidence-generated/manifest.yaml
```

Pass the split and synthetic manifests explicitly to check their fixture and
registry identities and carry generated assumptions into evaluation. A fitted
generator binds the registry fingerprint and requires the same registry during
generation. Manifests include referenced records, build and platform information;
these fields are informational and do not guarantee repeatability across
environments. Version 1 registry links use `study.follow_up_days`,
`study.variable.NAME`, `generator.coverage`, `generator.variable.NAME.value`,
and `generator.repeated.NAME.count` or `.timing`.

The repository includes a small Rust vertical slice. `janus-cli generate` samples
fictional events from a separate, versioned configuration of categorical
distributions and builds a cohort from them. Version 1 samples independently;
version 2 supports conditional tables over earlier variables. The generator is an
illustrative assumption, not a fitted model or evidence of clinical validity.
`janus-cli validate` checks
fictional YAML study and event fixture files; `janus-cli run` constructs an
episode cohort and writes `cohort.yaml`, `exclusions.yaml`, `summary.yaml`,
`integrity.yaml`, `diagnostics.yaml`, and `manifest.yaml`. The summary counts
source patients separately from episodes; derived value counts cover included
episodes. Diagnostics describe this fixture and do not establish clinical or
other external validity.

Included rows retain index-time eligibility values and their source event IDs
separately from values selected through the end of follow-up. Eligibility
equality compares the input strings after type validation.

Study schema version 2 assigns each variable a baseline, index-context, or
follow-up-outcome role. Cohort values follow those time windows and declared
observation bounds, and retain a source event ID per variable. Evaluation
compares those selected values and includes their source IDs; its temporal
windows instead describe source events and are not role-specific summaries.
Reports and source IDs can be sensitive. Version 1 studies retain their earlier
selection behavior. See `docs/ARCHITECTURE.md` for the contract.

Run with `cargo run -p janus-cli -- run examples/study.yaml examples/fixture.yaml /tmp/janus-output`.
Generate with `cargo run -p janus-cli -- generate examples/study.yaml examples/generator.yaml /tmp/janus-generated`.
For a fictional conditional association, use `examples/generator-conditional.yaml`
as the generator input. Its probabilities do not establish a treatment effect.
Choose a fresh output path; the study configuration, generated fixture, cohort artifacts, and manifest
are published together. Repeatability depends on software and environment, not
the seed alone.

Evaluate a generated fixture against the separately prepared fictional reference:

```text
cargo run --offline -p janus-cli -- evaluate examples/study.yaml /tmp/janus-generated/fixture.yaml examples/reference.yaml examples/evaluation.yaml /tmp/janus-evaluation
```

Create a reproducible patient-level development/holdout split and verify the holdout during evaluation:

```text
cargo run --offline -p janus-cli -- split examples/study.yaml examples/fixture.yaml examples/split-for-fit.yaml /tmp/janus-split
cargo run --offline -p janus-cli -- evaluate examples/study.yaml /tmp/janus-generated/fixture.yaml /tmp/janus-split/holdout.yaml examples/evaluation.yaml /tmp/janus-evaluation --split-manifest /tmp/janus-split/manifest.yaml
```

Splitting uses seeded FNV-1a followed by a SplitMix64 finalizer and keeps all episodes and events together. The finalizer changes assignments from earlier FNV-1a-only versions; retain existing split artifacts to reproduce earlier analyses. A matching manifest proves only fixture identity; it does not prove correct development use, prevent upstream leakage, or establish temporal, site, clinical, privacy, causal, or external validity. The holdout is an internal partition, not independent external validation.

The caller is responsible for holding reference patients out of model
development. Janus cannot verify independence or upstream leakage. The report
compares descriptive categorical frequencies, not clinical, privacy, causal,
forensic, or external validity. Outputs may themselves be sensitive and must
be handled under applicable data-use terms.

The fictional protocol in `examples/study.yaml` uses the exposure event as the
episode index and describes a 365-day follow-up. `examples/evaluation.yaml`
enables baseline and follow-up windows. Temporal output includes source-event
and recording-day distributions, recording delays, index-relative event days,
episodes per patient, event spans, last observed follow-up and window-specific
observed, missing and absent counts. Window boundaries are inclusive and values
must be recorded by the window end. Indexed episodes are the window denominator;
source-event distributions include excluded episodes. Version 2 fixtures may
declare episode-specific observation start/end days and a descriptive censoring
reason; version 1 fixtures retain unknown coverage. Derived follow-up values are
restricted to both periods, while out-of-period events remain diagnosed source
data. Temporal output reports complete, incomplete, and unknown indexed
episodes, capped observed duration, and censoring-reason counts. Last observed
event remains separate from observation coverage. These metadata do not
establish censoring mechanism or clinical validity. The manifest defines the
populations and records input fingerprints.
Validation rejects recording before the event before a report is written.

Fit the empirical baseline on development data, then generate and evaluate against the untouched holdout:

```text
cargo run --offline -p janus-cli -- fit examples/study.yaml /tmp/janus-split/development.yaml examples/fit.yaml /tmp/janus-fit
cargo run --offline -p janus-cli -- generate examples/study.yaml /tmp/janus-fit/generator.yaml /tmp/janus-synthetic
cargo run --offline -p janus-cli -- evaluate examples/study.yaml /tmp/janus-synthetic/fixture.yaml /tmp/janus-split/holdout.yaml examples/evaluation.yaml /tmp/janus-evaluation --split-manifest /tmp/janus-split/manifest.yaml
```

For a fictional conditional association, split `examples/fixture-conditional.yaml`
with `examples/split-for-fit.yaml`, fit the resulting development fixture with
`examples/fit-conditional.yaml`, then generate and evaluate the holdout with
`examples/evaluation-pairs.yaml` and the split manifest. Fit schema version 2
uses complete observed parent-child records and rejects unsupported parent
combinations; manifest counts describe missing, absent, and excluded records.
These tables represent associations, not causal effects or validated clinical
relationships. The small fictional fixture is only a workflow demonstration.

`examples/fixture-longitudinal.yaml` and `examples/evaluation-longitudinal.yaml`
demonstrate repeated-event evaluation. Version 4 reports event counts, timing,
first-event ordering, coverage, and source IDs. Zero observed events do not
prove absence; incomplete and unknown coverage remain separate. Outputs can
contain sensitive identifiers. See decision 0010 for denominators and limits.

Generator version 4 in `examples/generator-repeated.yaml` samples bounded
repeated follow-up histories from explicit categorical value and count
assumptions. A zero count emits no event; this is not evidence of absence when
coverage is unknown. Complete coverage is emitted only when explicitly
configured. Days are uniformly sampled over inclusive bounds and recording day
equals event day. This fictional model does not estimate trajectories,
missingness, censoring, or causal effects. See decision 0011.

For descriptive outcome timing, use the same fictional split with
`examples/fit-timing.yaml` and `examples/evaluation-timing.yaml`. Version 3
fits outcome value and inclusive day bin together only for episodes with
declared complete follow-up, then samples a day uniformly within the selected
bin and declares generated follow-up coverage. The fit manifest reports excluded
states and conditional table denominators. Coverage is an input assumption,
not evidence that censoring is independent or that timing is clinically valid.

Fitting counts observed follow-up values among included development episodes and records missing and absent values separately. The independent marginals are descriptive only: they have no causal interpretation, do not model temporal dependencies or repeated trajectories, and provide no privacy or clinical-validity guarantee. A fingerprint records fixture identity but does not prove the development data was used without leakage.

- [Contributor and agent guidance](AGENTS.md)
- [Current state and session handoff](docs/PROJECT_STATE.md)
- [Proposed architecture](docs/ARCHITECTURE.md)
- [Retrospective study model](docs/STUDY_MODEL.md)
- [Roadmap](docs/ROADMAP.md)
- [Decision records](docs/decisions/README.md)
Continuous evaluation is opt-in via `continuous.variables` in the evaluation
configuration. Selected integer and number variables report observed, missing,
absent counts, moments, interpolated quantiles, empirical CDF distance, and,
when uncertainty is configured, patient-level bootstrap intervals. These are
descriptive comparisons conditional on the supplied cohorts.

## Local data procedure

The CSV / Parquet / DuckDB event adapter is verified with fictional examples;
it has not been used to convert the Early Drain table into event records. The
separate flat descriptive workflow has been run on the user-provided approved,
deidentified cohort. Before using any dataset adapter, confirm governance and
data-use scope, validate mappings and outputs, and retain only permitted
artifacts. These steps do not constitute anonymization, clinical validity,
causal validity, privacy validation, or external validation. Janus does not
perform pseudonymization or key management. Reports and fingerprints may be
sensitive.

For an explicitly configured flat patient-level CSV, `describe-flat` emits only
aggregate categorical counts and numeric quantiles, development/holdout
summaries, a development-fitted empirical whole-row resampling cohort,
thresholded synthetic-to-holdout descriptive comparisons, and a reproducibility
manifest; it does not build event episodes or estimate treatment effects. The
input, config, and new output directory must resolve beneath the local ignored
`datasets/` directory. For example, from the repository root:

Flat input must contain one row per patient and a unique stable row key. Set
`row_key_column` to its column name; the legacy default is `no`. The key cannot
be selected as a modeled variable. `allocation_column` and its nonempty
`allocation_values` define one or more groups used for stratified fitting and
splitting; these groups need not be treatments. A single constant group is
supported. This does not support multiple rows per patient.

The optional flat configuration field `generator: independent_marginals` fits
one empirical distribution per selected variable and allocation group using
development patients, then samples those distributions independently to form
new combinations for aggregate comparison with holdout. Omitting the field uses
the existing `row_bootstrap` baseline. The marginal model preserves observed,
missing, and configured structural-absence tokens individually, but does not
learn relationships between variables or, without optional numeric perturbation, produce unseen individual values; it
can generate implausible combinations and offers no privacy guarantee. No
patient-level generated rows are written.

For a specified association, set `generator:
conditional_empirical` (the older `conditional_categorical` name is also
accepted) and `dependencies: [{parent: baseline_category,
child: outcome_category}]`, using configured variable names with the parent
before the child. A categorical or numeric child may have multiple categorical
parents, declared as separate dependencies. The model fits
empirical conditional values from development patients within each allocation
group, including missing tokens. For categorical children it reports joint
total variation distance for each declared pair against holdout. For numeric
children it reports conditional quantile differences and observed/missing counts
for each raw parent state. These reports include patient denominators. This is
descriptive association, not a causal effect; unsupported relationships and
implausible combinations remain possible. Distinct raw missing tokens are
distinct states in the pair comparison. If a sampled combination of parents
was not observed in development, the default `sparse_policy: fail` stops the
run without publishing a report. An explicit `sparse_policy: drop_last_parent`
instead drops parents in reverse dependency-list order until development support
exists. The report counts backoffs by group, child, and number of parents
dropped. Backoff assumes less specific relationships apply and can produce
unsupported combinations; it is not clinical validation. Sparse tables may
otherwise fail often.
The same `dependencies` list can be supplied with either baseline generator to
compare the same pairs against holdout without changing baseline sampling.
`generator: learned_categorical_forest` learns at most one earlier categorical
parent for each categorical variable within each development group, selecting
the strongest positive empirical mutual information. Variable order is supplied
by the study; it is not evidence of causation or temporal availability. Raw
missing tokens participate as states. Numeric variables remain independent.
The report records learned edges and compares their joint frequencies against
holdout. Small groups and high-cardinality fields can produce spurious edges;
learned edges and aggregate reports may themselves be sensitive.
`generator: learned_mixed_forest` additionally learns a categorical parent for
each numeric child. It scores parents by mutual information with four
development-derived quantile bins, keeping tied values together and treating
raw missing and structural-absence tokens as distinct states. It samples
actual development values conditional on the selected parent, not values from
the bins. Reports compare numeric quantiles and value-state counts by parent
state. Binning loses information and sparse groups can overfit; numeric parents
and numeric-to-numeric dependence are not learned. This is an exploratory
descriptive model, and comparisons require independently reserved evaluation.
`generator: learned_binned_forest` also considers earlier numeric variables as
parents. It learns numeric bins from development values, scores one-parent
associations using the binned states, and samples original values from rows
matching the sampled parent state. The report gives the development bin bounds
and selected-pair binned distances. This loses within-bin relationships, can
overfit, and without optional numeric perturbation creates no unseen numeric values. Selected-pair distances alone
are not a full evaluation. The same model can be compared using the inner
development-validation split below.
`generator: regularized_binned_forest` uses the same bins and sampling but
penalizes each candidate edge for the number of observed parent and child
states. It selects an edge only when its empirical mutual information exceeds
the penalty, and reports both values. This is a model-selection heuristic,
not a significance test or a guarantee against overfitting. See decision 0027.
For an in-process workflow, Rust callers can fit
`janus::tabular::FlatFittedBinnedForest` from one group's modeled rows and
`FlatVariable` declarations, inspect learned edges, and call `sample(seed,
count)` repeatedly. `sample_interpolated` can also interpolate explicitly
selected continuous columns and returns change counts. The model borrows its
training rows and writes no files. The caller must exclude identifiers, keep training
and holdout patients separate, and honor the applicable data-use and output
policy. The model and samples may contain sensitive or copied information.
The library interface does not include the CLI's optional bounded numeric
smoothing. The [fictional Rust integration example](crates/janus/tests/flat_transient.rs)
shows fitting, sampling and interpolation. See decision 0030.
`FlatFittedStratifiedForest` fits multiple caller-partitioned groups and samples
an explicit count for each group in one call, returning rows keyed by group.
Every fitted group must appear in the requested allocation; zero counts are
allowed. Group-derived random streams keep a group's sample unchanged when
other requested group sizes change. This interface also stays in memory and inherits
the same data-use and validation limitations. See decision 0031.
For a numeric variable that a study explicitly treats as continuous, optional
`numeric_smoothing: [{variable: measurement, bandwidth: 0.5, minimum: 0.0,
maximum: 10.0}]` perturbs each sampled observed value within the specified
range and bandwidth, in that variable's units. These numbers are fictional
examples, not clinical limits. Training observations outside the declared
range cause an error; missing and structural-absence values are unchanged.
For a binned model the perturbed value remains in its original fitted bin.
Reports count changed and unchanged values. Smoothing may change dependencies
and create implausible values; it is unsuitable for discrete counts or scores
unless an appropriate discrete model is used. It does not provide privacy. See
decision 0028.
For binned forests, `numeric_interpolation: [measurement]` instead uses the
nearest distinct training values within the sampled parent state and the
variable's fitted bin. It samples between the original value and one of those
neighbors, without extrapolating; values with no usable neighbor stay unchanged.
Only explicitly selected continuous variables are interpolated. This is an
empirical modeling assumption, can alter relationships, and does not establish
clinical validity or privacy. Reports count changed and unchanged values. See
decision 0029.
For categorical or numeric variables with explicitly adjudicated structural
absence tokens, use `missing_policy: structural_tokens` and list them in
`structural_absence_values`. The older `categorical_tokens` policy name is also
accepted. Other blank and `NA` values remain missing.
For a categorical child with multiple parents the report also compares the full
parent-and-child joint raw-token distribution with holdout. Multi-parent numeric
relationships are currently evaluated one parent at a time. These metrics are
point estimates and do not establish privacy or clinical validity.
Set `generator_seed` to vary generated draws while keeping `split_seed` and the
patient partition fixed. When omitted, it defaults to `split_seed` for backward
compatibility. The effective generator seed is recorded in the report; repeated
draws reflect simulation variability, not uncertainty about the holdout sample.
Predeclare the seeds and comparisons. Repeated inspection of one holdout can
influence model choices, so reserve separate data for a final assessment.

Set `development_validation_proportion` to a value strictly between zero and
one to split each outer development group into inner training and validation
patients. Each candidate is refitted on inner training and evaluated against
the same inner validation patients. The `development_validation` section
contains group counts, generated and validation summaries, learned edges,
backoffs, and comparisons for explicitly configured dependencies; use the same
split, dependencies and seeds across candidates. The inner split and generator
use domain-separated seeds recorded in that section. Empty inner partitions
are rejected. One split and draw do not measure model-selection uncertainty.
Set `development_validation_seed` to vary the inner split while keeping the
outer `split_seed` fixed; omitting it retains the derived default. Use
`generator_seed` to vary draws separately. The explicit inner seed requires a
validation proportion.
To compare a predeclared numeric relationship, add `numeric_associations:
[{parent: first_numeric, child: second_numeric}]`. The inner report gives the
generated and validation Spearman rank correlations, complete-pair counts and
absolute difference for each group. Ties use average ranks; missing and
structurally absent values are excluded. Correlations are unavailable for fewer
than two complete pairs or a constant variable. This descriptive metric captures
monotonic association, not causal direction or the full joint distribution.
To measure a simple downstream prediction task, declare for example:

```yaml
utility_tasks:
  - name: fictional_prediction
    predictor: baseline_category
    additional_predictors: [another_baseline_category]
    target: later_outcome
    positive_values: ["yes"]
    negative_values: ["no"]
    alpha: 1.0
    predictor_available_before_target: true
```

All predictors and the target must be distinct categorical variables. The
timing declaration applies to every predictor and is the study owner's
responsibility; flat rows cannot verify it. Within each group, a smoothed binary conditional
frequency model is trained separately on inner real training and generated
rows, then scored on the same inner validation patients using Brier score
(lower is better). Each state uses `(positives + alpha) / (labeled + 2*alpha)`;
each state is the full combination of predictor tokens. Unseen combinations
fall back to the smoothed overall training rate. Other target
values are excluded and counted; missing predictor tokens are separate states.
Scores are unavailable without labeled training or validation rows. Reports
include counts and unseen-state totals, never predictions or fitted tables.
Generated training size equals validation size and may differ from real
training size. Joint-state counts can become sparse as predictors are added.
This baseline is exploratory, not clinical or
causal validation; repeated use of the same patients is not independent testing.

For a predeclared categorical joint relationship, set for example
`joint_families: [[baseline_a, baseline_b, later_outcome]]` with development
validation enabled. Each inner group reports the total variation distance over
the full raw-token combinations and both patient counts. Missing and structural
absence spellings are distinct states. All variables must be distinct declared
categorical columns; duplicate families are rejected regardless of order. This
is a descriptive measure, not a causal or privacy assessment. With many states
and small validation groups it can be unstable and near its maximum even when
models are similar. Comparisons should use the same prespecified family and
validation patients.

The optional `generator: learned_binned_dag` can learn a second earlier
parent for each variable. It starts with the penalized binned forest and greedily
adds a second parent only when its conditional information exceeds a complexity
penalty. Edges with `conditioned_on` report conditional information given the
first parent. A generated parent combination may lack training support: the
default fails, while `sparse_policy: drop_last_parent` drops the second parent
and reports backoff counts. Numeric interpolation leaves a value unchanged if
there is no matching full parent state. Selection depends on variable order,
bins and training data, can miss interactions without a strong first parent,
and does not establish clinical, causal or privacy validity. Rust callers can fit this model in memory with `FlatFittedBinnedForest::fit_dag`
and sample with `sample_with_policy`, which returns rows and backoff counts.
Fitted models and generated rows can contain sensitive source values.

The optional `generator: interaction_binned_dag` also searches pairs of
earlier variables jointly when no single parent passes the penalty. It can
capture patterns such as XOR that a greedy single-parent start misses. In
reports, `joint_selection_gain_nats` is the penalized gain for the pair, repeated
on both edges; the ordinary information fields give the first parent's mutual
information and the second parent's conditional information. This search can
be substantially slower and can overfit, particularly because it tests many
pairs and its state-count penalty does not correct for that search. It does not
reconsider pairs when a single parent was selected. The Rust equivalent is
`FlatFittedBinnedForest::fit_interaction_dag`.
For multiple caller-partitioned groups, `FlatFittedStratifiedForest::fit_dag`
and `fit_interaction_dag` fit each group separately. Its `sample_with_policy`
requires a count for every group and returns in-memory rows and backoff counts
per group. Group-derived seeds isolate draws when another group's allocation
changes. If any group fails, the call returns an error instead of a partial
cohort. Callers must exclude identifiers, maintain patient splits, and enforce
applicable data-use rules for fitted models and generated rows.
For explicitly continuous columns, `sample_interpolated_with_policy` combines
interpolation and sparse-state handling in one call for either single-group or
stratified models, returning both kinds of counts. Interpolation uses the full
fitted parent state; after a backoff with no full-state support, the value is
left unchanged. The generated draws themselves are unaffected by interpolation.

Set `exact_match_diagnostics: true` with development validation to count
generated rows matching an inner training or validation row, validation rows
matching training, and duplicate generated rows, within each group. Equality
uses every modeled raw value, including distinct missing tokens; the row key and
group column are not compared. Counts use rows, not unique records. These
sensitive aggregate diagnostics detect exact copying only: zero matches does
not rule out near copies, membership inference or disclosure risk, and is not
a privacy guarantee.
The regular report still contains outer-holdout results and whole-source
summaries: do not inspect those sections when reserving that holdout for final
evaluation. The tool cannot verify previous holdout use or independence.

```text
cargo run --offline -p janus-cli -- describe-flat datasets/ed.csv datasets/early-drain-descriptive.yaml datasets/results/description-run
```

The local workflow runner checks aggregate denominator reconciliation and byte-
identical replay of its report and manifest. Thresholds are workflow diagnostics,
not clinical acceptance criteria. The configured patient-level bootstrap emits
percentile intervals and valid-replicate counts; intervals are unavailable if
any replicate has a zero observed denominator:

```text
scripts/run-flat-descriptive.sh datasets/ed.csv datasets/early-drain-descriptive.yaml datasets/results/description-run
```

Blank and `NA` are missing unless a field-specific structural-absence policy is
explicitly approved. Fingerprints are identity checks, not privacy controls.
An explicitly configured structural-absence token takes precedence over that
default. Flat missingness-rate comparisons divide missing counts by observed
plus missing counts, excluding structural absences; an empty denominator is
reported as unavailable.
The repository-local `datasets/` directory is not secure storage; use the
command only where local processing and output retention are approved.

The flat workflow does not replace the governance checklist. Keep the source,
dictionary, dataset-specific config, report, and manifest in approved storage;
record the approval scope and output-retention decision there. The generated
row bootstrap is empirical and descriptive, not a clinical or causal model.

The fictional adapter workflow uses a mapping containing the study,
column names, a pre-deidentification declaration, an explicit permitted-use
declaration and a description of that use:

```text
cargo run --offline -p janus-cli -- validate-csv examples/source.csv examples/mapping.yaml
cargo run --offline -p janus-cli -- import-csv examples/source.csv examples/mapping.yaml /tmp/janus-import-example
cargo run --offline -p janus-cli -- inspect /tmp/janus-import-example/data.parquet /tmp/janus-inspect-example
cargo run --offline -p janus-cli -- run examples/study.yaml /tmp/janus-import-example/data.parquet /tmp/janus-cohort-example
cargo run --offline -p janus-cli -- export-csv /tmp/janus-import-example/data.parquet /tmp/janus-export-example.csv
```

Use fresh output paths. Mapping values name CSV headers, with one row per source
event. Empty value and missing-reason cells represent no value in that column;
exactly one must be populated. To represent an observed empty string, map the
optional `value_present` column and set it to `true`; set it to `false` for a
missing value. Exports retain the mapped column names, including this column when mapped. If
the mapping cannot represent an existing empty string or observation metadata,
export uses canonical headers; map those headers when reimporting. An absent event has no row. Optional observation
columns must be mapped together and agree on every row of an episode. Parquet
retains original value spelling in addition to typed columns. Fingerprints are
identity checks, not cryptographic proofs or privacy guarantees. Inspection is
source-event analysis and does not replace study-specific cohort selection.
The current adapter materializes source events in memory during import and
validation; size inputs to available memory. Arrow, Parquet, DuckDB and software
version changes can affect serialized bytes or floating-point results.

The bundled DuckDB 1.2.2 Parquet extension needs a C++ compiler. On the tested
`x86_64-unknown-linux-gnu` target, workspace Cargo configuration includes the
standard `cstdint` header omitted by its bundled Thrift sources. This workaround
is scoped to that target and is not passed to MSVC targets. A preexisting
target-specific `CXXFLAGS_x86_64_unknown_linux_gnu` overrides the configured
value and must retain the workaround if needed. Other targets and compilers
require independent verification.
