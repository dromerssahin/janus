# Roadmap

The milestones are proposals and may change as evidence and clinical
collaboration clarify the first study. Each milestone should leave a runnable,
tested increment, with documentation reflecting what is actually implemented.

## 0. Scope and protocol

- Select a narrow retrospective research question with clinical collaborators.
- Define population, index, variables, exposure, comparator, outcome, follow-up,
  intended estimand, and plausible data sources. Identify unknowns explicitly.
- Check access terms and whether source variables support the question. No real
  dataset is needed for the initial software fixture.

Done when a fictional example study and its assumptions can be reviewed without
implying it is a validated clinical protocol.

The current concrete example uses exposure as the day-0 episode index, baseline
severity and comorbidity, index-time treatment context, and a binary outcome
within an inclusive 365-day window. It is descriptive and fictional; causal and
clinical claims remain out of scope.

## 1. Minimal vertical slice

- Create a small Rust workspace and CLI when implementation begins.
- Define versioned study and scenario inputs, basic temporal validation, a seeded
  illustrative generator, and a run manifest.
- Generate fictional records and a basic integrity report; preserve errors and
  document reproducibility limits.
- Test deterministic behavior under the stated conditions and invalid inputs.

Done when a fresh checkout can run the example and reproduce its documented
output under the stated environment, without real data or clinical claims.

Status: a versioned categorical generator and fictional examples are implemented.
Version 1 samples independent distributions; version 2 supports complete
conditional tables over earlier variables. Generated output retains the parsed study, fixture, cohort artifacts
and generator configuration in the manifest. Reproducibility is tested in the
same environment. Manifests record a lockfile fingerprint and build and platform
metadata; code revision and compiler flags are not recorded.

## 2. Retrospective cohort semantics

- Add source mapping and cohort construction over invented event fixtures.
- Implement eligibility, index and follow-up semantics, missingness, exclusions,
  and temporal leakage checks for the selected example.
- Report patient and episode counts and reasons for exclusion.

Done when boundary cases and patient-level splits are tested against the study
contract. Expand semantics only as concrete studies require them.

Status: implemented for the fictional event fixture contract, including seeded
patient-level splits that keep every event and episode together. The library and
CLI validate versioned study/fixture YAML, construct stable episode-level rows,
retain exclusions and source IDs, distinguish missing from absent values, and
emit inspectable cohort, exclusion, summary, integrity, diagnostics, and manifest artifacts. Equality
eligibility is the only supported expression; broader rule languages remain
deferred.

Cohort summaries and initial descriptive fixture diagnostics are implemented.
Extend diagnostics as concrete study contracts require, without treating fixture
results as validation against independent data.

## 3. Evidence and evaluation

Optional patient-level percentile bootstrap intervals cover categorical marginals
and selected pairs, with fixed category domains and explicit denominator failures.
 Timing and longitudinal uncertainty, real-data
assessment remain open. See decision 0013.

- Add manually reviewed, provenance-bearing evidence records before automated
  retrieval or extraction. Connect parameters to evidence and assumptions.
- Add distributional and temporal diagnostics with explicit reference datasets,
  sample sizes and uncertainty. Specify what cannot be evaluated without real data.
- Integrate an approved dataset only after checking its suitability and terms.

Done when an evaluation report distinguishes fictional demonstration from any
real-data assessment and records held-out partitions and known limitations.

Status: a first descriptive categorical evaluation accepts a caller-supplied
reference fixture and can verify it against a split manifest fingerprint, constructs both cohorts under the same study, and
reports marginal and optional explicitly selected joint frequencies. The example
is fictional. Independence and upstream leakage cannot be verified; the
fingerprint proves fixture identity only. Optional descriptive temporal evaluation
now reports event and recording timing, episode structure and window missingness.
Observation and censoring coverage is implemented descriptively for declared
version 2 fixture periods; temporal transitions and temporal uncertainty remain
open. Continuous descriptive comparisons are implemented as an opt-in numeric
evaluation with patient bootstrap intervals; real-data assessment remains open.
This milestone is not complete.

## 4. Research extensions

The version 1 empirical fit workflow is implemented and documented as a
descriptive baseline over included development episodes. It publishes generator
configuration and provenance metadata without causal, temporal, privacy, or
clinical-validity guarantees.

Introduce causal interventions, counterfactuals, domain-specific anatomy or
biomechanics, privacy attacks, forensic models, external validation, and optional
language interoperability only with explicit contracts and appropriate expertise.
Prioritize based on the chosen protocol and data availability; none is implied to
be validated by the earlier milestones.

## CSV / Parquet / DuckDB milestone (implemented)

Mapped CSV import, typed Parquet storage and fixture conversion, embedded DuckDB
aggregate inspection, and CSV export are implemented and tested on fictional
data. The adapter preserves missing versus absent values, original spellings,
observation metadata and provenance; validates mappings; records fingerprints and
data-use declarations; and retains YAML fixtures. Existing workflows accept
Parquet input. Real-data adapter validation, privacy attacks and external
validation remain future work. See decision 0015.
