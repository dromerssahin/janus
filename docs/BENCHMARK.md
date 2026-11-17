# Concrete descriptive benchmark

This repository's benchmark is a deterministic, fictional workflow for the
episode-indexed protocol in `examples/study.yaml`. An episode is a patient
episode indexed by the exposure event at day 0. Severity and comorbidity are
baseline values available by index; treatment group is index context; the
binary outcome is followed through inclusive day 365. Repeated episodes remain
grouped by patient. Event time and recording time are retained separately.

The populations are source rows, distinct patients, indexed episodes, eligible
episodes, complete-observation episodes, and patient-level development, holdout,
and reference populations. Every descriptive metric records its population and
denominator: value/missing/absent frequencies, numeric distributions and
quantiles, event and recording timing and delay, ordering, observation and
censoring coverage, repeated-event counts, and replay behavior of split, fit,
generate, and evaluate. A metric is unavailable when its denominator is zero or
its required observation declaration is absent; invalid input or integrity
failure is a benchmark failure. Threshold direction and limits are declared in
`examples/benchmark.yaml`.

Run the complete fictional benchmark with:

```text
scripts/run-benchmark.sh /tmp/janus-benchmark
```

The command validates inputs, creates a patient-level split, fits only on
development, generates synthetic data, evaluates against untouched holdout and
reference fixtures, and writes a compact report and manifest. It refuses an
existing output directory. Re-running in a fresh directory under the same
documented toolchain produces byte-identical artifacts. Perturbations used by
tests must fail their named timing, missingness, category, leakage, or
provenance check before publication. Reports contain summaries and fingerprints,
not raw records.

These metrics are descriptive diagnostics. They do not establish clinical
validity, treatment effects, causal validity, privacy, anonymity, independence,
or external validity.
