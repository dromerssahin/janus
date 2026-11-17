# 0030: Reusable in-memory fitted binned forest

Status: implemented. Date: 2026-11-11.

`FlatFittedBinnedForest` exposes the existing binned and penalized binned
fitting and sampling functions to Rust callers. It fits a single caller-supplied
group of modeled rows, validates row widths, variable declarations, category
domains, finite numeric values and trimmed input, and borrows its training rows
for its lifetime. Callers can inspect learned edges and generate up to one
million rows per call using an explicit seed. Identical training rows, variable
order, group, seed and implementation replay the same sample in the tested
environment. It does no file I/O and does not serialize the model or samples.

The caller must partition patients, exclude identifiers and enforce the study's
permitted use, storage and output rules. The API cannot prove that data are
deidentified, training and evaluation are independent, or generated rows are
safe to release. A fitted model borrows raw values, and samples can reproduce
training records. Edge metadata and samples can be sensitive. The public API
samples empirical values and can optionally interpolate explicitly selected
continuous columns using decision 0029; it does not expose the flat CLI's
bounded numeric smoothing. This descriptive
forest does not establish clinical, causal, privacy or external validity.

The EARLYDRAIN local contract permits aggregate reports and transient synthetic
rows, not row-level release or fitted-model files. The CLI therefore continues
to write aggregates only. Persistence or release of a fitted model needs an
applicable output policy; it is not implied by this interface.
