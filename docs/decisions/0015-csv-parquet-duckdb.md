# 0015: CSV boundary, typed Parquet, and DuckDB

Status: accepted and implemented for fictional examples.

CSV is the human-facing boundary, with an explicit versioned YAML mapping and
study contract. Canonical datasets use typed Parquet columns and retain original
value spellings because the existing study model compares values textually.
A missing reason is a source event; absence is represented by no event. Episode
observation metadata is repeated on source rows and checked for consistency.
YAML fixtures remain supported for small fictional examples and compatibility.

DuckDB is an embedded analytical reader over Parquet, with fixed aggregate
queries. Rust cohort construction retains authority over eligibility and temporal
selection. YAML reports and manifests describe contracts and limitations. Parquet
is the exchange artifact; a DuckDB database is not required.

Alternatives considered include YAML for large datasets, which is cumbersome
for columnar analysis, and a required database file, which would couple the
exchange format to an engine. CSV alone would lose typed column information.
Local checks and identity fingerprints do not establish deidentification,
governance approval, clinical validity, privacy, causal validity, or external
validation. Data and reports may be sensitive. Determinism is conditional on the
software and execution environment.
