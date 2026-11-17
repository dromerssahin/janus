# 0026: Aggregate exact-match diagnostics

Status: implemented. Date: 2026-11-11.

Opt-in `exact_match_diagnostics: true` requires development validation and
compares rows within each group. It reports the number of generated rows with
any exact match in inner training or validation, the number of validation rows
matching training, and generated rows beyond the first occurrence of each
distinct generated row. Counts and denominators refer to rows, not distinct
matched records. No rows or identifiers are emitted.

Equality covers every configured modeled value as a raw string, including
missing and structural-absence tokens. It excludes row keys and the group
column, although comparisons are group-specific. Equivalent numeric spellings
can differ, and changing the modeled columns changes the result. The validation
comparison provides context for naturally repeated records; row bootstrap
provides a positive-control generator. Aggregate counts may be sensitive.

This detects complete exact matches only. It does not detect near copies,
linkage, attribute inference, membership inference or disclosures from the
model or reports, and zero matches does not establish privacy or permit release.
These diagnostics are descriptive, not a threat model or privacy guarantee.
