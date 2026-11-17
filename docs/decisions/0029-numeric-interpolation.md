# 0029: Training-derived numeric interpolation

Status: implemented. Date: 2026-11-11.

A study can opt continuous variables into `numeric_interpolation` for either
binned forest. After an empirical row is sampled, the method finds distinct
observed training values in the same fitted child bin and, when a parent is
selected, the same sampled parent state. It chooses an adjacent lower or higher
value uniformly when both exist, then draws uniformly between that neighbor and
the sampled value. It does not extrapolate beyond the matched training values.
Missing and structural-absence states are unchanged, and the sampled child bin
is preserved so downstream fitted parent states remain consistent.

If no distinct usable neighbor exists, or no valid distinct floating-point
candidate is found after 16 attempts, the original value is retained. Aggregate
counts report observed candidates and changed and unchanged values. The
interpolation RNG is separate from empirical sampling and bounded smoothing.
A variable cannot use both interpolation and bounded smoothing. Both the
outer fit and inner validation use neighbors from their respective training
partitions only. No generated rows or identifiers are published.

This is a data-derived interpolation rule, not a validated continuous density.
It assumes the selected variables can vary continuously between recorded
values; it can change within-bin dependence and create implausible combinations.
It should not be used for discrete scores or counts without a suitable model.
No extrapolation does not establish clinical validity or privacy, and exact
match counts can be obscured by interpolation. Sparse parent states or bins
often force unchanged values. Alternatives are empirical resampling, explicitly
bounded smoothing, or a separately validated continuous model.
