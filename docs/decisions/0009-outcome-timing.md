# Descriptive outcome timing

Fit version 3 models one follow-up outcome as a joint categorical value and
inclusive index-relative day bin, optionally conditioned on earlier baseline or
index-context variables. Only included episodes with declared complete follow-up
and observed outcome and parents contribute. The manifest records exclusion
categories and each parent table's counts and denominator. Sparse combinations
are rejected; no imputation or censoring mechanism is assumed.

Generator version 3 uses an index at day zero, samples value and bin together,
and samples uniformly within the bin. It can declare generated follow-up
coverage. Evaluation version 3 compares value-and-bin counts among completely
observed included episodes, keeping unknown and incomplete coverage separate.
Coverage is an assumption, not evidence of clinical validity, causal effects,
privacy, or independent censoring. Repeated trajectories are not modeled.
