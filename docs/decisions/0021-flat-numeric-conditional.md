# 0021: Numeric children in flat conditional generation

Status: implemented. Date: 2026-11-11. Extends decisions 0017–0019.

The flat empirical conditional model accepts numeric as well as categorical
children of explicitly declared categorical parents. It samples a child's raw
development value, including missing tokens, among rows matching all sampled
parent states within the group. As with categorical children it does not create
unseen individual values; sparse policy and backoff reporting are unchanged.
The configuration accepts `conditional_empirical`; the legacy
`conditional_categorical` name remains an alias.

For each declared categorical-parent/numeric-child edge, evaluation reports
patient, observed and missing counts by raw parent state, and absolute q25,
median and q75 differences against holdout. Quantiles are linearly
interpolated; empty or non-finite comparisons are unavailable. These reports
can contain rare parent states and may be sensitive. Numeric multi-parent
relationships are evaluated one parent at a time, not as a full joint family.
The diagnostics are point estimates and have no causal, clinical, privacy or
external-validity interpretation. Conditional empirical sampling is a baseline,
not a continuous density model or evidence of realistic unseen values.
