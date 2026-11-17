# 0023: Development-fitted mixed-child forest

Status: implemented. Date: 2026-11-11.

The categorical forest does not learn relationships involving numeric children.
The separate `learned_mixed_forest` option keeps its categorical behavior and
selects at most one earlier categorical parent for each numeric child, within
each group. To score parents it discretizes observed numeric development values
at sorted positions `floor((n-1)*i/4)` for i = 1, 2, 3; tied values stay in the
same bin. Missing and explicitly structural-absence tokens are separate raw
states. Mutual information and its development denominator are reported, as
for categorical edges. Conditional sampling uses the original empirical values,
including missing and absence states, not the bins. Holdout comparisons report
numeric quantiles and value-state counts per selected parent state.

This is a simple exploratory baseline. Quartile binning can hide relationships;
unregularized selection can overfit, particularly with small groups and many
parent states. Numeric parents, numeric-to-numeric dependence, new numeric
values, causal direction and temporal availability are not modeled. An ordered
categorical parent is a modeling assumption. Alternatives include explicitly
declared relationships, independent marginals and complete-row bootstrap;
more expressive continuous models require their own evaluation and assumptions.
Fitting does not use holdout rows. Repeated inspection of a holdout prevents its
use as an independent final assessment. No privacy or clinical validity follows.

Numeric conditional comparisons now count structural absence separately from
missing and observed values. Earlier comparisons with structural-absence tokens
should be regenerated; nonnumeric absence tokens could previously cause a
failure and numeric absence tokens could be counted as observations.
The validator accepts `missing_policy: structural_tokens` for categorical and
numeric variables with explicit absence tokens. The older `categorical_tokens`
name remains accepted. Numeric absence declarations were previously rejected
by the flat input validator despite being handled by internal diagnostics.
