# 0025: Binned mixed-type dependency forest

Status: implemented. Date: 2026-11-11.

The `learned_binned_forest` option learns at most one earlier parent per child,
within each group, for both categorical and numeric variables. For every
numeric variable, observed training values are sorted and cut at positions
`floor((n-1)*i/4)` for i = 1, 2, 3. A value belongs to the first bin whose
upper bound is at least that value, or the final bin; equal values stay
together. Duplicate bounds can yield empty bins. Missing and explicit
structural-absence tokens remain distinct raw states. Numeric states are
separate from raw states. Categorical variables retain raw states.

The existing empirical mutual-information rule selects an earlier parent using
these states, and sampling conditions on the selected parent's state while
returning the original empirical child value. A generated parent state is
always supported because it came from a training row and there is at most one
parent. Numeric bin bounds and selected-pair binned total variation distances
against the designated reference are reported. Development validation refits
the bins and graph on inner training and compares against inner validation.
These aggregate reports and boundaries may be sensitive.

Binning can hide within-bin dependencies and unstable high-cardinality
categories can dominate parent selection; unregularized selection can overfit.
Only one parent is learned, numeric values are resampled rather than newly
created, and selected-pair diagnostics do not assess the full joint distribution.
The supplied order does not establish temporal availability or causality.
This is a descriptive baseline, not a privacy, clinical or causal guarantee.
Alternatives include independent marginals, row bootstrap, and the categorical
and mixed-child forests; an independent final assessment remains necessary.
