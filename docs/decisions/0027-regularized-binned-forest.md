# 0027: Penalized binned dependency forest

Status: implemented. Date: 2026-11-11.

The unregularized forest selects a positive empirical association even when it
comes from sparse or high-cardinality states. `regularized_binned_forest`
uses the same training-derived bins and empirical sampling as decision 0025,
but scores each candidate parent as `I - (r-1)(c-1) ln(n)/(2n)`, where `I` is
the empirical mutual information in nats, `r` and `c` are the numbers of
observed parent and child states, and `n` is the training group size. Missing
and structural-absence tokens are states. It selects the earlier parent with
the largest positive score, with near ties resolved by configured order; a
child with no positive score remains independent. Reports include both the
mutual information and penalized gain. Inner validation refits everything on
inner training only.

The penalty is motivated by the difference in parameter counts for a
one-parent multinomial model versus independence. It is a heuristic here:
states and numeric bins are learned from the data, candidate parents are
searched, and sparse cells violate conditions often used for asymptotic
approximations. It is not a p-value, causal claim, privacy guarantee or proof
against overfitting. It can suppress real relationships and still select
spurious ones. Bins lose within-bin relationships; numeric values remain
empirical. Compare this option with unregularized and independent baselines
using predeclared development validation and separate final data.
