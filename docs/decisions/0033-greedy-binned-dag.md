# 0033: Greedy two-parent binned generator

The optional flat CLI generator starts with the penalized ordered binned forest,
then considers an additional earlier parent conditional on the first. It adds the
best positive conditional-information gain after a BIC-inspired penalty based on
the observed first-parent, candidate-parent and child state counts. Ties retain
the earlier candidate. The edge's `conditioned_on` field distinguishes this
conditional information from the first edge's mutual information. Numeric bins
and all selection use the current training partition only.

Generated combinations of two parents may have no training support. The default
fails without publishing output; the opt-in `drop_last_parent` policy backs off
to the first parent and counts each backoff. Interpolation requires the full
parent state and leaves unsupported values unchanged. Outputs remain aggregate.

A greedy first edge can miss pure interactions such as XOR. State-count penalties
are heuristic with data-derived bins, do not prevent overfitting, and do not
establish statistical significance. Empirical numeric samples, variable order,
sparse states and reuse of development patients limit the model. Associations
are not causal; this is not clinical or privacy validation. A full multivariate
model and independent assessment remain open. The in-memory fitted API also supports this model and returns sparse backoff
counts when explicitly requested. Its default sampling fails on unsupported
combinations; callers must enforce permissions and exclude identifiers.
