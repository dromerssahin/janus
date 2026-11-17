# 0035: Pair search for interaction-only dependencies

The opt-in `interaction_binned_dag` and in-memory `fit_interaction_dag` retain
the greedy DAG for children with a selected single parent. When none passes the
penalty, they examine all pairs of earlier variables jointly and select the pair
with the highest positive BIC-inspired gain over an independent child. Degrees
of freedom use the observed joint parent state count and observed child states.
Ties retain the first pair in variable order. Numeric states use training-derived
bins and missing states remain explicit. Sparse combinations use the existing
failure or counted drop-last-parent policy.

Both selected edges carry the same `joint_selection_gain_nats`, the pair's
penalized gain, not an individual edge gain. The first edge reports its marginal
mutual information and the second reports information conditional on the first.
No pair search occurs when any single parent was selected, so other interactions
can still be missed. This searches many candidate pairs without an explicit
multiple-search correction and may overfit; using observed state counts can
understate complexity in sparse samples. Binning, finite samples, variable order
and backoff affect results. The pair search increases computation. These learned
associations do not establish causation, clinical validity, or privacy.
