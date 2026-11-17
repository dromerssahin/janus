# 0036: Stratified fitted DAG sampling

The reusable stratified API fits the greedy or interaction-search binned DAG
independently in each caller-provided group. It requires an explicit allocation
for every fitted group and can apply the existing sparse failure or counted
backoff policy, returning rows and diagnostics per group. Group-derived seeds
isolate one group's draws from changes to another group's allocation. All
results are constructed in memory; if a group fails, the call returns an error
rather than a partial cohort. Existing total row limits and input validation
apply. The fit and rows can retain source values, so the caller must enforce
the data-use policy, exclude identifiers and reserve patient-level test data.

This provides orchestration, not evidence of group-specific model quality,
clinical validity, causal meaning, or privacy. The models' search and sparse
limitations remain as described in decisions 0033 and 0035.
