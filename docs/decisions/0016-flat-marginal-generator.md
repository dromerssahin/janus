# 0016: Flat empirical marginal generator

Status: implemented. Date: 2026-11-11.

The flat descriptive workflow supports an opt-in independent marginal model in
addition to its default complete-row bootstrap. It estimates each configured
column's empirical distribution within allocation groups using only development
rows, including missing and explicitly configured structural-absence tokens.
Generation samples columns independently and uses the holdout group sizes;
only aggregate comparisons are written. The configured split seed determines
the samples in the tested environment.

This simple fitted baseline can produce combinations not present in development,
unlike the row bootstrap. It cannot generate unseen individual values or learn
relationships, joint missingness, trajectories, or causal mechanisms. It can
produce implausible combinations and reproduce development rows. Neither model
provides privacy, clinical validity, or external validation. Holdout group sizes
are used for evaluation; they are not a general-purpose population model.

Conditional or richer generative models are future alternatives and require
explicit temporal and missing-data semantics, evaluation of joint behavior,
and appropriate privacy analysis. The independent model is an interpretable
baseline for that comparison, not a substitute for those capabilities.
