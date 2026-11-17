# 0019: Explicit sparse backoff and family evaluation

Status: implemented. Date: 2026-11-11. Extends decisions 0017 and 0018.

Conditional flat generation defaults to failing when sampled parent states have
no joint development support. An opt-in `drop_last_parent` policy instead
removes the last configured parent for that child, repeating in reverse
dependency-list order until a supported subset exists. With no parents left,
the child's development marginal is used. Reports identify the policy and
count backoffs by allocation group, child and number of dropped parents.
Backoff assumes the resulting less-specific distribution is appropriate; it
does not establish support for the original combination or clinical plausibility.

For a child with multiple parents, report total variation distance between the
generated and holdout joint raw-token distributions of all parents and the
child, plus patient denominators. This complements edge-level comparisons:
matching all pairs does not imply matching the full family. Empty populations
produce unavailable distances. These are descriptive point estimates; sparse
cells, repeated holdout use and simulation variability limit interpretation.
Reports can be sensitive and do not establish privacy, causality or validity.
