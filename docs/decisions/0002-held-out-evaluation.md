# Held-out descriptive evaluation

Status: accepted. Date: 2026-11-09.

The caller supplies separate synthetic and held-out reference fixtures and a
versioned evaluation configuration. Both fixtures are validated and transformed
with the same study rules. Janus records the reference role but cannot establish
independence or prevent upstream leakage. The report contains aggregate counts
and descriptive categorical comparisons; it does not certify validity or privacy.

Observed proportions exclude missing and absent values. Joint comparisons are
opt-in and use complete pairs, avoiding inferred exposure or outcome roles.
Boolean categories are exhaustive; other categories are the observed union.
Numerical values are compared as literal categories, not continuous metrics.
Null denotes an unavailable proportion or difference when its denominator is
zero. Alternatives such as patient splitting, uncertainty intervals, continuous
metrics, and fitted models require separate designs. The report and manifest are
staged together; inputs and input fingerprints are not copied into the output.
