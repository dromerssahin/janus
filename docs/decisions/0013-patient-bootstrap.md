# 0013: Patient-level categorical bootstrap

Status: accepted, 2026-11-10.

Bootstrap included patients independently in the synthetic and reference cohorts,
retaining all included episodes per patient. Estimate categorical proportions and
their signed synthetic-minus-reference differences with percentile intervals.
Sorted identifiers, a seeded SplitMix64 stream and rejection sampling make draws
repeatable in the same environment. Percentiles are linearly interpolated.
One zero observed denominator in any replicate makes the affected interval
unavailable; valid replicate counts are retained. This avoids silently omitting
failed draws. Simple independent-episode analytic intervals were rejected because
episodes from the same patient need not be independent.

These intervals characterize sampling variability conditional on supplied data
and model outputs. Small or nonrepresentative samples and upstream leakage remain
limitations. No clinical, causal, privacy or external-validity conclusion follows.
Timing and longitudinal uncertainty and multiple-comparison adjustment are deferred.
