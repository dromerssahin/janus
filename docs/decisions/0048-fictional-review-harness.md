# Decision 0048: fictional aggregate-only review harness

The core provides an in-memory `ReviewPresentation` validator and
`aggregate_ratings` helper for fictional review fixtures. It enforces blinding,
allowed and prohibited fields, patient grouping, duplicate rejection, reserved
key exclusion and complete missing-value display. Ratings are reduced to
reviewer counts, disagreement counts, insufficient-information counts and
suppressed severity/reason cells; rows and individual ratings are not emitted
by the aggregate type.

This validates workflow mechanics before any real-record presentation. It does
not define clinical severity, judge plausibility, anonymize data, or authorize
reviewer access. The HeMoVal review remains a draft pending custodian and
clinician/study-researcher approval.
