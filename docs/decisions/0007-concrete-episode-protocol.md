# Decision 0007: Concrete fictional episode protocol

The example protocol defines an episode by `(patient_id, episode_id)` and uses
the exposure event at day 0 as its index. Baseline severity (integer), baseline
comorbidity (boolean), and treatment group (categorical index-time context) must
be event- and recording-available by the index when used for baseline summaries
or eligibility. A binary outcome is selected during the inclusive 365-day
follow-up window.

Fixture observations may declare coverage and a descriptive censoring reason.
Events outside the index-relative follow-up or declared observation period stay
in source diagnostics and cannot supply derived follow-up values. Missing,
absent, observed, and censored states remain distinct. Repeated episodes retain
their pair identity and are kept together by patient-level splits.

This milestone establishes software contracts and a reproducible fictional
workflow. The independent-marginal generator and fit are descriptive; causal
effects, clinical recommendations, imputation, privacy claims, and temporal
trajectory modeling remain out of scope.
