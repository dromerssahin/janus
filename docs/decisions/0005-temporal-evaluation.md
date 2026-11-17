# Temporal evaluation before fitting

Date: 2026-11-09

Status: accepted.

The independent empirical fit samples marginal values and one episode per
patient. It cannot reproduce trajectories or repeated episodes. A fictional
evaluation contract can reveal those discrepancies without assuming a temporal
generating mechanism or a causal effect.

Evaluate source-event timing, recording delays, episode structure and states in
declared index-relative windows first. Report denominators and limitations;
reject invalid recording order through existing validation. Observation end and
censoring are not encoded, so event spans do not measure complete follow-up.
Input fingerprints identify serialized inputs and do not prove independence.

An alternative is to fit a trajectory or causal model immediately. That would
require unsupported assumptions about measurement, censoring, dependencies and
causal mechanisms. Defer it until an explicit protocol and a measurable
acceptance criterion justify a model. Temporal distributions alone do not
establish clinical plausibility, privacy, causal or external validity.
