# Decision 0006: Declared Observation and Censoring

Fixture schema version 2 optionally declares one observation period per episode.
The start and end are source metadata; Janus never infers coverage from event
timestamps. Version 1 fixtures remain valid and report unknown coverage.

An end before the index, an end before the start, duplicate declarations, and
unknown episode references are rejected. Events outside a declaration remain
source events and are counted diagnostically, but cannot supply derived
follow-up values. Censoring reasons are descriptive metadata only and do not
establish mechanism, informativeness, or a causal adjustment.
