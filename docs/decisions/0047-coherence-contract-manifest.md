# Decision 0047: machine-validated study contract and protocol manifest

## Decision

Janus adds strict, study-agnostic Rust representations for a study-specific
coherence contract and a candidate-comparison protocol manifest. The contract
records variable meanings, timing statements, units, allowed states and
missingness semantics. The manifest records the candidate, preprocessing,
populations, seeds, software, source fingerprints, reviewer protocol version
and reservation status.

Validation checks completeness, duplicate variables and empty states. It does
not infer clinical relationships, repair records, establish independence, or
turn a draft into an approved clinical protocol. The fictional HeMoVal files
in `examples/` exercise the schema only; their data and outcomes are not
clinical evidence.

## Consequences

Future candidate reports can require a validated contract and manifest before
comparing aggregates. The current HeMoVal draft remains limited to two
variables and previously inspected development data. A genuinely independent
evaluation and any patient-level review remain separate gates.
