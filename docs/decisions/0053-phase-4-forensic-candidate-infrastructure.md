# Decision 0053: Phase 4 forensic and candidate-comparison infrastructure

## Context

Phase 4 extends the research extensions (Roadmap §4) with production forensic
workflows and unified candidate-comparison infrastructure. Prior milestones
established bounded technical validity (decision 0052), study contracts and
protocol manifests (decision 0047), and an experimental standalone neural
candidate, but lacked multi-seed forensic orchestration, a unified generator
trait for in-memory candidate comparison, and an automated reproducibility audit
gate.

## Decision

Implement the Phase 4 forensic and candidate-comparison infrastructure across
the core library, CLI, and documentation:

- `crates/janus/src/forensic_workflow.rs` (1,068 lines): multi-seed forensic
  workflow orchestrator producing aggregate-only manifests with provenance,
  seeds, configuration, and environment metadata.
- `crates/janus/src/candidate.rs` (1,236 lines): unified generator trait and
  in-memory-only candidate-comparison harness enforcing identical comparison
  conditions without serializing patient or synthetic rows.
- `janus-cli audit-reproducibility`: CLI command that verifies replay
  reproducibility and catches platform, dependency, or version drift.
- `docs/FORENSIC_WORKFLOWS.md` and `docs/NEURAL_OPERATIONS.md`: operational
  documentation for forensic workflows and neural candidate operations.

## Consequences

- Enables parallel candidate evaluation under identical conditions (splits,
  seeds, sample sizes, and metrics) with in-memory-only enforcement.
- Aggregate manifests record provenance, seeds, software versions, and
  environment metadata without exposing patient rows.
- Reproducibility audit gates catch platform or version drift before
  comparisons are accepted.

## Limitations

- The experimental neural bridge still writes `samples.csv`, which is an
  aggregate-only violation documented for remediation in Phase 5.
- GPU operation and GPU reproducibility remain untested in this environment.
- No baseline-neural orchestration runner is implemented yet (future work).
- This infrastructure supports bounded technical comparisons only; no clinical
  plausibility, independent validation, causality, privacy, or model-superiority
  claim follows.

## Alternatives considered

- **Single-seed forensic workflow**: rejected as insufficient for seed-variation
  analysis across stochastic generators.
- **File-based candidate API**: rejected because writing intermediate patient or
  synthetic rows to disk violates aggregate-only data-handling constraints.
- **No audit CLI**: rejected because manual replay verification is error-prone
  and does not provide a repeatable gate against platform or version drift.
