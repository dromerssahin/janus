# Patient-level development and holdout splits

Status: accepted. Date: 2026-11-09.

Janus assigns each distinct patient with seeded FNV-1a over the seed and UTF-8 identifier followed by the SplitMix64 finalizer, comparing the unsigned result against `floor(proportion * 2^64)` (with explicit 0 and 1 bounds), keeping all events and episodes together. The finalizer improves mixing for similar identifiers and changes assignments from the earlier FNV-1a-only implementation. Retain old split artifacts for reproduction; the manifest identifies the algorithm. This is a probabilistic split and does not guarantee exact proportions or representative partitions. It publishes development and holdout fixtures with counts and a fingerprint. The split fingerprint is FNV-1a 64-bit over the canonical YAML serialization emitted by the current `serde_yaml` implementation. Evaluation can require the reference fixture to match that recorded holdout.

This establishes reproducible internal partition provenance only. It does not verify model fitting, leakage prevention, or any temporal, site, clinical, privacy, causal, or external validity claim. Temporal and site splits remain future extensions.
