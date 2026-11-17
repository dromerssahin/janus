# Rust vertical slice

The first implementation uses one Rust library crate and one CLI crate. YAML is
the authoring format and outputs are inspectable YAML files. `StdRng` with an
explicit seed provides deterministic single-process generation under the pinned
dependency set; platform and dependency changes may still alter results.

The example is fictional. Clinical validity, privacy, forensic detectability,
and causal interpretation remain unperformed evaluations.
