# Architecture decision records

Add a short numbered record when a consequential choice is made during
implementation. Include status, context, decision, alternatives, consequences,
and date. Link superseding records instead of rewriting history. Proposals in
the architecture document are not decisions until recorded or implemented.

- [Configured fictional generator](0001-configured-generator.md)
- [Held-out descriptive evaluation](0002-held-out-evaluation.md)
- [Patient-level development and holdout splits](0003-patient-splits.md)
- [Empirical descriptive fitting](0004-empirical-fitting.md)
- [Temporal evaluation before fitting](0005-temporal-evaluation.md)
- [Declared observation and censoring](0006-observation-censoring.md)
- [Concrete fictional episode protocol](0007-concrete-episode-protocol.md)
- [Conditional categorical fitting](0008-conditional-fitting.md)
- [Descriptive outcome timing](0009-outcome-timing.md)
- [Longitudinal evaluation](0010-longitudinal-evaluation.md)
- [Configured repeated-event generation](0011-repeated-event-generation.md)
- [Manual fictional evidence registry](0012-evidence-registry.md)
- [Patient-level categorical bootstrap](0013-patient-bootstrap.md)
- [0015 CSV, Parquet, and DuckDB](0015-csv-parquet-duckdb.md) (implemented for fictional examples).
- [Flat empirical marginal generator](0016-flat-marginal-generator.md)
- [Flat conditional categorical baseline](0017-flat-conditional-generator.md)
- [Multiple categorical parents for flat generation](0018-flat-multiple-parents.md)
- [Explicit sparse backoff and family evaluation](0019-flat-sparse-backoff.md)
- [Configurable flat row keys and groups](0020-flat-row-key-and-groups.md)
- [Numeric children in flat conditional generation](0021-flat-numeric-conditional.md)
- [Development-fitted ordered categorical forest](0022-learned-categorical-forest.md)
- [Development-fitted mixed-child forest](0023-learned-mixed-forest.md)
- [Development-only validation for flat generators](0024-flat-development-validation.md)
- [Binned mixed-type dependency forest](0025-binned-mixed-forest.md)
- [Aggregate exact-match diagnostics](0026-flat-exact-matches.md)
- [Penalized binned dependency forest](0027-regularized-binned-forest.md)
- [Explicit bounded numeric smoothing](0028-bounded-numeric-smoothing.md)
- [Training-derived numeric interpolation](0029-numeric-interpolation.md)
- [Reusable in-memory fitted binned forest](0030-in-memory-fitted-forest.md)
- [In-memory stratified fitted generator](0031-stratified-fitted-forest.md)
- [Aggregate predictive utility baseline](0032-flat-predictive-utility.md)
- [Greedy binned two-parent generator](0033-greedy-binned-dag.md)
- [Predeclared inner joint families](0034-inner-joint-families.md)
- [Interaction-only parent pair search](0035-interaction-search.md)
- [Stratified fitted DAG sampling](0036-stratified-dag.md)
- [Joint-state predictive utility](0037-multivariable-utility.md)
- [Policy-aware interpolated in-memory sampling](0038-policy-interpolation.md)
- [In-memory external flat evaluation](0039-external-flat-evaluation.md)
- [Prespecified HeMoVal within-cohort utility task](0040-hemoval-reserved-utility.md)
- [Additional learned stochastic flat generator](0041-additional-stochastic-generator.md)
- [Neural generator for clinical-reader realism](0042-neural-clinical-realism.md)
- [Clinician-guided generation and evaluation](0043-clinician-guided-realism.md)
# Experimental neural candidate

See [0044](0044-experimental-neural-candidate.md) for the optional conditional
neural implementation and its exploratory comparison.
See [0045](0045-neural-conditioning-diagnostic.md) for the bounded conditioning
policy diagnostic and training-loop replay fix.

- [Machine-validated coherence contract and protocol manifest](0047-coherence-contract-manifest.md)
- [Fictional aggregate-only review harness](0048-fictional-review-harness.md)
- [Identical candidate comparison conditions](0049-candidate-comparison-freeze.md)
- [Contract validation CLI](0050-contract-validation-cli.md)
- [Bounded Suwa SAH technical comparison](0051-suwa-technical-contract.md)
- [Bounded technical validity claim](0052-bounded-technical-validity-claim.md)
