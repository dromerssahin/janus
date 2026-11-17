# 0041: Additional learned stochastic flat generator

Status: superseded as the primary development direction by
[decision 0043](0043-clinician-guided-realism.md); preserved for historical
context. Its copula remains an optional statistical comparator, not the selected
next generator. Research and design only. No new
generator, dependency, data analysis, or validation has been implemented.

## Motivation and current baseline

Janus aims to generate and evaluate study-agnostic synthetic clinical cohorts.
Its fitted flat DAG and forest baselines already learn conditional associations
from development data and draw stochastically. A fixed seed gives reproducible
draws in a fixed execution environment; different seeds normally yield different
draws, although a different result is not guaranteed. The fictional test
`generator_seed_changes_samples_without_changing_split_or_fit_population`
checks changed-seed behavior. Preserve these existing generators as baselines.
The requested additional method should learn a joint distribution from data; it
cannot discover clinical meaning, eligibility, timing, or constraints without a
declared schema and study protocol.

## Candidates and evidence

| Candidate | Evidence and fit | Cost and limitations |
| --- | --- | --- |
| Gaussian copula with learned mixed marginals | [SDV documents](https://docs.sdv.dev/sdv/modeling/single-table-synthesizers/gaussiancopulasynthesizer.md) fitting and sampling a statistical Gaussian copula. Rust provides components such as [nalgebra Cholesky](https://docs.rs/nalgebra/latest/nalgebra/linalg/struct.Cholesky.html), [normal sampling](https://docs.rs/rand_distr/latest/rand_distr/struct.Normal.html), and [distribution transforms](https://docs.rs/statrs/latest/statrs/distribution/trait.ContinuousCDF.html). | A Janus mixed-data and missingness adapter must be designed and built; these primitives are not a turnkey synthesizer. Latent Gaussian dependence can miss nonlinear interactions and tails; encoding nominal categories may impose artificial order. |
| Autoregressive gradient-boosted conditionals | [LightGBM](https://lightgbm.readthedocs.io/en/latest/Advanced-Topics.html) documents missing-value routing and categorical predictors; [lightgbm3](https://docs.rs/lightgbm3/latest/lightgbm3/) documents Rust training and prediction bindings. | LightGBM is predictive, not a cohort generator. Sequential categorical probability models, continuous conditional distributions, missingness, structural absence, sampling and variable order all need explicit implementation. C++ build, API coverage and reproducibility need verification. |
| Conditional GAN or diffusion | [CTAB-GAN](https://proceedings.mlr.press/v157/zhao21a.html) and [TabDDPM](https://proceedings.mlr.press/v202/kotelnikov23a.html) describe generative models for mixed tabular data. | Published benchmark results do not establish performance on Janus data. A Rust-callable implementation, mixed missingness/absence handling, training stability, and reproducibility require investigation and validation. More integration and tuning complexity is not justified yet. |

The citations document methods and component capabilities, not a tested Rust
mixed-clinical synthesizer. Crate versions, licenses, offline availability,
compatibility with the workspace's existing `rand` version, numerical behavior,
and the LightGBM binding's categorical and probability interfaces remain to be
checked before any dependency decision. No Python bridge is selected.

## Decision

Prototype a **Rust-native mixed-data Gaussian copula** as the primary additional
learned statistical machine-learning model. It learns development-set marginals
and a joint latent dependence structure, then makes stochastic joint draws. It
offers a distinct global dependence assumption from Janus's learned conditional
DAG, with fewer moving parts than a neural training stack. This is a design
choice, not evidence that it will outperform the DAG or be clinically valid.

The fallback is **Rust-callable LightGBM autoregressive conditional models** if
the copula cannot represent required dependence or missingness adequately and a
small proof establishes binding coverage and deterministic configuration. This
fallback still requires a generative wrapper; routing missing predictors alone
does not generate missing outcomes or distinguish structural absence.

For a copula prototype, model each variable's value and missing/structural
absence as explicitly distinguished states, with missingness indicators in the
joint model rather than independently injected. Define and validate a concrete
mixed-type encoding and decoding method before implementation; nominal category
ordering, ties, non-positive-definite covariance, rare states, all-missing and
constant columns must have documented handling. An indicator plus value
representation can produce incompatible combinations, so decoding must preserve
declared structural rules, count any repairs or rejections, and never silently
invent observations. Empirical inverse marginals can retain source values and
do not provide privacy or prevent memorization. The exact categorical encoding,
continuous marginal and regularization policy are unresolved design choices,
not proven capabilities of this decision.

## Integration sketch

The intended reusable, in-memory API mirrors the existing fitted flat APIs:

```text
fit(development_rows, declared_schema, fit_options, fit_seed?) -> fitted_model
sample(fitted_model, row_count, sample_seed, sampling_options) -> rows, diagnostics
```

Schema supplies types, allowed categories, missing and structural-absence
semantics, constraints and excluded identifiers; split patients before fitting.
Record a fit seed if fitting uses randomness and always require and record a
sample seed. Fresh draws use a newly chosen seed; store it in permitted aggregate
metadata. Same inputs, seeds, row order, algorithm and numerical software,
threading, hardware and environment are required for replay; cross-version or
cross-platform bit identity is not promised. Distinct seeds can coincide.
Return errors for unsupported or invalid inputs and report constraint and
numerical diagnostics; do not silently discard violations. Keep fitted models,
synthetic rows and predictions in memory where the EARLYDRAIN contract requires
it; only permitted aggregates and provenance go under ignored `datasets/`.

## Evidence needed before use or claims

1. First validate encoding and numerical feasibility with fictional mixed data:
   nominal/continuous dependence, correlated missingness and structural absence,
   all-missing, constant, rare and singular cases, valid decoding, stable counts,
   seed replay and variation. Check bounds, failures and reported repairs.
2. Compare with existing baselines using fixed patient-level splits and matching
   training/sample sizes. Report marginals, joint and conditional dependence,
   missingness patterns, constraint violations, downstream utility, seed
   variation and uncertainty where validated. Do not equate seed variation with
   a confidence interval. Exploratory use of inspected data cannot establish
   final model selection or independent validity.
3. Before claims of generalization or selection, freeze the task and model
   protocol and use a suitable genuinely reserved reference with documented
   permission, patient independence, measurement and timing. The EARLYDRAIN
   holdout, Suwa and HeMoVal outcomes have been inspected; none qualifies as an
   untouched reference. Do not resume an open-ended source search without a new
   reason or request.
4. Assess disclosure under a specified attacker and access scenario; assess
   clinical plausibility with domain expertise and source-aware constraints.
   Statistical similarity alone establishes neither anonymity, clinical or
   causal validity, transportability, nor treatment benefit.

The next implementation session should first specify and test the encoding on
fictional data and verify candidate dependencies, then implement only if those
gates pass. No clinical evaluation has been performed for this proposal.
