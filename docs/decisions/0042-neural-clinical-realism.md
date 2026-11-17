# 0042: Neural generator for clinical-reader realism

Status: historical model proposal, superseded as the primary direction by
[decision 0043](0043-clinician-guided-realism.md) on 2026-11-12. It had
superseded decision [0041](0041-additional-stochastic-generator.md)'s copula direction;
0041's statistical and data-governance cautions remain applicable. No neural
generator is implemented or clinically validated. The aim is clinically
credible joint patient records, assessed by clinical readers as well as numeric
metrics, not merely a different dependence structure.

## Evidence and candidate comparison

The project has development sets of 229 and 298 patients. Their holdouts and
other named reference outcomes have been inspected. It is unknown whether any
candidate below can generate realistic Janus cohorts at this scale. A
[253-patient pharmacogenetics study](https://ascpt.onlinelibrary.wiley.com/doi/10.1002/psp4.13240)
reported that CTGAN and TVAE produced distorted summaries; TVAE had the largest
hazard-ratio deviation and CTGAN distorted some survival estimates. This is one
task, not a universal ranking. An
[unreviewed 2026 size-ladder preprint](https://arxiv.org/abs/2610.03500), using
downstream AUROC rather than clinical-reader realism, reports that seed variation
often exceeded differences between methods. It supports repeated runs and strong
baselines, not a claim that any model is generally unsuitable.

| Candidate | Evidence and benefits | Small-data failure and integration risk |
| --- | --- | --- |
| CTGAN / CTAB-GAN | [CTGAN paper](https://papers.nips.cc/paper_files/paper/2019/hash/254ed7d2de3b23ab10936522dd547b78-Abstract.html) uses conditional training for imbalanced discrete columns. [CTAB-GAN](https://proceedings.mlr.press/v157/zhao21a.html) addresses mixed types and skew. Both can learn nonlinear joint distributions. | In the 253-patient study CTGAN distorted survival estimates and some variables, despite being a stronger performer on its aggregate criteria. Rare combinations may still disappear or be memorized; adversarial optimization can collapse or vary across runs. CTAB-GAN is a distinct, more complex design, not interchangeable. The [standalone CTGAN package](https://github.com/sdv-dev/CTGAN) requires data without missing values; explicit joint missing/absence encoding is needed. A faithful Rust implementation and its reproducibility are unverified. |
| TVAE | [SDV TVAE](https://docs.sdv.dev/sdv/modeling/single-table-synthesizers/tvaesynthesizer.md) documents a mixed-tabular VAE with adjustable network and training settings; the standalone [CTGAN package](https://github.com/sdv-dev/CTGAN) also implements it. Non-adversarial training is a useful fallback. | On the 253-patient study, TVAE had the greatest reported hazard-ratio deviation. Rare patterns may be smoothed away or memorized; neither stable optimization nor faithful joint records is guaranteed. The standalone package requires no missing values, so explicit encoding is necessary. |
| TabDDPM | The [paper](https://proceedings.mlr.press/v202/kotelnikov23a.html) describes diffusion over mixed continuous and discrete features, and the [authors' implementation](https://github.com/yandex-research/tab-ddpm) is PyTorch. | Published benchmark performance does not demonstrate realism at 229–298 patients; rare modes, memorization, seed sensitivity and cost remain practical risks at this size, with no Janus-specific failure measured. Missing/structural-absence generation requires a verified encoding; the reference README does not establish it. Many denoising steps add sampling and training cost; Rust reproduction and parity remain unverified. |
| Rust-native neural infrastructure | [Burn](https://burn.dev/) documents Rust neural training/inference; [scirs2-neural](https://docs.rs/scirs2-neural/latest/scirs2_neural/) documents layers, losses and training; [tenflowers-neural](https://docs.rs/tenflowers-neural/latest/tenflowers_neural/) documents neural primitives. | These are frameworks, not evidence of a ready-to-fit CTGAN/TVAE or of realistic clinical generation. The model, mixed-data encoder, sampling and reproducibility must be implemented and tested. At hundreds of records they inherit the failures of the model built with them; no comparative small-clinical-cohort result for these Rust frameworks was found. Check supported operations, maintained versions, licenses and builds before selecting dependencies. |
| Existing learned DAG / forest and optional copula | Janus's fitted statistical baselines are already seeded stochastic samplers. Decision 0041 documents a possible copula, not an implemented one. | May miss nonlinear, higher-order and rare patterns; the copula's latent dependence and nominal encoding are restrictive. Conversely they may outperform neural methods on a particular small dataset. None establishes a clinical realism or privacy guarantee. |

No documented Rust turnkey synthesizer for these requirements was established in
this survey. No feasibility spike was run: documentation supports testing a
model, but only the planned multi-seed and clinical-reader evaluation can answer
whether it works for the project's data. The 253-patient result shows training
is possible in one clinical setting, not that Janus can reproduce it.

## Decision

The primary candidate is a **CTGAN-style conditional tabular GAN trained and
sampled in Rust**, using Burn as the first framework to evaluate. It targets
nonlinear, multimodal joint structure and explicitly exposes rare categorical
states to training. The 253-patient evidence makes it a defensible experiment,
not a promise that rare clinical combinations or downstream analyses will be
faithful. Verify Burn operations, CPU training, version/license compatibility,
and a small reference-parity fixture before committing to the dependency; use
another Rust framework if those gates fail. This is a proposed implementation,
not a claim that Burn already supplies CTGAN.

The fallback is a **Rust-native TVAE-style mixed-output variational autoencoder**
if GAN training repeatedly collapses or cannot meet the feasibility gates. Its
non-adversarial objective may be easier to diagnose, but the cited 253-patient
result is a specific warning about loss of clinically meaningful relationships.
Compare both fairly with the existing generators; retain the copula from 0041 as
an optional statistical comparator, not an implementation prerequisite. Do not
select a winner using inspected holdouts or clinical impressions alone.

For mixed data, explicitly encode categorical states, observed continuous
values and missing versus structural-absence indicators jointly. Define
continuous transformations, masking, decoding and hard constraints from the
declared schema, fit transformations only on development patients, and assess
whether missingness is associated with other variables. Reject unknown codes and
report invalid combinations and repairs or rejections; do not silently convert
missing into a measured value. Rare states and all-missing, constant and
high-cardinality columns require explicit behavior. Absence or missingness is
never an outcome label. Reference implementations' preprocessing is not proof
that these project-specific semantics are satisfied.

## Proposed Rust boundary and reproducibility

```text
fit(development_rows, schema, config, fit_seed) -> in_memory_model, fit_diagnostics
sample(in_memory_model, count, sample_seed) -> in_memory_rows, sampling_diagnostics
```

Rust owns patient splitting, typed schema and source permissions; the fitted
model owns preprocessing and weights. Fix seed streams for initialization,
minibatch ordering, conditional draws, training noise and sampling, and record
their derivation, row order, schema/config fingerprints, generator revision,
dependency versions, backend, hardware, thread settings and environment in
permitted aggregate metadata. First target replay on a pinned single-thread CPU
backend; verify independently repeated fits and samples. Different seeds can
coincide. Never promise bit identity across devices, parallel settings, versions
or platforms; fail or report when deterministic operations are unavailable.

Training rows, preprocessing state, weights, generated rows and predictions
remain in memory for EARLYDRAIN, whose contract permits aggregate reports only.
Do not write checkpoints, caches or model files there, including temporary
files. A fitted model may memorize input values. Explicitly disable or audit
framework checkpointing, logging and temporary artifacts; fail closed if this
cannot be assured. Only permitted aggregate metrics and provenance may be saved
under ignored `datasets/`. Other source terms must be checked independently.
The present decision does not require a Python trainer or bridge; if Rust
feasibility fails, reassess rather than silently introduce one. Any future
Python benchmark must obey the same in-memory policy and reproducibility rules.

## Feasibility gates and failure criteria

Before accessing additional real outcomes, freeze a small experimental protocol
and schema. Start with fictional data and known nonlinear associations, mixed
types, rare categories and correlated missing/absence states at sizes 153, 229
and 298. Include constants, all-missing columns and invalid inputs. Establish
that training actually updates weights, loss/gradients remain finite, generated
rows obey supported schema, rare states and dependence are detectable, and
identical pinned-environment seeds replay. Compare against a reference
implementation on fictional data by documented transformations, losses and
distributional behavior, not by expecting identical weights. Show repeated
sampling changes for a nondegenerate fixture; distinguish fixed model sampling
seeds from independent training seeds. Record wall time and peak memory.

Next run a prespecified **exploratory**, development-only comparison using
patient-level inner splits, excluding the original reserved HeMoVal keys before
outcome extraction. Use the same training patients, variables and sample counts
for Janus baselines and the neural candidate, at least three prespecified split
seeds and five independent fit seeds per split, with multiple sample seeds per
fit. Report all failures and all runs, not a selected seed; compare paired
scores and variation without treating correlated folds or seeds as independent
patients. No extra cohort search is authorized by this decision.

Record per-variable and joint/conditional fidelity, rare-state coverage and
counts, observed/missing/absence patterns, constraint violations before and
after any repair, duplication and nearest-neighbor diagnostics, and downstream
utility against the same held-out patients. Fix clinically meaningful metrics,
minimum subgroup counts, tolerances, run/time budget and handling of sparse
denominators before accessing outcomes; do not declare success from aggregate
averages alone. A feasible prototype must complete the prespecified runs with no
unexplained nonfinite losses, replay failures, invalid output or unreported
discarding; produce diverse outputs across fit/sample seeds; and recover the
fictional fixture's prespecified rare and conditional patterns within its
predeclared sampling uncertainty. On development data it must show stable,
clinically relevant joint fidelity and no material utility/constraint regression
versus the baselines using prespecified tolerances. Rare patterns unsupported by
data must be reported as unresolved, not scored as faithful.

Failure to train or replay, frequent mode collapse, systematic loss of rare
states, implausible combinations, excessive copying, or large unexplained
between-fit variation falsifies feasibility for that configuration. Failure to
improve the prespecified realism-relevant measures relative to baselines, or
clinician reviewers finding important recurring errors, falsifies the claim of
added realism for that setting even if training succeeds. Results may be
inconclusive when counts or intervals cannot distinguish methods; do not tune
on the reference or invent thresholds after results. If CTGAN fails, try the
prespecified TVAE fallback on fresh development splits as an exploration, with
the same rules, or retain the best-supported existing baseline if neither works.

## Evidence required for realistic clinical use

Develop a blinded reader protocol with qualified clinicians, representative
records, explicit source permission and safe in-memory presentation; compare
real and synthetic records without revealing source, solicit specific errors and
severity, document inter-reader agreement and uncertainty, and avoid exposing
identifiers. A reader's inability to distinguish records alone is not validity.
Evaluate patient-level dependence, time/eligibility consistency where actually
documented, clinically important subgroups, utility and independent site or
cohort transport using a frozen protocol and a genuinely reserved permitted
reference. EARLYDRAIN holdout, Suwa and HeMoVal outcomes are inspected; none is
an untouched reference. For privacy, specify attacker/access and perform a
separate risk assessment: nearest-neighbor checks do not establish safety.
None of the proposed checks alone proves clinical, causal or privacy validity.
