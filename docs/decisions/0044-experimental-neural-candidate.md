# 0044: Experimental conditional neural candidate

Date: 2026-11-12. Status: experimental implementation, not selected as a
preferred generator.

Add a standalone optional `tch` 0.20.0 CPU crate with in-memory fitting and
sampling, explicit schema and missing states, separate seeds, conditioned
categorical sampling, Wasserstein loss and gradient penalty. Keep statistical
baselines. This is CTGAN-style rather than a faithful reproduction: numeric
values use bounded min-max scaling, categorical outputs use softmax/argmax,
and the critic is not packed. States absent in training are masked; structural
absence is rejected. No model, preprocessing or patient rows need be saved.
See `experimental/neural/README.md` for the interface and reproducibility limits.

The fictional tests passed at 153, 229 and 298 rows, with rare categories,
dependent missingness, constants, invalid inputs, finite losses and outputs,
updates to both networks, in-process CPU replay and seed variation. They do
not demonstrate that the missingness dependency is learned. The default
configuration's exploratory EARLYDRAIN development comparison used three
overlapping inner splits, five neural fit seeds and two sample seeds each, and
two forests on the same training and reference patients. Mean neural marginal
TVD was 0.398–0.414 and pairwise TVD 0.610–0.624 across splits; corresponding
forest ranges were 0.106–0.119 and 0.208–0.219. This candidate did not improve
these metrics. Eight-field joint TVD was 1.0 for all neural draws; sparse joint
states make that measure difficult to interpret. The numeric schema treats
scores as continuous, allowing fractional values; the reported integrity check
only checks declared types and domains, not clinical or ordinal constraints.

The references and outer holdout were previously inspected. The outer holdout
was excluded from this comparison. Aggregate reports, protocol, detailed metrics
and provenance are under ignored `datasets/results/neural-comparison-20261112/`.
The matched real-versus-real context uses reference-sized samples of 46 real
training patients and is not independent validation; see
`datasets/results/earlydrain-real-calibration-20261112/summary.md`. Different
sample sizes make its distances contextual rather than directly matched to the
183-row generator draws here. Split and seed variation are not confidence
intervals. These findings do not establish clinical validity, privacy safety,
causality or external performance. Do not select this neural candidate as a
winner; investigate representation and training behavior before further claims.
