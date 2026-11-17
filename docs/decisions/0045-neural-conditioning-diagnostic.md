# 0045: Neural conditioning diagnostic

The experimental neural candidate now exposes its conditional-state sampling
policy. `LogFrequency` is the historical configuration used by the exploratory
EARLYDRAIN comparison and remains the default comparator. `EmpiricalFrequency`
samples states in proportion to their observed counts and is intended for a
bounded diagnostic of unconditional marginal fidelity.

The policy is stored in the fitted model and reused by `sample`, so fitting and
generation cannot silently use different condition distributions. A duplicated
condition-batch call in the training loop was removed; it consumed an unused
random batch on every step and obscured replay and diagnosis. Fictional tests
cover both policies, replay, missingness and mixed numeric/categorical inputs.

The same-seed aggregate diagnostic was run on the existing three overlapping
EARLYDRAIN development splits (183 training and 46 inspected reference patients
per split). Mean marginal TVD across splits was 0.4066 for log-frequency and
0.4081 for empirical-frequency conditioning; mean pairwise TVD was 0.6151 and
0.6167, respectively. Eight-field joint TVD remained 1.0 for every policy and
draw. The change therefore did not improve the observed neural gap and is
retained as a diagnostic option, not a preferred configuration. The aggregate
report and fingerprints are in the ignored
`datasets/results/neural-comparison-20261112/conditioning-diagnostic.yaml`.

This is a representation/training diagnostic, not a model-selection claim.
The neural candidate remains exploratory; fractional numeric scores, collapse,
sparse joint states and inspected references limit interpretation. No clinical,
causal, privacy or independent-validation claim follows.
