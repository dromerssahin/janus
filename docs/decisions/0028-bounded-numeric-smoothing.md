# 0028: Explicit bounded numeric smoothing

Status: implemented. Date: 2026-11-11.

The flat generators otherwise sample numeric values observed in training. A
study may explicitly configure `numeric_smoothing` per continuous variable,
with a finite positive bandwidth and finite ordered minimum and maximum in the
variable's units. Each observed sampled value is perturbed uniformly in the
intersection of its bandwidth interval and the declared range. Training values
outside that range cause an error; the validation or holdout data do not set
the bounds. Missing and structural-absence tokens remain unchanged.

For binned forests, candidates must remain in the original training-derived
bin; the fitted dependency structure is therefore still conditioned on the
same binned parent states. A candidate that would become an explicitly
configured structural-absence token is also rejected. At most 16 attempts are
made; if no distinct acceptable floating-point value is found, the original is
retained. The aggregate report records observed candidates, changed and
unchanged values for each variable and group. The smoothing RNG is separately
derived from the generator seed, so enabling smoothing does not alter the
underlying empirical draw sequence. Results depend on the declared smoothing
order and implementation; a seed alone does not guarantee portability.

This is an explicit modeling assumption, not a learned continuous density or
evidence of clinical plausibility. Uniform perturbation can alter dependencies,
create inappropriate combinations or values, and obscure exact-row matches.
It is inappropriate for discrete scores and counts without a suitable discrete
model. Bounds and bandwidths must come from the study; no clinical defaults are
inferred. Numeric novelty does not establish privacy or authorize release.
Fitted bins, empirical values, reports and model-derived artifacts can remain
sensitive. Alternatives include no smoothing and a separately validated
continuous model.
