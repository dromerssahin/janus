# 0032: Aggregate predictive utility baseline

A study may declare a binary target and one categorical predictor available before
that target. For each inner group, fit smoothed conditional frequencies separately
on real training and generated rows and evaluate both on the same validation
patients. The report exposes Brier scores, labeled and unlabeled counts, and
unseen predictor counts, but no individual predictions or fitted tables.
Unknown target labels are excluded; raw predictor tokens, including missing
ones, remain states. Unseen states use the smoothed overall rate. Zero labeled
training or validation cases yield an unavailable score.

This deliberately small baseline has no extra dependency and gives an
interpretable comparison, but cannot test multivariable prediction. It does not
verify timing, represent causality, establish clinical utility or privacy, or
make reused validation patients independent. The generated sample is sized to
the validation partition, so real and synthetic training budgets may differ.
A general configurable multivariable evaluator and independent test remain
future work; they must not infer temporal eligibility from column order.
