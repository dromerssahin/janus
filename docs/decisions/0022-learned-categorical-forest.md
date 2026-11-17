# 0022: Development-fitted ordered categorical forest

Status: implemented. Date: 2026-11-11.

The flat workflow can learn a one-parent categorical dependency forest from
development rows within each configured group. For each categorical child, it
selects the strongest earlier categorical variable by empirical mutual
information in natural units; values within `1e-12` are treated as ties and the
earliest parent wins. Scores at or below `1e-12` leave the child independent.
Raw missing and explicitly declared structural-absence tokens are states.
Numeric variables are independently sampled from development marginals. The
ordered graph and fitted categorical samples never use holdout rows.

The report records each selected edge, score and development denominator and
compares that pair's joint distribution with the designated holdout. If a
caller explicitly requested the same diagnostic, it is reported once. Learned
edges may contain sensitive information. Variable order is a study assumption,
not proof of temporal availability or causality. High-cardinality and small
groups can overfit; the model has no regularization or clinical, privacy,
causal or external-validation guarantee. Generated values remain empirical and
individual row-level records are not published. Repeated model evaluation on
the same holdout can leak into model choices and requires a separate final test.

This simple trained statistical model complements, rather than replaces,
configured dependencies and the independent and row-resampling baselines. It
does not fit continuous densities or model dependencies of numeric children.
