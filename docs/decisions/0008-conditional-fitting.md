# Conditional categorical fitting

Fit schema version 2 declares ordered variables and earlier parents. It fits
descriptive complete-case categorical tables and rejects any generated parent
combination without an observed child. There is no smoothing or imputation.
The manifest records marginal counts, each table's observed, missing and absent
child counts, missing and absent parent counts, and observed parent combinations
outside the generated domain. Parent-absent takes precedence over parent-missing
when both occur. Excluded episodes are reported separately.

Evaluation schema version 2 supports named categorical pairs and integer
categories; version 1 keeps its original single-pair interface. Pair counts use
complete observed pairs. These associations do not establish causation, privacy,
clinical validity, or temporal trajectory fidelity. The fictional fixture is a
software demonstration, not evidence of a clinical association.
