# 0034: Predeclared inner-validation joint distributions

A flat study may declare sets of at least two distinct categorical variables
for development-only validation. Each group reports total variation over the
full raw-token joint distribution of generated and inner-validation rows, with
patient denominators. The same declared family can be used across candidates,
without relying on their different learned edges. Missing and structural-absence
tokens are retained as distinct states; no cell counts or individual rows are
reported. Duplicate families, unknown variables, numeric variables and missing
inner validation are rejected.

Total variation is descriptive; sparse combinations and small samples can make
it unstable. It does not explain which cells differ or validate causality,
clinical usefulness or privacy. Reused inner splits are not independent final
validation. Numeric relationships require separately specified handling and are
not silently categorized by this option.
