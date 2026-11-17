# 0024: Development-only validation for flat generators

Status: implemented. Date: 2026-11-11.

An optional `development_validation_proportion` creates a group-stratified
inner split within the existing development patients. Patient keys are ranked
with the existing hash and a domain-separated split seed; the requested
validation fraction is rounded within each group. Empty inner partitions are
rejected. The candidate is fitted only on inner training and generates the
inner validation size using a separately derived generator seed. The section
reports both group summaries, fixed configured-pair comparisons, learned
edges and sparse backoffs. The same inputs, split and dependency declarations
allow candidates to be compared on the same patients and measurements. No
patient identifiers or generated rows are emitted.

The ordinary report still includes whole-source and outer-holdout summaries and
metrics. This option does not seal them or establish that the outer holdout is
independent or unused; do not inspect those sections during model selection.
Candidate comparison on one split and draw is noisy and repeated selection can
overfit the inner validation data. Explicit pair comparisons do not evaluate
all dependencies, privacy, clinical plausibility or causal validity. Model
choice and evaluation require declared criteria and separately reserved data.
Seed derivation and counts are recorded, but seeds alone do not guarantee
reproducibility across code or environments.

An optional `development_validation_seed` selects the inner ranking directly,
allowing comparisons across inner splits without changing the outer split. It
requires a validation proportion; when omitted, the derived seed above remains
the default. It is recorded in the validation section.

Optional predeclared `numeric_associations` compare Spearman rank correlations
for numeric variable pairs in generated and inner validation rows. Ties receive
average ranks; missing and structural-absence values are excluded pairwise.
The report records complete-pair counts, each coefficient and their absolute
difference, or null for fewer than two pairs or constant ranks. Reversed or
duplicate pairs are rejected. These point estimates assess monotonic
association only and do not establish clinical or causal validity.
