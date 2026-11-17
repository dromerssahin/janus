# 0017: Flat conditional categorical baseline

Status: implemented. Date: 2026-11-11.

The aggregate-only flat workflow can fit a configured categorical child from
one earlier categorical parent, within allocation groups, using development
rows. Configured parent values, including raw missing and structural-absence
tokens, select the matching development child distribution. All other columns
use independent empirical marginals. The model never samples a parent value
without development support; sparse strata can still be unstable. Configured
order is a sampling convention and does not establish causality.

For each configured pair, all three flat generators can report the total
variation distance between generated and holdout joint raw-token distributions,
with patient denominators. The distance is unavailable for an empty population;
distinct missing tokens remain distinct. These are point estimates, not privacy,
clinical, causal, or external-validity assessments. No row-level generated data
are published. One-parent categorical conditioning cannot capture arbitrary
multivariate dependence or continuous conditional distributions.

While adding joint evaluation, the previous flat bootstrap was found to retain
partition rows once per configured variable. It now retains each patient once.
Consequently bootstrap intervals from earlier flat reports should not be
treated as equivalent to those produced by the corrected implementation.
