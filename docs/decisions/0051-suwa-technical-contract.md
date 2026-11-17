# Decision 0051: bounded Suwa SAH technical comparison

The existing aggregate Suwa comparison is recorded as a frozen technical
development exercise under `docs/SUWA_SAH_COHERENCE_CONTRACT.md`. It compares
the two forest baselines under identical seeds and metrics against the already
inspected 153-patient Suwa SAH cohort. It is useful for checking cross-cohort
behavior and denominator handling, but it is not independent validation:
patient-level non-overlap is unverified, outcome timing differs from
EARLYDRAIN, and the sex mapping is inferred from aggregate counts.
