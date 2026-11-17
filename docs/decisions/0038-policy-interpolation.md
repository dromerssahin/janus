# 0038: Policy-aware interpolated in-memory sampling

Single-group and stratified fitted binned generators provide
`sample_interpolated_with_policy`, combining the existing sparse-parent policy
with numeric interpolation and returning both aggregate diagnostics alongside
the transient rows. This lets callers use DAG backoff and continuous generation
in one call. Existing `sample_interpolated` retains fail-on-unsupported behavior.

Interpolation requires the complete fitted parent state and child bin. If the
sampling policy backed off because the full state lacked support, that value has
no full-state training neighbor and remains unchanged, as recorded by the
interpolation counts. Interpolation uses its separate random stream, so it does
not change the underlying empirical draw. Group failures return no partial cohort.
This is not a learned density or a privacy or clinical guarantee; callers must
exclude identifiers and respect the applicable data-use policy.
