# 0018: Multiple categorical parents for flat generation

Status: implemented. Date: 2026-11-11. Extends decision 0017.

A categorical child may declare multiple earlier categorical parents by listing
one dependency per edge. The model samples the child from development rows that
match all sampled parent states within the allocation group. Raw missing and
structural-absence tokens participate as states. Duplicate edges, noncategorical
endpoints, and parents at or after the child are rejected.

Parent combinations generated upstream may have no matching development row.
The run fails explicitly without publishing output in that case; it does not
silently discard a sample, back off, smooth, or fabricate a conditional cell.
Sparse models can therefore fail even when individual parent values are observed.
Pairwise holdout diagnostics still assess individual edges, not the full joint
distribution. Configured ordering and association have no causal interpretation.

Alternatives include declared backoff, smoothing, or constrained sampling, each
of which requires explicit assumptions and validation. They remain future work.
