# 0031: In-memory stratified fitted generator

Status: implemented. Date: 2026-11-11.

`FlatFittedStratifiedForest` fits one `FlatFittedBinnedForest` for each
caller-supplied group using shared ordered variables and either the empirical
or penalized selection rule. Groups and their training rows must be nonempty.
The caller supplies a count for every fitted group; zero counts are allowed,
and unknown or missing groups and totals above one million are rejected.
Sampling returns rows keyed by group and supports the same optional continuous
interpolation and aggregate change counts as the single-group API.

Each group derives its random stream from the supplied seed and group name.
With the same fitted input, seed and group name, changing another group's
requested size does not change that group's sample. This is deterministic
orchestration, not a proof of statistical independence. As in decision 0030,
the model borrows training rows, does no I/O, and cannot verify patient
independence, identifier removal, data-use authorization, privacy or clinical
validity. Group labels and generated records can be sensitive; callers must
honor the study's storage and output policy. The EARLYDRAIN flat CLI continues
to publish aggregate reports only.
