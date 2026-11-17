# 0020: Configurable flat row keys and groups

Status: implemented. Date: 2026-11-11.

The aggregate-only flat workflow accepts an explicit `row_key_column` and one
or more declared values of its configured grouping column. The row key defaults
to `no` for legacy configurations, must identify one patient row uniquely,
and cannot also be the group column or a modeled variable. Groups need not be
treatments; a constant group represents an unstratified patient table. Existing
stratified splitting and fitting continue independently within each group.

This removes Early Drain-specific assumptions about the identifier's name and
the number of allocation groups. It does not support multiple rows per patient,
longitudinal event conversion, or identifiers being emitted as model variables.
An absent declared group remains an empty group, and a group with holdout rows
but no development rows cannot support generation. Patient identifiers and
aggregate reports remain subject to the configured data-use terms.
