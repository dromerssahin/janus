# Local validation readiness checklist

This is procedural guidance for a future approved local dataset. It is not
approval, anonymization, clinical validation, causal validation, or external
validation.

- [ ] Governance and data-use approval is recorded.
- [ ] The provider confirms that input is already deidentified.
- [ ] Permitted derivatives, retention, and destruction terms are documented.
- [ ] Access controls and secure-storage location are documented.
- [ ] Mapping review and validation approval are complete.
- [ ] Audit trail and manifest retention are configured.
- [ ] An expert reviewed variable definitions, timing, censoring, and limitations.
- [ ] A decision states whether outputs may leave the approved environment.
- [ ] The source dataset and data dictionary are stored outside the repository
      worktree in the approved access-controlled location.
- [ ] The study owner has approved the analysis contract and variable-level
      temporal and missingness semantics.

No real records, credentials, identifiers, or restricted derivatives belong in
this repository.

For local work, place approved source files in `/datasets` or another
access-controlled location outside the repository worktree. `/datasets/` is
ignored by Git as a guard against accidental staging; ignore rules do not
provide access control, encryption, or secure retention. If the source is
currently under the repository, relocate it to the approved secure location
before processing. Do not copy source files, row-level derivatives, or
dataset-specific mappings into tracked paths.

For a flat patient-level table, do not convert columns into event records until
the study owner has approved the event/index interpretation. Record the
following decisions in the approved environment before writing a dataset
mapping:

| Decision | Required documentation |
| --- | --- |
| Population and unit | Source-row meaning, patient identity scope, duplicates/repeated admissions |
| Time origin | Which source timestamp defines day zero and its precision |
| Variable roles | Baseline, treatment context, acute course, discharge, follow-up |
| Observation coverage | Which fields require complete or incomplete follow-up declarations |
| Missing states | Per-variable meaning of each missing token: unavailable, not measured, not applicable, or other |
| Treatment definitions | Separate allocation, per-protocol, and as-treated variables and permitted descriptive comparison |
| Output policy | Allowed aggregate outputs, small-cell restrictions, location, retention, and deletion |

Do not infer these rules from a column name or a generic dictionary statement.
Keep this decision record, mappings, fingerprints, and reports with the approved
data in its secure environment. A readiness checklist or successful parser run
does not itself establish permission or validate the study semantics.

The flat descriptive report accepts per-variable `missing_policy: missing` or
`missing_policy: categorical_tokens` plus exact `structural_absence_values`.
Only those explicitly named values count as structural absence; blank and `NA`
remain missing unless themselves explicitly named. Use this only when the study
owner has approved the row-level interpretation. The software cannot infer it
from a generic `NA` token.
