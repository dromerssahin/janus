# Independent Evaluation Intake

Decision [0043](decisions/0043-clinician-guided-realism.md) uses this intake at
the clinician-guided loop's final reserved-reference step. Its gates below are
unchanged; the frozen target in this document is a specific historical task,
and a different study requires its own frozen protocol and the same gates.
Do not expose the final reference to iterative clinician review or selection.

## Purpose

Obtain a compatible, approved reference for the frozen generator evaluation
before inspecting its outcome values. This is a source-readiness work package,
not an evaluation result or a claim that the current generator is valid.

## Frozen evaluation target

Use `datasets/results/independent-evaluation-20261112/plan.md` as the protocol
authority. It compares the existing `learned_binned_dag` and
`interaction_binned_dag` baselines on:

- Discharge infarct from sex and admission sedation.
- Discharge infection from sex and initial CT intraventricular hemorrhage.

Do not substitute variables, alter outcome definitions, or add a model after
seeing reference outcomes. A different task requires a separately dated
pre-outcome protocol amendment.

## Intake gates

Complete these gates in order. Stop at the first failed gate and record `no-go`;
do not inspect outcome distributions to decide suitability.

1. **Permission and storage:** Record custodian approval, data-use terms,
   permitted aggregate outputs, retention location and any limits on hashes or
   manifests. Keep source and derived artifacts under the approved ignored
   `datasets/` storage. Never persist patient rows, predictions or fitted models.
2. **Source identity:** Record source citation, version, retrieval/provision date,
   license or agreement, file identity/hash where permitted, and responsible
   custodian. Do not redownload existing cohorts.
3. **Schema-only compatibility:** Inspect documentation and headers only. Confirm
   one patient per row or document a valid patient key and grouping rule; map all
   frozen predictors and outcomes to explicit source definitions and categories.
   Missing definitions, incompatible coding, or absent frozen fields are `no-go`.
4. **Timing and population:** Confirm predictors are available before outcomes,
   follow-up and outcome ascertainment match the frozen definitions, and the
   target population is interpretable for the comparison. Do not infer timestamps
   or equate merely similar endpoints.
5. **Independence and reservation:** Establish patient-level non-overlap with
   model-development data, or create a genuinely reserved patient partition
   before any outcomes are inspected. Different sites alone are insufficient.
6. **Protocol freeze:** Record the source-to-schema mapping, exclusions,
   seeds, metrics, missingness rules, denominators, uncertainty method, replay
   checks and limitations. Obtain domain/custodian review where required before
   reading outcome values.

## Decision record template

For each source, record:

```text
Source/version:
Custodian and permission status:
Permitted storage and outputs:
Identity evidence (and whether hashing is permitted):
Patient unit/key and duplicate handling:
Frozen predictor mapping and source definitions:
Frozen outcome mapping, definition and ascertainment time:
Missing/category coding:
Development-reference overlap evidence:
Protocol deviations: none, or describe before outcome inspection
Decision: go / no-go / pending
Reason and unresolved requirements:
Reviewer/date:
```

`go` means only that a source is ready for the frozen descriptive evaluation;
it does not establish clinical validity, privacy, causality, or general
external validity. `pending` means no patient-level outcome values may be
inspected for suitability or model selection. A `no-go` source must not be
rescued by silent variable substitution.

## Evaluation after a go decision

Execute the frozen runner only after gates 1-6 pass. Keep generated rows and
fitted models in memory. Write permitted aggregate metrics, fingerprints and
execution metadata under `datasets/`. Verify patient disjointness, source and
configuration identity, denominator reconciliation, missing-target handling,
unseen-state and backoff counts, deterministic replay, and unavailable metrics.
Run offline tests, Clippy with warnings denied, formatting and `git diff
--check`. Label results descriptive and conditional on the specified source and
model; do not use them to imply privacy protection or clinical utility.

## Current source queue

- EARLYDRAIN holdout: excluded; repeatedly inspected.
- Suwa SAH: excluded for new model selection; outcomes inspected and frozen
  tasks unsupported.
- SAH-complications: no-go for frozen tasks; required admission predictors are
  absent and infarction/infection definitions and timing are unresolved. See
  `datasets/results/source-suitability-20261112/complications-audit.md`.
- HeMoVal: no-go for frozen tasks after documentation and header-only review;
  required fields and endpoint equivalence are unestablished, and different
  recruitment years do not prove patient-level non-overlap. See
  `datasets/hemoval/analysis-log.md`. A separate within-HeMoVal task was later
  run under decision 0040; its outcomes have been inspected and are not an
  untouched reference for subsequent model selection.

No source is currently ready for the frozen independent evaluation. The user
stopped the open-ended source search; apply these gates to a newly offered
source or a separately justified future task.
