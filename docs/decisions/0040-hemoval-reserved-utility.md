# 0040: Prespecified HeMoVal within-cohort utility task

Status: frozen 2026-11-12, before individual outcome inspection. This is a new
task, not an amendment claiming equivalence to the EARLYDRAIN discharge tasks.
The evaluator is not yet authorized to run: the gates below must first pass.

## Source and question

Use the acquired HeMoVal clinical file, DOI
[10.5281/zenodo.6609707](https://doi.org/10.5281/zenodo.6609707), CC BY 4.0,
retrieved 2026-11-12, SHA-256
`b2f9b2d5f54aacc210f4c6de8b811d4e9079711a660b5480701e667cfa4b6ce8`.
The [study protocol](https://pmc.ncbi.nlm.nih.gov/articles/PMC9290286/)
describes admission WFNS and mRS at 12 weeks, plus or minus two weeks. The
source headers include `ha_wfns` and `fup_mrs`; their raw coding and mapping
to those concepts require documentation verification before outcome access.
Ask how well a generator fitted on HeMoVal development patients preserves the
association between admission WFNS and poor 12-week mRS on held-out HeMoVal
patients. Define poor outcome as mRS 3–6 and favorable as 0–2, only if the
source codebook confirms the standard 0–6 scale and its meaning. Use WFNS
grades 1–5 as five categorical predictor states if their source coding is
confirmed. No other predictors or endpoints are substituted.

This is an internal patient-held-out study of a generator trained on HeMoVal;
it does **not** validate an EARLYDRAIN-trained model, establish external
transport, clinical fitness, causality or privacy. Include that statement in
every result. Published HeMoVal aggregate results have been seen; individual
outcomes in the acquired file have not been inspected. Follow-up loss and
differences between intended and actual assessment time limit interpretation.

## Patient unit and reservation

The intended unit is one participant, keyed by `record_id`; repeated REDCap
events and instruments belong to the same participant and must remain together.
Before using outcomes, verify from source documentation that this key identifies
one patient and determine the event/instrument coding for the single admission
WFNS and 12-week mRS assessment. Do not infer a mapping from outcome values.
Reject missing keys, conflicting or ambiguous baseline or follow-up records,
and unresolved repeat assessments; report aggregate counts. If participant
identity or deterministic extraction cannot be established, stop rather than
split repeated rows or guess. No patient identifiers are persisted.

### Population clarification, 2026-11-12

Before inspecting individual outcomes, apply the source authors' published
participant selection to the existing reservation, without re-ranking: exclude
test-site records; include a participant with an end-of-study event marked
complete, or marked incomplete with reason `2` (representative withdrew), `4`
(lost to follow-up), or `5` (died), provided no contradictory consent or
eligibility information is present. The authors' analysis uses these categories;
it excludes screening failures, patient withdrawal, and declined participation.
Do not infer a label from death or exclude patients for missing follow-up. If
eligibility, contradictory consent, unknown or missing status, or masked test-site
coding cannot be resolved from documentation, stop and report the ambiguity.
The study protocol states that data collected until withdrawal remain coded for
analysis, while the study paper states that data for initially incapable
patients who later declined were excluded; do not reinterpret those statements
as blanket consent to use other withdrawal categories. The source code and
protocol references are recorded in `datasets/hemoval/documentation-audit.md`.

After these documentation gates pass and before outcome inspection, reserve
20% of unique participants by ranking the existing seeded FNV-1a 64-bit row-key
hash with SplitMix64 finalizer, seed 101; use the rounded 80% development count,
with hash ties broken by key in memory. Assign all events for each key to the
same partition. Record algorithm, seed, counts and permitted source fingerprint,
not identifiers. Do not stratify or select the partition on outcomes. The
inspected EARLYDRAIN holdout and Suwa outcomes are excluded entirely.

## Analysis and safeguards

Fit the existing greedy and interaction binned DAG baselines independently on
the same development participants, generating one synthetic training row per
development patient with seeds 11, 22 and 33. Fit the real utility baseline on
the same development patients. Score each synthetic-trained and real-trained
binary smoothed-frequency utility model on the same reserved participants. Treat
unseen WFNS states using the existing overall-rate fallback and report counts.
Use smoothing alpha 1.0 and the `DropLastParent` sparse policy, with the
existing unstratified DAG fit defaults. Treat this as a full-cohort analysis of
the selected HeMoVal patients, rather than conditioning on a treatment group.
Report per-seed Brier scores and paired synthetic-minus-real differences,
mean across seeds, labeled and unlabeled counts and their denominators, WFNS
and mRS marginal and joint total variation, and generation backoff counts.
Report uncertainty intervals as unavailable unless a patient-level method is
implemented and validated. Seed variation is not a confidence interval.

Blank and `NA` are missing; treat other documented missing or structural-absence
codes as missing only according to the verified codebook. Missing/absent mRS is
never a positive or negative label; retain and count missing predictor states.
Reject unknown codes rather than silently map or impute them. If the source
does not document whether mRS was assessed in the intended window, report that
limitation; do not infer the date. No individual data, predictions, synthetic
rows or fitted models are written. Store permitted aggregates, manifest,
software/environment details and hashes under ignored `datasets/`.

Before reporting, check source hash, key grouping and partition disjointness,
row and label reconciliation, missing and unseen states, backoffs and replay;
run meaningful fictional tests, the offline suite, Clippy, formatting and
`git diff --check`. Do not alter this task in response to reserved outcomes;
any change needs a separately documented protocol and a fresh reference.
