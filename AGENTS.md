# Janus: guidance for contributors and coding agents

Read this file and `docs/PROJECT_STATE.md` at the start of a session. Consult
`README.md` for supported workflows and `docs/ARCHITECTURE.md` before changing
interfaces, data models or boundaries. Architecture describes proposed design,
not a list of implemented features.

## Fresh-session handoff (2026-11-13)

The primary goal is a study-agnostic research engine for plausible synthetic
clinical cohorts, with claims limited to evidence actually established. The
current completed milestone is bounded technical validity: the workflow is
contracted, reproducible and auditable on fictional and inspected development
evidence. It is not clinical plausibility, independent validation, causality,
privacy validation, transportability or model selection. See decision 0052 and
the current-state handoff in `docs/PROJECT_STATE.md`.

This session's implementation includes `coherence` contract and protocol
manifest types, candidate-comparison validation, the in-memory fictional review
harness, and `janus-cli validate-contract`. The Suwa SAH technical comparison
is documented by `docs/SUWA_SAH_COHERENCE_CONTRACT.md`; its outcomes were
inspected and patient-level non-overlap is unverified. Continue from these
artifacts rather than treating Suwa, HeMoVal or EARLYDRAIN holdouts as
untouched references.

## Delegation

Use `gpt-6-luna` with low reasoning for bounded routine subagent work and
`gpt-6.1-terra` with medium reasoning for orchestration where available.
Multi-agent mode is V1; V2 was disabled after sealed-task backend failures. A
later Luna delegation also failed in V1. Do not change the setting or model
defaults without testing that a subagent successfully returns its result.

## Purpose and current direction

Janus is a research engine for generating and evaluating synthetic clinical
cohorts across studies, diseases, institutions, interventions and outcomes.
Neurosurgery is an initial motivation, not a core requirement. The aim is
plausible patient-variable combinations, including unusual patients and
missingness; see decision 0043. Generators express assumptions and learned
associations, and evaluators assess fidelity, utility and other properties.
Generated outcomes and counterfactuals are model-dependent hypotheses, not
clinical recommendations or evidence of treatment benefit.

The existing flat Rust baselines fit statistical models and sample
stochastically with explicit seeds. Keep them as baselines. Neural models are
candidates, not assumed winners; the experimental standalone neural candidate
and its exploratory results are described in decision 0044. Technical
synthetic-versus-real comparisons on permitted data do not require clinician
sign-off or a reserved reference. Label inspected references and limitations;
do not claim clinical validity, causality or privacy safety. Clinical claims
need appropriate clinical assessment and independent evidence. Current results
and the user-requested neural development next step are in
`docs/PROJECT_STATE.md`; its LibTorch probe is not a cohort generator.

## Do not change without reason

- Keep study protocols, data sources, clinical scores and domain extensions out
  of the general core. Prefer Rust-native, composable interfaces; use external
  scientific tools through explicit replaceable boundaries when justified.
- Respect source terms and preserve patient grouping, eligibility, timing,
  provenance and consent. Never infer a timestamp or assume a diagnosis was
  available before it was recorded. Treat blank and `NA` as missing unless
  explicitly structural absence; neither missing nor absent targets are utility
  labels. Preserve this validation and regression coverage.
- EARLYDRAIN is already at `datasets/ed.csv`; do not redownload it. Its contract
  permits aggregate reports under ignored `datasets/` only: patient rows,
  synthetic rows, predictions and fitted models stay in memory. Its CSV has no
  diagnosis timestamp. Never commit real records, credentials or restricted
  derivatives; honor other sources' permissions and retention requirements.
- EARLYDRAIN holdout, Suwa and HeMoVal outcomes were inspected. Do not call
  them untouched. Preserve HeMoVal's original seed-101 reservation over all
  442 keys before eligibility; exclude reserved keys before extracting WFNS or
  mRS and do not rerank selected patients. Decision 0040 is a frozen snapshot.
  Workbook encounter keys are not proven patient identifiers; a full workbook
  row including an outcome was inadvertently displayed during header inspection,
  so its outcomes are not wholly uninspected. See the project state for details.
- Existing open cohorts and acquisition records are under ignored `datasets/`.
  JinhuaNSICU is listed as controlled access; no patient data were downloaded.
  An open-ended source search found no ready independent reference. Apply the
  independent intake gates to any newly offered reference when a frozen
  validation task requires them; do not assume different sites or periods prove
  patient independence. The current bounded milestone does not require a new
  reference.
- Reject a nonempty holdout group with no development rows before fitting, and
  retain unbiased bounded sampling in `flat_sample_index` and their tests.
  Record seeds, configuration, software, permitted fingerprints and environment;
  reproducibility is limited to conditions actually verified.
- Do not weaken the bounded-claim boundary in decision 0052. The coherence and
  review schemas are integrity/preflight tools, not clinical rules; do not add
  undocumented plausibility constraints or relabel inspected Suwa, HeMoVal or
  EARLYDRAIN data as independent validation.
- Keep `ReviewPresentation` and `aggregate_ratings` in-memory and
  aggregate-only; the fictional review harness does not authorize real-record
  presentation. Keep protocol and candidate-comparison contract identity and
  shared-condition checks before candidate comparisons.

## Working practice

Inspect the repository and current changes before editing; preserve unrelated
work. Separate hard integrity checks from study-specific plausibility rules,
record failures rather than silently discarding records, and distinguish
association, intervention and counterfactual claims. State uncertainty and
relevant limits including confounding, selection, positivity, follow-up,
censoring, competing events and external validation. Prefer deterministic
fictional fixtures and meaningful tests. Update `docs/PROJECT_STATE.md` after
meaningful milestones or incomplete work, and record concrete decisions under
`docs/decisions/`. Do not implement empty placeholders or all components in the
architecture merely to match the diagram.

Reading order: `README.md`, `docs/PROJECT_STATE.md`, `docs/ARCHITECTURE.md`,
`docs/STUDY_MODEL.md`, then `docs/ROADMAP.md`.

For the experimental neural candidate, preserve the historical log-frequency
configuration as a comparator. Use its explicit empirical-frequency policy only
for bounded diagnostics, keep fit and sample policies aligned, and record the
policy with seeds and environment in aggregate reports. Do not treat changes in
conditioning, ordinal encoding or softmax decoding as clinical validation.
