# Project state and handoff

## Fresh-session handoff (2026-11-13)

The primary goal is a study-agnostic synthetic-cohort research engine; see
`README.md` for supported workflows. The current evidence supports a bounded
technical-validity milestone only. Contracts, candidate comparison conditions,
fictional review mechanics, and inspected aggregate comparisons are implemented
and tested. No clinical-plausibility, independent-validation, causal, privacy,
transportability, or model-superiority claim is supported.

Start with `AGENTS.md`, this section, and `README.md`. The key implementation
artifacts are `crates/janus/src/coherence.rs`, `crates/janus/src/review.rs`,
`janus-cli validate-contract`, and `docs/SUWA_SAH_COHERENCE_CONTRACT.md`.
The existing Suwa SAH, HeMoVal, and EARLYDRAIN outcomes were inspected;
different sites or periods do not prove patient independence. An approved
independent reference is a future extension governed by
`docs/INDEPENDENT_EVALUATION_INTAKE.md`.

Janus is a study-agnostic research engine for generating and evaluating clinical
cohorts. The goal is plausible combinations of patient variables, including
missingness and unusual cases. See `README.md` for implemented workflows;
`docs/ARCHITECTURE.md` is a proposed design, not a feature checklist. Statistical
similarity does not establish clinical, causal or privacy validity.

## Current capabilities and evidence

- Flat Rust forest and DAG baselines learn associations and sample stochastically
  in memory; fixed seeds replay in the verified environment. Evaluators provide
  aggregate fidelity and predictive utility, but do not establish permissions,
  timing, patient independence or clinical plausibility. An experimental
  standalone neural generator is implemented; clinical-reader evaluation is
  not. See `README.md` and decision 0044 for workflows and limitations.
- The latest exploratory EARLYDRAIN analysis compared existing and weight-first
  orders across three overlapping development splits, four baselines and ten
  seeds. Targeted sex/weight and weight/WFNS differences were mixed; mean
  pairwise TVD worsened in five of six split/model groups. The pairs were chosen
  after inspecting prior results; neither order is a validated default. Its
  aggregate-only protocol, results and replay metadata are under ignored
  `datasets/results/earlydrain-order-sensitivity-20261112/`.
- The matched real-versus-real comparator used 46 training patients versus 46
  reference patients. Mean eight-field joint TVD was 0.9000, 0.9000 and 0.9326
  across splits, illustrating sparse-state limits. The comparator comes from the
  generator's training population; it is not independent validation. See
  ignored `datasets/results/earlydrain-real-calibration-20261112/` and
  `datasets/results/earlydrain-wide-comparison-20261112/`. Field meanings and
  timing remain incompletely documented; seed variation is not a confidence
  interval. No clinical, causal or privacy claim follows.
- Earlier HeMoVal development work found the unregularized forest retained a
  WFNS/mRS association missed by the penalized DAGs; its exploratory advantage
  did not establish a general winner. The cranial/spine comparison showed a
  different tradeoff. Its encounter keys are not verified patient keys, and one
  full row including an outcome was inadvertently displayed during header
  inspection; outcomes cannot be called wholly uninspected. Detailed reports
  and provenance remain under ignored `datasets/results/`.
- The latest review reported no actionable introduced defects and a passing
  full workspace test run via `cargo test --workspace --all-targets --offline -q`; it assessed
  moderate risk from broad shared implementation and public APIs. The most
  recent experiment also passed fictional metric tests and replayed in its
  recorded environment. These checks do not establish validity.
- Neural backend feasibility: Burn 0.22.0's documented autodiff supports only
  first-order derivatives, whereas reference CTGAN's gradient penalty requires
  higher-order differentiation. A separate Rust `tch` 0.20.0 / PyTorch 2.7.0
  CPU probe computed a derivative through a derivative and trained a small
  fictional WGAN with gradient penalty at 153, 229 and 298 rows. Weights changed,
  losses were finite and identical fit seeds replayed locally. This is not a
  mixed-data CTGAN, a cohort comparison or evidence of realistic generation.
  The probe and its pinned Rust manifest are under ignored
  `datasets/results/tch-neural-preflight-20261112/`; its PyTorch and Cargo
  dependencies under `/tmp` are ephemeral. `tch` 0.20.0 rejected PyTorch
  2.7.1's version; the successful probe used 2.7.0. Do not bypass that check.
  CUDA was unavailable in this execution environment; the user's RTX 4060
  laptop GPU and Intel i5-12450 are the intended hardware, but GPU operation
  and reproducibility have not been tested. The installed PyTorch was CPU-only.
  The `oxicuda-tabular` CTGAN source describes fixed network bodies and an
  adversarial term that does not update the generator, so its training scaffold
  is not a substitute for end-to-end CTGAN training. See the probe summary and
  earlier Burn preflight for commands, sources and limitations.

## Constraints that must not change without reason

- Technical synthetic-versus-real comparisons on permitted data do not require
  clinician sign-off or a reserved reference. Identify prior inspection and do
  not make clinical-validity or privacy claims. Clinical assessment and an
  appropriate genuinely reserved reference are needed for stronger clinical
  claims; see decision 0043. The unreviewed HeMoVal coherence and review drafts
  are `docs/HEMOVAL_COHERENCE_DRAFT.md` and
  `docs/HEMOVAL_BLIND_REVIEW_DRAFT.md`.
- Honor each source's terms. For EARLYDRAIN, keep real and synthetic rows,
  predictions and fitted models in memory and save only permitted aggregates
  under ignored `datasets/`; its CSV has no diagnosis timestamp. Blank and `NA`
  are missing unless explicitly structural absence; neither is a utility label.
  Do not infer missing outcomes from death or silently substitute variables.
- EARLYDRAIN holdout, Suwa and HeMoVal outcomes have been inspected; never call
  them untouched. For comparable EARLYDRAIN development work, reconstruct its
  allocation-stratified seed-41 split and exclude the 58-patient outer holdout;
  the development partition has 229 patients. For HeMoVal, reconstruct the
  original seed-101 reservation over all 442 source keys before eligibility, and exclude
  reserved keys before extracting WFNS or mRS. Preserve documented selection and
  consent rules; do not rerank selected patients. Decision 0040 is a frozen
  pre-outcome protocol, not the current inspection status.
- The acquired datasets and provenance are under ignored `datasets/`; do not
  redownload them. JinhuaNSICU's public schema was checked but the source is
  listed as controlled access and patient data were not downloaded. The user
  stopped open-ended source search. For a newly offered reference and a frozen
  task, apply `docs/INDEPENDENT_EVALUATION_INTAKE.md`; see
  `docs/NEUROSURGICAL_SOURCE_LEADS.md` for source-specific limitations.
- Preserve pre-fit rejection of nonempty holdout groups with no development
  rows, and unbiased bounded sampling in `flat_sample_index`. Keep their
  regression tests. Preserve reproducibility metadata and note that seeds alone
  do not guarantee identical output across versions or environments.
- For the aggregate-only study, do not serialize preprocessing state, model
  weights or checkpoints, write temporary patient or synthetic rows, or print
  records or predictions. An in-memory model can retain source-derived values.

## Experimental neural milestone (2026-11-12)

The optional standalone `experimental/neural` Rust crate now implements an
in-memory conditional WGAN with mixed numeric/categorical fields, explicit
missing states, gradient penalty and independent fit/sample seeds. Structural
absence is rejected. It is CTGAN-style, not a full CTGAN implementation; see
decision 0044 and the crate README. The default workspace remains independent
of LibTorch. CPU fictional feasibility tests passed for 153, 229 and 298 rows,
including both networks' weight updates, finite losses and outputs, rare states,
dependent missingness in the input, constants, invalid inputs, replay and seed
variation. The tests do not establish recovery of missingness dependencies.
The final test process including Cargo used 120772 KiB peak child RSS and
0.84 s on the local PyTorch 2.7.0+cpu / tch 0.20.0 setup. CPU tests
use one LibTorch thread. No CUDA backend was available or tested.

An exploratory eight-field EARLYDRAIN comparison excluded the 58 outer holdout
patients and used inner splits 101, 202 and 303 of the 229 development patients.
Five neural fit seeds and two sample seeds per fit were compared against ten
draws from each forest with the same 183 training and 46 reference patients.
Mean neural marginal TVD was 0.398–0.414 and mean pairwise TVD 0.610–0.624;
forest ranges were 0.106–0.119 and 0.208–0.219. Neural eight-field joint TVD
was 1.0 in every draw. Reported schema integrity violations were zero, but
numeric ordinal scores were treated as continuous and can be fractional; this
check does not establish clinical plausibility. Real-versus-real context uses
46 training patients, not the 183 synthetic rows used in this experiment.
The inspected references, overlapping splits, sparse states and limited seeds
preclude clinical, causal, privacy or independent-validation claims. Detailed
aggregates and provenance are under ignored `datasets/results/neural-comparison-20261112/`.

## Next action

The first bounded neural diagnostic is implemented in decision 0045. The
candidate now makes conditional-state sampling explicit: the historical
log-frequency policy remains the default, while empirical-frequency sampling
can be compared on identical fictional or permitted development inputs. The
fit policy is retained for sampling, and an unused duplicate condition batch
was removed from training. CPU fictional tests pass for both policies. A
same-seed aggregate comparison on the three existing development splits found
mean marginal TVD 0.4066 (log) versus 0.4081 (empirical), pairwise TVD 0.6151
versus 0.6167, and joint TVD 1.0 for both; empirical conditioning therefore
did not improve the observed gap. The aggregate report is under ignored
`datasets/results/neural-comparison-20261112/conditioning-diagnostic.yaml`.
This does not justify selecting the neural candidate; ordinal-value
representation, rare-state recovery and joint sparsity remain open. Preserve
the baselines and source constraints, and test CUDA only with a compatible GPU
environment. Do not infer undocumented clinical relationships or call
inspected references untouched. See `AGENTS.md` for contributor constraints.

## Whole-patient realism plan audit (2026-11-12)

The plan's repository-contained deliverables are present as drafts:
`docs/HEMOVAL_COHERENCE_DRAFT.md` defines a bounded admission-WFNS / follow-up-
mRS development task, and `docs/HEMOVAL_BLIND_REVIEW_DRAFT.md` defines a
proposed blinded review protocol. Both explicitly remain unreviewed by
clinicians and study researchers; real-record review is not ready to run.
Fictional review mechanics, protocol-manifest validation, and a bounded Suwa
technical contract are implemented. Existing technical comparisons and their
limitations are described above; no model is selected.

The source cohorts are real-world by their documented provenance. That fact
does not require clinician review. Clinical review addresses plausibility of
generated patient combinations, not whether source patients are real. Local
aggregate analysis can proceed under documented source terms; third-party
presentation of records remains a separate permission question. The blind
review draft states this distinction.

The existing HeMoVal runner was rerun locally from its fingerprint-checked
source using the original seed-101 reservation and the 298/68 patient split.
All 442 input keys reconciled as 366 selected, 52 source-excluded, 9 test-site,
and 15 ambiguous-metadata keys. The two DAGs produced identical results on
this two-variable task: real-trained Brier 0.1752 and synthetic-trained mean
Brier 0.2205 across seeds 11, 22 and 33, with byte-identical same-environment
replay. The split is internal, outcomes and published aggregates were already
inspected, actual follow-up timing is unverified, and confidence intervals are
unavailable. This does not establish clinical plausibility, external validity,
or a winner; neither baseline is selected. Details are in the aggregate-only
`datasets/hemoval/summary.yaml` and `manifest.json`.

No independent evaluation was run because the intake has no source ready for
its frozen task; inspected references remain excluded from untouched-validation
claims. Workspace tests, formatting, Clippy with warnings denied, and
`git diff --check` passed in this audit. These
technical checks do not validate clinical assumptions. The optional neural
crate was not runnable offline in this environment because its `tch` dependency
is absent from the local Cargo cache; no neural result is claimed from that
failed command.

## Contract and protocol-manifest milestone (2026-11-12)

The core now exposes machine-validated `CoherenceContract` and
`ProtocolManifest` types in `crates/janus/src/coherence.rs`. Fictional HeMoVal
examples exercise required meanings, timing statements, allowed states,
missingness semantics, candidate identity, populations, seeds, software,
fingerprints and reservation status. Validation rejects incomplete contracts,
duplicate variables, empty allowed states and incomplete manifests. This is a
schema and integrity gate only: it does not approve the draft, encode clinical
plausibility, prove independence, or implement blinded review. See decision
0047. The next dependency remains clinician/study-researcher review of the
contract and a permitted independent reference; no such reference is ready.

## Fictional review harness milestone (2026-11-12)

The core now includes an in-memory review harness in
`crates/janus/src/review.rs`. Fictional tests cover blinding, prohibited
fields, missing-value display, duplicate/group handling, reserved-key
exclusion, reviewer disagreement, insufficient-information responses and
small-cell suppression. The aggregate result contains only counts and does not
retain rows or individual ratings. This is a workflow test, not clinical
review, anonymization, permission approval or evidence of plausibility.

The same module also validates `CandidateComparisonManifest`, requiring at
least two uniquely named candidates, identical preprocessing, shared mapping
fingerprint, populations, sample counts, seeds and metrics. This freezes the
comparison conditions but does not select a winner; the existing forests remain
baselines and the neural candidate remains exploratory.

The CLI command `janus-cli validate-contract` now validates the contract and,
optionally, the protocol and candidate-comparison manifests together before a
workflow is run. A fictional invocation using all three example files passed.

## Independent-source search resumed (2026-11-12)

At the user's request, a metadata-only web search resumed. The NYU
“Neurological Emergencies Outcomes” catalog is the strongest newly identified
lead: it describes adult ICH/SAH admissions, admission examination and CT data,
hospital complications, discharge mRS and infection-focused outcomes. The
catalog requires an access application and author approval; exact frozen
sedation/infarct/infection mappings and patient non-overlap are not yet proven.
No patient data were downloaded or inspected. MIMIC-IV-Ext Note Extract was
also reviewed, but its documented fields do not establish the frozen task
mappings. See `docs/NEUROSURGICAL_SOURCE_LEADS.md`.

The credential-free alternative is now documented as a bounded Suwa SAH
technical contract in `docs/SUWA_SAH_COHERENCE_CONTRACT.md`. Its existing
aggregate comparison is reproducible and uses the same seeds, candidate
baselines, sample sizes and metrics. It remains inspected cross-cohort
development evidence, not independent validation; outcome timing, sex mapping
and patient-level non-overlap remain limited as documented.

## Bounded technical-validity completion (2026-11-13)

Per decision 0052, the project is complete at a deliberately bounded technical
milestone. The repository now has the study contract, protocol and candidate
comparison manifests, CLI preflight validation, fictional aggregate-only review
harness, inspected HeMoVal and Suwa technical comparisons, reproducibility
metadata, and passing workspace checks. The supported claim is that this
specified workflow can be executed and audited on the available fictional and
inspected development evidence. No clinical-plausibility, independent,
causal, privacy, transportability, or model-superiority claim is made.

An approved independent reference remains a future extension rather than a
missing prerequisite for this bounded technical milestone.

## Phase 4: Production Forensics & Candidate Infrastructure (2026-11-13)

Per decision 0053, Phase 4 adds production forensic workflows and unified
candidate-comparison infrastructure:

- **Summary**: forensic workflow orchestrator with multi-seed support and
  aggregate-only manifests, unified `Candidate` generator trait with
  in-memory-only comparison enforcement, `janus-cli audit-reproducibility`
  command, and operational documentation for forensic and neural workflows.
- **Implementation artifacts**:
  - `crates/janus/src/forensic_workflow.rs` (1,070 lines)
  - `crates/janus/src/candidate.rs` (1,236 lines)
  - `docs/FORENSIC_WORKFLOWS.md`
  - `docs/NEURAL_OPERATIONS.md`
  - `janus-cli audit-reproducibility`
- **Tests**: 139 workspace tests passing, zero clippy warnings.
- **Known limitations**:
  - The experimental neural bridge still writes `samples.csv` (an
    aggregate-only violation documented for Phase 5).
  - GPU operation and GPU reproducibility remain untested.
  - No baseline-neural orchestration runner is implemented yet (future work).
- **Next actions**: remains user-directed—Phase 4 completes infrastructure for
  candidate comparison; no clinical, causal, or privacy claim follows.

## Phase 5: In-Memory Neural IPC, Candidate Comparison & End-to-End CLI Orchestration (2026-11-16)

Phase 5 completes end-to-end CLI orchestration, in-memory neural IPC, and
multi-candidate comparison while keeping `experimental/neural/` untouched and
preserving all `AGENTS.md` bounded-claim limits:

- **In-memory neural IPC (`--stdio`)**:
  - `scripts/neural_bridge.py --stdio` and
    `crates/janus/src/neural.rs::run_neural_bridge_in_memory` transfer training
    rows, `fit_seed`, `sample_seeds`, and sampled rows over JSON stdio without
    writing `samples.csv` or `model.pkl` in aggregate-only workflows.
  - `NeuralGeneratorManifest` adds optional `fixture_fingerprint` so standard
    fictional `janus-cli generate --generator neural-ctgan` runs that match a
    `Study` emit `fixture.yaml` and integrate with
    `janus-cli evaluate --synthetic-manifest`.
- **Candidate comparison & interaction search**:
  - `crates/janus/src/candidate.rs` includes `InteractionDagCandidate` and
    `NeuralCandidate` alongside `ForestCandidate` and `DagCandidate`.
  - `janus-cli compare-candidates` runs `compare_candidates` and
    `orchestrate_forensic_suite` strictly in memory across shared splits,
    denominators, and seeds, emitting only `comparison_manifest.json` and
    `forensic_manifests.json`.
- **CLI upgrades & Phase 3 validation hooks**:
  - `janus-cli generate --generator neural-ctgan --in-memory-aggregate` emits
    aggregate observed/missing variable summaries (`aggregate_summary.json`)
    and `manifest.json` only, with no synthetic rows or model weights on disk.
  - `janus-cli forensic-eval` adds `--suite`, `--training`, and `--seeds` to
    run `orchestrate_forensic_suite` (Chi-squared, KS, Logistic Regression,
    Random Forest, `<10` row small-cell suppression) and adds Chi-squared tests
    for categorical columns in legacy single-seed mixed-cohort mode.
  - `janus-cli validate-contract` adds optional `--neurosurgical-cohort`,
    `--survival-real`, `--survival-synthetic`, and `--output` hooks invoking
    `validate_neurosurgical_cohort` and `compare_survival_curves`.
  - `audit-reproducibility` supports both `fnv1a64-yaml` and `sha256` lockfile
    fingerprints.
- **Bounded claims & remaining limits**:
  - Technical validity, reproducibility, and aggregate comparisons remain
    bounded to recorded software, seeds, and fictional or inspected inputs.
  - No clinical validity, independent validation, causality, privacy safety, or
    GPU reproducibility claim is established; GPU operation remains untested.
