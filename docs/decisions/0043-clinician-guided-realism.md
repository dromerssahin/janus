# 0043: Clinician-guided generation and evaluation

Status: governing direction, 2026-11-12. This decision supersedes the model
priorities in [0041](0041-additional-stochastic-generator.md) and
[0042](0042-neural-clinical-realism.md); those decisions remain historical
research and their applicable technical and data-governance cautions remain.
It introduces no implemented generator, review system or validation result.

## Development clarification

The user authorized descriptive synthetic-versus-real technical comparisons
without clinician sign-off or a reserved reference. Those are not clinical
validity or privacy claims; label any inspected reference and its limitations.
Clinical review and a suitably independent reserved reference remain relevant
before a clinical-validity claim, not prerequisites for exploratory development
or comparing distributions. Honor source terms, patient-grouping uncertainty,
and the existing exclusions; do not call an inspected holdout untouched.

## Goal

Janus is a study-agnostic research engine for synthetic clinical cohorts. An
organic cohort is one in which a clinician can read each patient's variables
together and find the combination plausible, including unusual patients,
missing observations and realistic variation. Clinical claims of realism require
clinician-guided assessment, not a single model or score; exploratory technical
comparisons may proceed independently. More complex
models are not assumed to improve realism; at current cohort sizes, data
diversity and clinician feedback are expected to matter more. This expectation
is a research hypothesis, not a measured result.

## Loop

1. **Specify coherence for clinical assessment.** Qualified clinicians and study
   researchers jointly author and version a per-study clinical specification. It must
   describe population, patient unit, variable meanings and availability,
   timing and units, missing versus structural absence, clinically meaningful
   relationships and unusual cases, hard integrity constraints versus reviewable
   plausibility concerns, evidence and assumptions, limitations, and the
   criteria for evaluation and handling violations. Document uncertainty and
   disagreements. It is not a universal rubric or a license to discard unusual
   patients. If clinician input is unavailable, mark a draft as unreviewed.
2. **Generate comparable candidates.** Retain existing statistical generators
   as baselines, using patient-level development partitions and consistent
   schemas, training populations and sample sizes. Neural tabular methods are
   candidates, not default winners: CTGAN-style and TabPFN-style methods may be
   considered along with others when the protocol and evidence justify them.
   Their mixed-type, missingness, rare-state, licensing, access and reproducible
   integration requirements must be verified. No candidate or new model is
   currently selected for implementation by this decision.
3. **Review records blindly.** Prepare a protocol for qualified clinicians to
   review a permitted, blinded mixture of real and synthetic development or
   separately designated review records. Specify presentation, sampling,
   blinding, severity-rated implausibilities, unusual cases, missingness,
   reviewer independence, disagreement, aggregation and limitations. Verify
   source permission and access before presentation. Do not use the final
   reserved reference for iterative review or selection; reviews can themselves
   expose outcomes and contaminate a reference.
4. **Check cohorts across seeds.** Prespecify comparable checks for rare
   combinations, missingness patterns, associations and correlations, duplicate
   and proximity diagnostics, downstream utility and constraints, including
   violations before and after any repair or rejection. Record denominators,
   unsupported cases, uncertainty where validated, failures, and all seeds;
   distinguish repeated draws from independent patients. Report clinical review
   and quantitative results together when available; neither alone establishes validity.
5. **Freeze and evaluate once.** After development, freeze the selected method,
   specification, mappings, review/evaluation protocol and metrics before
   examining outcomes in a genuinely reserved, permitted reference. Apply
   [intake gates](../INDEPENDENT_EVALUATION_INTAKE.md), including documented
   patient independence, timing, permissions and schema. Run the frozen
   evaluation once and report limitations; subsequent changes require a new
   protocol and genuinely reserved reference. Do not treat previously inspected
   holdouts or outcomes as untouched references for selection or final claims.

## Boundaries and next action

Source terms govern review, temporary data, fitted models, outputs and retention.
For the aggregate-only local study, rows, predictions and fitted models remain
in memory; blind presentation requires permitted access and cannot create
persisted patient-level review artifacts. Store only allowed aggregate reports
and provenance under ignored `datasets/`. Do not infer privacy safety from
synthetic data or duplication checks. A separate threat model and assessment are
required for a privacy claim; clinical, causal and transport validity are also
unproven. Respect existing source-specific constraints and intake gates.

Reproducibility requires seeds, algorithms, configuration, software versions,
inputs and permitted fingerprints, execution environment and relevant numerical
and threading details. Same-seed replay applies only under the verified
conditions; different seeds need not give different outputs and cross-platform
bit identity is not promised. The current fitted statistical generators already
sample stochastically and remain unchanged baselines.

Technical comparisons can proceed using available permitted real data and the
existing generators. The coherence and review drafts remain unreviewed; do not
fabricate clinician input or call them reviewed. The existing aggregate
evaluation harness does not establish source permissions or independence.
