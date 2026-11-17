# Retrospective study model

The current fictional example uses exposure as the day-0 episode index,
baseline severity and comorbidity as index-available baseline variables,
treatment group as index-time context, and a binary outcome assessed through
day 365 inclusive. Observation coverage and censoring metadata are descriptive
and do not imply complete follow-up or causal interpretation.

Janus should describe many retrospective cohort studies without assuming one
specialty or requiring all studies to share the same variables. A study-specific
configuration maps source data and clinical concepts into general primitives.
Examples and tests must use invented values and make no clinical claims.

## Concepts to represent

1. **Population and observation:** source population, site, available observation
   period, data availability, and inclusion and exclusion criteria.
2. **Unit and index:** patient versus episode, multiple episodes per patient,
   index event and timestamp, and the time at which eligibility is assessed.
3. **Variables:** definitions, units, permissible values, provenance, event time
   versus recording time, measurement process, and missing or unknown status.
4. **Exposure and comparator:** treatment, procedure, or other exposure; timing,
   assignment rules, and explicitly defined comparison groups.
5. **Follow-up:** start and end, outcomes and their ascertainment, censoring,
   loss to follow-up, competing events, and incomplete observation.
6. **Analysis:** target population and estimand, descriptive versus causal
   purpose, subgroup definitions, assumptions, and uncertainty.

Time and missingness are first-class. Absence of a record is not necessarily
absence of an event. A future measurement must not determine baseline eligibility
unless the protocol explicitly defines an appropriate retrospective window and
its interpretation. Tests should cover boundary timestamps and overlapping
episodes, and describe how ambiguous dates are handled.

## Models and interventions

A structural causal model comprises variables, a graph or equivalent dependency
structure, mechanisms, and exogenous variables. It requires justified assumptions
about causal direction, confounding, selection, measurement, and support. A
statistical association is not by itself a mechanism or an intervention effect.

An intervention changes a specified mechanism under declared assumptions; a
counterfactual additionally requires assumptions connecting outcomes for the same
unit across hypothetical worlds. The generated result is conditional on the
model, not observed ground truth. Track which variables are observed, inferred,
or simulated, and document uncertainty and unsupported regions. A hypothetical
scenario outside observed treatment support must be clearly marked extrapolative.

## Evaluation boundaries

Use patient-level splits before fitting preprocessing or models; account for
multiple episodes and temporal/site dependencies. Reserve an untouched test set
and distinguish internal holdout from independent external validation. For each
metric report its population, sample size, uncertainty, missingness, and failure
conditions. Include subgroup analysis where sample size and permissions allow.

Statistical similarity does not prove physiological plausibility, privacy,
clinical utility, or causal validity. A forensic classifier's performance depends
on the datasets, features, and generators tested; it is not a universal detector.
Privacy attacks require a defined threat model and do not prove anonymity.
Clinical score stress tests identify model-dependent cases for expert review,
not changes to patient care.
