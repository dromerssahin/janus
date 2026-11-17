//! Study-specific coherence contracts and reproducibility manifests.
//!
//! These types deliberately describe the contract and its limits; they do not
//! encode clinical plausibility rules or infer timing and missingness.
use crate::Error;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use statrs::distribution::{ChiSquared, ContinuousCDF};
use std::collections::{BTreeMap, BTreeSet, HashSet};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct CoherenceContract {
    pub schema_version: String,
    pub contract_id: String,
    pub status: String,
    pub study: String,
    pub patient_unit: String,
    pub variables: Vec<ContractVariable>,
    pub limitations: Vec<String>,
    pub unresolved_assumptions: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ContractVariable {
    pub name: String,
    pub role: String,
    pub meaning: String,
    pub timing: String,
    pub units: String,
    pub allowed_values: Vec<String>,
    pub missing_is_structural_absence: bool,
}

impl CoherenceContract {
    pub fn validate(&self) -> Result<(), Error> {
        if self.schema_version != "1" || self.status.trim().is_empty() {
            return Err(Error::Input(
                "unsupported or incomplete coherence contract".into(),
            ));
        }
        if [&self.contract_id, &self.study, &self.patient_unit]
            .iter()
            .any(|s| s.trim().is_empty())
            || self.variables.is_empty()
        {
            return Err(Error::Input(
                "coherence contract has empty required fields".into(),
            ));
        }
        let mut names = HashSet::new();
        for variable in &self.variables {
            if [
                &variable.name,
                &variable.role,
                &variable.meaning,
                &variable.timing,
                &variable.units,
            ]
            .iter()
            .any(|s| s.trim().is_empty())
                || variable.allowed_values.is_empty()
                || variable.allowed_values.iter().any(|v| v.trim().is_empty())
                || !names.insert(&variable.name)
            {
                return Err(Error::Input(format!(
                    "invalid or duplicate contract variable: {}",
                    variable.name
                )));
            }
        }
        if self.limitations.is_empty() || self.limitations.iter().any(|s| s.trim().is_empty()) {
            return Err(Error::Input(
                "coherence contract must state limitations".into(),
            ));
        }
        if self
            .unresolved_assumptions
            .iter()
            .any(|s| s.trim().is_empty())
        {
            return Err(Error::Input(
                "contract has an empty unresolved assumption".into(),
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ProtocolManifest {
    pub schema_version: String,
    pub protocol_id: String,
    pub contract_id: String,
    pub candidate: String,
    pub preprocessing: String,
    pub populations: Vec<String>,
    pub seeds: Vec<u64>,
    pub software: String,
    pub source_fingerprints: Vec<String>,
    pub reviewer_protocol_version: Option<String>,
    pub reservation_status: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct CandidateComparisonManifest {
    pub schema_version: String,
    pub protocol_id: String,
    pub contract_id: String,
    pub candidates: Vec<ProtocolCandidate>,
    pub mapping_fingerprint: String,
    pub populations: Vec<String>,
    pub sample_count: usize,
    pub seeds: Vec<u64>,
    pub metrics: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ProtocolCandidate {
    pub name: String,
    pub preprocessing: String,
    pub status: String,
}

impl CandidateComparisonManifest {
    pub fn validate(&self) -> Result<(), Error> {
        if self.schema_version != "1"
            || self.protocol_id.trim().is_empty()
            || self.contract_id.trim().is_empty()
            || self.mapping_fingerprint.trim().is_empty()
            || self.sample_count == 0
            || self.candidates.len() < 2
            || self.populations.is_empty()
            || self.seeds.is_empty()
            || self.metrics.is_empty()
            || self.populations.iter().any(|v| v.trim().is_empty())
            || self.metrics.iter().any(|v| v.trim().is_empty())
            || self.candidates.iter().any(|candidate| {
                candidate.name.trim().is_empty()
                    || candidate.preprocessing.trim().is_empty()
                    || candidate.status.trim().is_empty()
            })
        {
            return Err(Error::Input(
                "incomplete candidate comparison manifest".into(),
            ));
        }
        let names: HashSet<_> = self
            .candidates
            .iter()
            .map(|candidate| &candidate.name)
            .collect();
        if names.len() != self.candidates.len() {
            return Err(Error::Input(
                "candidate comparison has duplicate candidates".into(),
            ));
        }
        let preprocessing: HashSet<_> = self
            .candidates
            .iter()
            .map(|candidate| &candidate.preprocessing)
            .collect();
        if preprocessing.len() != 1 {
            return Err(Error::Input(
                "candidate comparison preprocessing is not identical".into(),
            ));
        }
        Ok(())
    }
}

impl ProtocolManifest {
    pub fn validate(&self) -> Result<(), Error> {
        if self.schema_version != "1"
            || self.protocol_id.trim().is_empty()
            || self.contract_id.trim().is_empty()
            || self.candidate.trim().is_empty()
            || self.preprocessing.trim().is_empty()
            || self.software.trim().is_empty()
            || self.reservation_status.trim().is_empty()
            || self.populations.is_empty()
            || self.seeds.is_empty()
            || self.source_fingerprints.is_empty()
            || self.populations.iter().any(|s| s.trim().is_empty())
            || self.source_fingerprints.iter().any(|s| s.trim().is_empty())
        {
            return Err(Error::Input("incomplete protocol manifest".into()));
        }
        Ok(())
    }
}

/// Explicit representation of an optional field value that distinguishes missingness
/// (`""` or `"NA"`) from observed tokens without inferring values or assuming structural absence.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum OptionalField<T> {
    Observed(T),
    Missing,
}

impl<T> OptionalField<T> {
    pub fn as_option(&self) -> Option<&T> {
        match self {
            Self::Observed(value) => Some(value),
            Self::Missing => None,
        }
    }

    pub fn is_missing(&self) -> bool {
        matches!(self, Self::Missing)
    }
}

/// Parse a raw string cell into `OptionalField<f64>`.
/// Blank strings and `"NA"` are treated as `Missing`, never structural absence.
pub fn parse_optional_f64(raw: &str, field_name: &str) -> Result<OptionalField<f64>, String> {
    let trimmed = raw.trim();
    if trimmed.is_empty() || trimmed.eq_ignore_ascii_case("NA") {
        return Ok(OptionalField::Missing);
    }
    match trimmed.parse::<f64>() {
        Ok(val) if val.is_finite() => Ok(OptionalField::Observed(val)),
        _ => Err(format!(
            "invalid numeric token '{trimmed}' for field '{field_name}'"
        )),
    }
}

/// Parse a raw string cell into `OptionalField<u8>`.
/// Blank strings and `"NA"` are treated as `Missing`, never structural absence.
pub fn parse_optional_u8(raw: &str, field_name: &str) -> Result<OptionalField<u8>, String> {
    let trimmed = raw.trim();
    if trimmed.is_empty() || trimmed.eq_ignore_ascii_case("NA") {
        return Ok(OptionalField::Missing);
    }
    match trimmed.parse::<u8>() {
        Ok(val) => Ok(OptionalField::Observed(val)),
        Err(_) => Err(format!(
            "invalid integer token '{trimmed}' for field '{field_name}'"
        )),
    }
}

/// Distinguishes hard data/chronological integrity checks from study-specific
/// clinical plausibility rules.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ValidationCategory {
    /// Hard structural, chronological, or domain-range integrity violation.
    Integrity,
    /// Study-specific clinical coherence / plausibility rule violation.
    Plausibility,
}

/// Recorded validation failure for a single patient record.
/// Failures are explicitly recorded rather than silently discarding records.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ValidationFailure {
    pub record_id: String,
    pub category: ValidationCategory,
    pub rule: String,
    pub message: String,
}

/// Summary report of cohort validation preserving all record-level failures and missingness counts.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ValidationReport {
    pub total_records: usize,
    pub valid_records: usize,
    pub rejected_records: usize,
    pub integrity_failure_count: usize,
    pub plausibility_failure_count: usize,
    pub missing_timestamp_count: usize,
    pub missing_grade_count: usize,
    pub failures: Vec<ValidationFailure>,
}

impl ValidationReport {
    pub fn is_valid(&self) -> bool {
        self.failures.is_empty()
    }
}

/// Patient timestamps (in hours from reference epoch or baseline) across the care pathway:
/// Admission -> Surgery -> Discharge -> 90-day Follow-up.
/// Missing timestamps are represented as `None` and are never inferred.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TemporalTransitionRecord {
    pub record_id: String,
    pub admission_time_hours: Option<f64>,
    pub surgery_time_hours: Option<f64>,
    pub discharge_time_hours: Option<f64>,
    pub followup_90d_time_hours: Option<f64>,
}

impl TemporalTransitionRecord {
    /// Construct a `TemporalTransitionRecord` from raw string fields (`""` and `"NA"` -> `None`),
    /// recording any unparseable tokens as `ValidationCategory::Integrity` failures.
    pub fn from_raw_strings(
        record_id: impl Into<String>,
        admission_raw: &str,
        surgery_raw: &str,
        discharge_raw: &str,
        followup_90d_raw: &str,
    ) -> Result<Self, Vec<ValidationFailure>> {
        let record_id = record_id.into();
        let mut failures = Vec::new();
        let mut parse_field = |raw: &str, name: &str| -> Option<f64> {
            match parse_optional_f64(raw, name) {
                Ok(OptionalField::Observed(v)) => Some(v),
                Ok(OptionalField::Missing) => None,
                Err(msg) => {
                    failures.push(ValidationFailure {
                        record_id: record_id.clone(),
                        category: ValidationCategory::Integrity,
                        rule: "timestamp_token_integrity".into(),
                        message: msg,
                    });
                    None
                }
            }
        };

        let admission_time_hours = parse_field(admission_raw, "admission_time_hours");
        let surgery_time_hours = parse_field(surgery_raw, "surgery_time_hours");
        let discharge_time_hours = parse_field(discharge_raw, "discharge_time_hours");
        let followup_90d_time_hours = parse_field(followup_90d_raw, "followup_90d_time_hours");

        if failures.is_empty() {
            Ok(Self {
                record_id,
                admission_time_hours,
                surgery_time_hours,
                discharge_time_hours,
                followup_90d_time_hours,
            })
        } else {
            Err(failures)
        }
    }

    /// Count how many of the 4 pathway timestamps are missing on this record.
    pub fn missing_timestamp_count(&self) -> usize {
        [
            self.admission_time_hours,
            self.surgery_time_hours,
            self.discharge_time_hours,
            self.followup_90d_time_hours,
        ]
        .iter()
        .filter(|t| t.is_none())
        .count()
    }
}

/// Validate temporal transitions for a single record:
/// Admission -> Surgery -> Discharge -> 90-day Follow-up.
///
/// - Never infers missing timestamps.
/// - Checks that all observed timestamps are finite numbers.
/// - Checks chronological ordering (`t_earlier <= t_later`) across every observed pair in the
///   sequence `admission -> surgery -> discharge -> followup_90d`.
pub fn validate_temporal_transitions(record: &TemporalTransitionRecord) -> Vec<ValidationFailure> {
    let mut failures = Vec::new();
    if record.record_id.trim().is_empty() {
        failures.push(ValidationFailure {
            record_id: record.record_id.clone(),
            category: ValidationCategory::Integrity,
            rule: "record_id_present".into(),
            message: "record_id must not be empty".into(),
        });
    }

    let stages: [(&str, Option<f64>); 4] = [
        ("admission", record.admission_time_hours),
        ("surgery", record.surgery_time_hours),
        ("discharge", record.discharge_time_hours),
        ("followup_90d", record.followup_90d_time_hours),
    ];

    for (name, value) in &stages {
        if let Some(t) = value {
            if !t.is_finite() {
                failures.push(ValidationFailure {
                    record_id: record.record_id.clone(),
                    category: ValidationCategory::Integrity,
                    rule: "finite_timestamp".into(),
                    message: format!("{name} timestamp must be finite, got {t}"),
                });
            }
        }
    }

    // Check pairwise chronological ordering across all observed finite timestamps in sequence.
    for i in 0..stages.len() {
        for j in (i + 1)..stages.len() {
            let (earlier_name, earlier_opt) = stages[i];
            let (later_name, later_opt) = stages[j];
            if let (Some(t_early), Some(t_late)) = (earlier_opt, later_opt) {
                if t_early.is_finite() && t_late.is_finite() && t_late < t_early {
                    failures.push(ValidationFailure {
                        record_id: record.record_id.clone(),
                        category: ValidationCategory::Integrity,
                        rule: "temporal_transition_order".into(),
                        message: format!(
                            "{earlier_name} -> {later_name} violates chronological sequence ({t_early} > {t_late})"
                        ),
                    });
                }
            }
        }
    }

    failures
}

/// Validate surgical timing window (<72h from admission to surgery when surgery timestamp is present).
///
/// - If `surgery_time_hours` is `None`, missingness is preserved without error or inference.
/// - If `surgery_time_hours` is `Some` and `admission_time_hours` is `None`, records a
///   `ValidationCategory::Integrity` failure (admission time is never inferred).
/// - If `surgery_time_hours < admission_time_hours`, records a `ValidationCategory::Integrity`
///   chronological failure.
/// - If `surgery_time_hours - admission_time_hours >= 72.0`, records a
///   `ValidationCategory::Plausibility` failure for the `<72h` neurosurgical window contract.
pub fn validate_surgical_window_72h(record: &TemporalTransitionRecord) -> Vec<ValidationFailure> {
    let mut failures = Vec::new();
    let Some(surgery_t) = record.surgery_time_hours else {
        return failures;
    };
    let Some(admission_t) = record.admission_time_hours else {
        failures.push(ValidationFailure {
            record_id: record.record_id.clone(),
            category: ValidationCategory::Integrity,
            rule: "surgical_window_requires_admission_timestamp".into(),
            message: "surgery timestamp is present but admission timestamp is missing; admission timestamp cannot be inferred".into(),
        });
        return failures;
    };
    if !surgery_t.is_finite() || !admission_t.is_finite() {
        failures.push(ValidationFailure {
            record_id: record.record_id.clone(),
            category: ValidationCategory::Integrity,
            rule: "finite_timestamp".into(),
            message: "admission and surgery timestamps must be finite".into(),
        });
        return failures;
    }
    let elapsed = surgery_t - admission_t;
    if elapsed < 0.0 {
        failures.push(ValidationFailure {
            record_id: record.record_id.clone(),
            category: ValidationCategory::Integrity,
            rule: "temporal_transition_order".into(),
            message: format!(
                "surgery timestamp ({surgery_t}) precedes admission timestamp ({admission_t})"
            ),
        });
    } else if elapsed >= 72.0 {
        failures.push(ValidationFailure {
            record_id: record.record_id.clone(),
            category: ValidationCategory::Plausibility,
            rule: "surgical_window_under_72h".into(),
            message: format!(
                "surgery_time - admission_time must be < 72.0h when surgery timestamp is present, got {elapsed}h"
            ),
        });
    }
    failures
}

/// Scale type for neurological severity grading.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum NeuroGradeScale {
    Wfns,
    HuntHess,
}

impl NeuroGradeScale {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Wfns => "wfns",
            Self::HuntHess => "hunt_hess",
        }
    }
}

/// Validate WFNS or Hunt-Hess grade progression between initial and subsequent assessment.
///
/// - Hard integrity check: observed grades must be within `1..=5`.
/// - Missing grades (`None`) are preserved without inference.
/// - Study-specific clinical coherence contract: `initial_grade <= subsequent_grade` (where higher
///   numbers indicate worse severity, so `initial_grade > subsequent_grade` represents clinical
///   improvement) is required unless `documented_intervention` is `true`.
pub fn validate_grade_progression(
    record_id: &str,
    scale: NeuroGradeScale,
    initial_grade: Option<u8>,
    subsequent_grade: Option<u8>,
    documented_intervention: bool,
) -> Vec<ValidationFailure> {
    let mut failures = Vec::new();
    let scale_name = scale.as_str();

    if let Some(g) = initial_grade {
        if !(1..=5).contains(&g) {
            failures.push(ValidationFailure {
                record_id: record_id.into(),
                category: ValidationCategory::Integrity,
                rule: format!("{scale_name}_grade_range"),
                message: format!("initial {scale_name} grade {g} is outside valid domain 1..=5"),
            });
        }
    }
    if let Some(g) = subsequent_grade {
        if !(1..=5).contains(&g) {
            failures.push(ValidationFailure {
                record_id: record_id.into(),
                category: ValidationCategory::Integrity,
                rule: format!("{scale_name}_grade_range"),
                message: format!("subsequent {scale_name} grade {g} is outside valid domain 1..=5"),
            });
        }
    }

    if let (Some(init), Some(sub)) = (initial_grade, subsequent_grade) {
        if (1..=5).contains(&init)
            && (1..=5).contains(&sub)
            && init > sub
            && !documented_intervention
        {
            failures.push(ValidationFailure {
                record_id: record_id.into(),
                category: ValidationCategory::Plausibility,
                rule: format!("{scale_name}_progression_monotonicity"),
                message: format!(
                    "initial {scale_name} grade ({init}) > subsequent grade ({sub}) indicates spontaneous improvement without documented intervention"
                ),
            });
        }
    }

    failures
}

/// Validate discharge vs 90-day modified Rankin Scale (mRS) monotonicity.
///
/// - Hard integrity check: observed mRS scores must be within `0..=6`.
/// - Hard integrity check: if `discharge_mrs == 6` (death at discharge), `followup_90d_mrs < 6`
///   is impossible even with intervention.
/// - Missing mRS values (`None`) are preserved and never inferred from death or other fields.
/// - Study-specific clinical coherence contract: `discharge_mrs <= followup_90d_mrs` (no
///   spontaneous recovery to a lower mRS score without documented post-discharge/rehab intervention).
pub fn validate_mrs_monotonicity(
    record_id: &str,
    discharge_mrs: Option<u8>,
    followup_90d_mrs: Option<u8>,
    documented_intervention: bool,
) -> Vec<ValidationFailure> {
    let mut failures = Vec::new();

    if let Some(m) = discharge_mrs {
        if m > 6 {
            failures.push(ValidationFailure {
                record_id: record_id.into(),
                category: ValidationCategory::Integrity,
                rule: "mrs_grade_range".into(),
                message: format!("discharge mRS {m} is outside valid domain 0..=6"),
            });
        }
    }
    if let Some(m) = followup_90d_mrs {
        if m > 6 {
            failures.push(ValidationFailure {
                record_id: record_id.into(),
                category: ValidationCategory::Integrity,
                rule: "mrs_grade_range".into(),
                message: format!("90-day follow-up mRS {m} is outside valid domain 0..=6"),
            });
        }
    }

    if let (Some(d_mrs), Some(f_mrs)) = (discharge_mrs, followup_90d_mrs) {
        if d_mrs <= 6 && f_mrs <= 6 {
            if d_mrs == 6 && f_mrs < 6 {
                failures.push(ValidationFailure {
                    record_id: record_id.into(),
                    category: ValidationCategory::Integrity,
                    rule: "mrs_death_irreversibility".into(),
                    message: format!("discharge mRS is 6 (death) but 90-day mRS is {f_mrs}"),
                });
            } else if d_mrs > f_mrs && !documented_intervention {
                failures.push(ValidationFailure {
                    record_id: record_id.into(),
                    category: ValidationCategory::Plausibility,
                    rule: "mrs_discharge_to_90d_monotonicity".into(),
                    message: format!(
                        "discharge mRS ({d_mrs}) > 90-day mRS ({f_mrs}) indicates spontaneous recovery without documented intervention"
                    ),
                });
            }
        }
    }

    failures
}

/// Complete neurosurgical patient record for Phase 3 temporal and clinical coherence validation.
/// Missing values are represented as `None` and are never inferred.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct NeurosurgicalCoherenceRecord {
    pub temporal: TemporalTransitionRecord,
    pub initial_wfns: Option<u8>,
    pub subsequent_wfns: Option<u8>,
    pub initial_hunt_hess: Option<u8>,
    pub subsequent_hunt_hess: Option<u8>,
    pub documented_acute_intervention: bool,
    pub discharge_mrs: Option<u8>,
    pub followup_90d_mrs: Option<u8>,
    pub documented_recovery_intervention: bool,
}

impl NeurosurgicalCoherenceRecord {
    /// Count missing clinical grade fields (`initial_wfns`, `subsequent_wfns`,
    /// `initial_hunt_hess`, `subsequent_hunt_hess`, `discharge_mrs`, `followup_90d_mrs`).
    pub fn missing_grade_count(&self) -> usize {
        [
            self.initial_wfns,
            self.subsequent_wfns,
            self.initial_hunt_hess,
            self.subsequent_hunt_hess,
            self.discharge_mrs,
            self.followup_90d_mrs,
        ]
        .iter()
        .filter(|g| g.is_none())
        .count()
    }
}

/// Validate a single `NeurosurgicalCoherenceRecord` against both hard integrity rules and
/// study-specific neurosurgical clinical coherence contracts.
pub fn validate_neurosurgical_record(
    record: &NeurosurgicalCoherenceRecord,
) -> Vec<ValidationFailure> {
    let mut failures = validate_temporal_transitions(&record.temporal);
    for failure in validate_surgical_window_72h(&record.temporal) {
        if !failures.contains(&failure) {
            failures.push(failure);
        }
    }
    failures.extend(validate_grade_progression(
        &record.temporal.record_id,
        NeuroGradeScale::Wfns,
        record.initial_wfns,
        record.subsequent_wfns,
        record.documented_acute_intervention,
    ));
    failures.extend(validate_grade_progression(
        &record.temporal.record_id,
        NeuroGradeScale::HuntHess,
        record.initial_hunt_hess,
        record.subsequent_hunt_hess,
        record.documented_acute_intervention,
    ));
    failures.extend(validate_mrs_monotonicity(
        &record.temporal.record_id,
        record.discharge_mrs,
        record.followup_90d_mrs,
        record.documented_recovery_intervention,
    ));
    failures
}

/// Validate a batch of `NeurosurgicalCoherenceRecord` items, recording every failure and
/// missingness count without silently dropping any record.
pub fn validate_neurosurgical_cohort(records: &[NeurosurgicalCoherenceRecord]) -> ValidationReport {
    let mut failures = Vec::new();
    let mut valid_records = 0usize;
    let mut rejected_records = 0usize;
    let mut missing_timestamp_count = 0usize;
    let mut missing_grade_count = 0usize;

    for record in records {
        missing_timestamp_count += record.temporal.missing_timestamp_count();
        missing_grade_count += record.missing_grade_count();
        let rec_failures = validate_neurosurgical_record(record);
        if rec_failures.is_empty() {
            valid_records += 1;
        } else {
            rejected_records += 1;
            failures.extend(rec_failures);
        }
    }

    let integrity_failure_count = failures
        .iter()
        .filter(|f| f.category == ValidationCategory::Integrity)
        .count();
    let plausibility_failure_count = failures
        .iter()
        .filter(|f| f.category == ValidationCategory::Plausibility)
        .count();

    ValidationReport {
        total_records: records.len(),
        valid_records,
        rejected_records,
        integrity_failure_count,
        plausibility_failure_count,
        missing_timestamp_count,
        missing_grade_count,
        failures,
    }
}

/// Single time-to-event observation for survival analysis.
/// `time` is in nonnegative units (e.g., hours or days from index/admission).
/// `event_occurred` is `true` if the event (e.g., death) occurred at `time`, or `false` if right-censored.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SurvivalObservation {
    pub record_id: String,
    pub time: Option<f64>,
    pub event_occurred: Option<bool>,
}

impl SurvivalObservation {
    /// Parse a survival observation from raw string tokens (`""` and `"NA"` -> `None`).
    /// Never infers missing time or event status.
    pub fn from_raw_strings(
        record_id: impl Into<String>,
        time_raw: &str,
        event_raw: &str,
    ) -> Result<Self, Vec<ValidationFailure>> {
        let record_id = record_id.into();
        let mut failures = Vec::new();

        let time = match parse_optional_f64(time_raw, "survival_time") {
            Ok(OptionalField::Observed(t)) => Some(t),
            Ok(OptionalField::Missing) => None,
            Err(msg) => {
                failures.push(ValidationFailure {
                    record_id: record_id.clone(),
                    category: ValidationCategory::Integrity,
                    rule: "survival_time_token".into(),
                    message: msg,
                });
                None
            }
        };

        let trimmed_event = event_raw.trim();
        let event_occurred = if trimmed_event.is_empty() || trimmed_event.eq_ignore_ascii_case("NA")
        {
            None
        } else {
            match trimmed_event.to_ascii_lowercase().as_str() {
                "1" | "true" | "event" | "dead" | "deceased" => Some(true),
                "0" | "false" | "censored" | "alive" => Some(false),
                _ => {
                    failures.push(ValidationFailure {
                        record_id: record_id.clone(),
                        category: ValidationCategory::Integrity,
                        rule: "survival_event_token".into(),
                        message: format!(
                            "invalid survival event token '{trimmed_event}' for record '{record_id}'"
                        ),
                    });
                    None
                }
            }
        };

        if failures.is_empty() {
            Ok(Self {
                record_id,
                time,
                event_occurred,
            })
        } else {
            Err(failures)
        }
    }
}

/// Step point on a Kaplan-Meier survival curve.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct KaplanMeierPoint {
    pub time: f64,
    pub at_risk: usize,
    pub events: usize,
    pub censored: usize,
    pub survival_prob: f64,
    pub greenwood_se: f64,
}

/// Estimated Kaplan-Meier survival distribution and summary statistics for a cohort.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct KaplanMeierCurve {
    pub total_records: usize,
    pub evaluable_records: usize,
    pub missing_records: usize,
    pub total_events: usize,
    pub total_censored: usize,
    pub restricted_mean_survival_time: f64,
    pub median_survival_time: Option<f64>,
    pub points: Vec<KaplanMeierPoint>,
    pub failures: Vec<ValidationFailure>,
}

/// Comparison result between real and synthetic Kaplan-Meier survival distributions.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SurvivalCurveComparison {
    pub real_curve: KaplanMeierCurve,
    pub synthetic_curve: KaplanMeierCurve,
    /// Two-sample Mantel-Haenszel log-rank chi-squared statistic (1 degree of freedom).
    pub log_rank_statistic: f64,
    /// Two-sample log-rank p-value.
    pub log_rank_pvalue: f64,
    /// Maximum absolute difference between the two step survival functions across all observed times.
    pub max_survival_difference: f64,
    /// Absolute difference in restricted mean survival time (RMST) evaluated up to the common horizon.
    pub rmst_difference: f64,
}

/// Compute the Kaplan-Meier survival curve for a cohort of `SurvivalObservation` records.
///
/// - Records with missing (`None`) time or event status are counted in `missing_records` and not
///   inferred or treated as zero-time events.
/// - Negative or non-finite times are recorded as `ValidationCategory::Integrity` failures.
pub fn compute_kaplan_meier(
    observations: &[SurvivalObservation],
) -> Result<KaplanMeierCurve, Error> {
    let mut failures = Vec::new();
    let mut missing_records = 0usize;
    let mut valid_obs: Vec<(f64, bool)> = Vec::new();

    for obs in observations {
        if obs.record_id.trim().is_empty() {
            failures.push(ValidationFailure {
                record_id: obs.record_id.clone(),
                category: ValidationCategory::Integrity,
                rule: "survival_record_id".into(),
                message: "survival observation has empty record_id".into(),
            });
            continue;
        }
        match (obs.time, obs.event_occurred) {
            (None, _) | (_, None) => {
                missing_records += 1;
            }
            (Some(t), Some(ev)) => {
                if !t.is_finite() || t < 0.0 {
                    failures.push(ValidationFailure {
                        record_id: obs.record_id.clone(),
                        category: ValidationCategory::Integrity,
                        rule: "survival_time_nonnegative".into(),
                        message: format!("survival time must be finite and nonnegative, got {t}"),
                    });
                } else {
                    valid_obs.push((t, ev));
                }
            }
        }
    }

    if valid_obs.is_empty() {
        return Err(Error::Input(
            "Kaplan-Meier estimation requires at least one valid non-missing observation".into(),
        ));
    }

    valid_obs.sort_by(|a, b| a.0.total_cmp(&b.0));

    let evaluable_records = valid_obs.len();
    let mut points = Vec::new();
    let mut at_risk = evaluable_records;
    let mut survival_prob = 1.0_f64;
    let mut greenwood_sum = 0.0_f64;
    let mut total_events = 0usize;
    let mut total_censored = 0usize;

    let mut idx = 0usize;
    while idx < valid_obs.len() {
        let current_time = valid_obs[idx].0;
        let mut d_i = 0usize;
        let mut c_i = 0usize;
        while idx < valid_obs.len() && valid_obs[idx].0 == current_time {
            if valid_obs[idx].1 {
                d_i += 1;
            } else {
                c_i += 1;
            }
            idx += 1;
        }

        total_events += d_i;
        total_censored += c_i;

        if d_i > 0 && at_risk > 0 {
            let n_f = at_risk as f64;
            let d_f = d_i as f64;
            survival_prob *= 1.0 - (d_f / n_f);
            if at_risk > d_i {
                greenwood_sum += d_f / (n_f * (n_f - d_f));
            }
        }
        let greenwood_se = if survival_prob > 0.0 {
            survival_prob * greenwood_sum.sqrt()
        } else {
            0.0
        };

        points.push(KaplanMeierPoint {
            time: current_time,
            at_risk,
            events: d_i,
            censored: c_i,
            survival_prob,
            greenwood_se,
        });

        at_risk = at_risk.saturating_sub(d_i + c_i);
    }

    let max_time = points.last().map(|p| p.time).unwrap_or(0.0);
    let restricted_mean_survival_time = integrate_km_step(&points, max_time);
    let median_survival_time = points
        .iter()
        .find(|p| p.survival_prob <= 0.5)
        .map(|p| p.time);

    Ok(KaplanMeierCurve {
        total_records: observations.len(),
        evaluable_records,
        missing_records,
        total_events,
        total_censored,
        restricted_mean_survival_time,
        median_survival_time,
        points,
        failures,
    })
}

fn evaluate_km_at(points: &[KaplanMeierPoint], t: f64) -> f64 {
    let mut surv = 1.0;
    for pt in points {
        if pt.time <= t {
            surv = pt.survival_prob;
        } else {
            break;
        }
    }
    surv
}

fn integrate_km_step(points: &[KaplanMeierPoint], horizon: f64) -> f64 {
    if horizon <= 0.0 {
        return 0.0;
    }
    let mut area = 0.0;
    let mut prev_t = 0.0;
    let mut prev_s = 1.0;
    for pt in points {
        if pt.time >= horizon {
            area += (horizon - prev_t) * prev_s;
            prev_t = horizon;
            break;
        }
        if pt.time > prev_t {
            area += (pt.time - prev_t) * prev_s;
            prev_t = pt.time;
        }
        prev_s = pt.survival_prob;
    }
    if prev_t < horizon {
        area += (horizon - prev_t) * prev_s;
    }
    area
}

/// Compare survival distributions between real and synthetic cohorts using Kaplan-Meier
/// estimation, the Mantel-Haenszel log-rank test, supremum survival gap, and RMST difference.
pub fn compare_survival_curves(
    real_observations: &[SurvivalObservation],
    synthetic_observations: &[SurvivalObservation],
) -> Result<SurvivalCurveComparison, Error> {
    let real_curve = compute_kaplan_meier(real_observations)?;
    let synthetic_curve = compute_kaplan_meier(synthetic_observations)?;

    if !real_curve.failures.is_empty() || !synthetic_curve.failures.is_empty() {
        return Err(Error::Input(
            "cannot compare survival curves when integrity failures are present in observations"
                .into(),
        ));
    }

    // Collect all unique observed times across both cohorts for step comparison and log-rank test.
    let mut all_times: BTreeSet<u64> = BTreeSet::new();
    for pt in &real_curve.points {
        all_times.insert(pt.time.to_bits());
    }
    for pt in &synthetic_curve.points {
        all_times.insert(pt.time.to_bits());
    }

    let mut ordered_times: Vec<f64> = all_times.into_iter().map(f64::from_bits).collect();
    ordered_times.sort_by(|a, b| a.total_cmp(b));

    let mut o1_minus_e1 = 0.0_f64;
    let mut variance_sum = 0.0_f64;
    let mut max_survival_difference = 0.0_f64;

    // Map time bits to (at_risk, events, censored) for each curve.
    let mut n1 = real_curve.evaluable_records;
    let mut n2 = synthetic_curve.evaluable_records;
    let mut i1 = 0usize;
    let mut i2 = 0usize;

    for &t in &ordered_times {
        let mut d1 = 0usize;
        let mut c1 = 0usize;
        if i1 < real_curve.points.len() && real_curve.points[i1].time == t {
            n1 = real_curve.points[i1].at_risk;
            d1 = real_curve.points[i1].events;
            c1 = real_curve.points[i1].censored;
            i1 += 1;
        }

        let mut d2 = 0usize;
        let mut c2 = 0usize;
        if i2 < synthetic_curve.points.len() && synthetic_curve.points[i2].time == t {
            n2 = synthetic_curve.points[i2].at_risk;
            d2 = synthetic_curve.points[i2].events;
            c2 = synthetic_curve.points[i2].censored;
            i2 += 1;
        }

        let n = n1 + n2;
        let d = d1 + d2;
        if n > 1 && d > 0 {
            let n_f = n as f64;
            let n1_f = n1 as f64;
            let n2_f = n2 as f64;
            let d_f = d as f64;
            let e1 = n1_f * d_f / n_f;
            let v = (n1_f * n2_f * d_f * (n_f - d_f)) / (n_f * n_f * (n_f - 1.0));
            o1_minus_e1 += (d1 as f64) - e1;
            variance_sum += v;
        }

        let s1 = evaluate_km_at(&real_curve.points, t);
        let s2 = evaluate_km_at(&synthetic_curve.points, t);
        max_survival_difference = max_survival_difference.max((s1 - s2).abs());

        n1 = n1.saturating_sub(d1 + c1);
        n2 = n2.saturating_sub(d2 + c2);
    }

    let (log_rank_statistic, log_rank_pvalue) = if variance_sum > 0.0 {
        let stat = (o1_minus_e1 * o1_minus_e1) / variance_sum;
        let pval = if stat == 0.0 {
            1.0
        } else {
            let chi2 = ChiSquared::new(1.0)
                .map_err(|e| Error::Input(format!("chi-squared error: {e}")))?;
            (1.0 - chi2.cdf(stat)).clamp(0.0, 1.0)
        };
        (stat, pval)
    } else {
        (0.0, 1.0)
    };

    let common_horizon = real_curve
        .points
        .last()
        .map(|p| p.time)
        .unwrap_or(0.0)
        .min(synthetic_curve.points.last().map(|p| p.time).unwrap_or(0.0));
    let rmst_real = integrate_km_step(&real_curve.points, common_horizon);
    let rmst_synth = integrate_km_step(&synthetic_curve.points, common_horizon);
    let rmst_difference = (rmst_real - rmst_synth).abs();

    Ok(SurvivalCurveComparison {
        real_curve,
        synthetic_curve,
        log_rank_statistic,
        log_rank_pvalue,
        max_survival_difference,
        rmst_difference,
    })
}

/// Compute a canonical `sha256:<hex>` fingerprint over arbitrary bytes.
pub fn sha256_fingerprint(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    format!("sha256:{digest:x}")
}

/// Compute the SHA-256 fingerprint (`sha256:<hex>`) of the workspace `Cargo.lock`.
pub fn cargo_lock_sha256() -> String {
    sha256_fingerprint(include_bytes!("../../../Cargo.lock"))
}

/// Extract resolved versions of key dependencies from the workspace `Cargo.lock`.
pub fn cargo_lock_dependency_versions() -> BTreeMap<String, String> {
    const KEY_DEPS: &[&str] = &[
        "arrow-array",
        "clap",
        "csv",
        "duckdb",
        "linfa",
        "linfa-logistic",
        "ndarray",
        "parquet",
        "rand",
        "serde",
        "serde_json",
        "serde_yaml",
        "sha2",
        "statrs",
        "thiserror",
    ];
    let key_set: HashSet<&str> = KEY_DEPS.iter().copied().collect();
    let lock_text = include_str!("../../../Cargo.lock");
    let mut versions = BTreeMap::new();
    let mut current_name: Option<String> = None;
    for line in lock_text.lines() {
        let trimmed = line.trim();
        if trimmed == "[[package]]" {
            current_name = None;
        } else if let Some(rest) = trimmed.strip_prefix("name = ") {
            let name = rest.trim_matches('"');
            if key_set.contains(name) {
                current_name = Some(name.to_string());
            } else {
                current_name = None;
            }
        } else if let Some(rest) = trimmed.strip_prefix("version = ") {
            if let Some(name) = current_name.take() {
                let version = rest.trim_matches('"').to_string();
                versions.entry(name).or_insert(version);
            }
        }
    }
    versions
}

/// Single check entry in a [`ReproducibilityAuditReport`].
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ReproducibilityCheck {
    pub name: String,
    pub passed: bool,
    pub status: String,
    pub expected: String,
    pub actual: String,
    pub message: String,
}

/// Output report produced by `janus-cli audit-reproducibility`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ReproducibilityAuditReport {
    pub reproducible: bool,
    pub checks: Vec<ReproducibilityCheck>,
    pub warnings: Vec<String>,
}

fn normalize_hash_token(raw: &str) -> String {
    let trimmed = raw.trim();
    let stripped = if trimmed
        .get(..7)
        .is_some_and(|prefix| prefix.eq_ignore_ascii_case("sha256:"))
    {
        &trimmed[7..]
    } else {
        trimmed
    };
    stripped.to_ascii_lowercase()
}

fn lookup_str<'a>(value: &'a serde_json::Value, paths: &[&[&str]]) -> Option<&'a str> {
    for path in paths {
        let mut cur = value;
        let mut found = true;
        for &segment in *path {
            if let Some(next) = cur.get(segment) {
                cur = next;
            } else {
                found = false;
                break;
            }
        }
        if found {
            if let Some(s) = cur.as_str() {
                let trimmed = s.trim();
                if !trimmed.is_empty() {
                    return Some(trimmed);
                }
            }
        }
    }
    None
}

fn lookup_object<'a>(
    value: &'a serde_json::Value,
    paths: &[&[&str]],
) -> Option<&'a serde_json::Map<String, serde_json::Value>> {
    for path in paths {
        let mut cur = value;
        let mut found = true;
        for &segment in *path {
            if let Some(next) = cur.get(segment) {
                cur = next;
            } else {
                found = false;
                break;
            }
        }
        if found {
            if let Some(map) = cur.as_object() {
                return Some(map);
            }
        }
    }
    None
}

/// Audit a run manifest JSON value (and optional split manifest) against the current replay environment.
///
/// Validation checks:
/// 1. `lockfile_fingerprint`: matches current `Cargo.lock` (`sha256:...` or `fnv1a64-yaml:...`).
/// 2. `platform_metadata`: compares OS and CPU architecture against manifest (warns if different in non-strict mode).
/// 3. `split_fingerprint`: if `splits_value` is provided, verifies holdout / patient-allocation fingerprint matches manifest.
/// 4. `software_versions`: reports current Rust toolchain, package version, and key dependency versions vs manifest.
pub fn audit_reproducibility(
    manifest_value: &serde_json::Value,
    splits_value: Option<&serde_json::Value>,
    strict: bool,
) -> Result<ReproducibilityAuditReport, Error> {
    if !manifest_value.is_object() {
        return Err(Error::Input("run manifest must be a JSON object".into()));
    }

    let mut checks = Vec::new();
    let mut warnings = Vec::new();
    let mut hard_failure = false;

    // 1. Lockfile fingerprint check (SHA-256 or FNV-1a YAML fingerprint of Cargo.lock)
    let expected_lock_opt = lookup_str(
        manifest_value,
        &[
            &["lockfile_hash"],
            &["lockfile_fingerprint"],
            &["dependency_lock_fingerprint"],
            &["fingerprints", "lockfile_hash"],
            &["fingerprints", "lockfile_fingerprint"],
            &["fingerprints", "dependency_lock_fingerprint"],
            &["software", "lockfile_hash"],
            &["software", "lockfile_fingerprint"],
            &["software", "dependency_lock_fingerprint"],
        ],
    );
    let (actual_lock, lock_passed, expected_lock_str, lock_msg) = match expected_lock_opt {
        Some(exp) if exp.to_ascii_lowercase().starts_with("fnv1a64-yaml:") => {
            let actual_yaml_lock = crate::yaml_fingerprint(&include_str!("../../../Cargo.lock"))?;
            let matched = normalize_hash_token(exp) == normalize_hash_token(&actual_yaml_lock);
            let msg = if matched {
                "Cargo.lock fnv1a64-yaml fingerprint matches manifest".to_string()
            } else {
                format!(
                    "lockfile_fingerprint mismatch: manifest recorded '{exp}', current Cargo.lock is '{actual_yaml_lock}'"
                )
            };
            (actual_yaml_lock, matched, exp.to_string(), msg)
        }
        Some(exp) => {
            let actual_sha_lock = cargo_lock_sha256();
            let matched = normalize_hash_token(exp) == normalize_hash_token(&actual_sha_lock);
            let msg = if matched {
                "Cargo.lock SHA-256 fingerprint matches manifest".to_string()
            } else {
                format!(
                    "lockfile_fingerprint mismatch: manifest recorded '{exp}', current Cargo.lock is '{actual_sha_lock}'"
                )
            };
            (actual_sha_lock, matched, exp.to_string(), msg)
        }
        None => {
            let actual_sha_lock = cargo_lock_sha256();
            let msg = format!(
                "lockfile_fingerprint missing in manifest (current Cargo.lock is '{actual_sha_lock}')"
            );
            (actual_sha_lock, false, "missing".to_string(), msg)
        }
    };
    if !lock_passed {
        hard_failure = true;
        warnings.push(lock_msg.clone());
    }
    checks.push(ReproducibilityCheck {
        name: "lockfile_fingerprint".into(),
        passed: lock_passed,
        status: if lock_passed {
            "pass".into()
        } else if strict {
            "fail".into()
        } else {
            "warn".into()
        },
        expected: expected_lock_str,
        actual: actual_lock,
        message: lock_msg,
    });

    // 2. Platform metadata check (OS, architecture)
    let actual_os = std::env::consts::OS;
    let actual_arch = std::env::consts::ARCH;
    let actual_platform = format!("{actual_os}/{actual_arch}");
    let expected_os = lookup_str(
        manifest_value,
        &[
            &["platform", "os"],
            &["platform", "operating_system"],
            &["environment", "operating_system"],
            &["environment", "os"],
            &["software", "operating_system"],
            &["software", "os"],
            &["os"],
            &["operating_system"],
        ],
    );
    let expected_arch = lookup_str(
        manifest_value,
        &[
            &["platform", "arch"],
            &["platform", "architecture"],
            &["environment", "architecture"],
            &["environment", "arch"],
            &["software", "architecture"],
            &["software", "arch"],
            &["arch"],
            &["architecture"],
        ],
    );
    let expected_platform = format!(
        "{}/{}",
        expected_os.unwrap_or("unspecified"),
        expected_arch.unwrap_or("unspecified")
    );
    let platform_passed = expected_os == Some(actual_os) && expected_arch == Some(actual_arch);
    let platform_msg = if platform_passed {
        format!("platform metadata matches current environment ({actual_platform})")
    } else {
        format!(
            "platform_metadata mismatch: manifest recorded '{expected_platform}', current environment is '{actual_platform}'"
        )
    };
    if !platform_passed {
        warnings.push(platform_msg.clone());
    }
    checks.push(ReproducibilityCheck {
        name: "platform_metadata".into(),
        passed: platform_passed,
        status: if platform_passed {
            "pass".into()
        } else if strict {
            "fail".into()
        } else {
            "warn".into()
        },
        expected: expected_platform,
        actual: actual_platform,
        message: platform_msg,
    });

    // 3. Split fingerprint check (if --splits provided)
    if let Some(splits) = splits_value {
        if !splits.is_object() {
            return Err(Error::Input(
                "split manifest must be a JSON/YAML object".into(),
            ));
        }
        let expected_split_opt = lookup_str(
            manifest_value,
            &[
                &["holdout_fingerprint"],
                &["holdout_fixture_hash"],
                &["split_fingerprint"],
                &["patient_allocation_fingerprint"],
                &["allocation_fingerprint"],
                &["reference_fingerprint"],
                &["fingerprints", "holdout_fingerprint"],
                &["fingerprints", "holdout_fixture_hash"],
                &["fingerprints", "split_fingerprint"],
                &["fingerprints", "patient_allocation_fingerprint"],
                &["fingerprints", "allocation_fingerprint"],
                &["fingerprints", "reference_fingerprint"],
            ],
        );
        let actual_split_opt = lookup_str(
            splits,
            &[
                &["holdout_fingerprint"],
                &["holdout_fixture_hash"],
                &["patient_allocation_fingerprint"],
                &["allocation_fingerprint"],
                &["split_fingerprint"],
                &["reference_fingerprint"],
                &["fingerprints", "holdout_fingerprint"],
                &["fingerprints", "holdout_fixture_hash"],
                &["fingerprints", "patient_allocation_fingerprint"],
                &["fingerprints", "allocation_fingerprint"],
                &["fingerprints", "split_fingerprint"],
            ],
        );
        let (split_passed, exp_str, act_str, split_msg) = match (expected_split_opt, actual_split_opt)
        {
            (Some(exp), Some(act))
                if normalize_hash_token(exp) == normalize_hash_token(act) =>
            {
                (
                    true,
                    exp.to_string(),
                    act.to_string(),
                    "split holdout/allocation fingerprint matches run manifest".to_string(),
                )
            }
            (Some(exp), Some(act)) => (
                false,
                exp.to_string(),
                act.to_string(),
                format!(
                    "split_fingerprint mismatch: run manifest expected '{exp}', split manifest recorded '{act}'"
                ),
            ),
            (None, Some(act)) => (
                false,
                "missing".to_string(),
                act.to_string(),
                format!(
                    "split_fingerprint missing in run manifest (split manifest recorded '{act}')"
                ),
            ),
            (Some(exp), None) => (
                false,
                exp.to_string(),
                "missing".to_string(),
                format!(
                    "split_fingerprint missing in split manifest (run manifest expected '{exp}')"
                ),
            ),
            (None, None) => (
                false,
                "missing".to_string(),
                "missing".to_string(),
                "split_fingerprint missing in both run manifest and split manifest".to_string(),
            ),
        };
        if !split_passed {
            hard_failure = true;
            warnings.push(split_msg.clone());
        }
        checks.push(ReproducibilityCheck {
            name: "split_fingerprint".into(),
            passed: split_passed,
            status: if split_passed {
                "pass".into()
            } else if strict {
                "fail".into()
            } else {
                "warn".into()
            },
            expected: exp_str,
            actual: act_str,
            message: split_msg,
        });
    }

    // 4. Software versions check (Rust toolchain, package version, key dependency versions)
    let actual_pkg_version = env!("CARGO_PKG_VERSION");
    let actual_rustc = option_env!("JANUS_RUSTC").unwrap_or("unavailable");
    let lock_deps = cargo_lock_dependency_versions();
    let key_deps_summary = [
        "serde",
        "serde_json",
        "serde_yaml",
        "rand",
        "statrs",
        "sha2",
    ]
    .iter()
    .filter_map(|&k| lock_deps.get(k).map(|v| format!("{k}={v}")))
    .collect::<Vec<_>>()
    .join(", ");
    let actual_software_summary =
        format!("janus={actual_pkg_version}, rustc={actual_rustc}, deps=[{key_deps_summary}]");

    let expected_pkg_version = lookup_str(
        manifest_value,
        &[
            &["software_version"],
            &["package_version"],
            &["software", "package_version"],
            &["software", "software_version"],
            &["software", "version"],
            &["software_versions", "janus"],
            &["software_versions", "package_version"],
            &["versions", "janus"],
        ],
    );
    let expected_rustc = lookup_str(
        manifest_value,
        &[
            &["rustc"],
            &["rust_toolchain"],
            &["toolchain"],
            &["software", "rustc"],
            &["software", "rust_toolchain"],
            &["software", "toolchain"],
            &["environment", "rustc"],
            &["software_versions", "rustc"],
            &["software_versions", "rust_toolchain"],
            &["versions", "rustc"],
        ],
    );
    let expected_deps_map = lookup_object(
        manifest_value,
        &[
            &["dependencies"],
            &["software", "dependencies"],
            &["software_versions", "dependencies"],
            &["versions", "dependencies"],
        ],
    );

    let mut sw_mismatches = Vec::new();
    let mut expected_sw_parts = Vec::new();

    if let Some(exp_pkg) = expected_pkg_version {
        expected_sw_parts.push(format!("janus={exp_pkg}"));
        if exp_pkg != actual_pkg_version {
            sw_mismatches.push(format!(
                "package version mismatch (expected '{exp_pkg}', actual '{actual_pkg_version}')"
            ));
        }
    }
    if let Some(exp_rustc) = expected_rustc {
        expected_sw_parts.push(format!("rustc={exp_rustc}"));
        if exp_rustc != "unavailable"
            && actual_rustc != "unavailable"
            && exp_rustc != actual_rustc
            && !actual_rustc.contains(exp_rustc)
            && !exp_rustc.contains(actual_rustc)
        {
            sw_mismatches.push(format!(
                "Rust toolchain mismatch (expected '{exp_rustc}', actual '{actual_rustc}')"
            ));
        }
    }
    if let Some(deps) = expected_deps_map {
        let mut dep_parts = Vec::new();
        for (dep_name, dep_val) in deps {
            if let Some(exp_ver) = dep_val.as_str() {
                dep_parts.push(format!("{dep_name}={exp_ver}"));
                match lock_deps.get(dep_name) {
                    Some(act_ver)
                        if act_ver == exp_ver || act_ver.starts_with(&format!("{exp_ver}.")) => {}
                    Some(act_ver) => {
                        sw_mismatches.push(format!(
                            "dependency '{dep_name}' version mismatch (expected '{exp_ver}', actual '{act_ver}')"
                        ));
                    }
                    None => {
                        sw_mismatches.push(format!(
                            "dependency '{dep_name}' not found in Cargo.lock (expected '{exp_ver}')"
                        ));
                    }
                }
            }
        }
        if !dep_parts.is_empty() {
            expected_sw_parts.push(format!("deps=[{}]", dep_parts.join(", ")));
        }
    }

    let expected_sw_summary = if expected_sw_parts.is_empty() {
        "unspecified".to_string()
    } else {
        expected_sw_parts.join(", ")
    };
    let sw_passed = sw_mismatches.is_empty();
    let sw_msg = if sw_passed {
        format!("software versions verified ({actual_software_summary})")
    } else {
        format!("software_versions mismatch: {}", sw_mismatches.join("; "))
    };
    if !sw_passed {
        warnings.push(sw_msg.clone());
    }
    checks.push(ReproducibilityCheck {
        name: "software_versions".into(),
        passed: sw_passed,
        status: if sw_passed {
            "pass".into()
        } else if strict {
            "fail".into()
        } else {
            "warn".into()
        },
        expected: expected_sw_summary,
        actual: actual_software_summary,
        message: sw_msg,
    });

    let reproducible = !hard_failure && (!strict || warnings.is_empty());

    Ok(ReproducibilityAuditReport {
        reproducible,
        checks,
        warnings,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fictional_contract_and_manifest_validate() {
        let contract: CoherenceContract =
            serde_yaml::from_str(include_str!("../../../examples/coherence-contract.yaml"))
                .unwrap();
        contract.validate().unwrap();
        let manifest: ProtocolManifest =
            serde_yaml::from_str(include_str!("../../../examples/protocol-manifest.yaml")).unwrap();
        manifest.validate().unwrap();
        assert_eq!(manifest.contract_id, contract.contract_id);
        let comparison: CandidateComparisonManifest =
            serde_yaml::from_str(include_str!("../../../examples/candidate-comparison.yaml"))
                .unwrap();
        comparison.validate().unwrap();
    }

    #[test]
    fn contract_rejects_duplicate_variables_and_missing_limits() {
        let mut contract: CoherenceContract =
            serde_yaml::from_str(include_str!("../../../examples/coherence-contract.yaml"))
                .unwrap();
        contract.variables.push(contract.variables[0].clone());
        assert!(contract.validate().is_err());
        contract.variables.pop();
        contract.limitations.clear();
        assert!(contract.validate().is_err());
    }

    #[test]
    fn candidate_comparison_requires_identical_preprocessing() {
        let manifest = CandidateComparisonManifest {
            schema_version: "1".into(),
            protocol_id: "fictional".into(),
            contract_id: "contract".into(),
            candidates: vec![
                ProtocolCandidate {
                    name: "forest".into(),
                    preprocessing: "same".into(),
                    status: "baseline".into(),
                },
                ProtocolCandidate {
                    name: "neural".into(),
                    preprocessing: "same".into(),
                    status: "exploratory".into(),
                },
            ],
            mapping_fingerprint: "map".into(),
            populations: vec!["development".into()],
            sample_count: 10,
            seeds: vec![1, 2],
            metrics: vec!["marginal_tvd".into(), "joint_tvd".into()],
        };
        manifest.validate().unwrap();
        let mut changed = manifest.clone();
        changed.candidates[1].preprocessing = "different".into();
        assert!(changed.validate().is_err());
    }

    #[test]
    fn temporal_transitions_valid_and_invalid_sequences() {
        // Valid complete chronological sequence: admission (0h) -> surgery (18h) -> discharge (240h) -> 90d follow-up (2160h)
        let valid = TemporalTransitionRecord {
            record_id: "fictional-p1".into(),
            admission_time_hours: Some(0.0),
            surgery_time_hours: Some(18.0),
            discharge_time_hours: Some(240.0),
            followup_90d_time_hours: Some(2160.0),
        };
        assert!(validate_temporal_transitions(&valid).is_empty());

        // Invalid: surgery before admission
        let surgery_before_admission = TemporalTransitionRecord {
            record_id: "fictional-p2".into(),
            admission_time_hours: Some(10.0),
            surgery_time_hours: Some(5.0),
            discharge_time_hours: Some(240.0),
            followup_90d_time_hours: Some(2160.0),
        };
        let failures = validate_temporal_transitions(&surgery_before_admission);
        assert!(!failures.is_empty());
        assert!(failures
            .iter()
            .all(|f| f.category == ValidationCategory::Integrity));
        assert!(failures
            .iter()
            .any(|f| f.rule == "temporal_transition_order"));

        // Invalid: discharge before surgery and 90d follow-up before discharge
        let out_of_order = TemporalTransitionRecord {
            record_id: "fictional-p3".into(),
            admission_time_hours: Some(0.0),
            surgery_time_hours: Some(48.0),
            discharge_time_hours: Some(24.0),
            followup_90d_time_hours: Some(12.0),
        };
        let failures = validate_temporal_transitions(&out_of_order);
        assert!(failures.len() >= 2);
        assert!(failures
            .iter()
            .all(|f| f.category == ValidationCategory::Integrity));
    }

    #[test]
    fn temporal_transitions_preserve_missing_timestamps_without_inference() {
        // Raw strings with blank "" and "NA" must parse as None without inferring timestamps
        let parsed = TemporalTransitionRecord::from_raw_strings(
            "fictional-missing-1",
            "0.0",
            "NA",
            "200.0",
            "",
        )
        .unwrap();
        assert_eq!(parsed.admission_time_hours, Some(0.0));
        assert_eq!(parsed.surgery_time_hours, None);
        assert_eq!(parsed.discharge_time_hours, Some(200.0));
        assert_eq!(parsed.followup_90d_time_hours, None);
        assert_eq!(parsed.missing_timestamp_count(), 2);
        assert!(validate_temporal_transitions(&parsed).is_empty());
        assert!(validate_surgical_window_72h(&parsed).is_empty());

        // Non-adjacent observed timestamps (admission and 90-day follow-up present, surgery and discharge missing)
        // still enforce chronological ordering when violated:
        let non_adjacent_violation = TemporalTransitionRecord::from_raw_strings(
            "fictional-missing-2",
            "500.0",
            "",
            "NA",
            "100.0",
        )
        .unwrap();
        let failures = validate_temporal_transitions(&non_adjacent_violation);
        assert_eq!(failures.len(), 1);
        assert_eq!(failures[0].category, ValidationCategory::Integrity);
        assert_eq!(failures[0].rule, "temporal_transition_order");

        // Invalid token in raw string is recorded as an Integrity failure rather than silently treated as missing
        let bad_token = TemporalTransitionRecord::from_raw_strings(
            "fictional-bad-token",
            "0.0",
            "unknown_time",
            "120.0",
            "2160.0",
        );
        assert!(bad_token.is_err());
        let token_failures = bad_token.unwrap_err();
        assert_eq!(token_failures.len(), 1);
        assert_eq!(token_failures[0].category, ValidationCategory::Integrity);
        assert_eq!(token_failures[0].rule, "timestamp_token_integrity");
    }

    #[test]
    fn surgical_window_72h_boundary_and_missing_cases() {
        // Valid: 71.999 hours (< 72h)
        let under_72h = TemporalTransitionRecord {
            record_id: "fictional-surg-1".into(),
            admission_time_hours: Some(10.0),
            surgery_time_hours: Some(81.999),
            discharge_time_hours: Some(240.0),
            followup_90d_time_hours: Some(2160.0),
        };
        assert!(validate_surgical_window_72h(&under_72h).is_empty());

        // Boundary condition: exactly 72.0 hours must be rejected (contract requires < 72h)
        let exactly_72h = TemporalTransitionRecord {
            record_id: "fictional-surg-boundary".into(),
            admission_time_hours: Some(10.0),
            surgery_time_hours: Some(82.0),
            discharge_time_hours: Some(240.0),
            followup_90d_time_hours: Some(2160.0),
        };
        let failures_72h = validate_surgical_window_72h(&exactly_72h);
        assert_eq!(failures_72h.len(), 1);
        assert_eq!(failures_72h[0].category, ValidationCategory::Plausibility);
        assert_eq!(failures_72h[0].rule, "surgical_window_under_72h");

        // Over 72h: 96.0 hours
        let over_72h = TemporalTransitionRecord {
            record_id: "fictional-surg-late".into(),
            admission_time_hours: Some(0.0),
            surgery_time_hours: Some(96.0),
            discharge_time_hours: Some(240.0),
            followup_90d_time_hours: Some(2160.0),
        };
        let failures_late = validate_surgical_window_72h(&over_72h);
        assert_eq!(failures_late.len(), 1);
        assert_eq!(failures_late[0].category, ValidationCategory::Plausibility);

        // Surgery timestamp present but admission timestamp missing -> Integrity failure (do not infer admission)
        let missing_admission = TemporalTransitionRecord {
            record_id: "fictional-surg-no-adm".into(),
            admission_time_hours: None,
            surgery_time_hours: Some(12.0),
            discharge_time_hours: None,
            followup_90d_time_hours: None,
        };
        let failures_no_adm = validate_surgical_window_72h(&missing_admission);
        assert_eq!(failures_no_adm.len(), 1);
        assert_eq!(failures_no_adm[0].category, ValidationCategory::Integrity);
        assert_eq!(
            failures_no_adm[0].rule,
            "surgical_window_requires_admission_timestamp"
        );
    }

    #[test]
    fn wfns_and_hunt_hess_progression_contracts() {
        // Stable or worsening grade without intervention is valid (initial <= subsequent)
        assert!(validate_grade_progression(
            "fictional-g1",
            NeuroGradeScale::Wfns,
            Some(2),
            Some(2),
            false
        )
        .is_empty());
        assert!(validate_grade_progression(
            "fictional-g2",
            NeuroGradeScale::HuntHess,
            Some(2),
            Some(4),
            false
        )
        .is_empty());

        // Spontaneous improvement (initial > subsequent) without documented intervention -> Plausibility failure
        let wfns_spontaneous = validate_grade_progression(
            "fictional-g3",
            NeuroGradeScale::Wfns,
            Some(4),
            Some(2),
            false,
        );
        assert_eq!(wfns_spontaneous.len(), 1);
        assert_eq!(
            wfns_spontaneous[0].category,
            ValidationCategory::Plausibility
        );
        assert_eq!(wfns_spontaneous[0].rule, "wfns_progression_monotonicity");

        let hh_spontaneous = validate_grade_progression(
            "fictional-g4",
            NeuroGradeScale::HuntHess,
            Some(5),
            Some(3),
            false,
        );
        assert_eq!(hh_spontaneous.len(), 1);
        assert_eq!(hh_spontaneous[0].category, ValidationCategory::Plausibility);
        assert_eq!(hh_spontaneous[0].rule, "hunt_hess_progression_monotonicity");

        // Improvement WITH documented intervention is valid
        assert!(validate_grade_progression(
            "fictional-g5",
            NeuroGradeScale::Wfns,
            Some(4),
            Some(2),
            true
        )
        .is_empty());

        // Missing grade is preserved without inference
        assert!(validate_grade_progression(
            "fictional-g6",
            NeuroGradeScale::Wfns,
            Some(4),
            None,
            false
        )
        .is_empty());

        // Out-of-bounds grade (0 or 6) is an Integrity failure
        let invalid_grade = validate_grade_progression(
            "fictional-g7",
            NeuroGradeScale::Wfns,
            Some(0),
            Some(6),
            true,
        );
        assert_eq!(invalid_grade.len(), 2);
        assert!(invalid_grade
            .iter()
            .all(|f| f.category == ValidationCategory::Integrity));
    }

    #[test]
    fn discharge_vs_90d_mrs_monotonicity_contracts() {
        // Stable or worsening mRS (discharge <= 90d) is valid
        assert!(validate_mrs_monotonicity("fictional-m1", Some(3), Some(3), false).is_empty());
        assert!(validate_mrs_monotonicity("fictional-m2", Some(3), Some(5), false).is_empty());

        // Spontaneous recovery (discharge > 90d) without documented intervention -> Plausibility failure
        let spontaneous_recovery =
            validate_mrs_monotonicity("fictional-m3", Some(4), Some(2), false);
        assert_eq!(spontaneous_recovery.len(), 1);
        assert_eq!(
            spontaneous_recovery[0].category,
            ValidationCategory::Plausibility
        );
        assert_eq!(
            spontaneous_recovery[0].rule,
            "mrs_discharge_to_90d_monotonicity"
        );

        // Recovery WITH documented intervention is allowed for non-deceased patients
        assert!(validate_mrs_monotonicity("fictional-m4", Some(4), Some(2), true).is_empty());

        // Death at discharge (mRS 6) followed by mRS < 6 at 90d is an Integrity failure even with intervention
        let impossible_resurrection =
            validate_mrs_monotonicity("fictional-m5", Some(6), Some(5), true);
        assert_eq!(impossible_resurrection.len(), 1);
        assert_eq!(
            impossible_resurrection[0].category,
            ValidationCategory::Integrity
        );
        assert_eq!(impossible_resurrection[0].rule, "mrs_death_irreversibility");

        // Missing 90-day mRS is preserved and never inferred from discharge mRS
        assert!(validate_mrs_monotonicity("fictional-m6", Some(6), None, false).is_empty());

        // Out-of-range mRS (> 6) is an Integrity failure
        let out_of_range = validate_mrs_monotonicity("fictional-m7", Some(7), Some(2), false);
        assert_eq!(out_of_range.len(), 1);
        assert_eq!(out_of_range[0].category, ValidationCategory::Integrity);
    }

    #[test]
    fn cohort_validation_records_all_failures_without_discarding() {
        let records = vec![
            NeurosurgicalCoherenceRecord {
                temporal: TemporalTransitionRecord {
                    record_id: "fictional-cohort-valid".into(),
                    admission_time_hours: Some(0.0),
                    surgery_time_hours: Some(24.0),
                    discharge_time_hours: Some(240.0),
                    followup_90d_time_hours: Some(2160.0),
                },
                initial_wfns: Some(3),
                subsequent_wfns: Some(3),
                initial_hunt_hess: Some(3),
                subsequent_hunt_hess: None,
                documented_acute_intervention: false,
                discharge_mrs: Some(3),
                followup_90d_mrs: Some(3),
                documented_recovery_intervention: false,
            },
            NeurosurgicalCoherenceRecord {
                temporal: TemporalTransitionRecord {
                    record_id: "fictional-cohort-invalid".into(),
                    admission_time_hours: Some(0.0),
                    surgery_time_hours: Some(72.0), // boundary >= 72h plausibility failure
                    discharge_time_hours: Some(48.0), // discharge < surgery integrity failure
                    followup_90d_time_hours: None,
                },
                initial_wfns: Some(4),
                subsequent_wfns: Some(1), // spontaneous improvement plausibility failure
                initial_hunt_hess: None,
                subsequent_hunt_hess: None,
                documented_acute_intervention: false,
                discharge_mrs: Some(4),
                followup_90d_mrs: Some(2), // spontaneous mRS recovery plausibility failure
                documented_recovery_intervention: false,
            },
        ];

        let report = validate_neurosurgical_cohort(&records);
        assert_eq!(report.total_records, 2);
        assert_eq!(report.valid_records, 1);
        assert_eq!(report.rejected_records, 1);
        assert_eq!(report.integrity_failure_count, 1);
        assert_eq!(report.plausibility_failure_count, 3);
        assert_eq!(report.missing_timestamp_count, 1);
        assert_eq!(report.missing_grade_count, 3);
        assert_eq!(report.failures.len(), 4);
    }

    #[test]
    fn kaplan_meier_survival_curve_comparison_valid_and_divergent() {
        let real_cohort = vec![
            SurvivalObservation::from_raw_strings("r1", "10.0", "1").unwrap(),
            SurvivalObservation::from_raw_strings("r2", "20.0", "1").unwrap(),
            SurvivalObservation::from_raw_strings("r3", "30.0", "0").unwrap(),
            SurvivalObservation::from_raw_strings("r4", "40.0", "1").unwrap(),
            SurvivalObservation::from_raw_strings("r5", "50.0", "0").unwrap(),
            SurvivalObservation::from_raw_strings("r6", "NA", "0").unwrap(), // missing time preserved
        ];

        // Identical evaluable distribution
        let synth_identical = real_cohort.clone();
        let cmp_same = compare_survival_curves(&real_cohort, &synth_identical).unwrap();
        assert_eq!(cmp_same.real_curve.total_records, 6);
        assert_eq!(cmp_same.real_curve.evaluable_records, 5);
        assert_eq!(cmp_same.real_curve.missing_records, 1);
        assert_eq!(cmp_same.real_curve.total_events, 3);
        assert_eq!(cmp_same.real_curve.total_censored, 2);
        assert!((cmp_same.log_rank_statistic - 0.0).abs() < 1e-12);
        assert!((cmp_same.log_rank_pvalue - 1.0).abs() < 1e-12);
        assert!((cmp_same.max_survival_difference - 0.0).abs() < 1e-12);
        assert!((cmp_same.rmst_difference - 0.0).abs() < 1e-12);

        // Divergent synthetic cohort (early high mortality vs late survival)
        let real_large: Vec<SurvivalObservation> = (0..20)
            .map(|i| SurvivalObservation {
                record_id: format!("real-{i}"),
                time: Some(80.0 + i as f64),
                event_occurred: Some(i % 2 == 0),
            })
            .collect();
        let synth_early_deaths: Vec<SurvivalObservation> = (0..20)
            .map(|i| SurvivalObservation {
                record_id: format!("synth-{i}"),
                time: Some(2.0 + (i % 5) as f64),
                event_occurred: Some(true),
            })
            .collect();

        let cmp_diff = compare_survival_curves(&real_large, &synth_early_deaths).unwrap();
        assert!(cmp_diff.log_rank_statistic > 10.0);
        assert!(cmp_diff.log_rank_pvalue < 0.005);
        assert!(cmp_diff.max_survival_difference > 0.8);
        assert_eq!(cmp_diff.synthetic_curve.median_survival_time, Some(4.0));

        // Negative survival time is recorded as an Integrity failure and rejected by comparison
        let invalid_obs = vec![SurvivalObservation {
            record_id: "bad-time".into(),
            time: Some(-5.0),
            event_occurred: Some(true),
        }];
        assert!(compute_kaplan_meier(&invalid_obs).is_err());
    }

    #[test]
    fn reproducibility_audit_validates_lockfile_platform_splits_and_software() {
        let lock_hash = cargo_lock_sha256();
        assert!(lock_hash.starts_with("sha256:"));
        let manifest = serde_json::json!({
            "schema_version": "1",
            "lockfile_hash": lock_hash,
            "platform": {
                "os": std::env::consts::OS,
                "arch": std::env::consts::ARCH
            },
            "fingerprints": {
                "holdout_fingerprint": "sha256:fictional-holdout-1234"
            },
            "software": {
                "package_version": env!("CARGO_PKG_VERSION"),
                "dependencies": {
                    "serde": "1",
                    "sha2": "0.10"
                }
            }
        });
        let splits = serde_json::json!({
            "schema_version": "1",
            "holdout_fingerprint": "sha256:fictional-holdout-1234"
        });

        let report = audit_reproducibility(&manifest, Some(&splits), true).unwrap();
        assert!(report.reproducible);
        assert!(report.warnings.is_empty());
        assert_eq!(report.checks.len(), 4);
        assert!(report.checks.iter().all(|c| c.passed));

        // Mismatched lockfile hash fails in strict mode
        let bad_lock = serde_json::json!({
            "lockfile_hash": "sha256:0000000000000000000000000000000000000000000000000000000000000000",
            "platform": {
                "os": std::env::consts::OS,
                "arch": std::env::consts::ARCH
            }
        });
        let report_bad_lock = audit_reproducibility(&bad_lock, None, true).unwrap();
        assert!(!report_bad_lock.reproducible);
        assert!(!report_bad_lock.warnings.is_empty());
        assert!(report_bad_lock
            .warnings
            .iter()
            .any(|w| w.contains("lockfile_fingerprint mismatch")));

        // Platform difference in warn-only mode emits warning while keeping reproducible = true
        let diff_platform = serde_json::json!({
            "lockfile_hash": lock_hash,
            "platform": {
                "os": "fictional-os",
                "arch": "fictional-arch"
            }
        });
        let report_warn = audit_reproducibility(&diff_platform, None, false).unwrap();
        assert!(report_warn.reproducible);
        assert_eq!(report_warn.warnings.len(), 1);
        assert!(report_warn.warnings[0].contains("platform_metadata mismatch"));

        // Same platform difference in strict mode sets reproducible = false
        let report_strict_platform = audit_reproducibility(&diff_platform, None, true).unwrap();
        assert!(!report_strict_platform.reproducible);
    }

    #[test]
    fn reproducibility_audit_accepts_fnv1a64_yaml_lockfile_and_split_fingerprints() {
        let yaml_lock = crate::yaml_fingerprint(&include_str!("../../../Cargo.lock")).unwrap();
        assert!(yaml_lock.starts_with("fnv1a64-yaml:"));
        let software = crate::software_identity().unwrap();
        assert_eq!(software.dependency_lock_fingerprint, yaml_lock);

        let manifest = serde_json::json!({
            "schema_version": "1",
            "software_version": software.package_version,
            "reference_fingerprint": "fnv1a64-yaml:abcdef0123456789",
            "software": {
                "package_version": software.package_version,
                "rustc": software.rustc,
                "target": software.target,
                "dependency_lock_fingerprint": yaml_lock,
                "operating_system": software.operating_system,
                "architecture": software.architecture,
            }
        });
        let splits = serde_json::json!({
            "schema_version": "1",
            "holdout_fingerprint": "FNV1A64-YAML:ABCDEF0123456789"
        });

        let report = audit_reproducibility(&manifest, Some(&splits), true).unwrap();
        assert!(report.reproducible);
        assert!(report.warnings.is_empty());
        assert_eq!(report.checks.len(), 4);
        assert!(report.checks.iter().all(|c| c.passed));
        assert_eq!(report.checks[0].name, "lockfile_fingerprint");
        assert_eq!(report.checks[0].actual, yaml_lock);

        // Mismatched fnv1a64-yaml lockfile fingerprint fails strictly and reports fnv1a64-yaml actual
        let bad_yaml_lock = serde_json::json!({
            "software": {
                "dependency_lock_fingerprint": "fnv1a64-yaml:0000000000000000",
                "operating_system": std::env::consts::OS,
                "architecture": std::env::consts::ARCH
            }
        });
        let report_bad = audit_reproducibility(&bad_yaml_lock, None, true).unwrap();
        assert!(!report_bad.reproducible);
        assert!(!report_bad.checks[0].passed);
        assert_eq!(report_bad.checks[0].actual, yaml_lock);
        assert!(report_bad
            .warnings
            .iter()
            .any(|w| w.contains("lockfile_fingerprint mismatch")));
    }
}
