use serde::{Deserialize, Serialize};
use std::{
    collections::{BTreeMap, BTreeSet, HashSet},
    fs,
    path::Path,
};
use thiserror::Error;

pub mod candidate;
pub mod coherence;
pub mod forensic;
pub mod forensic_workflow;
pub mod neural;
pub mod review;

pub use candidate::{
    compare_candidates, token_is_missing, AggregateComparisonManifest, BatchSampleResult,
    CandidateAggregateSummary, CandidateComparisonConfig, CandidateMetricsConfig,
    CandidateSeedMetrics, CohortGenerator, ComparisonDenominators, DagCandidate, FittedModel,
    ForestCandidate, InteractionDagCandidate, NeuralCandidate, Row, UtilityBrierConfig,
    UtilityBrierSummary,
};
pub use coherence::{
    audit_reproducibility, cargo_lock_sha256, compare_survival_curves, compute_kaplan_meier,
    parse_optional_f64, parse_optional_u8, validate_grade_progression, validate_mrs_monotonicity,
    validate_neurosurgical_cohort, validate_neurosurgical_record, validate_surgical_window_72h,
    validate_temporal_transitions, KaplanMeierCurve, KaplanMeierPoint, NeuroGradeScale,
    NeurosurgicalCoherenceRecord, OptionalField, ReproducibilityAuditReport, ReproducibilityCheck,
    SurvivalCurveComparison, SurvivalObservation, TemporalTransitionRecord, ValidationCategory,
    ValidationFailure, ValidationReport,
};
pub use forensic_workflow::{
    orchestrate_forensic_suite, train_logistic_discriminator, train_random_forest_discriminator,
    AggregateClassifierMetrics, DiscriminatorType, ForensicAggregateMetrics, ForensicCohort,
    ForensicEnvironment, ForensicFingerprints, ForensicManifest, ForensicOutcome, ForensicRow,
    SeedDiscriminatorMetrics, MINIMUM_FORENSIC_COHORT_ROWS,
};
pub use neural::GeneratorKind;

#[derive(Debug, Error)]
pub enum Error {
    #[error("input error: {0}")]
    Input(String),
    #[error("I/O: {0}")]
    Io(#[from] std::io::Error),
    #[error("yaml: {0}")]
    Yaml(#[from] serde_yaml::Error),
}

/// Manually curated descriptions of the assumptions used by a protocol.
#[derive(Debug, Deserialize, Serialize, Clone)]
#[serde(deny_unknown_fields)]
pub struct EvidenceRegistry {
    pub schema_version: String,
    pub records: Vec<EvidenceRecord>,
}

#[derive(Debug, Deserialize, Serialize, Clone, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct EvidenceRecord {
    pub id: String,
    pub title: String,
    pub description: String,
    pub source_type: EvidenceSource,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_locator: Option<String>,
    pub population_context: String,
    pub rationale: String,
    pub limitations: String,
    pub review_status: String,
    pub links: Vec<String>,
}

#[derive(Debug, Deserialize, Serialize, Clone, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum EvidenceSource {
    FictionalAssumption,
    ReviewedLiterature,
    DatasetNote,
}

impl EvidenceRegistry {
    pub fn validate(&self) -> Result<(), Error> {
        if self.schema_version != "1" {
            return Err(Error::Input(
                "unsupported evidence registry schema version".into(),
            ));
        }
        let mut ids = HashSet::new();
        for record in &self.records {
            if [
                &record.id,
                &record.title,
                &record.description,
                &record.population_context,
                &record.rationale,
                &record.limitations,
                &record.review_status,
            ]
            .iter()
            .any(|s| s.trim().is_empty())
                || record.links.is_empty()
                || record.links.iter().any(|s| s.trim().is_empty())
                || record
                    .source_locator
                    .as_ref()
                    .is_some_and(|s| s.trim().is_empty())
            {
                return Err(Error::Input(
                    "evidence record contains empty required fields".into(),
                ));
            }
            if !ids.insert(&record.id) {
                return Err(Error::Input(format!(
                    "duplicate evidence ID: {}",
                    record.id
                )));
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod evidence_tests {
    use super::*;

    #[test]
    fn registry_validates_records_references_and_fingerprints() {
        let yaml = include_str!("../../../examples/evidence.yaml");
        let registry: EvidenceRegistry = serde_yaml::from_str(yaml).unwrap();
        registry.validate().unwrap();
        let round_trip: EvidenceRegistry =
            serde_yaml::from_str(&serde_yaml::to_string(&registry).unwrap()).unwrap();
        assert_eq!(
            yaml_fingerprint(&registry).unwrap(),
            yaml_fingerprint(&round_trip).unwrap()
        );
        let mut changed = registry.clone();
        changed.records[0].description.push('!');
        assert_ne!(
            yaml_fingerprint(&registry).unwrap(),
            yaml_fingerprint(&changed).unwrap()
        );
        changed.records.push(changed.records[0].clone());
        assert!(changed.validate().is_err());
        assert!(serde_yaml::from_str::<EvidenceRegistry>(
            &yaml.replace("fictional_assumption", "unsupported")
        )
        .is_err());
        assert!(serde_yaml::from_str::<EvidenceRegistry>(
            &yaml.replace("    title: Fictional follow-up\n", "")
        )
        .is_err());
        let refs = BTreeMap::from([("study.follow_up_days".into(), "fictional-follow-up".into())]);
        assert!(evidence_provenance(Some(&registry), &[&refs]).is_ok());
        assert!(evidence_provenance(None, &[&refs]).is_err());
        let bad = BTreeMap::from([("study.wrong".into(), "fictional-follow-up".into())]);
        assert!(evidence_provenance(Some(&registry), &[&bad]).is_err());
        let missing = BTreeMap::from([("study.follow_up_days".into(), "unknown".into())]);
        assert!(evidence_provenance(Some(&registry), &[&missing]).is_err());
    }
}

#[derive(Debug, Deserialize, Serialize, Clone, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct EvidenceProvenance {
    pub registry_fingerprint: String,
    pub referenced_ids: Vec<String>,
    pub records: Vec<EvidenceRecord>,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct SoftwareIdentity {
    pub package_version: String,
    pub rustc: String,
    pub target: String,
    pub dependency_lock_fingerprint: String,
    pub operating_system: String,
    pub architecture: String,
    pub limitations: String,
}

pub fn software_identity() -> Result<SoftwareIdentity, Error> {
    Ok(SoftwareIdentity {
        package_version: env!("CARGO_PKG_VERSION").into(),
        rustc: option_env!("JANUS_RUSTC").unwrap_or("unavailable").into(),
        target: option_env!("JANUS_TARGET").unwrap_or("unavailable").into(),
        dependency_lock_fingerprint: yaml_fingerprint(&include_str!("../../../Cargo.lock"))?,
        operating_system: std::env::consts::OS.into(),
        architecture: std::env::consts::ARCH.into(),
        limitations: "Build toolchain and target are informational; environment, compiler flags, floating-point behavior and serialization can affect reproducibility. Lockfile identity does not prove the dependencies used at runtime.".into(),
    })
}

pub fn evidence_provenance(
    registry: Option<&EvidenceRegistry>,
    references: &[&BTreeMap<String, String>],
) -> Result<Option<EvidenceProvenance>, Error> {
    let Some(registry) = registry else {
        if references.iter().any(|r| !r.is_empty()) {
            return Err(Error::Input(
                "evidence references require a registry".into(),
            ));
        }
        return Ok(None);
    };
    registry.validate()?;
    let mut ids = std::collections::BTreeSet::new();
    for (link, id) in references.iter().flat_map(|r| r.iter()) {
        let record = registry
            .records
            .iter()
            .find(|record| &record.id == id)
            .ok_or_else(|| Error::Input(format!("unknown evidence ID: {id}")))?;
        if !record.links.contains(link) {
            return Err(Error::Input(format!(
                "evidence {id} does not support link {link}"
            )));
        }
        ids.insert(id.clone());
    }
    let records = ids
        .iter()
        .map(|id| {
            registry
                .records
                .iter()
                .find(|record| &record.id == id)
                .unwrap()
                .clone()
        })
        .collect();
    Ok(Some(EvidenceProvenance {
        registry_fingerprint: yaml_fingerprint(registry)?,
        referenced_ids: ids.into_iter().collect(),
        records,
    }))
}

pub fn validate_evidence_links(
    study: &Study,
    generator: Option<&GeneratorConfig>,
) -> Result<(), Error> {
    for link in study.evidence.keys() {
        if link != "study.follow_up_days"
            && !study
                .variables
                .iter()
                .any(|v| link == &format!("study.variable.{}", v.name))
        {
            return Err(Error::Input(format!("unknown study evidence link: {link}")));
        }
    }
    if let Some(generator) = generator {
        for link in generator.evidence.keys() {
            let valid = link == "generator.coverage"
                || generator
                    .variables
                    .iter()
                    .any(|v| link == &format!("generator.variable.{}.value", v.name))
                || generator.repeated_events.iter().any(|v| {
                    link == &format!("generator.repeated.{}.count", v.variable)
                        || link == &format!("generator.repeated.{}.timing", v.variable)
                });
            if !valid {
                return Err(Error::Input(format!(
                    "unknown generator evidence link: {link}"
                )));
            }
        }
    }
    Ok(())
}

pub fn validate_registry_links(
    registry: &EvidenceRegistry,
    study: &Study,
    generator: Option<&GeneratorConfig>,
) -> Result<(), Error> {
    for record in &registry.records {
        for link in &record.links {
            let parts: Vec<_> = link.split('.').collect();
            let valid = match parts.as_slice() {
                ["study", "follow_up_days"] | ["generator", "coverage"] => true,
                ["study", "variable", name] => study
                    .variables
                    .iter()
                    .any(|variable| variable.name == *name),
                ["generator", "variable", name, "value"] => generator.is_none_or(|config| {
                    config
                        .variables
                        .iter()
                        .any(|variable| variable.name == *name)
                        || study
                            .variables
                            .iter()
                            .any(|variable| variable.name == *name)
                }),
                ["generator", "repeated", name, "count" | "timing"] => {
                    generator.is_none_or(|config| {
                        config
                            .repeated_events
                            .iter()
                            .any(|event| event.variable == *name)
                            || study
                                .variables
                                .iter()
                                .any(|variable| variable.name == *name)
                    })
                }
                _ => false,
            };
            if !valid {
                return Err(Error::Input(format!(
                    "unknown registry parameter link: {link}"
                )));
            }
        }
    }
    Ok(())
}

#[derive(Debug, Deserialize, Serialize, Clone)]
#[serde(deny_unknown_fields)]
pub struct Study {
    pub schema_version: String,
    pub name: String,
    pub unit: String,
    pub variables: Vec<Variable>,
    pub index: String,
    pub eligibility: Vec<Rule>,
    pub follow_up_days: i64,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub evidence: BTreeMap<String, String>,
}
#[derive(Debug, Deserialize, Serialize, Clone)]
#[serde(deny_unknown_fields)]
pub struct Variable {
    pub name: String,
    pub kind: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub role: Option<VariableRole>,
}
#[derive(Debug, Deserialize, Serialize, Clone, Copy, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum VariableRole {
    Baseline,
    IndexContext,
    FollowUpOutcome,
}
#[derive(Debug, Deserialize, Serialize, Clone)]
#[serde(deny_unknown_fields)]
pub struct Rule {
    pub variable: String,
    pub equals: String,
}
#[derive(Debug, Deserialize, Serialize, Clone)]
#[serde(deny_unknown_fields)]
pub struct Fixture {
    pub schema_version: String,
    pub events: Vec<Event>,
    #[serde(default)]
    pub observations: Vec<Observation>,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
#[serde(deny_unknown_fields)]
pub struct Observation {
    pub patient_id: String,
    pub episode_id: String,
    pub start_day: i64,
    pub end_day: Option<i64>,
    pub censoring_reason: Option<String>,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
#[serde(deny_unknown_fields)]
pub struct SplitConfig {
    pub schema_version: String,
    pub seed: u64,
    pub development_proportion: f64,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct SplitManifest {
    pub schema_version: String,
    pub configuration: SplitConfig,
    pub algorithm: String,
    pub development_patients: usize,
    pub holdout_patients: usize,
    pub development_episodes: usize,
    pub holdout_episodes: usize,
    pub requested_development_proportion: f64,
    pub actual_development_patient_proportion: Option<f64>,
    pub holdout_fingerprint: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub development_fingerprint: Option<String>,
    pub artifacts: Vec<String>,
    pub reproducibility_limits: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub evidence: Option<EvidenceProvenance>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub software: Option<SoftwareIdentity>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub study_fingerprint: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub configuration_fingerprint: Option<String>,
}

pub struct SplitResult {
    pub development: Fixture,
    pub holdout: Fixture,
    pub manifest: SplitManifest,
}

fn patient_hash(seed: u64, patient_id: &str) -> u64 {
    let mut hash = 0xcbf29ce484222325u64;
    for byte in seed.to_le_bytes().iter().chain(patient_id.as_bytes()) {
        hash ^= *byte as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash ^= hash >> 30;
    hash = hash.wrapping_mul(0xbf58476d1ce4e5b9);
    hash ^= hash >> 27;
    hash = hash.wrapping_mul(0x94d049bb133111eb);
    hash ^ (hash >> 31)
}

fn in_development(hash: u64, proportion: f64) -> bool {
    if proportion <= 0.0 {
        return false;
    }
    if proportion >= 1.0 {
        return true;
    }
    let threshold = (proportion * 18_446_744_073_709_551_616.0).floor() as u128;
    (hash as u128) < threshold
}

/// Patient assignment avalanches the seeded FNV-1a hash before applying a threshold.
pub fn split_fixture(
    study: &Study,
    fixture: &Fixture,
    config: &SplitConfig,
) -> Result<SplitResult, Error> {
    validate(study, fixture)?;
    if config.schema_version != "1"
        || !config.development_proportion.is_finite()
        || !(0.0..=1.0).contains(&config.development_proportion)
    {
        return Err(Error::Input("invalid split configuration".into()));
    }
    let mut assignments = BTreeMap::new();
    for event in &fixture.events {
        assignments
            .entry(event.patient_id.clone())
            .or_insert_with(|| {
                in_development(
                    patient_hash(config.seed, &event.patient_id),
                    config.development_proportion,
                )
            });
    }
    let mut development = Vec::new();
    let mut holdout = Vec::new();
    for event in &fixture.events {
        if assignments[&event.patient_id] {
            development.push(event.clone());
        } else {
            holdout.push(event.clone());
        }
    }
    let dev_patients = assignments.values().filter(|v| **v).count();
    let total = assignments.len();
    let episode_count = |events: &[Event]| {
        events
            .iter()
            .map(|e| (&e.patient_id, &e.episode_id))
            .collect::<HashSet<_>>()
            .len()
    };
    let holdout = Fixture {
        schema_version: fixture.schema_version.clone(),
        events: holdout,
        observations: fixture
            .observations
            .iter()
            .filter(|o| !assignments.get(&o.patient_id).copied().unwrap_or(false))
            .cloned()
            .collect(),
    };
    let fingerprint = fixture_fingerprint(&holdout)?;
    let holdout_episodes = episode_count(&holdout.events);
    let development_fixture = Fixture {
        schema_version: fixture.schema_version.clone(),
        events: development.clone(),
        observations: fixture
            .observations
            .iter()
            .filter(|o| assignments.get(&o.patient_id).copied().unwrap_or(false))
            .cloned()
            .collect(),
    };
    let development_fingerprint = fixture_fingerprint(&development_fixture)?;
    Ok(SplitResult { development: development_fixture, holdout, manifest: SplitManifest { schema_version:"1".into(), configuration:config.clone(), algorithm:"FNV-1a 64-bit over little-endian seed bytes followed by UTF-8 patient ID, then SplitMix64 finalizer; integer threshold floor(proportion * 2^64), with explicit 0 and 1 bounds".into(), development_patients:dev_patients, holdout_patients:total-dev_patients, development_episodes:episode_count(&development), holdout_episodes, requested_development_proportion:config.development_proportion, actual_development_patient_proportion:(total>0).then_some(dev_patients as f64/total as f64), holdout_fingerprint:Some(fingerprint), development_fingerprint: Some(development_fingerprint), artifacts:vec!["development.yaml".into(),"holdout.yaml".into(),"manifest.yaml".into()], reproducibility_limits:"Stable assignment for identical UTF-8 identifiers, configuration and algorithm; previous FNV-1a-only versions produce different assignments. Canonical YAML fingerprint may vary if serialization behavior changes. This does not prove leakage prevention or validity. Empty fixtures have no actual proportion.".into(), evidence: None, software: None, study_fingerprint: Some(yaml_fingerprint(study)?), configuration_fingerprint: Some(yaml_fingerprint(config)?) } })
}
#[derive(Debug, Deserialize, Serialize, Clone)]
#[serde(deny_unknown_fields)]
pub struct GeneratorConfig {
    pub schema_version: String,
    pub seed: u64,
    pub episode_count: usize,
    pub variables: Vec<GeneratedVariable>,
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub complete_observation: bool,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub repeated_events: Vec<RepeatedEventConfig>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub evidence: BTreeMap<String, String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub registry_fingerprint: Option<String>,
}
#[derive(Debug, Deserialize, Serialize, Clone)]
#[serde(deny_unknown_fields)]
pub struct RepeatedEventConfig {
    pub variable: String,
    pub counts: Vec<GeneratedCount>,
    pub start_day: i64,
    pub end_day: i64,
}
#[derive(Debug, Deserialize, Serialize, Clone)]
#[serde(deny_unknown_fields)]
pub struct GeneratedCount {
    pub count: usize,
    pub probability: f64,
}
#[derive(Debug, Deserialize, Serialize, Clone)]
#[serde(deny_unknown_fields)]
pub struct GeneratedVariable {
    pub name: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub parents: Vec<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub outcomes: Vec<GeneratedOutcome>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub table: Vec<ConditionalDistribution>,
}
#[derive(Debug, Deserialize, Serialize, Clone)]
#[serde(deny_unknown_fields)]
pub struct ConditionalDistribution {
    pub when: BTreeMap<String, String>,
    pub outcomes: Vec<GeneratedOutcome>,
}
#[derive(Debug, Deserialize, Serialize, Clone)]
#[serde(deny_unknown_fields)]
pub struct GeneratedOutcome {
    pub value: String,
    pub probability: f64,
    pub event_day: i64,
    pub recording_day: i64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bin_end_day: Option<i64>,
}

#[derive(Debug, Deserialize, Serialize, Clone, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct OutcomeTimeBin {
    pub start_day: i64,
    pub end_day: i64,
}

impl OutcomeTimeBin {
    pub fn validate_ordered(bins: &[Self], follow_up_days: i64) -> Result<(), Error> {
        if bins.is_empty() || follow_up_days < 0 {
            return Err(Error::Input(
                "outcome timing requires bins and nonnegative follow-up".into(),
            ));
        }
        let mut previous_end = None;
        for bin in bins {
            if bin.start_day < 0
                || bin.start_day > bin.end_day
                || bin.end_day > follow_up_days
                || previous_end.is_some_and(|end| bin.start_day <= end)
            {
                return Err(Error::Input(
                    "outcome timing bins must be ordered, nonoverlapping and within follow-up"
                        .into(),
                ));
            }
            previous_end = Some(bin.end_day);
        }
        Ok(())
    }
}

/// Sample an inclusive nonnegative day uniformly using a SplitMix64 stream.
pub fn sample_outcome_day(state: &mut u64, bin: &OutcomeTimeBin) -> Result<i64, Error> {
    if bin.start_day < 0 || bin.start_day > bin.end_day {
        return Err(Error::Input("invalid outcome time bin".into()));
    }
    let width = (bin.end_day as u64) - (bin.start_day as u64) + 1;
    let reject_below = width.wrapping_neg() % width;
    loop {
        *state = state.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = *state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        let value = z ^ (z >> 31);
        if value >= reject_below {
            return Ok((bin.start_day as u64 + value % width) as i64);
        }
    }
}

#[derive(Debug, Deserialize, Serialize, Clone)]
#[serde(deny_unknown_fields)]
pub struct FitConfig {
    pub schema_version: String,
    pub seed: u64,
    pub episode_count: usize,
    pub event_day: i64,
    pub recording_day: i64,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub variables: Vec<FitVariable>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub timing: Option<OutcomeTimingConfig>,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
#[serde(deny_unknown_fields)]
pub struct OutcomeTimingConfig {
    pub variable: String,
    pub bins: Vec<OutcomeTimeBin>,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
#[serde(deny_unknown_fields)]
pub struct FitVariable {
    pub name: String,
    #[serde(default)]
    pub parents: Vec<String>,
}

#[derive(Debug, Serialize)]
pub struct FitManifest {
    pub schema_version: String,
    pub model_kind: String,
    pub study_name: String,
    pub study_fingerprint: String,
    pub development_fingerprint: String,
    pub configuration: FitConfig,
    pub patients: usize,
    pub episodes: usize,
    pub included_episodes: usize,
    pub excluded_episodes: usize,
    pub included_patients: usize,
    pub variables: BTreeMap<String, FitVariableCounts>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub timing_counts: Option<BTreeMap<String, usize>>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub timing_tables: Vec<TimingFitTable>,
    pub limitations: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub evidence: Option<EvidenceProvenance>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub software: Option<SoftwareIdentity>,
    pub configuration_fingerprint: String,
}
#[derive(Debug, Serialize)]
pub struct TimingFitTable {
    pub when: BTreeMap<String, String>,
    pub denominator: usize,
    pub cells: Vec<TimingFitCell>,
}
#[derive(Debug, Serialize)]
pub struct TimingFitCell {
    pub value: String,
    pub bin: OutcomeTimeBin,
    pub count: usize,
}

#[derive(Debug, Serialize)]
pub struct FitVariableCounts {
    pub observed: usize,
    pub missing: usize,
    pub absent: usize,
    pub categories: BTreeMap<String, usize>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub conditional: Vec<FitConditionalCounts>,
    pub parent_missing: usize,
    pub parent_absent: usize,
    pub parent_unsupported: usize,
}

#[derive(Debug, Serialize)]
pub struct FitConditionalCounts {
    pub when: BTreeMap<String, String>,
    pub observed: usize,
    pub missing: usize,
    pub absent: usize,
    pub categories: BTreeMap<String, usize>,
}

pub struct FitResult {
    pub generator: GeneratorConfig,
    pub manifest: FitManifest,
}

pub fn fit(study: &Study, development: &Fixture, config: &FitConfig) -> Result<FitResult, Error> {
    if config.schema_version == "3" {
        return fit_outcome_timing(study, development, config);
    }
    if config.timing.is_some() {
        return Err(Error::Input("timing requires fit version 3".into()));
    }
    validate(study, development)?;
    if !["1", "2"].contains(&config.schema_version.as_str())
        || config.episode_count == 0
        || config.episode_count > 1_000_000
    {
        return Err(Error::Input("invalid fit configuration".into()));
    }
    if config.schema_version == "1" && !config.variables.is_empty() {
        return Err(Error::Input(
            "fit variables require schema version 2".into(),
        ));
    }
    let report = run(study, development)?;
    let mut fitted = Vec::new();
    let mut counts = BTreeMap::new();
    let selections: Vec<FitVariable> = if config.schema_version == "1" {
        study
            .variables
            .iter()
            .map(|v| FitVariable {
                name: v.name.clone(),
                parents: vec![],
            })
            .collect()
    } else {
        config.variables.clone()
    };
    let mut seen = HashSet::new();
    for selection in &selections {
        let variable = study
            .variables
            .iter()
            .find(|v| v.name == selection.name)
            .ok_or_else(|| Error::Input(format!("unknown fit variable {}", selection.name)))?;
        if config.schema_version == "2" && variable.kind == "number" {
            return Err(Error::Input(format!(
                "conditional fit requires categorical variables; {} is number",
                variable.name
            )));
        }
        if !seen.insert(selection.name.clone())
            || selection
                .parents
                .iter()
                .any(|p| !seen.contains(p) || p == &selection.name)
            || selection.parents.iter().collect::<HashSet<_>>().len() != selection.parents.len()
        {
            return Err(Error::Input(format!(
                "invalid parent order or duplicate variable for {}",
                selection.name
            )));
        }
        let mut categories = BTreeMap::<String, usize>::new();
        let mut missing = 0;
        let mut absent = 0;
        for row in &report.cohort {
            match row
                .values
                .get(&variable.name)
                .unwrap_or(&ValueState::Absent)
            {
                ValueState::Observed(value) => *categories.entry(value.clone()).or_default() += 1,
                ValueState::Missing(_) => missing += 1,
                ValueState::Absent => absent += 1,
            }
        }
        let observed = categories.values().sum::<usize>();
        if observed == 0 {
            return Err(Error::Input(format!(
                "variable {} has no observed fit values",
                variable.name
            )));
        }
        let outcomes = categories
            .iter()
            .map(|(value, count)| GeneratedOutcome {
                value: value.clone(),
                probability: *count as f64 / observed as f64,
                event_day: config.event_day,
                recording_day: config.recording_day,
                bin_end_day: None,
            })
            .collect();
        fitted.push(GeneratedVariable {
            name: variable.name.clone(),
            parents: vec![],
            outcomes,
            table: vec![],
        });
        counts.insert(
            variable.name.clone(),
            FitVariableCounts {
                observed,
                missing,
                absent,
                categories,
                conditional: vec![],
                parent_missing: 0,
                parent_absent: 0,
                parent_unsupported: 0,
            },
        );
    }
    if !seen.contains(&study.index) {
        return Err(Error::Input("fit requires index variable".into()));
    }
    if config.schema_version == "2" {
        for (i, selection) in selections.iter().enumerate() {
            if selection.parents.is_empty() {
                continue;
            }
            let mut combinations = vec![BTreeMap::new()];
            for parent in &selection.parents {
                let pv = fitted
                    .iter()
                    .find(|v| &v.name == parent)
                    .expect("parent already fitted");
                let domain: Vec<_> = pv
                    .outcomes
                    .iter()
                    .chain(pv.table.iter().flat_map(|row| row.outcomes.iter()))
                    .map(|o| o.value.clone())
                    .collect::<std::collections::BTreeSet<_>>()
                    .into_iter()
                    .collect();
                combinations = combinations
                    .iter()
                    .flat_map(|c| {
                        domain.iter().map(move |value| {
                            let mut next = c.clone();
                            next.insert(parent.clone(), value.clone());
                            next
                        })
                    })
                    .collect();
            }
            let mut table = vec![];
            let mut details = vec![];
            for when in combinations {
                let mut categories = BTreeMap::<String, usize>::new();
                let mut missing = 0;
                let mut absent = 0;
                for row in &report.cohort {
                    if !when.iter().all(|(name, value)| matches!(row.values.get(name), Some(ValueState::Observed(v)) if v == value)) { continue; }
                    match &row.values[&selection.name] {
                        ValueState::Observed(v) => *categories.entry(v.clone()).or_default() += 1,
                        ValueState::Missing(_) => missing += 1,
                        ValueState::Absent => absent += 1,
                    }
                }
                let observed: usize = categories.values().sum();
                if observed == 0 {
                    return Err(Error::Input(format!(
                        "variable {} has no complete observations for parent combination {:?}",
                        selection.name, when
                    )));
                }
                let outcomes = categories
                    .iter()
                    .map(|(value, count)| GeneratedOutcome {
                        value: value.clone(),
                        probability: *count as f64 / observed as f64,
                        event_day: config.event_day,
                        recording_day: config.recording_day,
                        bin_end_day: None,
                    })
                    .collect();
                table.push(ConditionalDistribution {
                    when: when.clone(),
                    outcomes,
                });
                details.push(FitConditionalCounts {
                    when,
                    observed,
                    missing,
                    absent,
                    categories,
                });
            }
            fitted[i].parents = selection.parents.clone();
            fitted[i].outcomes.clear();
            fitted[i].table = table;
            let entry = counts
                .get_mut(&selection.name)
                .expect("fit variable exists");
            entry.conditional = details;
            for row in &report.cohort {
                if selection.parents.iter().any(|p| matches!(row.values[p], ValueState::Absent)) {
                    entry.parent_absent += 1;
                } else if selection.parents.iter().any(|p| matches!(row.values[p], ValueState::Missing(_))) {
                    entry.parent_missing += 1;
                } else if !entry.conditional.iter().any(|c| c.when.iter().all(|(name, value)| matches!(row.values.get(name), Some(ValueState::Observed(v)) if v == value))) {
                    entry.parent_unsupported += 1;
                }
            }
        }
    }
    let patients = development
        .events
        .iter()
        .map(|event| &event.patient_id)
        .collect::<HashSet<_>>()
        .len();
    let included_patients = report
        .cohort
        .iter()
        .map(|row| &row.patient_id)
        .collect::<HashSet<_>>()
        .len();
    let episodes = development
        .events
        .iter()
        .map(|event| (&event.patient_id, &event.episode_id))
        .collect::<HashSet<_>>()
        .len();
    let generator = GeneratorConfig {
        schema_version: config.schema_version.clone(),
        seed: config.seed,
        episode_count: config.episode_count,
        variables: fitted,
        complete_observation: false,
        repeated_events: vec![],
        evidence: BTreeMap::new(),
        registry_fingerprint: None,
    };
    validate_generator(study, &generator)?;
    Ok(FitResult {
        generator,
        manifest: FitManifest {
            schema_version: config.schema_version.clone(), model_kind: if config.schema_version == "2" { "empirical conditional categorical" } else { "fictional empirical descriptive baseline" }.into(),
            study_name: study.name.clone(), study_fingerprint: yaml_fingerprint(study)?, development_fingerprint: fixture_fingerprint(development)?,
            configuration: config.clone(), patients, episodes, included_episodes: report.cohort.len(), excluded_episodes: report.exclusions.len(), included_patients, variables: counts, timing_counts: None, timing_tables: vec![],
            limitations: vec!["Descriptive categorical distributions only; no causal interpretation or temporal dependency/repeated-trajectory modeling.".into(), "No guarantee of privacy or clinical validity; fixture fingerprint does not prove development use or absence of leakage.".into(), "Reproducibility is limited to the same software, dependencies and environment.".into()], evidence: None, software: None, configuration_fingerprint: yaml_fingerprint(config)?,
        },
    })
}

fn fit_outcome_timing(
    study: &Study,
    fixture: &Fixture,
    config: &FitConfig,
) -> Result<FitResult, Error> {
    let timing = config
        .timing
        .as_ref()
        .ok_or_else(|| Error::Input("fit version 3 requires timing configuration".into()))?;
    OutcomeTimeBin::validate_ordered(&timing.bins, study.follow_up_days)?;
    if study.schema_version != "2"
        || variable_role(study, &timing.variable) != Some(VariableRole::FollowUpOutcome)
    {
        return Err(Error::Input(
            "timing variable must be a version 2 follow_up_outcome".into(),
        ));
    }
    if config.event_day != 0 || config.recording_day != 0 {
        return Err(Error::Input("timing fit requires index day 0".into()));
    }
    let selection = config
        .variables
        .iter()
        .find(|v| v.name == timing.variable)
        .ok_or_else(|| Error::Input("timing variable must be selected for fitting".into()))?;
    if config
        .variables
        .last()
        .is_none_or(|v| v.name != timing.variable)
    {
        return Err(Error::Input(
            "timing variable must be last in fit order".into(),
        ));
    }
    if selection.parents.iter().any(|name| {
        variable_role(study, name) != Some(VariableRole::Baseline)
            && variable_role(study, name) != Some(VariableRole::IndexContext)
    }) {
        return Err(Error::Input(
            "timing parents must be baseline or index context".into(),
        ));
    }
    let mut base = config.clone();
    base.schema_version = "2".into();
    base.timing = None;
    let mut result = fit(study, fixture, &base)?;
    let report = run(study, fixture)?;
    let mut totals: BTreeMap<String, usize> = [
        "observed",
        "unknown_coverage",
        "incomplete_coverage",
        "missing",
        "absent",
        "outside_bins",
        "incomplete_parents",
        "unsupported_parents",
    ]
    .into_iter()
    .map(|key| (key.into(), 0))
    .collect();
    let mut groups = BTreeMap::<Vec<String>, BTreeMap<(String, usize), usize>>::new();
    for row in &report.cohort {
        let (value, day) = match outcome_timing_state(study, fixture, row, &timing.variable)? {
            OutcomeTimingState::UnknownCoverage => {
                *totals.entry("unknown_coverage".into()).or_default() += 1;
                continue;
            }
            OutcomeTimingState::IncompleteCoverage => {
                *totals.entry("incomplete_coverage".into()).or_default() += 1;
                continue;
            }
            OutcomeTimingState::Missing => {
                *totals.entry("missing".into()).or_default() += 1;
                continue;
            }
            OutcomeTimingState::Absent => {
                *totals.entry("absent".into()).or_default() += 1;
                continue;
            }
            OutcomeTimingState::Observed { value, day } => (value, day),
        };
        let Some(bin) = timing
            .bins
            .iter()
            .position(|b| day >= b.start_day && day <= b.end_day)
        else {
            *totals.entry("outside_bins".into()).or_default() += 1;
            continue;
        };
        let parents: Option<Vec<String>> = selection
            .parents
            .iter()
            .map(|p| match &row.values[p] {
                ValueState::Observed(v) => Some(v.clone()),
                _ => None,
            })
            .collect();
        let Some(parents) = parents else {
            *totals.entry("incomplete_parents".into()).or_default() += 1;
            continue;
        };
        *groups
            .entry(parents)
            .or_default()
            .entry((value, bin))
            .or_default() += 1;
        *totals.entry("observed".into()).or_default() += 1;
    }
    let target = result
        .generator
        .variables
        .iter_mut()
        .find(|v| v.name == timing.variable)
        .expect("fit selection exists");
    let combinations: Vec<BTreeMap<String, String>> = if target.parents.is_empty() {
        vec![BTreeMap::new()]
    } else {
        target.table.iter().map(|r| r.when.clone()).collect()
    };
    let supported: HashSet<Vec<String>> = combinations
        .iter()
        .map(|when| selection.parents.iter().map(|p| when[p].clone()).collect())
        .collect();
    for (parents, cells) in &groups {
        if !supported.contains(parents) {
            let n: usize = cells.values().sum();
            *totals.get_mut("unsupported_parents").unwrap() += n;
            *totals.get_mut("observed").unwrap() -= n;
        }
    }
    let mut table = Vec::new();
    let mut timing_tables = Vec::new();
    for when in combinations {
        let key: Vec<_> = selection.parents.iter().map(|p| when[p].clone()).collect();
        let cells = groups.get(&key).ok_or_else(|| {
            Error::Input(format!(
                "no complete timing observations for parent combination {when:?}"
            ))
        })?;
        let denominator: usize = cells.values().sum();
        timing_tables.push(TimingFitTable {
            when: when.clone(),
            denominator,
            cells: cells
                .iter()
                .map(|((value, bin), count)| TimingFitCell {
                    value: value.clone(),
                    bin: timing.bins[*bin].clone(),
                    count: *count,
                })
                .collect(),
        });
        let outcomes = cells
            .iter()
            .map(|((value, bin), count)| GeneratedOutcome {
                value: value.clone(),
                probability: *count as f64 / denominator as f64,
                event_day: timing.bins[*bin].start_day,
                recording_day: timing.bins[*bin].start_day,
                bin_end_day: Some(timing.bins[*bin].end_day),
            })
            .collect();
        table.push(ConditionalDistribution { when, outcomes });
    }
    if target.parents.is_empty() {
        target.outcomes = table.remove(0).outcomes;
        target.table.clear();
    } else {
        target.outcomes.clear();
        target.table = table;
    }
    result.generator.schema_version = "3".into();
    result.generator.complete_observation = true;
    validate_generator(study, &result.generator)?;
    result.manifest.schema_version = "3".into();
    result.manifest.model_kind = "descriptive categorical outcome timing".into();
    result.manifest.configuration = config.clone();
    result.manifest.timing_counts = Some(totals);
    result.manifest.timing_tables = timing_tables;
    result.manifest.limitations.push("Timing uses complete declared follow-up only and uniform sampling within bins; no censoring mechanism is estimated.".into());
    Ok(result)
}

fn validate_generator(study: &Study, config: &GeneratorConfig) -> Result<(), Error> {
    if config.schema_version == "4" && config.complete_observation && study.follow_up_days < 0 {
        return Err(Error::Input(
            "complete generated observation requires nonnegative follow-up".into(),
        ));
    }
    if !["3", "4"].contains(&config.schema_version.as_str())
        && (config.complete_observation
            || config
                .variables
                .iter()
                .flat_map(|v| {
                    v.outcomes
                        .iter()
                        .chain(v.table.iter().flat_map(|r| r.outcomes.iter()))
                })
                .any(|o| o.bin_end_day.is_some()))
    {
        return Err(Error::Input(
            "timed outcomes and generated observation require generator version 3".into(),
        ));
    }
    validate(
        study,
        &Fixture {
            schema_version: "1".into(),
            events: vec![],
            observations: vec![],
        },
    )?;
    if !["1", "2", "3", "4"].contains(&config.schema_version.as_str())
        || config.episode_count == 0
        || config.episode_count > 1_000_000
    {
        return Err(Error::Input(
            "unsupported generator version or episode count (1..=1000000 required)".into(),
        ));
    }
    if config.schema_version != "4" && !config.repeated_events.is_empty() {
        return Err(Error::Input(
            "repeated events require generator version 4".into(),
        ));
    }
    let mut repeated_names = HashSet::new();
    for repeated in &config.repeated_events {
        let declared = study
            .variables
            .iter()
            .find(|v| v.name == repeated.variable)
            .ok_or_else(|| {
                Error::Input(format!("unknown repeated variable {}", repeated.variable))
            })?;
        if repeated.variable == study.index
            || !repeated_names.insert(&repeated.variable)
            || declared.role != Some(VariableRole::FollowUpOutcome)
            || repeated.start_day < 0
            || repeated.start_day > repeated.end_day
            || repeated.end_day > study.follow_up_days
            || repeated.counts.is_empty()
            || repeated.counts.iter().any(|c| {
                c.count > 100 || !c.probability.is_finite() || !(0.0..=1.0).contains(&c.probability)
            })
            || (repeated.counts.iter().map(|c| c.probability).sum::<f64>() - 1.0).abs() > 1e-12
        {
            return Err(Error::Input(format!(
                "invalid repeated event configuration for {}",
                repeated.variable
            )));
        }
        if !config.variables.iter().any(|v| v.name == repeated.variable) {
            return Err(Error::Input(format!(
                "repeated variable {} requires a categorical distribution",
                repeated.variable
            )));
        }
        let repeated_variable = config
            .variables
            .iter()
            .find(|v| v.name == repeated.variable)
            .unwrap();
        if !repeated_variable.parents.is_empty() {
            return Err(Error::Input(format!(
                "repeated variable {} must use a root categorical distribution",
                repeated.variable
            )));
        }
    }
    let mut names = HashSet::new();
    let mut timed_variables = HashSet::new();
    let mut domains: BTreeMap<String, HashSet<String>> = BTreeMap::new();
    for variable in &config.variables {
        let declared = study
            .variables
            .iter()
            .find(|v| v.name == variable.name)
            .ok_or_else(|| Error::Input(format!("unknown generated variable {}", variable.name)))?;
        if !names.insert(&variable.name) {
            return Err(Error::Input(
                "duplicate generated variable or empty outcomes".into(),
            ));
        }
        let mut parent_names = HashSet::new();
        for parent in &variable.parents {
            if !parent_names.insert(parent) || !domains.contains_key(parent) {
                return Err(Error::Input(format!(
                    "invalid parent {parent} for {}",
                    variable.name
                )));
            }
        }
        if config.schema_version == "1"
            && (!variable.parents.is_empty() || !variable.table.is_empty())
        {
            return Err(Error::Input(
                "version 1 does not support conditional tables".into(),
            ));
        }
        let distributions: Vec<&Vec<GeneratedOutcome>> = if variable.parents.is_empty() {
            if !variable.table.is_empty() {
                return Err(Error::Input("root variable cannot have a table".into()));
            }
            vec![&variable.outcomes]
        } else {
            if !variable.outcomes.is_empty() {
                return Err(Error::Input(
                    "conditional variable cannot have root outcomes".into(),
                ));
            }
            let mut combinations = HashSet::new();
            for row in &variable.table {
                if row.when.len() != variable.parents.len()
                    || variable
                        .parents
                        .iter()
                        .any(|p| !row.when.get(p).is_some_and(|v| domains[p].contains(v)))
                {
                    return Err(Error::Input("invalid parent combination".into()));
                }
                let key: Vec<_> = variable
                    .parents
                    .iter()
                    .map(|p| row.when[p].clone())
                    .collect();
                if !combinations.insert(key) {
                    return Err(Error::Input("duplicate parent combination".into()));
                }
            }
            let count = variable
                .parents
                .iter()
                .try_fold(1usize, |n, p| n.checked_mul(domains[p].len()))
                .ok_or_else(|| Error::Input("too many parent combinations".into()))?;
            if combinations.len() != count {
                return Err(Error::Input("incomplete parent combinations".into()));
            }
            variable.table.iter().map(|row| &row.outcomes).collect()
        };
        let mut domain = HashSet::new();
        for outcomes in distributions {
            if outcomes.is_empty() {
                return Err(Error::Input("empty outcomes".into()));
            }
            let mut sum = 0.0;
            let mut values = HashSet::new();
            for outcome in outcomes {
                if !values.insert((&outcome.value, outcome.event_day, outcome.bin_end_day)) {
                    return Err(Error::Input("duplicate outcome value".into()));
                }
                if config.schema_version != "3"
                    && outcomes.iter().filter(|o| o.value == outcome.value).count() > 1
                {
                    return Err(Error::Input("duplicate outcome value".into()));
                }
                if let Some(end) = outcome.bin_end_day {
                    if !["3", "4"].contains(&config.schema_version.as_str())
                        || declared.role != Some(VariableRole::FollowUpOutcome)
                        || outcome.recording_day != outcome.event_day
                    {
                        return Err(Error::Input(format!(
                            "invalid timed outcome for {}",
                            variable.name
                        )));
                    }
                    OutcomeTimeBin::validate_ordered(
                        &[OutcomeTimeBin {
                            start_day: outcome.event_day,
                            end_day: end,
                        }],
                        study.follow_up_days,
                    )?;
                    timed_variables.insert(variable.name.clone());
                }
                if ["3", "4"].contains(&config.schema_version.as_str())
                    && variable.name == study.index
                    && (outcome.event_day != 0
                        || outcome.recording_day != 0
                        || outcome.bin_end_day.is_some())
                {
                    return Err(Error::Input(
                        "version 3 index must occur and be recorded at day 0".into(),
                    ));
                }
                domain.insert(outcome.value.clone());
                if !outcome.probability.is_finite()
                    || !(0.0..=1.0).contains(&outcome.probability)
                    || outcome.recording_day < outcome.event_day
                    || !valid_value(&declared.kind, &outcome.value)
                {
                    return Err(Error::Input(format!(
                        "invalid outcome for {}",
                        variable.name
                    )));
                }
                if variable.name == study.index
                    && outcome
                        .event_day
                        .checked_add(study.follow_up_days)
                        .is_none()
                {
                    return Err(Error::Input("index follow-up boundary overflows".into()));
                }
                sum += outcome.probability;
            }
            if (sum - 1.0).abs() > 1e-12 {
                return Err(Error::Input(format!(
                    "probabilities for {} must sum to 1",
                    variable.name
                )));
            }
        }
        domains.insert(variable.name.clone(), domain);
    }
    if config.schema_version == "4" {
        for repeated in &config.repeated_events {
            let variable = config
                .variables
                .iter()
                .find(|v| v.name == repeated.variable)
                .expect("validated repeated variable");
            for outcome in &variable.outcomes {
                if outcome.event_day < 0
                    || outcome.recording_day < outcome.event_day
                    || outcome.event_day > study.follow_up_days
                    || outcome.bin_end_day.is_some()
                {
                    return Err(Error::Input(format!("repeated variable {} requires values valid within follow-up and without outcome timing bins", repeated.variable)));
                }
            }
        }
    }
    if !names.contains(&study.index) {
        return Err(Error::Input("generator requires index variable".into()));
    }
    if config.schema_version == "3" && timed_variables.len() != 1 {
        return Err(Error::Input(
            "version 3 requires exactly one timed outcome variable".into(),
        ));
    }
    if ["3", "4"].contains(&config.schema_version.as_str()) && !timed_variables.is_empty() {
        let name = timed_variables.iter().next().expect("one timed variable");
        let variable = config
            .variables
            .iter()
            .find(|v| &v.name == name)
            .expect("timed variable exists");
        let outcomes: Vec<_> = variable
            .outcomes
            .iter()
            .chain(variable.table.iter().flat_map(|r| r.outcomes.iter()))
            .collect();
        if outcomes.iter().any(|o| o.bin_end_day.is_none()) {
            return Err(Error::Input(
                "timed variable requires a bin for every outcome".into(),
            ));
        }
        let bins: Vec<_> = outcomes
            .iter()
            .map(|o| (o.event_day, o.bin_end_day.expect("checked")))
            .collect::<std::collections::BTreeSet<_>>()
            .into_iter()
            .map(|(start_day, end_day)| OutcomeTimeBin { start_day, end_day })
            .collect();
        OutcomeTimeBin::validate_ordered(&bins, study.follow_up_days)?;
    }
    Ok(())
}

pub fn generate(study: &Study, config: &GeneratorConfig) -> Result<Fixture, Error> {
    validate_generator(study, config)?;
    let mut state = config.seed;
    let mut events = Vec::new();
    let mut observations = Vec::new();
    for i in 0..config.episode_count {
        let event_start = events.len();
        let repeated_set: HashSet<&str> = config
            .repeated_events
            .iter()
            .map(|r| r.variable.as_str())
            .collect();
        let mut sampled = BTreeMap::new();
        for (j, variable) in config.variables.iter().enumerate() {
            if config.schema_version == "4" && repeated_set.contains(variable.name.as_str()) {
                continue;
            }
            // SplitMix64, with the upper 53 bits mapped to [0, 1).
            state = state.wrapping_add(0x9e3779b97f4a7c15);
            let mut z = state;
            z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
            let sample = ((z ^ (z >> 31)) >> 11) as f64 / 9007199254740992.0;
            let mut cumulative = 0.0;
            let outcomes = if variable.parents.is_empty() {
                &variable.outcomes
            } else {
                &variable
                    .table
                    .iter()
                    .find(|row| {
                        row.when
                            .iter()
                            .all(|(key, value)| sampled.get(key) == Some(value))
                    })
                    .expect("validated table")
                    .outcomes
            };
            let mut selected = outcomes
                .iter()
                .rev()
                .find(|outcome| outcome.probability > 0.0)
                .expect("validated probability sum");
            for outcome in outcomes {
                cumulative += outcome.probability;
                if sample < cumulative {
                    selected = outcome;
                    break;
                }
            }
            sampled.insert(variable.name.clone(), selected.value.clone());
            let event_day = if let Some(end) = selected.bin_end_day {
                sample_outcome_day(
                    &mut state,
                    &OutcomeTimeBin {
                        start_day: selected.event_day,
                        end_day: end,
                    },
                )?
            } else {
                selected.event_day
            };
            events.push(Event {
                patient_id: format!("patient-{i:08}"),
                episode_id: format!("episode-{i:08}"),
                id: format!("event-{i:08}-{j:04}"),
                variable: variable.name.clone(),
                value: Some(selected.value.clone()),
                missing: None,
                event_day,
                recording_day: if selected.bin_end_day.is_some() {
                    event_day
                } else {
                    selected.recording_day
                },
                provenance: format!("configured-generator-v{}", config.schema_version),
            });
        }
        if config.schema_version == "4" {
            for repeated in &config.repeated_events {
                state = state.wrapping_add(0x9e3779b97f4a7c15);
                let mut z = state;
                z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
                z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
                let sample = ((z ^ (z >> 31)) >> 11) as f64 / 9007199254740992.0;
                let mut cumulative = 0.0;
                let mut count = repeated.counts.last().unwrap().count;
                for c in &repeated.counts {
                    cumulative += c.probability;
                    if sample < cumulative {
                        count = c.count;
                        break;
                    }
                }
                let var = config
                    .variables
                    .iter()
                    .find(|v| v.name == repeated.variable)
                    .unwrap();
                for n in 0..count {
                    state = state.wrapping_add(0x9e3779b97f4a7c15);
                    let mut z = state;
                    z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
                    z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
                    let sample = ((z ^ (z >> 31)) >> 11) as f64 / 9007199254740992.0;
                    let mut cumulative = 0.0;
                    let mut selected = var
                        .outcomes
                        .iter()
                        .rev()
                        .find(|o| o.probability > 0.0)
                        .unwrap();
                    for o in &var.outcomes {
                        cumulative += o.probability;
                        if sample < cumulative {
                            selected = o;
                            break;
                        }
                    }
                    state = state.wrapping_add(0x9e3779b97f4a7c15);
                    let mut z = state;
                    z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
                    z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
                    let mixed = z ^ (z >> 31);
                    let width = (repeated.end_day as i128 - repeated.start_day as i128 + 1) as u64;
                    let offset = ((mixed as u128 * width as u128) >> 64) as u64;
                    let day = (repeated.start_day as i128 + offset as i128) as i64;
                    events.push(Event {
                        patient_id: format!("patient-{i:08}"),
                        episode_id: format!("episode-{i:08}"),
                        id: format!("event-{i:08}-repeated-{}-{n}", repeated.variable),
                        variable: repeated.variable.clone(),
                        value: Some(selected.value.clone()),
                        missing: None,
                        event_day: day,
                        recording_day: day,
                        provenance: "configured-generator-v4".into(),
                    });
                }
            }
            events[event_start..]
                .sort_by(|a, b| a.event_day.cmp(&b.event_day).then_with(|| a.id.cmp(&b.id)));
        }
        if config.complete_observation {
            observations.push(Observation {
                patient_id: format!("patient-{i:08}"),
                episode_id: format!("episode-{i:08}"),
                start_day: 0,
                end_day: Some(study.follow_up_days),
                censoring_reason: None,
            });
        }
    }
    let fixture = Fixture {
        schema_version: if config.complete_observation {
            "2"
        } else {
            "1"
        }
        .into(),
        events,
        observations,
    };
    validate(study, &fixture)?;
    Ok(fixture)
}
#[derive(Debug, Deserialize, Serialize, Clone)]
#[serde(deny_unknown_fields)]
pub struct Event {
    pub patient_id: String,
    pub episode_id: String,
    pub id: String,
    pub variable: String,
    pub value: Option<String>,
    pub missing: Option<String>,
    pub event_day: i64,
    pub recording_day: i64,
    pub provenance: String,
}
#[derive(Debug, Serialize)]
pub struct CohortRow {
    pub patient_id: String,
    pub episode_id: String,
    pub index_day: i64,
    pub values: BTreeMap<String, ValueState>,
    pub eligibility_values: BTreeMap<String, EligibilityValue>,
    pub source_event_ids: Vec<String>,
    #[serde(skip_serializing_if = "BTreeMap::is_empty")]
    pub variable_source_event_ids: BTreeMap<String, Option<String>>,
    pub observation_start_day: Option<i64>,
    pub observation_end_day: Option<i64>,
    pub censoring_reason: Option<String>,
}

#[derive(Debug, PartialEq, Eq)]
pub enum OutcomeTimingState {
    UnknownCoverage,
    IncompleteCoverage,
    Missing,
    Absent,
    Observed { value: String, day: i64 },
}

/// Classify an included episode using declared coverage and the selected source event.
pub fn outcome_timing_state(
    study: &Study,
    fixture: &Fixture,
    row: &CohortRow,
    variable: &str,
) -> Result<OutcomeTimingState, Error> {
    if row.observation_start_day.is_none() {
        return Ok(OutcomeTimingState::UnknownCoverage);
    }
    let end = i128::from(row.index_day) + i128::from(study.follow_up_days);
    if row
        .observation_end_day
        .is_none_or(|day| i128::from(day) < end)
    {
        return Ok(OutcomeTimingState::IncompleteCoverage);
    }
    match row
        .values
        .get(variable)
        .ok_or_else(|| Error::Input(format!("unknown outcome variable {variable}")))?
    {
        ValueState::Absent => Ok(OutcomeTimingState::Absent),
        ValueState::Missing(_) => Ok(OutcomeTimingState::Missing),
        ValueState::Observed(value) => {
            let id = row
                .variable_source_event_ids
                .get(variable)
                .and_then(Option::as_ref)
                .ok_or_else(|| Error::Input(format!("missing source event for {variable}")))?;
            let event = fixture
                .events
                .iter()
                .find(|e| &e.id == id)
                .ok_or_else(|| Error::Input(format!("unknown source event {id}")))?;
            let day = event
                .event_day
                .checked_sub(row.index_day)
                .ok_or_else(|| Error::Input("outcome day difference overflows".into()))?;
            Ok(OutcomeTimingState::Observed {
                value: value.clone(),
                day,
            })
        }
    }
}
#[derive(Debug, Serialize)]
pub struct EligibilityValue {
    pub value: ValueState,
    pub source_event_id: Option<String>,
}
#[derive(Debug, Serialize, Clone, PartialEq)]
#[serde(tag = "status", content = "value")]
pub enum ValueState {
    Observed(String),
    Missing(String),
    Absent,
}
#[derive(Debug, Serialize)]
pub struct Exclusion {
    pub patient_id: String,
    pub episode_id: String,
    pub reason: String,
    pub source_event_ids: Vec<String>,
}
#[derive(Debug, Serialize)]
pub struct RunManifest {
    pub schema_version: String,
    pub artifacts: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub generator: Option<GeneratorManifest>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub evidence: Option<EvidenceProvenance>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub software: Option<SoftwareIdentity>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub study_fingerprint: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub generator_fingerprint: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub fixture_fingerprint: Option<String>,
}
#[derive(Debug, Serialize)]
pub struct GeneratorManifest {
    pub configuration: GeneratorConfig,
    pub rng: String,
    pub software_version: String,
    pub reproducibility_limits: String,
}
#[derive(Debug, Serialize)]
pub struct RunReport {
    pub cohort: Vec<CohortRow>,
    pub exclusions: Vec<Exclusion>,
    pub summary: CohortSummary,
    pub diagnostics: Vec<Diagnostic>,
    pub integrity: Vec<String>,
    pub manifest: RunManifest,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
#[serde(deny_unknown_fields)]
pub struct EvaluationConfig {
    pub schema_version: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub joint: Option<JointSelection>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub pairs: BTreeMap<String, JointSelection>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub temporal: Option<TemporalConfig>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub outcome_timing: Option<OutcomeTimingConfig>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub longitudinal: Option<LongitudinalConfig>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub uncertainty: Option<UncertaintyConfig>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub continuous: Option<ContinuousConfig>,
}
#[derive(Debug, Deserialize, Serialize, Clone)]
#[serde(deny_unknown_fields)]
pub struct ContinuousConfig {
    pub variables: Vec<String>,
    #[serde(default = "default_quantiles")]
    pub quantiles: Vec<f64>,
}
fn default_quantiles() -> Vec<f64> {
    vec![0.25, 0.5, 0.75]
}
#[derive(Debug, Serialize)]
pub struct ContinuousComparison {
    pub synthetic: ContinuousCohortSummary,
    pub reference: ContinuousCohortSummary,
    pub empirical_cdf_distance: Option<f64>,
    pub denominator: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bootstrap: Option<ContinuousBootstrap>,
}
#[derive(Debug, Serialize)]
pub struct ContinuousBootstrap {
    pub configuration: UncertaintyConfig,
    pub valid_replicates: usize,
    pub mean_interval: Option<(f64, f64)>,
    pub sample_standard_deviation_interval: Option<(f64, f64)>,
    pub signed_mean_difference_interval: Option<(f64, f64)>,
}
#[derive(Debug, Serialize)]
pub struct ContinuousCohortSummary {
    pub observed: usize,
    pub missing: usize,
    pub absent: usize,
    pub invalid: usize,
    pub patients: usize,
    pub episodes: usize,
    pub mean: Option<f64>,
    pub sample_standard_deviation: Option<f64>,
    pub quantiles: BTreeMap<String, Option<f64>>,
}
#[derive(Debug, Deserialize, Serialize, Clone)]
#[serde(deny_unknown_fields)]
pub struct UncertaintyConfig {
    pub seed: u64,
    pub replicates: usize,
    pub confidence_level: f64,
}
#[derive(Debug, Deserialize, Serialize, Clone)]
#[serde(deny_unknown_fields)]
pub struct TemporalConfig {
    pub windows: Vec<TemporalWindow>,
}
#[derive(Debug, Deserialize, Serialize, Clone)]
#[serde(deny_unknown_fields)]
pub struct TemporalWindow {
    pub name: String,
    pub start_day: i64,
    pub end_day: i64,
}
#[derive(Debug, Deserialize, Serialize, Clone)]
#[serde(deny_unknown_fields)]
pub struct LongitudinalConfig {
    pub start_day: i64,
    pub end_day: i64,
    pub variables: Vec<String>,
    pub pairs: BTreeMap<String, JointSelection>,
}

impl LongitudinalConfig {
    pub fn validate(&self, study: &Study) -> Result<(), Error> {
        if self.start_day > self.end_day || self.variables.is_empty() {
            return Err(Error::Input(
                "longitudinal window must be ordered and select variables".into(),
            ));
        }
        let selected: HashSet<_> = self.variables.iter().collect();
        if selected.len() != self.variables.len()
            || self
                .variables
                .iter()
                .any(|name| !study.variables.iter().any(|v| &v.name == name))
        {
            return Err(Error::Input(
                "longitudinal variables must be unique study variables".into(),
            ));
        }
        if self.pairs.iter().any(|(name, pair)| {
            name.trim().is_empty()
                || pair.exposure == pair.outcome
                || !selected.contains(&pair.exposure)
                || !selected.contains(&pair.outcome)
        }) {
            return Err(Error::Input(
                "longitudinal pairs require a name and two distinct selected variables".into(),
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Serialize, Default)]
pub struct LongitudinalEpisode {
    pub patient_id: String,
    pub episode_id: String,
    pub coverage: String,
    pub variables: BTreeMap<String, Vec<LongitudinalEvent>>,
}
#[derive(Debug, Serialize)]
pub struct LongitudinalEvent {
    pub source_event_id: String,
    pub provenance: String,
    pub day: i128,
    pub missing: bool,
    pub missing_reason: Option<String>,
    pub recording_day: i128,
}

#[derive(Debug, Serialize, Default)]
pub struct LongitudinalSummary {
    pub coverage: BTreeMap<String, usize>,
    pub variables: BTreeMap<String, LongitudinalVariableSummary>,
    pub pairs: BTreeMap<String, LongitudinalPairSummary>,
}
#[derive(Debug, Serialize, Default)]
pub struct LongitudinalVariableSummary {
    pub complete_episode_denominator: usize,
    pub observed_event_denominator: usize,
    pub complete_episode_counts: BTreeMap<String, usize>,
    pub event_count_distribution: BTreeMap<String, usize>,
    pub observed_event_days: BTreeMap<String, usize>,
    pub missing_events: usize,
}
#[derive(Debug, Serialize, Default)]
pub struct LongitudinalPairSummary {
    pub complete_episode_denominator: usize,
    pub both_observed_denominator: usize,
    pub order: BTreeMap<String, usize>,
    pub elapsed_days: BTreeMap<String, usize>,
}

pub fn summarize_longitudinal(
    config: &LongitudinalConfig,
    episodes: &[LongitudinalEpisode],
) -> LongitudinalSummary {
    let mut summary = LongitudinalSummary::default();
    for name in &config.variables {
        summary
            .variables
            .insert(name.clone(), LongitudinalVariableSummary::default());
    }
    for name in config.pairs.keys() {
        summary
            .pairs
            .insert(name.clone(), LongitudinalPairSummary::default());
    }
    for episode in episodes {
        *summary
            .coverage
            .entry(episode.coverage.clone())
            .or_default() += 1;
        for (name, events) in &episode.variables {
            let variable = summary
                .variables
                .get_mut(name)
                .expect("configured variable");
            variable.missing_events += events.iter().filter(|e| e.missing).count();
            let observed: Vec<_> = events.iter().filter(|e| !e.missing).collect();
            for event in &observed {
                variable.observed_event_denominator += 1;
                *variable
                    .observed_event_days
                    .entry(event.day.to_string())
                    .or_default() += 1;
            }
            if episode.coverage == "complete" {
                variable.complete_episode_denominator += 1;
                *variable
                    .event_count_distribution
                    .entry(observed.len().to_string())
                    .or_default() += 1;
                let category = match observed.len() {
                    0 => "zero",
                    1 => "one",
                    _ => "multiple",
                };
                *variable
                    .complete_episode_counts
                    .entry(category.into())
                    .or_default() += 1;
            }
        }
        if episode.coverage != "complete" {
            continue;
        }
        for (name, pair) in &config.pairs {
            summary
                .pairs
                .get_mut(name)
                .expect("configured pair")
                .complete_episode_denominator += 1;
            let first = episode.variables[&pair.exposure]
                .iter()
                .find(|e| !e.missing);
            let second = episode.variables[&pair.outcome].iter().find(|e| !e.missing);
            if let (Some(a), Some(b)) = (first, second) {
                let pair_summary = summary.pairs.get_mut(name).expect("configured pair");
                pair_summary.both_observed_denominator += 1;
                let order = match a.day.cmp(&b.day) {
                    std::cmp::Ordering::Less => "first_before_second",
                    std::cmp::Ordering::Equal => "same_day",
                    std::cmp::Ordering::Greater => "second_before_first",
                };
                *pair_summary.order.entry(order.into()).or_default() += 1;
                *pair_summary
                    .elapsed_days
                    .entry((b.day - a.day).to_string())
                    .or_default() += 1;
            }
        }
    }
    summary
}

pub fn longitudinal_episodes(
    config: &LongitudinalConfig,
    fixture: &Fixture,
    rows: &[CohortRow],
) -> Vec<LongitudinalEpisode> {
    rows.iter()
        .map(|row| {
            let start = i128::from(row.index_day) + i128::from(config.start_day);
            let end = i128::from(row.index_day) + i128::from(config.end_day);
            let coverage = match (row.observation_start_day, row.observation_end_day) {
                (None, _) => "unknown",
                (Some(s), Some(e)) if i128::from(s) <= start && i128::from(e) >= end => "complete",
                _ => "incomplete",
            };
            let mut variables = BTreeMap::new();
            for name in &config.variables {
                let mut events: Vec<_> = fixture
                    .events
                    .iter()
                    .filter(|e| {
                        e.patient_id == row.patient_id
                            && e.episode_id == row.episode_id
                            && e.variable == *name
                            && i128::from(e.event_day) >= start
                            && i128::from(e.event_day) <= end
                            && i128::from(e.recording_day) <= end
                            && row
                                .observation_start_day
                                .is_none_or(|day| e.event_day >= day)
                            && row
                                .observation_end_day
                                .is_none_or(|day| e.event_day <= day && e.recording_day <= day)
                    })
                    .map(|e| LongitudinalEvent {
                        source_event_id: e.id.clone(),
                        provenance: e.provenance.clone(),
                        day: i128::from(e.event_day) - i128::from(row.index_day),
                        missing: e.missing.is_some(),
                        missing_reason: e.missing.clone(),
                        recording_day: i128::from(e.recording_day) - i128::from(row.index_day),
                    })
                    .collect();
                events
                    .sort_by(|a, b| (a.day, &a.source_event_id).cmp(&(b.day, &b.source_event_id)));
                variables.insert(name.clone(), events);
            }
            LongitudinalEpisode {
                patient_id: row.patient_id.clone(),
                episode_id: row.episode_id.clone(),
                coverage: coverage.into(),
                variables,
            }
        })
        .collect()
}
#[derive(Debug, Deserialize, Serialize, Clone)]
#[serde(deny_unknown_fields)]
pub struct JointSelection {
    pub exposure: String,
    pub outcome: String,
}
#[derive(Debug, Serialize)]
pub struct EvaluationReport {
    pub schema_version: String,
    pub synthetic: EvaluationCohort,
    pub reference: EvaluationCohort,
    pub variables: BTreeMap<String, VariableComparison>,
    pub joint: Option<JointComparison>,
    #[serde(skip_serializing_if = "BTreeMap::is_empty")]
    pub pairs: BTreeMap<String, JointComparison>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temporal: Option<TemporalComparison>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub outcome_timing: Option<OutcomeTimingComparison>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub longitudinal: Option<LongitudinalComparison>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub uncertainty: Option<UncertaintySummary>,
    #[serde(skip_serializing_if = "BTreeMap::is_empty")]
    pub continuous: BTreeMap<String, ContinuousComparison>,
    pub limitations: Vec<String>,
}
#[derive(Debug, Serialize, Clone)]
pub struct UncertaintySummary {
    pub configuration: UncertaintyConfig,
    pub method: String,
    pub limitations: String,
}
#[derive(Debug, Serialize)]
pub struct BootstrapInterval {
    pub lower: Option<f64>,
    pub upper: Option<f64>,
    pub valid_replicates: usize,
    pub unavailable_reason: Option<String>,
}
#[derive(Debug, Serialize)]
pub struct ComparisonUncertainty {
    pub signed_difference: Option<f64>,
    pub synthetic: BootstrapInterval,
    pub reference: BootstrapInterval,
    pub signed_difference_interval: BootstrapInterval,
}
#[derive(Debug, Serialize)]
pub struct LongitudinalComparison {
    pub synthetic_episodes: Vec<LongitudinalEpisode>,
    pub reference_episodes: Vec<LongitudinalEpisode>,
    pub synthetic: LongitudinalSummary,
    pub reference: LongitudinalSummary,
    pub coverage: BTreeMap<String, CategoryComparison>,
    pub variable_counts: BTreeMap<String, BTreeMap<String, CategoryComparison>>,
    pub event_count_distributions: BTreeMap<String, BTreeMap<String, CategoryComparison>>,
    pub variable_days: BTreeMap<String, BTreeMap<String, CategoryComparison>>,
    pub pair_order: BTreeMap<String, BTreeMap<String, CategoryComparison>>,
    pub pair_elapsed_days: BTreeMap<String, BTreeMap<String, CategoryComparison>>,
}
#[derive(Debug, Serialize)]
pub struct OutcomeTimingComparison {
    pub synthetic: TimingCounts,
    pub reference: TimingCounts,
    pub cells: Vec<TimingCellComparison>,
}
#[derive(Debug, Serialize, Default)]
pub struct TimingCounts {
    pub observed: usize,
    pub missing: usize,
    pub absent: usize,
    pub incomplete_coverage: usize,
    pub unknown_coverage: usize,
    pub outside_bins: usize,
}
#[derive(Debug, Serialize)]
pub struct TimingCellComparison {
    pub value: String,
    pub bin: OutcomeTimeBin,
    pub comparison: CategoryComparison,
}
#[derive(Debug, Serialize)]
pub struct EvaluationCohort {
    pub role: String,
    pub included_episodes: usize,
    pub excluded_episodes: usize,
    pub exclusion_reasons: BTreeMap<String, usize>,
}
#[derive(Debug, Serialize)]
pub struct Frequency {
    pub count: usize,
    pub proportion: Option<f64>,
}
#[derive(Debug, Serialize)]
pub struct CategoryComparison {
    pub synthetic: Frequency,
    pub reference: Frequency,
    pub absolute_difference: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub uncertainty: Option<ComparisonUncertainty>,
}
#[derive(Debug, Serialize)]
pub struct VariableComparison {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<VariableRole>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub synthetic_source_event_ids: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reference_source_event_ids: Option<Vec<String>>,
    pub synthetic_counts: ValueCounts,
    pub reference_counts: ValueCounts,
    pub categories: BTreeMap<String, CategoryComparison>,
}
#[derive(Debug, Serialize)]
pub struct JointCategory {
    pub exposure: String,
    pub outcome: String,
    pub comparison: CategoryComparison,
}
#[derive(Debug, Serialize)]
pub struct JointComparison {
    pub selection: JointSelection,
    pub synthetic_denominator: usize,
    pub reference_denominator: usize,
    pub categories: Vec<JointCategory>,
}
#[derive(Debug, Serialize)]
pub struct EvaluationManifest {
    pub schema_version: String,
    pub configuration: EvaluationConfig,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub uncertainty: Option<UncertaintySummary>,
    pub artifacts: Vec<String>,
    pub software_version: String,
    pub reproducibility_limits: String,
    pub study_name: String,
    pub study_schema_version: String,
    pub study_fingerprint: String,
    pub synthetic_fingerprint: String,
    pub reference_fingerprint: String,
    pub metric_definitions: Vec<String>,
    pub unperformed_checks: Vec<String>,
    pub limitations: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub holdout_provenance: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub evidence: Option<EvidenceProvenance>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub software: Option<SoftwareIdentity>,
    pub configuration_fingerprint: String,
}

#[derive(Debug, Serialize)]
pub struct Distribution {
    pub denominator: usize,
    pub counts: BTreeMap<String, usize>,
}
impl Distribution {
    fn new() -> Self {
        Self {
            denominator: 0,
            counts: BTreeMap::new(),
        }
    }
    fn add(&mut self, value: i128) {
        self.denominator += 1;
        *self.counts.entry(value.to_string()).or_default() += 1;
    }
}
#[derive(Debug, Serialize)]
pub struct TemporalVariable {
    pub event_day: Distribution,
    pub recording_day: Distribution,
    pub recording_delay_days: Distribution,
    pub index_to_event_days: Distribution,
    pub windows: BTreeMap<String, WindowCounts>,
}
#[derive(Debug, Serialize, Default)]
pub struct WindowCounts {
    pub denominator_episodes: usize,
    pub observed: usize,
    pub missing: usize,
    pub absent: usize,
}
#[derive(Debug, Serialize)]
pub struct TemporalCohort {
    pub variables: BTreeMap<String, TemporalVariable>,
    pub episodes_per_patient: Distribution,
    pub episode_event_span_days: Distribution,
    pub last_observed_follow_up_day: Distribution,
    pub indexed_episodes_without_observed_follow_up: usize,
    pub indexed_episode_denominator: usize,
    pub recorded_before_event_count: usize,
    pub temporal_order_event_denominator: usize,
    pub observation_complete_episodes: usize,
    pub observation_incomplete_episodes: usize,
    pub observation_unknown_episodes: usize,
    pub observed_duration_days: Distribution,
    pub censoring_reasons: BTreeMap<String, usize>,
    pub events_outside_observation: usize,
}
#[derive(Debug, Serialize)]
pub struct TemporalComparison {
    pub synthetic: TemporalCohort,
    pub reference: TemporalCohort,
}

fn temporal_cohort(study: &Study, fixture: &Fixture, config: &TemporalConfig) -> TemporalCohort {
    let mut groups = BTreeMap::<(&str, &str), Vec<&Event>>::new();
    let mut patients = BTreeMap::<&str, usize>::new();
    for event in &fixture.events {
        groups
            .entry((&event.patient_id, &event.episode_id))
            .or_default()
            .push(event);
    }
    for (patient, _) in groups.keys() {
        *patients.entry(patient).or_default() += 1;
    }
    let mut episodes_per_patient = Distribution::new();
    for count in patients.values() {
        episodes_per_patient.add(*count as i128);
    }
    let mut variables = BTreeMap::new();
    for variable in &study.variables {
        variables.insert(
            variable.name.clone(),
            TemporalVariable {
                event_day: Distribution::new(),
                recording_day: Distribution::new(),
                recording_delay_days: Distribution::new(),
                index_to_event_days: Distribution::new(),
                windows: config
                    .windows
                    .iter()
                    .map(|w| (w.name.clone(), WindowCounts::default()))
                    .collect(),
            },
        );
    }
    let mut span = Distribution::new();
    let mut last = Distribution::new();
    let mut indexed = 0;
    let mut without_follow_up = 0;
    let mut complete = 0;
    let mut incomplete = 0;
    let mut unknown = 0;
    let mut duration = Distribution::new();
    let mut censoring_reasons = BTreeMap::new();
    let mut outside = 0;
    for events in groups.values() {
        let min = events.iter().map(|e| e.event_day).min().unwrap();
        let max = events.iter().map(|e| e.event_day).max().unwrap();
        span.add(i128::from(max) - i128::from(min));
        let index = events.iter().find(|e| e.variable == study.index);
        if index.is_some() {
            indexed += 1;
        }
        let mut follow_up = None;
        for event in events {
            let v = variables
                .get_mut(&event.variable)
                .expect("validated variable");
            v.event_day.add(i128::from(event.event_day));
            v.recording_day.add(i128::from(event.recording_day));
            v.recording_delay_days
                .add(i128::from(event.recording_day) - i128::from(event.event_day));
            if let Some(ix) = index {
                let day = i128::from(event.event_day) - i128::from(ix.event_day);
                v.index_to_event_days.add(day);
                if event.id != ix.id
                    && event.value.is_some()
                    && day >= 0
                    && day <= i128::from(study.follow_up_days)
                    && i128::from(event.recording_day)
                        <= i128::from(ix.event_day) + i128::from(study.follow_up_days)
                {
                    follow_up = Some(follow_up.map_or(day, |previous: i128| previous.max(day)));
                }
            }
        }
        if let Some(ix) = index {
            let observation = fixture.observations.iter().find(|o| {
                o.patient_id == events[0].patient_id && o.episode_id == events[0].episode_id
            });
            if let Some(o) = observation {
                if o.end_day.is_some_and(|end| {
                    i128::from(end) >= i128::from(ix.event_day) + i128::from(study.follow_up_days)
                }) {
                    complete += 1;
                } else {
                    incomplete += 1;
                }
                let elapsed = o.end_day.map_or(i128::from(study.follow_up_days), |end| {
                    (i128::from(end) - i128::from(ix.event_day))
                        .min(i128::from(study.follow_up_days))
                });
                duration.add(elapsed);
                if let Some(reason) = &o.censoring_reason {
                    *censoring_reasons.entry(reason.clone()).or_default() += 1;
                }
                outside += events
                    .iter()
                    .filter(|e| {
                        e.event_day < o.start_day || o.end_day.is_some_and(|end| e.event_day > end)
                    })
                    .count();
            } else {
                unknown += 1;
            }
            if let Some(day) = follow_up {
                last.add(day);
            } else {
                without_follow_up += 1;
            }
            for variable in &study.variables {
                for window in &config.windows {
                    let selected = events
                        .iter()
                        .filter(|e| {
                            e.variable == variable.name && {
                                let event_day = i128::from(e.event_day) - i128::from(ix.event_day);
                                let recorded =
                                    i128::from(e.recording_day) - i128::from(ix.event_day);
                                let observation_ok = fixture
                                    .observations
                                    .iter()
                                    .find(|o| {
                                        o.patient_id == e.patient_id && o.episode_id == e.episode_id
                                    })
                                    .is_none_or(|o| {
                                        e.event_day >= o.start_day
                                            && o.end_day.is_none_or(|end| e.event_day <= end)
                                            && e.recording_day <= o.end_day.unwrap_or(i64::MAX)
                                    });
                                observation_ok
                                    && event_day >= i128::from(window.start_day)
                                    && event_day <= i128::from(window.end_day)
                                    && recorded <= i128::from(window.end_day)
                            }
                        })
                        .max_by_key(|e| (e.event_day, e.recording_day, &e.id));
                    let counts = variables
                        .get_mut(&variable.name)
                        .unwrap()
                        .windows
                        .get_mut(&window.name)
                        .unwrap();
                    counts.denominator_episodes += 1;
                    match selected {
                        Some(e) if e.value.is_some() => counts.observed += 1,
                        Some(_) => counts.missing += 1,
                        None => counts.absent += 1,
                    }
                }
            }
        }
    }
    TemporalCohort {
        variables,
        episodes_per_patient,
        episode_event_span_days: span,
        last_observed_follow_up_day: last,
        indexed_episodes_without_observed_follow_up: without_follow_up,
        indexed_episode_denominator: indexed,
        recorded_before_event_count: fixture
            .events
            .iter()
            .filter(|e| e.recording_day < e.event_day)
            .count(),
        temporal_order_event_denominator: fixture.events.len(),
        observation_complete_episodes: complete,
        observation_incomplete_episodes: incomplete,
        observation_unknown_episodes: unknown,
        observed_duration_days: duration,
        censoring_reasons,
        events_outside_observation: outside,
    }
}

pub fn fixture_fingerprint(fixture: &Fixture) -> Result<String, Error> {
    yaml_fingerprint(fixture)
}

pub fn yaml_fingerprint<T: Serialize>(value: &T) -> Result<String, Error> {
    let bytes = serde_yaml::to_string(value)?;
    let mut hash = 0xcbf29ce484222325u64;
    for byte in bytes.as_bytes() {
        hash ^= *byte as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    Ok(format!("fnv1a64-yaml:{hash:016x}"))
}

fn comparison(a: usize, a_total: usize, b: usize, b_total: usize) -> CategoryComparison {
    let proportion = |count, total| (total != 0).then(|| count as f64 / total as f64);
    let ap = proportion(a, a_total);
    let bp = proportion(b, b_total);
    CategoryComparison {
        synthetic: Frequency {
            count: a,
            proportion: ap,
        },
        reference: Frequency {
            count: b,
            proportion: bp,
        },
        absolute_difference: ap.zip(bp).map(|(a, b)| (a - b).abs()),
        uncertainty: None,
    }
}

fn bootstrap_patients(rows: &[CohortRow]) -> Vec<Vec<&CohortRow>> {
    let mut patients = BTreeMap::<&str, Vec<&CohortRow>>::new();
    for row in rows {
        patients.entry(&row.patient_id).or_default().push(row);
    }
    patients.into_values().collect()
}

fn bootstrap_indices(count: usize, state: &mut u64) -> Vec<usize> {
    if count == 0 {
        return vec![];
    }
    let bin = OutcomeTimeBin {
        start_day: 0,
        end_day: (count - 1) as i64,
    };
    (0..count)
        .map(|_| sample_outcome_day(state, &bin).expect("valid patient count") as usize)
        .collect()
}

fn bootstrap_proportion(
    patients: &[Vec<&CohortRow>],
    indices: &[usize],
    variable: &str,
    value: &str,
    joint: Option<(&str, &str)>,
) -> Option<f64> {
    let mut numerator = 0usize;
    let mut denominator = 0usize;
    for row in indices.iter().flat_map(|index| patients[*index].iter()) {
        if let Some(ValueState::Observed(observed)) = row.values.get(variable) {
            if let Some((other, target)) = joint {
                if let Some(ValueState::Observed(other_value)) = row.values.get(other) {
                    denominator += 1;
                    numerator += usize::from(observed == value && other_value == target);
                }
            } else {
                denominator += 1;
                numerator += usize::from(observed == value);
            }
        }
    }
    (denominator > 0).then(|| numerator as f64 / denominator as f64)
}

fn bootstrap_interval(
    mut values: Vec<f64>,
    config: &UncertaintyConfig,
    reason: &str,
) -> BootstrapInterval {
    let valid_replicates = values.len();
    if valid_replicates != config.replicates {
        return BootstrapInterval {
            lower: None,
            upper: None,
            valid_replicates,
            unavailable_reason: Some(reason.into()),
        };
    }
    values.sort_by(f64::total_cmp);
    let alpha = (1.0 - config.confidence_level) / 2.0;
    let quantile = |p: f64| {
        let position = p * (values.len() - 1) as f64;
        let lo = position.floor() as usize;
        let hi = position.ceil() as usize;
        values[lo] + (values[hi] - values[lo]) * (position - lo as f64)
    };
    BootstrapInterval {
        lower: Some(quantile(alpha)),
        upper: Some(quantile(1.0 - alpha)),
        valid_replicates,
        unavailable_reason: None,
    }
}

fn attach_uncertainty(
    comparison: &mut CategoryComparison,
    config: &UncertaintyConfig,
    a: &[Vec<&CohortRow>],
    b: &[Vec<&CohortRow>],
    category: (&str, &str),
    joint: Option<(&str, &str)>,
) {
    let mut av = Vec::new();
    let mut bv = Vec::new();
    let mut differences = Vec::new();
    let mut state = config.seed;
    for _ in 0..config.replicates {
        let ai = bootstrap_indices(a.len(), &mut state);
        let bi = bootstrap_indices(b.len(), &mut state);
        let ap = bootstrap_proportion(a, &ai, category.0, category.1, joint);
        let bp = bootstrap_proportion(b, &bi, category.0, category.1, joint);
        if let Some(p) = ap {
            av.push(p);
        }
        if let Some(p) = bp {
            bv.push(p);
        }
        if let (Some(x), Some(y)) = (ap, bp) {
            differences.push(x - y);
        }
    }
    let reason = "empty cohort or zero observed denominator in at least one replicate";
    comparison.uncertainty = Some(ComparisonUncertainty {
        signed_difference: comparison
            .synthetic
            .proportion
            .zip(comparison.reference.proportion)
            .map(|(x, y)| x - y),
        synthetic: bootstrap_interval(av, config, reason),
        reference: bootstrap_interval(bv, config, reason),
        signed_difference_interval: bootstrap_interval(differences, config, reason),
    });
}

fn compare_distributions(
    a: &BTreeMap<String, usize>,
    b: &BTreeMap<String, usize>,
    denominator_a: usize,
    denominator_b: usize,
    required: &[&str],
) -> BTreeMap<String, CategoryComparison> {
    a.keys()
        .chain(b.keys())
        .cloned()
        .chain(required.iter().map(|s| (*s).to_owned()))
        .collect::<std::collections::BTreeSet<_>>()
        .into_iter()
        .map(|key| {
            let value = comparison(
                *a.get(&key).unwrap_or(&0),
                denominator_a,
                *b.get(&key).unwrap_or(&0),
                denominator_b,
            );
            (key, value)
        })
        .collect()
}

fn compare_longitudinal(
    config: &LongitudinalConfig,
    synthetic: &Fixture,
    a: &[CohortRow],
    reference: &Fixture,
    b: &[CohortRow],
) -> LongitudinalComparison {
    let synthetic_episodes = longitudinal_episodes(config, synthetic, a);
    let reference_episodes = longitudinal_episodes(config, reference, b);
    let synthetic = summarize_longitudinal(config, &synthetic_episodes);
    let reference = summarize_longitudinal(config, &reference_episodes);
    let coverage = compare_distributions(
        &synthetic.coverage,
        &reference.coverage,
        synthetic_episodes.len(),
        reference_episodes.len(),
        &["complete", "incomplete", "unknown"],
    );
    let mut variable_counts = BTreeMap::new();
    let mut event_count_distributions = BTreeMap::new();
    let mut variable_days = BTreeMap::new();
    let mut pair_order = BTreeMap::new();
    let mut pair_elapsed_days = BTreeMap::new();
    for name in &config.variables {
        let x = &synthetic.variables[name];
        let y = &reference.variables[name];
        variable_counts.insert(
            name.clone(),
            compare_distributions(
                &x.complete_episode_counts,
                &y.complete_episode_counts,
                x.complete_episode_counts.values().sum(),
                y.complete_episode_counts.values().sum(),
                &["zero", "one", "multiple"],
            ),
        );
        event_count_distributions.insert(
            name.clone(),
            compare_distributions(
                &x.event_count_distribution,
                &y.event_count_distribution,
                x.event_count_distribution.values().sum(),
                y.event_count_distribution.values().sum(),
                &[],
            ),
        );
        variable_days.insert(
            name.clone(),
            compare_distributions(
                &x.observed_event_days,
                &y.observed_event_days,
                x.observed_event_days.values().sum(),
                y.observed_event_days.values().sum(),
                &[],
            ),
        );
    }
    for name in config.pairs.keys() {
        let x = &synthetic.pairs[name];
        let y = &reference.pairs[name];
        pair_order.insert(
            name.clone(),
            compare_distributions(
                &x.order,
                &y.order,
                x.order.values().sum(),
                y.order.values().sum(),
                &["first_before_second", "same_day", "second_before_first"],
            ),
        );
        pair_elapsed_days.insert(
            name.clone(),
            compare_distributions(
                &x.elapsed_days,
                &y.elapsed_days,
                x.elapsed_days.values().sum(),
                y.elapsed_days.values().sum(),
                &[],
            ),
        );
    }
    LongitudinalComparison {
        synthetic_episodes,
        reference_episodes,
        synthetic,
        reference,
        coverage,
        variable_counts,
        event_count_distributions,
        variable_days,
        pair_order,
        pair_elapsed_days,
    }
}

fn numeric_values<'a>(
    rows: impl IntoIterator<Item = &'a CohortRow>,
    name: &str,
    kind: &str,
) -> Result<Vec<f64>, Error> {
    let mut out = Vec::new();
    for row in rows {
        if let Some(ValueState::Observed(s)) = row.values.get(name) {
            let v = if kind == "integer" {
                let integer = s
                    .parse::<i64>()
                    .map_err(|_| Error::Input(format!("invalid integer value for {name}")))?;
                integer as f64
            } else {
                s.parse::<f64>()
                    .map_err(|_| Error::Input(format!("invalid numeric value for {name}")))?
            };
            if !v.is_finite() {
                return Err(Error::Input(format!("non-finite numeric value for {name}")));
            }
            out.push(v);
        }
    }
    Ok(out)
}
fn continuous_summary(
    rows: &[CohortRow],
    name: &str,
    kind: &str,
    qs: &[f64],
) -> Result<ContinuousCohortSummary, Error> {
    let values = numeric_values(rows, name, kind)?;
    let mut missing = 0;
    let mut absent = 0;
    let mut patients = BTreeSet::new();
    let mut episodes = 0;
    for row in rows {
        episodes += 1;
        patients.insert(&row.patient_id);
        match row.values.get(name) {
            Some(ValueState::Missing(_)) => missing += 1,
            Some(ValueState::Absent) | None => absent += 1,
            _ => {}
        }
    }
    let mean = (!values.is_empty()).then(|| values.iter().sum::<f64>() / values.len() as f64);
    let sd = (values.len() > 1).then(|| {
        let m = mean.unwrap();
        (values.iter().map(|v| (v - m).powi(2)).sum::<f64>() / (values.len() - 1) as f64).sqrt()
    });
    let mut quantiles = BTreeMap::new();
    let mut sorted = values.clone();
    sorted.sort_by(f64::total_cmp);
    for &q in qs {
        let x = if sorted.is_empty() {
            None
        } else if sorted.len() == 1 {
            Some(sorted[0])
        } else {
            let p = q * (sorted.len() - 1) as f64;
            let i = p.floor() as usize;
            let d = p - i as f64;
            Some(sorted[i] + (sorted[(i + 1).min(sorted.len() - 1)] - sorted[i]) * d)
        };
        quantiles.insert(q.to_string(), x);
    }
    Ok(ContinuousCohortSummary {
        observed: values.len(),
        missing,
        absent,
        invalid: 0,
        patients: patients.len(),
        episodes,
        mean,
        sample_standard_deviation: sd,
        quantiles,
    })
}
fn cdf_distance(a: &[f64], b: &[f64]) -> f64 {
    let mut xs = a.iter().chain(b).copied().collect::<Vec<_>>();
    xs.sort_by(f64::total_cmp);
    xs.dedup();
    xs.into_iter()
        .map(|x| {
            ((a.iter().filter(|v| **v <= x).count() as f64 / a.len() as f64)
                - (b.iter().filter(|v| **v <= x).count() as f64 / b.len() as f64))
                .abs()
        })
        .fold(0.0, f64::max)
}
fn continuous_bootstrap(
    settings: &UncertaintyConfig,
    a: &[Vec<&CohortRow>],
    b: &[Vec<&CohortRow>],
    name: &str,
    kind: &str,
) -> Result<ContinuousBootstrap, Error> {
    let mut state = settings.seed;
    let mut means_a = Vec::new();
    let mut sds_a = Vec::new();
    let mut diffs = Vec::new();
    for _ in 0..settings.replicates {
        let ia = bootstrap_indices(a.len(), &mut state);
        let ib = bootstrap_indices(b.len(), &mut state);
        let ra: Vec<_> = ia.iter().flat_map(|i| a[*i].iter().copied()).collect();
        let rb: Vec<_> = ib.iter().flat_map(|i| b[*i].iter().copied()).collect();
        let va = numeric_values(ra.iter().copied(), name, kind)?;
        let vb = numeric_values(rb.iter().copied(), name, kind)?;
        if va.is_empty() || vb.is_empty() {
            continue;
        }
        let ma = va.iter().sum::<f64>() / va.len() as f64;
        let mb = vb.iter().sum::<f64>() / vb.len() as f64;
        means_a.push(ma);
        diffs.push(ma - mb);
        if va.len() > 1 {
            sds_a.push(
                (va.iter().map(|v| (v - ma).powi(2)).sum::<f64>() / (va.len() - 1) as f64).sqrt(),
            );
        }
    }
    let pct = |mut v: Vec<f64>| -> Option<(f64, f64)> {
        if v.is_empty() {
            return None;
        }
        v.sort_by(f64::total_cmp);
        let lo = ((1.0 - settings.confidence_level) / 2.0 * (v.len() - 1) as f64).round() as usize;
        let hi = ((1.0 - (1.0 - settings.confidence_level) / 2.0) * (v.len() - 1) as f64).round()
            as usize;
        Some((v[lo], v[hi]))
    };
    Ok(ContinuousBootstrap {
        configuration: settings.clone(),
        valid_replicates: means_a.len(),
        mean_interval: pct(means_a),
        sample_standard_deviation_interval: pct(sds_a),
        signed_mean_difference_interval: pct(diffs),
    })
}
fn marginal(rows: &[CohortRow], name: &str) -> (ValueCounts, BTreeMap<String, usize>) {
    let mut counts = ValueCounts::default();
    let mut values = BTreeMap::new();
    for row in rows {
        match &row.values[name] {
            ValueState::Observed(value) => {
                counts.observed += 1;
                *values.entry(value.clone()).or_default() += 1;
            }
            ValueState::Missing(_) => counts.missing += 1,
            ValueState::Absent => counts.absent += 1,
        }
    }
    (counts, values)
}

type JointCounts = BTreeMap<(String, String), usize>;
type TimingCells = BTreeMap<(String, usize), usize>;
fn timing_counts(
    study: &Study,
    timing: &OutcomeTimingConfig,
    fixture: &Fixture,
    rows: &[CohortRow],
) -> Result<(TimingCounts, TimingCells), Error> {
    let mut counts = TimingCounts::default();
    let mut cells = BTreeMap::new();
    for row in rows {
        match outcome_timing_state(study, fixture, row, &timing.variable)? {
            OutcomeTimingState::UnknownCoverage => counts.unknown_coverage += 1,
            OutcomeTimingState::IncompleteCoverage => counts.incomplete_coverage += 1,
            OutcomeTimingState::Missing => counts.missing += 1,
            OutcomeTimingState::Absent => counts.absent += 1,
            OutcomeTimingState::Observed { value, day } => {
                if let Some(bin) = timing
                    .bins
                    .iter()
                    .position(|b| day >= b.start_day && day <= b.end_day)
                {
                    counts.observed += 1;
                    *cells.entry((value, bin)).or_default() += 1;
                } else {
                    counts.outside_bins += 1;
                }
            }
        }
    }
    Ok((counts, cells))
}
fn compare_outcome_timing(
    study: &Study,
    timing: &OutcomeTimingConfig,
    synthetic: &Fixture,
    a: &[CohortRow],
    reference: &Fixture,
    b: &[CohortRow],
) -> Result<OutcomeTimingComparison, Error> {
    let (ac, av) = timing_counts(study, timing, synthetic, a)?;
    let (bc, bv) = timing_counts(study, timing, reference, b)?;
    let values: std::collections::BTreeSet<_> =
        av.keys().chain(bv.keys()).map(|(v, _)| v.clone()).collect();
    let mut cells = Vec::new();
    for value in values {
        for (i, bin) in timing.bins.iter().enumerate() {
            let key = (value.clone(), i);
            cells.push(TimingCellComparison {
                value: value.clone(),
                bin: bin.clone(),
                comparison: comparison(
                    *av.get(&key).unwrap_or(&0),
                    ac.observed,
                    *bv.get(&key).unwrap_or(&0),
                    bc.observed,
                ),
            });
        }
    }
    Ok(OutcomeTimingComparison {
        synthetic: ac,
        reference: bc,
        cells,
    })
}
fn joint_counts(rows: &[CohortRow], selection: &JointSelection) -> (usize, JointCounts) {
    let mut counts = BTreeMap::new();
    let mut denominator = 0;
    for row in rows {
        if let (ValueState::Observed(a), ValueState::Observed(b)) = (
            &row.values[&selection.exposure],
            &row.values[&selection.outcome],
        ) {
            denominator += 1;
            *counts.entry((a.clone(), b.clone())).or_default() += 1;
        }
    }
    (denominator, counts)
}

/// Compare cohorts constructed from separately supplied fixtures using the same study.
pub fn evaluate(
    study: &Study,
    synthetic: &Fixture,
    reference: &Fixture,
    config: &EvaluationConfig,
) -> Result<EvaluationReport, Error> {
    if let Some(uncertainty) = &config.uncertainty {
        if !(1..=100_000).contains(&uncertainty.replicates)
            || !uncertainty.confidence_level.is_finite()
            || uncertainty.confidence_level <= 0.0
            || uncertainty.confidence_level >= 1.0
        {
            return Err(Error::Input("uncertainty requires 1 to 100000 replicates and a confidence level strictly between 0 and 1".into()));
        }
    }
    if !["1", "2", "3", "4"].contains(&config.schema_version.as_str()) {
        return Err(Error::Input("unsupported evaluation version".into()));
    }
    if let Some(longitudinal) = &config.longitudinal {
        if config.schema_version != "4" {
            return Err(Error::Input(
                "longitudinal evaluation requires version 4".into(),
            ));
        }
        longitudinal.validate(study)?;
    } else if config.schema_version == "4" {
        return Err(Error::Input(
            "evaluation version 4 requires longitudinal configuration".into(),
        ));
    }
    if let Some(timing) = &config.outcome_timing {
        if config.schema_version != "3"
            || study.schema_version != "2"
            || variable_role(study, &timing.variable) != Some(VariableRole::FollowUpOutcome)
        {
            return Err(Error::Input(
                "outcome timing requires evaluation version 3 and a follow_up_outcome".into(),
            ));
        }
        OutcomeTimeBin::validate_ordered(&timing.bins, study.follow_up_days)?;
    } else if config.schema_version == "3" {
        return Err(Error::Input(
            "evaluation version 3 requires outcome_timing".into(),
        ));
    }
    if config.schema_version == "1" && !config.pairs.is_empty() {
        return Err(Error::Input(
            "named pairs require evaluation schema version 2".into(),
        ));
    }
    if let Some(continuous) = &config.continuous {
        let mut selected = HashSet::new();
        if continuous.variables.is_empty()
            || continuous.variables.iter().any(|n| {
                !selected.insert(n)
                    || !study
                        .variables
                        .iter()
                        .any(|v| &v.name == n && matches!(v.kind.as_str(), "integer" | "number"))
            })
        {
            return Err(Error::Input(
                "continuous variables must be unique declared integer or number variables".into(),
            ));
        }
        if continuous
            .quantiles
            .iter()
            .any(|q| !q.is_finite() || !(0.0..=1.0).contains(q))
        {
            return Err(Error::Input(
                "continuous quantiles must be finite values in [0, 1]".into(),
            ));
        }
    }
    if let Some(temporal) = &config.temporal {
        let mut names = HashSet::new();
        if temporal.windows.is_empty()
            || temporal.windows.iter().any(|w| {
                w.name.trim().is_empty() || w.start_day > w.end_day || !names.insert(&w.name)
            })
        {
            return Err(Error::Input(
                "temporal windows must have unique nonempty names and ordered bounds".into(),
            ));
        }
    }
    if config.pairs.keys().any(|name| name.trim().is_empty()) {
        return Err(Error::Input("pair names must be nonempty".into()));
    }
    for joint in config.joint.iter().chain(config.pairs.values()) {
        if joint.exposure == joint.outcome
            || [&joint.exposure, &joint.outcome].iter().any(|name| {
                !study.variables.iter().any(|v| {
                    &v.name == *name
                        && (matches!(v.kind.as_str(), "string" | "boolean")
                            || (config.schema_version == "2" && v.kind == "integer"))
                })
            })
        {
            return Err(Error::Input("joint selection requires two distinct categorical (string, boolean or integer) study variables".into()));
        }
    }
    let a = run(study, synthetic)?;
    let b = run(study, reference)?;
    let mut variables = BTreeMap::new();
    for variable in &study.variables {
        let (ac, mut av) = marginal(&a.cohort, &variable.name);
        let (bc, bv) = marginal(&b.cohort, &variable.name);
        if variable.kind == "boolean" {
            av.entry("false".into()).or_default();
            av.entry("true".into()).or_default();
        }
        let categories = av
            .keys()
            .chain(bv.keys())
            .map(|key| {
                (
                    key.clone(),
                    comparison(
                        *av.get(key).unwrap_or(&0),
                        ac.observed,
                        *bv.get(key).unwrap_or(&0),
                        bc.observed,
                    ),
                )
            })
            .collect();
        variables.insert(
            variable.name.clone(),
            VariableComparison {
                role: variable.role,
                synthetic_source_event_ids: variable.role.map(|_| {
                    a.cohort
                        .iter()
                        .filter_map(|r| {
                            r.variable_source_event_ids
                                .get(&variable.name)
                                .and_then(Clone::clone)
                        })
                        .collect()
                }),
                reference_source_event_ids: variable.role.map(|_| {
                    b.cohort
                        .iter()
                        .filter_map(|r| {
                            r.variable_source_event_ids
                                .get(&variable.name)
                                .and_then(Clone::clone)
                        })
                        .collect()
                }),
                synthetic_counts: ac,
                reference_counts: bc,
                categories,
            },
        );
    }
    let compare_joint = |selection: &JointSelection| {
        let (ad, av) = joint_counts(&a.cohort, selection);
        let (bd, bv) = joint_counts(&b.cohort, selection);
        let exposures = variables[&selection.exposure].categories.keys();
        let outcomes: Vec<_> = variables[&selection.outcome]
            .categories
            .keys()
            .cloned()
            .collect();
        let mut categories = Vec::new();
        for exposure in exposures {
            for outcome in &outcomes {
                let key = (exposure.clone(), outcome.clone());
                categories.push(JointCategory {
                    exposure: exposure.clone(),
                    outcome: outcome.clone(),
                    comparison: comparison(
                        *av.get(&key).unwrap_or(&0),
                        ad,
                        *bv.get(&key).unwrap_or(&0),
                        bd,
                    ),
                });
            }
        }
        JointComparison {
            selection: selection.clone(),
            synthetic_denominator: ad,
            reference_denominator: bd,
            categories,
        }
    };
    let mut joint = config.joint.as_ref().map(compare_joint);
    let mut pairs: BTreeMap<String, JointComparison> = config
        .pairs
        .iter()
        .map(|(name, selection)| (name.clone(), compare_joint(selection)))
        .collect();
    let mut continuous_results = BTreeMap::new();
    if let Some(settings) = &config.continuous {
        for name in &settings.variables {
            let kind = study
                .variables
                .iter()
                .find(|v| &v.name == name)
                .unwrap()
                .kind
                .as_str();
            let av = continuous_summary(&a.cohort, name, kind, &settings.quantiles)?;
            let bv = continuous_summary(&b.cohort, name, kind, &settings.quantiles)?;
            let ax = numeric_values(a.cohort.iter(), name, kind)?;
            let bx = numeric_values(b.cohort.iter(), name, kind)?;
            let bootstrap = config
                .uncertainty
                .as_ref()
                .map(|u| {
                    continuous_bootstrap(
                        u,
                        &bootstrap_patients(&a.cohort),
                        &bootstrap_patients(&b.cohort),
                        name,
                        kind,
                    )
                })
                .transpose()?;
            continuous_results.insert(name.clone(), ContinuousComparison { synthetic: av, reference: bv, empirical_cdf_distance: if ax.is_empty() || bx.is_empty() { None } else { Some(cdf_distance(&ax, &bx)) }, denominator: "Observed numeric values among included cohort episodes; patient and episode counts describe those episodes".into(), bootstrap });
        }
    }
    let uncertainty = config.uncertainty.as_ref().map(|settings| {
        let ap = bootstrap_patients(&a.cohort);
        let bp = bootstrap_patients(&b.cohort);
        for (name, variable) in &mut variables {
            for (value, comparison) in &mut variable.categories {
                attach_uncertainty(comparison, settings, &ap, &bp, (name, value), None);
            }
        }
        for pair in joint.iter_mut().chain(pairs.values_mut()) {
            for cell in &mut pair.categories {
                attach_uncertainty(&mut cell.comparison, settings, &ap, &bp, (&pair.selection.exposure, &cell.exposure), Some((&pair.selection.outcome, &cell.outcome)));
            }
        }
        UncertaintySummary { configuration: settings.clone(), method: "Independent patient-level percentile bootstrap; sorted patient IDs; SplitMix64 with rejection sampling; linear interpolation of quantiles".into(), limitations: "Conditional sampling variability only; small samples may produce unstable intervals. No independence, clinical, privacy, causal or external-validity guarantee. Timing and longitudinal metrics have no intervals. Repeatability depends on software and environment.".into() }
    });
    let cohort = |role: &str, summary: CohortSummary| EvaluationCohort {
        role: role.into(),
        included_episodes: summary.included_rows,
        excluded_episodes: summary.excluded_episodes,
        exclusion_reasons: summary.exclusion_reasons,
    };
    let temporal = config.temporal.as_ref().map(|config| TemporalComparison {
        synthetic: temporal_cohort(study, synthetic, config),
        reference: temporal_cohort(study, reference, config),
    });
    let outcome_timing = config
        .outcome_timing
        .as_ref()
        .map(|timing| {
            compare_outcome_timing(study, timing, synthetic, &a.cohort, reference, &b.cohort)
        })
        .transpose()?;
    let longitudinal = config
        .longitudinal
        .as_ref()
        .map(|c| compare_longitudinal(c, synthetic, &a.cohort, reference, &b.cohort));
    let metric_limit = if uncertainty.is_some() {
        "Descriptive and temporal similarity do not establish clinical plausibility, privacy, causal, forensic, or external validity; intervals cover categorical comparisons only and no continuous metrics are computed."
    } else {
        "Descriptive and temporal similarity do not establish clinical plausibility, privacy, causal, forensic, or external validity; no uncertainty intervals or continuous metrics are computed."
    };
    Ok(EvaluationReport { schema_version: config.schema_version.clone(), synthetic: cohort("synthetic", a.summary), reference: cohort("caller-supplied held-out reference", b.summary), variables, joint, pairs, temporal, outcome_timing, longitudinal, uncertainty, continuous: continuous_results, limitations: vec!["Reference independence and upstream leakage cannot be verified by Janus.".into(), metric_limit.into(), "Continuous comparisons are conditional descriptive comparisons; they do not establish clinical validity, causal effects, privacy, independence, or external validity.".into(), "Empirical fitting does not model trajectories or repeated episodes.".into(), "Observation coverage and censoring metadata are descriptive and do not establish censoring mechanism or causal validity.".into(), "Validation rejects recording before event, so invalid inputs produce no evaluation report.".into()] })
}
#[derive(Debug, Serialize, Default, PartialEq, Eq)]
pub struct ValueCounts {
    pub observed: usize,
    pub missing: usize,
    pub absent: usize,
}
#[derive(Debug, Serialize, Default, PartialEq, Eq)]
pub struct CohortSummary {
    pub source_events: usize,
    pub distinct_patients: usize,
    pub distinct_episodes: usize,
    pub episodes_with_index: usize,
    pub included_rows: usize,
    pub excluded_episodes: usize,
    pub exclusion_reasons: BTreeMap<String, usize>,
    pub derived_values: ValueCounts,
}
#[derive(Debug, Serialize)]
pub struct Diagnostic {
    pub metric: String,
    pub numerator: usize,
    pub denominator: usize,
    pub status: String,
    pub limitation: String,
}

fn metric(
    name: String,
    numerator: usize,
    denominator: usize,
    anomaly: bool,
    limitation: &str,
) -> Diagnostic {
    Diagnostic {
        metric: name,
        numerator,
        denominator,
        status: if denominator == 0 {
            "not_available"
        } else if anomaly && numerator > 0 {
            "detected"
        } else {
            "descriptive"
        }
        .into(),
        limitation: limitation.into(),
    }
}

/// Describes fixture construction only; callers can inspect invalid inputs before validation.
pub fn diagnostics(
    study: &Study,
    fixture: &Fixture,
    cohort: &[CohortRow],
    exclusions: &[Exclusion],
) -> Vec<Diagnostic> {
    let events = &fixture.events;
    let indexes: BTreeMap<_, _> = events
        .iter()
        .filter(|e| e.variable == study.index)
        .map(|e| ((e.patient_id.as_str(), e.episode_id.as_str()), e))
        .collect();
    let ids: BTreeMap<_, _> = events.iter().map(|e| (e.id.as_str(), e)).collect();
    let mut episodes = BTreeMap::<&str, HashSet<&str>>::new();
    for e in events {
        episodes
            .entry(&e.patient_id)
            .or_default()
            .insert(&e.episode_id);
    }
    let mut out = vec![
        metric("recorded_before_event".into(), events.iter().filter(|e| e.recording_day < e.event_day).count(), events.len(), true, "Input validation rejects these events."),
        metric("outside_follow_up".into(), events.iter().filter(|e| indexes.get(&(e.patient_id.as_str(), e.episode_id.as_str())).is_some_and(|ix| ix.event_day.checked_add(study.follow_up_days).is_some_and(|end| e.event_day < ix.event_day || e.event_day > end))).count(), events.iter().filter(|e| indexes.contains_key(&(e.patient_id.as_str(), e.episode_id.as_str()))).count(), false, "Events before index can supply baseline eligibility; only episodes with an index are counted."),
        metric("missing_provenance".into(), events.iter().filter(|e| e.provenance.trim().is_empty()).count(), events.len(), true, "Input validation rejects empty provenance; content is not independently verified."),
        metric("index_coverage".into(), indexes.len(), episodes.values().map(HashSet::len).sum(), false, "Episodes are identified by patient and episode identifiers."),
        metric("eligibility_exclusions".into(), exclusions.iter().filter(|e| e.reason.starts_with("eligibility:")).count(), indexes.len(), false, "Only the first failed equality rule is retained."),
    ];
    let refs: Vec<_> = cohort
        .iter()
        .flat_map(|r| {
            r.source_event_ids
                .iter()
                .map(move |id| (r.patient_id.as_str(), r.episode_id.as_str(), id.as_str()))
        })
        .chain(cohort.iter().flat_map(|r| {
            r.eligibility_values.values().filter_map(move |value| {
                value
                    .source_event_id
                    .as_deref()
                    .map(|id| (r.patient_id.as_str(), r.episode_id.as_str(), id))
            })
        }))
        .chain(exclusions.iter().flat_map(|r| {
            r.source_event_ids
                .iter()
                .map(move |id| (r.patient_id.as_str(), r.episode_id.as_str(), id.as_str()))
        }))
        .collect();
    out.push(metric(
        "unresolved_source_references".into(),
        refs.iter()
            .filter(|(_, _, id)| !ids.contains_key(id))
            .count(),
        refs.len(),
        true,
        "Only fixture event identifiers are resolvable.",
    ));
    out.push(metric(
        "inconsistent_source_provenance".into(),
        refs.iter()
            .filter(|(p, ep, id)| {
                ids.get(id).is_some_and(|e| {
                    e.patient_id != *p || e.episode_id != *ep || e.provenance.trim().is_empty()
                })
            })
            .count(),
        refs.len(),
        true,
        "Checks episode identity and nonempty provenance, not source authenticity.",
    ));
    let candidates: Vec<_> = events
        .iter()
        .filter(|e| {
            study.eligibility.iter().any(|r| r.variable == e.variable)
                && indexes.contains_key(&(e.patient_id.as_str(), e.episode_id.as_str()))
        })
        .collect();
    let late = candidates
        .iter()
        .filter(|e| {
            let ix = indexes[&(e.patient_id.as_str(), e.episode_id.as_str())];
            e.event_day <= ix.event_day && e.recording_day > ix.event_day
        })
        .count();
    out.push(metric("eligibility_late_recorded_candidates".into(), late, candidates.len(), true, "Potential future information excluded from index-time eligibility; does not indicate actual leakage."));
    let inconsistent = cohort
        .iter()
        .filter(|row| {
            study.eligibility.iter().any(|rule| {
                let baseline = events
                    .iter()
                    .filter(|e| {
                        e.patient_id == row.patient_id
                            && e.episode_id == row.episode_id
                            && e.variable == rule.variable
                            && e.event_day <= row.index_day
                            && e.recording_day <= row.index_day
                    })
                    .max_by_key(|e| (e.event_day, e.recording_day));
                !matches!(baseline.and_then(|e| e.value.as_ref()), Some(v) if v == &rule.equals)
            })
        })
        .count();
    out.push(metric("eligibility_future_information_leakage".into(), inconsistent, cohort.len(), true, "Included episodes whose eligibility cannot be established using index-time information; does not identify the cause."));
    for v in &study.variables {
        for (name, count) in [
            (
                "missing",
                cohort
                    .iter()
                    .filter(|r| matches!(r.values.get(&v.name), Some(ValueState::Missing(_))))
                    .count(),
            ),
            (
                "absent",
                cohort
                    .iter()
                    .filter(|r| matches!(r.values.get(&v.name), Some(ValueState::Absent) | None))
                    .count(),
            ),
        ] {
            out.push(metric(
                format!("{name}:{}", v.name),
                count,
                cohort.len(),
                false,
                "Included episodes only; absence does not imply an event did not occur.",
            ));
        }
    }
    let mut distribution = BTreeMap::<usize, usize>::new();
    for eps in episodes.values() {
        *distribution.entry(eps.len()).or_default() += 1;
    }
    for (n, count) in distribution {
        out.push(metric(
            format!("patients_with_{n}_episodes"),
            count,
            episodes.len(),
            false,
            "Counts source episodes per distinct patient.",
        ));
    }
    out
}
pub fn load<T: for<'a> Deserialize<'a>>(p: &Path) -> Result<T, Error> {
    Ok(serde_yaml::from_str(&fs::read_to_string(p)?)?)
}
fn valid_value(kind: &str, value: &str) -> bool {
    match kind {
        "string" => true,
        "boolean" => value.parse::<bool>().is_ok(),
        "integer" => value.parse::<i64>().is_ok(),
        "number" => value.parse::<f64>().is_ok_and(f64::is_finite),
        _ => false,
    }
}
fn event_value(event: &Event) -> ValueState {
    match (&event.value, &event.missing) {
        (Some(value), _) => ValueState::Observed(value.clone()),
        (_, Some(reason)) => ValueState::Missing(reason.clone()),
        _ => ValueState::Absent,
    }
}
fn variable_role(study: &Study, name: &str) -> Option<VariableRole> {
    study
        .variables
        .iter()
        .find(|v| v.name == name)
        .and_then(|v| v.role)
}
pub fn validate(study: &Study, fixture: &Fixture) -> Result<(), Error> {
    if !["1", "2"].contains(&study.schema_version.as_str())
        || !["1", "2"].contains(&fixture.schema_version.as_str())
    {
        return Err(Error::Input(
            "unsupported study or fixture schema_version".into(),
        ));
    }
    if study.unit != "episode" {
        return Err(Error::Input("unit must be episode".into()));
    }
    if study.follow_up_days < 0 {
        return Err(Error::Input("follow_up_days must be nonnegative".into()));
    }
    let vars: HashSet<_> = study.variables.iter().map(|v| v.name.as_str()).collect();
    if vars.len() != study.variables.len() {
        return Err(Error::Input("duplicate study variable".into()));
    }
    if !vars.contains(study.index.as_str()) {
        return Err(Error::Input("index must name a study variable".into()));
    }
    for v in &study.variables {
        if study.schema_version == "1" && v.role.is_some() {
            return Err(Error::Input(format!(
                "variable {}: roles require study schema version 2",
                v.name
            )));
        }
        if study.schema_version == "2" && v.role.is_none() {
            return Err(Error::Input(format!("variable {} requires a role", v.name)));
        }
        if !["string", "integer", "number", "boolean"].contains(&v.kind.as_str()) {
            return Err(Error::Input(format!(
                "unsupported type '{}' for {}",
                v.kind, v.name
            )));
        }
    }
    if study.schema_version == "2"
        && study
            .variables
            .iter()
            .find(|v| v.name == study.index)
            .is_some_and(|v| v.role != Some(VariableRole::IndexContext))
    {
        return Err(Error::Input(
            "index variable must have index_context role".into(),
        ));
    }
    let mut ids = HashSet::new();
    for e in &fixture.events {
        if !ids.insert(&e.id) {
            return Err(Error::Input(format!("duplicate event id {}", e.id)));
        }
        if !vars.contains(e.variable.as_str()) {
            return Err(Error::Input(format!("unknown variable '{}'", e.variable)));
        }
        if e.recording_day < e.event_day {
            return Err(Error::Input(format!(
                "event {} recorded before event time",
                e.id
            )));
        }
        if e.value.is_some() == e.missing.is_some() {
            return Err(Error::Input(format!(
                "event {} must have exactly one of value or missing",
                e.id
            )));
        }
        if let Some(value) = &e.value {
            let variable = study
                .variables
                .iter()
                .find(|v| v.name == e.variable)
                .expect("variable existence checked above");
            if !valid_value(&variable.kind, value) {
                return Err(Error::Input(format!(
                    "event {} has invalid {} value for {}",
                    e.id, variable.kind, variable.name
                )));
            }
        }
        if e.provenance.trim().is_empty() {
            return Err(Error::Input(format!("event {} requires provenance", e.id)));
        }
    }
    for r in &study.eligibility {
        if !vars.contains(r.variable.as_str()) {
            return Err(Error::Input(format!(
                "eligibility references unknown variable {}",
                r.variable
            )));
        }
        let variable = study
            .variables
            .iter()
            .find(|v| v.name == r.variable)
            .expect("variable existence checked above");
        if study.schema_version == "2"
            && !matches!(
                variable.role,
                Some(VariableRole::Baseline | VariableRole::IndexContext)
            )
        {
            return Err(Error::Input(format!(
                "eligibility variable {} must have a baseline or index_context role",
                r.variable
            )));
        }
        if !valid_value(&variable.kind, &r.equals) {
            return Err(Error::Input(format!(
                "eligibility for {} has invalid {} value",
                r.variable, variable.kind
            )));
        }
    }
    let mut episode_indexes: HashSet<(&str, &str)> = HashSet::new();
    for event in &fixture.events {
        if event.variable == study.index
            && !episode_indexes.insert((&event.patient_id, &event.episode_id))
        {
            return Err(Error::Input(format!(
                "episode '{}:{}' has multiple index events",
                event.patient_id, event.episode_id
            )));
        }
    }
    let mut observation_keys = HashSet::new();
    for observation in &fixture.observations {
        if !observation_keys.insert((&observation.patient_id, &observation.episode_id)) {
            return Err(Error::Input("duplicate observation episode".into()));
        }
        if observation
            .end_day
            .is_some_and(|end| end < observation.start_day)
        {
            return Err(Error::Input("observation end before start".into()));
        }
        let episode_exists = fixture.events.iter().any(|event| {
            event.patient_id == observation.patient_id && event.episode_id == observation.episode_id
        });
        if !episode_exists {
            return Err(Error::Input(
                "observation references unknown episode".into(),
            ));
        }
        if let Some(index) = fixture.events.iter().find(|event| {
            event.patient_id == observation.patient_id
                && event.episode_id == observation.episode_id
                && event.variable == study.index
        }) {
            if observation.start_day > index.event_day
                || observation.end_day.is_some_and(|end| end < index.event_day)
            {
                return Err(Error::Input("observation must include index".into()));
            }
        }
    }
    Ok(())
}
pub fn run(study: &Study, fixture: &Fixture) -> Result<RunReport, Error> {
    validate(study, fixture)?;
    let mut groups: BTreeMap<(String, String), Vec<&Event>> = BTreeMap::new();
    for e in &fixture.events {
        groups
            .entry((e.patient_id.clone(), e.episode_id.clone()))
            .or_default()
            .push(e)
    }
    let mut cohort = vec![];
    let mut exclusions = vec![];
    let distinct_episodes = groups.len();
    let distinct_patients = fixture
        .events
        .iter()
        .map(|e| &e.patient_id)
        .collect::<HashSet<_>>()
        .len();
    let episodes_with_index = groups
        .values()
        .filter(|es| es.iter().any(|e| e.variable == study.index))
        .count();
    for ((p, ep), es) in groups {
        let indexes: Vec<_> = es.iter().filter(|e| e.variable == study.index).collect();
        if indexes.is_empty() {
            exclusions.push(Exclusion {
                patient_id: p.clone(),
                episode_id: ep.clone(),
                reason: "missing_index".into(),
                source_event_ids: es.iter().map(|e| e.id.clone()).collect(),
            });
        }
        for ix in indexes {
            let observation = fixture
                .observations
                .iter()
                .find(|o| o.patient_id == p && o.episode_id == ep);
            let follow_up_end =
                ix.event_day
                    .checked_add(study.follow_up_days)
                    .ok_or_else(|| {
                        Error::Input(format!("follow-up boundary overflows for event {}", ix.id))
                    })?;
            let available: Vec<_> = es
                .iter()
                .filter(|e| {
                    e.recording_day <= follow_up_end
                        && e.event_day <= follow_up_end
                        && observation.is_none_or(|o| {
                            e.event_day >= o.start_day
                                && o.end_day.is_none_or(|end| e.event_day <= end)
                                && e.recording_day <= o.end_day.unwrap_or(i64::MAX)
                        })
                })
                .collect();
            let mut values = BTreeMap::new();
            let mut sources = vec![];
            let mut variable_sources = BTreeMap::new();
            for v in &study.variables {
                let selected = available
                    .iter()
                    .filter(|e| {
                        e.variable == v.name
                            && match v.role {
                                Some(VariableRole::Baseline) => {
                                    e.event_day <= ix.event_day && e.recording_day <= ix.event_day
                                }
                                Some(VariableRole::IndexContext) => {
                                    e.event_day == ix.event_day && e.recording_day <= ix.event_day
                                }
                                Some(VariableRole::FollowUpOutcome) => e.event_day >= ix.event_day,
                                None => true,
                            }
                    })
                    .max_by_key(|e| (e.event_day, e.recording_day));
                if let Some(e) = selected {
                    values.insert(v.name.clone(), event_value(e));
                    sources.push(e.id.clone());
                } else {
                    values.insert(v.name.clone(), ValueState::Absent);
                }
                if study.schema_version == "2" {
                    variable_sources.insert(v.name.clone(), selected.map(|e| e.id.clone()));
                }
            }
            let mut eligibility_values = BTreeMap::new();
            for rule in &study.eligibility {
                if study.schema_version == "2" {
                    eligibility_values.insert(
                        rule.variable.clone(),
                        EligibilityValue {
                            value: values[&rule.variable].clone(),
                            source_event_id: variable_sources[&rule.variable].clone(),
                        },
                    );
                    continue;
                }
                let baseline = available
                    .iter()
                    .filter(|e| {
                        e.variable == rule.variable
                            && e.recording_day <= ix.event_day
                            && e.event_day <= ix.event_day
                            && (variable_role(study, &rule.variable)
                                != Some(VariableRole::IndexContext)
                                || e.event_day == ix.event_day)
                    })
                    .max_by_key(|e| (e.event_day, e.recording_day));
                eligibility_values.insert(
                    rule.variable.clone(),
                    EligibilityValue {
                        value: baseline.map_or(ValueState::Absent, |e| event_value(e)),
                        source_event_id: baseline.map(|e| e.id.clone()),
                    },
                );
            }
            let failed = study.eligibility.iter().find(|r| {
                !matches!(&eligibility_values[&r.variable].value, ValueState::Observed(v) if v == &r.equals)
            });
            if let Some(rule) = failed {
                exclusions.push(Exclusion {
                    patient_id: p.clone(),
                    episode_id: ep.clone(),
                    reason: format!("eligibility:{} != {}", rule.variable, rule.equals),
                    source_event_ids: es.iter().map(|e| e.id.clone()).collect(),
                })
            } else {
                cohort.push(CohortRow {
                    patient_id: p.clone(),
                    episode_id: ep.clone(),
                    index_day: ix.event_day,
                    values,
                    eligibility_values,
                    source_event_ids: sources,
                    variable_source_event_ids: variable_sources,
                    observation_start_day: observation.map(|o| o.start_day),
                    observation_end_day: observation.and_then(|o| o.end_day),
                    censoring_reason: observation.and_then(|o| o.censoring_reason.clone()),
                })
            }
        }
    }
    let mut summary = CohortSummary {
        source_events: fixture.events.len(),
        distinct_patients,
        distinct_episodes,
        episodes_with_index,
        included_rows: cohort.len(),
        excluded_episodes: exclusions.len(),
        ..Default::default()
    };
    for exclusion in &exclusions {
        *summary
            .exclusion_reasons
            .entry(exclusion.reason.clone())
            .or_default() += 1;
    }
    for value in cohort.iter().flat_map(|r| r.values.values()) {
        match value {
            ValueState::Observed(_) => summary.derived_values.observed += 1,
            ValueState::Missing(_) => summary.derived_values.missing += 1,
            ValueState::Absent => summary.derived_values.absent += 1,
        }
    }
    if summary.included_rows + summary.excluded_episodes != summary.distinct_episodes
        || summary.exclusion_reasons.values().sum::<usize>() != exclusions.len()
        || summary.derived_values.observed
            + summary.derived_values.missing
            + summary.derived_values.absent
            != cohort.len() * study.variables.len()
    {
        return Err(Error::Input(
            "cohort summary does not reconcile with artifacts".into(),
        ));
    }
    let diagnostics = diagnostics(study, fixture, &cohort, &exclusions);
    Ok(RunReport {
        cohort,
        exclusions,
        summary,
        diagnostics,
        integrity: vec![
            "eligibility_availability: event_day <= index_day and recording_day <= index_day".into(),
            "derived_values_availability: event_day <= follow_up_end and recording_day <= follow_up_end".into(),
            "follow_up_boundary: inclusive".into(),
            "causal_interpretation: none".into(),
        ],
        manifest: RunManifest {
            schema_version: "1".into(),
            generator: None,
            evidence: None,
            software: None,
            study_fingerprint: None,
            generator_fingerprint: None,
            fixture_fingerprint: None,
            artifacts: vec![
                "cohort.yaml".into(),
                "exclusions.yaml".into(),
                "summary.yaml".into(),
                "integrity.yaml".into(),
                "diagnostics.yaml".into(),
                "manifest.yaml".into(),
            ],
        },
    })
}

#[cfg(test)]
mod tests {
    #[test]
    fn patient_bootstrap_preserves_clusters_and_handles_missingness() {
        let mut events = Vec::new();
        for (id, patient, episode, value) in [
            ("a", "p1", "e1", Some("true")),
            ("b", "p1", "e2", Some("true")),
            ("c", "p2", "e3", Some("false")),
        ] {
            let mut index = event(&format!("{id}i"), "index", Some("true"), None, 0, 0);
            index.patient_id = patient.into();
            index.episode_id = episode.into();
            let mut outcome = event(&format!("{id}o"), "outcome", value, None, 1, 1);
            outcome.patient_id = patient.into();
            outcome.episode_id = episode.into();
            events.extend([index, outcome]);
        }
        let fixture = Fixture {
            schema_version: "1".into(),
            events,
            observations: vec![],
        };
        let rows = run(&study(), &fixture).unwrap().cohort;
        let groups = bootstrap_patients(&rows);
        assert_eq!(groups.iter().map(Vec::len).collect::<Vec<_>>(), vec![2, 1]);
        assert_eq!(
            bootstrap_proportion(&groups, &[0, 1], "outcome", "true", None),
            Some(2.0 / 3.0)
        );
        assert_eq!(
            bootstrap_proportion(&groups, &[0, 0], "outcome", "true", None),
            Some(1.0)
        );
        let config: EvaluationConfig = serde_yaml::from_str("schema_version: '1'\njoint: {exposure: index, outcome: outcome}\nuncertainty: {seed: 3, replicates: 200, confidence_level: 0.95}\n").unwrap();
        let result = evaluate(&study(), &fixture, &fixture, &config).unwrap();
        assert_eq!(
            serde_yaml::to_string(&result).unwrap(),
            serde_yaml::to_string(&evaluate(&study(), &fixture, &fixture, &config).unwrap())
                .unwrap()
        );
        let interval = result.variables["outcome"].categories["true"]
            .uncertainty
            .as_ref()
            .unwrap();
        assert_eq!(interval.signed_difference, Some(0.0));
        assert_eq!(interval.synthetic.valid_replicates, 200);
        assert_eq!(interval.synthetic.lower, Some(0.0));
        assert_eq!(interval.synthetic.upper, Some(1.0));
        assert!(result
            .joint
            .unwrap()
            .categories
            .iter()
            .all(|cell| cell.comparison.uncertainty.is_some()));
        let mut different = fixture.clone();
        for event in &mut different.events {
            if event.variable == "outcome" {
                event.value = Some("false".into());
            }
        }
        let shifted = evaluate(&study(), &fixture, &different, &config).unwrap();
        assert_eq!(
            shifted.variables["outcome"].categories["true"]
                .uncertainty
                .as_ref()
                .unwrap()
                .signed_difference,
            Some(2.0 / 3.0)
        );
        let mut partially_missing = fixture.clone();
        for event in &mut partially_missing.events {
            if event.variable == "outcome" && event.patient_id == "p2" {
                event.value = None;
                event.missing = Some("unknown".into());
            }
        }
        let missing = evaluate(&study(), &partially_missing, &fixture, &config).unwrap();
        let interval = &missing.variables["outcome"].categories["true"]
            .uncertainty
            .as_ref()
            .unwrap()
            .synthetic;
        assert!(interval.valid_replicates > 0 && interval.valid_replicates < 200);
        assert!(interval.lower.is_none());
        let empty = Fixture {
            schema_version: "1".into(),
            events: vec![],
            observations: vec![],
        };
        let empty_result = evaluate(&study(), &empty, &fixture, &config).unwrap();
        let interval = empty_result.variables["outcome"].categories["true"]
            .uncertainty
            .as_ref()
            .unwrap();
        assert_eq!(interval.synthetic.valid_replicates, 0);
        assert!(interval.synthetic.lower.is_none());
        assert!(interval.synthetic.unavailable_reason.is_some());
        for bad in [
            "{seed: 0, replicates: 0, confidence_level: 0.95}",
            "{seed: 0, replicates: 5, confidence_level: 0}",
            "{seed: 0, replicates: 5, confidence_level: 1}",
        ] {
            let invalid: EvaluationConfig =
                serde_yaml::from_str(&format!("schema_version: '1'\nuncertainty: {bad}\n"))
                    .unwrap();
            assert!(evaluate(&study(), &fixture, &fixture, &invalid).is_err());
        }
    }
    use super::*;

    #[test]
    fn longitudinal_selection_validates_window_and_variables() {
        let mut config = LongitudinalConfig {
            start_day: 0,
            end_day: 10,
            variables: vec!["index".into(), "outcome".into()],
            pairs: BTreeMap::from([(
                "sequence".into(),
                JointSelection {
                    exposure: "index".into(),
                    outcome: "outcome".into(),
                },
            )]),
        };
        assert!(config.validate(&study()).is_ok());
        config.end_day = -1;
        assert!(config.validate(&study()).is_err());
        config.end_day = 10;
        config.variables.pop();
        assert!(config.validate(&study()).is_err());
    }

    #[test]
    fn longitudinal_events_preserve_coverage_and_source_ids() {
        let config = LongitudinalConfig {
            start_day: 0,
            end_day: 10,
            variables: vec!["outcome".into()],
            pairs: BTreeMap::new(),
        };
        let mut fixture = Fixture {
            schema_version: "2".into(),
            events: vec![
                event("i", "index", Some("true"), None, 0, 0),
                event("a", "outcome", Some("true"), None, 1, 1),
                event("m", "outcome", None, Some("unknown"), 10, 10),
                event("late", "outcome", Some("true"), None, 10, 11),
            ],
            observations: vec![Observation {
                patient_id: "p".into(),
                episode_id: "e".into(),
                start_day: 0,
                end_day: Some(10),
                censoring_reason: None,
            }],
        };
        let report = run(&study(), &fixture).unwrap();
        let result = longitudinal_episodes(&config, &fixture, &report.cohort);
        assert_eq!(result[0].coverage, "complete");
        assert_eq!(result[0].variables["outcome"].len(), 2);
        assert!(result[0].variables["outcome"][1].missing);
        assert_eq!(result[0].variables["outcome"][1].source_event_id, "m");
        let summary = summarize_longitudinal(&config, &result);
        assert_eq!(
            summary.variables["outcome"].complete_episode_counts["one"],
            1
        );
        assert_eq!(summary.variables["outcome"].missing_events, 1);
        assert_eq!(summary.variables["outcome"].complete_episode_denominator, 1);
        assert_eq!(summary.variables["outcome"].observed_event_denominator, 1);
        let eval_config: EvaluationConfig = serde_yaml::from_str(
            r#"
schema_version: "4"
longitudinal:
  start_day: 0
  end_day: 10
  variables: [index, outcome]
  pairs:
    sequence: {exposure: index, outcome: outcome}
"#,
        )
        .unwrap();
        let mut changed = fixture.clone();
        changed
            .events
            .iter_mut()
            .find(|e| e.id == "a")
            .unwrap()
            .event_day = 0;
        let evaluation = evaluate(&study(), &changed, &fixture, &eval_config).unwrap();
        let longitudinal = evaluation.longitudinal.unwrap();
        assert_eq!(
            longitudinal.pair_order["sequence"]["same_day"].absolute_difference,
            Some(1.0)
        );
        let empty = Fixture {
            schema_version: "1".into(),
            events: vec![],
            observations: vec![],
        };
        let empty_report = evaluate(&study(), &empty, &empty, &eval_config)
            .unwrap()
            .longitudinal
            .unwrap();
        assert_eq!(
            empty_report.variable_counts["outcome"]["zero"].absolute_difference,
            None
        );
        fixture.observations.clear();
        let report = run(&study(), &fixture).unwrap();
        assert_eq!(
            longitudinal_episodes(&config, &fixture, &report.cohort)[0].coverage,
            "unknown"
        );
    }

    #[test]
    fn longitudinal_coverage_denominators_do_not_treat_unknown_as_zero() {
        let config = LongitudinalConfig {
            start_day: 0,
            end_day: 10,
            variables: vec!["index".into(), "outcome".into()],
            pairs: BTreeMap::from([(
                "pair".into(),
                JointSelection {
                    exposure: "index".into(),
                    outcome: "outcome".into(),
                },
            )]),
        };
        let mut fixture = Fixture {
            schema_version: "2".into(),
            events: vec![],
            observations: vec![],
        };
        for i in 0..3 {
            let patient = format!("p{i}");
            let mut ix = event(&format!("i{i}"), "index", Some("true"), None, 0, 0);
            ix.patient_id = patient.clone();
            fixture.events.push(ix);
            if i == 0 {
                let mut missing = event("m", "outcome", None, Some("unknown"), 1, 1);
                missing.patient_id = patient.clone();
                fixture.events.push(missing);
            }
            if i != 2 {
                fixture.observations.push(Observation {
                    patient_id: patient,
                    episode_id: "e".into(),
                    start_day: 0,
                    end_day: Some(if i == 0 { 10 } else { 5 }),
                    censoring_reason: None,
                });
            }
        }
        let report = run(&study(), &fixture).unwrap();
        let episodes = longitudinal_episodes(&config, &fixture, &report.cohort);
        let summary = summarize_longitudinal(&config, &episodes);
        assert_eq!(summary.coverage["complete"], 1);
        assert_eq!(summary.coverage["incomplete"], 1);
        assert_eq!(summary.coverage["unknown"], 1);
        assert_eq!(
            summary.variables["outcome"].complete_episode_counts["zero"],
            1
        );
        assert_eq!(summary.variables["outcome"].missing_events, 1);
        assert_eq!(summary.pairs["pair"].complete_episode_denominator, 1);
        assert_eq!(summary.pairs["pair"].both_observed_denominator, 0);
    }

    #[test]
    fn timing_fit_uses_complete_coverage_and_emits_bins() {
        let mut s = study();
        s.schema_version = "2".into();
        s.variables[0].role = Some(VariableRole::IndexContext);
        s.variables[1].role = Some(VariableRole::FollowUpOutcome);
        let fixture = Fixture {
            schema_version: "2".into(),
            events: vec![
                event("i", "index", Some("true"), None, 0, 0),
                event("o", "outcome", Some("false"), None, 10, 10),
            ],
            observations: vec![Observation {
                patient_id: "p".into(),
                episode_id: "e".into(),
                start_day: 0,
                end_day: Some(10),
                censoring_reason: None,
            }],
        };
        let config: FitConfig = serde_yaml::from_str(
            r#"
schema_version: "3"
seed: 1
episode_count: 5
event_day: 0
recording_day: 0
variables:
  - {name: index}
  - {name: outcome, parents: [index]}
timing:
  variable: outcome
  bins:
    - {start_day: 0, end_day: 5}
    - {start_day: 6, end_day: 10}
"#,
        )
        .unwrap();
        let result = fit(&s, &fixture, &config).unwrap();
        assert_eq!(
            result.manifest.timing_counts.as_ref().unwrap()["observed"],
            1
        );
        assert_eq!(
            result.generator.variables[1].table[0].outcomes[0].bin_end_day,
            Some(10)
        );
        assert_eq!(
            generate(&s, &result.generator).unwrap().observations.len(),
            5
        );
        let evaluation = EvaluationConfig {
            uncertainty: None,
            continuous: None,
            schema_version: "3".into(),
            longitudinal: None,
            joint: None,
            pairs: BTreeMap::new(),
            temporal: None,
            outcome_timing: config.timing.clone(),
        };
        let mut shifted = fixture.clone();
        shifted.events[1].event_day = 0;
        shifted.events[1].recording_day = 0;
        let comparison = evaluate(&s, &shifted, &fixture, &evaluation)
            .unwrap()
            .outcome_timing
            .unwrap();
        assert_eq!(comparison.synthetic.observed, 1);
        assert!(comparison
            .cells
            .iter()
            .any(|c| c.comparison.absolute_difference == Some(1.0)));
        let mut incomplete = fixture.clone();
        incomplete.observations[0].end_day = Some(9);
        assert!(fit(&s, &incomplete, &config).is_err());
        let mut mixed = fixture.clone();
        for i in 1..=4 {
            let patient = format!("p{i}");
            let mut ix = event(&format!("i{i}"), "index", Some("true"), None, 0, 0);
            ix.patient_id = patient.clone();
            mixed.events.push(ix);
            if i != 4 {
                let mut outcome = if i == 3 {
                    event(&format!("o{i}"), "outcome", None, Some("unknown"), 1, 1)
                } else {
                    event(&format!("o{i}"), "outcome", Some("true"), None, 1, 1)
                };
                outcome.patient_id = patient.clone();
                mixed.events.push(outcome);
            }
            if i != 1 {
                mixed.observations.push(Observation {
                    patient_id: patient,
                    episode_id: "e".into(),
                    start_day: 0,
                    end_day: Some(if i == 2 { 0 } else { 10 }),
                    censoring_reason: None,
                });
            }
        }
        let fitted = fit(&s, &mixed, &config).unwrap();
        let counts = fitted.manifest.timing_counts.unwrap();
        assert_eq!(counts["observed"], 1);
        assert_eq!(counts["unknown_coverage"], 1);
        assert_eq!(counts["incomplete_coverage"], 1);
        assert_eq!(counts["missing"], 1);
        assert_eq!(counts["absent"], 1);
        assert_eq!(counts.values().sum::<usize>(), 5);
        assert_eq!(fitted.manifest.timing_tables[0].denominator, 1);
    }

    #[test]
    fn timing_state_requires_complete_declared_coverage() {
        let mut s = study();
        s.schema_version = "2".into();
        s.variables[0].role = Some(VariableRole::IndexContext);
        s.variables[1].role = Some(VariableRole::FollowUpOutcome);
        let mut f = Fixture {
            schema_version: "2".into(),
            events: vec![
                event("i", "index", Some("true"), None, 0, 0),
                event("o", "outcome", Some("false"), None, 10, 10),
            ],
            observations: vec![],
        };
        let report = run(&s, &f).unwrap();
        assert_eq!(
            outcome_timing_state(&s, &f, &report.cohort[0], "outcome").unwrap(),
            OutcomeTimingState::UnknownCoverage
        );
        f.observations.push(Observation {
            patient_id: "p".into(),
            episode_id: "e".into(),
            start_day: 0,
            end_day: Some(9),
            censoring_reason: None,
        });
        let report = run(&s, &f).unwrap();
        assert_eq!(
            outcome_timing_state(&s, &f, &report.cohort[0], "outcome").unwrap(),
            OutcomeTimingState::IncompleteCoverage
        );
        f.observations[0].end_day = Some(10);
        let report = run(&s, &f).unwrap();
        assert_eq!(
            outcome_timing_state(&s, &f, &report.cohort[0], "outcome").unwrap(),
            OutcomeTimingState::Observed {
                value: "false".into(),
                day: 10
            }
        );
        f.events[1].value = None;
        f.events[1].missing = Some("unknown".into());
        let report = run(&s, &f).unwrap();
        assert_eq!(
            outcome_timing_state(&s, &f, &report.cohort[0], "outcome").unwrap(),
            OutcomeTimingState::Missing
        );
        f.events.pop();
        let report = run(&s, &f).unwrap();
        assert_eq!(
            outcome_timing_state(&s, &f, &report.cohort[0], "outcome").unwrap(),
            OutcomeTimingState::Absent
        );
    }

    #[test]
    fn version_three_generator_samples_bins_and_declares_coverage() {
        let mut s = study();
        s.schema_version = "2".into();
        s.variables[0].role = Some(VariableRole::IndexContext);
        s.variables[1].role = Some(VariableRole::FollowUpOutcome);
        let config: GeneratorConfig = serde_yaml::from_str(
            r#"
schema_version: "3"
seed: 42
episode_count: 20
complete_observation: true
variables:
  - name: index
    outcomes:
      - {value: "true", probability: 1.0, event_day: 0, recording_day: 0}
  - name: outcome
    outcomes:
      - {value: "true", probability: 0.5, event_day: 0, recording_day: 0, bin_end_day: 3}
      - {value: "true", probability: 0.5, event_day: 4, recording_day: 4, bin_end_day: 10}
"#,
        )
        .unwrap();
        let a = generate(&s, &config).unwrap();
        let b = generate(&s, &config).unwrap();
        assert_eq!(
            serde_yaml::to_string(&a).unwrap(),
            serde_yaml::to_string(&b).unwrap()
        );
        assert_eq!(a.observations.len(), 20);
        assert!(a
            .events
            .iter()
            .filter(|e| e.variable == "outcome")
            .all(|e| (0..=10).contains(&e.event_day) && e.event_day == e.recording_day));
        let mut invalid = config.clone();
        invalid.variables[1].outcomes[1].event_day = 3;
        invalid.variables[1].outcomes[1].recording_day = 3;
        assert!(generate(&s, &invalid).is_err());
    }

    #[test]
    fn outcome_time_bins_validate_inclusive_boundaries() {
        let bins = vec![
            OutcomeTimeBin {
                start_day: 0,
                end_day: 0,
            },
            OutcomeTimeBin {
                start_day: 1,
                end_day: 365,
            },
        ];
        assert!(OutcomeTimeBin::validate_ordered(&bins, 365).is_ok());
        assert!(OutcomeTimeBin::validate_ordered(&bins, 364).is_err());
        assert!(
            OutcomeTimeBin::validate_ordered(&[bins[0].clone(), bins[0].clone()], 365).is_err()
        );
        assert!(OutcomeTimeBin::validate_ordered(&[], 365).is_err());
        assert!(OutcomeTimeBin::validate_ordered(
            &[OutcomeTimeBin {
                start_day: 0,
                end_day: i64::MAX
            }],
            i64::MAX
        )
        .is_ok());
    }
    #[test]
    fn outcome_day_sampling_is_deterministic_and_bounded() {
        let bin = OutcomeTimeBin {
            start_day: 3,
            end_day: 7,
        };
        let mut a = 42;
        let mut b = 42;
        let first: Vec<_> = (0..100)
            .map(|_| sample_outcome_day(&mut a, &bin).unwrap())
            .collect();
        let second: Vec<_> = (0..100)
            .map(|_| sample_outcome_day(&mut b, &bin).unwrap())
            .collect();
        assert_eq!(first, second);
        assert!(first.iter().all(|day| (3..=7).contains(day)));
        assert!(first.contains(&3) && first.contains(&7));
        assert_eq!(
            sample_outcome_day(
                &mut a,
                &OutcomeTimeBin {
                    start_day: i64::MAX,
                    end_day: i64::MAX
                }
            )
            .unwrap(),
            i64::MAX
        );
        assert!(sample_outcome_day(
            &mut a,
            &OutcomeTimeBin {
                start_day: -1,
                end_day: 1
            }
        )
        .is_err());
    }

    #[test]
    fn conditional_fit_counts_and_named_pairs() {
        let mut events = vec![];
        for (i, (parent, child)) in [("true", "true"), ("true", "false"), ("false", "false")]
            .iter()
            .enumerate()
        {
            let mut ix = event(&format!("i{i}"), "index", Some(parent), None, 0, 0);
            let mut out = event(&format!("o{i}"), "outcome", Some(child), None, 1, 1);
            ix.patient_id = format!("p{i}");
            out.patient_id = ix.patient_id.clone();
            events.extend([ix, out]);
        }
        let fixture = Fixture {
            schema_version: "1".into(),
            events,
            observations: vec![],
        };
        let config = FitConfig {
            schema_version: "2".into(),
            timing: None,
            seed: 7,
            episode_count: 100,
            event_day: 0,
            recording_day: 0,
            variables: vec![
                FitVariable {
                    name: "index".into(),
                    parents: vec![],
                },
                FitVariable {
                    name: "outcome".into(),
                    parents: vec!["index".into()],
                },
            ],
        };
        let fitted = fit(&study(), &fixture, &config).unwrap();
        let mut invalid_order = config.clone();
        invalid_order.variables.swap(0, 1);
        assert!(fit(&study(), &fixture, &invalid_order)
            .err()
            .unwrap()
            .to_string()
            .contains("parent order"));
        assert_eq!(fitted.generator.variables[1].table.len(), 2);
        let row = fitted.manifest.variables["outcome"]
            .conditional
            .iter()
            .find(|r| r.when["index"] == "true")
            .unwrap();
        assert_eq!(row.observed, 2);
        assert_eq!(row.categories["true"], 1);
        assert_eq!(
            serde_yaml::to_string(&fitted.generator).unwrap(),
            serde_yaml::to_string(&fit(&study(), &fixture, &config).unwrap().generator).unwrap()
        );
        let evaluation = EvaluationConfig {
            uncertainty: None,
            continuous: None,
            schema_version: "2".into(),
            longitudinal: None,
            outcome_timing: None,
            joint: None,
            temporal: None,
            pairs: BTreeMap::from([(
                "relationship".into(),
                JointSelection {
                    exposure: "index".into(),
                    outcome: "outcome".into(),
                },
            )]),
        };
        let result = evaluate(&study(), &fixture, &fixture, &evaluation).unwrap();
        assert_eq!(result.pairs["relationship"].synthetic_denominator, 3);
        assert!(result.pairs["relationship"]
            .categories
            .iter()
            .all(|c| c.comparison.absolute_difference == Some(0.0)));
        let mut changed = fixture.clone();
        changed
            .events
            .iter_mut()
            .find(|e| e.id == "o0")
            .unwrap()
            .value = Some("false".into());
        let different = evaluate(&study(), &changed, &fixture, &evaluation).unwrap();
        assert!(different.pairs["relationship"]
            .categories
            .iter()
            .any(|c| c.comparison.absolute_difference.is_some_and(|v| v > 0.0)));
        let mut incomplete = fixture.clone();
        incomplete.events.retain(|e| e.id != "o2");
        assert!(fit(&study(), &incomplete, &config)
            .err()
            .unwrap()
            .to_string()
            .contains("parent combination"));
    }

    #[test]
    fn conditional_fit_excludes_missing_parents_and_late_events() {
        let mut s = study();
        s.schema_version = "2".into();
        s.variables[0].role = Some(VariableRole::IndexContext);
        s.variables[1].role = Some(VariableRole::FollowUpOutcome);
        s.variables.push(Variable {
            name: "baseline".into(),
            kind: "boolean".into(),
            role: Some(VariableRole::Baseline),
        });
        let mut events = Vec::new();
        for i in 0..4 {
            let patient = format!("p{i}");
            let mut ix = event(&format!("i{i}"), "index", Some("true"), None, 0, 0);
            ix.patient_id = patient.clone();
            let mut outcome = event(&format!("o{i}"), "outcome", Some("true"), None, 1, 1);
            outcome.patient_id = patient.clone();
            events.extend([ix, outcome]);
            if i != 3 {
                let mut baseline = match i {
                    0 => event("b0", "baseline", Some("true"), None, 0, 0),
                    1 => event("b1", "baseline", None, Some("unknown"), 0, 0),
                    _ => event("b2", "baseline", Some("false"), None, 0, 1),
                };
                baseline.patient_id = patient;
                events.push(baseline);
            }
        }
        let fixture = Fixture {
            schema_version: "1".into(),
            events,
            observations: vec![],
        };
        let config = FitConfig {
            schema_version: "2".into(),
            timing: None,
            seed: 1,
            episode_count: 10,
            event_day: 0,
            recording_day: 0,
            variables: vec![
                FitVariable {
                    name: "index".into(),
                    parents: vec![],
                },
                FitVariable {
                    name: "baseline".into(),
                    parents: vec![],
                },
                FitVariable {
                    name: "outcome".into(),
                    parents: vec!["baseline".into()],
                },
            ],
        };
        let result = fit(&s, &fixture, &config).unwrap();
        let counts = &result.manifest.variables["outcome"];
        assert_eq!((counts.parent_missing, counts.parent_absent), (1, 2));
        assert_eq!(counts.conditional[0].observed, 1);
        assert_eq!(result.generator.variables[1].outcomes.len(), 1);
        let mut censored = fixture.clone();
        censored.schema_version = "2".into();
        censored.observations.push(Observation {
            patient_id: "p0".into(),
            episode_id: "e".into(),
            start_day: 0,
            end_day: Some(0),
            censoring_reason: Some("transfer".into()),
        });
        assert!(fit(&s, &censored, &config)
            .err()
            .unwrap()
            .to_string()
            .contains("parent combination"));
    }

    #[test]
    fn empirical_fit_is_deterministic_and_reconciles_value_states() {
        let s = study();
        let mut i1 = event("i1", "index", Some("true"), None, 0, 0);
        let mut o1 = event("o1", "outcome", Some("false"), None, 1, 1);
        let mut i2 = event("i2", "index", Some("true"), None, 0, 0);
        let mut o2 = event("o2", "outcome", None, Some("unknown"), 1, 1);
        i1.episode_id = "e1".into();
        o1.episode_id = "e1".into();
        i2.episode_id = "e2".into();
        o2.episode_id = "e2".into();
        let f = Fixture {
            schema_version: "1".into(),
            events: vec![i1, o1, i2, o2],
            observations: vec![],
        };
        let c = FitConfig {
            schema_version: "1".into(),
            timing: None,
            seed: 7,
            episode_count: 20,
            event_day: 0,
            recording_day: 0,
            variables: vec![],
        };
        let a = fit(&s, &f, &c).unwrap();
        let b = fit(&s, &f, &c).unwrap();
        assert_eq!(
            serde_yaml::to_string(&a.generator).unwrap(),
            serde_yaml::to_string(&b.generator).unwrap()
        );
        assert_eq!(
            serde_yaml::to_string(&a.manifest).unwrap(),
            serde_yaml::to_string(&b.manifest).unwrap()
        );
        let outcome = &a
            .generator
            .variables
            .iter()
            .find(|v| v.name == "outcome")
            .unwrap()
            .outcomes;
        assert_eq!(outcome.len(), 1);
        assert_eq!(outcome[0].value, "false");
        assert_eq!(outcome[0].probability, 1.0);
        assert_eq!((outcome[0].event_day, outcome[0].recording_day), (0, 0));
        let counts = &a.manifest.variables["outcome"];
        assert_eq!((counts.observed, counts.missing, counts.absent), (1, 1, 0));
        assert_eq!(counts.categories.values().sum::<usize>(), counts.observed);
        assert_eq!(a.manifest.included_episodes, 2);
    }

    #[test]
    fn fit_rejects_invalid_config_and_empty_observed_variable() {
        let s = study();
        let f = Fixture {
            schema_version: "1".into(),
            events: vec![event("i", "index", Some("true"), None, 0, 0)],
            observations: vec![],
        };
        let c = FitConfig {
            schema_version: "1".into(),
            timing: None,
            seed: 7,
            episode_count: 20,
            event_day: 0,
            recording_day: 0,
            variables: vec![],
        };
        assert!(fit(
            &s,
            &f,
            &FitConfig {
                episode_count: 0,
                ..c.clone()
            }
        )
        .is_err());
        assert!(fit(
            &s,
            &Fixture {
                schema_version: "2".into(),
                events: f.events.clone(),
                observations: vec![],
            },
            &c
        )
        .is_err());
        assert!(fit(&s, &f, &c).is_err());
    }
    #[test]
    fn fit_counts_patient_episode_pairs_and_validates_generator_timing() {
        let mut second_index = event("i2", "index", Some("false"), None, 0, 0);
        let mut second_outcome = event("o2", "outcome", Some("true"), None, 1, 1);
        second_index.patient_id = "p2".into();
        second_outcome.patient_id = "p2".into();
        let fixture = Fixture {
            schema_version: "1".into(),
            events: vec![
                event("i1", "index", Some("true"), None, 0, 0),
                event("o1", "outcome", Some("false"), None, 1, 1),
                second_index,
                second_outcome,
            ],
            observations: vec![],
        };
        let config = FitConfig {
            schema_version: "1".into(),
            timing: None,
            seed: 7,
            episode_count: 2,
            event_day: 0,
            recording_day: 0,
            variables: vec![],
        };
        let fitted = fit(&study(), &fixture, &config).unwrap();
        assert_eq!(
            (
                fitted.manifest.patients,
                fitted.manifest.episodes,
                fitted.manifest.included_episodes
            ),
            (2, 2, 2)
        );
        assert!(generate(&study(), &fitted.generator).is_ok());
        let before_event = FitConfig {
            event_day: 1,
            recording_day: 0,
            ..config.clone()
        };
        assert!(fit(&study(), &fixture, &before_event)
            .err()
            .unwrap()
            .to_string()
            .contains("invalid outcome"));
        let overflow = FitConfig {
            event_day: i64::MAX,
            recording_day: i64::MAX,
            ..config.clone()
        };
        assert!(fit(&study(), &fixture, &overflow)
            .err()
            .unwrap()
            .to_string()
            .contains("follow-up boundary overflows"));
        let boundary = FitConfig {
            event_day: i64::MAX - study().follow_up_days,
            recording_day: i64::MAX - study().follow_up_days,
            ..config
        };
        let fitted = fit(&study(), &fixture, &boundary).unwrap();
        assert!(generate(&study(), &fitted.generator).is_ok());
    }
    #[test]
    fn patient_split_is_stable_and_keeps_repeated_episodes_together() {
        let mut second = event("i2", "index", Some("true"), None, 0, 0);
        second.patient_id = "p2".into();
        second.episode_id = "e2".into();
        let mut repeat = event("o1", "outcome", Some("false"), None, 1, 1);
        repeat.episode_id = "e2".into();
        let fixture = Fixture {
            schema_version: "1".into(),
            events: vec![
                event("i", "index", Some("true"), None, 0, 0),
                second,
                repeat,
            ],
            observations: vec![],
        };
        let config = SplitConfig {
            schema_version: "1".into(),
            seed: 42,
            development_proportion: 0.5,
        };
        let a = split_fixture(&study(), &fixture, &config).unwrap();
        let b = split_fixture(&study(), &fixture, &config).unwrap();
        assert_eq!(
            serde_yaml::to_string(&a.development).unwrap(),
            serde_yaml::to_string(&b.development).unwrap()
        );
        assert_eq!(
            serde_yaml::to_string(&a.holdout).unwrap(),
            serde_yaml::to_string(&b.holdout).unwrap()
        );
        for patient in ["patient-1", "p2"] {
            let in_dev = a.development.events.iter().any(|e| e.patient_id == patient);
            let in_hold = a.holdout.events.iter().any(|e| e.patient_id == patient);
            assert!(!(in_dev && in_hold));
        }
        assert_eq!(
            a.manifest.development_patients + a.manifest.holdout_patients,
            2
        );
        assert_eq!(
            a.manifest.development_episodes + a.manifest.holdout_episodes,
            3
        );
    }
    #[test]
    fn patient_split_mixes_sequential_identifiers() {
        for seed in [0, 1, 42] {
            let development = (0..1000)
                .filter(|i| in_development(patient_hash(seed, &format!("patient-{i:06}")), 0.5))
                .count();
            assert!(
                (400..=600).contains(&development),
                "seed {seed}: {development}"
            );
        }
        assert_ne!(
            patient_hash(0, "patient-000001"),
            patient_hash(1, "patient-000001")
        );
    }

    #[test]
    fn patient_split_counts_reused_episode_ids_per_patient() {
        let mut first = event("i1", "index", Some("true"), None, 0, 0);
        first.patient_id = "p1".into();
        first.episode_id = "shared".into();
        let mut second = event("i2", "index", Some("true"), None, 0, 0);
        second.patient_id = "p2".into();
        second.episode_id = "shared".into();
        let fixture = Fixture {
            schema_version: "1".into(),
            events: vec![first, second],
            observations: vec![],
        };
        let result = split_fixture(
            &study(),
            &fixture,
            &SplitConfig {
                schema_version: "1".into(),
                seed: 42,
                development_proportion: 0.5,
            },
        )
        .unwrap();
        assert_eq!(
            result.manifest.development_episodes + result.manifest.holdout_episodes,
            2
        );
    }

    #[test]
    fn patient_split_handles_empty_and_extreme_partitions_and_rejects_bad_proportions() {
        let empty = Fixture {
            schema_version: "1".into(),
            events: vec![],
            observations: vec![],
        };
        let make = |proportion| SplitConfig {
            schema_version: "1".into(),
            seed: 7,
            development_proportion: proportion,
        };
        let result = split_fixture(&study(), &empty, &make(0.5)).unwrap();
        assert!(result.development.events.is_empty() && result.holdout.events.is_empty());
        assert_eq!(result.manifest.actual_development_patient_proportion, None);
        assert!(split_fixture(&study(), &empty, &make(-0.1)).is_err());
        assert!(split_fixture(&study(), &empty, &make(f64::NAN)).is_err());
        let one = Fixture {
            schema_version: "1".into(),
            events: vec![event("i", "index", Some("true"), None, 0, 0)],
            observations: vec![],
        };
        for proportion in [0.0, 1.0] {
            let part = split_fixture(&study(), &one, &make(proportion)).unwrap();
            assert_eq!(
                part.manifest.development_patients + part.manifest.holdout_patients,
                1
            );
            assert_eq!(
                part.manifest.development_episodes + part.manifest.holdout_episodes,
                1
            );
        }
    }
    #[test]
    fn evaluation_counts_marginals_joints_and_empty_denominators() {
        let mut second = event("i2", "index", Some("false"), None, 0, 0);
        second.episode_id = "e2".into();
        let mut missing = event("m2", "outcome", None, Some("unknown"), 1, 1);
        missing.episode_id = "e2".into();
        let mut excluded = event("o3", "outcome", Some("true"), None, 0, 0);
        excluded.episode_id = "e3".into();
        let synthetic = Fixture {
            schema_version: "1".into(),
            events: vec![
                event("i", "index", Some("true"), None, 0, 0),
                event("o", "outcome", Some("false"), None, 1, 1),
                second,
                missing,
                excluded,
            ],
            observations: vec![],
        };
        let reference = Fixture {
            schema_version: "1".into(),
            events: vec![
                event("r", "index", Some("true"), None, 0, 0),
                event("ro", "outcome", Some("true"), None, 1, 1),
            ],
            observations: vec![],
        };
        let config = EvaluationConfig {
            uncertainty: None,
            continuous: None,
            schema_version: "1".into(),
            longitudinal: None,
            outcome_timing: None,
            pairs: BTreeMap::new(),
            temporal: None,
            joint: Some(JointSelection {
                exposure: "index".into(),
                outcome: "outcome".into(),
            }),
        };
        let report = evaluate(&study(), &synthetic, &reference, &config).unwrap();
        assert_eq!(
            (
                report.synthetic.included_episodes,
                report.synthetic.excluded_episodes
            ),
            (2, 1)
        );
        let outcome = &report.variables["outcome"];
        assert_eq!(
            (
                outcome.synthetic_counts.observed,
                outcome.synthetic_counts.missing,
                outcome.synthetic_counts.absent
            ),
            (1, 1, 0)
        );
        assert_eq!(outcome.categories["false"].reference.count, 0);
        assert_eq!(outcome.categories["false"].absolute_difference, Some(1.0));
        let joint = report.joint.unwrap();
        assert_eq!(
            (
                joint.synthetic_denominator,
                joint.reference_denominator,
                joint.categories.len()
            ),
            (1, 1, 4)
        );
        assert_eq!(
            joint
                .categories
                .iter()
                .find(|c| c.exposure == "false" && c.outcome == "true")
                .unwrap()
                .comparison
                .synthetic
                .count,
            0
        );
        let empty = Fixture {
            schema_version: "1".into(),
            events: vec![],
            observations: vec![],
        };
        let report = evaluate(&study(), &synthetic, &empty, &config).unwrap();
        assert_eq!(
            report.variables["outcome"].categories["false"].absolute_difference,
            None
        );
        assert_eq!(
            report.joint.unwrap().categories[0]
                .comparison
                .reference
                .proportion,
            None
        );
        let mut invalid = config.clone();
        invalid.joint.as_mut().unwrap().outcome = "index".into();
        assert!(evaluate(&study(), &synthetic, &reference, &invalid).is_err());
        invalid.joint.as_mut().unwrap().outcome = "unknown".into();
        assert!(evaluate(&study(), &synthetic, &reference, &invalid).is_err());
        let mut bad = reference.clone();
        bad.events[0].recording_day = -1;
        assert!(evaluate(&study(), &synthetic, &bad, &config).is_err());
        let absent = Fixture {
            schema_version: "1".into(),
            events: vec![event("only-index", "index", Some("true"), None, 0, 0)],
            observations: vec![],
        };
        let no_joint = EvaluationConfig {
            uncertainty: None,
            continuous: None,
            schema_version: "1".into(),
            longitudinal: None,
            outcome_timing: None,
            pairs: BTreeMap::new(),
            joint: None,
            temporal: None,
        };
        let report = evaluate(&study(), &absent, &reference, &no_joint).unwrap();
        assert_eq!(report.variables["outcome"].synthetic_counts.absent, 1);
        assert!(report.joint.is_none());
        assert_eq!(
            report.variables["outcome"].categories["true"].absolute_difference,
            None
        );
        let mut numeric = study();
        numeric.variables[1].kind = "integer".into();
        assert!(evaluate(&numeric, &absent, &empty, &config).is_err());
    }
    #[test]
    fn temporal_evaluation_reconciles_windows_and_timing() {
        let mut ix = event("ix", "index", Some("true"), None, 0, 0);
        ix.patient_id = "p".into();
        let mut first = event("a", "outcome", Some("true"), None, 1, 2);
        first.patient_id = "p".into();
        let mut boundary = event("b", "outcome", None, Some("unknown"), 10, 10);
        boundary.patient_id = "p".into();
        let mut second_ix = event("ix2", "index", Some("false"), None, 20, 20);
        second_ix.patient_id = "p".into();
        second_ix.episode_id = "second".into();
        let fixture = Fixture {
            schema_version: "1".into(),
            events: vec![ix, first, boundary, second_ix],
            observations: vec![],
        };
        let empty = Fixture {
            schema_version: "1".into(),
            events: vec![],
            observations: vec![],
        };
        let config = EvaluationConfig {
            uncertainty: None,
            continuous: None,
            schema_version: "1".into(),
            longitudinal: None,
            outcome_timing: None,
            pairs: BTreeMap::new(),
            joint: None,
            temporal: Some(TemporalConfig {
                windows: vec![TemporalWindow {
                    name: "follow".into(),
                    start_day: 1,
                    end_day: 10,
                }],
            }),
        };
        let report = evaluate(&study(), &fixture, &empty, &config).unwrap();
        let temporal = report.temporal.unwrap();
        let a = &temporal.synthetic;
        assert_eq!(a.episodes_per_patient.counts["2"], 1);
        assert_eq!(a.episode_event_span_days.counts["10"], 1);
        assert_eq!(a.last_observed_follow_up_day.counts["1"], 1);
        assert_eq!(a.indexed_episodes_without_observed_follow_up, 1);
        let outcome = &a.variables["outcome"];
        assert_eq!(outcome.event_day.denominator, 2);
        assert_eq!(outcome.recording_day.counts["2"], 1);
        assert_eq!(outcome.recording_delay_days.counts["1"], 1);
        assert_eq!(outcome.index_to_event_days.counts["10"], 1);
        let window = &outcome.windows["follow"];
        assert_eq!(
            (
                window.denominator_episodes,
                window.observed,
                window.missing,
                window.absent
            ),
            (2, 0, 1, 1)
        );
        assert_eq!(
            temporal.reference.variables["outcome"]
                .event_day
                .denominator,
            0
        );
        assert_eq!(
            serde_yaml::to_string(&evaluate(&study(), &fixture, &empty, &config).unwrap()).unwrap(),
            serde_yaml::to_string(&evaluate(&study(), &fixture, &empty, &config).unwrap()).unwrap()
        );
        let mut bad = fixture.clone();
        bad.events[1].recording_day = 0;
        assert!(evaluate(&study(), &bad, &empty, &config).is_err());
        let mut invalid = config.clone();
        invalid.temporal.as_mut().unwrap().windows[0].start_day = 11;
        assert!(evaluate(&study(), &fixture, &empty, &invalid).is_err());
    }

    #[test]
    fn temporal_completeness_handles_overflowing_follow_up_boundary() {
        let fixture = Fixture {
            schema_version: "2".into(),
            events: vec![event("i", "index", Some("true"), None, i64::MAX, i64::MAX)],
            observations: vec![Observation {
                patient_id: "p".into(),
                episode_id: "e".into(),
                start_day: i64::MAX,
                end_day: Some(i64::MAX),
                censoring_reason: None,
            }],
        };
        let result = temporal_cohort(&study(), &fixture, &TemporalConfig { windows: vec![] });
        assert_eq!(result.observation_complete_episodes, 0);
        assert_eq!(result.observation_incomplete_episodes, 1);
    }

    #[test]
    fn observation_bounds_restrict_values_and_report_censoring() {
        let mut index = event("i", "index", Some("true"), None, 0, 0);
        index.patient_id = "p".into();
        index.episode_id = "e1".into();
        let mut inside = event("inside", "outcome", Some("false"), None, 2, 2);
        inside.patient_id = "p".into();
        inside.episode_id = "e1".into();
        let mut outside = event("outside", "outcome", Some("true"), None, 9, 9);
        outside.patient_id = "p".into();
        outside.episode_id = "e1".into();
        let fixture = Fixture {
            schema_version: "2".into(),
            events: vec![index, inside, outside],
            observations: vec![Observation {
                patient_id: "p".into(),
                episode_id: "e1".into(),
                start_day: 0,
                end_day: Some(5),
                censoring_reason: Some("transfer".into()),
            }],
        };
        let run = run(&study(), &fixture).unwrap();
        assert_eq!(
            run.cohort[0].values["outcome"],
            ValueState::Observed("false".into())
        );
        assert_eq!(run.cohort[0].observation_end_day, Some(5));
        let evaluated = temporal_cohort(&study(), &fixture, &TemporalConfig { windows: vec![] });
        assert_eq!(evaluated.observation_incomplete_episodes, 1);
        assert_eq!(evaluated.observed_duration_days.counts["5"], 1);
        assert_eq!(evaluated.censoring_reasons["transfer"], 1);
        assert_eq!(evaluated.events_outside_observation, 1);
        let mut bad = fixture.clone();
        bad.observations[0].episode_id = "missing".into();
        assert!(validate(&study(), &bad).is_err());
        let mut bad = fixture.clone();
        bad.observations[0].end_day = Some(-1);
        assert!(validate(&study(), &bad).is_err());
    }
    fn conditional_config() -> GeneratorConfig {
        serde_yaml::from_str(
            r#"
schema_version: "2"
seed: 42
episode_count: 100
variables:
  - name: index
    outcomes:
      - {value: "true", probability: 0.5, event_day: 0, recording_day: 0}
      - {value: "false", probability: 0.5, event_day: 0, recording_day: 0}
  - name: outcome
    parents: [index]
    table:
      - when: {index: "true"}
        outcomes:
          - {value: "true", probability: 1.0, event_day: 1, recording_day: 1}
          - {value: "false", probability: 0.0, event_day: 1, recording_day: 1}
      - when: {index: "false"}
        outcomes:
          - {value: "true", probability: 0.0, event_day: 1, recording_day: 1}
          - {value: "false", probability: 1.0, event_day: 1, recording_day: 1}
"#,
        )
        .unwrap()
    }
    #[test]
    fn conditional_generation_selects_matching_distribution() {
        let config = conditional_config();
        let fixture = generate(&study(), &config).unwrap();
        validate(&study(), &fixture).unwrap();
        assert_eq!(run(&study(), &fixture).unwrap().cohort.len(), 100);
        for events in fixture.events.as_chunks::<2>().0 {
            assert_eq!(events[0].value, events[1].value);
            assert_eq!(events[1].provenance, "configured-generator-v2");
        }
        assert!(fixture
            .events
            .as_chunks::<2>()
            .0
            .iter()
            .any(|e| e[0].value.as_deref() == Some("true")));
        assert!(fixture
            .events
            .as_chunks::<2>()
            .0
            .iter()
            .any(|e| e[0].value.as_deref() == Some("false")));
    }
    #[test]
    fn parent_domains_union_all_conditional_rows_and_cross_product() {
        let mut s = study();
        s.variables.push(Variable {
            name: "third".into(),
            kind: "boolean".into(),
            role: None,
        });
        let mut c = conditional_config();
        c.variables[1].table[0]
            .outcomes
            .retain(|o| o.value == "true");
        c.variables[1].table[1]
            .outcomes
            .retain(|o| o.value == "false");
        let rows = [
            ("true", "true"),
            ("true", "false"),
            ("false", "true"),
            ("false", "false"),
        ]
        .iter()
        .map(|(a, b)| ConditionalDistribution {
            when: [
                ("index".into(), (*a).into()),
                ("outcome".into(), (*b).into()),
            ]
            .into_iter()
            .collect(),
            outcomes: vec![GeneratedOutcome {
                value: (*b).into(),
                probability: 1.0,
                event_day: 2,
                recording_day: 2,
                bin_end_day: None,
            }],
        })
        .collect();
        c.variables.push(GeneratedVariable {
            name: "third".into(),
            parents: vec!["index".into(), "outcome".into()],
            outcomes: vec![],
            table: rows,
        });
        let fixture = generate(&s, &c).unwrap();
        assert!(fixture
            .events
            .as_chunks::<3>()
            .0
            .iter()
            .all(|e| e[1].value == e[2].value));
        c.variables[2].table.pop();
        assert!(generate(&s, &c).is_err());
    }
    #[test]
    fn conditional_tables_reject_invalid_inputs() {
        let base = conditional_config();
        let mut variants = Vec::new();
        let mut c = base.clone();
        c.variables[1].table.pop();
        variants.push(c);
        let mut c = base.clone();
        c.variables[1].table[1] = c.variables[1].table[0].clone();
        variants.push(c);
        let mut c = base.clone();
        c.variables[1].table[0]
            .when
            .insert("index".into(), "invalid".into());
        variants.push(c);
        let mut c = base.clone();
        c.variables[1].table[0]
            .when
            .insert("extra".into(), "true".into());
        variants.push(c);
        let mut c = base.clone();
        c.variables[1].parents = vec!["unknown".into()];
        variants.push(c);
        let mut c = base.clone();
        c.variables[0].parents = vec!["outcome".into()];
        variants.push(c);
        let mut c = base.clone();
        c.variables[1].parents.push("index".into());
        variants.push(c);
        let mut c = base.clone();
        c.variables[1].table[0].outcomes[0].value = "invalid".into();
        variants.push(c);
        let mut c = base.clone();
        c.variables[1].table[0].outcomes[0].probability = f64::NAN;
        variants.push(c);
        let mut c = base.clone();
        c.variables[1].table[0].outcomes[0].probability = 0.5;
        variants.push(c);
        let mut c = base.clone();
        c.variables[1].table[0].outcomes[0].recording_day = 0;
        variants.push(c);
        let mut c = base.clone();
        c.variables[1].table[0].outcomes[1].value = "true".into();
        variants.push(c);
        let mut c = base.clone();
        c.variables.remove(0);
        variants.push(c);
        let mut c = base.clone();
        c.variables.push(c.variables[0].clone());
        variants.push(c);
        let mut c = base.clone();
        c.schema_version = "1".into();
        variants.push(c);
        for (i, config) in variants.iter().enumerate() {
            assert!(generate(&study(), config).is_err(), "variant {i}");
        }
        let mut c = base;
        c.variables.remove(0);
        c.variables[0].parents.clear();
        c.variables[0].outcomes = c.variables[0].table.remove(0).outcomes;
        c.variables[0].table.clear();
        assert!(generate(&study(), &c)
            .unwrap_err()
            .to_string()
            .contains("index"));
    }
    #[test]
    fn configured_generation_validates_and_repeats() {
        let mut s = study();
        s.eligibility.push(Rule {
            variable: "index".into(),
            equals: "true".into(),
        });
        let mut c = GeneratorConfig {
            schema_version: "1".into(),
            seed: 42,
            episode_count: 100,
            complete_observation: false,
            repeated_events: vec![],
            evidence: BTreeMap::new(),
            registry_fingerprint: None,
            variables: vec![GeneratedVariable {
                name: "index".into(),
                parents: vec![],
                table: vec![],
                outcomes: vec![
                    GeneratedOutcome {
                        value: "true".into(),
                        probability: 0.5,
                        event_day: 0,
                        recording_day: 0,
                        bin_end_day: None,
                    },
                    GeneratedOutcome {
                        value: "false".into(),
                        probability: 0.5,
                        event_day: 0,
                        recording_day: 0,
                        bin_end_day: None,
                    },
                ],
            }],
        };
        let a = generate(&s, &c).unwrap();
        assert_eq!(
            serde_yaml::to_string(&a).unwrap(),
            serde_yaml::to_string(&generate(&s, &c).unwrap()).unwrap()
        );
        let report = run(&s, &a).unwrap();
        assert!(!report.cohort.is_empty() && !report.exclusions.is_empty());
        assert!(a
            .events
            .iter()
            .all(|e| e.provenance == "configured-generator-v1"));
        c.seed += 1;
        assert_ne!(
            serde_yaml::to_string(&a).unwrap(),
            serde_yaml::to_string(&generate(&s, &c).unwrap()).unwrap()
        );
        c.variables[0].outcomes[0].probability = -0.1;
        assert!(generate(&s, &c).is_err());
        c.variables[0].outcomes[0].probability = 0.5;
        c.variables[0].outcomes[0].value = "invalid".into();
        assert!(generate(&s, &c).is_err());
        c.variables[0].outcomes[0].value = "true".into();
        c.variables[0].outcomes[0].recording_day = -1;
        assert!(generate(&s, &c).is_err());
        c.variables[0].name = "unknown".into();
        assert!(generate(&s, &c).is_err());
        c.variables.clear();
        assert!(generate(&s, &c).is_err());
    }

    #[test]
    fn repeated_generation_bounds_replay_and_validation() {
        let mut s = study();
        s.schema_version = "2".into();
        s.follow_up_days = 10;
        s.variables[0].role = Some(VariableRole::IndexContext);
        s.variables[1].role = Some(VariableRole::FollowUpOutcome);
        let c: GeneratorConfig = serde_yaml::from_str(
            r#"
schema_version: "4"
seed: 22
episode_count: 4
complete_observation: true
variables:
  - name: index
    outcomes: [{value: "true", probability: 1.0, event_day: 0, recording_day: 0}]
  - name: outcome
    outcomes:
      - {value: "true", probability: 0.5, event_day: 1, recording_day: 1}
      - {value: "false", probability: 0.5, event_day: 1, recording_day: 1}
repeated_events:
  - variable: outcome
    counts: [{count: 100, probability: 1.0}]
    start_day: 0
    end_day: 10
"#,
        )
        .unwrap();
        let a = generate(&s, &c).unwrap();
        assert_eq!(
            a.events.iter().filter(|e| e.variable == "outcome").count(),
            400
        );
        assert_eq!(
            serde_yaml::to_string(&a).unwrap(),
            serde_yaml::to_string(&generate(&s, &c).unwrap()).unwrap()
        );
        assert!(a
            .events
            .iter()
            .filter(|e| e.variable == "outcome")
            .all(|e| (0..=10).contains(&e.event_day) && e.event_day == e.recording_day));
        assert_eq!(a.observations.len(), 4);
        assert_eq!(a.events.iter().filter(|e| e.variable == "index").count(), 4);
        let groups: std::collections::HashSet<_> = a
            .events
            .iter()
            .filter(|e| e.variable == "outcome")
            .map(|e| e.id.as_str())
            .collect();
        let mut multiple = c.clone();
        multiple.repeated_events.push(RepeatedEventConfig {
            variable: "other".into(),
            counts: vec![GeneratedCount {
                count: 1,
                probability: 1.0,
            }],
            start_day: 0,
            end_day: 10,
        });
        multiple.variables.push(GeneratedVariable {
            name: "other".into(),
            parents: vec![],
            table: vec![],
            outcomes: vec![GeneratedOutcome {
                value: "true".into(),
                probability: 1.0,
                event_day: 1,
                recording_day: 1,
                bin_end_day: None,
            }],
        });
        s.variables.push(Variable {
            name: "other".into(),
            kind: "boolean".into(),
            role: Some(VariableRole::FollowUpOutcome),
        });
        let multiple_fixture = generate(&s, &multiple).unwrap();
        let ids: std::collections::HashSet<_> = multiple_fixture
            .events
            .iter()
            .map(|e| e.id.as_str())
            .collect();
        assert_eq!(ids.len(), multiple_fixture.events.len());
        assert!(groups.len() <= 400);
        let eval:EvaluationConfig=serde_yaml::from_str("schema_version: '4'\nlongitudinal:\n  start_day: 0\n  end_day: 10\n  variables: [index, outcome]\n  pairs:\n    sequence: {exposure: index, outcome: outcome}\n").unwrap();
        let report = evaluate(&s, &a, &a, &eval).unwrap();
        assert_eq!(
            report.longitudinal.unwrap().variable_counts["outcome"]["multiple"].absolute_difference,
            Some(0.0)
        );
        let mut zero = c.clone();
        zero.repeated_events[0].counts = vec![GeneratedCount {
            count: 0,
            probability: 1.0,
        }];
        let z = generate(&s, &zero).unwrap();
        assert_eq!(
            z.events.iter().filter(|e| e.variable == "outcome").count(),
            0
        );
        let zreport = evaluate(&s, &z, &a, &eval).unwrap();
        assert_ne!(
            zreport.longitudinal.unwrap().variable_counts["outcome"]["zero"].absolute_difference,
            Some(0.0)
        );
        let mut unknown = zero.clone();
        unknown.complete_observation = false;
        let unknown_fixture = generate(&s, &unknown).unwrap();
        assert!(unknown_fixture.observations.is_empty());
        let unknown_run = run(&s, &unknown_fixture).unwrap();
        let longitudinal = longitudinal_episodes(
            eval.longitudinal.as_ref().unwrap(),
            &unknown_fixture,
            &unknown_run.cohort,
        );
        assert!(longitudinal
            .iter()
            .all(|episode| episode.coverage == "unknown"));
        let mut bad = c.clone();
        bad.repeated_events[0].end_day = 11;
        assert!(generate(&s, &bad).is_err());
        let mut bad = c.clone();
        bad.repeated_events[0].counts[0].count = 101;
        assert!(generate(&s, &bad).is_err());
        let mut bad = c.clone();
        bad.repeated_events[0].counts[0].probability = f64::NAN;
        assert!(generate(&s, &bad).is_err());
        let mut bad = c.clone();
        bad.repeated_events[0].counts[0].probability = f64::INFINITY;
        assert!(generate(&s, &bad).is_err());
        let mut bad = c.clone();
        bad.variables[1].outcomes[0].probability = f64::NAN;
        assert!(generate(&s, &bad).is_err());
        let mut bad = c.clone();
        bad.repeated_events[0].start_day = i64::MAX;
        bad.repeated_events[0].end_day = i64::MAX;
        assert!(generate(&s, &bad).is_err());
        let mut bad = c.clone();
        bad.repeated_events[0].variable = "index".into();
        assert!(generate(&s, &bad).is_err());
        let mut bad = c.clone();
        bad.repeated_events[0].start_day = -1;
        assert!(generate(&s, &bad).is_err());
        let mut bad = c.clone();
        bad.repeated_events[0].counts = vec![GeneratedCount {
            count: 0,
            probability: 0.9,
        }];
        assert!(generate(&s, &bad).is_err());
    }
    fn study() -> Study {
        Study {
            schema_version: "1".into(),
            name: "x".into(),
            unit: "episode".into(),
            variables: vec![
                Variable {
                    name: "index".into(),
                    kind: "boolean".into(),
                    role: None,
                },
                Variable {
                    name: "outcome".into(),
                    kind: "boolean".into(),
                    role: None,
                },
            ],
            index: "index".into(),
            eligibility: vec![],
            follow_up_days: 10,
            evidence: BTreeMap::new(),
        }
    }
    fn event(
        id: &str,
        var: &str,
        value: Option<&str>,
        missing: Option<&str>,
        day: i64,
        record: i64,
    ) -> Event {
        Event {
            patient_id: "p".into(),
            episode_id: "e".into(),
            id: id.into(),
            variable: var.into(),
            value: value.map(str::to_string),
            missing: missing.map(str::to_string),
            event_day: day,
            recording_day: record,
            provenance: "fixture".into(),
        }
    }
    #[test]
    fn boundary_and_leakage() {
        let f = Fixture {
            schema_version: "1".into(),
            events: vec![
                event("i", "index", Some("true"), None, 0, 0),
                event("o", "outcome", Some("true"), None, 10, 10),
                event("future", "outcome", Some("false"), None, 11, 11),
            ],
            observations: vec![],
        };
        let r = run(&study(), &f).unwrap();
        assert_eq!(
            r.cohort[0].values["outcome"],
            ValueState::Observed("true".into())
        );
        assert_eq!(r.summary.derived_values.observed, 2);
        assert!(r
            .integrity
            .iter()
            .any(|s| s.starts_with("eligibility_availability:")));
        assert!(r
            .integrity
            .iter()
            .any(|s| s.starts_with("derived_values_availability:")));
    }
    #[test]
    fn version_two_roles_enforce_availability_and_provenance() {
        let mut s = study();
        s.schema_version = "2".into();
        s.variables[0].role = Some(VariableRole::IndexContext);
        s.variables[1].role = Some(VariableRole::FollowUpOutcome);
        s.variables.push(Variable {
            name: "baseline".into(),
            kind: "boolean".into(),
            role: Some(VariableRole::Baseline),
        });
        s.eligibility.push(Rule {
            variable: "baseline".into(),
            equals: "true".into(),
        });
        let mut f = Fixture {
            schema_version: "2".into(),
            events: vec![
                event("i", "index", Some("true"), None, 0, 0),
                event("base", "baseline", Some("true"), None, -1, 0),
                event("late", "baseline", Some("false"), None, 0, 1),
                event("out", "outcome", Some("true"), None, 10, 10),
                event("outside", "outcome", Some("false"), None, 11, 11),
            ],
            observations: vec![Observation {
                patient_id: "p".into(),
                episode_id: "e".into(),
                start_day: -1,
                end_day: Some(10),
                censoring_reason: None,
            }],
        };
        let report = run(&s, &f).unwrap();
        let row = &report.cohort[0];
        assert_eq!(row.values["baseline"], ValueState::Observed("true".into()));
        assert_eq!(
            row.variable_source_event_ids["baseline"].as_deref(),
            Some("base")
        );
        assert_eq!(row.values["outcome"], ValueState::Observed("true".into()));
        assert_eq!(
            row.variable_source_event_ids["outcome"].as_deref(),
            Some("out")
        );
        f.observations[0].start_day = 0;
        assert!(run(&s, &f).unwrap().cohort.is_empty());
        s.variables[2].role = None;
        assert!(validate(&s, &f)
            .unwrap_err()
            .to_string()
            .contains("requires a role"));
        s.schema_version = "1".into();
        assert!(validate(&s, &f)
            .unwrap_err()
            .to_string()
            .contains("roles require"));
    }

    #[test]
    fn version_two_context_and_missing_values_are_distinct() {
        let mut s = study();
        s.schema_version = "2".into();
        s.variables[0].role = Some(VariableRole::IndexContext);
        s.variables[1].role = Some(VariableRole::FollowUpOutcome);
        s.variables.push(Variable {
            name: "context".into(),
            kind: "boolean".into(),
            role: Some(VariableRole::IndexContext),
        });
        s.variables.push(Variable {
            name: "baseline".into(),
            kind: "boolean".into(),
            role: Some(VariableRole::Baseline),
        });
        let fixture = Fixture {
            schema_version: "2".into(),
            observations: vec![],
            events: vec![
                event("index", "index", Some("true"), None, 0, 0),
                event("early", "context", Some("true"), None, -1, -1),
                event("late", "context", Some("true"), None, 0, 1),
                event("missing", "baseline", None, Some("unknown"), 0, 0),
                event("before", "outcome", Some("true"), None, -1, -1),
                event("boundary", "outcome", Some("false"), None, 0, 0),
            ],
        };
        let run = run(&s, &fixture).unwrap();
        let row = &run.cohort[0];
        assert_eq!(row.values["context"], ValueState::Absent);
        assert_eq!(row.variable_source_event_ids["context"], None);
        assert_eq!(
            row.values["baseline"],
            ValueState::Missing("unknown".into())
        );
        assert_eq!(
            row.variable_source_event_ids["baseline"].as_deref(),
            Some("missing")
        );
        assert_eq!(row.values["outcome"], ValueState::Observed("false".into()));
        let mut invalid = s.clone();
        invalid.eligibility.push(Rule {
            variable: "outcome".into(),
            equals: "true".into(),
        });
        assert!(validate(&invalid, &fixture)
            .unwrap_err()
            .to_string()
            .contains("eligibility variable"));
        invalid = s.clone();
        invalid.variables[0].role = Some(VariableRole::Baseline);
        assert!(validate(&invalid, &fixture)
            .unwrap_err()
            .to_string()
            .contains("index variable"));
    }
    #[test]
    fn typed_values_and_eligibility_operands_are_validated() {
        for (kind, valid, invalid) in [
            ("boolean", "true", "TRUE"),
            ("integer", "-42", "1.5"),
            ("number", "1.5e2", "not-a-number"),
            ("number", "0", "NaN"),
            ("number", "0", "inf"),
        ] {
            let mut s = study();
            s.variables[1].kind = kind.into();
            let mut f = Fixture {
                schema_version: "1".into(),
                events: vec![
                    event("i", "index", Some("true"), None, 0, 0),
                    event("v", "outcome", Some(valid), None, 0, 0),
                ],
                observations: vec![],
            };
            assert!(validate(&s, &f).is_ok(), "{kind}: {valid}");
            f.events[1].value = Some(invalid.into());
            assert!(validate(&s, &f)
                .unwrap_err()
                .to_string()
                .contains("event v has invalid"));
            f.events[1].value = Some(valid.into());
            s.eligibility.push(Rule {
                variable: "outcome".into(),
                equals: invalid.into(),
            });
            assert!(validate(&s, &f)
                .unwrap_err()
                .to_string()
                .contains("eligibility for outcome has invalid"));
        }
        let mut s = study();
        s.variables[1].kind = "string".into();
        let f = Fixture {
            schema_version: "1".into(),
            events: vec![event("v", "outcome", Some("arbitrary text"), None, 0, 0)],
            observations: vec![],
        };
        assert!(validate(&s, &f).is_ok());
    }
    #[test]
    fn eligibility_baseline_and_follow_up_are_both_retained() {
        let mut s = study();
        s.eligibility.push(Rule {
            variable: "outcome".into(),
            equals: "true".into(),
        });
        let f = Fixture {
            schema_version: "1".into(),
            events: vec![
                event("i", "index", Some("true"), None, 0, 0),
                event("baseline", "outcome", Some("true"), None, -1, 0),
                event("late_recorded", "outcome", Some("false"), None, 0, 1),
                event("followup", "outcome", Some("false"), None, 10, 10),
            ],
            observations: vec![],
        };
        let report = run(&s, &f).unwrap();
        let row = &report.cohort[0];
        assert_eq!(row.values["outcome"], ValueState::Observed("false".into()));
        assert_eq!(
            row.eligibility_values["outcome"].value,
            ValueState::Observed("true".into())
        );
        assert_eq!(
            row.eligibility_values["outcome"].source_event_id.as_deref(),
            Some("baseline")
        );
        assert!(row.source_event_ids.contains(&"followup".to_string()));
        assert_eq!(
            report
                .diagnostics
                .iter()
                .find(|d| d.metric == "eligibility_future_information_leakage")
                .unwrap()
                .numerator,
            0
        );
    }
    #[test]
    fn version_one_eligibility_respects_observation_start() {
        let mut s = study();
        s.eligibility.push(Rule {
            variable: "outcome".into(),
            equals: "true".into(),
        });
        let mut fixture = Fixture {
            schema_version: "2".into(),
            events: vec![
                event("i", "index", Some("true"), None, 0, 0),
                event("baseline", "outcome", Some("true"), None, -1, 0),
            ],
            observations: vec![Observation {
                patient_id: "p".into(),
                episode_id: "e".into(),
                start_day: 0,
                end_day: Some(10),
                censoring_reason: None,
            }],
        };
        let report = run(&s, &fixture).unwrap();
        assert!(report.cohort.is_empty());
        assert_eq!(report.exclusions[0].reason, "eligibility:outcome != true");
        fixture.observations[0].start_day = -1;
        let report = run(&s, &fixture).unwrap();
        assert_eq!(report.cohort.len(), 1);
        assert_eq!(
            report.cohort[0].eligibility_values["outcome"]
                .source_event_id
                .as_deref(),
            Some("baseline")
        );
        fixture.observations.clear();
        assert_eq!(run(&s, &fixture).unwrap().cohort.len(), 1);
    }
    #[test]
    fn missing_differs_from_absent() {
        let f = Fixture {
            schema_version: "1".into(),
            events: vec![
                event("i", "index", Some("true"), None, 0, 0),
                event("m", "outcome", None, Some("not_recorded"), 1, 1),
            ],
            observations: vec![],
        };
        let r = run(&study(), &f).unwrap();
        assert_eq!(
            r.cohort[0].values["outcome"],
            ValueState::Missing("not_recorded".into())
        );
    }
    #[test]
    fn duplicate_ids_rejected() {
        let mut f = Fixture {
            schema_version: "1".into(),
            events: vec![
                event("i", "index", Some("true"), None, 0, 0),
                event("i", "outcome", Some("true"), None, 0, 0),
            ],
            observations: vec![],
        };
        assert!(validate(&study(), &f).is_err());
        f.events[1].id = "o".into();
        f.events[1].recording_day = -1;
        assert!(validate(&study(), &f).is_err());
    }
    #[test]
    fn multiple_index_events_per_episode_rejected() {
        let f = Fixture {
            schema_version: "1".into(),
            events: vec![
                event("i1", "index", Some("true"), None, 0, 0),
                event("i2", "index", Some("true"), None, 2, 2),
            ],
            observations: vec![],
        };
        assert!(validate(&study(), &f)
            .unwrap_err()
            .to_string()
            .contains("multiple index events"));
    }
    #[test]
    fn follow_up_overflow_is_actionable_error() {
        let mut s = study();
        s.follow_up_days = i64::MAX;
        let f = Fixture {
            schema_version: "1".into(),
            events: vec![event("i", "index", Some("true"), None, 1, 1)],
            observations: vec![],
        };
        assert!(run(&s, &f)
            .unwrap_err()
            .to_string()
            .contains("follow-up boundary overflows"));
    }
    #[test]
    fn summary_reconciles_episodes_and_value_states() {
        let mut s = study();
        s.eligibility.push(Rule {
            variable: "index".into(),
            equals: "true".into(),
        });
        let mut excluded = event("i2", "index", Some("false"), None, 0, 0);
        excluded.episode_id = "e2".into();
        let mut no_index = event("o3", "outcome", Some("true"), None, 0, 0);
        no_index.episode_id = "e3".into();
        let f = Fixture {
            schema_version: "1".into(),
            events: vec![
                event("i", "index", Some("true"), None, 0, 0),
                event("m", "outcome", None, Some("unknown"), 10, 10),
                excluded,
                no_index,
            ],
            observations: vec![],
        };
        let r = run(&s, &f).unwrap();
        assert_eq!(
            (
                r.summary.source_events,
                r.summary.distinct_patients,
                r.summary.distinct_episodes,
                r.summary.episodes_with_index
            ),
            (4, 1, 3, 2)
        );
        assert_eq!(
            (r.summary.included_rows, r.summary.excluded_episodes),
            (1, 2)
        );
        assert_eq!(r.summary.exclusion_reasons["missing_index"], 1);
        assert_eq!(r.summary.exclusion_reasons["eligibility:index != true"], 1);
        assert_eq!(
            r.summary.derived_values,
            ValueCounts {
                observed: 1,
                missing: 1,
                absent: 0
            }
        );
        let empty = run(
            &s,
            &Fixture {
                schema_version: "1".into(),
                events: vec![],
                observations: vec![],
            },
        )
        .unwrap();
        assert_eq!(empty.summary, CohortSummary::default());
        let absent = run(
            &study(),
            &Fixture {
                schema_version: "1".into(),
                events: vec![event("i", "index", Some("true"), None, 0, 0)],
                observations: vec![],
            },
        )
        .unwrap();
        assert_eq!(absent.summary.derived_values.absent, 1);
        assert_eq!(
            serde_yaml::to_string(&r.summary).unwrap(),
            serde_yaml::to_string(&run(&s, &f).unwrap().summary).unwrap()
        );
        assert_eq!(
            serde_yaml::to_string(&r.diagnostics).unwrap(),
            serde_yaml::to_string(&run(&s, &f).unwrap().diagnostics).unwrap()
        );
    }
    #[test]
    fn diagnostics_find_invalid_inputs_and_references() {
        let mut s = study();
        s.eligibility.push(Rule {
            variable: "outcome".into(),
            equals: "true".into(),
        });
        let mut invalid = event("bad", "outcome", Some("true"), None, -1, 1);
        invalid.provenance.clear();
        let f = Fixture {
            schema_version: "1".into(),
            events: vec![
                event("i", "index", Some("true"), None, 0, 0),
                invalid,
                event("early", "outcome", Some("true"), None, 3, 2),
            ],
            observations: vec![],
        };
        let row = CohortRow {
            patient_id: "p".into(),
            episode_id: "e".into(),
            index_day: 0,
            values: BTreeMap::new(),
            eligibility_values: BTreeMap::new(),
            source_event_ids: vec!["unknown".into(), "bad".into()],
            variable_source_event_ids: BTreeMap::new(),
            observation_start_day: None,
            observation_end_day: None,
            censoring_reason: None,
        };
        let d = diagnostics(&s, &f, &[row], &[]);
        for name in [
            "recorded_before_event",
            "missing_provenance",
            "unresolved_source_references",
            "inconsistent_source_provenance",
            "eligibility_late_recorded_candidates",
            "eligibility_future_information_leakage",
        ] {
            assert_eq!(
                d.iter().find(|x| x.metric == name).unwrap().numerator,
                1,
                "{name}"
            );
        }
    }
}

pub mod tabular;
