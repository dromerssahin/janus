//! Unified in-memory candidate comparison harness.
//!
//! All training, reference, and synthetic patient rows as well as fitted models stay strictly
//! in memory. No file I/O is performed by this module, and [`AggregateComparisonManifest`]
//! emits only aggregate fidelity/utility metrics, shared denominators, fingerprints, and
//! bounded interpretation limits extending [`CandidateComparisonManifest`].

use crate::{
    coherence::{CandidateComparisonManifest, ProtocolCandidate},
    neural::{run_neural_bridge_in_memory, NeuralGeneratorConfig, NeuralModelType},
    software_identity, yaml_fingerprint, Error,
};
use serde::{Deserialize, Serialize};
use std::{
    collections::{BTreeMap, BTreeSet, HashSet},
    path::PathBuf,
};

/// Module result alias using [`crate::Error`].
pub type Result<T> = std::result::Result<T, Error>;

/// In-memory patient row mapping variable names to string tokens.
///
/// Blank strings (`""`) and `"NA"` (case-insensitive) are treated as missing values, never as
/// utility labels. Rows must remain in memory and are never serialized into comparison manifests.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(deny_unknown_fields)]
pub struct Row {
    pub fields: BTreeMap<String, String>,
}

impl Row {
    pub fn new(fields: BTreeMap<String, String>) -> Self {
        Self { fields }
    }

    pub fn from_pairs<I, K, V>(pairs: I) -> Self
    where
        I: IntoIterator<Item = (K, V)>,
        K: Into<String>,
        V: Into<String>,
    {
        Self {
            fields: pairs
                .into_iter()
                .map(|(k, v)| (k.into(), v.into()))
                .collect(),
        }
    }

    pub fn get(&self, variable: &str) -> Option<&str> {
        self.fields.get(variable).map(String::as_str)
    }
}

/// Check whether a raw token represents a missing value (`""` or ASCII case-insensitive `"NA"`).
pub fn token_is_missing(raw: &str) -> bool {
    let trimmed = raw.trim();
    trimmed.is_empty() || trimmed.eq_ignore_ascii_case("NA")
}

/// Canonicalize a cell token so blank strings and `"NA"` variants map to a single missing state
/// while preserving observed trimmed tokens.
fn canonical_state(raw: &str) -> String {
    let trimmed = raw.trim();
    if token_is_missing(trimmed) {
        "MISSING:NA".into()
    } else {
        format!("OBS:{trimmed}")
    }
}

/// In-memory fitted model representation produced by [`CohortGenerator::fit`].
///
/// Deliberately omits `Serialize` and `Deserialize` so fitted model state and training rows
/// cannot be written to disk.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FittedModel {
    pub candidate_name: String,
    pub variable_order: Vec<String>,
    pub parent_indices: Vec<Vec<usize>>,
    pub training_rows: Vec<Row>,
}

/// Per-seed sampled rows and per-variable sparse parent backoff counts.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct BatchSampleResult {
    pub draws: BTreeMap<u64, Vec<Row>>,
    pub sparse_backoff_counts: BTreeMap<u64, BTreeMap<String, usize>>,
}

/// Unified in-memory cohort generator trait.
///
/// Implementations must operate strictly in memory without performing any file I/O.
pub trait CohortGenerator {
    fn fit(&self, training_rows: &[Row]) -> Result<FittedModel>;
    fn sample(&self, fitted: &FittedModel, count: usize, seed: u64) -> Result<Vec<Row>>;

    fn sample_batch(
        &self,
        fitted: &FittedModel,
        count: usize,
        seeds: &[u64],
    ) -> Result<BTreeMap<u64, Vec<Row>>> {
        let mut out = BTreeMap::new();
        for &seed in seeds {
            out.insert(seed, self.sample(fitted, count, seed)?);
        }
        Ok(out)
    }

    fn sample_batch_with_backoffs(
        &self,
        fitted: &FittedModel,
        count: usize,
        seeds: &[u64],
    ) -> Result<BatchSampleResult> {
        Ok(BatchSampleResult {
            draws: self.sample_batch(fitted, count, seeds)?,
            sparse_backoff_counts: BTreeMap::new(),
        })
    }
}

fn default_schema_version() -> String {
    "1".into()
}

fn default_protocol_id() -> String {
    "candidate-comparison-protocol-v1".into()
}

fn default_contract_id() -> String {
    "coherence-contract-v1".into()
}

fn default_preprocessing() -> String {
    "shared_in_memory_split_and_schema".into()
}

fn default_candidate_status() -> String {
    "baseline".into()
}

fn default_populations() -> Vec<String> {
    vec![
        "training_development".into(),
        "reference_evaluation".into(),
        "same_seed_synthetic".into(),
    ]
}

fn default_alpha() -> f64 {
    1.0
}

fn default_true() -> bool {
    true
}

/// Binary predictive utility task configuration evaluated via Brier score on reference rows.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct UtilityBrierConfig {
    pub task_name: String,
    pub predictor: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub additional_predictors: Vec<String>,
    pub target: String,
    pub positive_values: Vec<String>,
    pub negative_values: Vec<String>,
    #[serde(default = "default_alpha")]
    pub alpha: f64,
    #[serde(default = "default_true")]
    pub predictor_available_before_target: bool,
}

impl UtilityBrierConfig {
    pub fn validate(&self) -> Result<()> {
        if self.task_name.trim().is_empty()
            || self.predictor.trim().is_empty()
            || self.target.trim().is_empty()
            || !self.predictor_available_before_target
            || !self.alpha.is_finite()
            || self.alpha <= 0.0
            || self.alpha > f64::MAX / 4.0
            || self.positive_values.is_empty()
            || self.negative_values.is_empty()
        {
            return Err(Error::Input("invalid utility Brier configuration".into()));
        }
        let mut predictors = HashSet::new();
        for pred in std::iter::once(&self.predictor).chain(self.additional_predictors.iter()) {
            if pred.trim().is_empty() || pred == &self.target || !predictors.insert(pred) {
                return Err(Error::Input(
                    "utility Brier predictors must be non-empty, unique, and distinct from target"
                        .into(),
                ));
            }
        }
        let mut labels = HashSet::new();
        for label in self
            .positive_values
            .iter()
            .chain(self.negative_values.iter())
        {
            if token_is_missing(label) || !labels.insert(label.trim()) {
                return Err(Error::Input(
                    "utility Brier labels must be non-missing and disjoint".into(),
                ));
            }
        }
        Ok(())
    }
}

/// Metric selection for [`CandidateComparisonConfig`].
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct CandidateMetricsConfig {
    pub marginal_tvd: bool,
    pub pairwise_tvd: bool,
    pub joint_tvd: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub utility_brier: Option<UtilityBrierConfig>,
}

impl CandidateMetricsConfig {
    pub fn validate(&self) -> Result<()> {
        if !self.marginal_tvd
            && !self.pairwise_tvd
            && !self.joint_tvd
            && self.utility_brier.is_none()
        {
            return Err(Error::Input(
                "candidate comparison requires at least one enabled metric".into(),
            ));
        }
        if let Some(utility) = &self.utility_brier {
            utility.validate()?;
        }
        Ok(())
    }

    pub fn enabled_metric_names(&self) -> Vec<String> {
        let mut names = Vec::new();
        if self.marginal_tvd {
            names.push("marginal_tvd".into());
        }
        if self.pairwise_tvd {
            names.push("pairwise_tvd".into());
        }
        if self.joint_tvd {
            names.push("joint_tvd".into());
        }
        if self.utility_brier.is_some() {
            names.push("utility_brier".into());
        }
        names
    }
}

/// Configuration for running a unified multi-candidate comparison across shared splits and seeds.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct CandidateComparisonConfig {
    #[serde(default = "default_schema_version")]
    pub schema_version: String,
    #[serde(default = "default_protocol_id")]
    pub protocol_id: String,
    #[serde(default = "default_contract_id")]
    pub contract_id: String,
    #[serde(default = "default_preprocessing")]
    pub preprocessing: String,
    #[serde(default = "default_candidate_status")]
    pub candidate_status: String,
    #[serde(default = "default_populations")]
    pub populations: Vec<String>,
    pub candidate_names: Vec<String>,
    pub split_fingerprint: String,
    pub sample_count_per_seed: usize,
    pub seeds: Vec<u64>,
    pub metrics: CandidateMetricsConfig,
}

impl CandidateComparisonConfig {
    pub fn validate(&self) -> Result<()> {
        if self.schema_version != "1"
            || self.protocol_id.trim().is_empty()
            || self.contract_id.trim().is_empty()
            || self.preprocessing.trim().is_empty()
            || self.candidate_status.trim().is_empty()
            || self.split_fingerprint.trim().is_empty()
            || self.candidate_names.len() < 2
            || self.sample_count_per_seed == 0
            || self.sample_count_per_seed > 1_000_000
            || self.seeds.is_empty()
            || self.populations.is_empty()
            || self.populations.iter().any(|p| p.trim().is_empty())
            || self.candidate_names.iter().any(|n| n.trim().is_empty())
        {
            return Err(Error::Input(
                "incomplete or invalid candidate comparison configuration".into(),
            ));
        }
        let unique_names: HashSet<_> = self.candidate_names.iter().collect();
        if unique_names.len() != self.candidate_names.len() {
            return Err(Error::Input(
                "candidate comparison configuration has duplicate candidate names".into(),
            ));
        }
        let unique_seeds: HashSet<_> = self.seeds.iter().copied().collect();
        if unique_seeds.len() != self.seeds.len() {
            return Err(Error::Input(
                "candidate comparison configuration has duplicate seeds".into(),
            ));
        }
        self.metrics.validate()?;
        Ok(())
    }

    /// Build the protocol-level [`CandidateComparisonManifest`] corresponding to this configuration.
    pub fn to_manifest(&self) -> Result<CandidateComparisonManifest> {
        self.validate()?;
        let manifest = CandidateComparisonManifest {
            schema_version: self.schema_version.clone(),
            protocol_id: self.protocol_id.clone(),
            contract_id: self.contract_id.clone(),
            candidates: self
                .candidate_names
                .iter()
                .map(|name| ProtocolCandidate {
                    name: name.clone(),
                    preprocessing: self.preprocessing.clone(),
                    status: self.candidate_status.clone(),
                })
                .collect(),
            mapping_fingerprint: self.split_fingerprint.clone(),
            populations: self.populations.clone(),
            sample_count: self.sample_count_per_seed,
            seeds: self.seeds.clone(),
            metrics: self.metrics.enabled_metric_names(),
        };
        manifest.validate()?;
        Ok(manifest)
    }
}

/// Shared evaluation denominators verified to be identical across all candidates and seeds.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ComparisonDenominators {
    pub training_rows: usize,
    pub reference_rows: usize,
    pub generated_rows: usize,
    pub marginal_reference_observed: BTreeMap<String, usize>,
    pub marginal_reference_missing: BTreeMap<String, usize>,
    pub pairwise_reference_complete: BTreeMap<String, usize>,
    pub joint_reference_rows: usize,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub utility_training_labeled: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub utility_reference_labeled: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub utility_reference_unlabeled: Option<usize>,
}

/// Utility Brier score details for a single seed draw.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct UtilityBrierSummary {
    pub task_name: String,
    pub real_training_brier: Option<f64>,
    pub synthetic_training_brier: Option<f64>,
    pub synthetic_training_labeled: usize,
    pub reference_labeled: usize,
    pub reference_unlabeled: usize,
}

/// Per-seed evaluation metrics for a single candidate.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct CandidateSeedMetrics {
    pub seed: u64,
    pub denominators: ComparisonDenominators,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub marginal_tvd_by_variable: BTreeMap<String, f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mean_marginal_tvd: Option<f64>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub pairwise_tvd_by_pair: BTreeMap<String, f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mean_pairwise_tvd: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub joint_tvd: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub utility_brier: Option<UtilityBrierSummary>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub sparse_backoff_counts: BTreeMap<String, usize>,
}

/// Seed-aggregated summary for a single candidate across all configured seeds.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct CandidateAggregateSummary {
    pub candidate_name: String,
    pub preprocessing: String,
    pub status: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mean_marginal_tvd: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mean_pairwise_tvd: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mean_joint_tvd: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub real_training_brier: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mean_synthetic_brier: Option<f64>,
    pub per_seed: Vec<CandidateSeedMetrics>,
}

/// Aggregate comparison manifest extending [`CandidateComparisonManifest`].
///
/// Contains only aggregate metrics, shared denominators, fingerprints, software identity,
/// and bounded limitations.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct AggregateComparisonManifest {
    pub manifest: CandidateComparisonManifest,
    pub split_fingerprint: String,
    pub training_fingerprint: String,
    pub reference_fingerprint: String,
    pub variables: Vec<String>,
    pub denominators: ComparisonDenominators,
    pub candidate_summaries: Vec<CandidateAggregateSummary>,
    pub software_version: String,
    pub dependency_lock_fingerprint: String,
    pub limitations: Vec<String>,
}

impl AggregateComparisonManifest {
    /// Validate both the underlying [`CandidateComparisonManifest`] and all shared-condition
    /// and identical-denominator invariants across candidates and seeds.
    pub fn validate(&self) -> Result<()> {
        self.manifest.validate()?;
        if self.split_fingerprint.trim().is_empty()
            || self.split_fingerprint != self.manifest.mapping_fingerprint
            || self.training_fingerprint.trim().is_empty()
            || self.reference_fingerprint.trim().is_empty()
            || self.software_version.trim().is_empty()
            || self.dependency_lock_fingerprint.trim().is_empty()
            || self.variables.is_empty()
            || self.variables.iter().any(|v| v.trim().is_empty())
            || self.denominators.training_rows == 0
            || self.denominators.reference_rows == 0
            || self.denominators.generated_rows != self.manifest.sample_count
            || self.denominators.joint_reference_rows != self.denominators.reference_rows
            || self.candidate_summaries.len() != self.manifest.candidates.len()
            || self.limitations.is_empty()
            || self.limitations.iter().any(|l| l.trim().is_empty())
        {
            return Err(Error::Input(
                "incomplete or inconsistent aggregate comparison manifest".into(),
            ));
        }

        for (summary, proto) in self
            .candidate_summaries
            .iter()
            .zip(self.manifest.candidates.iter())
        {
            if summary.candidate_name != proto.name
                || summary.preprocessing != proto.preprocessing
                || summary.status != proto.status
                || summary.per_seed.len() != self.manifest.seeds.len()
            {
                return Err(Error::Input(
                    "candidate aggregate summary does not match protocol manifest candidate".into(),
                ));
            }
            for (seed_metrics, &expected_seed) in
                summary.per_seed.iter().zip(self.manifest.seeds.iter())
            {
                if seed_metrics.seed != expected_seed
                    || seed_metrics.denominators != self.denominators
                {
                    return Err(Error::Input(
                        "candidate seed metrics violate shared seed or identical denominator contract"
                            .into(),
                    ));
                }
            }
        }
        Ok(())
    }
}

fn validate_cohort_rows(rows: &[Row], cohort_label: &str) -> Result<Vec<String>> {
    if rows.is_empty() {
        return Err(Error::Input(format!("{cohort_label} cohort is empty")));
    }
    let variables: Vec<String> = rows[0].fields.keys().cloned().collect();
    if variables.is_empty() || variables.iter().any(|v| v.trim().is_empty()) {
        return Err(Error::Input(format!(
            "{cohort_label} cohort has empty variable schema"
        )));
    }
    for (idx, row) in rows.iter().enumerate() {
        if row.fields.len() != variables.len()
            || variables.iter().any(|v| !row.fields.contains_key(v))
        {
            return Err(Error::Input(format!(
                "{cohort_label} row {idx} does not match expected variable schema"
            )));
        }
    }
    Ok(variables)
}

fn compute_denominators(
    training_rows: &[Row],
    reference_rows: &[Row],
    variables: &[String],
    sample_count: usize,
    utility_config: Option<&UtilityBrierConfig>,
) -> Result<ComparisonDenominators> {
    let mut marginal_reference_observed = BTreeMap::new();
    let mut marginal_reference_missing = BTreeMap::new();
    for var in variables {
        let mut obs = 0usize;
        let mut miss = 0usize;
        for row in reference_rows {
            let val = row.get(var).unwrap_or("");
            if token_is_missing(val) {
                miss += 1;
            } else {
                obs += 1;
            }
        }
        marginal_reference_observed.insert(var.clone(), obs);
        marginal_reference_missing.insert(var.clone(), miss);
    }

    let mut pairwise_reference_complete = BTreeMap::new();
    for i in 0..variables.len() {
        for j in (i + 1)..variables.len() {
            let a = &variables[i];
            let b = &variables[j];
            let complete = reference_rows
                .iter()
                .filter(|row| {
                    !token_is_missing(row.get(a).unwrap_or(""))
                        && !token_is_missing(row.get(b).unwrap_or(""))
                })
                .count();
            pairwise_reference_complete.insert(format!("{a}__{b}"), complete);
        }
    }

    let (utility_training_labeled, utility_reference_labeled, utility_reference_unlabeled) =
        if let Some(task) = utility_config {
            for pred in std::iter::once(&task.predictor).chain(task.additional_predictors.iter()) {
                if !variables.contains(pred) {
                    return Err(Error::Input(format!(
                        "utility predictor '{pred}' not found in cohort schema"
                    )));
                }
            }
            if !variables.contains(&task.target) {
                return Err(Error::Input(format!(
                    "utility target '{}' not found in cohort schema",
                    task.target
                )));
            }
            let is_labeled = |row: &Row| -> bool {
                let val = row.get(&task.target).unwrap_or("").trim();
                !token_is_missing(val)
                    && (task.positive_values.iter().any(|p| p.trim() == val)
                        || task.negative_values.iter().any(|n| n.trim() == val))
            };
            let train_labeled = training_rows.iter().filter(|r| is_labeled(r)).count();
            let ref_labeled = reference_rows.iter().filter(|r| is_labeled(r)).count();
            (
                Some(train_labeled),
                Some(ref_labeled),
                Some(reference_rows.len() - ref_labeled),
            )
        } else {
            (None, None, None)
        };

    Ok(ComparisonDenominators {
        training_rows: training_rows.len(),
        reference_rows: reference_rows.len(),
        generated_rows: sample_count,
        marginal_reference_observed,
        marginal_reference_missing,
        pairwise_reference_complete,
        joint_reference_rows: reference_rows.len(),
        utility_training_labeled,
        utility_reference_labeled,
        utility_reference_unlabeled,
    })
}

fn family_tvd(generated: &[Row], reference: &[Row], columns: &[&str]) -> f64 {
    let mut counts: BTreeMap<Vec<String>, (usize, usize)> = BTreeMap::new();
    for row in generated {
        let key: Vec<String> = columns
            .iter()
            .map(|c| canonical_state(row.get(c).unwrap_or("")))
            .collect();
        counts.entry(key).or_default().0 += 1;
    }
    for row in reference {
        let key: Vec<String> = columns
            .iter()
            .map(|c| canonical_state(row.get(c).unwrap_or("")))
            .collect();
        counts.entry(key).or_default().1 += 1;
    }
    let n_gen = generated.len() as f64;
    let n_ref = reference.len() as f64;
    0.5 * counts
        .values()
        .map(|(g, r)| (*g as f64 / n_gen - *r as f64 / n_ref).abs())
        .sum::<f64>()
}

fn compute_brier_score(
    train_rows: &[Row],
    reference_rows: &[Row],
    task: &UtilityBrierConfig,
) -> (usize, usize, Option<f64>) {
    let label_of = |row: &Row| -> Option<f64> {
        let raw = row.get(&task.target).unwrap_or("").trim();
        if token_is_missing(raw) {
            return None;
        }
        if task.positive_values.iter().any(|p| p.trim() == raw) {
            Some(1.0)
        } else if task.negative_values.iter().any(|n| n.trim() == raw) {
            Some(0.0)
        } else {
            None
        }
    };
    let predictor_key = |row: &Row| -> Vec<String> {
        std::iter::once(&task.predictor)
            .chain(task.additional_predictors.iter())
            .map(|p| canonical_state(row.get(p).unwrap_or("")))
            .collect()
    };

    let mut state_counts: BTreeMap<Vec<String>, (usize, usize)> = BTreeMap::new();
    let mut total_labeled = 0usize;
    let mut total_positive = 0usize;
    for row in train_rows {
        if let Some(y) = label_of(row) {
            let entry = state_counts.entry(predictor_key(row)).or_default();
            entry.0 += 1;
            entry.1 += y as usize;
            total_labeled += 1;
            total_positive += y as usize;
        }
    }

    let mut ref_labeled = 0usize;
    let mut loss = 0.0;
    for row in reference_rows {
        if let Some(y) = label_of(row) {
            let key = predictor_key(row);
            let (count, positives) = state_counts
                .get(&key)
                .copied()
                .unwrap_or((total_labeled, total_positive));
            let prob = (positives as f64 + task.alpha) / (count as f64 + 2.0 * task.alpha);
            loss += (prob - y).powi(2);
            ref_labeled += 1;
        }
    }

    let brier = (total_labeled > 0 && ref_labeled > 0).then_some(loss / ref_labeled as f64);
    (total_labeled, ref_labeled, brier)
}

fn mean_of(values: impl Iterator<Item = f64>) -> Option<f64> {
    let collected: Vec<f64> = values.collect();
    (!collected.is_empty()).then(|| collected.iter().sum::<f64>() / collected.len() as f64)
}

/// Run all candidate generators on identical training/reference cohorts, split fingerprint,
/// sample counts, and seeds, returning an [`AggregateComparisonManifest`] extending
/// [`CandidateComparisonManifest`].
pub fn compare_candidates(
    training_rows: &[Row],
    reference_rows: &[Row],
    candidates: &[&dyn CohortGenerator],
    config: &CandidateComparisonConfig,
) -> Result<AggregateComparisonManifest> {
    let protocol_manifest = config.to_manifest()?;
    if candidates.len() != config.candidate_names.len() {
        return Err(Error::Input(format!(
            "candidate generator count ({}) does not match configured candidate_names ({})",
            candidates.len(),
            config.candidate_names.len()
        )));
    }

    let train_vars = validate_cohort_rows(training_rows, "training")?;
    let ref_vars = validate_cohort_rows(reference_rows, "reference")?;
    if train_vars != ref_vars {
        return Err(Error::Input(
            "training and reference cohorts do not share an identical variable schema".into(),
        ));
    }

    let denominators = compute_denominators(
        training_rows,
        reference_rows,
        &train_vars,
        config.sample_count_per_seed,
        config.metrics.utility_brier.as_ref(),
    )?;

    let real_brier = config
        .metrics
        .utility_brier
        .as_ref()
        .map(|task| compute_brier_score(training_rows, reference_rows, task));

    let sw = software_identity()?;
    let training_fingerprint = yaml_fingerprint(&training_rows)?;
    let reference_fingerprint = yaml_fingerprint(&reference_rows)?;

    let mut candidate_summaries = Vec::with_capacity(candidates.len());
    for (idx, generator) in candidates.iter().enumerate() {
        let candidate_name = &config.candidate_names[idx];
        let fitted = generator.fit(training_rows)?;
        let BatchSampleResult {
            mut draws,
            mut sparse_backoff_counts,
        } = generator.sample_batch_with_backoffs(
            &fitted,
            config.sample_count_per_seed,
            &config.seeds,
        )?;
        let mut per_seed = Vec::with_capacity(config.seeds.len());

        for &seed in &config.seeds {
            let sampled_rows = draws.remove(&seed).ok_or_else(|| {
                Error::Input(format!(
                    "candidate '{candidate_name}' did not return sampled rows for seed {seed}"
                ))
            })?;
            let backoff_counts = sparse_backoff_counts.remove(&seed).unwrap_or_default();
            if sampled_rows.len() != config.sample_count_per_seed {
                return Err(Error::Input(format!(
                    "candidate '{candidate_name}' returned {} rows for seed {seed}, expected {}",
                    sampled_rows.len(),
                    config.sample_count_per_seed
                )));
            }
            let sample_vars = validate_cohort_rows(&sampled_rows, "generated")?;
            if sample_vars != train_vars {
                return Err(Error::Input(format!(
                    "candidate '{candidate_name}' generated schema does not match training schema"
                )));
            }

            let mut marginal_tvd_by_variable = BTreeMap::new();
            let mean_marginal_tvd = if config.metrics.marginal_tvd {
                for var in &train_vars {
                    let tvd = family_tvd(&sampled_rows, reference_rows, &[var.as_str()]);
                    marginal_tvd_by_variable.insert(var.clone(), tvd);
                }
                mean_of(marginal_tvd_by_variable.values().copied())
            } else {
                None
            };

            let mut pairwise_tvd_by_pair = BTreeMap::new();
            let mean_pairwise_tvd = if config.metrics.pairwise_tvd {
                for i in 0..train_vars.len() {
                    for j in (i + 1)..train_vars.len() {
                        let a = &train_vars[i];
                        let b = &train_vars[j];
                        let tvd =
                            family_tvd(&sampled_rows, reference_rows, &[a.as_str(), b.as_str()]);
                        pairwise_tvd_by_pair.insert(format!("{a}__{b}"), tvd);
                    }
                }
                mean_of(pairwise_tvd_by_pair.values().copied())
            } else {
                None
            };

            let joint_tvd = if config.metrics.joint_tvd {
                let cols: Vec<&str> = train_vars.iter().map(String::as_str).collect();
                Some(family_tvd(&sampled_rows, reference_rows, &cols))
            } else {
                None
            };

            let utility_brier = if let (Some(task), Some((_, ref_labeled, real_score))) =
                (config.metrics.utility_brier.as_ref(), real_brier)
            {
                let (synth_labeled, _, synth_score) =
                    compute_brier_score(&sampled_rows, reference_rows, task);
                Some(UtilityBrierSummary {
                    task_name: task.task_name.clone(),
                    real_training_brier: real_score,
                    synthetic_training_brier: synth_score,
                    synthetic_training_labeled: synth_labeled,
                    reference_labeled: ref_labeled,
                    reference_unlabeled: reference_rows.len() - ref_labeled,
                })
            } else {
                None
            };

            per_seed.push(CandidateSeedMetrics {
                seed,
                denominators: denominators.clone(),
                marginal_tvd_by_variable,
                mean_marginal_tvd,
                pairwise_tvd_by_pair,
                mean_pairwise_tvd,
                joint_tvd,
                utility_brier,
                sparse_backoff_counts: backoff_counts,
            });
        }

        let mean_marginal_tvd = mean_of(per_seed.iter().filter_map(|s| s.mean_marginal_tvd));
        let mean_pairwise_tvd = mean_of(per_seed.iter().filter_map(|s| s.mean_pairwise_tvd));
        let mean_joint_tvd = mean_of(per_seed.iter().filter_map(|s| s.joint_tvd));
        let real_training_brier = real_brier.and_then(|(_, _, score)| score);
        let mean_synthetic_brier = mean_of(
            per_seed
                .iter()
                .filter_map(|s| s.utility_brier.as_ref()?.synthetic_training_brier),
        );

        candidate_summaries.push(CandidateAggregateSummary {
            candidate_name: candidate_name.clone(),
            preprocessing: config.preprocessing.clone(),
            status: config.candidate_status.clone(),
            mean_marginal_tvd,
            mean_pairwise_tvd,
            mean_joint_tvd,
            real_training_brier,
            mean_synthetic_brier,
            per_seed,
        });
    }

    let aggregate = AggregateComparisonManifest {
        manifest: protocol_manifest,
        split_fingerprint: config.split_fingerprint.clone(),
        training_fingerprint,
        reference_fingerprint,
        variables: train_vars,
        denominators,
        candidate_summaries,
        software_version: sw.package_version,
        dependency_lock_fingerprint: sw.dependency_lock_fingerprint,
        limitations: vec![
            "Aggregate in-memory technical candidate comparison only; statistical similarity or predictive utility does not establish clinical plausibility, causal validity, privacy safety, or external transportability.".into(),
            "All candidates are evaluated under identical splits, seeds, and reference denominators; seed variation does not constitute a formal confidence interval or establish a general winner.".into(),
            "Blank and NA values are treated as missing states and are excluded from utility target labels without imputation.".into(),
        ],
    };
    aggregate.validate()?;
    Ok(aggregate)
}

fn splitmix64_next(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9e3779b97f4a7c15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
    z ^ (z >> 31)
}

fn sample_bounded_index(state: &mut u64, len: usize) -> usize {
    assert!(len > 0, "cannot sample index from empty slice");
    let bound = len as u64;
    let reject_below = bound.wrapping_neg() % bound;
    loop {
        let val = splitmix64_next(state);
        if val >= reject_below {
            return (val % bound) as usize;
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BinnedLearningMode {
    Forest,
    GreedyDag,
    InteractionDag,
}

fn build_binned_states(rows: &[Row], variables: &[String]) -> Vec<Vec<String>> {
    if rows.is_empty() || variables.is_empty() {
        return Vec::new();
    }
    let mut states = vec![vec![String::new(); variables.len()]; rows.len()];
    for (col_idx, var_name) in variables.iter().enumerate() {
        let mut all_non_missing_numeric = true;
        let mut has_non_missing = false;
        let mut numeric_values = Vec::new();
        for row in rows {
            let raw = row.get(var_name).unwrap_or("").trim();
            if token_is_missing(raw) {
                continue;
            }
            has_non_missing = true;
            match raw.parse::<f64>() {
                Ok(val) if val.is_finite() => numeric_values.push(val),
                _ => {
                    all_non_missing_numeric = false;
                    break;
                }
            }
        }
        let is_numeric = has_non_missing && all_non_missing_numeric;
        let cuts: Vec<f64> = if is_numeric {
            numeric_values.sort_by(f64::total_cmp);
            (1..4)
                .map(|j| numeric_values[(numeric_values.len() - 1) * j / 4])
                .collect()
        } else {
            Vec::new()
        };
        for (row, state_row) in rows.iter().zip(states.iter_mut()) {
            let raw = row.get(var_name).unwrap_or("").trim();
            state_row[col_idx] = if token_is_missing(raw) {
                "MISSING:NA".into()
            } else if is_numeric {
                let val = raw.parse::<f64>().expect("validated finite numeric token");
                format!("BIN:{}", cuts.partition_point(|cut| val > *cut))
            } else {
                format!("OBS:{raw}")
            };
        }
    }
    states
}

fn find_second_parent<'s>(
    states: &'s [Vec<String>],
    child_counts: &BTreeMap<&'s str, usize>,
    child: usize,
    first_parent: usize,
    n: f64,
) -> Option<usize> {
    let mut best_second: Option<(usize, f64)> = None;
    for candidate in 0..child {
        if candidate == first_parent {
            continue;
        }
        let mut first_counts: BTreeMap<&str, usize> = BTreeMap::new();
        let mut first_cand: BTreeMap<(&str, &str), usize> = BTreeMap::new();
        let mut first_child: BTreeMap<(&str, &str), usize> = BTreeMap::new();
        let mut joint: BTreeMap<(&str, &str, &str), usize> = BTreeMap::new();
        let mut cand_states: BTreeSet<&str> = BTreeSet::new();
        for row in states {
            let a = row[first_parent].as_str();
            let b = row[candidate].as_str();
            let c = row[child].as_str();
            *first_counts.entry(a).or_default() += 1;
            *first_cand.entry((a, b)).or_default() += 1;
            *first_child.entry((a, c)).or_default() += 1;
            *joint.entry((a, b, c)).or_default() += 1;
            cand_states.insert(b);
        }
        let cmi: f64 = joint
            .iter()
            .map(|(&(a, b, c), &count)| {
                let k = count as f64;
                (k / n)
                    * (k * first_counts[a] as f64
                        / (first_cand[&(a, b)] as f64 * first_child[&(a, c)] as f64))
                        .ln()
            })
            .sum();
        let df = (first_counts.len()
            * cand_states.len().saturating_sub(1)
            * child_counts.len().saturating_sub(1)) as f64;
        let penalty = df * n.ln() / (2.0 * n);
        let score = cmi - penalty;
        if score > 1e-12 && best_second.is_none_or(|(_, best)| score > best + 1e-12) {
            best_second = Some((candidate, score));
        }
    }
    best_second.map(|(parent, _)| parent)
}

fn find_joint_parent_pair<'s>(
    states: &'s [Vec<String>],
    child_counts: &BTreeMap<&'s str, usize>,
    child: usize,
    n: f64,
) -> Option<(usize, usize)> {
    let mut best: Option<(usize, usize, f64)> = None;
    for a in 0..child {
        for b in (a + 1)..child {
            let mut ab: BTreeMap<(&str, &str), usize> = BTreeMap::new();
            let mut abc: BTreeMap<(&str, &str, &str), usize> = BTreeMap::new();
            for row in states {
                let x = row[a].as_str();
                let y = row[b].as_str();
                let z = row[child].as_str();
                *ab.entry((x, y)).or_default() += 1;
                *abc.entry((x, y, z)).or_default() += 1;
            }
            let joint_info: f64 = abc
                .iter()
                .map(|(&(x, y, z), &count)| {
                    let k = count as f64;
                    (k / n) * (k * n / (ab[&(x, y)] as f64 * child_counts[z] as f64)).ln()
                })
                .sum();
            let df = (ab.len().saturating_sub(1) * child_counts.len().saturating_sub(1)) as f64;
            let gain = joint_info - df * n.ln() / (2.0 * n);
            if gain > 1e-12 && best.is_none_or(|(_, _, old_gain)| gain > old_gain + 1e-12) {
                best = Some((a, b, gain));
            }
        }
    }
    best.map(|(a, b, _)| (a, b))
}

fn fit_binned_structure(
    training_rows: &[Row],
    variables: &[String],
    mode: BinnedLearningMode,
) -> Vec<Vec<usize>> {
    let states = build_binned_states(training_rows, variables);
    let mut parents = vec![Vec::new(); variables.len()];
    if states.is_empty() {
        return parents;
    }
    let n = states.len() as f64;
    let penalized = mode != BinnedLearningMode::Forest;

    for child in 1..variables.len() {
        let mut child_counts: BTreeMap<&str, usize> = BTreeMap::new();
        for row in &states {
            *child_counts.entry(row[child].as_str()).or_default() += 1;
        }
        let mut best_first: Option<(usize, f64)> = None;
        for candidate in 0..child {
            let mut cand_counts: BTreeMap<&str, usize> = BTreeMap::new();
            let mut joint: BTreeMap<(&str, &str), usize> = BTreeMap::new();
            for row in &states {
                let p = row[candidate].as_str();
                let c = row[child].as_str();
                *cand_counts.entry(p).or_default() += 1;
                *joint.entry((p, c)).or_default() += 1;
            }
            let mi: f64 = joint
                .iter()
                .map(|(&(p, c), &count)| {
                    let k = count as f64;
                    (k / n) * (k * n / (cand_counts[p] as f64 * child_counts[c] as f64)).ln()
                })
                .sum();
            let penalty = if penalized {
                let df = (cand_counts.len().saturating_sub(1)
                    * child_counts.len().saturating_sub(1)) as f64;
                df * n.ln() / (2.0 * n)
            } else {
                0.0
            };
            let score = mi - penalty;
            if score > 1e-12 && best_first.is_none_or(|(_, best)| score > best + 1e-12) {
                best_first = Some((candidate, score));
            }
        }

        if let Some((first_parent, _)) = best_first {
            parents[child].push(first_parent);
            if matches!(
                mode,
                BinnedLearningMode::GreedyDag | BinnedLearningMode::InteractionDag
            ) {
                if let Some(second_parent) =
                    find_second_parent(&states, &child_counts, child, first_parent, n)
                {
                    parents[child].push(second_parent);
                }
            }
        } else if mode == BinnedLearningMode::InteractionDag {
            if let Some((a, b)) = find_joint_parent_pair(&states, &child_counts, child, n) {
                parents[child].extend([a, b]);
            }
        }
    }
    parents
}

fn validate_fitted_for_sampling(fitted: &FittedModel, count: usize) -> Result<()> {
    if fitted.training_rows.is_empty() || fitted.variable_order.is_empty() {
        return Err(Error::Input(
            "cannot sample from an empty fitted model".into(),
        ));
    }
    if count == 0 || count > 1_000_000 {
        return Err(Error::Input(
            "sample count must be between 1 and 1_000_000".into(),
        ));
    }
    let train_vars = validate_cohort_rows(&fitted.training_rows, "training")?;
    if train_vars != fitted.variable_order {
        return Err(Error::Input(
            "fitted model variable_order does not match training_rows schema".into(),
        ));
    }
    if fitted.parent_indices.len() != fitted.variable_order.len() {
        return Err(Error::Input(
            "fitted model parent_indices length does not match variable_order".into(),
        ));
    }
    for (col_idx, parents) in fitted.parent_indices.iter().enumerate() {
        if parents.iter().any(|&p| p >= col_idx) {
            return Err(Error::Input(format!(
                "fitted model parent_indices for column {col_idx} must only reference earlier columns"
            )));
        }
    }
    Ok(())
}

fn sample_for_seed_with_states(
    fitted: &FittedModel,
    states: &[Vec<String>],
    count: usize,
    seed: u64,
) -> (Vec<Row>, BTreeMap<String, usize>) {
    let mut state = seed;
    for byte in fitted.candidate_name.as_bytes() {
        state ^= u64::from(*byte);
        state = splitmix64_next(&mut state);
    }

    let mut out = Vec::with_capacity(count);
    let mut backoff_counts: BTreeMap<String, usize> = BTreeMap::new();
    for _ in 0..count {
        let mut fields = BTreeMap::new();
        let mut sampled_states: Vec<String> = Vec::with_capacity(fitted.variable_order.len());
        for (col_idx, var_name) in fitted.variable_order.iter().enumerate() {
            let parent_cols = &fitted.parent_indices[col_idx];
            let mut retained = parent_cols.len();
            let matching_indices: Vec<usize> = loop {
                let found: Vec<usize> = states
                    .iter()
                    .enumerate()
                    .filter(|(_, row_states)| {
                        parent_cols[..retained]
                            .iter()
                            .all(|&p_idx| row_states[p_idx] == sampled_states[p_idx])
                    })
                    .map(|(row_idx, _)| row_idx)
                    .collect();
                if !found.is_empty() || retained == 0 {
                    break found;
                }
                retained -= 1;
            };
            if retained < parent_cols.len() {
                *backoff_counts.entry(var_name.clone()).or_default() += 1;
            }
            let selected_row_idx =
                matching_indices[sample_bounded_index(&mut state, matching_indices.len())];
            let raw_value = fitted.training_rows[selected_row_idx]
                .get(var_name)
                .unwrap_or("")
                .to_owned();
            sampled_states.push(states[selected_row_idx][col_idx].clone());
            fields.insert(var_name.clone(), raw_value);
        }
        out.push(Row { fields });
    }
    (out, backoff_counts)
}

fn sample_batch_from_fitted_with_backoffs(
    fitted: &FittedModel,
    count: usize,
    seeds: &[u64],
) -> Result<BatchSampleResult> {
    validate_fitted_for_sampling(fitted, count)?;
    let states = build_binned_states(&fitted.training_rows, &fitted.variable_order);
    let mut draws = BTreeMap::new();
    let mut sparse_backoff_counts = BTreeMap::new();
    for &seed in seeds {
        let (rows, backoffs) = sample_for_seed_with_states(fitted, &states, count, seed);
        draws.insert(seed, rows);
        if !backoffs.is_empty() {
            sparse_backoff_counts.insert(seed, backoffs);
        }
    }
    Ok(BatchSampleResult {
        draws,
        sparse_backoff_counts,
    })
}

fn sample_from_fitted_with_backoffs(
    fitted: &FittedModel,
    count: usize,
    seed: u64,
) -> Result<(Vec<Row>, BTreeMap<String, usize>)> {
    let mut batch = sample_batch_from_fitted_with_backoffs(fitted, count, &[seed])?;
    let rows = batch.draws.remove(&seed).unwrap_or_default();
    let backoffs = batch.sparse_backoff_counts.remove(&seed).unwrap_or_default();
    Ok((rows, backoffs))
}

fn fit_binned_candidate(
    training_rows: &[Row],
    candidate_name: &str,
    mode: BinnedLearningMode,
) -> Result<FittedModel> {
    let variable_order = validate_cohort_rows(training_rows, "training")?;
    let parent_indices = fit_binned_structure(training_rows, &variable_order, mode);
    Ok(FittedModel {
        candidate_name: candidate_name.into(),
        variable_order,
        parent_indices,
        training_rows: training_rows.to_vec(),
    })
}

/// Single-parent ordered mutual-information binned forest baseline generator.
#[derive(Debug, Clone, Default)]
pub struct ForestCandidate;

impl CohortGenerator for ForestCandidate {
    fn fit(&self, training_rows: &[Row]) -> Result<FittedModel> {
        fit_binned_candidate(training_rows, "forest", BinnedLearningMode::Forest)
    }

    fn sample(&self, fitted: &FittedModel, count: usize, seed: u64) -> Result<Vec<Row>> {
        sample_from_fitted_with_backoffs(fitted, count, seed).map(|(rows, _)| rows)
    }

    fn sample_batch(
        &self,
        fitted: &FittedModel,
        count: usize,
        seeds: &[u64],
    ) -> Result<BTreeMap<u64, Vec<Row>>> {
        self.sample_batch_with_backoffs(fitted, count, seeds)
            .map(|res| res.draws)
    }

    fn sample_batch_with_backoffs(
        &self,
        fitted: &FittedModel,
        count: usize,
        seeds: &[u64],
    ) -> Result<BatchSampleResult> {
        sample_batch_from_fitted_with_backoffs(fitted, count, seeds)
    }
}

/// Greedy two-parent penalized binned directed acyclic graph (DAG) baseline generator.
#[derive(Debug, Clone, Default)]
pub struct DagCandidate;

impl CohortGenerator for DagCandidate {
    fn fit(&self, training_rows: &[Row]) -> Result<FittedModel> {
        fit_binned_candidate(training_rows, "dag", BinnedLearningMode::GreedyDag)
    }

    fn sample(&self, fitted: &FittedModel, count: usize, seed: u64) -> Result<Vec<Row>> {
        sample_from_fitted_with_backoffs(fitted, count, seed).map(|(rows, _)| rows)
    }

    fn sample_batch(
        &self,
        fitted: &FittedModel,
        count: usize,
        seeds: &[u64],
    ) -> Result<BTreeMap<u64, Vec<Row>>> {
        self.sample_batch_with_backoffs(fitted, count, seeds)
            .map(|res| res.draws)
    }

    fn sample_batch_with_backoffs(
        &self,
        fitted: &FittedModel,
        count: usize,
        seeds: &[u64],
    ) -> Result<BatchSampleResult> {
        sample_batch_from_fitted_with_backoffs(fitted, count, seeds)
    }
}

/// Two-parent penalized binned DAG generator with joint parent-pair interaction search.
#[derive(Debug, Clone, Default)]
pub struct InteractionDagCandidate;

impl CohortGenerator for InteractionDagCandidate {
    fn fit(&self, training_rows: &[Row]) -> Result<FittedModel> {
        fit_binned_candidate(
            training_rows,
            "interaction_dag",
            BinnedLearningMode::InteractionDag,
        )
    }

    fn sample(&self, fitted: &FittedModel, count: usize, seed: u64) -> Result<Vec<Row>> {
        sample_from_fitted_with_backoffs(fitted, count, seed).map(|(rows, _)| rows)
    }

    fn sample_batch(
        &self,
        fitted: &FittedModel,
        count: usize,
        seeds: &[u64],
    ) -> Result<BTreeMap<u64, Vec<Row>>> {
        self.sample_batch_with_backoffs(fitted, count, seeds)
            .map(|res| res.draws)
    }

    fn sample_batch_with_backoffs(
        &self,
        fitted: &FittedModel,
        count: usize,
        seeds: &[u64],
    ) -> Result<BatchSampleResult> {
        sample_batch_from_fitted_with_backoffs(fitted, count, seeds)
    }
}

/// Neural tabular cohort generator candidate backed by the in-memory Python CTGAN/TVAE bridge.
#[derive(Debug, Clone, Default)]
pub struct NeuralCandidate {
    pub config: NeuralGeneratorConfig,
    pub bridge_script: Option<PathBuf>,
}

impl CohortGenerator for NeuralCandidate {
    fn fit(&self, training_rows: &[Row]) -> Result<FittedModel> {
        let variable_order = validate_cohort_rows(training_rows, "training")?;
        self.config.validate()?;
        if let Some(discrete) = &self.config.discrete_columns {
            for col in discrete {
                if !variable_order.contains(col) {
                    return Err(Error::Input(format!(
                        "neural discrete_columns entry '{col}' not found in training schema"
                    )));
                }
            }
        }
        let candidate_name = match self.config.model_type {
            NeuralModelType::Ctgan => "neural-ctgan",
            NeuralModelType::Tvae => "neural-tvae",
        }
        .into();
        Ok(FittedModel {
            candidate_name,
            parent_indices: vec![Vec::new(); variable_order.len()],
            variable_order,
            training_rows: training_rows.to_vec(),
        })
    }

    fn sample(&self, fitted: &FittedModel, count: usize, seed: u64) -> Result<Vec<Row>> {
        let mut draws = self.sample_batch(fitted, count, &[seed])?;
        draws.remove(&seed).ok_or_else(|| {
            Error::Input(format!(
                "neural candidate did not return sampled rows for seed {seed}"
            ))
        })
    }

    fn sample_batch(
        &self,
        fitted: &FittedModel,
        count: usize,
        seeds: &[u64],
    ) -> Result<BTreeMap<u64, Vec<Row>>> {
        if fitted.training_rows.is_empty() || fitted.variable_order.is_empty() {
            return Err(Error::Input(
                "cannot sample from an empty fitted model".into(),
            ));
        }
        if count == 0 || count > 1_000_000 {
            return Err(Error::Input(
                "sample count must be between 1 and 1_000_000".into(),
            ));
        }
        if seeds.is_empty() {
            return Err(Error::Input("sample seeds must not be empty".into()));
        }
        let train_vars = validate_cohort_rows(&fitted.training_rows, "training")?;
        if train_vars != fitted.variable_order {
            return Err(Error::Input(
                "fitted model variable_order does not match training_rows schema".into(),
            ));
        }

        let mut cfg = self.config.clone();
        cfg.n_samples = count;
        cfg.fit_seed = Some(self.config.fit_seed.unwrap_or(self.config.seed));
        cfg.sample_seeds = Some(seeds.to_vec());
        cfg.validate()?;

        let (_, draws) = run_neural_bridge_in_memory(
            &fitted.training_rows,
            &cfg,
            seeds,
            self.bridge_script.as_deref(),
        )?;
        Ok(draws)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fictional_cohort(row_count: usize, seed_offset: u64) -> Vec<Row> {
        let mut state = 0x5eed_cafe_u64.wrapping_add(seed_offset);
        (0..row_count)
            .map(|idx| {
                let r = splitmix64_next(&mut state);
                let wfns = ((idx + (r as usize % 3)) % 5 + 1).to_string();
                let intervention = if (idx + (r as usize >> 3)).is_multiple_of(2) {
                    "evd"
                } else {
                    "lumbar_drain"
                };
                // Introduce deterministic missingness ("NA") on 6 of 100 rows while keeping
                // structured dependence on wfns and intervention for the remaining rows.
                let mrs = if idx.is_multiple_of(17) {
                    "NA".to_string()
                } else if wfns == "1" || wfns == "2" {
                    if intervention == "lumbar_drain" {
                        "1".to_string()
                    } else {
                        "2".to_string()
                    }
                } else if intervention == "lumbar_drain" {
                    "3".to_string()
                } else {
                    "5".to_string()
                };
                Row::from_pairs([
                    ("ha_wfns", wfns.as_str()),
                    ("intervention", intervention),
                    ("fup_mrs", mrs.as_str()),
                ])
            })
            .collect()
    }

    #[test]
    fn deterministic_fictional_forest_vs_dag_comparison_matches_manifest_validation() {
        let training_rows = fictional_cohort(100, 1);
        let reference_rows = fictional_cohort(100, 2);
        assert_eq!(training_rows.len(), 100);
        assert_eq!(reference_rows.len(), 100);

        let config = CandidateComparisonConfig {
            schema_version: "1".into(),
            protocol_id: "fictional-forest-vs-dag-v1".into(),
            contract_id: "fictional-coherence-v1".into(),
            preprocessing: "fictional_shared_split_missing_preserved".into(),
            candidate_status: "baseline".into(),
            populations: vec![
                "fictional_training_100".into(),
                "fictional_reference_100".into(),
                "fictional_synthetic_100".into(),
            ],
            candidate_names: vec!["forest".into(), "dag".into()],
            split_fingerprint: "fnv1a64-split:fictional-100-row-split".into(),
            sample_count_per_seed: 100,
            seeds: vec![11, 22, 33],
            metrics: CandidateMetricsConfig {
                marginal_tvd: true,
                pairwise_tvd: true,
                joint_tvd: true,
                utility_brier: Some(UtilityBrierConfig {
                    task_name: "favorable_ vs_unfavorable_mrs".into(),
                    predictor: "ha_wfns".into(),
                    additional_predictors: vec!["intervention".into()],
                    target: "fup_mrs".into(),
                    positive_values: vec!["3".into(), "4".into(), "5".into(), "6".into()],
                    negative_values: vec!["0".into(), "1".into(), "2".into()],
                    alpha: 1.0,
                    predictor_available_before_target: true,
                }),
            },
        };

        let forest = ForestCandidate;
        let dag = DagCandidate;
        let candidates: Vec<&dyn CohortGenerator> = vec![&forest, &dag];

        let report_a =
            compare_candidates(&training_rows, &reference_rows, &candidates, &config).unwrap();
        let report_b =
            compare_candidates(&training_rows, &reference_rows, &candidates, &config).unwrap();

        // Deterministic replay across identical inputs and seeds
        assert_eq!(report_a, report_b);
        assert_eq!(
            serde_yaml::to_string(&report_a).unwrap(),
            serde_yaml::to_string(&report_b).unwrap()
        );

        // Validate underlying CandidateComparisonManifest and extended AggregateComparisonManifest
        report_a.manifest.validate().unwrap();
        report_a.validate().unwrap();
        assert_eq!(report_a.manifest.candidates.len(), 2);
        assert_eq!(report_a.manifest.seeds, vec![11, 22, 33]);
        assert_eq!(report_a.manifest.sample_count, 100);
        assert_eq!(
            report_a.manifest.metrics,
            vec!["marginal_tvd", "pairwise_tvd", "joint_tvd", "utility_brier"]
        );

        // Verify identical denominators across both candidates and all 3 seeds
        assert_eq!(report_a.denominators.training_rows, 100);
        assert_eq!(report_a.denominators.reference_rows, 100);
        assert_eq!(report_a.denominators.generated_rows, 100);
        assert_eq!(report_a.denominators.joint_reference_rows, 100);
        assert_eq!(report_a.denominators.utility_training_labeled, Some(94));
        assert_eq!(report_a.denominators.utility_reference_labeled, Some(94));
        assert_eq!(report_a.denominators.utility_reference_unlabeled, Some(6));

        for summary in &report_a.candidate_summaries {
            assert_eq!(summary.per_seed.len(), 3);
            assert!(summary.mean_marginal_tvd.is_some());
            assert!(summary.mean_pairwise_tvd.is_some());
            assert!(summary.mean_joint_tvd.is_some());
            assert!(summary.real_training_brier.is_some());
            assert!(summary.mean_synthetic_brier.is_some());

            for seed_metric in &summary.per_seed {
                assert_eq!(seed_metric.denominators, report_a.denominators);
                assert_eq!(seed_metric.marginal_tvd_by_variable.len(), 3);
                assert_eq!(seed_metric.pairwise_tvd_by_pair.len(), 3);
                let brier = seed_metric.utility_brier.as_ref().unwrap();
                assert_eq!(brier.reference_labeled, 94);
                assert_eq!(brier.reference_unlabeled, 6);
            }
        }
    }

    #[test]
    fn config_and_manifest_reject_invalid_conditions() {
        let training_rows = fictional_cohort(20, 10);
        let reference_rows = fictional_cohort(20, 20);
        let forest = ForestCandidate;
        let dag = DagCandidate;
        let candidates: Vec<&dyn CohortGenerator> = vec![&forest, &dag];

        let mut config = CandidateComparisonConfig {
            schema_version: "1".into(),
            protocol_id: "proto".into(),
            contract_id: "contract".into(),
            preprocessing: "shared".into(),
            candidate_status: "baseline".into(),
            populations: vec!["dev".into()],
            candidate_names: vec!["forest".into(), "dag".into()],
            split_fingerprint: "fp-1".into(),
            sample_count_per_seed: 20,
            seeds: vec![1, 2, 3],
            metrics: CandidateMetricsConfig {
                marginal_tvd: true,
                pairwise_tvd: false,
                joint_tvd: true,
                utility_brier: None,
            },
        };
        assert!(compare_candidates(&training_rows, &reference_rows, &candidates, &config).is_ok());

        // Duplicate candidate names rejected
        config.candidate_names = vec!["forest".into(), "forest".into()];
        assert!(compare_candidates(&training_rows, &reference_rows, &candidates, &config).is_err());
        config.candidate_names = vec!["forest".into(), "dag".into()];

        // Duplicate seeds rejected
        config.seeds = vec![1, 1, 2];
        assert!(compare_candidates(&training_rows, &reference_rows, &candidates, &config).is_err());
        config.seeds = vec![1, 2, 3];

        // Missing ("NA") utility target label rejected
        config.metrics.utility_brier = Some(UtilityBrierConfig {
            task_name: "bad_task".into(),
            predictor: "ha_wfns".into(),
            additional_predictors: vec![],
            target: "fup_mrs".into(),
            positive_values: vec!["NA".into()],
            negative_values: vec!["1".into()],
            alpha: 1.0,
            predictor_available_before_target: true,
        });
        assert!(compare_candidates(&training_rows, &reference_rows, &candidates, &config).is_err());
    }

    #[test]
    fn binned_numeric_forest_and_dag_learn_quantile_dependencies_and_preserve_raw_values() {
        let mut rows = Vec::with_capacity(100);
        for i in 0..100usize {
            let a_num = if i.is_multiple_of(20) {
                "NA".to_string()
            } else if i % 20 == 5 {
                "".to_string()
            } else if i % 20 == 10 {
                "na".to_string()
            } else {
                format!("{:.2}", 1.5 + i as f64)
            };
            let b_num = if i.is_multiple_of(20) {
                "NA".to_string()
            } else {
                format!("{:.2}", 10.0 + (i as f64) * 2.0)
            };
            rows.push(Row::from_pairs([
                ("a_num", a_num.as_str()),
                ("b_num", b_num.as_str()),
            ]));
        }

        let forest = ForestCandidate;
        let dag = DagCandidate;
        let forest_fit = forest.fit(&rows).unwrap();
        let dag_fit = dag.fit(&rows).unwrap();

        assert_eq!(forest_fit.variable_order, vec!["a_num", "b_num"]);
        assert_eq!(forest_fit.parent_indices, vec![vec![], vec![0]]);
        assert_eq!(dag_fit.parent_indices, vec![vec![], vec![0]]);

        let raw_a_values: HashSet<&str> = rows.iter().map(|r| r.get("a_num").unwrap()).collect();
        let raw_b_values: HashSet<&str> = rows.iter().map(|r| r.get("b_num").unwrap()).collect();
        assert!(!raw_b_values.contains("0"));
        assert!(!raw_b_values.contains("0.0"));

        let exact_pairs: HashSet<(&str, &str)> = rows
            .iter()
            .map(|r| (r.get("a_num").unwrap(), r.get("b_num").unwrap()))
            .collect();

        let sampled = dag.sample(&dag_fit, 300, 41).unwrap();
        assert_eq!(sampled.len(), 300);

        let mut saw_missing_na = false;
        let mut saw_missing_blank = false;
        let mut saw_missing_lowercase_na = false;
        let mut saw_within_bin_recombination = false;

        for row in &sampled {
            let a = row.get("a_num").unwrap();
            let b = row.get("b_num").unwrap();
            assert!(raw_a_values.contains(a));
            assert!(raw_b_values.contains(b));
            if a == "NA" {
                saw_missing_na = true;
            } else if a.is_empty() {
                saw_missing_blank = true;
            } else if a == "na" {
                saw_missing_lowercase_na = true;
            }
            if !exact_pairs.contains(&(a, b)) {
                saw_within_bin_recombination = true;
            }
        }
        assert!(saw_missing_na);
        assert!(saw_missing_blank);
        assert!(saw_missing_lowercase_na);
        assert!(saw_within_bin_recombination);
    }

    #[test]
    fn interaction_dag_learns_xor_and_tracks_sparse_backoff_counts() {
        let mut xor_rows = Vec::with_capacity(400);
        let mut independent_rows = Vec::with_capacity(800);
        for _ in 0..100 {
            for a in 0..2 {
                for b in 0..2 {
                    let c = a ^ b;
                    xor_rows.push(Row::from_pairs([
                        ("a", a.to_string()),
                        ("b", b.to_string()),
                        ("c", c.to_string()),
                    ]));
                    for ind_c in 0..2 {
                        independent_rows.push(Row::from_pairs([
                            ("a", a.to_string()),
                            ("b", b.to_string()),
                            ("c", ind_c.to_string()),
                        ]));
                    }
                }
            }
        }

        let dag = DagCandidate;
        let interaction = InteractionDagCandidate;

        let dag_xor_fit = dag.fit(&xor_rows).unwrap();
        assert!(dag_xor_fit.parent_indices[2].is_empty());

        let interaction_xor_fit = interaction.fit(&xor_rows).unwrap();
        assert_eq!(interaction_xor_fit.parent_indices[2], vec![0, 1]);

        let xor_batch = interaction
            .sample_batch_with_backoffs(&interaction_xor_fit, 200, &[41, 42])
            .unwrap();
        assert!(xor_batch.sparse_backoff_counts.is_empty());
        for row in &xor_batch.draws[&41] {
            let a = row.get("a").unwrap();
            let b = row.get("b").unwrap();
            let c = row.get("c").unwrap();
            assert_eq!(c, if a == b { "0" } else { "1" });
        }

        let independent_fit = interaction.fit(&independent_rows).unwrap();
        assert!(independent_fit.parent_indices[2].is_empty());

        // Direct sparse backoff verification when a parent pair combination is unsupported.
        let sparse_model = FittedModel {
            candidate_name: "interaction_dag".into(),
            variable_order: vec!["a".into(), "b".into(), "c".into()],
            parent_indices: vec![vec![], vec![], vec![0, 1]],
            training_rows: vec![
                Row::from_pairs([("a", "0"), ("b", "0"), ("c", "0")]),
                Row::from_pairs([("a", "1"), ("b", "1"), ("c", "1")]),
            ],
        };
        let sparse_res = interaction
            .sample_batch_with_backoffs(&sparse_model, 200, &[7, 11])
            .unwrap();
        assert!(sparse_res.sparse_backoff_counts[&7].get("c").copied().unwrap_or(0) > 0);
        for row in &sparse_res.draws[&7] {
            assert_eq!(row.get("c"), row.get("a"));
        }

        // End-to-end sparse backoff inside compare_candidates (4x4 grid missing (3, 3)).
        let mut hole_rows = Vec::with_capacity(120);
        for _ in 0..8 {
            for a in 0..4 {
                for b in 0..4 {
                    if a == 3 && b == 3 {
                        continue;
                    }
                    let c = usize::from((a >= 2) ^ (b >= 2));
                    hole_rows.push(Row::from_pairs([
                        ("a", a.to_string()),
                        ("b", b.to_string()),
                        ("c", c.to_string()),
                    ]));
                }
            }
        }
        let hole_dag_fit = dag.fit(&hole_rows).unwrap();
        assert!(hole_dag_fit.parent_indices[2].is_empty());
        let hole_int_fit = interaction.fit(&hole_rows).unwrap();
        assert_eq!(hole_int_fit.parent_indices[1], Vec::<usize>::new());
        assert_eq!(hole_int_fit.parent_indices[2], vec![0, 1]);

        let hole_cfg = CandidateComparisonConfig {
            schema_version: "1".into(),
            protocol_id: "fictional-xor-sparse-backoff-v1".into(),
            contract_id: "fictional-coherence-v1".into(),
            preprocessing: "shared_in_memory_split_and_schema".into(),
            candidate_status: "baseline".into(),
            populations: vec!["fictional_train".into(), "fictional_ref".into()],
            candidate_names: vec!["dag".into(), "interaction_dag".into()],
            split_fingerprint: "fnv1a64-split:fictional-hole".into(),
            sample_count_per_seed: 400,
            seeds: vec![101, 202],
            metrics: CandidateMetricsConfig {
                marginal_tvd: true,
                pairwise_tvd: true,
                joint_tvd: true,
                utility_brier: None,
            },
        };
        let hole_candidates: Vec<&dyn CohortGenerator> = vec![&dag, &interaction];
        let hole_report =
            compare_candidates(&hole_rows, &hole_rows, &hole_candidates, &hole_cfg).unwrap();
        hole_report.validate().unwrap();
        assert!(hole_report.candidate_summaries[0]
            .per_seed
            .iter()
            .all(|s| s.sparse_backoff_counts.is_empty()));
        assert!(hole_report.candidate_summaries[1]
            .per_seed
            .iter()
            .all(|s| s.sparse_backoff_counts.get("c").copied().unwrap_or(0) > 0));
    }

    #[test]
    fn neural_candidate_in_compare_candidates_deterministic_and_no_disk_io() {
        let mut training_rows = Vec::with_capacity(20);
        let mut reference_rows = Vec::with_capacity(20);
        for i in 0..20 {
            let make_row = |idx: usize, shift: usize| {
                let age = if idx.is_multiple_of(5) {
                    "NA".to_string()
                } else {
                    (30 + (idx + shift) % 15).to_string()
                };
                let group = if idx.is_multiple_of(4) {
                    "".to_string()
                } else if (idx + shift).is_multiple_of(2) {
                    "control".to_string()
                } else {
                    "active".to_string()
                };
                let score = if idx.is_multiple_of(6) {
                    "na".to_string()
                } else {
                    format!("{:.2}", 1.5 + ((idx + shift) % 7) as f64 * 0.5)
                };
                let outcome = if idx.is_multiple_of(7) {
                    "NA".to_string()
                } else if (idx + shift).is_multiple_of(2) {
                    "yes".to_string()
                } else {
                    "no".to_string()
                };
                Row::from_pairs([
                    ("age", age.as_str()),
                    ("group", group.as_str()),
                    ("score", score.as_str()),
                    ("outcome", outcome.as_str()),
                ])
            };
            training_rows.push(make_row(i, 0));
            reference_rows.push(make_row(i, 1));
        }

        let forest = ForestCandidate;
        let dag = DagCandidate;
        let neural = NeuralCandidate {
            config: NeuralGeneratorConfig {
                model_type: NeuralModelType::Ctgan,
                epochs: 2,
                batch_size: 10,
                pac: Some(2),
                embedding_dim: 8,
                generator_dim: Some(vec![16, 16]),
                discriminator_dim: Some(vec![16, 16]),
                seed: 42,
                fit_seed: Some(77),
                ..Default::default()
            },
            bridge_script: None,
        };

        let empty_rows: Vec<Row> = Vec::new();
        assert!(neural.fit(&empty_rows).is_err());

        let fitted_ctgan = neural.fit(&training_rows).unwrap();
        assert_eq!(fitted_ctgan.candidate_name, "neural-ctgan");
        assert_eq!(fitted_ctgan.parent_indices, vec![Vec::<usize>::new(); 4]);
        assert!(neural.sample(&fitted_ctgan, 0, 101).is_err());
        assert!(neural.sample_batch(&fitted_ctgan, 5, &[]).is_err());

        let tvae = NeuralCandidate {
            config: NeuralGeneratorConfig {
                model_type: NeuralModelType::Tvae,
                epochs: 2,
                batch_size: 10,
                ..Default::default()
            },
            bridge_script: None,
        };
        assert_eq!(tvae.fit(&training_rows).unwrap().candidate_name, "neural-tvae");

        let config = CandidateComparisonConfig {
            schema_version: "1".into(),
            protocol_id: "fictional-forest-dag-neural-v1".into(),
            contract_id: "fictional-coherence-v1".into(),
            preprocessing: "fictional_shared_split_missing_preserved".into(),
            candidate_status: "exploratory".into(),
            populations: vec![
                "fictional_training_20".into(),
                "fictional_reference_20".into(),
                "fictional_synthetic_20".into(),
            ],
            candidate_names: vec!["forest".into(), "dag".into(), "neural-ctgan".into()],
            split_fingerprint: "fnv1a64-split:fictional-20-row-neural-split".into(),
            sample_count_per_seed: 20,
            seeds: vec![101, 202],
            metrics: CandidateMetricsConfig {
                marginal_tvd: true,
                pairwise_tvd: true,
                joint_tvd: true,
                utility_brier: Some(UtilityBrierConfig {
                    task_name: "fictional_group_outcome".into(),
                    predictor: "group".into(),
                    additional_predictors: vec![],
                    target: "outcome".into(),
                    positive_values: vec!["yes".into()],
                    negative_values: vec!["no".into()],
                    alpha: 1.0,
                    predictor_available_before_target: true,
                }),
            },
        };

        let crate_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
        let workspace_dir = crate_dir.join("../..");
        let snapshot_dir = |dir: &std::path::Path| -> BTreeSet<std::ffi::OsString> {
            std::fs::read_dir(dir)
                .unwrap()
                .filter_map(|entry| {
                    let name = entry.ok()?.file_name();
                    (name != "target").then_some(name)
                })
                .collect()
        };
        let before_crate = snapshot_dir(crate_dir);
        let before_workspace = snapshot_dir(&workspace_dir);

        let candidates: Vec<&dyn CohortGenerator> = vec![&forest, &dag, &neural];
        let report_a =
            compare_candidates(&training_rows, &reference_rows, &candidates, &config).unwrap();
        let report_b =
            compare_candidates(&training_rows, &reference_rows, &candidates, &config).unwrap();

        assert_eq!(snapshot_dir(crate_dir), before_crate);
        assert_eq!(snapshot_dir(&workspace_dir), before_workspace);
        for artifact in [
            "samples.csv",
            "model.pkl",
            "manifest.json",
            "neural_manifest.json",
            "config.json",
        ] {
            assert!(!crate_dir.join(artifact).exists());
            assert!(!workspace_dir.join(artifact).exists());
        }

        assert_eq!(report_a, report_b);
        report_a.manifest.validate().unwrap();
        report_a.validate().unwrap();
        assert_eq!(report_a.candidate_summaries.len(), 3);

        for summary in &report_a.candidate_summaries {
            assert_eq!(summary.per_seed.len(), 2);
            assert!(summary.mean_marginal_tvd.is_some());
            assert!(summary.mean_pairwise_tvd.is_some());
            assert!(summary.mean_joint_tvd.is_some());
            for seed_metric in &summary.per_seed {
                assert_eq!(seed_metric.denominators, report_a.denominators);
                assert!(seed_metric.utility_brier.is_some());
            }
        }
    }
}
