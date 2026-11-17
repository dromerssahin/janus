//! Production forensic workflow orchestrator for in-memory synthetic-versus-reference cohort
//! evaluation.
//!
//! All patient rows remain strictly in memory; no patient-level file I/O is performed and
//! [`ForensicManifest`] serializes only aggregate metrics, fingerprints, software/environment
//! identity, and bounded interpretation limits.

use crate::{
    forensic::{
        chi_squared_test, kolmogorov_smirnov_test, DiscriminatorMetrics, DistributionalTest,
    },
    software_identity, yaml_fingerprint, Error,
};
use linfa::prelude::*;
use linfa_logistic::LogisticRegression;
use ndarray::{Array1, Array2, Axis};
use rand::{rngs::StdRng, seq::SliceRandom, Rng, SeedableRng};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet, HashSet};

/// Small-cell suppression threshold: cohorts with fewer than 10 rows return
/// [`ForensicOutcome::Insufficient`] without running statistical tests or ML discriminators.
pub const MINIMUM_FORENSIC_COHORT_ROWS: usize = 10;

/// Discriminator and statistical test types executed by the forensic suite.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[serde(rename_all = "snake_case")]
pub enum DiscriminatorType {
    ChiSquared,
    KolmogorovSmirnov,
    LogisticRegression,
    RandomForest,
}

/// High-level forensic evaluation outcome.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ForensicOutcome {
    /// Synthetic and reference cohorts are statistically or predictively distinguishable.
    Discriminable,
    /// Discriminators and statistical tests did not separate the cohorts above threshold.
    Indistinguishable,
    /// Cohort size is below [`MINIMUM_FORENSIC_COHORT_ROWS`] (small-cell suppression).
    Insufficient,
}

/// In-memory representation of a single patient row for forensic evaluation.
/// Callers must keep rows in memory; rows are never serialized into [`ForensicManifest`].
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ForensicRow {
    pub patient_key: String,
    #[serde(default)]
    pub numeric_fields: BTreeMap<String, Option<f64>>,
    #[serde(default)]
    pub categorical_fields: BTreeMap<String, Option<String>>,
}

/// In-memory cohort container for forensic orchestration.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ForensicCohort {
    pub cohort_id: String,
    pub rows: Vec<ForensicRow>,
}

impl ForensicCohort {
    pub fn new(cohort_id: impl Into<String>, rows: Vec<ForensicRow>) -> Self {
        Self {
            cohort_id: cohort_id.into(),
            rows,
        }
    }

    /// Construct an in-memory [`ForensicCohort`] from schema-consistent candidate [`crate::candidate::Row`]
    /// slices without imputing missing values (`""` and `"NA"` become `None`).
    pub fn from_candidate_rows(
        cohort_id: impl Into<String>,
        rows: &[crate::candidate::Row],
    ) -> Result<Self, Error> {
        let cohort_id = cohort_id.into();
        if cohort_id.trim().is_empty() {
            return Err(Error::Input("forensic cohort_id must not be empty".into()));
        }
        if rows.is_empty() {
            return Err(Error::Input(format!(
                "forensic cohort '{cohort_id}' requires at least one row"
            )));
        }

        let columns: Vec<String> = rows[0].fields.keys().cloned().collect();
        if columns.is_empty() {
            return Err(Error::Input(format!(
                "forensic cohort '{cohort_id}' rows must contain at least one column"
            )));
        }
        for row in rows.iter().skip(1) {
            if row.fields.len() != columns.len()
                || row.fields.keys().ne(columns.iter())
            {
                return Err(Error::Input(format!(
                    "forensic cohort '{cohort_id}' has inconsistent row schemas"
                )));
            }
        }

        let mut is_numeric_col = Vec::with_capacity(columns.len());
        for col in &columns {
            let mut all_numeric = true;
            let mut has_observed = false;
            for row in rows {
                let raw = row.fields[col].trim();
                if !crate::candidate::token_is_missing(raw) {
                    if let Ok(v) = raw.parse::<f64>() {
                        if v.is_finite() {
                            has_observed = true;
                        } else {
                            all_numeric = false;
                            break;
                        }
                    } else {
                        all_numeric = false;
                        break;
                    }
                }
            }
            is_numeric_col.push(all_numeric && has_observed);
        }

        let mut forensic_rows = Vec::with_capacity(rows.len());
        for (idx, row) in rows.iter().enumerate() {
            let mut numeric_fields = BTreeMap::new();
            let mut categorical_fields = BTreeMap::new();
            for (col, &is_numeric) in columns.iter().zip(is_numeric_col.iter()) {
                let trimmed = row.fields[col].trim();
                if is_numeric {
                    let val = if crate::candidate::token_is_missing(trimmed) {
                        None
                    } else {
                        trimmed.parse::<f64>().ok().filter(|v| v.is_finite())
                    };
                    numeric_fields.insert(col.clone(), val);
                } else {
                    let val = if crate::candidate::token_is_missing(trimmed) {
                        None
                    } else {
                        Some(trimmed.to_string())
                    };
                    categorical_fields.insert(col.clone(), val);
                }
            }
            forensic_rows.push(ForensicRow {
                patient_key: format!("{cohort_id}-row-{idx:06}"),
                numeric_fields,
                categorical_fields,
            });
        }

        let cohort = Self::new(cohort_id, forensic_rows);
        cohort.validate_keys()?;
        Ok(cohort)
    }

    pub fn validate_keys(&self) -> Result<(), Error> {
        if self.cohort_id.trim().is_empty() {
            return Err(Error::Input("forensic cohort_id must not be empty".into()));
        }
        let mut seen = HashSet::new();
        for row in &self.rows {
            if row.patient_key.trim().is_empty() || !seen.insert(&row.patient_key) {
                return Err(Error::Input(format!(
                    "forensic cohort '{}' contains empty or duplicate patient_key",
                    self.cohort_id
                )));
            }
        }
        Ok(())
    }
}

/// Canonical fingerprints for the three in-memory cohorts.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ForensicFingerprints {
    pub training_fingerprint: String,
    pub generated_fingerprint: String,
    pub reference_fingerprint: String,
}

/// Execution environment metadata recorded in [`ForensicManifest`].
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ForensicEnvironment {
    pub operating_system: String,
    pub architecture: String,
    pub rustc: String,
    pub target: String,
}

/// Per-seed ML discriminator metrics for both classifiers.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct SeedDiscriminatorMetrics {
    pub seed: u64,
    pub logistic_regression: DiscriminatorMetrics,
    pub random_forest: DiscriminatorMetrics,
}

/// Seed-aggregated summary metrics for an ML discriminator family.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct AggregateClassifierMetrics {
    pub mean_auc_roc: f64,
    pub min_auc_roc: f64,
    pub max_auc_roc: f64,
    pub mean_accuracy: f64,
    pub mean_precision: f64,
    pub mean_recall: f64,
}

/// Aggregate statistical and ML discriminator metrics in [`ForensicManifest`].
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ForensicAggregateMetrics {
    pub training_rows: usize,
    pub generated_rows: usize,
    pub reference_rows: usize,
    pub minimum_cell_size: usize,
    pub suppressed: bool,
    pub chi_squared_tests: Vec<DistributionalTest>,
    pub ks_tests: Vec<DistributionalTest>,
    pub min_p_value: Option<f64>,
    pub mean_effect_size: Option<f64>,
    pub logistic_regression: Option<AggregateClassifierMetrics>,
    pub random_forest: Option<AggregateClassifierMetrics>,
    pub combined_mean_auc_roc: Option<f64>,
    pub per_seed_metrics: Vec<SeedDiscriminatorMetrics>,
}

/// Reproducibility and aggregate-outcome manifest for a forensic evaluation run.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ForensicManifest {
    pub schema_version: String,
    pub discriminator_types: Vec<DiscriminatorType>,
    pub seeds: Vec<u64>,
    pub fingerprints: ForensicFingerprints,
    pub software_version: String,
    pub dependency_lock_fingerprint: String,
    pub environment: ForensicEnvironment,
    pub outcome: ForensicOutcome,
    pub aggregate_metrics: ForensicAggregateMetrics,
    pub limitations: Vec<String>,
}

impl ForensicManifest {
    /// Validate manifest completeness and structural invariants.
    pub fn validate(&self) -> Result<(), Error> {
        if self.schema_version != "1"
            || self.discriminator_types.is_empty()
            || self.software_version.trim().is_empty()
            || self.dependency_lock_fingerprint.trim().is_empty()
            || self.environment.operating_system.trim().is_empty()
            || self.environment.architecture.trim().is_empty()
            || self.environment.rustc.trim().is_empty()
            || self.environment.target.trim().is_empty()
            || self.fingerprints.training_fingerprint.trim().is_empty()
            || self.fingerprints.generated_fingerprint.trim().is_empty()
            || self.fingerprints.reference_fingerprint.trim().is_empty()
            || self.limitations.is_empty()
            || self.limitations.iter().any(|s| s.trim().is_empty())
        {
            return Err(Error::Input("incomplete forensic manifest".into()));
        }

        match self.outcome {
            ForensicOutcome::Insufficient => {
                if !self.aggregate_metrics.suppressed
                    || !self.aggregate_metrics.per_seed_metrics.is_empty()
                    || self.aggregate_metrics.logistic_regression.is_some()
                    || self.aggregate_metrics.random_forest.is_some()
                {
                    return Err(Error::Input(
                        "insufficient forensic manifest must suppress discriminator metrics".into(),
                    ));
                }
            }
            ForensicOutcome::Discriminable | ForensicOutcome::Indistinguishable => {
                if self.seeds.is_empty()
                    || self.aggregate_metrics.suppressed
                    || self.aggregate_metrics.logistic_regression.is_none()
                    || self.aggregate_metrics.random_forest.is_none()
                    || self.aggregate_metrics.per_seed_metrics.len() != self.seeds.len()
                {
                    return Err(Error::Input(
                        "non-suppressed forensic manifest requires seeds and classifier metrics"
                            .into(),
                    ));
                }
            }
        }
        Ok(())
    }

    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }

    pub fn to_yaml(&self) -> Result<String, serde_yaml::Error> {
        serde_yaml::to_string(self)
    }
}

/// Orchestrate the full in-memory forensic evaluation suite across statistical tests
/// ($\chi^2$, Kolmogorov-Smirnov) and ML discriminators (logistic regression, random forest).
///
/// If any of `training`, `generated`, or `reference` has fewer than
/// [`MINIMUM_FORENSIC_COHORT_ROWS`] (10) rows, small-cell suppression applies and
/// [`ForensicOutcome::Insufficient`] is returned immediately without running discriminators.
pub fn orchestrate_forensic_suite(
    training: &ForensicCohort,
    generated: &ForensicCohort,
    reference: &ForensicCohort,
    seeds: &[u64],
) -> Result<ForensicManifest, Error> {
    training.validate_keys()?;
    generated.validate_keys()?;
    reference.validate_keys()?;
    if seeds.is_empty() {
        return Err(Error::Input(
            "orchestrate_forensic_suite requires at least one seed".into(),
        ));
    }

    let sw = software_identity()?;
    let fingerprints = ForensicFingerprints {
        training_fingerprint: yaml_fingerprint(training)?,
        generated_fingerprint: yaml_fingerprint(generated)?,
        reference_fingerprint: yaml_fingerprint(reference)?,
    };
    let environment = ForensicEnvironment {
        operating_system: sw.operating_system,
        architecture: sw.architecture,
        rustc: sw.rustc,
        target: sw.target,
    };
    let discriminator_types = vec![
        DiscriminatorType::ChiSquared,
        DiscriminatorType::KolmogorovSmirnov,
        DiscriminatorType::LogisticRegression,
        DiscriminatorType::RandomForest,
    ];
    let limitations = vec![
        "Aggregate in-memory forensic comparison only; statistical or classifier indistinguishability does not establish clinical plausibility, privacy safety, causal validity, or external transportability.".into(),
        "Reproducibility of discriminator metrics is bounded to identical software, seeds, feature encodings, and execution environment.".into(),
    ];

    let n_train_rows = training.rows.len();
    let n_gen_rows = generated.rows.len();
    let n_ref_rows = reference.rows.len();

    // Small-cell suppression: <10 rows in any cohort returns Insufficient without running discriminators.
    if n_train_rows < MINIMUM_FORENSIC_COHORT_ROWS
        || n_gen_rows < MINIMUM_FORENSIC_COHORT_ROWS
        || n_ref_rows < MINIMUM_FORENSIC_COHORT_ROWS
    {
        return Ok(ForensicManifest {
            schema_version: "1".into(),
            discriminator_types,
            seeds: seeds.to_vec(),
            fingerprints,
            software_version: sw.package_version,
            dependency_lock_fingerprint: sw.dependency_lock_fingerprint,
            environment,
            outcome: ForensicOutcome::Insufficient,
            aggregate_metrics: ForensicAggregateMetrics {
                training_rows: n_train_rows,
                generated_rows: n_gen_rows,
                reference_rows: n_ref_rows,
                minimum_cell_size: MINIMUM_FORENSIC_COHORT_ROWS,
                suppressed: true,
                chi_squared_tests: Vec::new(),
                ks_tests: Vec::new(),
                min_p_value: None,
                mean_effect_size: None,
                logistic_regression: None,
                random_forest: None,
                combined_mean_auc_roc: None,
                per_seed_metrics: Vec::new(),
            },
            limitations,
        });
    }

    // Collect sorted variable names across cohorts.
    let mut numeric_vars = BTreeSet::new();
    let mut categorical_vars = BTreeSet::new();
    for cohort in [training, generated, reference] {
        for row in &cohort.rows {
            for k in row.numeric_fields.keys() {
                numeric_vars.insert(k.clone());
            }
            for k in row.categorical_fields.keys() {
                categorical_vars.insert(k.clone());
            }
        }
    }

    // 1. Statistical tests: Chi-squared for categorical variables, KS for continuous variables.
    let mut chi_squared_tests = Vec::new();
    for var in &categorical_vars {
        let ref_vals: Vec<String> = reference
            .rows
            .iter()
            .filter_map(|r| r.categorical_fields.get(var).and_then(|v| v.as_ref()))
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty() && !s.eq_ignore_ascii_case("NA"))
            .collect();
        let gen_vals: Vec<String> = generated
            .rows
            .iter()
            .filter_map(|r| r.categorical_fields.get(var).and_then(|v| v.as_ref()))
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty() && !s.eq_ignore_ascii_case("NA"))
            .collect();
        if !ref_vals.is_empty() && !gen_vals.is_empty() {
            chi_squared_tests.push(chi_squared_test(var, &ref_vals, &gen_vals)?);
        }
    }

    let mut ks_tests = Vec::new();
    for var in &numeric_vars {
        let ref_vals: Vec<f64> = reference
            .rows
            .iter()
            .filter_map(|r| r.numeric_fields.get(var).and_then(|v| *v))
            .filter(|v| v.is_finite())
            .collect();
        let gen_vals: Vec<f64> = generated
            .rows
            .iter()
            .filter_map(|r| r.numeric_fields.get(var).and_then(|v| *v))
            .filter(|v| v.is_finite())
            .collect();
        if !ref_vals.is_empty() && !gen_vals.is_empty() {
            ks_tests.push(kolmogorov_smirnov_test(var, &ref_vals, &gen_vals)?);
        }
    }

    let all_tests: Vec<&DistributionalTest> =
        chi_squared_tests.iter().chain(ks_tests.iter()).collect();
    let min_p_value = all_tests.iter().map(|t| t.p_value).reduce(f64::min);
    let mean_effect_size = if all_tests.is_empty() {
        None
    } else {
        Some(all_tests.iter().map(|t| t.effect_size).sum::<f64>() / all_tests.len() as f64)
    };

    // 2. Build standardized feature matrices anchored on the training cohort.
    let (ref_features, gen_features) = build_standardized_features(
        training,
        reference,
        generated,
        &numeric_vars,
        &categorical_vars,
    )?;

    // 3. Run ML discriminators (Logistic Regression & Random Forest) for each seed.
    let mut per_seed_metrics = Vec::with_capacity(seeds.len());
    for &seed in seeds {
        let lr = train_logistic_discriminator(&ref_features, &gen_features, seed)?;
        let rf = train_random_forest_discriminator(&ref_features, &gen_features, seed)?;
        per_seed_metrics.push(SeedDiscriminatorMetrics {
            seed,
            logistic_regression: lr,
            random_forest: rf,
        });
    }

    let lr_agg = aggregate_classifier_metrics(
        &per_seed_metrics
            .iter()
            .map(|m| &m.logistic_regression)
            .collect::<Vec<_>>(),
    );
    let rf_agg = aggregate_classifier_metrics(
        &per_seed_metrics
            .iter()
            .map(|m| &m.random_forest)
            .collect::<Vec<_>>(),
    );
    let combined_mean_auc_roc = Some(0.5 * (lr_agg.mean_auc_roc + rf_agg.mean_auc_roc));

    let is_discriminable = lr_agg.mean_auc_roc >= 0.60
        || rf_agg.mean_auc_roc >= 0.60
        || min_p_value.is_some_and(|p| p < 0.01);
    let outcome = if is_discriminable {
        ForensicOutcome::Discriminable
    } else {
        ForensicOutcome::Indistinguishable
    };

    Ok(ForensicManifest {
        schema_version: "1".into(),
        discriminator_types,
        seeds: seeds.to_vec(),
        fingerprints,
        software_version: sw.package_version,
        dependency_lock_fingerprint: sw.dependency_lock_fingerprint,
        environment,
        outcome,
        aggregate_metrics: ForensicAggregateMetrics {
            training_rows: n_train_rows,
            generated_rows: n_gen_rows,
            reference_rows: n_ref_rows,
            minimum_cell_size: MINIMUM_FORENSIC_COHORT_ROWS,
            suppressed: false,
            chi_squared_tests,
            ks_tests,
            min_p_value,
            mean_effect_size,
            logistic_regression: Some(lr_agg),
            random_forest: Some(rf_agg),
            combined_mean_auc_roc,
            per_seed_metrics,
        },
        limitations,
    })
}

fn aggregate_classifier_metrics(metrics: &[&DiscriminatorMetrics]) -> AggregateClassifierMetrics {
    let n = metrics.len() as f64;
    let mean_auc_roc = metrics.iter().map(|m| m.auc_roc).sum::<f64>() / n;
    let min_auc_roc = metrics
        .iter()
        .map(|m| m.auc_roc)
        .reduce(f64::min)
        .unwrap_or(0.5);
    let max_auc_roc = metrics
        .iter()
        .map(|m| m.auc_roc)
        .reduce(f64::max)
        .unwrap_or(0.5);
    let mean_accuracy = metrics.iter().map(|m| m.accuracy).sum::<f64>() / n;
    let mean_precision = metrics.iter().map(|m| m.precision).sum::<f64>() / n;
    let mean_recall = metrics.iter().map(|m| m.recall).sum::<f64>() / n;

    AggregateClassifierMetrics {
        mean_auc_roc,
        min_auc_roc,
        max_auc_roc,
        mean_accuracy,
        mean_precision,
        mean_recall,
    }
}

const MISSING_CATEGORICAL_LEVEL: &str = "__JANUS_MISSING__";

fn extract_categorical_level(row: &ForensicRow, var: &str) -> Option<String> {
    match row.categorical_fields.get(var) {
        Some(Some(val)) => {
            let trimmed = val.trim();
            if trimmed.is_empty() || trimmed.eq_ignore_ascii_case("NA") {
                None
            } else {
                Some(trimmed.to_string())
            }
        }
        _ => None,
    }
}

fn build_standardized_features(
    training: &ForensicCohort,
    reference: &ForensicCohort,
    generated: &ForensicCohort,
    numeric_vars: &BTreeSet<String>,
    categorical_vars: &BTreeSet<String>,
) -> Result<(Array2<f64>, Array2<f64>), Error> {
    let num_list: Vec<String> = numeric_vars.iter().cloned().collect();

    let mut cat_levels: Vec<(String, String)> = Vec::new();
    for var in categorical_vars {
        let mut levels = BTreeSet::new();
        let mut has_missing = false;
        for cohort in [training, reference, generated] {
            for row in &cohort.rows {
                match extract_categorical_level(row, var) {
                    Some(level) => {
                        levels.insert(level);
                    }
                    None => {
                        has_missing = true;
                    }
                }
            }
        }
        if has_missing {
            levels.insert(MISSING_CATEGORICAL_LEVEL.to_string());
        }
        // Drop first level as reference category to prevent exact collinearity.
        if levels.len() > 1 {
            for level in levels.into_iter().skip(1) {
                cat_levels.push((var.clone(), level));
            }
        }
    }

    // Compute training mean and std for numeric columns so scaling uses training statistics only,
    // and track numeric columns with missing/non-finite entries for explicit missing indicators.
    let mut means = vec![0.0_f64; num_list.len()];
    let mut stds = vec![1.0_f64; num_list.len()];
    let mut numeric_missing_indices: Vec<usize> = Vec::new();

    for (col_idx, var) in num_list.iter().enumerate() {
        let vals: Vec<f64> = training
            .rows
            .iter()
            .filter_map(|r| r.numeric_fields.get(var).and_then(|v| *v))
            .filter(|v| v.is_finite())
            .collect();
        if !vals.is_empty() {
            let mean = vals.iter().sum::<f64>() / vals.len() as f64;
            let var_sum = vals.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / vals.len() as f64;
            let std = var_sum.sqrt();
            means[col_idx] = mean;
            stds[col_idx] = if std > 1e-9 { std } else { 1.0 };
        }

        let has_missing = [training, reference, generated].iter().any(|cohort| {
            cohort.rows.iter().any(|row| {
                row.numeric_fields
                    .get(var)
                    .and_then(|v| *v)
                    .filter(|v| v.is_finite())
                    .is_none()
            })
        });
        if has_missing {
            numeric_missing_indices.push(col_idx);
        }
    }

    let n_cols = num_list.len() + numeric_missing_indices.len() + cat_levels.len();
    if n_cols == 0 {
        return Err(Error::Input(
            "forensic cohorts must contain at least one observed numeric or multi-level categorical feature".into(),
        ));
    }

    let extract_matrix = |cohort: &ForensicCohort| -> Result<Array2<f64>, Error> {
        let mut data = Vec::with_capacity(cohort.rows.len() * n_cols);
        for row in &cohort.rows {
            for (col_idx, var) in num_list.iter().enumerate() {
                let standardized = match row
                    .numeric_fields
                    .get(var)
                    .and_then(|v| *v)
                    .filter(|v| v.is_finite())
                {
                    Some(v) => (v - means[col_idx]) / stds[col_idx],
                    None => 0.0,
                };
                data.push(standardized);
            }
            for &col_idx in &numeric_missing_indices {
                let var = &num_list[col_idx];
                let is_missing = row
                    .numeric_fields
                    .get(var)
                    .and_then(|v| *v)
                    .filter(|v| v.is_finite())
                    .is_none();
                data.push(if is_missing { 1.0 } else { 0.0 });
            }
            for (var, level) in &cat_levels {
                let observed = extract_categorical_level(row, var);
                let row_level = observed.as_deref().unwrap_or(MISSING_CATEGORICAL_LEVEL);
                data.push(if row_level == level { 1.0 } else { 0.0 });
            }
        }
        Array2::from_shape_vec((cohort.rows.len(), n_cols), data)
            .map_err(|e| Error::Input(format!("failed to construct feature matrix: {e}")))
    };

    Ok((extract_matrix(reference)?, extract_matrix(generated)?))
}

#[derive(Debug, Clone)]
enum DecisionNode {
    Leaf {
        probability: f64,
    },
    Split {
        feature_idx: usize,
        threshold: f64,
        left: Box<DecisionNode>,
        right: Box<DecisionNode>,
    },
}

impl DecisionNode {
    fn predict_prob(&self, row: &[f64]) -> f64 {
        match self {
            Self::Leaf { probability } => *probability,
            Self::Split {
                feature_idx,
                threshold,
                left,
                right,
            } => {
                if row[*feature_idx] <= *threshold {
                    left.predict_prob(row)
                } else {
                    right.predict_prob(row)
                }
            }
        }
    }
}

fn gini_impurity(count_pos: usize, total: usize) -> f64 {
    if total == 0 {
        return 0.0;
    }
    let p = count_pos as f64 / total as f64;
    2.0 * p * (1.0 - p)
}

fn build_cart_tree(
    x: &Array2<f64>,
    y: &[usize],
    sample_indices: &[usize],
    depth: usize,
    max_depth: usize,
    rng: &mut StdRng,
) -> DecisionNode {
    let total = sample_indices.len();
    let pos_count = sample_indices.iter().filter(|&&i| y[i] == 1).count();
    let probability = if total > 0 {
        pos_count as f64 / total as f64
    } else {
        0.5
    };

    if depth >= max_depth || total < 4 || pos_count == 0 || pos_count == total {
        return DecisionNode::Leaf { probability };
    }

    let n_cols = x.ncols();
    let n_try = (n_cols as f64).sqrt().ceil().max(1.0) as usize;
    let mut candidate_features: Vec<usize> = (0..n_cols).collect();
    candidate_features.shuffle(rng);
    candidate_features.truncate(n_try);

    let parent_gini = gini_impurity(pos_count, total);
    let mut best_gain = 0.0_f64;
    let mut best_split: Option<(usize, f64, Vec<usize>, Vec<usize>)> = None;

    for &feat in &candidate_features {
        let mut vals: Vec<f64> = sample_indices.iter().map(|&i| x[[i, feat]]).collect();
        vals.sort_by(|a, b| a.total_cmp(b));
        vals.dedup_by(|a, b| (*a - *b).abs() < 1e-12);
        if vals.len() < 2 {
            continue;
        }

        // Evaluate up to 16 quantile candidate thresholds for efficiency and determinism.
        let step = (vals.len() - 1).div_ceil(16).max(1);
        for idx in (0..(vals.len() - 1)).step_by(step) {
            let threshold = 0.5 * (vals[idx] + vals[idx + 1]);
            let mut left_idx = Vec::new();
            let mut right_idx = Vec::new();
            let mut left_pos = 0usize;
            let mut right_pos = 0usize;

            for &row_i in sample_indices {
                if x[[row_i, feat]] <= threshold {
                    left_idx.push(row_i);
                    if y[row_i] == 1 {
                        left_pos += 1;
                    }
                } else {
                    right_idx.push(row_i);
                    if y[row_i] == 1 {
                        right_pos += 1;
                    }
                }
            }

            if left_idx.is_empty() || right_idx.is_empty() {
                continue;
            }

            let w_left = left_idx.len() as f64 / total as f64;
            let w_right = right_idx.len() as f64 / total as f64;
            let child_gini = w_left * gini_impurity(left_pos, left_idx.len())
                + w_right * gini_impurity(right_pos, right_idx.len());
            let gain = parent_gini - child_gini;

            if gain > best_gain + 1e-12 {
                best_gain = gain;
                best_split = Some((feat, threshold, left_idx, right_idx));
            }
        }
    }

    if let Some((feature_idx, threshold, left_idx, right_idx)) = best_split {
        let left = Box::new(build_cart_tree(x, y, &left_idx, depth + 1, max_depth, rng));
        let right = Box::new(build_cart_tree(x, y, &right_idx, depth + 1, max_depth, rng));
        DecisionNode::Split {
            feature_idx,
            threshold,
            left,
            right,
        }
    } else {
        DecisionNode::Leaf { probability }
    }
}

/// Train an in-memory Logistic Regression discriminator (80/20 split, fixed seed) on numeric features.
/// Label: 0 = reference (real), 1 = generated (synthetic).
pub fn train_logistic_discriminator(
    real_features: &Array2<f64>,
    synthetic_features: &Array2<f64>,
    seed: u64,
) -> Result<DiscriminatorMetrics, Error> {
    let n_real = real_features.nrows();
    let n_synth = synthetic_features.nrows();
    let n_cols = real_features.ncols();
    if n_real == 0 || n_synth == 0 || n_cols == 0 || synthetic_features.ncols() != n_cols {
        return Err(Error::Input(
            "logistic regression discriminator requires non-empty feature matrices with matching columns"
                .into(),
        ));
    }

    let total = n_real + n_synth;
    let combined_x =
        ndarray::concatenate(Axis(0), &[real_features.view(), synthetic_features.view()])
            .map_err(|e| Error::Input(format!("failed to concatenate features: {e}")))?;
    let mut labels = Vec::with_capacity(total);
    labels.extend(std::iter::repeat_n(0usize, n_real));
    labels.extend(std::iter::repeat_n(1usize, n_synth));

    let mut indices: Vec<usize> = (0..total).collect();
    let mut rng = StdRng::seed_from_u64(seed);
    indices.shuffle(&mut rng);

    let n_train = ((total as f64) * 0.8).round() as usize;
    let n_train = n_train.clamp(1, total.saturating_sub(1));
    let n_test = total - n_train;
    if n_test == 0 {
        return Err(Error::Input(
            "logistic regression discriminator requires at least 2 samples for 80/20 split".into(),
        ));
    }

    let train_idx = &indices[..n_train];
    let test_idx = &indices[n_train..];

    let x_train = combined_x.select(Axis(0), train_idx);
    let y_train = Array1::from_iter(train_idx.iter().map(|&i| labels[i]));
    let x_test = combined_x.select(Axis(0), test_idx);
    let y_test: Vec<usize> = test_idx.iter().map(|&i| labels[i]).collect();

    let train_ds = Dataset::new(x_train, y_train);
    let model = LogisticRegression::default()
        .max_iterations(200)
        .fit(&train_ds)
        .map_err(|e| Error::Input(format!("logistic regression fit failed: {e}")))?;

    let raw_probs = model.predict_probabilities(&x_test);
    let preds = model.predict(&x_test);

    // Orient probabilities so they always represent P(label == 1).
    let invert_probs = raw_probs
        .iter()
        .zip(preds.iter())
        .any(|(&p, &pred)| (p > 0.5 && pred == 0) || (p < 0.5 && pred == 1));
    let probs: Vec<f64> = raw_probs
        .iter()
        .map(|&p| if invert_probs { 1.0 - p } else { p })
        .collect();

    let mut tp = 0usize;
    let mut fp = 0usize;
    let mut tn = 0usize;
    let mut fn_ = 0usize;
    for (&pred, &truth) in preds.iter().zip(y_test.iter()) {
        match (pred, truth) {
            (1, 1) => tp += 1,
            (1, 0) => fp += 1,
            (0, 0) => tn += 1,
            (0, 1) => fn_ += 1,
            _ => {}
        }
    }

    let accuracy = (tp + tn) as f64 / n_test as f64;
    let precision = if tp + fp > 0 {
        tp as f64 / (tp + fp) as f64
    } else {
        0.0
    };
    let recall = if tp + fn_ > 0 {
        tp as f64 / (tp + fn_) as f64
    } else {
        0.0
    };
    let auc_roc = compute_auc_roc(&probs, &y_test);

    Ok(DiscriminatorMetrics {
        auc_roc,
        accuracy,
        precision,
        recall,
        n_train,
        n_test,
    })
}

/// Train an in-memory Random Forest discriminator (80/20 split, fixed seed) on numeric features.
/// Label: 0 = reference (real), 1 = generated (synthetic).
pub fn train_random_forest_discriminator(
    real_features: &Array2<f64>,
    synthetic_features: &Array2<f64>,
    seed: u64,
) -> Result<DiscriminatorMetrics, Error> {
    let n_real = real_features.nrows();
    let n_synth = synthetic_features.nrows();
    let n_cols = real_features.ncols();
    if n_real == 0 || n_synth == 0 || n_cols == 0 || synthetic_features.ncols() != n_cols {
        return Err(Error::Input(
            "random forest discriminator requires non-empty feature matrices with matching columns"
                .into(),
        ));
    }

    let total = n_real + n_synth;
    let combined_x =
        ndarray::concatenate(Axis(0), &[real_features.view(), synthetic_features.view()])
            .map_err(|e| Error::Input(format!("failed to concatenate features: {e}")))?;
    let mut labels = Vec::with_capacity(total);
    labels.extend(std::iter::repeat_n(0usize, n_real));
    labels.extend(std::iter::repeat_n(1usize, n_synth));

    let mut indices: Vec<usize> = (0..total).collect();
    let mut rng = StdRng::seed_from_u64(seed);
    indices.shuffle(&mut rng);

    let n_train = ((total as f64) * 0.8).round() as usize;
    let n_train = n_train.clamp(1, total.saturating_sub(1));
    let n_test = total - n_train;
    if n_test == 0 {
        return Err(Error::Input(
            "random forest discriminator requires at least 2 samples for 80/20 split".into(),
        ));
    }

    let train_idx = &indices[..n_train];
    let test_idx = &indices[n_train..];

    let x_train = combined_x.select(Axis(0), train_idx);
    let y_train: Vec<usize> = train_idx.iter().map(|&i| labels[i]).collect();
    let x_test = combined_x.select(Axis(0), test_idx);
    let y_test: Vec<usize> = test_idx.iter().map(|&i| labels[i]).collect();

    let n_trees = 25usize;
    let max_depth = 4usize;
    let mut trees = Vec::with_capacity(n_trees);
    for _ in 0..n_trees {
        let bootstrap: Vec<usize> = (0..n_train).map(|_| rng.gen_range(0..n_train)).collect();
        trees.push(build_cart_tree(
            &x_train, &y_train, &bootstrap, 0, max_depth, &mut rng,
        ));
    }

    let mut probs = Vec::with_capacity(n_test);
    let mut tp = 0usize;
    let mut fp = 0usize;
    let mut tn = 0usize;
    let mut fn_ = 0usize;

    for (row_idx, &truth) in y_test.iter().enumerate() {
        let row_slice: Vec<f64> = (0..n_cols).map(|c| x_test[[row_idx, c]]).collect();
        let prob = trees
            .iter()
            .map(|t| t.predict_prob(&row_slice))
            .sum::<f64>()
            / n_trees as f64;
        probs.push(prob);
        let pred = usize::from(prob >= 0.5);
        match (pred, truth) {
            (1, 1) => tp += 1,
            (1, 0) => fp += 1,
            (0, 0) => tn += 1,
            (0, 1) => fn_ += 1,
            _ => {}
        }
    }

    let accuracy = (tp + tn) as f64 / n_test as f64;
    let precision = if tp + fp > 0 {
        tp as f64 / (tp + fp) as f64
    } else {
        0.0
    };
    let recall = if tp + fn_ > 0 {
        tp as f64 / (tp + fn_) as f64
    } else {
        0.0
    };
    let auc_roc = compute_auc_roc(&probs, &y_test);

    Ok(DiscriminatorMetrics {
        auc_roc,
        accuracy,
        precision,
        recall,
        n_train,
        n_test,
    })
}

fn compute_auc_roc(scores: &[f64], labels: &[usize]) -> f64 {
    let n_pos = labels.iter().filter(|&&y| y == 1).count() as f64;
    let n_neg = labels.iter().filter(|&&y| y == 0).count() as f64;
    if n_pos == 0.0 || n_neg == 0.0 || scores.len() != labels.len() {
        return 0.5;
    }

    let mut pairs: Vec<(f64, usize)> = scores.iter().copied().zip(labels.iter().copied()).collect();
    pairs.sort_by(|a, b| b.0.total_cmp(&a.0));

    let mut tp = 0.0;
    let mut fp = 0.0;
    let mut prev_tp = 0.0;
    let mut prev_fp = 0.0;
    let mut auc = 0.0;

    for (_, label) in pairs {
        if label == 1 {
            tp += 1.0;
        } else {
            fp += 1.0;
        }
        auc += (fp - prev_fp) * (tp + prev_tp) / 2.0;
        prev_tp = tp;
        prev_fp = fp;
    }

    auc / (n_pos * n_neg)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn build_fictional_cohort(
        cohort_id: &str,
        n_rows: usize,
        seed: u64,
        mean_shift: f64,
        high_grade_prob: f64,
    ) -> ForensicCohort {
        let mut rng = StdRng::seed_from_u64(seed);
        let mut rows = Vec::with_capacity(n_rows);
        for idx in 0..n_rows {
            let u1: f64 = rng.gen_range(1e-6..1.0);
            let u2: f64 = rng.gen_range(0.0..1.0);
            let z0 = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
            let z1 = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).sin();

            let age = 55.0 + 10.0 * z0 + mean_shift * 12.0;
            let biomarker = 2.5 + 0.8 * z1 + mean_shift * 1.5;
            let is_high = rng.gen_bool(high_grade_prob.clamp(0.0, 1.0));
            let grade_cat = if is_high { "high" } else { "low" };

            rows.push(ForensicRow {
                patient_key: format!("{cohort_id}-p{idx:03}"),
                numeric_fields: BTreeMap::from([
                    ("age_years".into(), Some(age)),
                    ("inflammatory_marker".into(), Some(biomarker)),
                ]),
                categorical_fields: BTreeMap::from([(
                    "severity_band".into(),
                    Some(grade_cat.into()),
                )]),
            });
        }
        ForensicCohort::new(cohort_id, rows)
    }

    #[test]
    fn small_cell_suppression_returns_insufficient_without_running_discriminators() {
        let training = build_fictional_cohort("train-small", 9, 10, 0.0, 0.25);
        let generated = build_fictional_cohort("gen-small", 50, 20, 1.2, 0.75);
        let reference = build_fictional_cohort("ref-small", 50, 30, 0.0, 0.25);

        let manifest =
            orchestrate_forensic_suite(&training, &generated, &reference, &[11, 22, 33]).unwrap();
        manifest.validate().unwrap();

        assert_eq!(manifest.outcome, ForensicOutcome::Insufficient);
        assert!(manifest.aggregate_metrics.suppressed);
        assert!(manifest.aggregate_metrics.chi_squared_tests.is_empty());
        assert!(manifest.aggregate_metrics.ks_tests.is_empty());
        assert!(manifest.aggregate_metrics.logistic_regression.is_none());
        assert!(manifest.aggregate_metrics.random_forest.is_none());
        assert!(manifest.aggregate_metrics.per_seed_metrics.is_empty());
    }

    #[test]
    fn deterministic_fictional_50_row_cohorts_with_3_seeds_verify_auc_and_manifest() {
        let seeds = [101_u64, 202_u64, 303_u64];
        let training = build_fictional_cohort("fictional-train", 50, 1, 0.0, 0.20);
        let reference = build_fictional_cohort("fictional-ref", 50, 2, 0.0, 0.20);
        // Known distributional difference in both continuous and categorical variables
        let generated = build_fictional_cohort("fictional-gen", 50, 3, 1.5, 0.80);

        let manifest =
            orchestrate_forensic_suite(&training, &generated, &reference, &seeds).unwrap();
        manifest.validate().unwrap();

        // Verify deterministic replay with identical inputs and seeds
        let replay = orchestrate_forensic_suite(&training, &generated, &reference, &seeds).unwrap();
        assert_eq!(manifest, replay);

        // Verify serialization round-trip (JSON and YAML)
        let json = manifest.to_json().unwrap();
        let from_json: ForensicManifest = serde_json::from_str(&json).unwrap();
        assert_eq!(manifest.outcome, from_json.outcome);
        assert_eq!(manifest.seeds, vec![101, 202, 303]);

        // Verify no patient keys are leaked into serialized manifest
        assert!(!json.contains("fictional-ref-p000"));
        assert!(!json.contains("fictional-gen-p000"));

        // Verify discriminator types, outcome, and AUC > 0.7
        assert_eq!(
            manifest.discriminator_types,
            vec![
                DiscriminatorType::ChiSquared,
                DiscriminatorType::KolmogorovSmirnov,
                DiscriminatorType::LogisticRegression,
                DiscriminatorType::RandomForest,
            ]
        );
        assert_eq!(manifest.outcome, ForensicOutcome::Discriminable);
        assert!(!manifest.aggregate_metrics.suppressed);
        assert_eq!(manifest.aggregate_metrics.training_rows, 50);
        assert_eq!(manifest.aggregate_metrics.generated_rows, 50);
        assert_eq!(manifest.aggregate_metrics.reference_rows, 50);
        assert_eq!(manifest.aggregate_metrics.per_seed_metrics.len(), 3);
        assert_eq!(manifest.aggregate_metrics.chi_squared_tests.len(), 1);
        assert_eq!(manifest.aggregate_metrics.ks_tests.len(), 2);

        let lr = manifest
            .aggregate_metrics
            .logistic_regression
            .as_ref()
            .unwrap();
        let rf = manifest.aggregate_metrics.random_forest.as_ref().unwrap();
        let combined_auc = manifest.aggregate_metrics.combined_mean_auc_roc.unwrap();

        assert!(
            lr.mean_auc_roc > 0.7,
            "expected logistic regression mean AUC > 0.7, got {}",
            lr.mean_auc_roc
        );
        assert!(
            rf.mean_auc_roc > 0.7,
            "expected random forest mean AUC > 0.7, got {}",
            rf.mean_auc_roc
        );
        assert!(
            combined_auc > 0.7,
            "expected combined mean AUC > 0.7, got {}",
            combined_auc
        );
    }

    #[test]
    fn indistinguishable_outcome_and_manifest_validation_rejects_incomplete_fields() {
        let seeds = [11_u64, 22_u64, 33_u64];
        let training = build_fictional_cohort("same-train", 50, 100, 0.0, 0.50);
        let reference = build_fictional_cohort("same-ref", 50, 100, 0.0, 0.50);
        let generated = build_fictional_cohort("same-gen", 50, 100, 0.0, 0.50);

        let mut manifest =
            orchestrate_forensic_suite(&training, &generated, &reference, &seeds).unwrap();
        manifest.validate().unwrap();
        assert_eq!(manifest.outcome, ForensicOutcome::Indistinguishable);

        manifest.software_version.clear();
        assert!(manifest.validate().is_err());
    }

    #[test]
    fn mixed_numeric_and_categorical_cohorts_detect_categorical_shift_even_when_numeric_matches() {
        let seeds = [101_u64, 202_u64, 303_u64];
        let training = build_fictional_cohort("mix-train", 60, 11, 0.0, 0.10);
        let reference = build_fictional_cohort("mix-ref", 60, 22, 0.0, 0.10);
        // Same numeric distribution (mean_shift = 0.0), only categorical distribution differs.
        let generated = build_fictional_cohort("mix-gen", 60, 33, 0.0, 0.90);

        let manifest =
            orchestrate_forensic_suite(&training, &generated, &reference, &seeds).unwrap();
        manifest.validate().unwrap();

        assert_eq!(manifest.outcome, ForensicOutcome::Discriminable);
        let lr = manifest
            .aggregate_metrics
            .logistic_regression
            .as_ref()
            .unwrap();
        let rf = manifest.aggregate_metrics.random_forest.as_ref().unwrap();
        assert!(
            lr.mean_auc_roc > 0.7,
            "expected LR AUC > 0.7 from categorical shift alongside numeric columns, got {}",
            lr.mean_auc_roc
        );
        assert!(
            rf.mean_auc_roc > 0.7,
            "expected RF AUC > 0.7 from categorical shift alongside numeric columns, got {}",
            rf.mean_auc_roc
        );
    }

    #[test]
    fn missing_numeric_values_are_distinguished_from_observed_zero_via_explicit_indicators() {
        let mut train_rows = Vec::new();
        let mut ref_rows = Vec::new();
        let mut gen_rows = Vec::new();
        for idx in 0..40 {
            train_rows.push(ForensicRow {
                patient_key: format!("tr-{idx:03}"),
                numeric_fields: BTreeMap::from([("biomarker".into(), Some(0.0))]),
                categorical_fields: BTreeMap::new(),
            });
            ref_rows.push(ForensicRow {
                patient_key: format!("rf-{idx:03}"),
                numeric_fields: BTreeMap::from([("biomarker".into(), Some(0.0))]),
                categorical_fields: BTreeMap::new(),
            });
            gen_rows.push(ForensicRow {
                patient_key: format!("gn-{idx:03}"),
                numeric_fields: BTreeMap::from([("biomarker".into(), None)]),
                categorical_fields: BTreeMap::new(),
            });
        }
        let training = ForensicCohort::new("zero-train", train_rows);
        let reference = ForensicCohort::new("zero-ref", ref_rows);
        let generated = ForensicCohort::new("na-gen", gen_rows);

        let num_vars = BTreeSet::from(["biomarker".to_string()]);
        let cat_vars = BTreeSet::new();
        let (ref_x, gen_x) =
            build_standardized_features(&training, &reference, &generated, &num_vars, &cat_vars)
                .unwrap();
        // Column 0: standardized value (0.0); Column 1: explicit is_missing indicator (0.0 vs 1.0)
        assert_eq!(ref_x.ncols(), 2);
        assert_eq!(gen_x.ncols(), 2);
        assert_eq!(ref_x[[0, 1]], 0.0);
        assert_eq!(gen_x[[0, 1]], 1.0);

        // NumericTable observed_column also keeps Some(0.0) and excludes None
        let mut real_mat = ndarray::Array2::from_elem((40, 1), Some(0.0));
        let synth_mat = ndarray::Array2::from_elem((40, 1), None);
        real_mat[[0, 0]] = Some(0.0);
        let real_tbl = crate::tabular::NumericTable {
            columns: vec!["biomarker".into()],
            data: real_mat,
        };
        let synth_tbl = crate::tabular::NumericTable {
            columns: vec!["biomarker".into()],
            data: synth_mat,
        };
        assert_eq!(real_tbl.observed_column(0).len(), 40);
        assert!(synth_tbl.observed_column(0).is_empty());
        assert_eq!(synth_tbl.missing_count(0), 40);

        let manifest =
            orchestrate_forensic_suite(&training, &generated, &reference, &[11, 22, 33]).unwrap();
        manifest.validate().unwrap();
        assert_eq!(manifest.outcome, ForensicOutcome::Discriminable);
        let lr = manifest
            .aggregate_metrics
            .logistic_regression
            .as_ref()
            .unwrap();
        let rf = manifest.aggregate_metrics.random_forest.as_ref().unwrap();
        assert!(lr.mean_auc_roc > 0.7, "expected LR AUC > 0.7, got {}", lr.mean_auc_roc);
        assert!(rf.mean_auc_roc > 0.7, "expected RF AUC > 0.7, got {}", rf.mean_auc_roc);
    }

    #[test]
    fn from_candidate_rows_preserves_missing_tokens_as_none_and_validates_schema() {
        use crate::candidate::Row;

        let rows = vec![
            Row::from_pairs([
                ("age", "55.5"),
                ("score", "0"),
                ("grade", "high"),
                ("all_missing", ""),
            ]),
            Row::from_pairs([
                ("age", ""),
                ("score", "NA"),
                ("grade", "na"),
                ("all_missing", "NA"),
            ]),
            Row::from_pairs([
                ("age", "62"),
                ("score", " 3.0 "),
                ("grade", ""),
                ("all_missing", "  "),
            ]),
        ];

        let cohort = ForensicCohort::from_candidate_rows("cand", &rows).unwrap();
        cohort.validate_keys().unwrap();
        assert_eq!(cohort.rows.len(), 3);
        assert_eq!(cohort.rows[0].patient_key, "cand-row-000000");
        assert_eq!(cohort.rows[1].patient_key, "cand-row-000001");
        assert_eq!(cohort.rows[2].patient_key, "cand-row-000002");

        assert_eq!(cohort.rows[0].numeric_fields.get("age"), Some(&Some(55.5)));
        assert_eq!(cohort.rows[1].numeric_fields.get("age"), Some(&None));
        assert_eq!(cohort.rows[0].numeric_fields.get("score"), Some(&Some(0.0)));
        assert_eq!(cohort.rows[1].numeric_fields.get("score"), Some(&None));
        assert_eq!(cohort.rows[2].numeric_fields.get("score"), Some(&Some(3.0)));

        assert_eq!(
            cohort.rows[0].categorical_fields.get("grade"),
            Some(&Some("high".to_string()))
        );
        assert_eq!(cohort.rows[1].categorical_fields.get("grade"), Some(&None));
        assert_eq!(cohort.rows[2].categorical_fields.get("grade"), Some(&None));
        assert_eq!(
            cohort.rows[0].categorical_fields.get("all_missing"),
            Some(&None)
        );

        // Reject empty rows, empty cohort_id, and inconsistent row schemas
        assert!(ForensicCohort::from_candidate_rows("cand", &[]).is_err());
        assert!(ForensicCohort::from_candidate_rows(" ", &rows).is_err());
        let bad_rows = vec![
            Row::from_pairs([("a", "1"), ("b", "x")]),
            Row::from_pairs([("a", "2"), ("c", "y")]),
        ];
        assert!(ForensicCohort::from_candidate_rows("bad", &bad_rows).is_err());
    }

    #[test]
    fn categorical_missingness_alongside_numeric_is_encoded_with_explicit_missing_level() {
        let mut train_rows = Vec::new();
        let mut ref_rows = Vec::new();
        let mut gen_rows = Vec::new();
        for idx in 0..40 {
            let x = (idx as f64) * 0.1;
            train_rows.push(ForensicRow {
                patient_key: format!("cm-tr-{idx:03}"),
                numeric_fields: BTreeMap::from([("x".into(), Some(x))]),
                categorical_fields: BTreeMap::from([(
                    "cat".into(),
                    Some(if idx % 2 == 0 { "A" } else { "B" }.to_string()),
                )]),
            });
            ref_rows.push(ForensicRow {
                patient_key: format!("cm-rf-{idx:03}"),
                numeric_fields: BTreeMap::from([("x".into(), Some(x))]),
                categorical_fields: BTreeMap::from([(
                    "cat".into(),
                    Some(if idx % 2 == 0 { "A" } else { "B" }.to_string()),
                )]),
            });
            gen_rows.push(ForensicRow {
                patient_key: format!("cm-gn-{idx:03}"),
                numeric_fields: BTreeMap::from([("x".into(), Some(x))]),
                categorical_fields: BTreeMap::from([("cat".into(), None)]),
            });
        }
        let training = ForensicCohort::new("cm-train", train_rows);
        let reference = ForensicCohort::new("cm-ref", ref_rows);
        let generated = ForensicCohort::new("cm-gen", gen_rows);

        let manifest =
            orchestrate_forensic_suite(&training, &generated, &reference, &[11, 22, 33]).unwrap();
        manifest.validate().unwrap();
        assert_eq!(manifest.outcome, ForensicOutcome::Discriminable);
        let lr = manifest
            .aggregate_metrics
            .logistic_regression
            .as_ref()
            .unwrap();
        let rf = manifest.aggregate_metrics.random_forest.as_ref().unwrap();
        assert!(lr.mean_auc_roc > 0.9, "expected LR AUC > 0.9, got {}", lr.mean_auc_roc);
        assert!(rf.mean_auc_roc > 0.9, "expected RF AUC > 0.9, got {}", rf.mean_auc_roc);
    }
}
