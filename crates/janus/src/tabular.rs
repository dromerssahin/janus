//! Versioned CSV boundary for canonical event datasets.
use crate::{validate, Event, Fixture, Observation, Study};
use serde::{Deserialize, Serialize};
use std::{
    collections::{BTreeMap, BTreeSet, HashSet},
    error::Error,
    path::Path,
};

/// Explicit input mapping. Every event is represented by a CSV row; absence is
/// represented by no row, while a missing observation has a missing reason.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CsvMapping {
    pub schema_version: String,
    pub study: Study,
    pub pre_deidentified: bool,
    pub permitted_data_use: String,
    pub data_use_permitted: bool,
    pub columns: BTreeMap<String, String>,
}

/// Minimal flat cohort schema for aggregate-only descriptive reports.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FlatCohortConfig {
    pub schema_version: String,
    pub allocation_column: String,
    pub allocation_values: Vec<String>,
    #[serde(
        default = "default_flat_row_key",
        skip_serializing_if = "is_default_flat_row_key"
    )]
    pub row_key_column: String,
    #[serde(default)]
    pub generator: FlatGenerator,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub dependencies: Vec<FlatDependency>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub numeric_associations: Vec<FlatDependency>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub joint_families: Vec<Vec<String>>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub utility_tasks: Vec<FlatUtilityTask>,
    #[serde(default, skip_serializing_if = "flat_is_false")]
    pub exact_match_diagnostics: bool,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub numeric_smoothing: Vec<FlatNumericSmoothing>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub numeric_interpolation: Vec<String>,
    #[serde(default)]
    pub sparse_policy: FlatSparsePolicy,
    #[serde(default = "default_flat_seed")]
    pub split_seed: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub generator_seed: Option<u64>,
    #[serde(default = "default_flat_proportion")]
    pub development_proportion: f64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub development_validation_proportion: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub development_validation_seed: Option<u64>,
    #[serde(default)]
    pub expected_source_columns: Vec<String>,
    pub thresholds: FlatThresholds,
    #[serde(default = "flat_bootstrap_replicates")]
    pub bootstrap_replicates: usize,
    #[serde(default = "flat_bootstrap_confidence")]
    pub bootstrap_confidence_level: f64,
    pub variables: Vec<FlatVariable>,
}

fn default_flat_row_key() -> String {
    "no".into()
}

fn flat_is_false(value: &bool) -> bool {
    !value
}

fn is_default_flat_row_key(value: &str) -> bool {
    value == "no"
}

/// The marginal model independently resamples each fitted development column.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum FlatGenerator {
    #[default]
    RowBootstrap,
    IndependentMarginals,
    #[serde(alias = "conditional_empirical")]
    ConditionalCategorical,
    LearnedCategoricalForest,
    LearnedMixedForest,
    LearnedBinnedForest,
    RegularizedBinnedForest,
    LearnedBinnedDag,
    InteractionBinnedDag,
}

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum FlatSparsePolicy {
    #[default]
    Fail,
    DropLastParent,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FlatDependency {
    pub parent: String,
    pub child: String,
}

/// A descriptive binary prediction task with declared available predictors.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FlatUtilityTask {
    pub name: String,
    pub predictor: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub additional_predictors: Vec<String>,
    pub target: String,
    pub positive_values: Vec<String>,
    pub negative_values: Vec<String>,
    pub alpha: f64,
    pub predictor_available_before_target: bool,
}

/// Explicit uniform numeric perturbation in the variable's original units.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FlatNumericSmoothing {
    pub variable: String,
    pub bandwidth: f64,
    pub minimum: f64,
    pub maximum: f64,
}

fn flat_bootstrap_replicates() -> usize {
    500
}
fn flat_bootstrap_confidence() -> f64 {
    0.95
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FlatThresholds {
    pub max_missingness_rate_difference: f64,
    pub max_categorical_tvd: f64,
    pub max_numeric_quantile_difference: f64,
}

fn default_flat_seed() -> u64 {
    41
}
fn default_flat_proportion() -> f64 {
    0.8
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FlatVariable {
    pub column: String,
    pub kind: String,
    #[serde(default = "default_flat_missing_policy")]
    pub missing_policy: String,
    #[serde(default)]
    pub structural_absence_values: Vec<String>,
    #[serde(default)]
    pub allowed_values: Vec<String>,
}

/// A trained, single-group binned generator that borrows its training rows.
///
/// This model and its samples can contain sensitive information. The caller is
/// responsible for permitted data use and storage; this API performs no I/O.
pub struct FlatFittedBinnedForest<'a> {
    training: &'a [Vec<String>],
    variables: &'a [FlatVariable],
    group: String,
    fit: FlatBinnedFit,
}

/// In-memory rows and aggregate sparse-parent backoff counts.
pub struct FlatBinnedSample {
    pub rows: Vec<Vec<String>>,
    pub sparse_backoffs: Vec<FlatBackoffReport>,
}

/// In-memory synthetic rows with interpolation and sparse-backoff counts.
pub struct FlatPolicyInterpolatedSample {
    pub rows: Vec<Vec<String>>,
    pub interpolation: Vec<FlatSmoothingReport>,
    pub sparse_backoffs: Vec<FlatBackoffReport>,
}

/// In-memory synthetic rows and aggregate interpolation counts.
pub struct FlatInterpolatedSample {
    pub rows: Vec<Vec<String>>,
    pub interpolation: Vec<FlatSmoothingReport>,
}

impl<'a> FlatFittedBinnedForest<'a> {
    /// Fit from modeled columns in one group. The caller must exclude patient
    /// keys and other identifiers. Use `regularized` for penalized selection.
    pub fn fit(
        training: &'a [Vec<String>],
        variables: &'a [FlatVariable],
        group: &str,
        regularized: bool,
    ) -> Result<Self, Box<dyn Error>> {
        if training.is_empty() || variables.is_empty() || group.is_empty() || group.trim() != group
        {
            return Err("fitting requires a nonempty training group and variables".into());
        }
        let mut names = HashSet::new();
        for variable in variables {
            if variable.column.trim().is_empty()
                || variable.column.trim() != variable.column
                || !names.insert(&variable.column)
                || !["categorical", "numeric"].contains(&variable.kind.as_str())
                || !["missing", "categorical_tokens", "structural_tokens"]
                    .contains(&variable.missing_policy.as_str())
                || (variable.missing_policy == "missing"
                    && !variable.structural_absence_values.is_empty())
                || (variable.missing_policy != "missing"
                    && variable.structural_absence_values.is_empty())
                || (variable.kind == "categorical" && variable.allowed_values.is_empty())
                || (variable.kind == "numeric" && !variable.allowed_values.is_empty())
            {
                return Err("invalid fitted forest variable declaration".into());
            }
        }
        for row in training {
            if row.len() != variables.len() {
                return Err("training row width does not match variables".into());
            }
            for (value, variable) in row.iter().zip(variables) {
                if value.trim() != value {
                    return Err("training values must be trimmed".into());
                }
                if flat_is_missing(value, variable)
                    || variable.structural_absence_values.contains(value)
                {
                    continue;
                }
                if variable.kind == "categorical" {
                    if !variable.allowed_values.contains(value) {
                        return Err(format!("undeclared category for {}", variable.column).into());
                    }
                } else if !value.parse::<f64>().is_ok_and(f64::is_finite) {
                    return Err(
                        format!("invalid finite numeric value for {}", variable.column).into(),
                    );
                }
            }
        }
        let rows: Vec<_> = training.iter().collect();
        let fit = fit_flat_binned_forest(&rows, variables, group, regularized);
        Ok(Self {
            training,
            variables,
            group: group.into(),
            fit,
        })
    }

    /// Fit a greedy, penalized binned model with up to two parents per variable.
    /// Callers must enforce data permissions and remove identifiers.
    pub fn fit_dag(
        training: &'a [Vec<String>],
        variables: &'a [FlatVariable],
        group: &str,
    ) -> Result<Self, Box<dyn Error>> {
        let mut model = Self::fit(training, variables, group, true)?;
        let rows: Vec<_> = training.iter().collect();
        model.fit =
            fit_flat_binned_selected(&rows, variables, group, FlatGenerator::LearnedBinnedDag);
        Ok(model)
    }

    /// Fit a binned DAG that also searches parent pairs for interaction-only patterns.
    pub fn fit_interaction_dag(
        training: &'a [Vec<String>],
        variables: &'a [FlatVariable],
        group: &str,
    ) -> Result<Self, Box<dyn Error>> {
        let mut model = Self::fit(training, variables, group, true)?;
        let rows: Vec<_> = training.iter().collect();
        model.fit =
            fit_flat_binned_selected(&rows, variables, group, FlatGenerator::InteractionBinnedDag);
        Ok(model)
    }

    /// Return the ordered variable declarations used by this model.
    pub fn variables(&self) -> &[FlatVariable] {
        self.variables
    }

    /// Inspect the learned descriptive edges. These may themselves be sensitive.
    pub fn learned_dependencies(&self) -> &[FlatLearnedDependency] {
        &self.fit.edges
    }

    /// Draw in memory, failing when a sampled parent combination lacks support.
    pub fn sample(&self, seed: u64, count: usize) -> Result<Vec<Vec<String>>, Box<dyn Error>> {
        Ok(self
            .sample_with_policy(seed, count, FlatSparsePolicy::Fail)?
            .rows)
    }

    /// Draw in memory using an explicit sparse policy and return backoff counts.
    pub fn sample_with_policy(
        &self,
        seed: u64,
        count: usize,
        policy: FlatSparsePolicy,
    ) -> Result<FlatBinnedSample, Box<dyn Error>> {
        if count > 1_000_000 {
            return Err("sample count exceeds one million rows".into());
        }
        let mut state = seed ^ flat_hash(seed, &self.group);
        let training: Vec<_> = self.training.iter().collect();
        let mut counts = vec![BTreeMap::new(); self.variables.len()];
        let rows = (0..count)
            .map(|_| {
                sample_flat_binned_checked(&training, &self.fit, &mut state, policy, &mut counts)
            })
            .collect::<Result<Vec<_>, _>>()?;
        let sparse_backoffs = counts
            .into_iter()
            .enumerate()
            .filter(|(_, count)| !count.is_empty())
            .map(|(i, dropped_parent_counts)| FlatBackoffReport {
                allocation: self.group.clone(),
                child: self.variables[i].column.clone(),
                dropped_parent_counts,
            })
            .collect();
        Ok(FlatBinnedSample {
            rows,
            sparse_backoffs,
        })
    }

    /// Draw in memory and interpolate explicitly selected continuous columns.
    pub fn sample_interpolated(
        &self,
        seed: u64,
        count: usize,
        columns: &[String],
    ) -> Result<FlatInterpolatedSample, Box<dyn Error>> {
        let sample =
            self.sample_interpolated_with_policy(seed, count, columns, FlatSparsePolicy::Fail)?;
        Ok(FlatInterpolatedSample {
            rows: sample.rows,
            interpolation: sample.interpolation,
        })
    }

    /// Draw and interpolate with an explicit sparse-parent policy and counts.
    pub fn sample_interpolated_with_policy(
        &self,
        seed: u64,
        count: usize,
        columns: &[String],
        policy: FlatSparsePolicy,
    ) -> Result<FlatPolicyInterpolatedSample, Box<dyn Error>> {
        let mut seen = HashSet::new();
        for column in columns {
            let variable = self
                .variables
                .iter()
                .find(|v| v.column == *column)
                .ok_or("unknown numeric interpolation variable")?;
            if variable.kind != "numeric" || !seen.insert(column) {
                return Err("numeric interpolation requires unique numeric variables".into());
            }
        }
        let mut sample = self.sample_with_policy(seed, count, policy)?;
        let training: Vec<_> = self.training.iter().collect();
        let interpolation = apply_flat_numeric_interpolation(
            &mut sample.rows,
            &training,
            self.variables,
            columns,
            &self.fit,
            &self.group,
            seed,
        );
        Ok(FlatPolicyInterpolatedSample {
            rows: sample.rows,
            interpolation,
            sparse_backoffs: sample.sparse_backoffs,
        })
    }
}

/// In-memory binned generators for caller-partitioned groups.
pub struct FlatFittedStratifiedForest<'a> {
    groups: BTreeMap<String, FlatFittedBinnedForest<'a>>,
}

/// Modeled rows keyed by group; callers must exclude patient identifiers.
pub type FlatStratifiedRows = BTreeMap<String, Vec<Vec<String>>>;

impl<'a> FlatFittedStratifiedForest<'a> {
    /// Fit each nonempty training group with the same ordered variables.
    pub fn fit(
        training: &'a FlatStratifiedRows,
        variables: &'a [FlatVariable],
        regularized: bool,
    ) -> Result<Self, Box<dyn Error>> {
        if training.is_empty() {
            return Err("stratified fitting requires at least one group".into());
        }
        let mut groups = BTreeMap::new();
        for (group, rows) in training {
            let model = FlatFittedBinnedForest::fit(rows, variables, group, regularized)
                .map_err(|error| format!("invalid training group {group}: {error}"))?;
            groups.insert(group.clone(), model);
        }
        Ok(Self { groups })
    }

    /// Fit the greedy two-parent model separately in every group.
    pub fn fit_dag(
        training: &'a FlatStratifiedRows,
        variables: &'a [FlatVariable],
    ) -> Result<Self, Box<dyn Error>> {
        Self::fit_groups(training, variables, FlatFittedBinnedForest::fit_dag)
    }

    /// Fit the interaction-search model separately in every group.
    pub fn fit_interaction_dag(
        training: &'a FlatStratifiedRows,
        variables: &'a [FlatVariable],
    ) -> Result<Self, Box<dyn Error>> {
        Self::fit_groups(
            training,
            variables,
            FlatFittedBinnedForest::fit_interaction_dag,
        )
    }

    fn fit_groups<F>(
        training: &'a FlatStratifiedRows,
        variables: &'a [FlatVariable],
        fit: F,
    ) -> Result<Self, Box<dyn Error>>
    where
        F: Fn(
            &'a [Vec<String>],
            &'a [FlatVariable],
            &str,
        ) -> Result<FlatFittedBinnedForest<'a>, Box<dyn Error>>,
    {
        if training.is_empty() {
            return Err("stratified fitting requires at least one group".into());
        }
        let mut groups = BTreeMap::new();
        for (group, rows) in training {
            let model = fit(rows, variables, group)
                .map_err(|error| format!("invalid training group {group}: {error}"))?;
            groups.insert(group.clone(), model);
        }
        Ok(Self { groups })
    }

    /// Access one fitted group and its learned edges.
    pub fn group(&self, name: &str) -> Option<&FlatFittedBinnedForest<'a>> {
        self.groups.get(name)
    }

    fn validate_allocation(
        &self,
        allocation: &BTreeMap<String, usize>,
    ) -> Result<(), Box<dyn Error>> {
        if allocation.len() != self.groups.len()
            || allocation
                .keys()
                .any(|group| !self.groups.contains_key(group))
        {
            return Err("allocation must specify each fitted group exactly once".into());
        }
        let total = allocation
            .values()
            .try_fold(0_usize, |sum, count| sum.checked_add(*count))
            .ok_or("sample count overflow")?;
        if total > 1_000_000 {
            return Err("total sample count exceeds one million rows".into());
        }
        Ok(())
    }

    /// Draw each group's requested number of rows without writing files.
    pub fn sample(
        &self,
        seed: u64,
        allocation: &BTreeMap<String, usize>,
    ) -> Result<FlatStratifiedRows, Box<dyn Error>> {
        self.validate_allocation(allocation)?;
        self.groups
            .iter()
            .map(|(group, model)| Ok((group.clone(), model.sample(seed, allocation[group])?)))
            .collect()
    }

    /// Draw each group with a sparse-state policy and per-group backoff counts.
    /// No partial sample is returned if any group fails.
    pub fn sample_with_policy(
        &self,
        seed: u64,
        allocation: &BTreeMap<String, usize>,
        policy: FlatSparsePolicy,
    ) -> Result<BTreeMap<String, FlatBinnedSample>, Box<dyn Error>> {
        self.validate_allocation(allocation)?;
        self.groups
            .iter()
            .map(|(group, model)| {
                let sample = model
                    .sample_with_policy(seed, allocation[group], policy)
                    .map_err(|error| format!("sampling group {group} failed: {error}"))?;
                Ok((group.clone(), sample))
            })
            .collect()
    }

    /// Draw and interpolate each group with per-group sparse-backoff counts.
    /// A failure returns no partial cohort.
    pub fn sample_interpolated_with_policy(
        &self,
        seed: u64,
        allocation: &BTreeMap<String, usize>,
        columns: &[String],
        policy: FlatSparsePolicy,
    ) -> Result<BTreeMap<String, FlatPolicyInterpolatedSample>, Box<dyn Error>> {
        self.validate_allocation(allocation)?;
        self.groups
            .iter()
            .map(|(group, model)| {
                let sample = model
                    .sample_interpolated_with_policy(seed, allocation[group], columns, policy)
                    .map_err(|error| format!("sampling group {group} failed: {error}"))?;
                Ok((group.clone(), sample))
            })
            .collect()
    }

    /// Draw each group and interpolate the same declared continuous columns.
    pub fn sample_interpolated(
        &self,
        seed: u64,
        allocation: &BTreeMap<String, usize>,
        columns: &[String],
    ) -> Result<BTreeMap<String, FlatInterpolatedSample>, Box<dyn Error>> {
        self.validate_allocation(allocation)?;
        self.groups
            .iter()
            .map(|(group, model)| {
                Ok((
                    group.clone(),
                    model.sample_interpolated(seed, allocation[group], columns)?,
                ))
            })
            .collect()
    }
}

fn default_flat_missing_policy() -> String {
    "missing".into()
}

fn flat_is_missing(value: &str, variable: &FlatVariable) -> bool {
    (value.is_empty() || value == "NA")
        && !variable
            .structural_absence_values
            .iter()
            .any(|token| token == value)
}

#[derive(Debug, Serialize)]
pub struct FlatGroupReport {
    pub group: String,
    pub patients: usize,
    pub variables: Vec<FlatVariableReport>,
}

#[derive(Debug, Serialize, Clone)]
pub struct FlatVariableReport {
    pub variable: String,
    pub kind: String,
    pub observed: usize,
    pub missing: usize,
    pub structurally_absent: usize,
    pub category_counts: BTreeMap<String, usize>,
    pub quantiles: BTreeMap<String, Option<f64>>,
}

#[derive(Debug, Serialize)]
pub struct FlatCohortReport {
    pub schema_version: String,
    pub allocation_column: String,
    pub total_patients: usize,
    pub split: FlatSplitReport,
    pub groups: Vec<FlatGroupReport>,
    pub partitions: Vec<FlatPartitionReport>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub development_validation: Vec<FlatDevelopmentValidation>,
    pub generated: Vec<FlatGroupReport>,
    pub generated_vs_holdout: Vec<FlatGeneratedMetric>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub joint_generated_vs_holdout: Vec<FlatJointMetric>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub conditional_numeric_generated_vs_holdout: Vec<FlatConditionalNumericMetric>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub family_generated_vs_holdout: Vec<FlatFamilyMetric>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub sparse_backoffs: Vec<FlatBackoffReport>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub learned_dependencies: Vec<FlatLearnedDependency>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub binned_models: Vec<FlatBinnedModel>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub numeric_smoothing: Vec<FlatSmoothingReport>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub numeric_interpolation: Vec<FlatSmoothingReport>,
    pub fidelity: Vec<FlatFidelityGroup>,
    pub benchmark: FlatBenchmarkResult,
    pub limitations: Vec<String>,
}

#[derive(Debug, Serialize)]
pub struct FlatDevelopmentValidation {
    pub group: String,
    pub split_seed: u64,
    pub generator_seed: u64,
    pub training_patients: usize,
    pub validation_patients: usize,
    pub generated: FlatGroupReport,
    pub validation: FlatGroupReport,
    pub categorical_pairs: Vec<FlatJointMetric>,
    pub numeric_pairs: Vec<FlatConditionalNumericMetric>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub numeric_associations: Vec<FlatNumericAssociation>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub joint_families: Vec<FlatJointFamilyMetric>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub utility: Vec<FlatUtilityReport>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub exact_matches: Option<FlatExactMatches>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub numeric_smoothing: Vec<FlatSmoothingReport>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub numeric_interpolation: Vec<FlatSmoothingReport>,
    pub learned_dependencies: Vec<FlatLearnedDependency>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub binned_model: Option<FlatBinnedModel>,
    pub sparse_backoffs: Vec<FlatBackoffReport>,
}

#[derive(Debug, Serialize)]
pub struct FlatSmoothingReport {
    pub allocation: String,
    pub variable: String,
    pub observed_candidates: usize,
    pub changed: usize,
    pub unchanged: usize,
}

#[derive(Debug, Serialize)]
pub struct FlatExactMatches {
    pub training_patients: usize,
    pub validation_patients: usize,
    pub generated_patients: usize,
    pub generated_matching_training: usize,
    pub generated_matching_validation: usize,
    pub validation_matching_training: usize,
    pub duplicate_generated_rows: usize,
}

#[derive(Debug, Serialize)]
pub struct FlatNumericAssociation {
    pub first: String,
    pub second: String,
    pub generated_complete_pairs: usize,
    pub validation_complete_pairs: usize,
    pub generated_spearman: Option<f64>,
    pub validation_spearman: Option<f64>,
    pub absolute_difference: Option<f64>,
}

/// Aggregate Brier scores for the same held-out patients.
#[derive(Debug, Serialize)]
pub struct FlatUtilityReport {
    pub task: String,
    pub real_training: FlatUtilityScore,
    pub synthetic_training: FlatUtilityScore,
    pub validation_unlabeled: usize,
}

#[derive(Debug, Serialize)]
pub struct FlatUtilityScore {
    pub training_labeled: usize,
    pub training_unlabeled: usize,
    pub validation_labeled: usize,
    pub unseen_predictor: usize,
    pub brier_score: Option<f64>,
}

/// Aggregate-only comparison against caller-supplied external patients.
/// The caller must establish provenance, patient independence and permissions.
#[derive(Debug, Serialize)]
pub struct FlatExternalEvaluation {
    pub generated_patients: usize,
    pub reference_patients: usize,
    pub marginal_tvd: Vec<Option<f64>>,
    pub joint_tvd: Option<f64>,
    pub utility: FlatUtilityReport,
}

/// Compare in-memory synthetic and external rows using a declared binary task.
/// Rows must use the supplied categorical schema in exactly the same order.
pub fn evaluate_flat_external(
    training: &[Vec<String>],
    generated: &[Vec<String>],
    reference: &[Vec<String>],
    variables: &[FlatVariable],
    task: &FlatUtilityTask,
) -> Result<FlatExternalEvaluation, Box<dyn Error>> {
    if training.is_empty() || generated.is_empty() || reference.is_empty() || variables.is_empty() {
        return Err("external evaluation requires nonempty cohorts and variables".into());
    }
    let mut names = HashSet::new();
    for v in variables {
        if v.kind != "categorical"
            || v.column.trim().is_empty()
            || v.column.trim() != v.column
            || v.allowed_values.is_empty()
            || !["missing", "categorical_tokens", "structural_tokens"]
                .contains(&v.missing_policy.as_str())
            || (v.missing_policy == "missing" && !v.structural_absence_values.is_empty())
            || (v.missing_policy != "missing" && v.structural_absence_values.is_empty())
            || !names.insert(&v.column)
        {
            return Err("external evaluation requires distinct categorical variables".into());
        }
    }
    for row in training.iter().chain(generated).chain(reference) {
        if row.len() != variables.len() {
            return Err("external row width does not match variables".into());
        }
        for (value, v) in row.iter().zip(variables) {
            if value.trim() != value
                || (!flat_is_missing(value, v)
                    && !v.structural_absence_values.contains(value)
                    && !v.allowed_values.contains(value))
            {
                return Err(format!("invalid external value for {}", v.column).into());
            }
        }
    }
    let target = variables
        .iter()
        .position(|v| v.column == task.target)
        .ok_or("target not found")?;
    let predictors: Vec<usize> = std::iter::once(&task.predictor)
        .chain(task.additional_predictors.iter())
        .map(|name| {
            variables
                .iter()
                .position(|v| &v.column == name)
                .ok_or("predictor not found")
        })
        .collect::<Result<_, _>>()?;
    let mut indices = HashSet::new();
    if !task.predictor_available_before_target
        || !task.alpha.is_finite()
        || task.alpha <= 0.0
        || task.alpha > f64::MAX / 4.0
        || predictors
            .iter()
            .any(|i| *i == target || !indices.insert(*i))
    {
        return Err("invalid external utility task".into());
    }
    let mut labels = HashSet::new();
    if task.positive_values.is_empty()
        || task.negative_values.is_empty()
        || task
            .positive_values
            .iter()
            .chain(&task.negative_values)
            .any(|label| {
                !labels.insert(label)
                    || flat_is_missing(label, &variables[target])
                    || variables[target].structural_absence_values.contains(label)
                    || !variables[target].allowed_values.contains(label)
            })
    {
        return Err("invalid external utility labels".into());
    }
    let training_refs: Vec<_> = training.iter().collect();
    let generated_refs: Vec<_> = generated.iter().collect();
    let real_training = flat_utility_score(&training_refs, reference, task, &predictors, target);
    let synthetic_training =
        flat_utility_score(&generated_refs, reference, task, &predictors, target);
    let marginal_tvd = (0..variables.len())
        .map(|i| flat_family_tvd(generated, reference, &[i]))
        .collect();
    let columns: Vec<_> = (0..variables.len()).collect();
    Ok(FlatExternalEvaluation {
        generated_patients: generated.len(),
        reference_patients: reference.len(),
        marginal_tvd,
        joint_tvd: flat_family_tvd(generated, reference, &columns),
        utility: FlatUtilityReport {
            task: task.name.clone(),
            validation_unlabeled: reference.len() - real_training.validation_labeled,
            real_training,
            synthetic_training,
        },
    })
}

#[cfg(test)]
mod external_evaluation_tests {
    use super::*;

    #[test]
    fn validates_external_rows_and_missing_labels_and_scores_reference() {
        let variables: Vec<FlatVariable> = serde_yaml::from_str("- {column: x, kind: categorical, allowed_values: [a, b]}\n- {column: y, kind: categorical, allowed_values: ['yes', 'no', 'NA']}\n").unwrap();
        let mut task: FlatUtilityTask = serde_yaml::from_str("name: task\npredictor: x\ntarget: y\npositive_values: ['yes']\nnegative_values: ['no']\nalpha: 1.0\npredictor_available_before_target: true\n").unwrap();
        let training = vec![
            vec!["a".into(), "yes".into()],
            vec!["a".into(), "no".into()],
        ];
        let generated = vec![vec!["a".into(), "yes".into()]];
        let reference = vec![
            vec!["b".into(), "yes".into()],
            vec!["a".into(), "NA".into()],
        ];
        let report =
            evaluate_flat_external(&training, &generated, &reference, &variables, &task).unwrap();
        assert_eq!(report.utility.real_training.validation_labeled, 1);
        assert_eq!(report.utility.validation_unlabeled, 1);
        assert_eq!(report.utility.real_training.unseen_predictor, 1);
        assert_eq!(report.utility.real_training.brier_score, Some(0.25));
        assert_eq!(report.joint_tvd, Some(1.0));
        task.positive_values.push("NA".into());
        assert!(
            evaluate_flat_external(&training, &generated, &reference, &variables, &task).is_err()
        );
        task.positive_values.pop();
        assert!(evaluate_flat_external(
            &training,
            &generated,
            &[vec!["a".into()]],
            &variables,
            &task
        )
        .is_err());
        let mut structural = variables.clone();
        structural[1].missing_policy = "structural_tokens".into();
        structural[1].structural_absence_values = vec!["absent".into()];
        structural[1].allowed_values.push("absent".into());
        task.negative_values.push("absent".into());
        assert!(
            evaluate_flat_external(&training, &generated, &reference, &structural, &task).is_err()
        );
        task.negative_values.pop();
        let unlabeled = vec![vec!["a".into(), "absent".into()]];
        let report =
            evaluate_flat_external(&unlabeled, &generated, &reference, &structural, &task).unwrap();
        assert_eq!(report.utility.real_training.training_unlabeled, 1);
        assert_eq!(report.utility.real_training.brier_score, None);
    }
}

#[derive(Debug, Serialize)]
pub struct FlatBinnedModel {
    pub allocation: String,
    pub numeric_bins: Vec<FlatNumericBins>,
    pub selected_pairs: Vec<FlatJointMetric>,
}

#[derive(Debug, Serialize)]
pub struct FlatNumericBins {
    pub variable: String,
    pub upper_bounds: Vec<f64>,
}

#[derive(Debug, Clone, Serialize)]
pub struct FlatLearnedDependency {
    pub allocation: String,
    pub parent: String,
    pub child: String,
    pub mutual_information_nats: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub conditioned_on: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub joint_selection_gain_nats: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub penalized_gain_nats: Option<f64>,
    pub development_patients: usize,
}

#[derive(Debug, Serialize)]
pub struct FlatConditionalNumericMetric {
    pub allocation: String,
    pub parent: String,
    pub parent_value: String,
    pub child: String,
    pub generated_patients: usize,
    pub holdout_patients: usize,
    pub generated_observed: usize,
    pub holdout_observed: usize,
    pub generated_missing: usize,
    pub holdout_missing: usize,
    pub generated_structurally_absent: usize,
    pub holdout_structurally_absent: usize,
    pub absolute_quantile_differences: BTreeMap<String, Option<f64>>,
}

/// Aggregate joint distance over a predeclared set of categorical variables.
#[derive(Debug, Serialize)]
pub struct FlatJointFamilyMetric {
    pub variables: Vec<String>,
    pub generated_patients: usize,
    pub validation_patients: usize,
    pub total_variation_distance: Option<f64>,
}

#[derive(Debug, Serialize)]
pub struct FlatFamilyMetric {
    pub allocation: String,
    pub parents: Vec<String>,
    pub child: String,
    pub generated_patients: usize,
    pub holdout_patients: usize,
    pub total_variation_distance: Option<f64>,
}

#[derive(Debug, Serialize)]
pub struct FlatBackoffReport {
    pub allocation: String,
    pub child: String,
    pub dropped_parent_counts: BTreeMap<usize, usize>,
}

#[derive(Debug, Serialize)]
pub struct FlatJointMetric {
    pub allocation: String,
    pub parent: String,
    pub child: String,
    pub generated_patients: usize,
    pub holdout_patients: usize,
    pub total_variation_distance: Option<f64>,
}

#[derive(Debug, Serialize)]
pub struct FlatGeneratedMetric {
    pub allocation: String,
    pub variable: String,
    pub kind: String,
    pub generated_observed: usize,
    pub holdout_observed: usize,
    pub generated_missing: usize,
    pub holdout_missing: usize,
    pub missingness_rate_difference: Option<f64>,
    pub categorical_tvd: Option<f64>,
    pub absolute_quantile_differences: BTreeMap<String, Option<f64>>,
}

#[derive(Debug, Serialize)]
pub struct FlatFidelityGroup {
    pub group: String,
    pub variables: Vec<FlatFidelityMetric>,
}

#[derive(Debug, Serialize)]
pub struct FlatFidelityMetric {
    pub variable: String,
    pub kind: String,
    pub development_observed: usize,
    pub holdout_observed: usize,
    pub development_missing: usize,
    pub holdout_missing: usize,
    pub missingness_rate_absolute_difference: Option<f64>,
    pub categorical_total_variation_distance: Option<f64>,
    pub absolute_quantile_differences: BTreeMap<String, Option<f64>>,
    pub status: String,
    pub bootstrap_intervals: BTreeMap<String, Option<(f64, f64)>>,
    pub valid_bootstrap_replicates: BTreeMap<String, usize>,
}

#[derive(Debug, Serialize)]
pub struct FlatBenchmarkResult {
    pub method: String,
    pub generator_seed: u64,
    pub sparse_policy: FlatSparsePolicy,
    pub uncertainty: String,
    pub passed: usize,
    pub failed: usize,
    pub unavailable: usize,
    pub thresholds: FlatThresholds,
}

#[derive(Debug, Serialize)]
pub struct FlatPartitionReport {
    pub partition: String,
    pub patients: usize,
    pub groups: Vec<FlatGroupReport>,
}

#[derive(Debug, Serialize)]
pub struct FlatCohortManifest {
    pub schema_version: String,
    pub source_fingerprint: String,
    pub config_fingerprint: String,
    pub report_fingerprint: String,
    pub artifacts: Vec<String>,
    pub limitations: Vec<String>,
}

pub fn flat_fingerprint(path: &Path) -> Result<String, Box<dyn Error>> {
    let bytes = std::fs::read(path)?;
    let hash = bytes.iter().fold(0xcbf29ce484222325u64, |h, b| {
        (h ^ u64::from(*b)).wrapping_mul(0x100000001b3)
    });
    Ok(format!("fnv1a64:{hash:016x}"))
}

#[derive(Debug, Serialize)]
pub struct FlatSplitReport {
    pub seed: u64,
    pub development_proportion: f64,
    pub development_patients: usize,
    pub holdout_patients: usize,
    pub groups: BTreeMap<String, (usize, usize)>,
    pub algorithm: String,
}

/// Summarize a flat CSV without retaining or emitting patient identifiers/rows.
pub fn describe_flat_csv(
    path: &Path,
    config: &FlatCohortConfig,
) -> Result<FlatCohortReport, Box<dyn Error>> {
    use std::collections::BTreeSet;
    if config.schema_version != "1"
        || config.allocation_values.is_empty()
        || config.allocation_column.trim().is_empty()
        || config.row_key_column.trim().is_empty()
        || config.row_key_column == config.allocation_column
        || !config.development_proportion.is_finite()
        || !(0.0..1.0).contains(&config.development_proportion)
        || config
            .development_validation_proportion
            .is_some_and(|p| !p.is_finite() || p <= 0.0 || p >= 1.0)
        || (config.development_validation_seed.is_some()
            && config.development_validation_proportion.is_none())
        || (config.exact_match_diagnostics && config.development_validation_proportion.is_none())
        || !config
            .thresholds
            .max_missingness_rate_difference
            .is_finite()
        || !(0.0..=1.0).contains(&config.thresholds.max_missingness_rate_difference)
        || !config.thresholds.max_categorical_tvd.is_finite()
        || !(0.0..=1.0).contains(&config.thresholds.max_categorical_tvd)
        || !config
            .thresholds
            .max_numeric_quantile_difference
            .is_finite()
        || config.thresholds.max_numeric_quantile_difference < 0.0
        || !(100..=10000).contains(&config.bootstrap_replicates)
        || !config.bootstrap_confidence_level.is_finite()
        || !(0.5..1.0).contains(&config.bootstrap_confidence_level)
    {
        return Err("invalid flat cohort configuration".into());
    }
    let values: BTreeSet<_> = config.allocation_values.iter().collect();
    if values.len() != config.allocation_values.len()
        || config
            .allocation_values
            .iter()
            .any(|v| v.is_empty() || v.trim() != v)
        || config.variables.is_empty()
    {
        return Err(
            "allocation groups must be nonempty, distinct, and trimmed; variables must be nonempty"
                .into(),
        );
    }
    let mut reader = csv::Reader::from_path(path)?;
    let headers = reader.headers()?.clone();
    if headers.iter().collect::<HashSet<_>>().len() != headers.len() {
        return Err("duplicate flat CSV headers".into());
    }
    if !config.expected_source_columns.is_empty()
        && (config
            .expected_source_columns
            .iter()
            .collect::<HashSet<_>>()
            .len()
            != config.expected_source_columns.len()
            || config.expected_source_columns.len() != headers.len()
            || config
                .expected_source_columns
                .iter()
                .any(|column| !headers.iter().any(|header| header == column)))
    {
        return Err("source columns do not match the declared flat cohort schema".into());
    }
    let group_index = headers
        .iter()
        .position(|h| h == config.allocation_column)
        .ok_or("allocation column missing")?;
    let indexes: Vec<_> = config
        .variables
        .iter()
        .map(|v| {
            if !["categorical", "numeric"].contains(&v.kind.as_str())
                || v.column == config.allocation_column
                || v.column == config.row_key_column
                || !["missing", "categorical_tokens", "structural_tokens"]
                    .contains(&v.missing_policy.as_str())
                || (v.missing_policy == "missing" && !v.structural_absence_values.is_empty())
                || (v.missing_policy != "missing" && v.structural_absence_values.is_empty())
                || (v.kind == "categorical" && v.allowed_values.is_empty())
                || (v.kind == "numeric" && !v.allowed_values.is_empty())
            {
                return Err(format!("invalid flat variable declaration: {}", v.column).into());
            }
            headers
                .iter()
                .position(|h| h == v.column)
                .ok_or_else(|| format!("missing column: {}", v.column).into())
        })
        .collect::<Result<_, Box<dyn Error>>>()?;
    if config
        .variables
        .iter()
        .map(|v| &v.column)
        .collect::<HashSet<_>>()
        .len()
        != config.variables.len()
    {
        return Err("flat variables must be unique".into());
    }
    let mut smoothing_variables = HashSet::new();
    for smoothing in &config.numeric_smoothing {
        let variable = config
            .variables
            .iter()
            .find(|v| v.column == smoothing.variable)
            .ok_or("unknown numeric smoothing variable")?;
        if variable.kind != "numeric"
            || !smoothing_variables.insert(&smoothing.variable)
            || !smoothing.bandwidth.is_finite()
            || smoothing.bandwidth <= 0.0
            || !smoothing.minimum.is_finite()
            || !smoothing.maximum.is_finite()
            || smoothing.minimum >= smoothing.maximum
            || !(smoothing.maximum - smoothing.minimum).is_finite()
        {
            return Err("numeric smoothing requires unique numeric variables, positive finite bandwidth and finite ordered bounds".into());
        }
    }
    if !config.numeric_interpolation.is_empty()
        && !matches!(
            config.generator,
            FlatGenerator::LearnedBinnedForest
                | FlatGenerator::RegularizedBinnedForest
                | FlatGenerator::LearnedBinnedDag
                | FlatGenerator::InteractionBinnedDag
        )
    {
        return Err("numeric interpolation requires a binned forest".into());
    }
    let mut interpolation_variables = HashSet::new();
    for name in &config.numeric_interpolation {
        let variable = config
            .variables
            .iter()
            .find(|v| v.column == *name)
            .ok_or("unknown numeric interpolation variable")?;
        if variable.kind != "numeric"
            || !interpolation_variables.insert(name)
            || smoothing_variables.contains(name)
        {
            return Err(
                "numeric interpolation requires unique numeric variables without smoothing overlap"
                    .into(),
            );
        }
    }
    if config.generator == FlatGenerator::ConditionalCategorical && config.dependencies.is_empty() {
        return Err("conditional categorical generation requires dependencies".into());
    }
    if !matches!(
        config.generator,
        FlatGenerator::ConditionalCategorical
            | FlatGenerator::LearnedBinnedDag
            | FlatGenerator::InteractionBinnedDag
    ) && config.sparse_policy != FlatSparsePolicy::Fail
    {
        return Err(
            "a sparse backoff policy requires a conditional or learned binned DAG generator".into(),
        );
    }
    let mut parent_indexes: Vec<Vec<usize>> = vec![Vec::new(); config.variables.len()];
    for dependency in &config.dependencies {
        let parent = config
            .variables
            .iter()
            .position(|v| v.column == dependency.parent)
            .ok_or("unknown dependency parent")?;
        let child = config
            .variables
            .iter()
            .position(|v| v.column == dependency.child)
            .ok_or("unknown dependency child")?;
        if parent >= child
            || config.variables[parent].kind != "categorical"
            || parent_indexes[child].contains(&parent)
        {
            return Err(
                "conditional dependencies must be unique and use earlier categorical parents"
                    .into(),
            );
        }
        parent_indexes[child].push(parent);
    }
    if !config.numeric_associations.is_empty() && config.development_validation_proportion.is_none()
    {
        return Err("numeric associations require development validation".into());
    }
    let mut seen_numeric_pairs = HashSet::new();
    for pair in &config.numeric_associations {
        let first = config
            .variables
            .iter()
            .position(|v| v.column == pair.parent)
            .ok_or("unknown numeric association variable")?;
        let second = config
            .variables
            .iter()
            .position(|v| v.column == pair.child)
            .ok_or("unknown numeric association variable")?;
        let key = (first.min(second), first.max(second));
        if first == second
            || config.variables[first].kind != "numeric"
            || config.variables[second].kind != "numeric"
            || !seen_numeric_pairs.insert(key)
        {
            return Err(
                "numeric associations must be distinct numeric variable pairs without duplicates"
                    .into(),
            );
        }
    }
    if !config.joint_families.is_empty() && config.development_validation_proportion.is_none() {
        return Err("joint families require development validation".into());
    }
    let mut families = HashSet::new();
    for family in &config.joint_families {
        let mut members = family.clone();
        members.sort();
        if members.len() < 2
            || members.windows(2).any(|pair| pair[0] == pair[1])
            || !families.insert(members.clone())
            || members.iter().any(|name| {
                config
                    .variables
                    .iter()
                    .find(|v| &v.column == name)
                    .is_none_or(|v| v.kind != "categorical")
            })
        {
            return Err(
                "joint families require unique categorical variables and no duplicate families"
                    .into(),
            );
        }
    }
    if !config.utility_tasks.is_empty() && config.development_validation_proportion.is_none() {
        return Err("utility tasks require development validation".into());
    }
    let mut task_names = HashSet::new();
    for task in &config.utility_tasks {
        let predictors: Vec<_> = std::iter::once(&task.predictor)
            .chain(task.additional_predictors.iter())
            .collect();
        let unique_predictors: HashSet<_> = predictors.iter().collect();
        let target = config.variables.iter().find(|v| v.column == task.target);
        let labels: HashSet<_> = task
            .positive_values
            .iter()
            .chain(&task.negative_values)
            .collect();
        if task.name.trim().is_empty()
            || !task_names.insert(&task.name)
            || unique_predictors.len() != predictors.len()
            || predictors.iter().any(|name| {
                **name == task.target
                    || config
                        .variables
                        .iter()
                        .find(|v| &v.column == *name)
                        .is_none_or(|v| v.kind != "categorical")
            })
            || !task.predictor_available_before_target
            || target.is_none_or(|v| v.kind != "categorical")
            || task.positive_values.is_empty()
            || task.negative_values.is_empty()
            || labels.len() != task.positive_values.len() + task.negative_values.len()
            || !task.alpha.is_finite()
            || task.alpha <= 0.0
            || task.alpha > f64::MAX / 4.0
            || labels.iter().any(|label| {
                label.trim().is_empty()
                    || flat_is_missing(label, target.unwrap())
                    || !target.unwrap().allowed_values.contains(label)
                    || target.unwrap().structural_absence_values.contains(label)
            })
        {
            return Err("invalid utility task declaration".into());
        }
    }
    let id_index = headers
        .iter()
        .position(|h| h == config.row_key_column)
        .ok_or("flat cohort row key column missing")?;
    let mut rows: BTreeMap<String, Vec<(String, Vec<String>)>> = config
        .allocation_values
        .iter()
        .map(|g| (g.clone(), Vec::new()))
        .collect();
    let mut seen_ids = HashSet::new();
    let mut seen_rows = 0usize;
    for record in reader.records() {
        let record = record?;
        seen_rows += 1;
        let id = record.get(id_index).unwrap_or("").trim();
        if id.is_empty() || !seen_ids.insert(id.to_owned()) {
            return Err("flat cohort row identifier is empty or duplicated".into());
        }
        let group = record.get(group_index).unwrap_or("").trim();
        let row = rows
            .get_mut(group)
            .ok_or_else(|| format!("unknown or empty allocation group: {group}"))?;
        let values: Vec<String> = indexes
            .iter()
            .map(|i| record.get(*i).unwrap_or("").trim().to_owned())
            .collect();
        for (index, variable) in config.variables.iter().enumerate() {
            let value = &values[index];
            if variable.kind == "categorical"
                && !value.is_empty()
                && value != "NA"
                && !variable.structural_absence_values.contains(value)
                && !variable.allowed_values.contains(value)
            {
                return Err(format!("undeclared category for {}", variable.column).into());
            }
        }
        row.push((id.to_owned(), values));
    }
    if seen_rows == 0 {
        return Err("flat cohort contains no rows".into());
    }
    let total_patients = rows.values().map(Vec::len).sum();
    let mut groups = Vec::new();
    let mut development_groups = Vec::new();
    let mut holdout_groups = Vec::new();
    let mut split_groups = BTreeMap::new();
    let mut partition_values: BTreeMap<String, [Vec<Vec<String>>; 2]> = BTreeMap::new();
    let mut synthetic_values: BTreeMap<String, Vec<Vec<String>>> = BTreeMap::new();
    let mut sparse_backoffs = Vec::new();
    let mut learned_dependencies = Vec::new();
    let mut binned_models = Vec::new();
    let mut numeric_smoothing = Vec::new();
    let mut numeric_interpolation = Vec::new();
    let mut development_validation = Vec::new();
    let mut development_patients = 0;
    let mut holdout_patients = 0;
    for (group, data) in rows {
        let mut ranked: Vec<_> = data
            .iter()
            .map(|(id, _)| (flat_hash(config.split_seed, id), id.clone()))
            .collect();
        ranked.sort();
        let dev_count = ((ranked.len() as f64) * config.development_proportion).round() as usize;
        let development_ids: HashSet<_> = ranked
            .iter()
            .take(dev_count)
            .map(|(_, id)| id.as_str())
            .collect();
        let development_rows: Vec<&Vec<String>> = data
            .iter()
            .filter(|(id, _)| development_ids.contains(id.as_str()))
            .map(|(_, row)| row)
            .collect();
        if development_rows.is_empty() && dev_count < data.len() {
            return Err("empty development group cannot support empirical generation".into());
        }
        development_patients += dev_count;
        holdout_patients += ranked.len() - dev_count;
        split_groups.insert(group.clone(), (dev_count, ranked.len() - dev_count));
        let mut variables = Vec::new();
        let mut partition_variables = [Vec::new(), Vec::new()];
        let mut partition_rows = [Vec::new(), Vec::new()];
        for (j, variable) in config.variables.iter().enumerate() {
            let absent_value = |v: &str| variable.structural_absence_values.iter().any(|x| x == v);
            let absent_count = data.iter().filter(|(_, r)| absent_value(&r[j])).count();
            let observed: Vec<_> = data
                .iter()
                .map(|(_, r)| r[j].as_str())
                .filter(|v| !v.is_empty() && *v != "NA" && !absent_value(v))
                .collect();
            let missing = data
                .iter()
                .filter(|(_, r)| flat_is_missing(&r[j], variable))
                .count();
            let mut category_counts = BTreeMap::new();
            let mut numbers = Vec::new();
            for value in &observed {
                if variable.kind == "numeric" {
                    let number: f64 = value
                        .parse()
                        .map_err(|_| format!("invalid numeric value for {}", variable.column))?;
                    if !number.is_finite() {
                        return Err(
                            format!("non-finite numeric value for {}", variable.column).into()
                        );
                    }
                    numbers.push(number);
                } else {
                    *category_counts.entry((*value).to_owned()).or_insert(0) += 1;
                }
            }
            numbers.sort_by(f64::total_cmp);
            let quantile = |p: f64| -> Option<f64> {
                if numbers.is_empty() {
                    return None;
                }
                let pos = p * (numbers.len() - 1) as f64;
                let lo = pos.floor() as usize;
                let hi = pos.ceil() as usize;
                Some(numbers[lo] + (numbers[hi] - numbers[lo]) * (pos - lo as f64))
            };
            let quantiles = if variable.kind == "numeric" {
                [
                    ("q25", quantile(0.25)),
                    ("median", quantile(0.5)),
                    ("q75", quantile(0.75)),
                ]
                .into_iter()
                .map(|(k, v)| (k.to_owned(), v))
                .collect()
            } else {
                BTreeMap::new()
            };
            variables.push(FlatVariableReport {
                variable: variable.column.clone(),
                kind: variable.kind.clone(),
                observed: observed.len(),
                missing,
                structurally_absent: absent_count,
                category_counts,
                quantiles,
            });
            for (part, selected) in [true, false].into_iter().enumerate() {
                let subset: Vec<_> = data
                    .iter()
                    .filter(|(id, _)| development_ids.contains(id.as_str()) == selected)
                    .collect();
                if j == 0 {
                    partition_rows[part].extend(subset.iter().map(|(_, row)| row.clone()));
                }
                let vals: Vec<_> = subset
                    .iter()
                    .map(|(_, row)| row[j].as_str())
                    .filter(|v| {
                        !v.is_empty()
                            && *v != "NA"
                            && !variable.structural_absence_values.iter().any(|x| x == *v)
                    })
                    .collect();
                let mut cats = BTreeMap::new();
                let mut nums = Vec::new();
                for value in &vals {
                    if variable.kind == "numeric" {
                        nums.push(value.parse::<f64>().map_err(|_| {
                            format!("invalid numeric value for {}", variable.column)
                        })?);
                    } else {
                        *cats.entry((*value).to_owned()).or_insert(0) += 1;
                    }
                }
                nums.sort_by(f64::total_cmp);
                let q = |p: f64| {
                    if nums.is_empty() {
                        None
                    } else {
                        let x = p * (nums.len() - 1) as f64;
                        let lo = x.floor() as usize;
                        let hi = x.ceil() as usize;
                        Some(nums[lo] + (nums[hi] - nums[lo]) * (x - lo as f64))
                    }
                };
                let quantiles = if variable.kind == "numeric" {
                    [("q25", q(0.25)), ("median", q(0.5)), ("q75", q(0.75))]
                        .into_iter()
                        .map(|(k, v)| (k.to_owned(), v))
                        .collect()
                } else {
                    BTreeMap::new()
                };
                let absent = subset
                    .iter()
                    .filter(|(_, row)| {
                        variable
                            .structural_absence_values
                            .iter()
                            .any(|v| v == &row[j])
                    })
                    .count();
                partition_variables[part].push(FlatVariableReport {
                    variable: variable.column.clone(),
                    kind: variable.kind.clone(),
                    observed: vals.len(),
                    missing: subset
                        .iter()
                        .filter(|(_, row)| flat_is_missing(&row[j], variable))
                        .count(),
                    structurally_absent: absent,
                    category_counts: cats,
                    quantiles,
                });
            }
        }
        if let Some(proportion) = config.development_validation_proportion {
            development_validation.push(flat_development_validation(
                &group,
                &data,
                &development_ids,
                proportion,
                config,
                &parent_indexes,
            )?);
        }
        let group_size = data.len();
        groups.push(FlatGroupReport {
            group,
            patients: group_size,
            variables,
        });
        let group_name = groups.last().unwrap().group.clone();
        development_groups.push(FlatGroupReport {
            group: group_name.clone(),
            patients: dev_count,
            variables: partition_variables[0].clone(),
        });
        holdout_groups.push(FlatGroupReport {
            group: group_name.clone(),
            patients: group_size - dev_count,
            variables: partition_variables[1].clone(),
        });
        partition_values.insert(group_name.clone(), partition_rows);
        let generator_seed = config.generator_seed.unwrap_or(config.split_seed);
        let mut synth_state = generator_seed ^ flat_hash(generator_seed, &group_name);
        let mut backoff_counts = vec![BTreeMap::new(); config.variables.len()];
        let binned_fit = if matches!(
            config.generator,
            FlatGenerator::LearnedBinnedForest
                | FlatGenerator::RegularizedBinnedForest
                | FlatGenerator::LearnedBinnedDag
                | FlatGenerator::InteractionBinnedDag
        ) {
            Some(fit_flat_binned_selected(
                &development_rows,
                &config.variables,
                &group_name,
                config.generator,
            ))
        } else {
            None
        };
        let learned_parents = if matches!(
            config.generator,
            FlatGenerator::LearnedCategoricalForest | FlatGenerator::LearnedMixedForest
        ) {
            let (parents, learned) = fit_flat_ordered_forest(
                &development_rows,
                &config.variables,
                &group_name,
                config.generator == FlatGenerator::LearnedMixedForest,
            );
            learned_dependencies.extend(learned);
            parents
        } else {
            vec![Vec::new(); config.variables.len()]
        };
        if let Some(fit) = &binned_fit {
            learned_dependencies.extend(fit.edges.iter().cloned());
        }
        let mut synthetic = (0..holdout_groups.last().unwrap().patients)
            .map(|attempt| {
                match config.generator {
                    FlatGenerator::RowBootstrap => Ok(development_rows
                        [flat_sample_index(&mut synth_state, development_rows.len())]
                    .clone()),
                    FlatGenerator::IndependentMarginals => Ok(sample_flat_marginals(
                        &development_rows,
                        config.variables.len(),
                        &mut synth_state,
                    )),
                    FlatGenerator::ConditionalCategorical => sample_flat_conditionals(
                        &development_rows,
                        &parent_indexes,
                        &mut synth_state,
                        config.sparse_policy,
                        &mut backoff_counts,
                    ),
                    FlatGenerator::LearnedCategoricalForest | FlatGenerator::LearnedMixedForest => {
                        sample_flat_conditionals(
                            &development_rows,
                            &learned_parents,
                            &mut synth_state,
                            FlatSparsePolicy::Fail,
                            &mut backoff_counts,
                        )
                    }
                    FlatGenerator::LearnedBinnedForest
                    | FlatGenerator::RegularizedBinnedForest
                    | FlatGenerator::LearnedBinnedDag
                    | FlatGenerator::InteractionBinnedDag => sample_flat_binned_checked(
                        &development_rows,
                        binned_fit.as_ref().expect("binned fit"),
                        &mut synth_state,
                        config.sparse_policy,
                        &mut backoff_counts,
                    ),
                }
                .map_err(|error: Box<dyn Error>| -> Box<dyn Error> {
                    format!(
                        "generation stopped after {} accepted rows in the current group: {error}",
                        attempt
                    )
                    .into()
                })
            })
            .collect::<Result<Vec<_>, Box<dyn Error>>>()?;
        numeric_smoothing.extend(apply_flat_numeric_smoothing(
            &mut synthetic,
            &development_rows,
            config,
            binned_fit.as_ref(),
            &group_name,
            generator_seed,
        )?);
        if !config.numeric_interpolation.is_empty() {
            numeric_interpolation.extend(apply_flat_numeric_interpolation(
                &mut synthetic,
                &development_rows,
                &config.variables,
                &config.numeric_interpolation,
                binned_fit.as_ref().expect("validated binned model"),
                &group_name,
                generator_seed,
            ));
        }
        if let Some(fit) = &binned_fit {
            binned_models.push(flat_binned_report(
                &group_name,
                fit,
                &config.variables,
                &synthetic,
                &partition_values[&group_name][1],
            ));
        }
        for (index, counts) in backoff_counts.into_iter().enumerate() {
            if !counts.is_empty() {
                sparse_backoffs.push(FlatBackoffReport {
                    allocation: group_name.clone(),
                    child: config.variables[index].column.clone(),
                    dropped_parent_counts: counts,
                });
            }
        }
        synthetic_values.insert(group_name, synthetic);
    }
    let mut fidelity = Vec::new();
    for (dev_group, test_group) in development_groups.iter().zip(&holdout_groups) {
        let mut metrics = Vec::new();
        for (dev, test) in dev_group.variables.iter().zip(&test_group.variables) {
            let dev_n = dev.observed + dev.missing;
            let test_n = test.observed + test.missing;
            let missingness = if dev_n > 0 && test_n > 0 {
                Some(
                    (dev.missing as f64 / dev_n as f64 - test.missing as f64 / test_n as f64).abs(),
                )
            } else {
                None
            };
            let tvd = if dev.kind == "categorical" && dev.observed > 0 && test.observed > 0 {
                let domain: BTreeSet<_> = dev
                    .category_counts
                    .keys()
                    .chain(test.category_counts.keys())
                    .collect();
                Some(
                    0.5 * domain
                        .iter()
                        .map(|value| {
                            let a = *dev.category_counts.get(*value).unwrap_or(&0) as f64
                                / dev.observed as f64;
                            let b = *test.category_counts.get(*value).unwrap_or(&0) as f64
                                / test.observed as f64;
                            (a - b).abs()
                        })
                        .sum::<f64>(),
                )
            } else {
                None
            };
            let mut quantile_differences = BTreeMap::new();
            for key in ["q25", "median", "q75"] {
                let difference = match (
                    dev.quantiles.get(key).copied().flatten(),
                    test.quantiles.get(key).copied().flatten(),
                ) {
                    (Some(a), Some(b)) => Some((a - b).abs()),
                    _ => None,
                };
                quantile_differences.insert(key.to_owned(), difference);
            }
            let mut applicable = vec![missingness];
            if dev.kind == "categorical" {
                applicable.push(tvd);
            } else {
                applicable.extend(quantile_differences.values().copied());
            }
            let status = if applicable.iter().any(Option::is_none) {
                "unavailable"
            } else if missingness
                .is_some_and(|v| v > config.thresholds.max_missingness_rate_difference)
                || tvd.is_some_and(|v| v > config.thresholds.max_categorical_tvd)
                || quantile_differences
                    .values()
                    .flatten()
                    .any(|v| *v > config.thresholds.max_numeric_quantile_difference)
            {
                "fail"
            } else {
                "pass"
            };
            let rows = partition_values
                .get(&dev_group.group)
                .expect("partition rows retained");
            let variable_index = config
                .variables
                .iter()
                .position(|v| v.column == dev.variable)
                .expect("configured variable");
            let (bootstrap_intervals, valid_bootstrap_replicates) = flat_bootstrap_intervals(
                &rows[0],
                &rows[1],
                variable_index,
                &config.variables[variable_index],
                config.bootstrap_replicates,
                config.bootstrap_confidence_level,
                config.split_seed ^ flat_hash(config.split_seed, &dev_group.group),
            );
            metrics.push(FlatFidelityMetric {
                variable: dev.variable.clone(),
                kind: dev.kind.clone(),
                development_observed: dev.observed,
                holdout_observed: test.observed,
                development_missing: dev.missing,
                holdout_missing: test.missing,
                missingness_rate_absolute_difference: missingness,
                categorical_total_variation_distance: tvd,
                absolute_quantile_differences: quantile_differences,
                status: status.into(),
                bootstrap_intervals,
                valid_bootstrap_replicates,
            });
        }
        fidelity.push(FlatFidelityGroup {
            group: dev_group.group.clone(),
            variables: metrics,
        });
    }
    let statuses = fidelity
        .iter()
        .flat_map(|group| group.variables.iter().map(|metric| metric.status.as_str()));
    let mut passed = 0;
    let mut failed = 0;
    let mut unavailable = 0;
    for status in statuses {
        match status {
            "pass" => passed += 1,
            "fail" => failed += 1,
            _ => unavailable += 1,
        }
    }
    let generated: Vec<_> = synthetic_values
        .iter()
        .map(|(group, rows)| summarize_flat_rows(group, rows, &config.variables))
        .collect();
    let mut generated_vs_holdout = Vec::new();
    for generated_group in &generated {
        let holdout_group = holdout_groups
            .iter()
            .find(|g| g.group == generated_group.group)
            .expect("matching holdout group");
        for (a, b) in generated_group
            .variables
            .iter()
            .zip(&holdout_group.variables)
        {
            let an = a.observed + a.missing;
            let bn = b.observed + b.missing;
            let missing = if an > 0 && bn > 0 {
                Some(a.missing as f64 / an as f64 - b.missing as f64 / bn as f64)
            } else {
                None
            };
            let tvd = if a.kind == "categorical" && a.observed > 0 && b.observed > 0 {
                let domain: BTreeSet<_> = a
                    .category_counts
                    .keys()
                    .chain(b.category_counts.keys())
                    .collect();
                Some(
                    0.5 * domain
                        .iter()
                        .map(|v| {
                            (a.category_counts.get(*v).copied().unwrap_or(0) as f64
                                / a.observed as f64
                                - b.category_counts.get(*v).copied().unwrap_or(0) as f64
                                    / b.observed as f64)
                                .abs()
                        })
                        .sum::<f64>(),
                )
            } else {
                None
            };
            let quantiles = ["q25", "median", "q75"]
                .into_iter()
                .map(|k| {
                    (
                        k.to_owned(),
                        match (
                            a.quantiles.get(k).copied().flatten(),
                            b.quantiles.get(k).copied().flatten(),
                        ) {
                            (Some(x), Some(y)) => Some((x - y).abs()),
                            _ => None,
                        },
                    )
                })
                .collect();
            generated_vs_holdout.push(FlatGeneratedMetric {
                allocation: generated_group.group.clone(),
                variable: a.variable.clone(),
                kind: a.kind.clone(),
                generated_observed: a.observed,
                holdout_observed: b.observed,
                generated_missing: a.missing,
                holdout_missing: b.missing,
                missingness_rate_difference: missing,
                categorical_tvd: tvd,
                absolute_quantile_differences: quantiles,
            });
        }
    }
    let mut joint_generated_vs_holdout = Vec::new();
    let mut conditional_numeric_generated_vs_holdout = Vec::new();
    let mut family_generated_vs_holdout = Vec::new();
    for (group, generated_rows) in &synthetic_values {
        let holdout_rows = &partition_values[group][1];
        for dependency in &config.dependencies {
            let parent = config
                .variables
                .iter()
                .position(|v| v.column == dependency.parent)
                .expect("validated dependency");
            let child = config
                .variables
                .iter()
                .position(|v| v.column == dependency.child)
                .expect("validated dependency");
            if config.variables[child].kind == "numeric" {
                conditional_numeric_generated_vs_holdout.extend(flat_conditional_numeric_metrics(
                    group,
                    &dependency.parent,
                    generated_rows,
                    holdout_rows,
                    parent,
                    child,
                    &config.variables[child],
                ));
                continue;
            }
            joint_generated_vs_holdout.push(FlatJointMetric {
                allocation: group.clone(),
                parent: dependency.parent.clone(),
                child: dependency.child.clone(),
                generated_patients: generated_rows.len(),
                holdout_patients: holdout_rows.len(),
                total_variation_distance: flat_joint_tvd(
                    generated_rows,
                    holdout_rows,
                    parent,
                    child,
                ),
            });
        }
        for edge in learned_dependencies
            .iter()
            .filter(|edge: &&FlatLearnedDependency| {
                edge.allocation == *group
                    && !matches!(
                        config.generator,
                        FlatGenerator::LearnedBinnedForest
                            | FlatGenerator::RegularizedBinnedForest
                            | FlatGenerator::LearnedBinnedDag
                            | FlatGenerator::InteractionBinnedDag
                    )
            })
        {
            if config
                .dependencies
                .iter()
                .any(|d| d.parent == edge.parent && d.child == edge.child)
            {
                continue;
            }
            let parent = config
                .variables
                .iter()
                .position(|v| v.column == edge.parent)
                .expect("learned parent");
            let child = config
                .variables
                .iter()
                .position(|v| v.column == edge.child)
                .expect("learned child");
            if config.variables[child].kind == "numeric" {
                conditional_numeric_generated_vs_holdout.extend(flat_conditional_numeric_metrics(
                    group,
                    &edge.parent,
                    generated_rows,
                    holdout_rows,
                    parent,
                    child,
                    &config.variables[child],
                ));
                continue;
            }
            joint_generated_vs_holdout.push(FlatJointMetric {
                allocation: group.clone(),
                parent: edge.parent.clone(),
                child: edge.child.clone(),
                generated_patients: generated_rows.len(),
                holdout_patients: holdout_rows.len(),
                total_variation_distance: flat_joint_tvd(
                    generated_rows,
                    holdout_rows,
                    parent,
                    child,
                ),
            });
        }
        for (child, parents) in parent_indexes.iter().enumerate() {
            if parents.len() < 2 || config.variables[child].kind != "categorical" {
                continue;
            }
            let mut columns = parents.clone();
            columns.push(child);
            family_generated_vs_holdout.push(FlatFamilyMetric {
                allocation: group.clone(),
                parents: parents
                    .iter()
                    .map(|p| config.variables[*p].column.clone())
                    .collect(),
                child: config.variables[child].column.clone(),
                generated_patients: generated_rows.len(),
                holdout_patients: holdout_rows.len(),
                total_variation_distance: flat_family_tvd(generated_rows, holdout_rows, &columns),
            });
        }
    }
    Ok(FlatCohortReport { schema_version: "1".into(), allocation_column: config.allocation_column.clone(), total_patients,
        split: FlatSplitReport { seed: config.split_seed, development_proportion: config.development_proportion, development_patients, holdout_patients, groups: split_groups, algorithm: "FNV-1a 64-bit seeded row-key hash with SplitMix64 finalizer; rank within allocation group and assign rounded proportion to development".into() }, groups,
        partitions: vec![FlatPartitionReport { partition:"development".into(), patients:development_patients, groups:development_groups }, FlatPartitionReport { partition:"holdout".into(), patients:holdout_patients, groups:holdout_groups }], development_validation, generated, generated_vs_holdout, joint_generated_vs_holdout, conditional_numeric_generated_vs_holdout, family_generated_vs_holdout, sparse_backoffs, learned_dependencies, binned_models, numeric_smoothing, numeric_interpolation, fidelity,
        benchmark: FlatBenchmarkResult { method: match config.generator { FlatGenerator::RowBootstrap => "development-fitted empirical complete-patient-row bootstrap", FlatGenerator::IndependentMarginals => "development-fitted independent empirical marginal distributions", FlatGenerator::ConditionalCategorical => "development-fitted empirical conditional distributions with categorical parents", FlatGenerator::LearnedCategoricalForest => "development-fitted ordered categorical dependency forest", FlatGenerator::LearnedMixedForest => "development-fitted ordered mixed-child dependency forest", FlatGenerator::LearnedBinnedForest => "development-fitted ordered binned mixed-type dependency forest", FlatGenerator::RegularizedBinnedForest => "development-fitted penalized ordered binned mixed-type dependency forest", FlatGenerator::LearnedBinnedDag => "development-fitted penalized ordered two-parent binned dependency model", FlatGenerator::InteractionBinnedDag => "development-fitted binned dependency model with pair interaction search" }.to_owned() + " compared with designated holdout, stratified by allocation", generator_seed: config.generator_seed.unwrap_or(config.split_seed), sparse_policy: config.sparse_policy, uncertainty: format!("{}-replicate patient bootstrap for development/holdout metrics; {:.1}% percentile intervals; seed derived from split seed; generated-vs-holdout metrics are point estimates", config.bootstrap_replicates, config.bootstrap_confidence_level*100.0), passed, failed, unavailable, thresholds: config.thresholds.clone() },
        limitations: vec!["Descriptive group summaries only; no treatment-effect or causal estimate.".into(), match config.generator { FlatGenerator::RowBootstrap => "Row resampling reproduces development rows before optional numeric smoothing.", FlatGenerator::IndependentMarginals => "Independent marginal sampling can create new combinations but does not model dependencies between variables or their missingness, and can create implausible records.", FlatGenerator::ConditionalCategorical => "Conditional sampling models only declared relationships with categorical parents, including recorded missing tokens; sparse cells may be unstable and other combinations can be implausible. The configured sparse policy determines whether unsupported combinations fail or drop parents.", FlatGenerator::LearnedCategoricalForest => "Learned categorical associations depend on the supplied order, include missing tokens, and may overfit sparse or high-cardinality categories. Numeric variables are sampled independently. Edges are descriptive, not causal.", FlatGenerator::LearnedMixedForest => "Learned associations depend on the supplied order and may overfit sparse or high-cardinality categories. Numeric parent selection uses development quartile bins and raw missing and structural-absence states; numeric samples remain empirical. Edges are descriptive, not causal.", FlatGenerator::LearnedBinnedForest => "Learned associations depend on supplied order and development quantile bins; sparse and high-cardinality states can overfit. Numeric samples remain empirical and binned joint comparisons lose within-bin information. Edges are descriptive, not causal.", FlatGenerator::RegularizedBinnedForest => "Penalized association selection depends on supplied order and data-derived bins; its penalty does not prevent overfitting or establish significance. Numeric values remain empirical, and binned comparisons lose within-bin information. Edges are descriptive, not causal.", FlatGenerator::LearnedBinnedDag => "Greedy two-parent selection depends on order and bins and can miss interactions or overfit. Sparse combinations follow the configured backoff policy. Edges are descriptive, not causal.", FlatGenerator::InteractionBinnedDag => "Pair interaction search considers pairs only when no single parent is selected; searching many pairs can overfit. Sparse combinations follow the configured backoff policy. Edges are descriptive, not causal." }.into(), "Holdout independence and prior use cannot be verified by this report.".into(), "Optional numeric smoothing or interpolation can create unseen values; neither corrects source biases or guarantees privacy.".into(), "Blank and NA tokens are counted as missing unless explicitly declared structural absence; structural absence requires separately adjudicated rules.".into(), "Joint comparisons treat distinct raw missing tokens as separate states; results may be sensitive.".into(), "Numeric conditional comparisons are pairwise; full multi-parent numeric relationships and uncertainty are not evaluated.".into(), "Utility scores use categorical predictor states, declared timing and binary labels; training sizes can differ, and they do not establish clinical or causal validity.".into(), "Thresholds are descriptive workflow diagnostics, not clinical standards.".into(), "Bootstrap intervals describe sampling variability conditional on this cohort and split; small strata can be unstable.".into(), "Generated records remain transient; only aggregate synthetic-to-holdout summaries are emitted.".into(), "Rows and identifiers are not included in this report.".into()] })
}

// Fit smoothed conditional frequencies; unseen states use the smoothed overall rate.
fn flat_utility_score(
    training: &[&Vec<String>],
    validation: &[Vec<String>],
    task: &FlatUtilityTask,
    predictors: &[usize],
    target: usize,
) -> FlatUtilityScore {
    let label = |value: &String| {
        if task.positive_values.contains(value) {
            Some(1.0)
        } else if task.negative_values.contains(value) {
            Some(0.0)
        } else {
            None
        }
    };
    let mut states: BTreeMap<Vec<&str>, (usize, usize)> = BTreeMap::new();
    let (mut total, mut positive) = (0usize, 0usize);
    for row in training {
        if let Some(y) = label(&row[target]) {
            let counts = states
                .entry(predictors.iter().map(|&i| row[i].as_str()).collect())
                .or_default();
            counts.0 += 1;
            counts.1 += y as usize;
            total += 1;
            positive += y as usize;
        }
    }
    let (mut n, mut unseen, mut loss) = (0usize, 0usize, 0.0);
    for row in validation {
        if let Some(y) = label(&row[target]) {
            let key: Vec<_> = predictors.iter().map(|&i| row[i].as_str()).collect();
            let (count, positives) = match states.get(&key) {
                Some(&counts) => counts,
                None => {
                    unseen += 1;
                    (total, positive)
                }
            };
            let probability = (positives as f64 + task.alpha) / (count as f64 + 2.0 * task.alpha);
            loss += (probability - y).powi(2);
            n += 1;
        }
    }
    FlatUtilityScore {
        training_labeled: total,
        training_unlabeled: training.len() - total,
        validation_labeled: n,
        unseen_predictor: unseen,
        brier_score: (total > 0 && n > 0).then_some(loss / n as f64),
    }
}

fn flat_development_validation(
    group: &str,
    data: &[(String, Vec<String>)],
    development_ids: &HashSet<&str>,
    proportion: f64,
    config: &FlatCohortConfig,
    configured_parents: &[Vec<usize>],
) -> Result<FlatDevelopmentValidation, Box<dyn Error>> {
    // A domain-separated seed prevents reusing the outer ranking for the inner split.
    let split_seed = config
        .development_validation_seed
        .unwrap_or(config.split_seed ^ 0x7661_6c69_6461_7465);
    let generator_seed = config.generator_seed.unwrap_or(config.split_seed) ^ 0x696e_6e65_725f_6765;
    let mut ranked: Vec<_> = data
        .iter()
        .filter(|(id, _)| development_ids.contains(id.as_str()))
        .map(|(id, _)| (flat_hash(split_seed, id), id.as_str()))
        .collect();
    ranked.sort();
    let validation_count = (ranked.len() as f64 * proportion).round() as usize;
    if validation_count == 0 || validation_count == ranked.len() {
        return Err(format!("development validation requires nonempty training and validation partitions in group {group}").into());
    }
    let validation_ids: HashSet<_> = ranked
        .iter()
        .take(validation_count)
        .map(|(_, id)| *id)
        .collect();
    let training: Vec<_> = data
        .iter()
        .filter(|(id, _)| {
            development_ids.contains(id.as_str()) && !validation_ids.contains(id.as_str())
        })
        .map(|(_, row)| row)
        .collect();
    let validation: Vec<_> = data
        .iter()
        .filter(|(id, _)| validation_ids.contains(id.as_str()))
        .map(|(_, row)| row.clone())
        .collect();
    let binned_fit = if matches!(
        config.generator,
        FlatGenerator::LearnedBinnedForest
            | FlatGenerator::RegularizedBinnedForest
            | FlatGenerator::LearnedBinnedDag
            | FlatGenerator::InteractionBinnedDag
    ) {
        Some(fit_flat_binned_selected(
            &training,
            &config.variables,
            group,
            config.generator,
        ))
    } else {
        None
    };
    let (learned_parents, mut learned_dependencies) = if matches!(
        config.generator,
        FlatGenerator::LearnedCategoricalForest | FlatGenerator::LearnedMixedForest
    ) {
        fit_flat_ordered_forest(
            &training,
            &config.variables,
            group,
            config.generator == FlatGenerator::LearnedMixedForest,
        )
    } else {
        (vec![Vec::new(); config.variables.len()], Vec::new())
    };
    if let Some(fit) = &binned_fit {
        learned_dependencies.extend(fit.edges.iter().cloned());
    }
    let mut state = generator_seed ^ flat_hash(generator_seed, group);
    let mut counts = vec![BTreeMap::new(); config.variables.len()];
    let mut generated: Vec<Vec<String>> = (0..validation.len())
        .map(|_| match config.generator {
            FlatGenerator::RowBootstrap => {
                Ok(training[flat_sample_index(&mut state, training.len())].clone())
            }
            FlatGenerator::IndependentMarginals => Ok(sample_flat_marginals(
                &training,
                config.variables.len(),
                &mut state,
            )),
            FlatGenerator::ConditionalCategorical => sample_flat_conditionals(
                &training,
                configured_parents,
                &mut state,
                config.sparse_policy,
                &mut counts,
            ),
            FlatGenerator::LearnedCategoricalForest | FlatGenerator::LearnedMixedForest => {
                sample_flat_conditionals(
                    &training,
                    &learned_parents,
                    &mut state,
                    FlatSparsePolicy::Fail,
                    &mut counts,
                )
            }
            FlatGenerator::LearnedBinnedForest
            | FlatGenerator::RegularizedBinnedForest
            | FlatGenerator::LearnedBinnedDag
            | FlatGenerator::InteractionBinnedDag => sample_flat_binned_checked(
                &training,
                binned_fit.as_ref().expect("binned fit"),
                &mut state,
                config.sparse_policy,
                &mut counts,
            ),
        })
        .collect::<Result<_, Box<dyn Error>>>()?;
    let numeric_smoothing = apply_flat_numeric_smoothing(
        &mut generated,
        &training,
        config,
        binned_fit.as_ref(),
        group,
        generator_seed,
    )?;
    let numeric_interpolation = if config.numeric_interpolation.is_empty() {
        Vec::new()
    } else {
        apply_flat_numeric_interpolation(
            &mut generated,
            &training,
            &config.variables,
            &config.numeric_interpolation,
            binned_fit.as_ref().expect("validated binned model"),
            group,
            generator_seed,
        )
    };
    let mut categorical_pairs = Vec::new();
    let mut numeric_pairs = Vec::new();
    for (child, parents) in configured_parents.iter().enumerate() {
        for &parent in parents {
            if config.variables[child].kind == "numeric" {
                numeric_pairs.extend(flat_conditional_numeric_metrics(
                    group,
                    &config.variables[parent].column,
                    &generated,
                    &validation,
                    parent,
                    child,
                    &config.variables[child],
                ));
            } else {
                categorical_pairs.push(FlatJointMetric {
                    allocation: group.into(),
                    parent: config.variables[parent].column.clone(),
                    child: config.variables[child].column.clone(),
                    generated_patients: generated.len(),
                    holdout_patients: validation.len(),
                    total_variation_distance: flat_joint_tvd(
                        &generated,
                        &validation,
                        parent,
                        child,
                    ),
                });
            }
        }
    }
    let sparse_backoffs = counts
        .into_iter()
        .enumerate()
        .filter(|(_, counts)| !counts.is_empty())
        .map(|(i, dropped_parent_counts)| FlatBackoffReport {
            allocation: group.into(),
            child: config.variables[i].column.clone(),
            dropped_parent_counts,
        })
        .collect();
    let numeric_associations = config
        .numeric_associations
        .iter()
        .map(|pair| {
            let first = config
                .variables
                .iter()
                .position(|v| v.column == pair.parent)
                .expect("validated numeric pair");
            let second = config
                .variables
                .iter()
                .position(|v| v.column == pair.child)
                .expect("validated numeric pair");
            let (generated_complete_pairs, generated_spearman) =
                flat_spearman(&generated, &config.variables, first, second);
            let (validation_complete_pairs, validation_spearman) =
                flat_spearman(&validation, &config.variables, first, second);
            FlatNumericAssociation {
                first: pair.parent.clone(),
                second: pair.child.clone(),
                generated_complete_pairs,
                validation_complete_pairs,
                generated_spearman,
                validation_spearman,
                absolute_difference: generated_spearman
                    .zip(validation_spearman)
                    .map(|(a, b)| (a - b).abs()),
            }
        })
        .collect();
    Ok(FlatDevelopmentValidation {
        group: group.into(),
        split_seed,
        generator_seed,
        training_patients: training.len(),
        validation_patients: validation.len(),
        generated: summarize_flat_rows(group, &generated, &config.variables),
        validation: summarize_flat_rows(group, &validation, &config.variables),
        categorical_pairs,
        numeric_pairs,
        numeric_associations,
        joint_families: config
            .joint_families
            .iter()
            .map(|family| {
                let columns: Vec<_> = family
                    .iter()
                    .map(|name| {
                        config
                            .variables
                            .iter()
                            .position(|v| &v.column == name)
                            .expect("validated family")
                    })
                    .collect();
                FlatJointFamilyMetric {
                    variables: family.clone(),
                    generated_patients: generated.len(),
                    validation_patients: validation.len(),
                    total_variation_distance: flat_family_tvd(&generated, &validation, &columns),
                }
            })
            .collect(),
        utility: config
            .utility_tasks
            .iter()
            .map(|task| {
                let predictors: Vec<_> = std::iter::once(&task.predictor)
                    .chain(task.additional_predictors.iter())
                    .map(|name| {
                        config
                            .variables
                            .iter()
                            .position(|v| &v.column == name)
                            .expect("validated predictor")
                    })
                    .collect();
                let target = config
                    .variables
                    .iter()
                    .position(|v| v.column == task.target)
                    .expect("validated target");
                let synthetic: Vec<_> = generated.iter().collect();
                let real_training =
                    flat_utility_score(&training, &validation, task, &predictors, target);
                let synthetic_training =
                    flat_utility_score(&synthetic, &validation, task, &predictors, target);
                FlatUtilityReport {
                    task: task.name.clone(),
                    validation_unlabeled: validation.len() - real_training.validation_labeled,
                    real_training,
                    synthetic_training,
                }
            })
            .collect(),
        exact_matches: config
            .exact_match_diagnostics
            .then(|| flat_exact_matches(&training, &validation, &generated)),
        numeric_smoothing,
        numeric_interpolation,
        learned_dependencies,
        binned_model: binned_fit
            .as_ref()
            .map(|fit| flat_binned_report(group, fit, &config.variables, &generated, &validation)),
        sparse_backoffs,
    })
}

fn flat_exact_matches(
    training: &[&Vec<String>],
    validation: &[Vec<String>],
    generated: &[Vec<String>],
) -> FlatExactMatches {
    let training_set: HashSet<&[String]> = training.iter().map(|row| row.as_slice()).collect();
    let validation_set: HashSet<&[String]> = validation.iter().map(|row| row.as_slice()).collect();
    let generated_set: HashSet<&[String]> = generated.iter().map(|row| row.as_slice()).collect();
    FlatExactMatches {
        training_patients: training.len(),
        validation_patients: validation.len(),
        generated_patients: generated.len(),
        generated_matching_training: generated
            .iter()
            .filter(|row| training_set.contains(row.as_slice()))
            .count(),
        generated_matching_validation: generated
            .iter()
            .filter(|row| validation_set.contains(row.as_slice()))
            .count(),
        validation_matching_training: validation
            .iter()
            .filter(|row| training_set.contains(row.as_slice()))
            .count(),
        duplicate_generated_rows: generated.len() - generated_set.len(),
    }
}

fn flat_spearman(
    rows: &[Vec<String>],
    variables: &[FlatVariable],
    first: usize,
    second: usize,
) -> (usize, Option<f64>) {
    let observed = |value: &str, variable: &FlatVariable| -> Option<f64> {
        if flat_is_missing(value, variable)
            || variable
                .structural_absence_values
                .iter()
                .any(|v| v == value)
        {
            None
        } else {
            value.parse::<f64>().ok().filter(|x| x.is_finite())
        }
    };
    let pairs: Vec<_> = rows
        .iter()
        .filter_map(|row| {
            Some((
                observed(&row[first], &variables[first])?,
                observed(&row[second], &variables[second])?,
            ))
        })
        .collect();
    let n = pairs.len();
    if n < 2 {
        return (n, None);
    }
    let ranks = |column: usize| {
        let mut values: Vec<_> = pairs
            .iter()
            .enumerate()
            .map(|(i, p)| (if column == 0 { p.0 } else { p.1 }, i))
            .collect();
        values.sort_by(|a, b| a.0.total_cmp(&b.0));
        let mut ranks = vec![0.0; n];
        let mut start = 0;
        while start < n {
            let mut end = start + 1;
            while end < n && values[end].0 == values[start].0 {
                end += 1;
            }
            let rank = (start + end - 1) as f64 / 2.0;
            for &(_, index) in &values[start..end] {
                ranks[index] = rank;
            }
            start = end;
        }
        ranks
    };
    let x = ranks(0);
    let y = ranks(1);
    let mean = (n - 1) as f64 / 2.0;
    let covariance: f64 = x.iter().zip(&y).map(|(a, b)| (a - mean) * (b - mean)).sum();
    let xx: f64 = x.iter().map(|a| (a - mean).powi(2)).sum();
    let yy: f64 = y.iter().map(|a| (a - mean).powi(2)).sum();
    let correlation = if xx > 0.0 && yy > 0.0 {
        Some((covariance / (xx.sqrt() * yy.sqrt())).clamp(-1.0, 1.0))
    } else {
        None
    };
    (n, correlation)
}

fn flat_random_u64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9e3779b97f4a7c15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
    z ^ (z >> 31)
}

fn flat_sample_index(state: &mut u64, len: usize) -> usize {
    assert!(len > 0, "cannot sample an index from an empty collection");
    let bound = len as u64;
    let threshold = bound.wrapping_neg() % bound;
    loop {
        let value = flat_random_u64(state);
        if value >= threshold {
            return (value % bound) as usize;
        }
    }
}

fn apply_flat_numeric_smoothing(
    generated: &mut [Vec<String>],
    training: &[&Vec<String>],
    config: &FlatCohortConfig,
    fit: Option<&FlatBinnedFit>,
    group: &str,
    generator_seed: u64,
) -> Result<Vec<FlatSmoothingReport>, Box<dyn Error>> {
    let mut random_state =
        generator_seed ^ flat_hash(generator_seed, group) ^ 0x736d_6f6f_7468_696e;
    let mut reports = Vec::new();
    for smoothing in &config.numeric_smoothing {
        let index = config
            .variables
            .iter()
            .position(|v| v.column == smoothing.variable)
            .expect("validated smoothing variable");
        let variable = &config.variables[index];
        let observed = |value: &str| -> Option<f64> {
            if flat_is_missing(value, variable)
                || variable
                    .structural_absence_values
                    .iter()
                    .any(|v| v == value)
            {
                None
            } else {
                value.parse::<f64>().ok()
            }
        };
        for row in training {
            if let Some(value) = observed(&row[index]) {
                if value < smoothing.minimum || value > smoothing.maximum {
                    return Err(format!(
                        "training value outside numeric smoothing bounds for {} in group {group}",
                        variable.column
                    )
                    .into());
                }
            }
        }
        let mut attempted = 0;
        let mut changed = 0;
        for row in generated.iter_mut() {
            let Some(value) = observed(&row[index]) else {
                continue;
            };
            attempted += 1;
            let lo = (value - smoothing.bandwidth).max(smoothing.minimum);
            let hi = (value + smoothing.bandwidth).min(smoothing.maximum);
            if lo >= hi {
                continue;
            }
            let original_state =
                fit.map(|model| flat_binned_state(&row[index], variable, &model.cuts[index]));
            for _ in 0..16 {
                let unit =
                    (flat_random_u64(&mut random_state) >> 11) as f64 / ((1_u64 << 53) as f64);
                let candidate = lo + (hi - lo) * unit;
                if !candidate.is_finite() || candidate == value {
                    continue;
                }
                let text = candidate.to_string();
                if variable.structural_absence_values.contains(&text) {
                    continue;
                }
                if let (Some(model), Some(original)) = (fit, &original_state) {
                    if flat_binned_state(&text, variable, &model.cuts[index]) != *original {
                        continue;
                    }
                }
                row[index] = text;
                changed += 1;
                break;
            }
        }
        reports.push(FlatSmoothingReport {
            allocation: group.into(),
            variable: variable.column.clone(),
            observed_candidates: attempted,
            changed,
            unchanged: attempted - changed,
        });
    }
    Ok(reports)
}

fn sample_flat_marginals(rows: &[&Vec<String>], columns: usize, state: &mut u64) -> Vec<String> {
    (0..columns)
        .map(|j| rows[flat_sample_index(state, rows.len())][j].clone())
        .collect()
}

fn apply_flat_numeric_interpolation(
    generated: &mut [Vec<String>],
    training: &[&Vec<String>],
    variables: &[FlatVariable],
    interpolation_columns: &[String],
    fit: &FlatBinnedFit,
    group: &str,
    generator_seed: u64,
) -> Vec<FlatSmoothingReport> {
    let mut random_state =
        generator_seed ^ flat_hash(generator_seed, group) ^ 0x696e_7465_7270_6f6c;
    let mut reports = Vec::new();
    for name in interpolation_columns {
        let index = variables
            .iter()
            .position(|v| v.column == *name)
            .expect("validated interpolation variable");
        let variable = &variables[index];
        let mut attempted = 0;
        let mut changed = 0;
        for row in generated.iter_mut() {
            let child_state = flat_binned_state(&row[index], variable, &fit.cuts[index]);
            if !child_state.starts_with("bin:") {
                continue;
            }
            let value = row[index].parse::<f64>().expect("validated numeric value");
            attempted += 1;
            let parent_states: Vec<_> = fit.parents[index]
                .iter()
                .map(|&parent| {
                    (
                        parent,
                        flat_binned_state(&row[parent], &variables[parent], &fit.cuts[parent]),
                    )
                })
                .collect();
            let mut candidates: Vec<f64> = training
                .iter()
                .enumerate()
                .filter(|(i, _)| {
                    fit.states[*i][index] == child_state
                        && parent_states
                            .iter()
                            .all(|(parent, state)| fit.states[*i][*parent] == *state)
                })
                .filter_map(|(_, training_row)| training_row[index].parse::<f64>().ok())
                .collect();
            candidates.sort_by(f64::total_cmp);
            candidates.dedup_by(|a, b| *a == *b);
            let position = candidates.partition_point(|candidate| *candidate < value);
            if candidates.get(position).copied() != Some(value) {
                continue;
            }
            let mut neighbors = Vec::new();
            if position > 0 {
                neighbors.push(candidates[position - 1]);
            }
            if position + 1 < candidates.len() {
                neighbors.push(candidates[position + 1]);
            }
            neighbors.retain(|neighbor| (*neighbor - value).is_finite());
            if neighbors.is_empty() {
                continue;
            }
            for _ in 0..16 {
                let neighbor = neighbors[flat_sample_index(&mut random_state, neighbors.len())];
                let unit =
                    (flat_random_u64(&mut random_state) >> 11) as f64 / ((1_u64 << 53) as f64);
                let candidate = value + (neighbor - value) * unit;
                if !candidate.is_finite()
                    || candidate <= value.min(neighbor)
                    || candidate >= value.max(neighbor)
                {
                    continue;
                }
                let text = candidate.to_string();
                if variable.structural_absence_values.contains(&text)
                    || flat_binned_state(&text, variable, &fit.cuts[index]) != child_state
                {
                    continue;
                }
                row[index] = text;
                changed += 1;
                break;
            }
        }
        reports.push(FlatSmoothingReport {
            allocation: group.into(),
            variable: name.clone(),
            observed_candidates: attempted,
            changed,
            unchanged: attempted - changed,
        });
    }
    reports
}

struct FlatBinnedFit {
    cuts: Vec<Vec<f64>>,
    states: Vec<Vec<String>>,
    parents: Vec<Vec<usize>>,
    edges: Vec<FlatLearnedDependency>,
}

fn flat_binned_state(value: &str, variable: &FlatVariable, cuts: &[f64]) -> String {
    if variable.kind == "numeric"
        && !variable
            .structural_absence_values
            .iter()
            .any(|v| v == value)
    {
        if let Ok(number) = value.parse::<f64>() {
            return format!("bin:{}", cuts.partition_point(|cut| number > *cut));
        }
    }
    format!("raw:{value}")
}

fn fit_flat_binned_forest(
    rows: &[&Vec<String>],
    variables: &[FlatVariable],
    group: &str,
    regularized: bool,
) -> FlatBinnedFit {
    let cuts: Vec<Vec<f64>> = variables
        .iter()
        .enumerate()
        .map(|(i, variable)| {
            if variable.kind != "numeric" {
                return Vec::new();
            }
            let mut values: Vec<f64> = rows
                .iter()
                .filter(|row| !variable.structural_absence_values.contains(&row[i]))
                .filter_map(|row| row[i].parse::<f64>().ok())
                .collect();
            values.sort_by(f64::total_cmp);
            if values.is_empty() {
                Vec::new()
            } else {
                (1..4).map(|j| values[(values.len() - 1) * j / 4]).collect()
            }
        })
        .collect();
    let states: Vec<Vec<String>> = rows
        .iter()
        .map(|row| {
            variables
                .iter()
                .enumerate()
                .map(|(i, variable)| flat_binned_state(&row[i], variable, &cuts[i]))
                .collect()
        })
        .collect();
    let state_refs: Vec<_> = states.iter().collect();
    let mut categorical_variables = variables.to_vec();
    for variable in &mut categorical_variables {
        variable.kind = "categorical".into();
    }
    let (parents, edges) = if regularized {
        fit_flat_penalized_forest(&state_refs, &categorical_variables, group)
    } else {
        fit_flat_ordered_forest(&state_refs, &categorical_variables, group, false)
    };
    FlatBinnedFit {
        cuts,
        states,
        parents,
        edges,
    }
}

// Add one earlier parent when its conditional information exceeds a BIC-like
// penalty. Greedy selection does not discover purely synergistic pairs.
fn fit_flat_binned_selected(
    rows: &[&Vec<String>],
    variables: &[FlatVariable],
    group: &str,
    generator: FlatGenerator,
) -> FlatBinnedFit {
    let mut fit = fit_flat_binned_forest(
        rows,
        variables,
        group,
        generator != FlatGenerator::LearnedBinnedForest,
    );
    if !matches!(
        generator,
        FlatGenerator::LearnedBinnedDag | FlatGenerator::InteractionBinnedDag
    ) || rows.is_empty()
    {
        return fit;
    }
    let n = rows.len() as f64;
    for child in 0..variables.len() {
        let Some(&first) = fit.parents[child].first() else {
            if generator == FlatGenerator::InteractionBinnedDag {
                fit_flat_joint_parent_pair(&mut fit, variables, group, child);
            }
            continue;
        };
        let mut best: Option<(usize, f64, f64)> = None;
        for candidate in 0..child {
            if candidate == first {
                continue;
            }
            let mut first_counts = BTreeMap::new();
            let mut first_candidate = BTreeMap::new();
            let mut first_child = BTreeMap::new();
            let mut joint = BTreeMap::new();
            let mut candidate_states = BTreeSet::new();
            let mut child_states = BTreeSet::new();
            for row in &fit.states {
                let (a, b, c) = (
                    row[first].as_str(),
                    row[candidate].as_str(),
                    row[child].as_str(),
                );
                *first_counts.entry(a).or_insert(0usize) += 1;
                *first_candidate.entry((a, b)).or_insert(0usize) += 1;
                *first_child.entry((a, c)).or_insert(0usize) += 1;
                *joint.entry((a, b, c)).or_insert(0usize) += 1;
                candidate_states.insert(b);
                child_states.insert(c);
            }
            let information: f64 = joint
                .iter()
                .map(|(&(a, b, c), &count)| {
                    let count = count as f64;
                    count / n
                        * (count * first_counts[a] as f64
                            / (first_candidate[&(a, b)] as f64 * first_child[&(a, c)] as f64))
                            .ln()
                })
                .sum();
            let degrees = first_counts.len() as f64
                * candidate_states.len().saturating_sub(1) as f64
                * child_states.len().saturating_sub(1) as f64;
            let gain = information - degrees * n.ln() / (2.0 * n);
            if gain > 1e-12 && best.is_none_or(|(_, _, best_gain)| gain > best_gain + 1e-12) {
                best = Some((candidate, information, gain));
            }
        }
        if let Some((candidate, information, gain)) = best {
            fit.parents[child].push(candidate);
            fit.edges.push(FlatLearnedDependency {
                allocation: group.into(),
                parent: variables[candidate].column.clone(),
                child: variables[child].column.clone(),
                mutual_information_nats: information,
                conditioned_on: Some(variables[first].column.clone()),
                joint_selection_gain_nats: None,
                penalized_gain_nats: Some(gain),
                development_patients: rows.len(),
            });
        }
    }
    fit
}

// Jointly evaluate pairs only for children with no penalized single parent.
// The chosen pair's score is relative to an independent child, not an edge score.
fn fit_flat_joint_parent_pair(
    fit: &mut FlatBinnedFit,
    variables: &[FlatVariable],
    group: &str,
    child: usize,
) {
    let n = fit.states.len() as f64;
    let mut best: Option<(usize, usize, f64, f64, f64)> = None;
    for a in 0..child {
        for b in a + 1..child {
            let mut a_counts = BTreeMap::new();
            let mut child_counts = BTreeMap::new();
            let mut ac = BTreeMap::new();
            let mut ab = BTreeMap::new();
            let mut abc = BTreeMap::new();
            for row in &fit.states {
                let (x, y, z) = (row[a].as_str(), row[b].as_str(), row[child].as_str());
                *a_counts.entry(x).or_insert(0usize) += 1;
                *child_counts.entry(z).or_insert(0usize) += 1;
                *ac.entry((x, z)).or_insert(0usize) += 1;
                *ab.entry((x, y)).or_insert(0usize) += 1;
                *abc.entry((x, y, z)).or_insert(0usize) += 1;
            }
            let joint_info: f64 = abc
                .iter()
                .map(|(&(x, y, z), &count)| {
                    let k = count as f64;
                    k / n * (k * n / (ab[&(x, y)] as f64 * child_counts[z] as f64)).ln()
                })
                .sum();
            let first_info: f64 = ac
                .iter()
                .map(|(&(x, z), &count)| {
                    let k = count as f64;
                    k / n * (k * n / (a_counts[x] as f64 * child_counts[z] as f64)).ln()
                })
                .sum();
            let degrees =
                ab.len().saturating_sub(1) as f64 * child_counts.len().saturating_sub(1) as f64;
            let gain = joint_info - degrees * n.ln() / (2.0 * n);
            if gain > 1e-12 && best.is_none_or(|(_, _, _, _, old)| gain > old + 1e-12) {
                best = Some((a, b, first_info, joint_info, gain));
            }
        }
    }
    if let Some((a, b, first_info, joint_info, gain)) = best {
        fit.parents[child].extend([a, b]);
        fit.edges.push(FlatLearnedDependency {
            allocation: group.into(),
            parent: variables[a].column.clone(),
            child: variables[child].column.clone(),
            mutual_information_nats: first_info,
            conditioned_on: None,
            joint_selection_gain_nats: Some(gain),
            penalized_gain_nats: None,
            development_patients: fit.states.len(),
        });
        fit.edges.push(FlatLearnedDependency {
            allocation: group.into(),
            parent: variables[b].column.clone(),
            child: variables[child].column.clone(),
            mutual_information_nats: (joint_info - first_info).max(0.0),
            conditioned_on: Some(variables[a].column.clone()),
            joint_selection_gain_nats: Some(gain),
            penalized_gain_nats: None,
            development_patients: fit.states.len(),
        });
    }
}

fn fit_flat_penalized_forest(
    rows: &[&Vec<String>],
    variables: &[FlatVariable],
    group: &str,
) -> (Vec<Vec<usize>>, Vec<FlatLearnedDependency>) {
    let mut parents = vec![Vec::new(); variables.len()];
    let mut edges = Vec::new();
    if rows.is_empty() {
        return (parents, edges);
    }
    let n = rows.len() as f64;
    for child in 0..variables.len() {
        let mut child_counts: BTreeMap<&str, usize> = BTreeMap::new();
        for row in rows {
            *child_counts.entry(&row[child]).or_default() += 1;
        }
        let mut best = None;
        let mut best_gain = 0.0;
        for parent in 0..child {
            let mut parent_counts: BTreeMap<&str, usize> = BTreeMap::new();
            let mut joint: BTreeMap<(&str, &str), usize> = BTreeMap::new();
            for row in rows {
                *parent_counts.entry(&row[parent]).or_default() += 1;
                *joint.entry((&row[parent], &row[child])).or_default() += 1;
            }
            let information: f64 = joint
                .iter()
                .map(|((p, c), count)| {
                    let count = *count as f64;
                    count / n
                        * (count * n / (parent_counts[p] as f64 * child_counts[c] as f64)).ln()
                })
                .sum();
            let degrees = (parent_counts.len().saturating_sub(1) as f64)
                * (child_counts.len().saturating_sub(1) as f64);
            let gain = information - degrees * n.ln() / (2.0 * n);
            if gain > best_gain + 1e-12 {
                best = Some((parent, information));
                best_gain = gain;
            }
        }
        if let Some((parent, information)) = best {
            parents[child].push(parent);
            edges.push(FlatLearnedDependency {
                allocation: group.into(),
                parent: variables[parent].column.clone(),
                child: variables[child].column.clone(),
                mutual_information_nats: information,
                conditioned_on: None,
                joint_selection_gain_nats: None,
                penalized_gain_nats: Some(best_gain),
                development_patients: rows.len(),
            });
        }
    }
    (parents, edges)
}

#[cfg(test)]
fn sample_flat_binned(rows: &[&Vec<String>], fit: &FlatBinnedFit, state: &mut u64) -> Vec<String> {
    let mut counts = vec![BTreeMap::new(); fit.parents.len()];
    sample_flat_binned_checked(rows, fit, state, FlatSparsePolicy::Fail, &mut counts)
        .expect("single-parent fitted forest has support")
}

fn sample_flat_binned_checked(
    rows: &[&Vec<String>],
    fit: &FlatBinnedFit,
    state: &mut u64,
    policy: FlatSparsePolicy,
    counts: &mut [BTreeMap<usize, usize>],
) -> Result<Vec<String>, Box<dyn Error>> {
    let mut sample = Vec::with_capacity(fit.parents.len());
    let mut sample_states = Vec::with_capacity(fit.parents.len());
    for (column, parents) in fit.parents.iter().enumerate() {
        let mut retained = parents.len();
        let candidates: Vec<_> = loop {
            let found: Vec<_> = fit
                .states
                .iter()
                .enumerate()
                .filter(|(_, row)| {
                    parents[..retained]
                        .iter()
                        .all(|parent| row[*parent] == sample_states[*parent])
                })
                .map(|(index, _)| index)
                .collect();
            if !found.is_empty() {
                break found;
            }
            if policy == FlatSparsePolicy::Fail || retained == 0 {
                return Err(format!(
                    "unsupported generated binned parent combination at variable index {column}"
                )
                .into());
            }
            retained -= 1;
        };
        let dropped = parents.len() - retained;
        if dropped > 0 {
            *counts[column].entry(dropped).or_default() += 1;
        }
        let selected = candidates[flat_sample_index(state, candidates.len())];
        sample.push(rows[selected][column].clone());
        sample_states.push(fit.states[selected][column].clone());
    }
    Ok(sample)
}

fn flat_binned_report(
    group: &str,
    fit: &FlatBinnedFit,
    variables: &[FlatVariable],
    generated: &[Vec<String>],
    reference: &[Vec<String>],
) -> FlatBinnedModel {
    let to_states = |rows: &[Vec<String>]| -> Vec<Vec<String>> {
        rows.iter()
            .map(|row| {
                variables
                    .iter()
                    .enumerate()
                    .map(|(i, variable)| flat_binned_state(&row[i], variable, &fit.cuts[i]))
                    .collect()
            })
            .collect()
    };
    let generated_states = to_states(generated);
    let reference_states = to_states(reference);
    let selected_pairs = fit
        .edges
        .iter()
        .map(|edge| {
            let parent = variables
                .iter()
                .position(|v| v.column == edge.parent)
                .expect("learned parent");
            let child = variables
                .iter()
                .position(|v| v.column == edge.child)
                .expect("learned child");
            FlatJointMetric {
                allocation: group.into(),
                parent: edge.parent.clone(),
                child: edge.child.clone(),
                generated_patients: generated.len(),
                holdout_patients: reference.len(),
                total_variation_distance: flat_joint_tvd(
                    &generated_states,
                    &reference_states,
                    parent,
                    child,
                ),
            }
        })
        .collect();
    let numeric_bins = variables
        .iter()
        .enumerate()
        .filter(|(_, v)| v.kind == "numeric")
        .map(|(i, variable)| FlatNumericBins {
            variable: variable.column.clone(),
            upper_bounds: fit.cuts[i].clone(),
        })
        .collect();
    FlatBinnedModel {
        allocation: group.into(),
        numeric_bins,
        selected_pairs,
    }
}

fn fit_flat_ordered_forest(
    rows: &[&Vec<String>],
    variables: &[FlatVariable],
    group: &str,
    include_numeric: bool,
) -> (Vec<Vec<usize>>, Vec<FlatLearnedDependency>) {
    let mut parents = vec![Vec::new(); variables.len()];
    let mut learned = Vec::new();
    if rows.is_empty() {
        return (parents, learned);
    }
    let n = rows.len() as f64;
    for child in 0..variables.len() {
        if variables[child].kind != "categorical" && !include_numeric {
            continue;
        }
        // Numeric cut points are learned from observed development values only.
        // Ties remain together; missing and structural absence are raw states.
        let mut numeric_values: Vec<f64> = if variables[child].kind == "numeric" {
            rows.iter()
                .filter(|row| {
                    !variables[child]
                        .structural_absence_values
                        .contains(&row[child])
                })
                .filter_map(|row| row[child].parse::<f64>().ok())
                .collect()
        } else {
            Vec::new()
        };
        numeric_values.sort_by(f64::total_cmp);
        let cuts: Vec<f64> = if numeric_values.is_empty() {
            Vec::new()
        } else {
            (1..4)
                .map(|i| numeric_values[(numeric_values.len() - 1) * i / 4])
                .collect()
        };
        let child_states: Vec<(u8, String)> = rows
            .iter()
            .map(|row| {
                let value = &row[child];
                if variables[child].kind == "numeric"
                    && !variables[child].structural_absence_values.contains(value)
                {
                    if let Ok(number) = value.parse::<f64>() {
                        return (1, cuts.partition_point(|cut| number > *cut).to_string());
                    }
                }
                (0, value.clone())
            })
            .collect();
        let mut best = None;
        let mut best_information = 0.0;
        for parent in 0..child {
            if variables[parent].kind != "categorical" {
                continue;
            }
            let mut joint: BTreeMap<(&str, &(u8, String)), usize> = BTreeMap::new();
            let mut parent_counts: BTreeMap<&str, usize> = BTreeMap::new();
            let mut child_counts: BTreeMap<&(u8, String), usize> = BTreeMap::new();
            for (row, c) in rows.iter().zip(&child_states) {
                let p = row[parent].as_str();
                *joint.entry((p, c)).or_default() += 1;
                *parent_counts.entry(p).or_default() += 1;
                *child_counts.entry(c).or_default() += 1;
            }
            let information: f64 = joint
                .iter()
                .map(|((p, c), count)| {
                    let count = *count as f64;
                    (count / n)
                        * (count * n / (parent_counts[p] as f64 * child_counts[c] as f64)).ln()
                })
                .sum();
            if information > best_information + 1e-12 {
                best = Some(parent);
                best_information = information;
            }
        }
        if let Some(parent) = best {
            parents[child].push(parent);
            learned.push(FlatLearnedDependency {
                allocation: group.into(),
                parent: variables[parent].column.clone(),
                child: variables[child].column.clone(),
                mutual_information_nats: best_information,
                conditioned_on: None,
                joint_selection_gain_nats: None,
                penalized_gain_nats: None,
                development_patients: rows.len(),
            });
        }
    }
    (parents, learned)
}

fn sample_flat_conditionals(
    rows: &[&Vec<String>],
    parents: &[Vec<usize>],
    state: &mut u64,
    sparse_policy: FlatSparsePolicy,
    backoff_counts: &mut [BTreeMap<usize, usize>],
) -> Result<Vec<String>, Box<dyn Error>> {
    let mut sample = Vec::with_capacity(parents.len());
    for (column, parent) in parents.iter().enumerate() {
        let mut retained = parent.len();
        let candidates = loop {
            let candidates: Vec<_> = rows
                .iter()
                .filter(|row| parent[..retained].iter().all(|p| row[*p] == sample[*p]))
                .collect();
            if !candidates.is_empty() {
                break candidates;
            }
            if sparse_policy == FlatSparsePolicy::Fail || retained == 0 {
                return Err(format!("unsupported generated parent combination at configured variable index {column}; no output published").into());
            }
            retained -= 1;
        };
        let dropped = parent.len() - retained;
        if dropped > 0 {
            *backoff_counts[column].entry(dropped).or_default() += 1;
        }
        let selected = candidates[flat_sample_index(state, candidates.len())];
        sample.push(selected[column].clone());
    }
    Ok(sample)
}

fn flat_family_tvd(a: &[Vec<String>], b: &[Vec<String>], columns: &[usize]) -> Option<f64> {
    if a.is_empty() || b.is_empty() {
        return None;
    }
    let mut counts: BTreeMap<Vec<&str>, (usize, usize)> = BTreeMap::new();
    for row in a {
        counts
            .entry(columns.iter().map(|i| row[*i].as_str()).collect())
            .or_default()
            .0 += 1;
    }
    for row in b {
        counts
            .entry(columns.iter().map(|i| row[*i].as_str()).collect())
            .or_default()
            .1 += 1;
    }
    Some(
        0.5 * counts
            .values()
            .map(|(x, y)| (*x as f64 / a.len() as f64 - *y as f64 / b.len() as f64).abs())
            .sum::<f64>(),
    )
}

fn flat_conditional_numeric_metrics(
    group: &str,
    parent_name: &str,
    generated: &[Vec<String>],
    holdout: &[Vec<String>],
    parent: usize,
    child: usize,
    variable: &FlatVariable,
) -> Vec<FlatConditionalNumericMetric> {
    let states: BTreeSet<_> = generated
        .iter()
        .chain(holdout)
        .map(|row| row[parent].as_str())
        .collect();
    states
        .into_iter()
        .map(|state| {
            let summarize = |rows: &[Vec<String>]| {
                let selected: Vec<_> = rows.iter().filter(|row| row[parent] == state).collect();
                let mut values: Vec<f64> = selected
                    .iter()
                    .filter_map(|row| {
                        let value = row[child].as_str();
                        if flat_is_missing(value, variable)
                            || variable
                                .structural_absence_values
                                .iter()
                                .any(|v| v == value)
                        {
                            None
                        } else {
                            Some(value.parse::<f64>().expect("validated numeric child"))
                        }
                    })
                    .collect();
                values.sort_by(f64::total_cmp);
                let missing = selected
                    .iter()
                    .filter(|row| flat_is_missing(&row[child], variable))
                    .count();
                let absent = selected
                    .iter()
                    .filter(|row| variable.structural_absence_values.contains(&row[child]))
                    .count();
                (selected.len(), values, missing, absent)
            };
            let (
                generated_patients,
                generated_values,
                generated_missing,
                generated_structurally_absent,
            ) = summarize(generated);
            let (holdout_patients, holdout_values, holdout_missing, holdout_structurally_absent) =
                summarize(holdout);
            let quantile = |values: &[f64], p: f64| {
                if values.is_empty() {
                    return None;
                }
                let pos = p * (values.len() - 1) as f64;
                let lo = pos.floor() as usize;
                let hi = pos.ceil() as usize;
                let weight = pos - lo as f64;
                let value = values[lo] * (1.0 - weight) + values[hi] * weight;
                value.is_finite().then_some(value)
            };
            let absolute_quantile_differences = [("q25", 0.25), ("median", 0.5), ("q75", 0.75)]
                .into_iter()
                .map(|(name, p)| {
                    let difference = quantile(&generated_values, p)
                        .zip(quantile(&holdout_values, p))
                        .map(|(a, b)| (a - b).abs())
                        .filter(|value| value.is_finite());
                    (name.to_owned(), difference)
                })
                .collect();
            FlatConditionalNumericMetric {
                allocation: group.into(),
                parent: parent_name.into(),
                parent_value: state.into(),
                child: variable.column.clone(),
                generated_patients,
                holdout_patients,
                generated_observed: generated_values.len(),
                holdout_observed: holdout_values.len(),
                generated_missing,
                holdout_missing,
                generated_structurally_absent,
                holdout_structurally_absent,
                absolute_quantile_differences,
            }
        })
        .collect()
}

fn flat_joint_tvd(
    a: &[Vec<String>],
    b: &[Vec<String>],
    parent: usize,
    child: usize,
) -> Option<f64> {
    if a.is_empty() || b.is_empty() {
        return None;
    }
    let mut counts: BTreeMap<(&str, &str), (usize, usize)> = BTreeMap::new();
    for row in a {
        counts.entry((&row[parent], &row[child])).or_default().0 += 1;
    }
    for row in b {
        counts.entry((&row[parent], &row[child])).or_default().1 += 1;
    }
    Some(
        0.5 * counts
            .values()
            .map(|(x, y)| (*x as f64 / a.len() as f64 - *y as f64 / b.len() as f64).abs())
            .sum::<f64>(),
    )
}

fn summarize_flat_rows(
    group: &str,
    rows: &[Vec<String>],
    variables: &[FlatVariable],
) -> FlatGroupReport {
    let summaries = variables
        .iter()
        .enumerate()
        .map(|(i, v)| {
            let observed: Vec<_> = rows
                .iter()
                .map(|r| r[i].as_str())
                .filter(|x| {
                    !x.is_empty()
                        && *x != "NA"
                        && !v.structural_absence_values.iter().any(|a| a == *x)
                })
                .collect();
            let missing = rows.iter().filter(|r| flat_is_missing(&r[i], v)).count();
            let structurally_absent = rows
                .iter()
                .filter(|r| v.structural_absence_values.contains(&r[i]))
                .count();
            let mut categories = BTreeMap::new();
            let mut numbers = Vec::new();
            for value in &observed {
                if v.kind == "categorical" {
                    *categories.entry((*value).to_owned()).or_insert(0) += 1;
                } else if let Ok(n) = value.parse::<f64>() {
                    numbers.push(n);
                }
            }
            numbers.sort_by(f64::total_cmp);
            let quantile = |p: f64| {
                if numbers.is_empty() {
                    None
                } else {
                    let x = p * (numbers.len() - 1) as f64;
                    let lo = x.floor() as usize;
                    let hi = x.ceil() as usize;
                    Some(numbers[lo] + (numbers[hi] - numbers[lo]) * (x - lo as f64))
                }
            };
            let quantiles = if v.kind == "numeric" {
                [
                    ("q25", quantile(0.25)),
                    ("median", quantile(0.5)),
                    ("q75", quantile(0.75)),
                ]
                .into_iter()
                .map(|(k, x)| (k.to_owned(), x))
                .collect()
            } else {
                BTreeMap::new()
            };
            FlatVariableReport {
                variable: v.column.clone(),
                kind: v.kind.clone(),
                observed: observed.len(),
                missing,
                structurally_absent,
                category_counts: categories,
                quantiles,
            }
        })
        .collect();
    FlatGroupReport {
        group: group.into(),
        patients: rows.len(),
        variables: summaries,
    }
}

fn flat_hash(seed: u64, id: &str) -> u64 {
    let mut hash = 0xcbf29ce484222325u64;
    for byte in seed.to_le_bytes().iter().chain(id.as_bytes()) {
        hash ^= *byte as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash ^= hash >> 30;
    hash = hash.wrapping_mul(0xbf58476d1ce4e5b9);
    hash ^= hash >> 27;
    hash = hash.wrapping_mul(0x94d049bb133111eb);
    hash ^ (hash >> 31)
}

type FlatBootstrapIntervals = BTreeMap<String, Option<(f64, f64)>>;
type FlatBootstrapValidCounts = BTreeMap<String, usize>;

fn flat_bootstrap_intervals(
    development: &[Vec<String>],
    holdout: &[Vec<String>],
    index: usize,
    variable: &FlatVariable,
    replicates: usize,
    confidence: f64,
    mut state: u64,
) -> (FlatBootstrapIntervals, FlatBootstrapValidCounts) {
    let mut samples: BTreeMap<String, Vec<f64>> = ["missingness", "tvd", "q25", "median", "q75"]
        .into_iter()
        .map(|key| (key.to_owned(), Vec::new()))
        .collect();
    let mut valid = BTreeMap::new();
    for _ in 0..replicates {
        let sample = |rows: &[Vec<String>], state: &mut u64| -> Vec<usize> {
            if rows.is_empty() {
                return Vec::new();
            }
            (0..rows.len())
                .map(|_| {
                    *state = state.wrapping_add(0x9e3779b97f4a7c15);
                    let mut z = *state;
                    z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
                    z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
                    z ^= z >> 31;
                    (z as usize) % rows.len()
                })
                .collect()
        };
        let a = sample(development, &mut state);
        let b = sample(holdout, &mut state);
        let va = flat_observed_values(development, &a, index, variable);
        let vb = flat_observed_values(holdout, &b, index, variable);
        if !a.is_empty() && !b.is_empty() {
            let observed_eligible = |rows: &[Vec<String>], ids: &[usize]| {
                ids.iter()
                    .filter(|i| {
                        !variable
                            .structural_absence_values
                            .contains(&rows[**i][index])
                    })
                    .count()
            };
            let da = observed_eligible(development, &a);
            let db = observed_eligible(holdout, &b);
            if da > 0 && db > 0 {
                let ma = a
                    .iter()
                    .filter(|i| {
                        let v = &development[**i][index];
                        flat_is_missing(v, variable)
                    })
                    .count() as f64
                    / da as f64;
                let mb = b
                    .iter()
                    .filter(|i| {
                        let v = &holdout[**i][index];
                        flat_is_missing(v, variable)
                    })
                    .count() as f64
                    / db as f64;
                samples
                    .get_mut("missingness")
                    .unwrap()
                    .push((ma - mb).abs());
            }
        }
        if variable.kind == "categorical" && !va.is_empty() && !vb.is_empty() {
            let domain: BTreeSet<_> = va.iter().chain(&vb).collect();
            let distance = 0.5
                * domain
                    .iter()
                    .map(|v| {
                        (va.iter().filter(|x| *x == *v).count() as f64 / va.len() as f64
                            - vb.iter().filter(|x| *x == *v).count() as f64 / vb.len() as f64)
                            .abs()
                    })
                    .sum::<f64>();
            samples.get_mut("tvd").unwrap().push(distance);
        }
        if variable.kind == "numeric" && !va.is_empty() && !vb.is_empty() {
            let quantile = |vals: &[&str], p: f64| -> Option<f64> {
                let mut n: Vec<f64> = vals.iter().filter_map(|v| v.parse().ok()).collect();
                if n.is_empty() {
                    return None;
                }
                n.sort_by(f64::total_cmp);
                let x = p * (n.len() - 1) as f64;
                let lo = x.floor() as usize;
                let hi = x.ceil() as usize;
                Some(n[lo] + (n[hi] - n[lo]) * (x - lo as f64))
            };
            for (key, p) in [("q25", 0.25), ("median", 0.5), ("q75", 0.75)] {
                if let (Some(x), Some(y)) = (quantile(&va, p), quantile(&vb, p)) {
                    samples.get_mut(key).unwrap().push((x - y).abs());
                }
            }
        }
    }
    for (key, values) in samples.iter_mut() {
        valid.insert(key.clone(), values.len());
    }
    let intervals = samples
        .into_iter()
        .map(|(key, mut values)| {
            let result = if values.len() == replicates {
                values.sort_by(f64::total_cmp);
                let alpha = (1.0 - confidence) / 2.0;
                let q = |p: f64| {
                    let pos = p * (values.len() - 1) as f64;
                    let lo = pos.floor() as usize;
                    let hi = pos.ceil() as usize;
                    values[lo] + (values[hi] - values[lo]) * (pos - lo as f64)
                };
                Some((q(alpha), q(1.0 - alpha)))
            } else {
                None
            };
            (key, result)
        })
        .collect();
    (intervals, valid)
}

fn flat_observed_values<'a>(
    rows: &'a [Vec<String>],
    indices: &[usize],
    column: usize,
    variable: &FlatVariable,
) -> Vec<&'a str> {
    indices
        .iter()
        .map(|i| rows[*i][column].as_str())
        .filter(|value| {
            !value.is_empty()
                && *value != "NA"
                && !variable
                    .structural_absence_values
                    .iter()
                    .any(|absent| absent == *value)
        })
        .collect()
}

type ObservationFields = Option<(i64, Option<i64>, Option<String>)>;
type EpisodeObservations = BTreeMap<(String, String), ObservationFields>;

const REQUIRED: &[&str] = &[
    "patient_id",
    "episode_id",
    "source_event_id",
    "variable",
    "value",
    "missing_reason",
    "event_day",
    "recording_day",
    "provenance",
];
const OBSERVATION: &[&str] = &[
    "observation_start_day",
    "observation_end_day",
    "censoring_reason",
];

/// Validate the versioned mapping and its governance declarations.
pub fn validate_mapping(mapping: &CsvMapping) -> Result<(), Box<dyn Error>> {
    if mapping.schema_version != "1"
        || !mapping.pre_deidentified
        || !mapping.data_use_permitted
        || mapping.permitted_data_use.trim().is_empty()
    {
        return Err(
            "mapping requires version 1, pre-deidentified input and permitted data-use declaration"
                .into(),
        );
    }
    if mapping.columns.keys().any(|k| {
        !REQUIRED.contains(&k.as_str())
            && !OBSERVATION.contains(&k.as_str())
            && k != "value_present"
    }) || (OBSERVATION
        .iter()
        .any(|key| mapping.columns.contains_key(*key))
        && OBSERVATION
            .iter()
            .any(|key| !mapping.columns.contains_key(*key)))
        || REQUIRED
            .iter()
            .any(|key| !mapping.columns.contains_key(*key))
    {
        return Err(
            "mapping must specify required event columns and all observation columns together"
                .into(),
        );
    }
    let mapped: HashSet<_> = mapping.columns.values().collect();
    if mapped.len() != mapping.columns.len() || mapped.iter().any(|v| v.trim().is_empty()) {
        return Err("mapped columns must be nonempty and distinct".into());
    }
    Ok(())
}

/// Read and validate the complete input before publishing any artifacts.
pub fn read_csv(path: &Path, mapping: &CsvMapping) -> Result<Fixture, Box<dyn Error>> {
    validate_mapping(mapping)?;
    let mut reader = csv::Reader::from_path(path)?;
    let headers = reader.headers()?.clone();
    let mut seen = HashSet::new();
    for header in &headers {
        let normalized: String = header
            .chars()
            .filter(|c| c.is_ascii_alphanumeric())
            .flat_map(char::to_lowercase)
            .collect();
        if [
            "name",
            "firstname",
            "lastname",
            "fullname",
            "patientname",
            "dob",
            "dateofbirth",
            "birthdate",
            "ssn",
            "socialsecuritynumber",
            "email",
            "phone",
            "address",
            "medicalrecordnumber",
            "mrn",
            "emailaddress",
            "phonenumber",
            "telephone",
            "streetaddress",
            "address1",
            "address2",
            "nationalid",
            "passport",
            "passportnumber",
        ]
        .contains(&normalized.as_str())
        {
            return Err(format!("direct identifier column rejected: {header}").into());
        }
        if !seen.insert(header) {
            return Err(format!("duplicate CSV column: {header}").into());
        }
    }
    let mut indexes = BTreeMap::new();
    for (key, name) in &mapping.columns {
        let index = headers
            .iter()
            .position(|header| header == name)
            .ok_or_else(|| format!("missing CSV column: {name}"))?;
        indexes.insert(key.as_str(), index);
    }
    let mut events = Vec::new();
    let mut observations: EpisodeObservations = BTreeMap::new();
    for row in reader.records() {
        let row = row?;
        let get = |key: &str| row.get(indexes[key]).unwrap_or("");
        let required = |key: &str| -> Result<String, Box<dyn Error>> {
            let value = get(key);
            if value.trim().is_empty() {
                return Err(format!("empty {key}").into());
            }
            Ok(value.to_owned())
        };
        let optional = |key: &str| {
            let value = get(key);
            if value.is_empty() {
                None
            } else {
                Some(value.to_owned())
            }
        };
        if mapping.columns.contains_key("observation_start_day") {
            let start = optional("observation_start_day");
            let end = optional("observation_end_day");
            let reason = optional("censoring_reason");
            let obs = if let Some(start) = start {
                Some((start.parse()?, end.map(|v| v.parse()).transpose()?, reason))
            } else {
                if end.is_some() || reason.is_some() {
                    return Err("observation fields require start day".into());
                }
                None
            };
            let key = (required("patient_id")?, required("episode_id")?);
            if let Some(previous) = observations.insert(key, obs.clone()) {
                if previous != obs {
                    return Err("inconsistent episode observation metadata".into());
                }
            }
        }
        let value = if mapping.columns.contains_key("value_present") {
            match get("value_present") {
                "true" => Some(get("value").to_owned()),
                "false" if get("value").is_empty() => None,
                _ => {
                    return Err(
                        "value_present must be true or false and false requires an empty value"
                            .into(),
                    )
                }
            }
        } else {
            optional("value")
        };
        events.push(Event {
            patient_id: required("patient_id")?,
            episode_id: required("episode_id")?,
            id: required("source_event_id")?,
            variable: required("variable")?,
            value,
            missing: optional("missing_reason"),
            event_day: required("event_day")?.parse()?,
            recording_day: required("recording_day")?.parse()?,
            provenance: required("provenance")?,
        });
    }
    events.sort_by(|a, b| {
        (&a.patient_id, &a.episode_id, &a.id).cmp(&(&b.patient_id, &b.episode_id, &b.id))
    });
    let observations: Vec<Observation> = observations
        .into_iter()
        .filter_map(|((patient_id, episode_id), o)| {
            o.map(|(start_day, end_day, censoring_reason)| Observation {
                patient_id,
                episode_id,
                start_day,
                end_day,
                censoring_reason,
            })
        })
        .collect();
    let fixture = Fixture {
        schema_version: if observations.is_empty() { "1" } else { "2" }.into(),
        events,
        observations,
    };
    validate(&mapping.study, &fixture)?;
    validate_identifiers(&fixture)?;
    Ok(fixture)
}

fn validate_identifiers(fixture: &Fixture) -> Result<(), Box<dyn Error>> {
    for event in &fixture.events {
        if [
            &event.patient_id,
            &event.episode_id,
            &event.id,
            &event.variable,
        ]
        .iter()
        .any(|v| v.trim().is_empty())
            || event.missing.as_ref().is_some_and(|v| v.trim().is_empty())
        {
            return Err("event identifiers, variable and missing reason must be nonempty".into());
        }
    }
    Ok(())
}

/// Serialize validated events as a typed Arrow record batch in Parquet form.
/// `value_original` retains textual equality semantics for numeric spellings.
pub fn write_parquet(
    fixture: &Fixture,
    study: &Study,
    output: &Path,
) -> Result<(), Box<dyn Error>> {
    use arrow_array::{ArrayRef, BooleanArray, Float64Array, Int64Array, RecordBatch, StringArray};
    use arrow_schema::{DataType, Field, Schema};
    use parquet::arrow::ArrowWriter;
    use std::{fs::File, sync::Arc};
    validate(study, fixture)?;
    validate_identifiers(fixture)?;
    let mut events: Vec<_> = fixture.events.iter().collect();
    events.sort_by(|a, b| {
        (&a.patient_id, &a.episode_id, &a.id).cmp(&(&b.patient_id, &b.episode_id, &b.id))
    });
    let types: BTreeMap<_, _> = study
        .variables
        .iter()
        .map(|v| (v.name.as_str(), v.kind.as_str()))
        .collect();
    let strings = |f: fn(&Event) -> &str| -> ArrayRef {
        Arc::new(StringArray::from(
            events.iter().map(|e| Some(f(e))).collect::<Vec<_>>(),
        ))
    };
    let typed = |kind: &str| {
        events
            .iter()
            .map(|e| {
                if types[e.variable.as_str()] == kind {
                    e.value.as_deref()
                } else {
                    None
                }
            })
            .collect::<Vec<_>>()
    };
    let observations: BTreeMap<_, _> = fixture
        .observations
        .iter()
        .map(|o| ((o.patient_id.as_str(), o.episode_id.as_str()), o))
        .collect();
    let observation = |e: &Event| {
        observations
            .get(&(e.patient_id.as_str(), e.episode_id.as_str()))
            .copied()
    };
    let columns: Vec<ArrayRef> = vec![
        strings(|e| &e.patient_id),
        strings(|e| &e.episode_id),
        strings(|e| &e.id),
        strings(|e| &e.variable),
        Arc::new(StringArray::from(typed("string"))),
        Arc::new(Int64Array::from(
            typed("integer")
                .iter()
                .map(|v| v.map(|s| s.parse::<i64>().expect("validated integer")))
                .collect::<Vec<_>>(),
        )),
        Arc::new(Float64Array::from(
            typed("number")
                .iter()
                .map(|v| v.map(|s| s.parse::<f64>().expect("validated number")))
                .collect::<Vec<_>>(),
        )),
        Arc::new(BooleanArray::from(
            typed("boolean")
                .iter()
                .map(|v| v.map(|s| s.parse::<bool>().expect("validated boolean")))
                .collect::<Vec<_>>(),
        )),
        Arc::new(StringArray::from(
            events
                .iter()
                .map(|e| e.value.as_deref())
                .collect::<Vec<_>>(),
        )),
        Arc::new(StringArray::from(
            events
                .iter()
                .map(|e| e.missing.as_deref())
                .collect::<Vec<_>>(),
        )),
        Arc::new(Int64Array::from(
            events.iter().map(|e| e.event_day).collect::<Vec<_>>(),
        )),
        Arc::new(Int64Array::from(
            events.iter().map(|e| e.recording_day).collect::<Vec<_>>(),
        )),
        strings(|e| &e.provenance),
        Arc::new(Int64Array::from(
            events
                .iter()
                .map(|e| observation(e).map(|o| o.start_day))
                .collect::<Vec<_>>(),
        )),
        Arc::new(Int64Array::from(
            events
                .iter()
                .map(|e| observation(e).and_then(|o| o.end_day))
                .collect::<Vec<_>>(),
        )),
        Arc::new(StringArray::from(
            events
                .iter()
                .map(|e| observation(e).and_then(|o| o.censoring_reason.as_deref()))
                .collect::<Vec<_>>(),
        )),
    ];
    let schema = Arc::new(Schema::new(vec![
        Field::new("patient_id", DataType::Utf8, false),
        Field::new("episode_id", DataType::Utf8, false),
        Field::new("source_event_id", DataType::Utf8, false),
        Field::new("variable", DataType::Utf8, false),
        Field::new("value_string", DataType::Utf8, true),
        Field::new("value_integer", DataType::Int64, true),
        Field::new("value_number", DataType::Float64, true),
        Field::new("value_boolean", DataType::Boolean, true),
        Field::new("value_original", DataType::Utf8, true),
        Field::new("missing_reason", DataType::Utf8, true),
        Field::new("event_day", DataType::Int64, false),
        Field::new("recording_day", DataType::Int64, false),
        Field::new("provenance", DataType::Utf8, false),
        Field::new("observation_start_day", DataType::Int64, true),
        Field::new("observation_end_day", DataType::Int64, true),
        Field::new("censoring_reason", DataType::Utf8, true),
    ]));
    let batch = RecordBatch::try_new(schema.clone(), columns)?;
    let mut writer = ArrowWriter::try_new(File::create(output)?, schema, None)?;
    writer.write(&batch)?;
    writer.close()?;
    Ok(())
}

/// Reconstruct events from canonical Parquet, checking types and original spelling.
pub fn read_parquet(path: &Path, study: &Study) -> Result<Fixture, Box<dyn Error>> {
    use arrow_array::{Array, BooleanArray, Float64Array, Int64Array, StringArray};
    use arrow_schema::DataType;
    use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
    use std::fs::File;
    let builder = ParquetRecordBatchReaderBuilder::try_new(File::open(path)?)?;
    let expected = [
        ("patient_id", DataType::Utf8),
        ("episode_id", DataType::Utf8),
        ("source_event_id", DataType::Utf8),
        ("variable", DataType::Utf8),
        ("value_string", DataType::Utf8),
        ("value_integer", DataType::Int64),
        ("value_number", DataType::Float64),
        ("value_boolean", DataType::Boolean),
        ("value_original", DataType::Utf8),
        ("missing_reason", DataType::Utf8),
        ("event_day", DataType::Int64),
        ("recording_day", DataType::Int64),
        ("provenance", DataType::Utf8),
        ("observation_start_day", DataType::Int64),
        ("observation_end_day", DataType::Int64),
        ("censoring_reason", DataType::Utf8),
    ];
    let schema = builder.schema();
    if schema.fields().len() != expected.len()
        || schema
            .fields()
            .iter()
            .zip(&expected)
            .any(|(f, (name, ty))| f.name() != name || f.data_type() != ty)
    {
        return Err("incompatible Parquet schema".into());
    }
    let kinds: BTreeMap<_, _> = study
        .variables
        .iter()
        .map(|v| (v.name.as_str(), v.kind.as_str()))
        .collect();
    let mut events = Vec::new();
    let mut observations: EpisodeObservations = BTreeMap::new();
    for batch in builder.build()? {
        let batch = batch?;
        let s = |i: usize| {
            batch
                .column(i)
                .as_any()
                .downcast_ref::<StringArray>()
                .expect("schema checked")
        };
        let integers = batch
            .column(5)
            .as_any()
            .downcast_ref::<Int64Array>()
            .expect("schema checked");
        let numbers = batch
            .column(6)
            .as_any()
            .downcast_ref::<Float64Array>()
            .expect("schema checked");
        let booleans = batch
            .column(7)
            .as_any()
            .downcast_ref::<BooleanArray>()
            .expect("schema checked");
        let days = batch
            .column(10)
            .as_any()
            .downcast_ref::<Int64Array>()
            .expect("schema checked");
        let recording = batch
            .column(11)
            .as_any()
            .downcast_ref::<Int64Array>()
            .expect("schema checked");
        let observation_start = batch
            .column(13)
            .as_any()
            .downcast_ref::<Int64Array>()
            .expect("schema checked");
        let observation_end = batch
            .column(14)
            .as_any()
            .downcast_ref::<Int64Array>()
            .expect("schema checked");
        for row in 0..batch.num_rows() {
            if [0, 1, 2, 3, 12].iter().any(|i| s(*i).is_null(row))
                || days.is_null(row)
                || recording.is_null(row)
            {
                return Err("null required Parquet field".into());
            }
            let observation = if observation_start.is_null(row) {
                if !observation_end.is_null(row) || !s(15).is_null(row) {
                    return Err("observation fields require start day".into());
                }
                None
            } else {
                Some((
                    observation_start.value(row),
                    if observation_end.is_null(row) {
                        None
                    } else {
                        Some(observation_end.value(row))
                    },
                    if s(15).is_null(row) {
                        None
                    } else {
                        Some(s(15).value(row).to_owned())
                    },
                ))
            };
            let key = (s(0).value(row).to_owned(), s(1).value(row).to_owned());
            if let Some(previous) = observations.insert(key, observation.clone()) {
                if previous != observation {
                    return Err("inconsistent episode observation metadata".into());
                }
            }
            let variable = s(3).value(row);
            let kind = kinds.get(variable).ok_or("unknown variable in Parquet")?;
            let original = if s(8).is_null(row) {
                None
            } else {
                Some(s(8).value(row).to_owned())
            };
            let populated = [
                !s(4).is_null(row),
                !integers.is_null(row),
                !numbers.is_null(row),
                !booleans.is_null(row),
            ];
            let matches = match (*kind, original.as_deref()) {
                (_, None) => !populated.iter().any(|v| *v),
                ("string", Some(v)) => {
                    populated == [true, false, false, false] && s(4).value(row) == v
                }
                ("integer", Some(v)) => {
                    populated == [false, true, false, false]
                        && v.parse::<i64>().ok() == Some(integers.value(row))
                }
                ("number", Some(v)) => {
                    populated == [false, false, true, false]
                        && v.parse::<f64>().ok() == Some(numbers.value(row))
                }
                ("boolean", Some(v)) => {
                    populated == [false, false, false, true]
                        && v.parse::<bool>().ok() == Some(booleans.value(row))
                }
                _ => false,
            };
            if !matches {
                return Err("typed value does not match original value and study".into());
            }
            events.push(Event {
                patient_id: s(0).value(row).into(),
                episode_id: s(1).value(row).into(),
                id: s(2).value(row).into(),
                variable: variable.into(),
                value: original,
                missing: if s(9).is_null(row) {
                    None
                } else {
                    Some(s(9).value(row).into())
                },
                event_day: days.value(row),
                recording_day: recording.value(row),
                provenance: s(12).value(row).into(),
            });
        }
    }
    let observations: Vec<Observation> = observations
        .into_iter()
        .filter_map(|((patient_id, episode_id), o)| {
            o.map(|(start_day, end_day, censoring_reason)| Observation {
                patient_id,
                episode_id,
                start_day,
                end_day,
                censoring_reason,
            })
        })
        .collect();
    let fixture = Fixture {
        schema_version: if observations.is_empty() { "1" } else { "2" }.into(),
        events,
        observations,
    };
    validate(study, &fixture)?;
    validate_identifiers(&fixture)?;
    Ok(fixture)
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ColumnSchema {
    pub name: String,
    pub data_type: String,
    pub nullable: bool,
}

/// Dataset contract recorded alongside the Parquet file.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DatasetSchema {
    pub schema_version: String,
    pub study_fingerprint: String,
    pub mapping_fingerprint: String,
    pub mapping: CsvMapping,
    pub source_row_count: usize,
    pub data_fingerprint: String,
    pub columns: Vec<ColumnSchema>,
    pub adapter_version: String,
    pub creation_software: serde_yaml::Value,
}

pub fn dataset_schema(
    mapping: &CsvMapping,
    source_row_count: usize,
) -> Result<DatasetSchema, Box<dyn Error>> {
    validate_mapping(mapping)?;
    Ok(DatasetSchema {
        schema_version: "1".into(),
        study_fingerprint: crate::yaml_fingerprint(&mapping.study)?,
        mapping_fingerprint: crate::yaml_fingerprint(mapping)?,
        mapping: mapping.clone(),
        source_row_count,
        data_fingerprint: String::new(),
        columns: [
            ("patient_id", "string", false),
            ("episode_id", "string", false),
            ("source_event_id", "string", false),
            ("variable", "string", false),
            ("value_string", "string", true),
            ("value_integer", "int64", true),
            ("value_number", "float64", true),
            ("value_boolean", "boolean", true),
            ("value_original", "string", true),
            ("missing_reason", "string", true),
            ("event_day", "int64", false),
            ("recording_day", "int64", false),
            ("provenance", "string", false),
            ("observation_start_day", "int64", true),
            ("observation_end_day", "int64", true),
            ("censoring_reason", "string", true),
        ]
        .iter()
        .map(|(name, data_type, nullable)| ColumnSchema {
            name: (*name).into(),
            data_type: (*data_type).into(),
            nullable: *nullable,
        })
        .collect(),
        adapter_version: "1".into(),
        creation_software: serde_yaml::to_value(crate::software_identity()?)?,
    })
}

/// Verify the sidecar contract and reconstruct a fixture using the supplied study.
pub fn load_dataset(path: &Path, study: &Study) -> Result<Fixture, Box<dyn Error>> {
    let schema_path = path.parent().unwrap_or(Path::new(".")).join("schema.yaml");
    let schema: DatasetSchema = serde_yaml::from_str(&std::fs::read_to_string(schema_path)?)?;
    validate_mapping(&schema.mapping)?;
    if schema.schema_version != "1"
        || schema.adapter_version != "1"
        || schema.study_fingerprint != crate::yaml_fingerprint(study)?
        || schema.study_fingerprint != crate::yaml_fingerprint(&schema.mapping.study)?
        || schema.mapping_fingerprint != crate::yaml_fingerprint(&schema.mapping)?
        || schema.columns != dataset_schema(&schema.mapping, schema.source_row_count)?.columns
        || !schema.mapping.pre_deidentified
        || !schema.mapping.data_use_permitted
        || schema.mapping.permitted_data_use.trim().is_empty()
    {
        return Err("dataset schema or fingerprints do not match".into());
    }
    if schema.data_fingerprint != data_fingerprint(path)? {
        return Err("Parquet fingerprint does not match schema".into());
    }
    let fixture = read_parquet(path, study)?;
    if fixture.events.len() != schema.source_row_count {
        return Err("dataset row count does not match schema".into());
    }
    Ok(fixture)
}

#[derive(Debug, Serialize)]
pub struct VariableSummary {
    pub variable: String,
    pub observed: i64,
    pub missing: i64,
    pub absent_episodes: i64,
    pub mean: Option<f64>,
    pub standard_deviation: Option<f64>,
    pub minimum: Option<f64>,
    pub lower_quartile: Option<f64>,
    pub median: Option<f64>,
    pub upper_quartile: Option<f64>,
    pub maximum: Option<f64>,
}
#[derive(Debug, Serialize)]
pub struct Frequency {
    pub variable: String,
    pub value: String,
    pub count: i64,
}
#[derive(Debug, Serialize)]
pub struct TimingSummary {
    pub event_minimum: Option<i64>,
    pub event_median: Option<f64>,
    pub event_maximum: Option<i64>,
    pub recording_minimum: Option<i64>,
    pub recording_median: Option<f64>,
    pub recording_maximum: Option<i64>,
    pub recorded_before_event: i64,
    pub duplicate_event_ids: i64,
}
#[derive(Debug, Serialize)]
pub struct Inspection {
    pub schema_version: String,
    pub source_rows: i64,
    pub patients: i64,
    pub episodes: i64,
    pub variables: Vec<VariableSummary>,
    pub categorical_frequencies: Vec<Frequency>,
    pub missing_reasons: Vec<Frequency>,
    pub timing: TimingSummary,
    pub limitations: Vec<String>,
}

/// Run fixed, parameterized aggregate queries over a canonical Parquet file.
pub fn inspect(path: &Path, study: &Study) -> Result<Inspection, Box<dyn Error>> {
    read_parquet(path, study)?;
    let connection = duckdb::Connection::open_in_memory()?;
    let path = path.to_str().ok_or("Parquet path is not UTF-8")?;
    let (source_rows, patients, episodes): (i64, i64, i64) = connection.query_row(
        "SELECT count(*), count(DISTINCT patient_id), count(DISTINCT (patient_id, episode_id)) FROM read_parquet(?)", [path], |row| Ok((row.get(0)?,row.get(1)?,row.get(2)?)))?;
    let mut statement = connection.prepare("SELECT variable, count(*) FILTER (WHERE missing_reason IS NULL), count(*) FILTER (WHERE missing_reason IS NOT NULL), avg(coalesce(value_number, value_integer::DOUBLE)), stddev_samp(coalesce(value_number, value_integer::DOUBLE)), min(coalesce(value_number, value_integer::DOUBLE)), quantile_cont(coalesce(value_number, value_integer::DOUBLE), 0.25), quantile_cont(coalesce(value_number, value_integer::DOUBLE), 0.5), quantile_cont(coalesce(value_number, value_integer::DOUBLE), 0.75), max(coalesce(value_number, value_integer::DOUBLE)) FROM read_parquet(?) GROUP BY variable ORDER BY variable")?;
    let mut variables = statement
        .query_map([path], |row| {
            Ok(VariableSummary {
                variable: row.get(0)?,
                observed: row.get(1)?,
                missing: row.get(2)?,
                absent_episodes: 0,
                mean: row.get(3)?,
                standard_deviation: row.get(4)?,
                minimum: row.get(5)?,
                lower_quartile: row.get(6)?,
                median: row.get(7)?,
                upper_quartile: row.get(8)?,
                maximum: row.get(9)?,
            })
        })?
        .collect::<Result<Vec<_>, _>>()?;
    for variable in &study.variables {
        let absent: i64 = connection.query_row("SELECT count(*) FROM (SELECT DISTINCT patient_id, episode_id FROM read_parquet(?)) e WHERE NOT EXISTS (SELECT 1 FROM read_parquet(?) r WHERE r.patient_id = e.patient_id AND r.episode_id = e.episode_id AND r.variable = ?)", duckdb::params![path, path, variable.name], |row| row.get(0))?;
        if let Some(summary) = variables.iter_mut().find(|v| v.variable == variable.name) {
            summary.absent_episodes = absent;
        } else {
            variables.push(VariableSummary {
                variable: variable.name.clone(),
                observed: 0,
                missing: 0,
                absent_episodes: absent,
                mean: None,
                standard_deviation: None,
                minimum: None,
                lower_quartile: None,
                median: None,
                upper_quartile: None,
                maximum: None,
            });
        }
    }
    variables.sort_by(|a, b| a.variable.cmp(&b.variable));
    let mut category_query = connection.prepare("SELECT variable, coalesce(value_string, value_boolean::VARCHAR), count(*) FROM read_parquet(?) WHERE value_string IS NOT NULL OR value_boolean IS NOT NULL GROUP BY 1,2 ORDER BY 1,2 LIMIT 1000")?;
    let categorical_frequencies = category_query
        .query_map([path], |row| {
            Ok(Frequency {
                variable: row.get(0)?,
                value: row.get(1)?,
                count: row.get(2)?,
            })
        })?
        .collect::<Result<Vec<_>, _>>()?;
    let mut missing_query = connection.prepare("SELECT variable, missing_reason, count(*) FROM read_parquet(?) WHERE missing_reason IS NOT NULL GROUP BY 1,2 ORDER BY 1,2 LIMIT 1000")?;
    let missing_reasons = missing_query
        .query_map([path], |row| {
            Ok(Frequency {
                variable: row.get(0)?,
                value: row.get(1)?,
                count: row.get(2)?,
            })
        })?
        .collect::<Result<Vec<_>, _>>()?;
    let timing = connection.query_row("SELECT min(event_day), quantile_cont(event_day, 0.5), max(event_day), min(recording_day), quantile_cont(recording_day, 0.5), max(recording_day), count(*) FILTER (WHERE recording_day < event_day), count(*) - count(DISTINCT source_event_id) FROM read_parquet(?)", [path], |row| Ok(TimingSummary { event_minimum: row.get(0)?, event_median: row.get(1)?, event_maximum: row.get(2)?, recording_minimum: row.get(3)?, recording_median: row.get(4)?, recording_maximum: row.get(5)?, recorded_before_event: row.get(6)?, duplicate_event_ids: row.get(7)? }))?;
    Ok(Inspection { schema_version: "1".into(), source_rows, patients, episodes, variables, categorical_frequencies, missing_reasons, timing, limitations: vec!["Descriptive source-event summaries do not establish clinical validity, privacy, causality, or external validation.".into(), "Observed and missing counts are source events; absent counts are distinct patient-episode pairs with no source event and do not establish clinical absence or complete observation.".into(), "Frequency tables are limited to the first 1000 sorted combinations.".into(), "Numeric summaries convert integers to floating point and can lose precision.".into()] })
}

/// Export canonical events in stable order using the version 1 CSV headers.
pub fn export_csv(fixture: &Fixture, output: &Path) -> Result<(), Box<dyn Error>> {
    let file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(output)?;
    let mut writer = csv::Writer::from_writer(file);
    writer.write_record(
        REQUIRED
            .iter()
            .chain(OBSERVATION.iter())
            .chain(std::iter::once(&"value_present")),
    )?;
    let observations: BTreeMap<_, _> = fixture
        .observations
        .iter()
        .map(|o| ((o.patient_id.as_str(), o.episode_id.as_str()), o))
        .collect();
    let mut events: Vec<_> = fixture.events.iter().collect();
    events.sort_by(|a, b| {
        (&a.patient_id, &a.episode_id, &a.id).cmp(&(&b.patient_id, &b.episode_id, &b.id))
    });
    for e in events {
        let observation = observations.get(&(e.patient_id.as_str(), e.episode_id.as_str()));
        writer.write_record([
            e.patient_id.as_str(),
            &e.episode_id,
            &e.id,
            &e.variable,
            e.value.as_deref().unwrap_or(""),
            e.missing.as_deref().unwrap_or(""),
            &e.event_day.to_string(),
            &e.recording_day.to_string(),
            &e.provenance,
            &observation
                .map(|o| o.start_day.to_string())
                .unwrap_or_default(),
            &observation
                .and_then(|o| o.end_day)
                .map(|v| v.to_string())
                .unwrap_or_default(),
            observation
                .and_then(|o| o.censoring_reason.as_deref())
                .unwrap_or(""),
            if e.value.is_some() { "true" } else { "false" },
        ])?;
    }
    writer.flush()?;
    Ok(())
}

pub fn read_dataset_schema(path: &Path) -> Result<DatasetSchema, Box<dyn Error>> {
    let schema_path = path.parent().unwrap_or(Path::new(".")).join("schema.yaml");
    Ok(serde_yaml::from_str(&std::fs::read_to_string(
        schema_path,
    )?)?)
}

/// Identity fingerprint, not a cryptographic integrity or privacy guarantee.
pub fn data_fingerprint(path: &Path) -> Result<String, Box<dyn Error>> {
    use std::io::Read;
    let mut file = std::fs::File::open(path)?;
    let mut hash = 0xcbf29ce484222325u64;
    let mut buffer = [0; 65536];
    loop {
        let n = file.read(&mut buffer)?;
        if n == 0 {
            break;
        }
        for byte in &buffer[..n] {
            hash ^= u64::from(*byte);
            hash = hash.wrapping_mul(0x100000001b3);
        }
    }
    Ok(format!("fnv1a64-bytes:{hash:016x}"))
}

/// Export using the source mapping's column names so the same mapping can replay it.
pub fn export_mapped_csv(
    fixture: &Fixture,
    mapping: &CsvMapping,
    output: &Path,
) -> Result<(), Box<dyn Error>> {
    validate_mapping(mapping)?;
    validate(&mapping.study, fixture)?;
    validate_identifiers(fixture)?;
    if !fixture.observations.is_empty() && !mapping.columns.contains_key("observation_start_day") {
        return export_csv(fixture, output);
    }
    if fixture
        .events
        .iter()
        .any(|e| e.value.as_deref() == Some(""))
        && !mapping.columns.contains_key("value_present")
    {
        return export_csv(fixture, output);
    }
    let mut keys = REQUIRED.to_vec();
    if mapping.columns.contains_key("observation_start_day") {
        keys.extend_from_slice(OBSERVATION);
    }
    if mapping.columns.contains_key("value_present") {
        keys.push("value_present");
    }
    let file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(output)?;
    let mut writer = csv::Writer::from_writer(file);
    writer.write_record(keys.iter().map(|k| mapping.columns[*k].as_str()))?;
    let observations: BTreeMap<_, _> = fixture
        .observations
        .iter()
        .map(|o| ((o.patient_id.as_str(), o.episode_id.as_str()), o))
        .collect();
    let mut events: Vec<_> = fixture.events.iter().collect();
    events.sort_by(|a, b| {
        (&a.patient_id, &a.episode_id, &a.id).cmp(&(&b.patient_id, &b.episode_id, &b.id))
    });
    for e in events {
        let obs = observations.get(&(e.patient_id.as_str(), e.episode_id.as_str()));
        let mut row = vec![
            e.patient_id.clone(),
            e.episode_id.clone(),
            e.id.clone(),
            e.variable.clone(),
            e.value.clone().unwrap_or_default(),
            e.missing.clone().unwrap_or_default(),
            e.event_day.to_string(),
            e.recording_day.to_string(),
            e.provenance.clone(),
        ];
        if mapping.columns.contains_key("observation_start_day") {
            row.push(obs.map(|o| o.start_day.to_string()).unwrap_or_default());
            row.push(
                obs.and_then(|o| o.end_day)
                    .map(|v| v.to_string())
                    .unwrap_or_default(),
            );
            row.push(
                obs.and_then(|o| o.censoring_reason.clone())
                    .unwrap_or_default(),
            );
        }
        if mapping.columns.contains_key("value_present") {
            row.push(e.value.is_some().to_string());
        }
        writer.write_record(row)?;
    }
    writer.flush()?;
    Ok(())
}

/// Numeric feature matrix extracted from a CSV or Parquet dataset.
/// Missing or NaN cells are preserved explicitly as `None` (never imputed to `0.0`).
#[derive(Debug, Clone)]
pub struct NumericTable {
    pub columns: Vec<String>,
    pub data: ndarray::Array2<Option<f64>>,
}

impl NumericTable {
    pub fn nrows(&self) -> usize {
        self.data.nrows()
    }

    pub fn ncols(&self) -> usize {
        self.data.ncols()
    }

    /// Return observed finite `Some(v)` values in column `col_idx` without imputation.
    pub fn observed_column(&self, col_idx: usize) -> Vec<f64> {
        self.data
            .column(col_idx)
            .iter()
            .filter_map(|cell| cell.filter(|v| v.is_finite()))
            .collect()
    }

    /// Count missing (`None` or non-finite) cells in column `col_idx`.
    pub fn missing_count(&self, col_idx: usize) -> usize {
        self.data
            .column(col_idx)
            .iter()
            .filter(|cell| !matches!(cell, Some(v) if v.is_finite()))
            .count()
    }

    /// Standardize numeric columns using observed `real` statistics and append explicit
    /// `is_missing` indicator columns whenever either `real` or `synthetic` has missing values.
    pub fn build_standardized_features_with_indicators(
        real: &NumericTable,
        synthetic: &NumericTable,
    ) -> Result<(ndarray::Array2<f64>, ndarray::Array2<f64>), Box<dyn Error>> {
        if real.columns.is_empty() || real.columns != synthetic.columns {
            return Err(crate::Error::Input(
                "standardized feature construction requires non-empty matching columns".into(),
            )
            .into());
        }
        if real.data.ncols() != real.columns.len()
            || synthetic.data.ncols() != synthetic.columns.len()
        {
            return Err(crate::Error::Input(
                "NumericTable column metadata does not match matrix width".into(),
            )
            .into());
        }
        if real.nrows() == 0 || synthetic.nrows() == 0 {
            return Err(crate::Error::Input(
                "standardized feature construction requires non-empty real and synthetic tables"
                    .into(),
            )
            .into());
        }

        let n_cols = real.ncols();
        let mut means = Vec::with_capacity(n_cols);
        let mut stds = Vec::with_capacity(n_cols);
        let mut missing_indicator_cols = Vec::new();

        for j in 0..n_cols {
            let observed = real.observed_column(j);
            let (mean, std) = if observed.is_empty() {
                (0.0, 1.0)
            } else {
                let n = observed.len() as f64;
                let mean = observed.iter().sum::<f64>() / n;
                let var = observed.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n;
                let s = var.sqrt();
                (mean, if s > 1e-9 { s } else { 1.0 })
            };
            means.push(mean);
            stds.push(std);

            if real.missing_count(j) > 0 || synthetic.missing_count(j) > 0 {
                missing_indicator_cols.push(j);
            }
        }

        let out_ncols = n_cols + missing_indicator_cols.len();
        let encode = |table: &NumericTable| -> Result<ndarray::Array2<f64>, Box<dyn Error>> {
            let nrows = table.nrows();
            let mut flat = Vec::with_capacity(nrows * out_ncols);
            for i in 0..nrows {
                for j in 0..n_cols {
                    let standardized = match table.data[[i, j]] {
                        Some(v) if v.is_finite() => (v - means[j]) / stds[j],
                        _ => 0.0,
                    };
                    flat.push(standardized);
                }
                for &j in &missing_indicator_cols {
                    let is_missing = match table.data[[i, j]] {
                        Some(v) if v.is_finite() => 0.0,
                        _ => 1.0,
                    };
                    flat.push(is_missing);
                }
            }
            Ok(ndarray::Array2::from_shape_vec((nrows, out_ncols), flat)?)
        };

        Ok((encode(real)?, encode(synthetic)?))
    }
}

/// Load numeric columns from a CSV or Parquet file into an `Array2<Option<f64>>`.
pub fn load_numeric_table(path: &Path) -> Result<NumericTable, Box<dyn Error>> {
    if !path.exists() {
        return Err(format!("dataset file not found: {}", path.display()).into());
    }
    if path
        .extension()
        .is_some_and(|ext| ext.eq_ignore_ascii_case("parquet"))
    {
        load_numeric_parquet(path)
    } else {
        load_numeric_csv(path)
    }
}

/// Load all columns (numeric and categorical/string/boolean) from a CSV or Parquet file into
/// an in-memory [`crate::forensic_workflow::ForensicCohort`] while preserving missing values as
/// `None` (never `0.0`).
pub fn load_forensic_cohort(
    path: &Path,
    cohort_id: &str,
) -> Result<crate::forensic_workflow::ForensicCohort, Box<dyn Error>> {
    if cohort_id.trim().is_empty() {
        return Err(crate::Error::Input("forensic cohort_id must not be empty".into()).into());
    }
    if !path.exists() {
        return Err(format!("dataset file not found: {}", path.display()).into());
    }
    if path
        .extension()
        .is_some_and(|ext| ext.eq_ignore_ascii_case("parquet"))
    {
        load_forensic_parquet(path, cohort_id)
    } else {
        load_forensic_csv(path, cohort_id)
    }
}

fn is_missing_cell(raw: &str) -> bool {
    let trimmed = raw.trim();
    trimmed.is_empty() || trimmed.eq_ignore_ascii_case("na") || trimmed.eq_ignore_ascii_case("nan")
}

fn is_numeric_arrow_type(dt: &arrow_schema::DataType) -> bool {
    use arrow_schema::DataType;
    matches!(
        dt,
        DataType::Int8
            | DataType::Int16
            | DataType::Int32
            | DataType::Int64
            | DataType::UInt8
            | DataType::UInt16
            | DataType::UInt32
            | DataType::UInt64
            | DataType::Float32
            | DataType::Float64
    )
}

fn append_parquet_numeric_column(
    arr: &dyn arrow_array::Array,
    col_name: &str,
    path: &Path,
    dst: &mut Vec<Option<f64>>,
) -> Result<(), Box<dyn Error>> {
    use arrow_array::cast::AsArray;
    use arrow_array::types::{
        Float32Type, Float64Type, Int16Type, Int32Type, Int64Type, Int8Type, UInt16Type,
        UInt32Type, UInt64Type, UInt8Type,
    };
    use arrow_array::Array;
    use arrow_schema::DataType;

    macro_rules! push_ints {
        ($ty:ty) => {{
            let a = arr.as_primitive::<$ty>();
            for i in 0..arr.len() {
                dst.push(if a.is_null(i) {
                    None
                } else {
                    Some(a.value(i) as f64)
                });
            }
        }};
    }

    match arr.data_type() {
        DataType::Int8 => push_ints!(Int8Type),
        DataType::Int16 => push_ints!(Int16Type),
        DataType::Int32 => push_ints!(Int32Type),
        DataType::Int64 => push_ints!(Int64Type),
        DataType::UInt8 => push_ints!(UInt8Type),
        DataType::UInt16 => push_ints!(UInt16Type),
        DataType::UInt32 => push_ints!(UInt32Type),
        DataType::UInt64 => push_ints!(UInt64Type),
        DataType::Float32 => {
            let a = arr.as_primitive::<Float32Type>();
            for i in 0..arr.len() {
                if a.is_null(i) {
                    dst.push(None);
                } else {
                    let v = a.value(i) as f64;
                    if v.is_nan() {
                        dst.push(None);
                    } else if v.is_infinite() {
                        return Err(format!(
                            "infinite float value in Parquet column '{}' of {}",
                            col_name,
                            path.display()
                        )
                        .into());
                    } else {
                        dst.push(Some(v));
                    }
                }
            }
        }
        DataType::Float64 => {
            let a = arr.as_primitive::<Float64Type>();
            for i in 0..arr.len() {
                if a.is_null(i) {
                    dst.push(None);
                } else {
                    let v = a.value(i);
                    if v.is_nan() {
                        dst.push(None);
                    } else if v.is_infinite() {
                        return Err(format!(
                            "infinite float value in Parquet column '{}' of {}",
                            col_name,
                            path.display()
                        )
                        .into());
                    } else {
                        dst.push(Some(v));
                    }
                }
            }
        }
        other => {
            return Err(format!(
                "unsupported numeric Arrow type {:?} for column '{}' in {}",
                other,
                col_name,
                path.display()
            )
            .into());
        }
    }
    Ok(())
}

fn load_numeric_csv(path: &Path) -> Result<NumericTable, Box<dyn Error>> {
    let mut reader = csv::Reader::from_path(path)?;
    let headers: Vec<String> = reader
        .headers()?
        .iter()
        .map(|h| h.trim().to_string())
        .collect();
    if headers.is_empty() {
        return Err(format!("dataset has no columns: {}", path.display()).into());
    }
    let mut rows: Vec<Vec<String>> = Vec::new();
    for record in reader.records() {
        let record = record?;
        rows.push(record.iter().map(|v| v.to_string()).collect());
    }
    if rows.is_empty() {
        return Err(format!("dataset is empty (0 rows): {}", path.display()).into());
    }
    let nrows = rows.len();
    let mut numeric_cols = Vec::new();
    let mut col_values: Vec<Vec<Option<f64>>> = Vec::new();

    for (col_idx, header) in headers.iter().enumerate() {
        let mut parsed = Vec::with_capacity(nrows);
        let mut all_numeric = true;
        let mut has_non_missing = false;
        for row in &rows {
            let raw = row.get(col_idx).map(String::as_str).unwrap_or("");
            if is_missing_cell(raw) {
                parsed.push(None);
            } else if let Ok(v) = raw.trim().parse::<f64>() {
                if v.is_finite() {
                    has_non_missing = true;
                    parsed.push(Some(v));
                } else {
                    all_numeric = false;
                    break;
                }
            } else {
                all_numeric = false;
                break;
            }
        }
        if all_numeric && has_non_missing {
            numeric_cols.push(header.clone());
            col_values.push(parsed);
        }
    }

    if numeric_cols.is_empty() {
        return Err(format!("no numeric columns found in {}", path.display()).into());
    }
    let mut data = ndarray::Array2::<Option<f64>>::from_elem((nrows, numeric_cols.len()), None);
    for (j, col) in col_values.iter().enumerate() {
        for (i, &val) in col.iter().enumerate() {
            data[[i, j]] = val;
        }
    }
    Ok(NumericTable {
        columns: numeric_cols,
        data,
    })
}

fn load_numeric_parquet(path: &Path) -> Result<NumericTable, Box<dyn Error>> {
    use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

    let file = std::fs::File::open(path)?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)?;
    let schema = builder.schema().clone();
    let reader = builder.build()?;

    let numeric_indices: Vec<(usize, String)> = schema
        .fields()
        .iter()
        .enumerate()
        .filter(|(_, f)| is_numeric_arrow_type(f.data_type()))
        .map(|(idx, f)| (idx, f.name().clone()))
        .collect();

    if numeric_indices.is_empty() {
        return Err(format!("no numeric columns found in {}", path.display()).into());
    }

    let mut col_values: Vec<Vec<Option<f64>>> = vec![Vec::new(); numeric_indices.len()];
    let mut total_rows = 0usize;

    for batch in reader {
        let batch = batch?;
        total_rows += batch.num_rows();
        for (out_j, (col_idx, col_name)) in numeric_indices.iter().enumerate() {
            let arr = batch.column(*col_idx);
            append_parquet_numeric_column(arr.as_ref(), col_name, path, &mut col_values[out_j])?;
        }
    }

    if total_rows == 0 {
        return Err(format!("dataset is empty (0 rows): {}", path.display()).into());
    }

    let columns: Vec<String> = numeric_indices.into_iter().map(|(_, name)| name).collect();
    let mut data = ndarray::Array2::<Option<f64>>::from_elem((total_rows, columns.len()), None);
    for (j, col) in col_values.iter().enumerate() {
        for (i, &val) in col.iter().enumerate() {
            data[[i, j]] = val;
        }
    }
    Ok(NumericTable { columns, data })
}

fn load_forensic_csv(
    path: &Path,
    cohort_id: &str,
) -> Result<crate::forensic_workflow::ForensicCohort, Box<dyn Error>> {
    use crate::forensic_workflow::{ForensicCohort, ForensicRow};

    let mut reader = csv::Reader::from_path(path)?;
    let headers: Vec<String> = reader
        .headers()?
        .iter()
        .map(|h| h.trim().to_string())
        .collect();
    if headers.is_empty() || headers.iter().any(|h| h.is_empty()) {
        return Err(format!("dataset has missing or empty column headers: {}", path.display()).into());
    }
    if headers.iter().collect::<HashSet<_>>().len() != headers.len() {
        return Err(format!("dataset contains duplicate column headers: {}", path.display()).into());
    }

    let mut raw_rows: Vec<Vec<String>> = Vec::new();
    for record in reader.records() {
        let record = record?;
        raw_rows.push(record.iter().map(|v| v.to_string()).collect());
    }
    if raw_rows.is_empty() {
        return Err(format!("dataset is empty (0 rows): {}", path.display()).into());
    }

    let nrows = raw_rows.len();
    enum ParsedColumn {
        Numeric(String, Vec<Option<f64>>),
        Categorical(String, Vec<Option<String>>),
    }

    let mut parsed_cols = Vec::with_capacity(headers.len());
    for (col_idx, header) in headers.iter().enumerate() {
        let mut num_vals = Vec::with_capacity(nrows);
        let mut all_numeric = true;
        let mut has_non_missing = false;
        for row in &raw_rows {
            let raw = row.get(col_idx).map(String::as_str).unwrap_or("");
            if is_missing_cell(raw) {
                num_vals.push(None);
            } else if let Ok(v) = raw.trim().parse::<f64>() {
                if v.is_finite() {
                    has_non_missing = true;
                    num_vals.push(Some(v));
                } else {
                    all_numeric = false;
                    break;
                }
            } else {
                all_numeric = false;
                break;
            }
        }
        if all_numeric && has_non_missing {
            parsed_cols.push(ParsedColumn::Numeric(header.clone(), num_vals));
        } else {
            let cat_vals = raw_rows
                .iter()
                .map(|row| {
                    let raw = row.get(col_idx).map(String::as_str).unwrap_or("");
                    let trimmed = raw.trim();
                    if crate::candidate::token_is_missing(trimmed) {
                        None
                    } else {
                        Some(trimmed.to_string())
                    }
                })
                .collect();
            parsed_cols.push(ParsedColumn::Categorical(header.clone(), cat_vals));
        }
    }

    let mut rows = Vec::with_capacity(nrows);
    for idx in 0..nrows {
        let mut numeric_fields = BTreeMap::new();
        let mut categorical_fields = BTreeMap::new();
        for col in &parsed_cols {
            match col {
                ParsedColumn::Numeric(name, vals) => {
                    numeric_fields.insert(name.clone(), vals[idx]);
                }
                ParsedColumn::Categorical(name, vals) => {
                    categorical_fields.insert(name.clone(), vals[idx].clone());
                }
            }
        }
        rows.push(ForensicRow {
            patient_key: format!("{cohort_id}-row-{idx:06}"),
            numeric_fields,
            categorical_fields,
        });
    }

    let cohort = ForensicCohort::new(cohort_id, rows);
    cohort.validate_keys()?;
    Ok(cohort)
}

fn load_forensic_parquet(
    path: &Path,
    cohort_id: &str,
) -> Result<crate::forensic_workflow::ForensicCohort, Box<dyn Error>> {
    use crate::forensic_workflow::{ForensicCohort, ForensicRow};
    use arrow_array::{Array, BooleanArray, LargeStringArray, StringArray};
    use arrow_schema::DataType;
    use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

    let file = std::fs::File::open(path)?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)?;
    let schema = builder.schema().clone();
    let fields = schema.fields();
    if fields.is_empty() {
        return Err(format!("dataset has no columns: {}", path.display()).into());
    }
    if fields
        .iter()
        .map(|f| f.name().as_str())
        .collect::<HashSet<_>>()
        .len()
        != fields.len()
    {
        return Err(format!("dataset contains duplicate column names: {}", path.display()).into());
    }

    enum ColumnStore {
        Numeric(String, Vec<Option<f64>>),
        Categorical(String, Vec<Option<String>>),
    }

    let mut stores: Vec<ColumnStore> = fields
        .iter()
        .map(|f| {
            let name = f.name().clone();
            if is_numeric_arrow_type(f.data_type()) {
                Ok(ColumnStore::Numeric(name, Vec::new()))
            } else if matches!(
                f.data_type(),
                DataType::Utf8 | DataType::LargeUtf8 | DataType::Boolean | DataType::Null
            ) {
                Ok(ColumnStore::Categorical(name, Vec::new()))
            } else {
                Err(format!(
                    "unsupported Parquet data type {:?} for column '{}' in {}",
                    f.data_type(),
                    name,
                    path.display()
                )
                .into())
            }
        })
        .collect::<Result<Vec<_>, Box<dyn Error>>>()?;

    let reader = builder.build()?;
    let mut total_rows = 0usize;

    for batch in reader {
        let batch = batch?;
        let n = batch.num_rows();
        total_rows += n;
        for (col_idx, store) in stores.iter_mut().enumerate() {
            let arr = batch.column(col_idx);
            match store {
                ColumnStore::Numeric(col_name, dst) => {
                    append_parquet_numeric_column(arr.as_ref(), col_name, path, dst)?;
                }
                ColumnStore::Categorical(col_name, dst) => match arr.data_type() {
                    DataType::Utf8 => {
                        let a = arr
                            .as_any()
                            .downcast_ref::<StringArray>()
                            .ok_or_else(|| format!("expected Utf8 array for '{}'", col_name))?;
                        for i in 0..n {
                            if a.is_null(i) {
                                dst.push(None);
                            } else {
                                let trimmed = a.value(i).trim();
                                if crate::candidate::token_is_missing(trimmed) {
                                    dst.push(None);
                                } else {
                                    dst.push(Some(trimmed.to_string()));
                                }
                            }
                        }
                    }
                    DataType::LargeUtf8 => {
                        let a = arr
                            .as_any()
                            .downcast_ref::<LargeStringArray>()
                            .ok_or_else(|| {
                                format!("expected LargeUtf8 array for '{}'", col_name)
                            })?;
                        for i in 0..n {
                            if a.is_null(i) {
                                dst.push(None);
                            } else {
                                let trimmed = a.value(i).trim();
                                if crate::candidate::token_is_missing(trimmed) {
                                    dst.push(None);
                                } else {
                                    dst.push(Some(trimmed.to_string()));
                                }
                            }
                        }
                    }
                    DataType::Boolean => {
                        let a = arr
                            .as_any()
                            .downcast_ref::<BooleanArray>()
                            .ok_or_else(|| format!("expected Boolean array for '{}'", col_name))?;
                        for i in 0..n {
                            dst.push(if a.is_null(i) {
                                None
                            } else {
                                Some(a.value(i).to_string())
                            });
                        }
                    }
                    DataType::Null => {
                        dst.extend(std::iter::repeat_n(None, n));
                    }
                    other => {
                        return Err(format!(
                            "unsupported categorical Arrow type {:?} for column '{}' in {}",
                            other,
                            col_name,
                            path.display()
                        )
                        .into());
                    }
                },
            }
        }
    }

    if total_rows == 0 {
        return Err(format!("dataset is empty (0 rows): {}", path.display()).into());
    }

    let mut rows = Vec::with_capacity(total_rows);
    for idx in 0..total_rows {
        let mut numeric_fields = BTreeMap::new();
        let mut categorical_fields = BTreeMap::new();
        for store in &stores {
            match store {
                ColumnStore::Numeric(name, vals) => {
                    numeric_fields.insert(name.clone(), vals[idx]);
                }
                ColumnStore::Categorical(name, vals) => {
                    categorical_fields.insert(name.clone(), vals[idx].clone());
                }
            }
        }
        rows.push(ForensicRow {
            patient_key: format!("{cohort_id}-row-{idx:06}"),
            numeric_fields,
            categorical_fields,
        });
    }

    let cohort = ForensicCohort::new(cohort_id, rows);
    cohort.validate_keys()?;
    Ok(cohort)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bounded_index_sampling_stays_in_range_and_replays() {
        let mut first = 0x1234_5678_9abc_def0;
        let mut replay = first;
        let values: Vec<_> = (0..10_000)
            .map(|_| flat_sample_index(&mut first, 7))
            .collect();
        let replayed: Vec<_> = (0..10_000)
            .map(|_| flat_sample_index(&mut replay, 7))
            .collect();
        assert_eq!(values, replayed);
        assert!(values.iter().all(|&index| index < 7));

        let mut state = 0;
        let before = state;
        assert_eq!(flat_sample_index(&mut state, 1), 0);
        assert_ne!(state, before);
    }

    #[test]
    fn ordered_forest_learns_association_without_inventing_independent_edge() {
        let rows: Vec<Vec<String>> = [
            ("a", "x", "0"),
            ("a", "x", "1"),
            ("b", "y", "0"),
            ("b", "y", "1"),
        ]
        .into_iter()
        .map(|(a, b, c)| vec![a.into(), b.into(), c.into(), "10".into()])
        .collect();
        let refs: Vec<_> = rows.iter().collect();
        let variables: Vec<FlatVariable> = serde_yaml::from_str("- {column: a, kind: categorical, allowed_values: [a, b]}\n- {column: b, kind: categorical, allowed_values: [x, y]}\n- {column: c, kind: categorical, allowed_values: ['0', '1']}\n- {column: number, kind: numeric}\n").unwrap();
        let (parents, edges) = fit_flat_ordered_forest(&refs, &variables, "g", false);
        assert_eq!(parents, vec![vec![], vec![0], vec![], vec![]]);
        assert_eq!(edges.len(), 1);
        assert_eq!(
            (&edges[0].parent, &edges[0].child),
            (&"a".to_owned(), &"b".to_owned())
        );
        assert!((edges[0].mutual_information_nats - std::f64::consts::LN_2).abs() < 1e-12);
        let mut seed = 9;
        let mut counts = vec![BTreeMap::new(); 4];
        for _ in 0..50 {
            let row = sample_flat_conditionals(
                &refs,
                &parents,
                &mut seed,
                FlatSparsePolicy::Fail,
                &mut counts,
            )
            .unwrap();
            assert!((row[0] == "a" && row[1] == "x") || (row[0] == "b" && row[1] == "y"));
        }
    }

    #[test]
    fn penalized_forest_rejects_high_cardinality_overfit_and_retains_signal() {
        let variables: Vec<FlatVariable> = serde_yaml::from_str("- {column: unique, kind: categorical, allowed_values: [unused]}\n- {column: signal, kind: categorical, allowed_values: [a, b]}\n- {column: outcome, kind: categorical, allowed_values: [x, y]}\n").unwrap();
        let rows: Vec<Vec<String>> = (0..200)
            .map(|i| {
                vec![
                    format!("id{i}"),
                    if i % 2 == 0 { "a".into() } else { "b".into() },
                    if i % 2 == 0 { "x".into() } else { "y".into() },
                ]
            })
            .collect();
        let refs: Vec<_> = rows.iter().collect();
        let (parents, edges) = fit_flat_penalized_forest(&refs, &variables, "g");
        assert_eq!(parents, vec![vec![], vec![], vec![1]]);
        assert_eq!(edges.len(), 1);
        let gain = std::f64::consts::LN_2 - (200.0_f64.ln() / 400.0);
        assert!((edges[0].mutual_information_nats - std::f64::consts::LN_2).abs() < 1e-12);
        assert!((edges[0].penalized_gain_nats.unwrap() - gain).abs() < 1e-12);
        let independent: Vec<Vec<String>> = (0..100)
            .map(|i| {
                vec![
                    if i % 2 == 0 { "a".into() } else { "b".into() },
                    if (i / 2) % 2 == 0 {
                        "x".into()
                    } else {
                        "y".into()
                    },
                ]
            })
            .collect();
        let independent_refs: Vec<_> = independent.iter().collect();
        let (independent_parents, _) =
            fit_flat_penalized_forest(&independent_refs, &variables[..2], "g");
        assert_eq!(independent_parents, vec![Vec::<usize>::new(), vec![]]);
    }

    #[test]
    fn binned_forest_learns_numeric_parent_and_samples_supported_bins() {
        let variables: Vec<FlatVariable> = serde_yaml::from_str("- {column: first, kind: numeric, missing_policy: structural_tokens, structural_absence_values: ['-999']}\n- {column: second, kind: numeric}\n").unwrap();
        let rows: Vec<Vec<String>> = (1..=8)
            .map(|i| vec![i.to_string(), (i * 10).to_string()])
            .collect();
        let refs: Vec<_> = rows.iter().collect();
        let fit = fit_flat_binned_forest(&refs, &variables, "g", false);
        assert_eq!(fit.parents, vec![vec![], vec![0]]);
        assert_eq!(fit.cuts[0], vec![2.0, 4.0, 6.0]);
        assert_eq!(fit.edges.len(), 1);
        assert_eq!(
            flat_binned_state("-999", &variables[0], &fit.cuts[0]),
            "raw:-999"
        );
        assert_eq!(
            flat_binned_state("NA", &variables[0], &fit.cuts[0]),
            "raw:NA"
        );
        assert_eq!(flat_binned_state("2", &variables[0], &fit.cuts[0]), "bin:0");
        assert_eq!(
            flat_binned_state("100", &variables[0], &fit.cuts[0]),
            "bin:3"
        );
        let mut seed = 17;
        for _ in 0..100 {
            let sample = sample_flat_binned(&refs, &fit, &mut seed);
            let parent = flat_binned_state(&sample[0], &variables[0], &fit.cuts[0]);
            let child = flat_binned_state(&sample[1], &variables[1], &fit.cuts[1]);
            assert!(fit
                .states
                .iter()
                .any(|row| row[0] == parent && row[1] == child));
        }
        let report = flat_binned_report("g", &fit, &variables, &rows, &rows);
        assert_eq!(report.selected_pairs[0].total_variation_distance, Some(0.0));
        let categorical_variables: Vec<FlatVariable> = serde_yaml::from_str("- {column: first, kind: numeric}\n- {column: second, kind: categorical, allowed_values: [low, high]}\n").unwrap();
        let categorical_rows: Vec<Vec<String>> = (1..=8)
            .map(|i| {
                vec![
                    i.to_string(),
                    if i <= 4 { "low".into() } else { "high".into() },
                ]
            })
            .collect();
        let categorical_refs: Vec<_> = categorical_rows.iter().collect();
        let categorical_fit =
            fit_flat_binned_forest(&categorical_refs, &categorical_variables, "g", false);
        assert_eq!(categorical_fit.parents, vec![vec![], vec![0]]);
        for _ in 0..100 {
            let sample = sample_flat_binned(&categorical_refs, &categorical_fit, &mut seed);
            let n: i32 = sample[0].parse().unwrap();
            assert_eq!(sample[1], if n <= 4 { "low" } else { "high" });
        }
    }

    #[test]
    fn numeric_interpolation_uses_matching_training_neighbors_and_preserves_bins() {
        let config: FlatCohortConfig = serde_yaml::from_str("schema_version: '1'\nallocation_column: site\nallocation_values: [g]\ngenerator: learned_binned_forest\nnumeric_interpolation: [number]\nthresholds: {max_missingness_rate_difference: 1.0, max_categorical_tvd: 1.0, max_numeric_quantile_difference: 100.0}\nvariables:\n  - {column: group, kind: categorical, allowed_values: [a, b]}\n  - {column: number, kind: numeric, missing_policy: structural_tokens, structural_absence_values: ['-999']}\n").unwrap();
        let training: Vec<Vec<String>> = [
            ("a", "1"),
            ("a", "2"),
            ("a", "3"),
            ("a", "4"),
            ("b", "50"),
            ("b", "60"),
            ("b", "70"),
            ("b", "80"),
            ("a", "NA"),
            ("b", "-999"),
        ]
        .iter()
        .map(|(a, b)| vec![(*a).into(), (*b).into()])
        .collect();
        let refs: Vec<_> = training.iter().collect();
        let fit = fit_flat_binned_forest(&refs, &config.variables, "g", false);
        assert_eq!(fit.parents[1], vec![0]);
        let before: Vec<Vec<String>> = (0..100)
            .map(|i| training[i % training.len()].clone())
            .collect();
        let mut generated = before.clone();
        let mut replay = before.clone();
        let report = apply_flat_numeric_interpolation(
            &mut generated,
            &refs,
            &config.variables,
            &config.numeric_interpolation,
            &fit,
            "g",
            11,
        );
        let repeated = apply_flat_numeric_interpolation(
            &mut replay,
            &refs,
            &config.variables,
            &config.numeric_interpolation,
            &fit,
            "g",
            11,
        );
        assert_eq!(generated, replay);
        assert_eq!(
            serde_yaml::to_string(&report).unwrap(),
            serde_yaml::to_string(&repeated).unwrap()
        );
        assert_eq!(report[0].observed_candidates, 80);
        assert!(report[0].changed > 0);
        assert_eq!(report[0].changed + report[0].unchanged, 80);
        for (original, altered) in before.iter().zip(&generated) {
            assert_eq!(original[0], altered[0]);
            assert_eq!(
                flat_binned_state(&original[1], &config.variables[1], &fit.cuts[1]),
                flat_binned_state(&altered[1], &config.variables[1], &fit.cuts[1])
            );
            if original[1] == "NA" || original[1] == "-999" {
                assert_eq!(original, altered);
                continue;
            }
            if original != altered {
                let value: f64 = altered[1].parse().unwrap();
                if original[0] == "a" {
                    assert!((1.0..=4.0).contains(&value));
                } else {
                    assert!((50.0..=80.0).contains(&value));
                }
                assert!(![1.0, 2.0, 3.0, 4.0, 50.0, 60.0, 70.0, 80.0].contains(&value));
            }
        }
        let single = [vec!["a".into(), "1".into()]];
        let single_refs: Vec<_> = single.iter().collect();
        let single_fit = fit_flat_binned_forest(&single_refs, &config.variables, "g", false);
        let mut single_sample = single.to_vec();
        let single_report = apply_flat_numeric_interpolation(
            &mut single_sample,
            &single_refs,
            &config.variables,
            &config.numeric_interpolation,
            &single_fit,
            "g",
            11,
        );
        assert_eq!(single_report[0].unchanged, 1);
        assert_eq!(single_sample, single);
    }

    #[test]
    fn numeric_smoothing_preserves_states_bounds_and_replays() {
        let config: FlatCohortConfig = serde_yaml::from_str("schema_version: '1'\nallocation_column: site\nallocation_values: [g]\nnumeric_smoothing: [{variable: number, bandwidth: 0.5, minimum: 0.0, maximum: 5.0}]\nthresholds: {max_missingness_rate_difference: 1.0, max_categorical_tvd: 1.0, max_numeric_quantile_difference: 10.0}\nvariables:\n  - {column: number, kind: numeric, missing_policy: structural_tokens, structural_absence_values: ['-999']}\n").unwrap();
        let training: Vec<Vec<String>> = ["1", "2", "3", "4", "NA", "-999"]
            .iter()
            .map(|s| vec![(*s).into()])
            .collect();
        let refs: Vec<_> = training.iter().collect();
        let fit = fit_flat_binned_forest(&refs, &config.variables, "g", false);
        let mut sample: Vec<Vec<String>> = (0..60)
            .map(|i| training[i % training.len()].clone())
            .collect();
        let mut replay = sample.clone();
        let before = sample.clone();
        let report =
            apply_flat_numeric_smoothing(&mut sample, &refs, &config, Some(&fit), "g", 11).unwrap();
        let repeated =
            apply_flat_numeric_smoothing(&mut replay, &refs, &config, Some(&fit), "g", 11).unwrap();
        assert_eq!(sample, replay);
        assert_eq!(
            serde_yaml::to_string(&report).unwrap(),
            serde_yaml::to_string(&repeated).unwrap()
        );
        assert_eq!(report[0].observed_candidates, 40);
        assert!(report[0].changed > 0);
        assert_eq!(report[0].changed + report[0].unchanged, 40);
        for (original, altered) in before.iter().zip(&sample) {
            assert_eq!(
                flat_binned_state(&original[0], &config.variables[0], &fit.cuts[0]),
                flat_binned_state(&altered[0], &config.variables[0], &fit.cuts[0])
            );
            if let Ok(value) = altered[0].parse::<f64>() {
                if altered[0] != "-999" {
                    assert!((0.0..=5.0).contains(&value));
                    if original != altered {
                        let before_value: f64 = original[0].parse().unwrap();
                        assert!((value - before_value).abs() <= 0.5);
                        assert!(![1.0, 2.0, 3.0, 4.0].contains(&value));
                    }
                }
            }
            if original[0] == "NA" || original[0] == "-999" {
                assert_eq!(original, altered);
            }
        }
        let outside = [vec!["6".to_owned()]];
        assert!(
            apply_flat_numeric_smoothing(&mut sample, &[&outside[0]], &config, None, "g", 11)
                .is_err()
        );
    }

    #[test]
    fn interaction_dag_learns_xor_but_not_independent_states() {
        let variables: Vec<FlatVariable> = ["a", "b", "outcome"]
            .iter()
            .map(|name| FlatVariable {
                column: (*name).into(),
                kind: "categorical".into(),
                missing_policy: "missing".into(),
                structural_absence_values: vec![],
                allowed_values: vec!["0".into(), "1".into()],
            })
            .collect();
        let mut xor = Vec::new();
        let mut independent = Vec::new();
        for _ in 0..100 {
            for a in 0..2 {
                for b in 0..2 {
                    xor.push(vec![a.to_string(), b.to_string(), (a ^ b).to_string()]);
                    for c in 0..2 {
                        independent.push(vec![a.to_string(), b.to_string(), c.to_string()]);
                    }
                }
            }
        }
        let refs: Vec<_> = xor.iter().collect();
        let greedy =
            fit_flat_binned_selected(&refs, &variables, "g", FlatGenerator::LearnedBinnedDag);
        assert!(greedy.parents[2].is_empty());
        let fit =
            fit_flat_binned_selected(&refs, &variables, "g", FlatGenerator::InteractionBinnedDag);
        assert_eq!(fit.parents[2], vec![0, 1]);
        assert_eq!(
            fit.edges
                .iter()
                .filter(|e| e.child == "outcome" && e.joint_selection_gain_nats.is_some())
                .count(),
            2
        );
        let mut state = 41;
        let mut counts = vec![BTreeMap::new(); 3];
        for _ in 0..200 {
            let row = sample_flat_binned_checked(
                &refs,
                &fit,
                &mut state,
                FlatSparsePolicy::Fail,
                &mut counts,
            )
            .unwrap();
            assert_eq!(row[2], if row[0] == row[1] { "0" } else { "1" });
        }
        let independent_refs: Vec<_> = independent.iter().collect();
        let empty = fit_flat_binned_selected(
            &independent_refs,
            &variables,
            "g",
            FlatGenerator::InteractionBinnedDag,
        );
        assert!(empty.parents[2].is_empty());
    }

    #[test]
    fn binned_dag_learns_incremental_information_and_handles_sparse_states() {
        let variables: Vec<FlatVariable> = ["a", "b", "outcome"]
            .iter()
            .map(|name| FlatVariable {
                column: (*name).into(),
                kind: "categorical".into(),
                missing_policy: "missing".into(),
                structural_absence_values: vec![],
                allowed_values: vec!["0".into(), "1".into()],
            })
            .collect();
        let mut rows = Vec::new();
        for _ in 0..100 {
            for a in 0..2 {
                for b in 0..2 {
                    rows.push(vec![a.to_string(), b.to_string(), (a | b).to_string()]);
                }
            }
        }
        let refs: Vec<_> = rows.iter().collect();
        let fit = fit_flat_binned_selected(&refs, &variables, "g", FlatGenerator::LearnedBinnedDag);
        assert_eq!(fit.parents[2].len(), 2);
        assert_eq!(
            fit.edges
                .iter()
                .filter(|edge| edge.child == "outcome" && edge.conditioned_on.is_some())
                .count(),
            1
        );
        let mut state = 9;
        let mut counts = vec![BTreeMap::new(); 3];
        for _ in 0..200 {
            let sample = sample_flat_binned_checked(
                &refs,
                &fit,
                &mut state,
                FlatSparsePolicy::Fail,
                &mut counts,
            )
            .unwrap();
            assert_eq!(
                sample[2],
                if sample[0] == "1" || sample[1] == "1" {
                    "1"
                } else {
                    "0"
                }
            );
        }
        // Force a training support hole while leaving both parent values individually sampleable.
        let sparse = [
            vec!["0".into(), "0".into(), "0".into()],
            vec!["1".into(), "1".into(), "1".into()],
        ];
        let sparse_refs: Vec<_> = sparse.iter().collect();
        let sparse_fit = FlatBinnedFit {
            cuts: vec![vec![]; 3],
            states: sparse
                .iter()
                .map(|r| r.iter().map(|v| format!("raw:{v}")).collect())
                .collect(),
            parents: vec![vec![], vec![], vec![0, 1]],
            edges: vec![],
        };
        let mut failures = 0;
        let mut backoffs = vec![BTreeMap::new(); 3];
        for seed in 0..100 {
            let mut fail_seed = seed;
            let result = sample_flat_binned_checked(
                &sparse_refs,
                &sparse_fit,
                &mut fail_seed,
                FlatSparsePolicy::Fail,
                &mut vec![BTreeMap::new(); 3],
            );
            let mut backoff_seed = seed;
            let backoff = sample_flat_binned_checked(
                &sparse_refs,
                &sparse_fit,
                &mut backoff_seed,
                FlatSparsePolicy::DropLastParent,
                &mut backoffs,
            )
            .unwrap();
            if result.is_err() {
                failures += 1;
                assert_eq!(backoff[0], backoff[2]);
            }
        }
        assert!(failures > 0);
        assert_eq!(backoffs[2].get(&1), Some(&failures));
        let stratified = FlatFittedStratifiedForest {
            groups: BTreeMap::from([(
                "sparse".into(),
                FlatFittedBinnedForest {
                    training: &sparse,
                    variables: &variables,
                    group: "sparse".into(),
                    fit: sparse_fit,
                },
            )]),
        };
        let allocation = BTreeMap::from([("sparse".into(), 200)]);
        let report = stratified
            .sample_with_policy(41, &allocation, FlatSparsePolicy::DropLastParent)
            .unwrap();
        assert_eq!(report["sparse"].rows.len(), 200);
        assert!(report["sparse"].sparse_backoffs[0].dropped_parent_counts[&1] > 0);
        let error = stratified
            .sample_with_policy(41, &allocation, FlatSparsePolicy::Fail)
            .err()
            .unwrap()
            .to_string();
        assert!(error.contains("sampling group sparse failed"));
    }

    #[test]
    fn interpolation_with_sparse_backoff_reports_both_and_preserves_draws() {
        let variables: Vec<FlatVariable> = serde_yaml::from_str("- {column: a, kind: categorical, allowed_values: ['0', '1']}\n- {column: b, kind: categorical, allowed_values: ['0', '1']}\n- {column: c, kind: categorical, allowed_values: ['0', '1']}\n- {column: value, kind: numeric}\n").unwrap();
        let training = [
            vec!["0".into(), "0".into(), "0".into(), "1".into()],
            vec!["1".into(), "1".into(), "1".into(), "2".into()],
        ];
        let states = training
            .iter()
            .map(|row| {
                vec![
                    format!("raw:{}", row[0]),
                    format!("raw:{}", row[1]),
                    format!("raw:{}", row[2]),
                    "bin:0".into(),
                ]
            })
            .collect();
        let fit = FlatBinnedFit {
            cuts: vec![vec![]; 4],
            states,
            parents: vec![vec![], vec![], vec![0, 1], vec![]],
            edges: vec![],
        };
        let model = FlatFittedBinnedForest {
            training: &training,
            variables: &variables,
            group: "g".into(),
            fit,
        };
        let columns = vec!["value".into()];
        let original = model
            .sample_with_policy(41, 200, FlatSparsePolicy::DropLastParent)
            .unwrap();
        let interpolated = model
            .sample_interpolated_with_policy(41, 200, &columns, FlatSparsePolicy::DropLastParent)
            .unwrap();
        let replay = model
            .sample_interpolated_with_policy(41, 200, &columns, FlatSparsePolicy::DropLastParent)
            .unwrap();
        assert_eq!(interpolated.rows, replay.rows);
        assert_eq!(
            original.sparse_backoffs[0].dropped_parent_counts,
            interpolated.sparse_backoffs[0].dropped_parent_counts
        );
        assert!(interpolated.sparse_backoffs[0].dropped_parent_counts[&1] > 0);
        assert!(interpolated.interpolation[0].changed > 0);
        assert!(original
            .rows
            .iter()
            .zip(&interpolated.rows)
            .all(|(a, b)| a[..3] == b[..3]));
        assert!(model
            .sample_interpolated_with_policy(41, 200, &columns, FlatSparsePolicy::Fail)
            .is_err());
        let stratified = FlatFittedStratifiedForest {
            groups: BTreeMap::from([("g".into(), model)]),
        };
        let allocation = BTreeMap::from([("g".into(), 200)]);
        let grouped = stratified
            .sample_interpolated_with_policy(
                41,
                &allocation,
                &columns,
                FlatSparsePolicy::DropLastParent,
            )
            .unwrap();
        assert_eq!(grouped["g"].rows, interpolated.rows);
        assert_eq!(
            grouped["g"].sparse_backoffs[0].dropped_parent_counts,
            interpolated.sparse_backoffs[0].dropped_parent_counts
        );
        assert!(stratified
            .sample_interpolated_with_policy(41, &allocation, &columns, FlatSparsePolicy::Fail)
            .err()
            .unwrap()
            .to_string()
            .contains("sampling group g failed"));
    }

    #[test]
    fn utility_task_runs_in_inner_validation_and_rejects_invalid_declarations() {
        let path = std::env::temp_dir().join(format!("janus-utility-{}.csv", std::process::id()));
        let mut csv = String::from("no,site,x,y\n");
        for i in 0..40 {
            csv.push_str(&format!(
                "p{i},g,{},{}\n",
                if i % 2 == 0 { "a" } else { "b" },
                if i % 3 == 0 { "yes" } else { "no" }
            ));
        }
        std::fs::write(&path, csv).unwrap();
        let config: FlatCohortConfig = serde_yaml::from_str("schema_version: '1'\nallocation_column: site\nallocation_values: [g]\ndevelopment_validation_proportion: 0.25\nbootstrap_replicates: 100\nthresholds: {max_missingness_rate_difference: 1.0, max_categorical_tvd: 1.0, max_numeric_quantile_difference: 1000.0}\nvariables:\n  - {column: x, kind: categorical, allowed_values: [a, b]}\n  - {column: y, kind: categorical, allowed_values: ['yes', 'no']}\nutility_tasks:\n  - {name: prediction, predictor: x, target: y, positive_values: ['yes'], negative_values: ['no'], alpha: 1.0, predictor_available_before_target: true}\n").unwrap();
        let mut config = config;
        config.joint_families.push(vec!["x".into(), "y".into()]);
        let first = describe_flat_csv(&path, &config).unwrap();
        let second = describe_flat_csv(&path, &config).unwrap();
        let family = &first.development_validation[0].joint_families[0];
        assert_eq!(family.generated_patients, family.validation_patients);
        assert!(family.total_variation_distance.is_some());
        assert_eq!(
            serde_yaml::to_string(family).unwrap(),
            serde_yaml::to_string(&second.development_validation[0].joint_families[0]).unwrap()
        );
        let mut duplicate = config.clone();
        duplicate.joint_families.push(vec!["y".into(), "x".into()]);
        assert!(describe_flat_csv(&path, &duplicate).is_err());
        duplicate = config.clone();
        duplicate.joint_families[0] = vec!["x".into(), "x".into()];
        assert!(describe_flat_csv(&path, &duplicate).is_err());
        duplicate.joint_families[0] = vec!["x".into(), "unknown".into()];
        assert!(describe_flat_csv(&path, &duplicate).is_err());
        let report = &first.development_validation[0].utility[0];
        assert!(report.real_training.brier_score.is_some());
        assert_eq!(
            serde_yaml::to_string(report).unwrap(),
            serde_yaml::to_string(&second.development_validation[0].utility[0]).unwrap()
        );
        let mut invalid = config.clone();
        invalid.utility_tasks[0]
            .additional_predictors
            .push("x".into());
        assert!(describe_flat_csv(&path, &invalid).is_err());
        invalid = config.clone();
        invalid.utility_tasks[0]
            .additional_predictors
            .push("y".into());
        assert!(describe_flat_csv(&path, &invalid).is_err());
        invalid = config.clone();
        invalid.utility_tasks[0]
            .additional_predictors
            .push("unknown".into());
        assert!(describe_flat_csv(&path, &invalid).is_err());
        invalid = config.clone();
        invalid.utility_tasks[0].predictor_available_before_target = false;
        assert!(describe_flat_csv(&path, &invalid).is_err());
        invalid = config.clone();
        invalid.utility_tasks[0].negative_values.push("yes".into());
        assert!(describe_flat_csv(&path, &invalid).is_err());
        invalid = config.clone();
        invalid.development_validation_proportion = None;
        assert!(describe_flat_csv(&path, &invalid).is_err());
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn utility_rejects_missing_target_labels_and_counts_missing_outcomes() {
        let path =
            std::env::temp_dir().join(format!("janus-utility-missing-{}.csv", std::process::id()));
        let mut csv = String::from("no,site,x,y\n");
        for i in 0..100 {
            let target = match i % 4 {
                0 => "NA",
                1 => "",
                2 => "yes",
                _ => "no",
            };
            csv.push_str(&format!("p{i},g,a,{target}\n"));
        }
        std::fs::write(&path, csv).unwrap();
        let config: FlatCohortConfig = serde_yaml::from_str("schema_version: '1'\nallocation_column: site\nallocation_values: [g]\ndevelopment_validation_proportion: 0.25\nbootstrap_replicates: 100\nthresholds: {max_missingness_rate_difference: 1.0, max_categorical_tvd: 1.0, max_numeric_quantile_difference: 1000.0}\nvariables:\n  - {column: x, kind: categorical, allowed_values: [a]}\n  - {column: y, kind: categorical, allowed_values: ['yes', 'no', 'NA']}\nutility_tasks:\n  - {name: prediction, predictor: x, target: y, positive_values: ['yes'], negative_values: ['no'], alpha: 1.0, predictor_available_before_target: true}\n").unwrap();
        for positive in [false, true] {
            let mut invalid = config.clone();
            let labels = if positive {
                &mut invalid.utility_tasks[0].positive_values
            } else {
                &mut invalid.utility_tasks[0].negative_values
            };
            labels.push("NA".into());
            assert!(
                describe_flat_csv(&path, &invalid).is_err(),
                "missing target must not be a utility label"
            );
        }
        let mut invalid = config.clone();
        invalid.variables[1].allowed_values.push(String::new());
        invalid.utility_tasks[0].negative_values.push(String::new());
        assert!(describe_flat_csv(&path, &invalid).is_err());
        let mut invalid_structural = config.clone();
        invalid_structural.variables[1].missing_policy = "structural_tokens".into();
        invalid_structural.variables[1]
            .structural_absence_values
            .push("NA".into());
        invalid_structural.utility_tasks[0]
            .negative_values
            .push("NA".into());
        assert!(describe_flat_csv(&path, &invalid_structural).is_err());
        let report = describe_flat_csv(&path, &config).unwrap();
        let validation = &report.development_validation[0];
        let utility = &validation.utility[0];
        assert!(utility.real_training.training_unlabeled > 0);
        assert!(utility.validation_unlabeled > 0);
        assert_eq!(
            utility.validation_unlabeled,
            validation.validation.variables[1].missing
        );
        assert_eq!(
            utility.synthetic_training.training_unlabeled,
            validation.generated.variables[1].missing
        );
        assert_eq!(
            utility.real_training.training_labeled + utility.real_training.training_unlabeled,
            validation.training_patients
        );
        assert_eq!(
            utility.real_training.validation_labeled + utility.validation_unlabeled,
            validation.validation_patients
        );
        assert_eq!(
            utility.synthetic_training.validation_labeled,
            utility.real_training.validation_labeled
        );
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn multivariable_utility_detects_lost_xor_interaction() {
        let variables: Vec<FlatVariable> = ["a", "b", "outcome"]
            .iter()
            .map(|name| FlatVariable {
                column: (*name).into(),
                kind: "categorical".into(),
                missing_policy: "missing".into(),
                structural_absence_values: vec![],
                allowed_values: vec!["0".into(), "1".into()],
            })
            .collect();
        let mut training = Vec::new();
        let mut validation = Vec::new();
        for _ in 0..100 {
            for a in 0..2 {
                for b in 0..2 {
                    training.push(vec![a.to_string(), b.to_string(), (a ^ b).to_string()]);
                }
            }
        }
        for a in 0..2 {
            for b in 0..2 {
                validation.push(vec![a.to_string(), b.to_string(), (a ^ b).to_string()]);
            }
        }
        let task = FlatUtilityTask {
            name: "xor".into(),
            predictor: "a".into(),
            additional_predictors: vec!["b".into()],
            target: "outcome".into(),
            positive_values: vec!["1".into()],
            negative_values: vec!["0".into()],
            alpha: 1.0,
            predictor_available_before_target: true,
        };
        let interaction = FlatFittedBinnedForest::fit_interaction_dag(&training, &variables, "g")
            .unwrap()
            .sample(41, 400)
            .unwrap();
        let greedy = FlatFittedBinnedForest::fit_dag(&training, &variables, "g")
            .unwrap()
            .sample(41, 400)
            .unwrap();
        let interaction_refs: Vec<_> = interaction.iter().collect();
        let greedy_refs: Vec<_> = greedy.iter().collect();
        let good = flat_utility_score(&interaction_refs, &validation, &task, &[0, 1], 2);
        let bad = flat_utility_score(&greedy_refs, &validation, &task, &[0, 1], 2);
        assert_eq!(good.validation_labeled, 4);
        assert!(good.brier_score.unwrap() < 0.01);
        assert!(bad.brier_score.unwrap() > 0.1);
        let single_state = [training[0].clone()];
        let single_refs: Vec<_> = single_state.iter().collect();
        let score = flat_utility_score(&single_refs, &validation, &task, &[0, 1], 2);
        assert_eq!(score.unseen_predictor, 3);
    }

    #[test]
    fn utility_scores_smoothed_predictions_and_unseen_states() {
        let task = FlatUtilityTask {
            name: "test".into(),
            predictor: "x".into(),
            additional_predictors: vec![],
            target: "y".into(),
            positive_values: vec!["yes".into()],
            negative_values: vec!["no".into()],
            alpha: 1.0,
            predictor_available_before_target: true,
        };
        let training = [
            vec!["a".into(), "yes".into()],
            vec!["a".into(), "no".into()],
            vec!["b".into(), "yes".into()],
            vec!["b".into(), "missing".into()],
        ];
        let refs: Vec<_> = training.iter().collect();
        let validation = vec![
            vec!["a".into(), "yes".into()],
            vec!["unseen".into(), "no".into()],
            vec!["a".into(), "missing".into()],
        ];
        let score = flat_utility_score(&refs, &validation, &task, &[0], 1);
        assert_eq!(
            (
                score.training_labeled,
                score.training_unlabeled,
                score.validation_labeled,
                score.unseen_predictor
            ),
            (3, 1, 2, 1)
        );
        assert!((score.brier_score.unwrap() - (0.25 + 0.36) / 2.0).abs() < 1e-12);
        assert!(flat_utility_score(&[], &validation, &task, &[0], 1)
            .brier_score
            .is_none());
        assert!(flat_utility_score(&refs, &[], &task, &[0], 1)
            .brier_score
            .is_none());
    }

    #[test]
    fn numeric_associations_handle_ties_missing_and_constant_values() {
        let variables: Vec<FlatVariable> = serde_yaml::from_str("- {column: a, kind: numeric, missing_policy: structural_tokens, structural_absence_values: ['-999']}\n- {column: b, kind: numeric}\n").unwrap();
        let rows: Vec<Vec<String>> = [
            ("1", "10"),
            ("1", "10"),
            ("2", "20"),
            ("3", "30"),
            ("-999", "40"),
            ("NA", "50"),
            ("4", "NA"),
        ]
        .iter()
        .map(|(a, b)| vec![(*a).into(), (*b).into()])
        .collect();
        let (n, correlation) = flat_spearman(&rows, &variables, 0, 1);
        assert_eq!(n, 4);
        assert!((correlation.unwrap() - 1.0).abs() < 1e-12);
        let constant: Vec<Vec<String>> = [("1", "3"), ("1", "2"), ("1", "1")]
            .iter()
            .map(|(a, b)| vec![(*a).into(), (*b).into()])
            .collect();
        assert_eq!(flat_spearman(&constant, &variables, 0, 1), (3, None));
        let reverse: Vec<Vec<String>> = [("1", "3"), ("2", "2"), ("3", "1")]
            .iter()
            .map(|(a, b)| vec![(*a).into(), (*b).into()])
            .collect();
        assert!((flat_spearman(&reverse, &variables, 0, 1).1.unwrap() + 1.0).abs() < 1e-12);
    }

    #[test]
    fn binned_forest_uses_development_only_in_outer_and_inner_fits() {
        let dir = std::env::temp_dir().join(format!("janus-binned-forest-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("fixture.csv");
        let mut ids: Vec<_> = (0..40).map(|i| format!("p{i}")).collect();
        ids.sort_by_key(|id| (flat_hash(41, id), id.clone()));
        let make_csv = |change_holdout: bool| {
            let mut csv = String::from("no,site,a,b\n");
            for (i, id) in ids.iter().enumerate() {
                let a = if i >= 32 && change_holdout {
                    500
                } else {
                    (i % 8) + 1
                };
                let b = if i >= 32 && change_holdout {
                    900
                } else {
                    a * 10
                };
                csv.push_str(&format!("{id},g,{a},{b}\n"));
            }
            csv
        };
        let config: FlatCohortConfig = serde_yaml::from_str("schema_version: '1'\nallocation_column: site\nallocation_values: [g]\ngenerator: learned_binned_forest\nnumeric_associations: [{parent: a, child: b}]\nsplit_seed: 41\ndevelopment_proportion: 0.8\ndevelopment_validation_proportion: 0.25\nbootstrap_replicates: 100\nthresholds: {max_missingness_rate_difference: 1.0, max_categorical_tvd: 1.0, max_numeric_quantile_difference: 1000.0}\nvariables:\n  - {column: a, kind: numeric}\n  - {column: b, kind: numeric}\n").unwrap();
        std::fs::write(&path, make_csv(false)).unwrap();
        let first = describe_flat_csv(&path, &config).unwrap();
        let mut interpolation_config = config.clone();
        interpolation_config.numeric_interpolation = vec!["b".into()];
        let interpolated = describe_flat_csv(&path, &interpolation_config).unwrap();
        assert!(interpolated.numeric_interpolation[0].changed > 0);
        assert!(interpolated.development_validation[0].numeric_interpolation[0].changed > 0);
        let mut invalid_interpolation = interpolation_config.clone();
        invalid_interpolation.generator = FlatGenerator::IndependentMarginals;
        assert!(describe_flat_csv(&path, &invalid_interpolation).is_err());
        invalid_interpolation = interpolation_config.clone();
        invalid_interpolation.numeric_interpolation.push("b".into());
        assert!(describe_flat_csv(&path, &invalid_interpolation).is_err());
        let mut smoothed_config = config.clone();
        smoothed_config
            .numeric_smoothing
            .push(FlatNumericSmoothing {
                variable: "b".into(),
                bandwidth: 0.5,
                minimum: 0.0,
                maximum: 100.0,
            });
        let smoothed = describe_flat_csv(&path, &smoothed_config).unwrap();
        assert!(smoothed.numeric_smoothing[0].changed > 0);
        assert!(smoothed.development_validation[0].numeric_smoothing[0].changed > 0);
        let mut invalid_smoothing = smoothed_config.clone();
        invalid_smoothing.numeric_smoothing[0].bandwidth = 0.0;
        assert!(describe_flat_csv(&path, &invalid_smoothing).is_err());
        assert_eq!(first.learned_dependencies.len(), 1);
        assert_eq!(first.binned_models[0].selected_pairs.len(), 1);
        assert!(first.development_validation[0].binned_model.is_some());
        let mut regularized = config.clone();
        regularized.generator = FlatGenerator::RegularizedBinnedForest;
        let regularized_report = describe_flat_csv(&path, &regularized).unwrap();
        assert_eq!(regularized_report.learned_dependencies.len(), 1);
        assert!(
            regularized_report.learned_dependencies[0]
                .penalized_gain_nats
                .unwrap()
                > 0.0
        );
        assert_eq!(regularized_report.binned_models[0].selected_pairs.len(), 1);
        assert!(regularized_report.development_validation[0]
            .binned_model
            .is_some());
        let mut dag = config.clone();
        dag.generator = FlatGenerator::LearnedBinnedDag;
        dag.sparse_policy = FlatSparsePolicy::DropLastParent;
        let dag_first = describe_flat_csv(&path, &dag).unwrap();
        let association = &first.development_validation[0].numeric_associations[0];
        assert_eq!(association.validation_complete_pairs, 8);
        assert!((association.validation_spearman.unwrap() - 1.0).abs() < 1e-12);
        let mut invalid = config.clone();
        invalid.numeric_associations.push(FlatDependency {
            parent: "b".into(),
            child: "a".into(),
        });
        assert!(describe_flat_csv(&path, &invalid).is_err());
        let mut missing_validation = config.clone();
        missing_validation.development_validation_proportion = None;
        assert!(describe_flat_csv(&path, &missing_validation).is_err());
        std::fs::write(&path, make_csv(true)).unwrap();
        let changed = describe_flat_csv(&path, &config).unwrap();
        let dag_changed = describe_flat_csv(&path, &dag).unwrap();
        assert_eq!(
            serde_yaml::to_string(&dag_first.generated).unwrap(),
            serde_yaml::to_string(&dag_changed.generated).unwrap()
        );
        assert_eq!(
            serde_yaml::to_string(&dag_first.development_validation).unwrap(),
            serde_yaml::to_string(&dag_changed.development_validation).unwrap()
        );
        let interpolated_changed = describe_flat_csv(&path, &interpolation_config).unwrap();
        assert_eq!(
            serde_yaml::to_string(&interpolated.generated).unwrap(),
            serde_yaml::to_string(&interpolated_changed.generated).unwrap()
        );
        assert_eq!(
            serde_yaml::to_string(&interpolated.development_validation).unwrap(),
            serde_yaml::to_string(&interpolated_changed.development_validation).unwrap()
        );
        let smoothed_changed = describe_flat_csv(&path, &smoothed_config).unwrap();
        assert_eq!(
            serde_yaml::to_string(&smoothed.generated).unwrap(),
            serde_yaml::to_string(&smoothed_changed.generated).unwrap()
        );
        assert_eq!(
            serde_yaml::to_string(&smoothed.development_validation).unwrap(),
            serde_yaml::to_string(&smoothed_changed.development_validation).unwrap()
        );
        let regularized_changed = describe_flat_csv(&path, &regularized).unwrap();
        assert_eq!(
            serde_yaml::to_string(&regularized_report.learned_dependencies).unwrap(),
            serde_yaml::to_string(&regularized_changed.learned_dependencies).unwrap()
        );
        assert_eq!(
            serde_yaml::to_string(&regularized_report.development_validation).unwrap(),
            serde_yaml::to_string(&regularized_changed.development_validation).unwrap()
        );
        assert_eq!(
            serde_yaml::to_string(&first.development_validation).unwrap(),
            serde_yaml::to_string(&changed.development_validation).unwrap()
        );
        assert_eq!(
            serde_yaml::to_string(&first.learned_dependencies).unwrap(),
            serde_yaml::to_string(&changed.learned_dependencies).unwrap()
        );
        assert_eq!(
            serde_yaml::to_string(&first.generated).unwrap(),
            serde_yaml::to_string(&changed.generated).unwrap()
        );
        assert_ne!(
            serde_yaml::to_string(&first.binned_models).unwrap(),
            serde_yaml::to_string(&changed.binned_models).unwrap()
        );
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn mixed_forest_learns_numeric_distribution_and_preserves_absence() {
        let variables: Vec<FlatVariable> = serde_yaml::from_str("- {column: group, kind: categorical, allowed_values: [a, b]}\n- {column: value, kind: numeric, structural_absence_values: ['-999']}\n").unwrap();
        let rows: Vec<Vec<String>> = [
            ("a", "1"),
            ("a", "2"),
            ("a", "-999"),
            ("b", "8"),
            ("b", "9"),
            ("b", "NA"),
        ]
        .iter()
        .map(|(a, b)| vec![(*a).into(), (*b).into()])
        .collect();
        let refs: Vec<_> = rows.iter().collect();
        let (parents, edges) = fit_flat_ordered_forest(&refs, &variables, "g", true);
        assert_eq!(parents, vec![vec![], vec![0]]);
        assert_eq!(edges.len(), 1);
        let mut seed = 4;
        let mut backoffs = vec![BTreeMap::new(); 2];
        for _ in 0..100 {
            let sample = sample_flat_conditionals(
                &refs,
                &parents,
                &mut seed,
                FlatSparsePolicy::Fail,
                &mut backoffs,
            )
            .unwrap();
            assert!(rows.contains(&sample));
        }
        let metrics =
            flat_conditional_numeric_metrics("g", "group", &rows, &rows, 0, 1, &variables[1]);
        let a = metrics.iter().find(|m| m.parent_value == "a").unwrap();
        assert_eq!(
            (
                a.generated_observed,
                a.generated_missing,
                a.generated_structurally_absent
            ),
            (2, 0, 1)
        );
        let b = metrics.iter().find(|m| m.parent_value == "b").unwrap();
        assert_eq!(
            (
                b.generated_observed,
                b.generated_missing,
                b.generated_structurally_absent
            ),
            (2, 1, 0)
        );
    }

    #[test]
    fn ordered_forest_fits_development_only_and_reports_heldout_pair() {
        let dir = std::env::temp_dir().join(format!("janus-flat-forest-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("fixture.csv");
        let mut ids: Vec<_> = (0..8).map(|i| format!("p{i}")).collect();
        ids.sort_by_key(|id| (flat_hash(41, id), id.clone()));
        let make_input = |change_holdout: bool| {
            let mut out = String::from("no,site,a,b\n");
            for (i, id) in ids.iter().enumerate() {
                let values = if i >= 4 && change_holdout {
                    "a,y"
                } else if i % 2 == 0 {
                    "a,x"
                } else {
                    "b,y"
                };
                out.push_str(&format!("{id},g,{values}\n"));
            }
            out
        };
        let config: FlatCohortConfig = serde_yaml::from_str("schema_version: '1'\nallocation_column: site\nallocation_values: [g]\ngenerator: learned_categorical_forest\nsplit_seed: 41\ndevelopment_proportion: 0.5\nbootstrap_replicates: 100\nthresholds: {max_missingness_rate_difference: 1.0, max_categorical_tvd: 1.0, max_numeric_quantile_difference: 1.0}\nvariables:\n  - {column: a, kind: categorical, allowed_values: [a, b]}\n  - {column: b, kind: categorical, allowed_values: [x, y]}\n").unwrap();
        std::fs::write(&path, make_input(false)).unwrap();
        let first = describe_flat_csv(&path, &config).unwrap();
        assert_eq!(first.learned_dependencies.len(), 1);
        assert_eq!(first.joint_generated_vs_holdout.len(), 1);
        assert_eq!(first.learned_dependencies[0].development_patients, 4);
        std::fs::write(&path, make_input(true)).unwrap();
        let changed = describe_flat_csv(&path, &config).unwrap();
        assert_eq!(
            serde_yaml::to_string(&first.learned_dependencies).unwrap(),
            serde_yaml::to_string(&changed.learned_dependencies).unwrap()
        );
        assert_eq!(
            serde_yaml::to_string(&first.generated).unwrap(),
            serde_yaml::to_string(&changed.generated).unwrap()
        );
        assert_ne!(
            serde_yaml::to_string(&first.joint_generated_vs_holdout).unwrap(),
            serde_yaml::to_string(&changed.joint_generated_vs_holdout).unwrap()
        );
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn exact_match_diagnostics_count_rows_and_respect_raw_states() {
        let training = [vec!["x".into(), "NA".into()], vec!["y".into(), "1".into()]];
        let refs: Vec<_> = training.iter().collect();
        let validation = vec![vec!["x".into(), "NA".into()], vec!["x".into(), "".into()]];
        let generated = vec![
            vec!["x".into(), "NA".into()],
            vec!["x".into(), "NA".into()],
            vec!["x".into(), "".into()],
            vec!["y".into(), "2".into()],
        ];
        let result = flat_exact_matches(&refs, &validation, &generated);
        assert_eq!(
            (
                result.generated_matching_training,
                result.generated_matching_validation,
                result.validation_matching_training,
                result.duplicate_generated_rows
            ),
            (2, 3, 1, 1)
        );
        assert_eq!(
            (
                result.training_patients,
                result.validation_patients,
                result.generated_patients
            ),
            (2, 2, 4)
        );
        let serialized = serde_yaml::to_string(&result).unwrap();
        assert!(!serialized.contains("NA"));
        let empty = flat_exact_matches(&[], &[], &[]);
        assert_eq!(
            (
                empty.generated_patients,
                empty.generated_matching_training,
                empty.duplicate_generated_rows
            ),
            (0, 0, 0)
        );
    }

    #[test]
    fn development_validation_is_nested_and_fitted_only_on_inner_training() {
        let dir =
            std::env::temp_dir().join(format!("janus-nested-validation-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("fixture.csv");
        let mut ids: Vec<_> = (0..40).map(|i| format!("p{i}")).collect();
        ids.sort_by_key(|id| (flat_hash(41, id), id.clone()));
        let outer: HashSet<_> = ids.iter().take(32).cloned().collect();
        let mut inner: Vec<_> = outer.iter().cloned().collect();
        inner.sort_by_key(|id| (flat_hash(41 ^ 0x7661_6c69_6461_7465, id), id.clone()));
        let validation: HashSet<_> = inner.iter().take(8).cloned().collect();
        let make_csv = |change_outer: bool, change_inner: bool| {
            let mut csv = String::from("no,site,a,b\n");
            for (i, id) in ids.iter().enumerate() {
                let a = if i % 2 == 0 { "a" } else { "b" };
                let b = if (change_outer && !outer.contains(id))
                    || (change_inner && validation.contains(id))
                {
                    "z"
                } else if a == "a" {
                    "x"
                } else {
                    "y"
                };
                csv.push_str(&format!("{id},g,{a},{b}\n"));
            }
            csv
        };
        let config: FlatCohortConfig = serde_yaml::from_str("schema_version: '1'\nallocation_column: site\nallocation_values: [g]\ngenerator: learned_categorical_forest\ndependencies: [{parent: a, child: b}]\nsplit_seed: 41\ndevelopment_proportion: 0.8\ndevelopment_validation_proportion: 0.25\nbootstrap_replicates: 100\nthresholds: {max_missingness_rate_difference: 1.0, max_categorical_tvd: 1.0, max_numeric_quantile_difference: 1.0}\nvariables:\n  - {column: a, kind: categorical, allowed_values: [a, b]}\n  - {column: b, kind: categorical, allowed_values: [x, y, z]}\n").unwrap();
        std::fs::write(&path, make_csv(false, false)).unwrap();
        let first = describe_flat_csv(&path, &config).unwrap();
        let nested = &first.development_validation[0];
        assert_eq!(
            (nested.training_patients, nested.validation_patients),
            (24, 8)
        );
        assert_eq!(nested.categorical_pairs.len(), 1);
        std::fs::write(&path, make_csv(true, false)).unwrap();
        let changed_outer = describe_flat_csv(&path, &config).unwrap();
        assert_eq!(
            serde_yaml::to_string(&first.development_validation).unwrap(),
            serde_yaml::to_string(&changed_outer.development_validation).unwrap()
        );
        std::fs::write(&path, make_csv(false, true)).unwrap();
        let changed_inner = describe_flat_csv(&path, &config).unwrap();
        let changed = &changed_inner.development_validation[0];
        assert_eq!(
            serde_yaml::to_string(&nested.learned_dependencies).unwrap(),
            serde_yaml::to_string(&changed.learned_dependencies).unwrap()
        );
        assert_eq!(
            serde_yaml::to_string(&nested.generated).unwrap(),
            serde_yaml::to_string(&changed.generated).unwrap()
        );
        assert_ne!(
            serde_yaml::to_string(&nested.categorical_pairs).unwrap(),
            serde_yaml::to_string(&changed.categorical_pairs).unwrap()
        );
        for generator in [
            FlatGenerator::RowBootstrap,
            FlatGenerator::IndependentMarginals,
            FlatGenerator::ConditionalCategorical,
            FlatGenerator::LearnedMixedForest,
        ] {
            let mut candidate = config.clone();
            candidate.generator = generator;
            candidate.exact_match_diagnostics = true;
            let compared = describe_flat_csv(&path, &candidate).unwrap();
            let result = &compared.development_validation[0];
            assert_eq!(
                (result.training_patients, result.validation_patients),
                (24, 8)
            );
            assert_eq!(
                serde_yaml::to_string(&result.validation).unwrap(),
                serde_yaml::to_string(&changed.validation).unwrap()
            );
            assert_eq!(result.categorical_pairs.len(), 1);
            let matches = result.exact_matches.as_ref().unwrap();
            assert_eq!(matches.generated_patients, 8);
            if generator == FlatGenerator::RowBootstrap {
                assert_eq!(
                    matches.generated_matching_training,
                    matches.generated_patients
                );
            }
        }
        let mut alternate_split = config.clone();
        alternate_split.development_validation_seed = Some(7);
        let alternate = describe_flat_csv(&path, &alternate_split).unwrap();
        assert_eq!(
            serde_yaml::to_string(&changed_inner.partitions).unwrap(),
            serde_yaml::to_string(&alternate.partitions).unwrap()
        );
        assert_eq!(
            serde_yaml::to_string(&changed_inner.generated).unwrap(),
            serde_yaml::to_string(&alternate.generated).unwrap()
        );
        assert_eq!(alternate.development_validation[0].split_seed, 7);
        assert_ne!(
            serde_yaml::to_string(&changed.validation).unwrap(),
            serde_yaml::to_string(&alternate.development_validation[0].validation).unwrap()
        );
        let mut alternate_draw = config.clone();
        alternate_draw.generator_seed = Some(99);
        let redrawn = describe_flat_csv(&path, &alternate_draw).unwrap();
        assert_eq!(
            serde_yaml::to_string(&changed.validation).unwrap(),
            serde_yaml::to_string(&redrawn.development_validation[0].validation).unwrap()
        );
        assert_eq!(
            serde_yaml::to_string(&changed.learned_dependencies).unwrap(),
            serde_yaml::to_string(&redrawn.development_validation[0].learned_dependencies).unwrap()
        );
        let mut invalid = config.clone();
        invalid.development_validation_proportion = Some(1.0);
        assert!(describe_flat_csv(&path, &invalid).is_err());
        invalid.development_validation_proportion = None;
        invalid.development_validation_seed = Some(7);
        assert!(describe_flat_csv(&path, &invalid).is_err());
        invalid.development_validation_seed = None;
        invalid.exact_match_diagnostics = true;
        assert!(describe_flat_csv(&path, &invalid).is_err());
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn mixed_forest_fits_only_development_and_reports_numeric_relationship() {
        let dir = std::env::temp_dir().join(format!("janus-flat-mixed-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("fixture.csv");
        let mut ids: Vec<_> = (0..12).map(|i| format!("p{i}")).collect();
        ids.sort_by_key(|id| (flat_hash(41, id), id.clone()));
        let make_input = |change_holdout: bool| {
            let mut out = String::from("no,site,a,value\n");
            for (i, id) in ids.iter().enumerate() {
                let value = if i >= 6 && change_holdout {
                    "50"
                } else if i % 2 == 0 {
                    "1"
                } else {
                    "9"
                };
                out.push_str(&format!(
                    "{id},g,{},{}\n",
                    if i % 2 == 0 { "a" } else { "b" },
                    value
                ));
            }
            out
        };
        let config: FlatCohortConfig = serde_yaml::from_str("schema_version: '1'\nallocation_column: site\nallocation_values: [g]\ngenerator: learned_mixed_forest\nsplit_seed: 41\ndevelopment_proportion: 0.5\nbootstrap_replicates: 100\nthresholds: {max_missingness_rate_difference: 1.0, max_categorical_tvd: 1.0, max_numeric_quantile_difference: 100.0}\nvariables:\n  - {column: a, kind: categorical, allowed_values: [a, b]}\n  - {column: value, kind: numeric}\n").unwrap();
        std::fs::write(&path, make_input(false)).unwrap();
        let first = describe_flat_csv(&path, &config).unwrap();
        assert_eq!(first.learned_dependencies.len(), 1);
        assert_eq!(first.conditional_numeric_generated_vs_holdout.len(), 2);
        std::fs::write(&path, make_input(true)).unwrap();
        let changed = describe_flat_csv(&path, &config).unwrap();
        assert_eq!(
            serde_yaml::to_string(&first.learned_dependencies).unwrap(),
            serde_yaml::to_string(&changed.learned_dependencies).unwrap()
        );
        assert_eq!(
            serde_yaml::to_string(&first.generated).unwrap(),
            serde_yaml::to_string(&changed.generated).unwrap()
        );
        assert_ne!(
            serde_yaml::to_string(&first.conditional_numeric_generated_vs_holdout).unwrap(),
            serde_yaml::to_string(&changed.conditional_numeric_generated_vs_holdout).unwrap()
        );
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn flat_numeric_absence_tokens_work_end_to_end() {
        let dir =
            std::env::temp_dir().join(format!("janus-numeric-absence-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("fixture.csv");
        let mut ids: Vec<_> = (0..16).map(|i| format!("p{i}")).collect();
        ids.sort_by_key(|id| (flat_hash(41, id), id.clone()));
        let mut csv = String::from("no,site,parent,value\n");
        for (i, id) in ids.iter().enumerate() {
            let value = ["1", "NONE", "-999", "NA"][i % 4];
            csv.push_str(&format!(
                "{id},g,{},{value}\n",
                if i % 2 == 0 { "a" } else { "b" }
            ));
        }
        std::fs::write(&path, csv).unwrap();
        let config: FlatCohortConfig = serde_yaml::from_str("schema_version: '1'\nallocation_column: site\nallocation_values: [g]\ngenerator: conditional_empirical\ndependencies: [{parent: parent, child: value}]\nsplit_seed: 41\ndevelopment_proportion: 0.5\nbootstrap_replicates: 100\nthresholds: {max_missingness_rate_difference: 1.0, max_categorical_tvd: 1.0, max_numeric_quantile_difference: 100.0}\nvariables:\n  - {column: parent, kind: categorical, allowed_values: [a, b]}\n  - {column: value, kind: numeric, missing_policy: structural_tokens, structural_absence_values: [NONE, '-999']}\n").unwrap();
        let report = describe_flat_csv(&path, &config).unwrap();
        let value = &report.groups[0].variables[1];
        assert_eq!(
            (value.observed, value.missing, value.structurally_absent),
            (4, 4, 8)
        );
        for metric in &report.conditional_numeric_generated_vs_holdout {
            assert_eq!(
                metric.holdout_observed
                    + metric.holdout_missing
                    + metric.holdout_structurally_absent,
                metric.holdout_patients
            );
            assert_eq!(
                metric.generated_observed
                    + metric.generated_missing
                    + metric.generated_structurally_absent,
                metric.generated_patients
            );
        }
        let mut learned_config = config.clone();
        learned_config.generator = FlatGenerator::LearnedMixedForest;
        learned_config.development_validation_proportion = Some(0.5);
        let learned = describe_flat_csv(&path, &learned_config).unwrap();
        assert_eq!(learned.learned_dependencies.len(), 1);
        assert_eq!(learned.conditional_numeric_generated_vs_holdout.len(), 2);
        assert!(!learned.development_validation[0].numeric_pairs.is_empty());
        for metric in &learned.development_validation[0].numeric_pairs {
            assert_eq!(
                metric.holdout_observed
                    + metric.holdout_missing
                    + metric.holdout_structurally_absent,
                metric.holdout_patients
            );
        }
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn conditional_numeric_quantiles_handle_missing_and_unshared_parent_states() {
        let first = vec!["x".to_owned(), "2".to_owned()];
        let second = vec!["y".to_owned(), "NA".to_owned()];
        let development = vec![&first, &second];
        let mut seed = 19;
        let mut backoffs = vec![BTreeMap::new(); 2];
        for _ in 0..64 {
            let sample = sample_flat_conditionals(
                &development,
                &[vec![], vec![0]],
                &mut seed,
                FlatSparsePolicy::Fail,
                &mut backoffs,
            )
            .unwrap();
            assert!(sample == first || sample == second);
        }
        let generated = vec![
            vec!["x".into(), "2".into()],
            vec!["x".into(), "4".into()],
            vec!["x".into(), "NA".into()],
            vec!["y".into(), "1".into()],
        ];
        let holdout = vec![
            vec!["x".into(), "4".into()],
            vec!["x".into(), "8".into()],
            vec!["z".into(), "".into()],
        ];
        let variable: FlatVariable =
            serde_yaml::from_str("column: child\nkind: numeric\n").unwrap();
        let metrics =
            flat_conditional_numeric_metrics("g", "parent", &generated, &holdout, 0, 1, &variable);
        let x = metrics.iter().find(|m| m.parent_value == "x").unwrap();
        assert_eq!(
            (
                x.generated_patients,
                x.generated_observed,
                x.generated_missing
            ),
            (3, 2, 1)
        );
        assert_eq!(
            (x.holdout_patients, x.holdout_observed, x.holdout_missing),
            (2, 2, 0)
        );
        assert_eq!(x.absolute_quantile_differences["median"], Some(3.0));
        assert!(metrics
            .iter()
            .find(|m| m.parent_value == "y")
            .unwrap()
            .absolute_quantile_differences["median"]
            .is_none());
        assert_eq!(
            metrics
                .iter()
                .find(|m| m.parent_value == "z")
                .unwrap()
                .holdout_missing,
            1
        );
    }

    #[test]
    fn conditional_numeric_child_is_generated_and_evaluated() {
        let dir =
            std::env::temp_dir().join(format!("janus-flat-numeric-parent-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("fixture.csv");
        std::fs::write(&path, "no,arm,a,b\na,LD,x,1\nb,LD,y,NA\nc,LD,x,2\nd,LD,y,3\ne,NoLD,x,1\nf,NoLD,y,NA\ng,NoLD,x,2\nh,NoLD,y,3\n").unwrap();
        let config: FlatCohortConfig = serde_yaml::from_str("schema_version: '1'\nallocation_column: arm\nallocation_values: [LD, NoLD]\ngenerator: conditional_empirical\ndependencies: [{parent: a, child: b}]\ndevelopment_proportion: 0.5\nbootstrap_replicates: 100\nthresholds: {max_missingness_rate_difference: 1.0, max_categorical_tvd: 1.0, max_numeric_quantile_difference: 10.0}\nvariables:\n  - {column: a, kind: categorical, allowed_values: [x, y]}\n  - {column: b, kind: numeric}\n").unwrap();
        let report = describe_flat_csv(&path, &config).unwrap();
        assert!(report.joint_generated_vs_holdout.is_empty());
        assert!(!report.conditional_numeric_generated_vs_holdout.is_empty());
        for metric in &report.conditional_numeric_generated_vs_holdout {
            assert_eq!(
                metric.generated_observed + metric.generated_missing,
                metric.generated_patients
            );
            assert_eq!(
                metric.holdout_observed + metric.holdout_missing,
                metric.holdout_patients
            );
        }
        assert_eq!(
            serde_yaml::to_string(&report).unwrap(),
            serde_yaml::to_string(&describe_flat_csv(&path, &config).unwrap()).unwrap()
        );
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn flat_generation_supports_explicit_row_key_and_nonbinary_groups() {
        let dir = std::env::temp_dir().join(format!("janus-flat-general-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("fixture.csv");
        std::fs::write(&path, "record_key,site,value\na,north,x\nb,north,y\nc,south,x\nd,south,y\ne,west,x\nf,west,y\n").unwrap();
        let mut config: FlatCohortConfig = serde_yaml::from_str("schema_version: '1'\nrow_key_column: record_key\nallocation_column: site\nallocation_values: [north, south, west]\ngenerator: independent_marginals\ndevelopment_proportion: 0.5\nbootstrap_replicates: 100\nthresholds: {max_missingness_rate_difference: 1.0, max_categorical_tvd: 1.0, max_numeric_quantile_difference: 1.0}\nvariables:\n  - {column: value, kind: categorical, allowed_values: [x, y]}\n").unwrap();
        let report = describe_flat_csv(&path, &config).unwrap();
        assert_eq!(report.total_patients, 6);
        assert_eq!(report.groups.len(), 3);
        assert!(report.split.groups.values().all(|counts| *counts == (1, 1)));
        assert_eq!(report.generated.len(), 3);
        assert!(!serde_yaml::to_string(&report)
            .unwrap()
            .contains("record_key"));
        config.variables[0].column = "record_key".into();
        assert!(describe_flat_csv(&path, &config).is_err());
        config.variables[0].column = "value".into();
        std::fs::write(&path, "record_key,site,value\na,north,x\nb,north,y\n").unwrap();
        config.allocation_values = vec!["north".into()];
        assert_eq!(describe_flat_csv(&path, &config).unwrap().groups.len(), 1);
        std::fs::write(&path, "record_key,site,value\na,north,x\na,north,y\n").unwrap();
        assert!(describe_flat_csv(&path, &config).is_err());
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn generator_seed_changes_samples_without_changing_split_or_fit_population() {
        let dir = std::env::temp_dir().join(format!("janus-flat-seeds-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("fixture.csv");
        let mut source = String::from("no,arm,value\n");
        for group in ["LD", "NoLD"] {
            for i in 0..40 {
                source.push_str(&format!(
                    "{group}{i},{group},{}\n",
                    if i % 2 == 0 { "x" } else { "y" }
                ));
            }
        }
        std::fs::write(&path, source).unwrap();
        let mut config: FlatCohortConfig = serde_yaml::from_str("schema_version: '1'\nallocation_column: arm\nallocation_values: [LD, NoLD]\ngenerator: independent_marginals\nsplit_seed: 41\ngenerator_seed: 1\ndevelopment_proportion: 0.5\nbootstrap_replicates: 100\nthresholds: {max_missingness_rate_difference: 1.0, max_categorical_tvd: 1.0, max_numeric_quantile_difference: 1.0}\nvariables:\n  - {column: value, kind: categorical, allowed_values: [x, y]}\n").unwrap();
        let first = describe_flat_csv(&path, &config).unwrap();
        config.generator_seed = Some(2);
        let second = describe_flat_csv(&path, &config).unwrap();
        assert_eq!(
            serde_yaml::to_string(&first.split).unwrap(),
            serde_yaml::to_string(&second.split).unwrap()
        );
        assert_eq!(
            serde_yaml::to_string(&first.partitions).unwrap(),
            serde_yaml::to_string(&second.partitions).unwrap()
        );
        assert_ne!(
            serde_yaml::to_string(&first.generated).unwrap(),
            serde_yaml::to_string(&second.generated).unwrap()
        );
        assert_eq!(
            (
                first.benchmark.generator_seed,
                second.benchmark.generator_seed
            ),
            (1, 2)
        );
        config.generator_seed = None;
        let default_seed = describe_flat_csv(&path, &config).unwrap();
        config.generator_seed = Some(config.split_seed);
        let explicit_seed = describe_flat_csv(&path, &config).unwrap();
        assert_eq!(
            serde_yaml::to_string(&default_seed.generated).unwrap(),
            serde_yaml::to_string(&explicit_seed.generated).unwrap()
        );
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn conditional_model_preserves_learned_pair_and_missing_states() {
        let first = vec!["x".to_owned(), "NA".to_owned(), "a".to_owned()];
        let second = vec!["y".to_owned(), "yes".to_owned(), "b".to_owned()];
        let rows = vec![&first, &second];
        let mut seed = 42;
        let samples: Vec<_> = (0..200)
            .map(|_| {
                sample_flat_conditionals(
                    &rows,
                    &[vec![], vec![0], vec![]],
                    &mut seed,
                    FlatSparsePolicy::Fail,
                    &mut vec![BTreeMap::new(); 3],
                )
                .unwrap()
            })
            .collect();
        assert!(samples
            .iter()
            .all(|row| (row[0] == "x" && row[1] == "NA") || (row[0] == "y" && row[1] == "yes")));
        assert!(samples.iter().any(|row| row != &first && row != &second));
        let holdout = vec![first.clone(), second.clone()];
        assert_eq!(flat_joint_tvd(&holdout, &holdout, 0, 1), Some(0.0));
        assert_eq!(flat_joint_tvd(&holdout, &[], 0, 1), None);
        let swapped = vec![
            vec!["x".into(), "yes".into()],
            vec!["y".into(), "NA".into()],
        ];
        assert_eq!(flat_joint_tvd(&holdout, &swapped, 0, 1), Some(1.0));
    }

    #[test]
    fn conditional_model_rejects_unsupported_multi_parent_combinations() {
        let first = vec!["a".to_owned(), "x".to_owned(), "one".to_owned()];
        let second = vec!["b".to_owned(), "y".to_owned(), "two".to_owned()];
        let rows = vec![&first, &second];
        let mut rejected = false;
        for seed in 0..100 {
            let mut state = seed;
            match sample_flat_conditionals(
                &rows,
                &[vec![], vec![], vec![0, 1]],
                &mut state,
                FlatSparsePolicy::Fail,
                &mut vec![BTreeMap::new(); 3],
            ) {
                Ok(row) => assert!(row == first || row == second),
                Err(error) => {
                    assert!(error
                        .to_string()
                        .contains("unsupported generated parent combination"));
                    rejected = true;
                }
            }
        }
        assert!(rejected);
    }

    #[test]
    fn opt_in_backoff_drops_last_parent_and_counts_fallbacks() {
        let first = vec!["a".to_owned(), "x".to_owned(), "one".to_owned()];
        let second = vec!["b".to_owned(), "y".to_owned(), "two".to_owned()];
        let rows = vec![&first, &second];
        let mut counts = vec![BTreeMap::new(); 3];
        for seed in 0..100 {
            let mut state = seed;
            let row = sample_flat_conditionals(
                &rows,
                &[vec![], vec![], vec![0, 1]],
                &mut state,
                FlatSparsePolicy::DropLastParent,
                &mut counts,
            )
            .unwrap();
            assert!((row[0] == "a" && row[2] == "one") || (row[0] == "b" && row[2] == "two"));
        }
        assert!(counts[2].get(&1).copied().unwrap_or(0) > 0);
        assert!(counts[0].is_empty() && counts[1].is_empty());
    }

    #[test]
    fn sparse_backoff_is_reported_end_to_end() {
        let dir = std::env::temp_dir().join(format!("janus-flat-backoff-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("fixture.csv");
        let first = vec!["a".to_owned(), "x".to_owned(), "one".to_owned()];
        let second = vec!["b".to_owned(), "y".to_owned(), "two".to_owned()];
        let rows = vec![&first, &second];
        let mut source = String::from("no,arm,a,b,c\n");
        for group in ["LD", "NoLD"] {
            let ids: Vec<_> = (0..4).map(|i| format!("{group}{i}")).collect();
            let mut sorted = ids.clone();
            sorted.sort_by_key(|id| (flat_hash(41, id), id.clone()));
            for (i, id) in sorted.iter().enumerate() {
                let row = if i == 1 { "b,y,two" } else { "a,x,one" };
                source.push_str(&format!("{id},{group},{row}\n"));
            }
        }
        std::fs::write(&path, source).unwrap();
        let seed = (0..100)
            .find(|seed| {
                let mut state = *seed ^ flat_hash(*seed, "LD");
                let mut counts = vec![BTreeMap::new(); 3];
                for _ in 0..2 {
                    sample_flat_conditionals(
                        &rows,
                        &[vec![], vec![], vec![0, 1]],
                        &mut state,
                        FlatSparsePolicy::DropLastParent,
                        &mut counts,
                    )
                    .unwrap();
                }
                !counts[2].is_empty()
            })
            .unwrap();
        let mut config: FlatCohortConfig = serde_yaml::from_str("schema_version: '1'\nallocation_column: arm\nallocation_values: [LD, NoLD]\ngenerator: conditional_categorical\nsparse_policy: drop_last_parent\ndependencies: [{parent: a, child: c}, {parent: b, child: c}]\nsplit_seed: 41\ndevelopment_proportion: 0.5\nbootstrap_replicates: 100\nthresholds: {max_missingness_rate_difference: 1.0, max_categorical_tvd: 1.0, max_numeric_quantile_difference: 1.0}\nvariables:\n  - {column: a, kind: categorical, allowed_values: [a, b]}\n  - {column: b, kind: categorical, allowed_values: [x, y]}\n  - {column: c, kind: categorical, allowed_values: [one, two]}\n").unwrap();
        config.generator_seed = Some(seed);
        let report = describe_flat_csv(&path, &config).unwrap();
        let fallback = report
            .sparse_backoffs
            .iter()
            .find(|r| r.allocation == "LD" && r.child == "c")
            .unwrap();
        assert!(fallback.dropped_parent_counts[&1] > 0);
        assert_eq!(report.family_generated_vs_holdout.len(), 2);
        assert_eq!(
            report.benchmark.sparse_policy,
            FlatSparsePolicy::DropLastParent
        );
        assert_eq!(
            serde_yaml::to_string(&report).unwrap(),
            serde_yaml::to_string(&describe_flat_csv(&path, &config).unwrap()).unwrap()
        );
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn family_distance_detects_higher_order_difference_with_matching_pairs() {
        let convert = |rows: &[(&str, &str, &str)]| {
            rows.iter()
                .map(|(a, b, c)| vec![a.to_string(), b.to_string(), c.to_string()])
                .collect::<Vec<_>>()
        };
        let even = convert(&[
            ("0", "0", "0"),
            ("0", "1", "1"),
            ("1", "0", "1"),
            ("1", "1", "0"),
        ]);
        let odd = convert(&[
            ("0", "0", "1"),
            ("0", "1", "0"),
            ("1", "0", "0"),
            ("1", "1", "1"),
        ]);
        assert_eq!(flat_joint_tvd(&even, &odd, 0, 2), Some(0.0));
        assert_eq!(flat_joint_tvd(&even, &odd, 1, 2), Some(0.0));
        assert_eq!(flat_family_tvd(&even, &odd, &[0, 1, 2]), Some(1.0));
        assert_eq!(flat_family_tvd(&even, &[], &[0, 1, 2]), None);
    }

    #[test]
    fn multiple_parents_are_accepted_but_duplicate_edges_are_rejected() {
        let dir = std::env::temp_dir().join(format!("janus-flat-parents-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("fixture.csv");
        std::fs::write(
            &path,
            "no,arm,a,b,c\na,LD,x,y,z\nb,LD,x,y,z\nc,NoLD,x,y,z\nd,NoLD,x,y,z\n",
        )
        .unwrap();
        let mut config: FlatCohortConfig = serde_yaml::from_str("schema_version: '1'\nallocation_column: arm\nallocation_values: [LD, NoLD]\ngenerator: conditional_categorical\ndependencies: [{parent: a, child: c}, {parent: b, child: c}]\ndevelopment_proportion: 0.5\nbootstrap_replicates: 100\nthresholds: {max_missingness_rate_difference: 1.0, max_categorical_tvd: 1.0, max_numeric_quantile_difference: 1.0}\nvariables:\n  - {column: a, kind: categorical, allowed_values: [x]}\n  - {column: b, kind: categorical, allowed_values: [y]}\n  - {column: c, kind: categorical, allowed_values: [z]}\n").unwrap();
        let report = describe_flat_csv(&path, &config).unwrap();
        assert_eq!(report.joint_generated_vs_holdout.len(), 4);
        assert_eq!(report.family_generated_vs_holdout.len(), 2);
        assert!(report.sparse_backoffs.is_empty());
        config.dependencies.push(config.dependencies[0].clone());
        assert!(describe_flat_csv(&path, &config).is_err());
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn conditional_configuration_reports_joint_holdout_and_rejects_invalid_graph() {
        let dir =
            std::env::temp_dir().join(format!("janus-flat-conditional-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("fixture.csv");
        std::fs::write(&path, "no,arm,a,b\na,LD,x,NA\nb,LD,y,yes\nc,LD,x,NA\nd,LD,y,yes\ne,NoLD,x,NA\nf,NoLD,y,yes\ng,NoLD,x,NA\nh,NoLD,y,yes\n").unwrap();
        let mut config: FlatCohortConfig = serde_yaml::from_str("schema_version: '1'\nallocation_column: arm\nallocation_values: [LD, NoLD]\ngenerator: conditional_categorical\ndependencies: [{parent: a, child: b}]\ndevelopment_proportion: 0.5\nbootstrap_replicates: 100\nthresholds: {max_missingness_rate_difference: 1.0, max_categorical_tvd: 1.0, max_numeric_quantile_difference: 1.0}\nvariables:\n  - {column: a, kind: categorical, allowed_values: [x, y]}\n  - {column: b, kind: categorical, allowed_values: [yes]}\n").unwrap();
        let report = describe_flat_csv(&path, &config).unwrap();
        assert_eq!(report.joint_generated_vs_holdout.len(), 2);
        assert!(report
            .joint_generated_vs_holdout
            .iter()
            .all(|m| m.generated_patients == 2
                && m.holdout_patients == 2
                && m.total_variation_distance.is_some()));
        assert_eq!(
            serde_yaml::to_string(&report).unwrap(),
            serde_yaml::to_string(&describe_flat_csv(&path, &config).unwrap()).unwrap()
        );
        let mut ranked = ["a", "b", "c", "d"];
        ranked.sort_by_key(|id| (flat_hash(config.split_seed, id), *id));
        let held_out = ranked[3];
        let source = std::fs::read_to_string(&path).unwrap();
        let changed = source
            .lines()
            .map(|line| {
                if line.starts_with(&format!("{held_out},")) {
                    if line.ends_with(",NA") {
                        line.replace(",NA", ",yes")
                    } else {
                        line.replace(",yes", ",NA")
                    }
                } else {
                    line.to_owned()
                }
            })
            .collect::<Vec<_>>()
            .join("\n")
            + "\n";
        std::fs::write(&path, changed).unwrap();
        let holdout_changed = describe_flat_csv(&path, &config).unwrap();
        assert_eq!(
            serde_yaml::to_string(&report.generated).unwrap(),
            serde_yaml::to_string(&holdout_changed.generated).unwrap()
        );
        assert_ne!(
            serde_yaml::to_string(&report.partitions[1]).unwrap(),
            serde_yaml::to_string(&holdout_changed.partitions[1]).unwrap()
        );
        config.generator = FlatGenerator::IndependentMarginals;
        assert_eq!(
            describe_flat_csv(&path, &config)
                .unwrap()
                .joint_generated_vs_holdout
                .len(),
            2
        );
        config.generator = FlatGenerator::ConditionalCategorical;
        config.dependencies[0] = FlatDependency {
            parent: "b".into(),
            child: "a".into(),
        };
        assert!(describe_flat_csv(&path, &config).is_err());
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn marginal_generator_recombines_fitted_values_and_replays() {
        let first = vec!["a".to_owned(), "1".to_owned()];
        let second = vec!["b".to_owned(), "NA".to_owned()];
        let rows = vec![&first, &second];
        let mut seed = 41;
        let samples: Vec<_> = (0..64)
            .map(|_| sample_flat_marginals(&rows, 2, &mut seed))
            .collect();
        assert!(samples.iter().any(|row| row != &first && row != &second));
        assert!(samples
            .iter()
            .all(|row| ["a", "b"].contains(&row[0].as_str())
                && ["1", "NA"].contains(&row[1].as_str())));
        let mut replay_seed = 41;
        assert_eq!(
            samples,
            (0..64)
                .map(|_| sample_flat_marginals(&rows, 2, &mut replay_seed))
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn marginal_fit_does_not_use_holdout_values() {
        let dir = std::env::temp_dir().join(format!(
            "janus-flat-marginal-holdout-{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("fixture.csv");
        let mut ids = vec!["a", "b", "c", "d"];
        ids.sort_by_key(|id| (flat_hash(41, id), *id));
        let holdout_id = ids[3];
        let make_input = |holdout_value: &str| {
            let mut text = String::from("no,arm,value\n");
            for id in &ids {
                text.push_str(&format!(
                    "{id},LD,{}\n",
                    if *id == holdout_id {
                        holdout_value
                    } else {
                        "x"
                    }
                ));
            }
            text.push_str("e,NoLD,x\nf,NoLD,x\n");
            text
        };
        let config: FlatCohortConfig = serde_yaml::from_str("schema_version: '1'\nallocation_column: arm\nallocation_values: [LD, NoLD]\ngenerator: independent_marginals\nsplit_seed: 41\ndevelopment_proportion: 0.5\nbootstrap_replicates: 100\nthresholds: {max_missingness_rate_difference: 1.0, max_categorical_tvd: 1.0, max_numeric_quantile_difference: 1.0}\nvariables:\n  - {column: value, kind: categorical, allowed_values: [x, y]}\n").unwrap();
        std::fs::write(&path, make_input("x")).unwrap();
        let first = describe_flat_csv(&path, &config).unwrap();
        std::fs::write(&path, make_input("y")).unwrap();
        let changed = describe_flat_csv(&path, &config).unwrap();
        assert_eq!(
            serde_yaml::to_string(&first.generated).unwrap(),
            serde_yaml::to_string(&changed.generated).unwrap()
        );
        assert_ne!(
            serde_yaml::to_string(&first.generated_vs_holdout).unwrap(),
            serde_yaml::to_string(&changed.generated_vs_holdout).unwrap()
        );
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn flat_summary_reports_only_aggregate_group_metrics() {
        let dir = std::env::temp_dir().join(format!("janus-flat-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("fixture.csv");
        std::fs::write(
            &path,
            "no,arm,score,state\na,LD,1,x\nb,LD,NA,y\nc,NoLD,3,x\n",
        )
        .unwrap();
        let config = FlatCohortConfig {
            schema_version: "1".into(),
            allocation_column: "arm".into(),
            allocation_values: vec!["LD".into(), "NoLD".into()],
            row_key_column: "no".into(),
            generator: FlatGenerator::RowBootstrap,
            dependencies: vec![],
            numeric_associations: vec![],
            joint_families: vec![],
            utility_tasks: vec![],
            exact_match_diagnostics: false,
            numeric_smoothing: vec![],
            numeric_interpolation: vec![],
            sparse_policy: FlatSparsePolicy::Fail,
            split_seed: 41,
            generator_seed: None,
            development_proportion: 0.67,
            development_validation_proportion: None,
            development_validation_seed: None,
            bootstrap_replicates: 100,
            bootstrap_confidence_level: 0.95,
            expected_source_columns: vec![
                "no".into(),
                "arm".into(),
                "score".into(),
                "state".into(),
            ],
            thresholds: FlatThresholds {
                max_missingness_rate_difference: 1.0,
                max_categorical_tvd: 1.0,
                max_numeric_quantile_difference: 1000.0,
            },
            variables: vec![
                FlatVariable {
                    column: "score".into(),
                    kind: "numeric".into(),
                    missing_policy: "missing".into(),
                    structural_absence_values: vec![],
                    allowed_values: vec![],
                },
                FlatVariable {
                    column: "state".into(),
                    kind: "categorical".into(),
                    missing_policy: "missing".into(),
                    structural_absence_values: vec![],
                    allowed_values: vec!["x".into(), "y".into()],
                },
            ],
        };
        let report = describe_flat_csv(&path, &config).unwrap();
        assert_eq!(report.total_patients, 3);
        assert_eq!(
            report.generated.iter().map(|g| g.patients).sum::<usize>(),
            report.split.holdout_patients
        );
        assert_eq!(report.generated_vs_holdout.len(), 4);
        let ld_metric = &report
            .fidelity
            .iter()
            .find(|g| g.group == "LD")
            .unwrap()
            .variables[0];
        assert!(ld_metric.bootstrap_intervals["missingness"].is_some());
        assert_eq!(ld_metric.valid_bootstrap_replicates["missingness"], 100);
        assert_eq!(
            serde_yaml::to_string(&report).unwrap(),
            serde_yaml::to_string(&describe_flat_csv(&path, &config).unwrap()).unwrap()
        );
        assert_eq!(
            report.benchmark.passed + report.benchmark.failed + report.benchmark.unavailable,
            4
        );
        assert_eq!(
            report.split.development_patients + report.split.holdout_patients,
            3
        );
        assert_eq!(report.split.groups["LD"], (1, 1));
        let ld_fidelity = report.fidelity.iter().find(|g| g.group == "LD").unwrap();
        assert_eq!(
            ld_fidelity.variables[0].absolute_quantile_differences["median"],
            None
        );
        assert_eq!(
            ld_fidelity.variables[1].categorical_total_variation_distance,
            Some(1.0)
        );
        let nold_fidelity = report.fidelity.iter().find(|g| g.group == "NoLD").unwrap();
        assert_eq!(
            nold_fidelity.variables[0].absolute_quantile_differences["median"],
            None
        );
        assert_eq!(
            report.partitions[0].patients + report.partitions[1].patients,
            3
        );
        for partition in &report.partitions {
            for group in &partition.groups {
                for variable in &group.variables {
                    assert_eq!(
                        variable.observed + variable.missing + variable.structurally_absent,
                        group.patients
                    );
                }
            }
        }
        assert_eq!(report.groups[0].patients, 2);
        assert_eq!(report.groups[0].variables[0].observed, 1);
        assert_eq!(report.groups[0].variables[0].missing, 1);
        assert_eq!(report.groups[0].variables[0].quantiles["median"], Some(1.0));
        let serialized = serde_yaml::to_string(&report).unwrap();
        assert!(!serialized.contains("patient_id"));
        assert!(!serialized.contains("source_rows"));
        let mut marginal_config = config.clone();
        marginal_config.generator = FlatGenerator::IndependentMarginals;
        let marginal = describe_flat_csv(&path, &marginal_config).unwrap();
        assert!(marginal
            .benchmark
            .method
            .contains("independent empirical marginal"));
        assert_eq!(
            marginal.generated.iter().map(|g| g.patients).sum::<usize>(),
            marginal.split.holdout_patients
        );
        assert_eq!(
            serde_yaml::to_string(&marginal).unwrap(),
            serde_yaml::to_string(&describe_flat_csv(&path, &marginal_config).unwrap()).unwrap()
        );
        assert!(describe_flat_csv(
            &path,
            &FlatCohortConfig {
                allocation_values: vec!["LD".into(), "NoLD".into()],
                ..config
            }
        )
        .is_ok());
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn flat_summary_rejects_duplicate_ids_unknown_groups_and_duplicate_headers() {
        let dir = std::env::temp_dir().join(format!("janus-flat-invalid-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("fixture.csv");
        let config = FlatCohortConfig {
            schema_version: "1".into(),
            allocation_column: "arm".into(),
            allocation_values: vec!["LD".into(), "NoLD".into()],
            row_key_column: "no".into(),
            generator: FlatGenerator::RowBootstrap,
            dependencies: vec![],
            numeric_associations: vec![],
            joint_families: vec![],
            utility_tasks: vec![],
            exact_match_diagnostics: false,
            numeric_smoothing: vec![],
            numeric_interpolation: vec![],
            sparse_policy: FlatSparsePolicy::Fail,
            split_seed: 41,
            generator_seed: None,
            development_proportion: 0.8,
            development_validation_proportion: None,
            development_validation_seed: None,
            bootstrap_replicates: 100,
            bootstrap_confidence_level: 0.95,
            expected_source_columns: vec!["no".into(), "arm".into(), "score".into()],
            thresholds: FlatThresholds {
                max_missingness_rate_difference: 0.25,
                max_categorical_tvd: 0.25,
                max_numeric_quantile_difference: 10.0,
            },
            variables: vec![FlatVariable {
                column: "score".into(),
                kind: "numeric".into(),
                missing_policy: "missing".into(),
                structural_absence_values: vec![],
                allowed_values: vec![],
            }],
        };
        std::fs::write(&path, "no,arm,score\na,LD,1\na,NoLD,2\n").unwrap();
        assert!(describe_flat_csv(&path, &config).is_err());
        std::fs::write(&path, "no,arm,score\na,other,1\n").unwrap();
        assert!(describe_flat_csv(&path, &config).is_err());
        std::fs::write(&path, "no,arm,score,score\na,LD,1,2\n").unwrap();
        assert!(describe_flat_csv(&path, &config).is_err());
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn flat_summary_separates_explicit_structural_tokens_from_missing() {
        let dir = std::env::temp_dir().join(format!("janus-flat-states-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("fixture.csv");
        std::fs::write(&path, "no,arm,value\na,LD,NA\nb,LD,NONE\nc,NoLD,4\n").unwrap();
        let config = FlatCohortConfig {
            schema_version: "1".into(),
            allocation_column: "arm".into(),
            allocation_values: vec!["LD".into(), "NoLD".into()],
            row_key_column: "no".into(),
            generator: FlatGenerator::RowBootstrap,
            dependencies: vec![],
            numeric_associations: vec![],
            joint_families: vec![],
            utility_tasks: vec![],
            exact_match_diagnostics: false,
            numeric_smoothing: vec![],
            numeric_interpolation: vec![],
            sparse_policy: FlatSparsePolicy::Fail,
            split_seed: 41,
            generator_seed: None,
            development_proportion: 0.5,
            development_validation_proportion: None,
            development_validation_seed: None,
            bootstrap_replicates: 100,
            bootstrap_confidence_level: 0.95,
            expected_source_columns: vec!["no".into(), "arm".into(), "value".into()],
            thresholds: FlatThresholds {
                max_missingness_rate_difference: 0.25,
                max_categorical_tvd: 0.25,
                max_numeric_quantile_difference: 10.0,
            },
            variables: vec![FlatVariable {
                column: "value".into(),
                kind: "categorical".into(),
                missing_policy: "categorical_tokens".into(),
                structural_absence_values: vec!["NONE".into()],
                allowed_values: vec!["4".into()],
            }],
        };
        let report = describe_flat_csv(&path, &config).unwrap();
        let ld = &report
            .groups
            .iter()
            .find(|g| g.group == "LD")
            .unwrap()
            .variables[0];
        assert_eq!((ld.missing, ld.structurally_absent, ld.observed), (1, 1, 0));
        for partition in &report.partitions {
            for group in &partition.groups {
                for variable in &group.variables {
                    assert_eq!(
                        variable.observed + variable.missing + variable.structurally_absent,
                        group.patients
                    );
                }
            }
        }
        let numeric = FlatCohortConfig {
            variables: vec![FlatVariable {
                column: "value".into(),
                kind: "numeric".into(),
                missing_policy: "categorical_tokens".into(),
                structural_absence_values: vec!["NONE".into()],
                allowed_values: vec![],
            }],
            ..config
        };
        assert!(describe_flat_csv(&path, &numeric).is_ok());
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn explicit_blank_and_na_absence_are_not_double_counted() {
        let dir =
            std::env::temp_dir().join(format!("janus-flat-absence-tokens-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("fixture.csv");
        std::fs::write(&path, "no,arm,value\na,LD,\nb,LD,NA\nc,LD,x\nd,LD,x\ne,NoLD,\nf,NoLD,NA\ng,NoLD,x\nh,NoLD,x\n").unwrap();
        let config: FlatCohortConfig = serde_yaml::from_str("schema_version: '1'\nallocation_column: arm\nallocation_values: [LD, NoLD]\ngenerator: independent_marginals\ndevelopment_proportion: 0.5\nbootstrap_replicates: 100\nthresholds: {max_missingness_rate_difference: 1.0, max_categorical_tvd: 1.0, max_numeric_quantile_difference: 1.0}\nvariables:\n  - {column: value, kind: categorical, missing_policy: categorical_tokens, structural_absence_values: ['', 'NA'], allowed_values: [x]}\n").unwrap();
        let report = describe_flat_csv(&path, &config).unwrap();
        for group in &report.groups {
            let variable = &group.variables[0];
            assert_eq!(
                (
                    variable.observed,
                    variable.missing,
                    variable.structurally_absent
                ),
                (2, 0, 2)
            );
        }
        for group in report
            .partitions
            .iter()
            .flat_map(|p| &p.groups)
            .chain(report.generated.iter())
        {
            let v = &group.variables[0];
            assert_eq!(
                v.observed + v.missing + v.structurally_absent,
                group.patients
            );
            assert_eq!(v.missing, 0);
        }
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn flat_bootstrap_marks_zero_observed_denominators_unavailable() {
        let categorical = FlatVariable {
            column: "state".into(),
            kind: "categorical".into(),
            missing_policy: "missing".into(),
            structural_absence_values: vec![],
            allowed_values: vec!["x".into()],
        };
        let (intervals, valid) = flat_bootstrap_intervals(
            &[vec!["NA".into()]],
            &[vec!["x".into()]],
            0,
            &categorical,
            100,
            0.95,
            99,
        );
        assert!(intervals["missingness"].is_some());
        assert_eq!(valid["missingness"], 100);
        assert!(intervals["tvd"].is_none());
        assert_eq!(valid["tvd"], 0);
        let numeric = FlatVariable {
            column: "score".into(),
            kind: "numeric".into(),
            missing_policy: "missing".into(),
            structural_absence_values: vec![],
            allowed_values: vec![],
        };
        let (intervals, valid) = flat_bootstrap_intervals(
            &[vec!["NA".into()]],
            &[vec!["1".into()]],
            0,
            &numeric,
            100,
            0.95,
            99,
        );
        for key in ["q25", "median", "q75"] {
            assert!(intervals[key].is_none());
            assert_eq!(valid[key], 0);
        }
    }

    #[test]
    fn flat_summary_rejects_source_schema_and_category_drift() {
        let dir = std::env::temp_dir().join(format!("janus-flat-schema-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("fixture.csv");
        std::fs::write(&path, "no,arm,state\na,LD,x\nb,NoLD,y\n").unwrap();
        let config = FlatCohortConfig {
            schema_version: "1".into(),
            allocation_column: "arm".into(),
            allocation_values: vec!["LD".into(), "NoLD".into()],
            row_key_column: "no".into(),
            generator: FlatGenerator::RowBootstrap,
            dependencies: vec![],
            numeric_associations: vec![],
            joint_families: vec![],
            utility_tasks: vec![],
            exact_match_diagnostics: false,
            numeric_smoothing: vec![],
            numeric_interpolation: vec![],
            sparse_policy: FlatSparsePolicy::Fail,
            split_seed: 2,
            generator_seed: None,
            development_proportion: 0.5,
            development_validation_proportion: None,
            development_validation_seed: None,
            bootstrap_replicates: 100,
            bootstrap_confidence_level: 0.95,
            expected_source_columns: vec!["no".into(), "arm".into(), "state".into()],
            thresholds: FlatThresholds {
                max_missingness_rate_difference: 0.25,
                max_categorical_tvd: 0.25,
                max_numeric_quantile_difference: 10.0,
            },
            variables: vec![FlatVariable {
                column: "state".into(),
                kind: "categorical".into(),
                missing_policy: "missing".into(),
                structural_absence_values: vec![],
                allowed_values: vec!["x".into(), "y".into()],
            }],
        };
        assert!(describe_flat_csv(&path, &config).is_ok());
        std::fs::write(&path, "no,arm,state,extra\na,LD,x,1\nb,NoLD,y,2\n").unwrap();
        assert!(describe_flat_csv(&path, &config).is_err());
        std::fs::write(&path, "no,arm,state\na,LD,z\nb,NoLD,y\n").unwrap();
        assert!(describe_flat_csv(&path, &config).is_err());
        std::fs::remove_dir_all(dir).unwrap();
    }
    use std::path::PathBuf;

    fn fictional_mapping() -> CsvMapping {
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../examples/mapping.yaml");
        serde_yaml::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
    }

    #[test]
    fn rejects_duplicate_ids_timing_and_direct_identifiers() {
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
        let input = std::fs::read_to_string(root.join("examples/source.csv")).unwrap();
        let path =
            std::env::temp_dir().join(format!("janus-csv-invalid-{}.csv", std::process::id()));
        let mapping = fictional_mapping();
        let mut unapproved = mapping.clone();
        unapproved.pre_deidentified = false;
        assert!(read_csv(&root.join("examples/source.csv"), &unapproved).is_err());
        let mut prohibited = mapping.clone();
        prohibited.data_use_permitted = false;
        assert!(read_csv(&root.join("examples/source.csv"), &prohibited).is_err());
        let mut invalid_mapping = mapping.clone();
        invalid_mapping.columns.remove("provenance");
        assert!(read_csv(&root.join("examples/source.csv"), &invalid_mapping).is_err());
        let duplicate = format!("{}{}\n", input, input.lines().nth(1).unwrap());
        std::fs::write(&path, duplicate).unwrap();
        assert!(read_csv(&path, &mapping).is_err());
        let direct = input.replacen("provenance", "provenance,email", 1);
        std::fs::write(&path, direct).unwrap();
        assert!(read_csv(&path, &mapping).is_err());
        let bad_value = input.replacen(",exposure,true,", ",exposure,not_boolean,", 1);
        std::fs::write(&path, bad_value).unwrap();
        assert!(read_csv(&path, &mapping).is_err());
        let invalid_integer = input.replacen(",baseline_severity,1,", ",baseline_severity,1.5,", 1);
        std::fs::write(&path, invalid_integer).unwrap();
        assert!(read_csv(&path, &mapping).is_err());
        let mut numeric_mapping = mapping.clone();
        numeric_mapping
            .study
            .variables
            .iter_mut()
            .find(|v| v.name == "baseline_severity")
            .unwrap()
            .kind = "number".into();
        let invalid_number = input.replacen(",baseline_severity,1,", ",baseline_severity,NaN,", 1);
        std::fs::write(&path, invalid_number).unwrap();
        assert!(read_csv(&path, &numeric_mapping).is_err());
        let conflicting = input.replacen(",exposure,true,,", ",exposure,true,unknown,", 1);
        std::fs::write(&path, conflicting).unwrap();
        assert!(read_csv(&path, &mapping).is_err());
        let no_provenance = input.replacen(",0,0,fictional_fixture", ",0,0,", 1);
        std::fs::write(&path, no_provenance).unwrap();
        assert!(read_csv(&path, &mapping).is_err());
        let bad_timing = input.replacen(",0,0,fictional_fixture", ",0,-1,fictional_fixture", 1);
        std::fs::write(&path, bad_timing).unwrap();
        assert!(read_csv(&path, &mapping).is_err());
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn typed_values_and_explicit_missing_survive_round_trip() {
        let mapping = fictional_mapping();
        let mut study = mapping.study.clone();
        study.variables.push(crate::Variable {
            name: "measurement".into(),
            kind: "number".into(),
            role: Some(crate::VariableRole::Baseline),
        });
        let fixture = Fixture {
            schema_version: "1".into(),
            observations: vec![],
            events: vec![
                Event {
                    patient_id: "p".into(),
                    episode_id: "e".into(),
                    id: "index".into(),
                    variable: study.index.clone(),
                    value: Some("true".into()),
                    missing: None,
                    event_day: 0,
                    recording_day: 0,
                    provenance: "fictional".into(),
                },
                Event {
                    patient_id: "p".into(),
                    episode_id: "e".into(),
                    id: "number".into(),
                    variable: "measurement".into(),
                    value: Some("1.2500".into()),
                    missing: None,
                    event_day: 0,
                    recording_day: 0,
                    provenance: "fictional".into(),
                },
                Event {
                    patient_id: "p".into(),
                    episode_id: "e".into(),
                    id: "empty".into(),
                    variable: "treatment_group".into(),
                    value: Some(String::new()),
                    missing: None,
                    event_day: 0,
                    recording_day: 0,
                    provenance: "fictional".into(),
                },
                Event {
                    patient_id: "p".into(),
                    episode_id: "e".into(),
                    id: "missing".into(),
                    variable: "baseline_severity".into(),
                    value: None,
                    missing: Some("unknown".into()),
                    event_day: 0,
                    recording_day: 0,
                    provenance: "fictional".into(),
                },
            ],
        };
        let path = std::env::temp_dir().join(format!("janus-typed-{}.parquet", std::process::id()));
        write_parquet(&fixture, &study, &path).unwrap();
        let reversed_path = std::env::temp_dir().join(format!(
            "janus-typed-reversed-{}.parquet",
            std::process::id()
        ));
        let mut reversed = fixture.clone();
        reversed.events.reverse();
        write_parquet(&reversed, &study, &reversed_path).unwrap();
        assert_eq!(
            std::fs::read(&path).unwrap(),
            std::fs::read(&reversed_path).unwrap()
        );
        std::fs::remove_file(reversed_path).unwrap();
        let result = read_parquet(&path, &study).unwrap();
        assert_eq!(
            result
                .events
                .iter()
                .find(|e| e.id == "number")
                .unwrap()
                .value
                .as_deref(),
            Some("1.2500")
        );
        assert_eq!(
            result
                .events
                .iter()
                .find(|e| e.id == "missing")
                .unwrap()
                .missing
                .as_deref(),
            Some("unknown")
        );
        assert!(!result
            .events
            .iter()
            .any(|e| e.variable == "baseline_comorbidity"));
        let csv_path =
            std::env::temp_dir().join(format!("janus-empty-value-{}.csv", std::process::id()));
        export_csv(&result, &csv_path).unwrap();
        let mut csv_mapping = mapping.clone();
        csv_mapping.study = study;
        csv_mapping
            .columns
            .insert("value_present".into(), "value_present".into());
        let replay = read_csv(&csv_path, &csv_mapping).unwrap();
        assert_eq!(
            replay
                .events
                .iter()
                .find(|e| e.id == "empty")
                .unwrap()
                .value
                .as_deref(),
            Some("")
        );
        std::fs::remove_file(csv_path).unwrap();
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn rejects_incompatible_study_and_malformed_parquet() {
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
        let mapping = fictional_mapping();
        let fixture = read_csv(&root.join("examples/source.csv"), &mapping).unwrap();
        let path = std::env::temp_dir().join(format!(
            "janus-invalid-parquet-{}.parquet",
            std::process::id()
        ));
        write_parquet(&fixture, &mapping.study, &path).unwrap();
        let mut incompatible = mapping.study.clone();
        incompatible
            .variables
            .iter_mut()
            .find(|v| v.name == "baseline_severity")
            .unwrap()
            .kind = "string".into();
        assert!(read_parquet(&path, &incompatible).is_err());
        std::fs::write(&path, b"not parquet").unwrap();
        assert!(read_parquet(&path, &mapping.study).is_err());
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn observation_metadata_round_trips() {
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
        let mapping = fictional_mapping();
        let fixture: Fixture = serde_yaml::from_str(
            &std::fs::read_to_string(root.join("examples/fixture-observation-v2.yaml")).unwrap(),
        )
        .unwrap();
        let path =
            std::env::temp_dir().join(format!("janus-observation-{}.parquet", std::process::id()));
        write_parquet(&fixture, &mapping.study, &path).unwrap();
        let result = read_parquet(&path, &mapping.study).unwrap();
        let mut expected = fixture
            .observations
            .iter()
            .map(|o| {
                (
                    &o.patient_id,
                    &o.episode_id,
                    o.start_day,
                    o.end_day,
                    &o.censoring_reason,
                )
            })
            .collect::<Vec<_>>();
        let mut actual = result
            .observations
            .iter()
            .map(|o| {
                (
                    &o.patient_id,
                    &o.episode_id,
                    o.start_day,
                    o.end_day,
                    &o.censoring_reason,
                )
            })
            .collect::<Vec<_>>();
        expected.sort();
        actual.sort();
        assert_eq!(expected, actual);
        let csv_path =
            std::env::temp_dir().join(format!("janus-observation-{}.csv", std::process::id()));
        export_csv(&result, &csv_path).unwrap();
        let mut csv_mapping = mapping.clone();
        for column in OBSERVATION {
            csv_mapping
                .columns
                .insert((*column).into(), (*column).into());
        }
        let replay = read_csv(&csv_path, &csv_mapping).unwrap();
        assert_eq!(
            crate::fixture_fingerprint(&result).unwrap(),
            crate::fixture_fingerprint(&replay).unwrap()
        );
        std::fs::remove_file(csv_path).unwrap();
        let mapped_path = std::env::temp_dir().join(format!(
            "janus-mapped-observation-{}.csv",
            std::process::id()
        ));
        for value in csv_mapping.columns.values_mut() {
            *value = format!("mapped_{value}");
        }
        export_mapped_csv(&result, &csv_mapping, &mapped_path).unwrap();
        let mapped_replay = read_csv(&mapped_path, &csv_mapping).unwrap();
        assert_eq!(
            crate::fixture_fingerprint(&result).unwrap(),
            crate::fixture_fingerprint(&mapped_replay).unwrap()
        );
        std::fs::remove_file(mapped_path).unwrap();
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn fictional_csv_round_trips_typed_parquet() {
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
        let mapping: CsvMapping = serde_yaml::from_str(
            &std::fs::read_to_string(root.join("examples/mapping.yaml")).unwrap(),
        )
        .unwrap();
        let fixture = read_csv(&root.join("examples/source.csv"), &mapping).unwrap();
        let path =
            std::env::temp_dir().join(format!("janus-tabular-test-{}.parquet", std::process::id()));
        write_parquet(&fixture, &mapping.study, &path).unwrap();
        let recovered = read_parquet(&path, &mapping.study).unwrap();
        assert_eq!(
            crate::fixture_fingerprint(&fixture).unwrap(),
            crate::fixture_fingerprint(&recovered).unwrap()
        );
        let report = inspect(&path, &mapping.study).unwrap();
        assert_eq!(report.source_rows, fixture.events.len() as i64);
        assert_eq!(report.patients, 2);
        assert_eq!(report.episodes, 3);
        let severity = report
            .variables
            .iter()
            .find(|v| v.variable == "baseline_severity")
            .unwrap();
        assert_eq!(severity.observed, 3);
        assert_eq!(severity.absent_episodes, 0);
        assert_eq!(severity.median, Some(1.0));
        assert!((severity.mean.unwrap() - 4.0 / 3.0).abs() < 1e-12);
        assert!((severity.standard_deviation.unwrap() - (1.0_f64 / 3.0).sqrt()).abs() < 1e-12);
        assert_eq!(severity.lower_quartile, Some(1.0));
        assert_eq!(severity.upper_quartile, Some(1.5));
        assert!(report
            .missing_reasons
            .iter()
            .any(|f| f.variable == "outcome" && f.value == "not_recorded"));
        assert!(report
            .categorical_frequencies
            .iter()
            .any(|f| f.variable == "treatment_group" && f.value == "standard"));
        assert_eq!(report.timing.event_minimum, Some(0));
        assert_eq!(report.timing.event_maximum, Some(410));
        assert_eq!(report.timing.recording_maximum, Some(410));
        assert_eq!(report.timing.recorded_before_event, 0);
        assert_eq!(report.timing.duplicate_event_ids, 0);
        let mut study_with_absent = mapping.study.clone();
        study_with_absent.variables.push(crate::Variable {
            name: "unrecorded".into(),
            kind: "string".into(),
            role: Some(crate::VariableRole::Baseline),
        });
        let with_absence = inspect(&path, &study_with_absent).unwrap();
        let absent = with_absence
            .variables
            .iter()
            .find(|v| v.variable == "unrecorded")
            .unwrap();
        assert_eq!(absent.absent_episodes, 3);
        assert_eq!(absent.missing, 0);
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn numeric_table_csv_preserves_missing_as_none_without_zero_imputation() {
        let path = std::env::temp_dir().join(format!(
            "janus-numeric-missing-csv-{}.csv",
            std::process::id()
        ));
        let csv_content = "\
age,score,note,all_na
20,0.0,alpha,NA
30,NA,beta,
40,,gamma,na
50,NaN,delta,NaN
60, 2.5 ,epsilon,
";
        std::fs::write(&path, csv_content).unwrap();

        let table = load_numeric_table(&path).unwrap();
        assert_eq!(table.columns, vec!["age", "score"]);
        assert_eq!(table.nrows(), 5);
        assert_eq!(table.ncols(), 2);

        // Verify missing values are None (NOT 0.0), while explicit 0.0 is Some(0.0)
        assert_eq!(table.data[[0, 1]], Some(0.0));
        assert_eq!(table.data[[1, 1]], None);
        assert_eq!(table.data[[2, 1]], None);
        assert_eq!(table.data[[3, 1]], None);
        assert_eq!(table.data[[4, 1]], Some(2.5));

        assert_eq!(table.observed_column(0), vec![20.0, 30.0, 40.0, 50.0, 60.0]);
        assert_eq!(table.observed_column(1), vec![0.0, 2.5]);
        assert_eq!(table.missing_count(0), 0);
        assert_eq!(table.missing_count(1), 3);

        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn numeric_table_parquet_preserves_null_and_nan_as_none_and_rejects_inf() {
        use arrow_array::{ArrayRef, Float64Array, Int32Array, RecordBatch, StringArray};
        use arrow_schema::{DataType, Field, Schema};
        use parquet::arrow::ArrowWriter;
        use std::sync::Arc;

        let path = std::env::temp_dir().join(format!(
            "janus-numeric-missing-parquet-{}.parquet",
            std::process::id()
        ));
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, true),
            Field::new("score", DataType::Float64, true),
            Field::new("label", DataType::Utf8, true),
        ]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int32Array::from(vec![Some(1), None, Some(3)])) as ArrayRef,
                Arc::new(Float64Array::from(vec![
                    Some(0.0),
                    None,
                    Some(f64::NAN),
                ])) as ArrayRef,
                Arc::new(StringArray::from(vec![Some("a"), Some("b"), None])) as ArrayRef,
            ],
        )
        .unwrap();
        let mut writer = ArrowWriter::try_new(std::fs::File::create(&path).unwrap(), schema.clone(), None).unwrap();
        writer.write(&batch).unwrap();
        writer.close().unwrap();

        let table = load_numeric_table(&path).unwrap();
        assert_eq!(table.columns, vec!["id", "score"]);
        assert_eq!(table.data[[0, 0]], Some(1.0));
        assert_eq!(table.data[[1, 0]], None);
        assert_eq!(table.data[[2, 0]], Some(3.0));
        assert_eq!(table.data[[0, 1]], Some(0.0));
        assert_eq!(table.data[[1, 1]], None);
        assert_eq!(table.data[[2, 1]], None);
        assert_eq!(table.observed_column(1), vec![0.0]);
        assert_eq!(table.missing_count(1), 2);
        std::fs::remove_file(&path).unwrap();

        // Infinite float in Parquet must return an error
        let inf_path = std::env::temp_dir().join(format!(
            "janus-numeric-inf-parquet-{}.parquet",
            std::process::id()
        ));
        let inf_schema = Arc::new(Schema::new(vec![Field::new(
            "score",
            DataType::Float64,
            true,
        )]));
        let inf_batch = RecordBatch::try_new(
            inf_schema.clone(),
            vec![Arc::new(Float64Array::from(vec![Some(1.0), Some(f64::INFINITY)])) as ArrayRef],
        )
        .unwrap();
        let mut inf_writer =
            ArrowWriter::try_new(std::fs::File::create(&inf_path).unwrap(), inf_schema, None)
                .unwrap();
        inf_writer.write(&inf_batch).unwrap();
        inf_writer.close().unwrap();
        assert!(load_numeric_table(&inf_path).is_err());
        std::fs::remove_file(&inf_path).unwrap();
    }

    #[test]
    fn numeric_table_standardized_features_emit_missing_indicators_only_when_missing() {
        let real_complete = NumericTable {
            columns: vec!["x".into(), "y".into()],
            data: ndarray::Array2::from_shape_vec(
                (2, 2),
                vec![Some(0.0), Some(10.0), Some(2.0), Some(20.0)],
            )
            .unwrap(),
        };
        let synth_complete = NumericTable {
            columns: vec!["x".into(), "y".into()],
            data: ndarray::Array2::from_shape_vec(
                (2, 2),
                vec![Some(1.0), Some(15.0), Some(3.0), Some(25.0)],
            )
            .unwrap(),
        };
        let (rx_c, sx_c) =
            NumericTable::build_standardized_features_with_indicators(&real_complete, &synth_complete)
                .unwrap();
        assert_eq!(rx_c.ncols(), 2);
        assert_eq!(sx_c.ncols(), 2);

        // Now synthetic has None in column `x`, while real has Some(0.0) and Some(2.0)
        let synth_with_na = NumericTable {
            columns: vec!["x".into(), "y".into()],
            data: ndarray::Array2::from_shape_vec(
                (2, 2),
                vec![None, Some(15.0), Some(0.0), Some(25.0)],
            )
            .unwrap(),
        };
        let (rx_m, sx_m) =
            NumericTable::build_standardized_features_with_indicators(&real_complete, &synth_with_na)
                .unwrap();
        // Column 0 (x std), Column 1 (y std), Column 2 (x is_missing)
        assert_eq!(rx_m.ncols(), 3);
        assert_eq!(sx_m.ncols(), 3);
        assert_eq!(rx_m[[0, 2]], 0.0);
        assert_eq!(rx_m[[1, 2]], 0.0);
        // Synthetic row 0 is None -> standardized residual 0.0 AND indicator 1.0
        assert_eq!(sx_m[[0, 0]], 0.0);
        assert_eq!(sx_m[[0, 2]], 1.0);
        // Synthetic row 1 is Some(0.0) -> standardized (0 - 1)/1 = -1.0 AND indicator 0.0
        assert!((sx_m[[1, 0]] - (-1.0)).abs() < 1e-12);
        assert_eq!(sx_m[[1, 2]], 0.0);

        // Zero observed values in `real` uses fallback mean=0.0, std=1.0 and emits indicator
        let real_all_missing = NumericTable {
            columns: vec!["x".into()],
            data: ndarray::Array2::from_shape_vec((2, 1), vec![None, None]).unwrap(),
        };
        let synth_some = NumericTable {
            columns: vec!["x".into()],
            data: ndarray::Array2::from_shape_vec((2, 1), vec![Some(2.0), None]).unwrap(),
        };
        let (rx_z, sx_z) = NumericTable::build_standardized_features_with_indicators(
            &real_all_missing,
            &synth_some,
        )
        .unwrap();
        assert_eq!(rx_z.ncols(), 2);
        assert_eq!(rx_z[[0, 0]], 0.0);
        assert_eq!(rx_z[[0, 1]], 1.0);
        assert_eq!(sx_z[[0, 0]], 2.0);
        assert_eq!(sx_z[[0, 1]], 0.0);
        assert_eq!(sx_z[[1, 0]], 0.0);
        assert_eq!(sx_z[[1, 1]], 1.0);
    }

    #[test]
    fn load_forensic_cohort_loads_mixed_csv_and_parquet_preserving_missing_values() {
        use arrow_array::{ArrayRef, BooleanArray, Float64Array, Int64Array, RecordBatch, StringArray};
        use arrow_schema::{DataType, Field, Schema};
        use parquet::arrow::ArrowWriter;
        use std::sync::Arc;

        let csv_path = std::env::temp_dir().join(format!(
            "janus-forensic-cohort-{}.csv",
            std::process::id()
        ));
        let csv_content = "\
age,score,grade,flag
50,0.0,high,true
,NA,na,false
60,NaN,,
";
        std::fs::write(&csv_path, csv_content).unwrap();

        let cohort = load_forensic_cohort(&csv_path, "csv-study").unwrap();
        cohort.validate_keys().unwrap();
        assert_eq!(cohort.rows.len(), 3);
        assert_eq!(cohort.rows[0].patient_key, "csv-study-row-000000");
        assert_eq!(cohort.rows[1].patient_key, "csv-study-row-000001");
        assert_eq!(cohort.rows[2].patient_key, "csv-study-row-000002");

        assert_eq!(cohort.rows[0].numeric_fields.get("age"), Some(&Some(50.0)));
        assert_eq!(cohort.rows[1].numeric_fields.get("age"), Some(&None));
        assert_eq!(cohort.rows[0].numeric_fields.get("score"), Some(&Some(0.0)));
        assert_eq!(cohort.rows[1].numeric_fields.get("score"), Some(&None));
        assert_eq!(cohort.rows[2].numeric_fields.get("score"), Some(&None));
        assert_eq!(
            cohort.rows[0].categorical_fields.get("grade"),
            Some(&Some("high".to_string()))
        );
        assert_eq!(cohort.rows[1].categorical_fields.get("grade"), Some(&None));
        assert_eq!(cohort.rows[2].categorical_fields.get("grade"), Some(&None));
        assert_eq!(
            cohort.rows[0].categorical_fields.get("flag"),
            Some(&Some("true".to_string()))
        );
        assert_eq!(cohort.rows[2].categorical_fields.get("flag"), Some(&None));
        std::fs::remove_file(&csv_path).unwrap();

        // Mixed Parquet
        let pq_path = std::env::temp_dir().join(format!(
            "janus-forensic-cohort-{}.parquet",
            std::process::id()
        ));
        let schema = Arc::new(Schema::new(vec![
            Field::new("age", DataType::Int64, true),
            Field::new("score", DataType::Float64, true),
            Field::new("grade", DataType::Utf8, true),
            Field::new("eligible", DataType::Boolean, true),
        ]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int64Array::from(vec![Some(45), None, Some(70)])) as ArrayRef,
                Arc::new(Float64Array::from(vec![
                    Some(0.0),
                    Some(f64::NAN),
                    None,
                ])) as ArrayRef,
                Arc::new(StringArray::from(vec![Some("low"), Some("NA"), Some("")])) as ArrayRef,
                Arc::new(BooleanArray::from(vec![Some(true), None, Some(false)])) as ArrayRef,
            ],
        )
        .unwrap();
        let mut writer =
            ArrowWriter::try_new(std::fs::File::create(&pq_path).unwrap(), schema, None).unwrap();
        writer.write(&batch).unwrap();
        writer.close().unwrap();

        let pq_cohort = load_forensic_cohort(&pq_path, "pq-study").unwrap();
        pq_cohort.validate_keys().unwrap();
        assert_eq!(pq_cohort.rows.len(), 3);
        assert_eq!(pq_cohort.rows[0].patient_key, "pq-study-row-000000");
        assert_eq!(pq_cohort.rows[0].numeric_fields.get("age"), Some(&Some(45.0)));
        assert_eq!(pq_cohort.rows[1].numeric_fields.get("age"), Some(&None));
        assert_eq!(pq_cohort.rows[0].numeric_fields.get("score"), Some(&Some(0.0)));
        assert_eq!(pq_cohort.rows[1].numeric_fields.get("score"), Some(&None));
        assert_eq!(pq_cohort.rows[2].numeric_fields.get("score"), Some(&None));
        assert_eq!(
            pq_cohort.rows[0].categorical_fields.get("grade"),
            Some(&Some("low".to_string()))
        );
        assert_eq!(pq_cohort.rows[1].categorical_fields.get("grade"), Some(&None));
        assert_eq!(pq_cohort.rows[2].categorical_fields.get("grade"), Some(&None));
        assert_eq!(
            pq_cohort.rows[0].categorical_fields.get("eligible"),
            Some(&Some("true".to_string()))
        );
        assert_eq!(pq_cohort.rows[1].categorical_fields.get("eligible"), Some(&None));
        assert_eq!(
            pq_cohort.rows[2].categorical_fields.get("eligible"),
            Some(&Some("false".to_string()))
        );
        std::fs::remove_file(&pq_path).unwrap();
    }
}
