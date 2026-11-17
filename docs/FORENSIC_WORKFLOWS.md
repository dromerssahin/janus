# Forensic Workflows

Janus provides an in-memory forensic evaluation suite (`crates/janus/src/forensic_workflow.rs`) to test whether synthetic cohorts can be distinguished from held-out real reference cohorts. Forensic evaluation is a bounded technical check on distributional and predictive separability; it does not establish clinical plausibility, privacy safety, or external validity.

## Theory: Statistical vs. ML Discriminators

The forensic suite (`run_forensic_suite`, `crates/janus/src/forensic_workflow.rs:230-436`) combines two complementary families of discriminators (`DiscriminatorType`, `crates/janus/src/forensic_workflow.rs:28-33`):

- **Univariate statistical tests**:
  - **Chi-squared test** (`DiscriminatorType::ChiSquared`): compares marginal category frequencies between generated and reference cohorts for discrete fields.
  - **Kolmogorov-Smirnov (KS) test** (`DiscriminatorType::KolmogorovSmirnov`): compares marginal empirical cumulative distributions for continuous fields.
  - These tests detect single-variable drift (e.g., shifted means, missing categories, or distorted tails) but cannot detect broken multivariate correlations.

- **Multivariate ML discriminators**:
  - **Logistic Regression** (`DiscriminatorType::LogisticRegression`, `crates/janus/src/forensic_workflow.rs:790-793`): fits a linear decision boundary on combined generated (label 0) and reference (label 1) rows.
  - **Random Forest** (`DiscriminatorType::RandomForest`, `crates/janus/src/forensic_workflow.rs:900-903`): fits an ensemble of decision trees capable of capturing non-linear interactions and higher-order feature combinations.
  - Both classifiers are evaluated using area under the ROC curve (`compute_auc_roc`, `crates/janus/src/forensic_workflow.rs:912-939`), accuracy, precision, and recall.

### Interpretation Limits

The suite assigns `ForensicOutcome::Discriminable` (`crates/janus/src/forensic_workflow.rs:403-410`) if any of the following hold:
- Logistic regression `mean_auc_roc >= 0.60`
- Random forest `mean_auc_roc >= 0.60`
- Minimum univariate `p_value < 0.01` (`min_p_value`)

Otherwise, it assigns `ForensicOutcome::Indistinguishable`.

**Caveats:**
- **AUC thresholds**: An AUC near `0.50` means the tested linear or tree-based classifier did not separate the cohorts on the available features. It does not prove the cohorts are identical; a more expressive model, different feature encoding, or larger sample size could still separate them.
- **P-value caveats**: Univariate p-values depend strongly on sample size. With large cohorts, tiny, clinically irrelevant differences yield `p < 0.01`. With small cohorts, large structural defects may yield `p >= 0.01` due to low statistical power. No multiple-testing correction is applied to `min_p_value`.

## Small-Cell Suppression

Cohorts with fewer than `MINIMUM_FORENSIC_COHORT_ROWS = 10` rows (`crates/janus/src/forensic_workflow.rs:23`) trigger immediate small-cell suppression (`crates/janus/src/forensic_workflow.rs:274-303`).

When any of the training, generated, or reference cohorts has `< 10` rows:
- Statistical tests and ML discriminators are **not** executed.
- The suite returns `ForensicOutcome::Insufficient` with `suppressed: true`.
- `chi_squared_tests`, `ks_tests`, `per_seed_metrics`, `min_p_value`, `mean_effect_size`, `logistic_regression`, `random_forest`, and `combined_mean_auc_roc` are all empty or `None`.

**Rationale:** Small cohorts produce unstable, high-variance classifier metrics and unreliable asymptotic test statistics, while also increasing the risk of memorizing or exposing individual records in aggregate summaries.

## Multi-Seed Orchestration

ML discriminators use train/test splits and stochastic tree/optimization steps. Running a single seed can produce misleadingly high or low AUC values due to a lucky or unlucky split.

`run_forensic_suite` accepts a slice of explicit `u64` seeds and records:
- `per_seed_metrics` (`SeedDiscriminatorMetrics`, `crates/janus/src/forensic_workflow.rs:114-118`): per-seed AUC-ROC, accuracy, precision, and recall for both Logistic Regression and Random Forest.
- `AggregateClassifierMetrics` (`crates/janus/src/forensic_workflow.rs:123-130`): `mean_auc_roc`, `min_auc_roc`, `max_auc_roc`, `mean_accuracy`, `mean_precision`, and `mean_recall` across all seeds.

**Interpreting seed variation:**
- Wide `[min_auc_roc, max_auc_roc]` spreads indicate high split sensitivity or insufficient sample size.
- The `0.60` discriminability threshold is evaluated against `mean_auc_roc`, so inspecting `min_auc_roc` and `max_auc_roc` is essential to distinguish consistent separability from single-seed noise.

## Aggregate-Only Enforcement

`ForensicRow` (`crates/janus/src/forensic_workflow.rs:51-57`) and `ForensicCohort` (`crates/janus/src/forensic_workflow.rs:62-65`) are strictly in-memory structures. No patient-level file I/O is performed by the forensic workflow.

**What stays in memory:**
- Individual patient keys (`patient_key`), numeric values, categorical values, per-row predictions, and fitted classifier objects.

**What goes into `ForensicManifest` (`crates/janus/src/forensic_workflow.rs:154-165`):**
- `schema_version`, `discriminator_types`, `seeds`
- `fingerprints` (`ForensicFingerprints`, `crates/janus/src/forensic_workflow.rs:95-99`): canonical SHA-256 fingerprints of training, generated, and reference cohorts
- `software_version`, `dependency_lock_fingerprint`, and `environment` (`ForensicEnvironment`, `crates/janus/src/forensic_workflow.rs:104-109`)
- `outcome` (`ForensicOutcome`) and `aggregate_metrics` (`ForensicAggregateMetrics`, `crates/janus/src/forensic_workflow.rs:135-149`)
- `limitations`: explicit list of bounded-claim statements

## Known Limitations

- **Not privacy validation**: `Indistinguishable` does not establish differential privacy, k-anonymity, or protection against membership inference or memorization.
- **Not clinical plausibility**: Discriminators only test statistical and predictive separability on observed columns; they do not verify physiological or clinical coherence.
- **Not external validity**: Passing on an inspected or single-site reference does not establish transportability to external institutions or future cohorts.

## Example Manifest Structure

```json
{
  "schema_version": "1",
  "discriminator_types": [
    "chi_squared",
    "kolmogorov_smirnov",
    "logistic_regression",
    "random_forest"
  ],
  "seeds": [101, 202, 303],
  "fingerprints": {
    "training_fingerprint": "sha256:...",
    "generated_fingerprint": "sha256:...",
    "reference_fingerprint": "sha256:..."
  },
  "software_version": "0.1.0",
  "dependency_lock_fingerprint": "sha256:...",
  "environment": {
    "operating_system": "linux",
    "architecture": "x86_64",
    "rustc": "1.85.0",
    "target": "x86_64-unknown-linux-gnu"
  },
  "outcome": "indistinguishable",
  "aggregate_metrics": {
    "training_rows": 50,
    "generated_rows": 50,
    "reference_rows": 50,
    "minimum_cell_size": 10,
    "suppressed": false,
    "chi_squared_tests": [],
    "ks_tests": [],
    "min_p_value": 0.142,
    "mean_effect_size": 0.08,
    "logistic_regression": {
      "mean_auc_roc": 0.52,
      "min_auc_roc": 0.48,
      "max_auc_roc": 0.56,
      "mean_accuracy": 0.51,
      "mean_precision": 0.51,
      "mean_recall": 0.50
    },
    "random_forest": {
      "mean_auc_roc": 0.54,
      "min_auc_roc": 0.50,
      "max_auc_roc": 0.58,
      "mean_accuracy": 0.53,
      "mean_precision": 0.53,
      "mean_recall": 0.52
    },
    "combined_mean_auc_roc": 0.53,
    "per_seed_metrics": []
  },
  "limitations": [
    "Forensic evaluation tests statistical/ML separability only; it does not establish clinical plausibility, privacy safety, or external validity."
  ]
}
```
