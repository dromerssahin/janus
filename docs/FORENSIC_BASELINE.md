# Forensic Baseline Report (Fictional Fixture)

## Scope and Bounded Claims

This report records aggregate-only technical forensic metrics comparing the
Empirical (`examples/fit.yaml`) and Forest/Conditional (`examples/fit-conditional.yaml`)
baselines against the fictional fixture (`examples/fixture-conditional.yaml`).

- **Evidence boundary**: Fictional fixture only. No clinical plausibility,
  independent validation, causal validity, privacy safety, or model superiority
  is claimed.
- **No patient-level records**: Only aggregate statistics, schema fingerprints,
  and configuration metadata are recorded.

## Reproduction

```bash
janus-cli fit examples/study.yaml examples/fixture-conditional.yaml examples/fit.yaml /tmp/janus-forensic-bench/fit-empirical
janus-cli generate examples/study.yaml /tmp/janus-forensic-bench/fit-empirical/generator.yaml /tmp/janus-forensic-bench/synth-empirical
janus-cli fit examples/study.yaml examples/fixture-conditional.yaml examples/fit-conditional.yaml /tmp/janus-forensic-bench/fit-forest
janus-cli generate examples/study.yaml /tmp/janus-forensic-bench/fit-forest/generator.yaml /tmp/janus-forensic-bench/synth-forest

janus-cli forensic-eval --real /tmp/janus-forensic-bench/real.csv --synthetic /tmp/janus-forensic-bench/empirical.csv --seed 42
janus-cli forensic-eval --real /tmp/janus-forensic-bench/real.csv --synthetic /tmp/janus-forensic-bench/forest.csv --seed 42
```

## Aggregate Results (Seed 42)

### Empirical Baseline (`examples/fit.yaml`)

- **Real fingerprint**: `fnv1a64-schema:5a346591fa61f3d8`
- **Synthetic fingerprint**: `fnv1a64-schema:c2ca7b7576a77213`
- **Timestamp**: `2026-11-13T09:38:07Z`
- **Distributional tests (Kolmogorov-Smirnov)**:
  - `event_day`: statistic = `0.2000`, p-value = `0.3844`, effect size = `0.2000`
  - `recording_day`: statistic = `0.2000`, p-value = `0.3844`, effect size = `0.2000`
- **Correlation distance**: Frobenius norm = `0.0`, normalized distance = `0.0` (`n_variables = 2`)
- **Discriminator (Logistic Regression, 80/20 split, `n_train = 416`, `n_test = 104`)**:
  - `auc_roc`: `0.5200`
  - `accuracy`: `0.9615`
  - `precision`: `0.9615`
  - `recall`: `1.0000`

### Forest / Conditional Baseline (`examples/fit-conditional.yaml`)

- **Real fingerprint**: `fnv1a64-schema:5a346591fa61f3d8`
- **Synthetic fingerprint**: `fnv1a64-schema:c2ca7b7576a77213`
- **Timestamp**: `2026-11-13T09:38:07Z`
- **Distributional tests (Kolmogorov-Smirnov)**:
  - `event_day`: statistic = `0.2000`, p-value = `0.3844`, effect size = `0.2000`
  - `recording_day`: statistic = `0.2000`, p-value = `0.3844`, effect size = `0.2000`
- **Correlation distance**: Frobenius norm = `0.0`, normalized distance = `0.0` (`n_variables = 2`)
- **Discriminator (Logistic Regression, 80/20 split, `n_train = 416`, `n_test = 104`)**:
  - `auc_roc`: `0.5200`
  - `accuracy`: `0.9615`
  - `precision`: `0.9615`
  - `recall`: `1.0000`

## Fictional Fixture vs. `neural-ctgan` Comparison (`2026-11-13T10:51:39Z`)

- **Reference**: `examples/fixture-conditional.yaml` exported to `real.csv` (`20` event rows, schema fingerprint `fnv1a64-schema:5a346591fa61f3d8`).
- **Comparable settings**: Generator seed `42`, synthetic sample size `500` event rows (`100` episodes × `5` variables for `examples/generator-conditional.yaml`; `n_samples: 500` for `neural-ctgan` with `epochs: 50`, `batch_size: 10`, `pac: 2`, `embedding_dim: 32`, `generator_dim: [64, 64]`, `discriminator_dim: [64, 64]`), and `janus-cli forensic-eval --seed 42` (`n_train = 416`, `n_test = 104`, synthetic fingerprint `fnv1a64-schema:c2ca7b7576a77213`).
- **Environment**: `janus-cli 0.1.0`, Python `3.13.18`, `ctgan 0.13.0`, `torch 2.15.0+cu132` (CPU execution, `cuda: false`).

| Generator | Seed | Synthetic Rows | `event_day` KS (stat / p-val) | `recording_day` KS (stat / p-val) | Corr Dist (Frobenius / Norm) | Discriminator AUC-ROC | Accuracy | Precision | Recall |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Fictional Fixture Baseline (`examples/generator-conditional.yaml`) | 42 | 500 | `0.0000` / `1.0000` | `0.0000` / `1.0000` | `0.0000` / `0.0000` | `0.4175` | `0.9615` | `0.9615` | `1.0000` |
| `neural-ctgan` (`ctgan`, 50 epochs) | 42 | 500 | `0.4960` / `8.23e-5` | `0.6760` / `1.43e-8` | `1.5096` / `0.7548` | `0.7100` | `0.9615` | `0.9615` | `1.0000` |

> **Notes**:
> - `accuracy`, `precision`, and `recall` reflect the `500:20` (`96.15%` synthetic) class imbalance in the `80/20` discriminator split; `auc_roc` and the Kolmogorov-Smirnov / correlation-distance metrics capture the distributional separation.
> - Fictional fixture only; these aggregate metrics do not establish clinical plausibility, privacy safety, causal validity, or general model superiority.
