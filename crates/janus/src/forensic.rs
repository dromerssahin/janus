use crate::Error;
use ndarray::Array2;
use ndarray_stats::CorrelationExt;
use serde::{Deserialize, Serialize};
use statrs::distribution::{ChiSquared, ContinuousCDF};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct DistributionalTest {
    pub variable_name: String,
    pub test_type: String,
    pub p_value: f64,
    pub statistic: f64,
    pub effect_size: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CorrelationDistance {
    pub frobenius_norm: f64,
    pub normalized_distance: f64,
    pub n_variables: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TemporalPatterns {
    pub sequence_entropy_real: f64,
    pub sequence_entropy_synthetic: f64,
    pub timing_ks_pvalue: f64,
    pub timing_ks_statistic: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct DiscriminatorMetrics {
    pub auc_roc: f64,
    pub accuracy: f64,
    pub precision: f64,
    pub recall: f64,
    pub n_train: usize,
    pub n_test: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ForensicReport {
    pub dataset_fingerprint_real: String,
    pub dataset_fingerprint_synthetic: String,
    pub seed: u64,
    pub distributional_tests: Vec<DistributionalTest>,
    pub correlation_distance: CorrelationDistance,
    pub temporal_patterns: Option<TemporalPatterns>,
    pub discriminator: DiscriminatorMetrics,
    pub timestamp: String,
}

impl ForensicReport {
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }
}

/// Two-sample Kolmogorov-Smirnov test for continuous variables.
pub fn kolmogorov_smirnov_test(
    variable_name: &str,
    real: &[f64],
    synthetic: &[f64],
) -> Result<DistributionalTest, Error> {
    let mut r: Vec<f64> = real.iter().copied().filter(|x| x.is_finite()).collect();
    let mut s: Vec<f64> = synthetic
        .iter()
        .copied()
        .filter(|x| x.is_finite())
        .collect();
    if r.is_empty() || s.is_empty() {
        return Err(Error::Input(format!(
            "KS test requires non-empty finite samples for '{variable_name}'"
        )));
    }
    r.sort_by(|a, b| a.total_cmp(b));
    s.sort_by(|a, b| a.total_cmp(b));

    let n1 = r.len() as f64;
    let n2 = s.len() as f64;
    let mut i = 0usize;
    let mut j = 0usize;
    let mut d = 0.0_f64;

    while i < r.len() && j < s.len() {
        let val = if r[i] <= s[j] { r[i] } else { s[j] };
        while i < r.len() && r[i] == val {
            i += 1;
        }
        while j < s.len() && s[j] == val {
            j += 1;
        }
        let cdf1 = i as f64 / n1;
        let cdf2 = j as f64 / n2;
        d = d.max((cdf1 - cdf2).abs());
    }

    let p_value = if d == 0.0 {
        1.0
    } else {
        let en = ((n1 * n2) / (n1 + n2)).sqrt();
        let lambda = (en + 0.12 + 0.11 / en) * d;
        let mut sum = 0.0;
        for k in 1..=100 {
            let kf = k as f64;
            let term = (-2.0 * kf * kf * lambda * lambda).exp();
            if k % 2 == 1 {
                sum += term;
            } else {
                sum -= term;
            }
            if term < 1e-12 {
                break;
            }
        }
        (2.0 * sum).clamp(0.0, 1.0)
    };

    Ok(DistributionalTest {
        variable_name: variable_name.to_string(),
        test_type: "kolmogorov_smirnov".to_string(),
        p_value,
        statistic: d,
        effect_size: d,
    })
}

/// Two-sample Chi-squared test of homogeneity for categorical variables.
pub fn chi_squared_test(
    variable_name: &str,
    real: &[String],
    synthetic: &[String],
) -> Result<DistributionalTest, Error> {
    if real.is_empty() || synthetic.is_empty() {
        return Err(Error::Input(format!(
            "Chi-squared test requires non-empty samples for '{variable_name}'"
        )));
    }

    let mut real_counts: BTreeMap<&str, usize> = BTreeMap::new();
    let mut synth_counts: BTreeMap<&str, usize> = BTreeMap::new();
    let mut categories: BTreeSet<&str> = BTreeSet::new();

    for v in real {
        *real_counts.entry(v.as_str()).or_insert(0) += 1;
        categories.insert(v.as_str());
    }
    for v in synthetic {
        *synth_counts.entry(v.as_str()).or_insert(0) += 1;
        categories.insert(v.as_str());
    }

    let n_r = real.len() as f64;
    let n_s = synthetic.len() as f64;
    let n_total = n_r + n_s;

    let mut chi2 = 0.0;
    for cat in &categories {
        let r_c = *real_counts.get(cat).unwrap_or(&0) as f64;
        let s_c = *synth_counts.get(cat).unwrap_or(&0) as f64;
        let t_c = r_c + s_c;
        if t_c > 0.0 {
            let e_r = n_r * t_c / n_total;
            let e_s = n_s * t_c / n_total;
            if e_r > 0.0 {
                chi2 += (r_c - e_r).powi(2) / e_r;
            }
            if e_s > 0.0 {
                chi2 += (s_c - e_s).powi(2) / e_s;
            }
        }
    }

    let df = categories.len().saturating_sub(1).max(1) as f64;
    let p_value = if chi2 == 0.0 {
        1.0
    } else {
        let dist = ChiSquared::new(df)
            .map_err(|e| Error::Input(format!("chi-squared distribution error: {e}")))?;
        (1.0 - dist.cdf(chi2)).clamp(0.0, 1.0)
    };
    let cramers_v = (chi2 / n_total).sqrt();

    Ok(DistributionalTest {
        variable_name: variable_name.to_string(),
        test_type: "chi_squared".to_string(),
        p_value,
        statistic: chi2,
        effect_size: cramers_v,
    })
}

/// Compare Pearson correlation matrices of continuous variables (rows=samples, cols=variables).
pub fn correlation_distance(
    real_matrix: &Array2<f64>,
    synthetic_matrix: &Array2<f64>,
) -> Result<CorrelationDistance, Error> {
    let n_vars = real_matrix.ncols();
    if n_vars == 0 || synthetic_matrix.ncols() != n_vars {
        return Err(Error::Input(
            "correlation_distance requires non-zero matching column counts".into(),
        ));
    }
    if real_matrix.nrows() < 2 || synthetic_matrix.nrows() < 2 {
        return Err(Error::Input(
            "correlation_distance requires at least 2 rows per dataset".into(),
        ));
    }

    let c_real = real_matrix
        .t()
        .pearson_correlation()
        .map_err(|e| Error::Input(format!("real correlation failed: {e}")))?;
    let c_synth = synthetic_matrix
        .t()
        .pearson_correlation()
        .map_err(|e| Error::Input(format!("synthetic correlation failed: {e}")))?;

    let diff = &c_real - &c_synth;
    let frobenius_norm = diff
        .mapv(|x| if x.is_finite() { x * x } else { 0.0 })
        .sum()
        .sqrt();
    let normalized_distance = frobenius_norm / ((n_vars * n_vars) as f64).sqrt();

    Ok(CorrelationDistance {
        frobenius_norm,
        normalized_distance,
        n_variables: n_vars,
    })
}

fn pairwise_optional_pearson_matrix(matrix: &Array2<Option<f64>>) -> Array2<f64> {
    let n_rows = matrix.nrows();
    let n_vars = matrix.ncols();
    let mut corr = Array2::<f64>::eye(n_vars);

    for i in 0..n_vars {
        for j in (i + 1)..n_vars {
            let mut pairs: Vec<(f64, f64)> = Vec::new();
            for r in 0..n_rows {
                if let (Some(x), Some(y)) = (matrix[[r, i]], matrix[[r, j]]) {
                    if x.is_finite() && y.is_finite() {
                        pairs.push((x, y));
                    }
                }
            }
            let r_val = if pairs.len() < 2 {
                0.0
            } else {
                let first_x = pairs[0].0;
                let first_y = pairs[0].1;
                if pairs.iter().all(|&(x, _)| x == first_x)
                    || pairs.iter().all(|&(_, y)| y == first_y)
                {
                    0.0
                } else {
                    let count = pairs.len() as f64;
                    let mean_x = pairs.iter().map(|&(x, _)| x).sum::<f64>() / count;
                    let mean_y = pairs.iter().map(|&(_, y)| y).sum::<f64>() / count;
                    let mut cov_xy = 0.0;
                    let mut var_x = 0.0;
                    let mut var_y = 0.0;
                    for &(x, y) in &pairs {
                        let dx = x - mean_x;
                        let dy = y - mean_y;
                        cov_xy += dx * dy;
                        var_x += dx * dx;
                        var_y += dy * dy;
                    }
                    let denom = (var_x * var_y).sqrt();
                    if denom <= 0.0 || !denom.is_finite() {
                        0.0
                    } else {
                        let raw = cov_xy / denom;
                        if raw.is_finite() {
                            raw.clamp(-1.0, 1.0)
                        } else {
                            0.0
                        }
                    }
                }
            };
            corr[[i, j]] = r_val;
            corr[[j, i]] = r_val;
        }
    }
    corr
}

/// Compare pairwise Pearson correlation matrices over optional continuous variables
/// (rows=samples, cols=variables) using only complete finite observed pairs `(Some(x), Some(y))`.
pub fn correlation_distance_optional(
    real_matrix: &Array2<Option<f64>>,
    synthetic_matrix: &Array2<Option<f64>>,
) -> Result<CorrelationDistance, Error> {
    let n_vars = real_matrix.ncols();
    if n_vars == 0 || synthetic_matrix.ncols() != n_vars {
        return Err(Error::Input(
            "correlation_distance_optional requires non-zero matching column counts".into(),
        ));
    }
    if real_matrix.nrows() < 2 || synthetic_matrix.nrows() < 2 {
        return Err(Error::Input(
            "correlation_distance_optional requires at least 2 rows per dataset".into(),
        ));
    }

    let c_real = pairwise_optional_pearson_matrix(real_matrix);
    let c_synth = pairwise_optional_pearson_matrix(synthetic_matrix);
    let diff = &c_real - &c_synth;
    let frobenius_norm = diff
        .mapv(|x| if x.is_finite() { x * x } else { 0.0 })
        .sum()
        .sqrt();
    let normalized_distance = frobenius_norm / ((n_vars * n_vars) as f64).sqrt();

    Ok(CorrelationDistance {
        frobenius_norm,
        normalized_distance,
        n_variables: n_vars,
    })
}

fn shannon_entropy(sequences: &[Vec<String>]) -> f64 {
    if sequences.is_empty() {
        return 0.0;
    }
    let mut counts: BTreeMap<String, usize> = BTreeMap::new();
    for seq in sequences {
        let key = seq.join("->");
        *counts.entry(key).or_insert(0) += 1;
    }
    let total = sequences.len() as f64;
    let mut entropy = 0.0;
    for count in counts.values() {
        let p = *count as f64 / total;
        if p > 0.0 {
            entropy -= p * p.log2();
        }
    }
    entropy
}

/// Temporal pattern tests: sequence Shannon entropy and inter-event timing K-S test.
pub fn temporal_pattern_tests(
    real_sequences: &[Vec<String>],
    synthetic_sequences: &[Vec<String>],
    real_deltas: &[f64],
    synthetic_deltas: &[f64],
) -> Result<TemporalPatterns, Error> {
    let ks = kolmogorov_smirnov_test("inter_event_timing", real_deltas, synthetic_deltas)?;
    Ok(TemporalPatterns {
        sequence_entropy_real: shannon_entropy(real_sequences),
        sequence_entropy_synthetic: shannon_entropy(synthetic_sequences),
        timing_ks_pvalue: ks.p_value,
        timing_ks_statistic: ks.statistic,
    })
}

/// Train logistic regression discriminator (80/20 split, fixed seed) on numeric features.
/// Label: 0 = real, 1 = synthetic.
pub fn train_discriminator(
    real_features: &Array2<f64>,
    synthetic_features: &Array2<f64>,
    seed: u64,
) -> Result<DiscriminatorMetrics, Error> {
    crate::forensic_workflow::train_logistic_discriminator(
        real_features,
        synthetic_features,
        seed,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{rngs::StdRng, Rng, SeedableRng};

    #[test]
    fn test_ks_identical_distributions() {
        let sample = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let res = kolmogorov_smirnov_test("age", &sample, &sample).unwrap();
        assert!((res.p_value - 1.0).abs() < 1e-9);
        assert_eq!(res.statistic, 0.0);
    }

    #[test]
    fn test_ks_different_distributions() {
        let mut rng = StdRng::seed_from_u64(42);
        let mut sample_a = Vec::with_capacity(200);
        let mut sample_b = Vec::with_capacity(200);
        // Box-Muller for N(0,1) vs N(1,1)
        for _ in 0..100 {
            let u1: f64 = rng.gen_range(1e-6..1.0);
            let u2: f64 = rng.gen_range(0.0..1.0);
            let z0 = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
            let z1 = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).sin();
            sample_a.push(z0);
            sample_a.push(z1);
            sample_b.push(z0 + 1.0);
            sample_b.push(z1 + 1.0);
        }
        let res = kolmogorov_smirnov_test("biomarker", &sample_a, &sample_b).unwrap();
        assert!(res.p_value < 0.05, "expected p < 0.05, got {}", res.p_value);
    }

    #[test]
    fn test_chisq_identical_categorical() {
        let cats: Vec<String> = vec!["A", "A", "B", "B", "C", "C"]
            .into_iter()
            .map(String::from)
            .collect();
        let res = chi_squared_test("sex", &cats, &cats).unwrap();
        assert!((res.p_value - 1.0).abs() < 1e-9);
        assert!((res.statistic - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_correlation_distance_zero() {
        let mat = Array2::from_shape_vec(
            (4, 3),
            vec![1.0, 2.0, 3.0, 2.0, 4.0, 5.0, 3.0, 6.0, 8.0, 4.0, 8.0, 9.0],
        )
        .unwrap();
        let dist = correlation_distance(&mat, &mat).unwrap();
        assert!(dist.frobenius_norm < 1e-12);
        assert!(dist.normalized_distance < 1e-12);
        assert_eq!(dist.n_variables, 3);
    }

    #[test]
    fn test_discriminator_random_labels() {
        let mut rng = StdRng::seed_from_u64(123);
        let n = 200;
        let mut real_vec = Vec::with_capacity(n * 3);
        let mut synth_vec = Vec::with_capacity(n * 3);
        for _ in 0..(n * 3) {
            real_vec.push(rng.gen_range(-1.0..1.0));
            synth_vec.push(rng.gen_range(-1.0..1.0));
        }
        let real = Array2::from_shape_vec((n, 3), real_vec).unwrap();
        let synth = Array2::from_shape_vec((n, 3), synth_vec).unwrap();
        let metrics = train_discriminator(&real, &synth, 99).unwrap();
        assert!(
            (metrics.auc_roc - 0.5).abs() < 0.15,
            "expected AUC near 0.5 for indistinguishable random features, got {}",
            metrics.auc_roc
        );
    }

    #[test]
    fn test_correlation_distance_optional_pairwise_complete_and_zero_variance() {
        // Real matrix with perfect correlation y = 2*x across observed rows
        let real_complete = Array2::from_shape_vec(
            (4, 2),
            vec![1.0, 2.0, 2.0, 4.0, 3.0, 6.0, 4.0, 8.0],
        )
        .unwrap();
        let real_opt = real_complete.mapv(Some);

        // Synthetic matrix has extra incomplete rows that must not be imputed as 0.0
        let synth_opt = Array2::from_shape_vec(
            (6, 2),
            vec![
                Some(1.0),
                Some(2.0),
                Some(2.0),
                Some(4.0),
                Some(3.0),
                Some(6.0),
                Some(4.0),
                Some(8.0),
                Some(100.0),
                None,
                None,
                Some(-100.0),
            ],
        )
        .unwrap();
        let dist = correlation_distance_optional(&real_opt, &synth_opt).unwrap();
        assert!(dist.frobenius_norm < 1e-12);
        assert!(dist.normalized_distance < 1e-12);
        assert_eq!(dist.n_variables, 2);

        // Verify exact match with correlation_distance on complete matrices
        let synth_flipped = Array2::from_shape_vec(
            (4, 2),
            vec![1.0, 8.0, 2.0, 6.0, 3.0, 4.0, 4.0, 2.0],
        )
        .unwrap();
        let d_full = correlation_distance(&real_complete, &synth_flipped).unwrap();
        let d_opt =
            correlation_distance_optional(&real_opt, &synth_flipped.mapv(Some)).unwrap();
        assert!((d_full.frobenius_norm - d_opt.frobenius_norm).abs() < 1e-12);
        assert!((d_full.normalized_distance - d_opt.normalized_distance).abs() < 1e-12);

        // Zero-variance column (repeated 0.1) and <2 complete pairs default to off-diagonal 0.0
        let zero_var = Array2::from_shape_vec(
            (3, 2),
            vec![
                Some(0.1),
                Some(1.0),
                Some(0.1),
                Some(2.0),
                Some(0.1),
                Some(3.0),
            ],
        )
        .unwrap();
        let one_pair = Array2::from_shape_vec(
            (3, 2),
            vec![Some(1.0), Some(1.0), Some(2.0), None, None, Some(3.0)],
        )
        .unwrap();
        let d_edge = correlation_distance_optional(&zero_var, &one_pair).unwrap();
        assert!(d_edge.frobenius_norm < 1e-12);
    }

    #[test]
    fn test_train_discriminator_separable_probability_orientation_across_seeds() {
        let mut rng = StdRng::seed_from_u64(7);
        let n = 60;
        let mut real_vec = Vec::with_capacity(n * 2);
        let mut synth_vec = Vec::with_capacity(n * 2);
        for _ in 0..n {
            real_vec.push(rng.gen_range(-2.0..-0.5));
            real_vec.push(rng.gen_range(-1.0..1.0));
            synth_vec.push(rng.gen_range(0.5..2.0));
            synth_vec.push(rng.gen_range(-1.0..1.0));
        }
        let real = Array2::from_shape_vec((n, 2), real_vec).unwrap();
        let synth = Array2::from_shape_vec((n, 2), synth_vec).unwrap();
        for seed in [1_u64, 2, 3, 42, 101] {
            let metrics = train_discriminator(&real, &synth, seed).unwrap();
            assert!(
                metrics.auc_roc > 0.9,
                "seed {seed} expected AUC > 0.9 (not inverted), got {}",
                metrics.auc_roc
            );
        }
    }
}
