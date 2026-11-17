use crate::{candidate::Row, Error};
use serde::{Deserialize, Serialize};
use std::{
    collections::BTreeMap,
    fs,
    io::Write,
    path::{Path, PathBuf},
    process::{Command, Stdio},
};

/// Selects which generator backend `janus-cli generate` executes.
///
/// Decision: Keep `Configured` as the default to preserve backwards
/// compatibility with existing YAML-configured cohort generation workflows,
/// and add `Neural` for the Python CTGAN/TVAE bridge (`--generator neural-ctgan`).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum GeneratorKind {
    #[default]
    Configured,
    #[serde(rename = "neural-ctgan", alias = "neural")]
    Neural,
}

/// Supported neural tabular model architectures in `scripts/neural_bridge.py`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum NeuralModelType {
    #[default]
    Ctgan,
    Tvae,
}

fn default_epochs() -> usize {
    300
}

fn default_batch_size() -> usize {
    500
}

fn default_seed() -> u64 {
    42
}

fn default_n_samples() -> usize {
    100
}

fn default_embedding_dim() -> usize {
    128
}

/// Hyperparameter and execution configuration for the Python CTGAN/TVAE bridge.
///
/// Decision: Keep this schema study-agnostic (no clinical variable assumptions)
/// and mirror `ctgan` package hyperparameters directly so the Python bridge
/// remains a thin interface layer.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct NeuralGeneratorConfig {
    #[serde(default)]
    pub model_type: NeuralModelType,
    #[serde(default = "default_epochs")]
    pub epochs: usize,
    #[serde(default = "default_batch_size")]
    pub batch_size: usize,
    #[serde(default = "default_seed")]
    pub seed: u64,
    #[serde(default = "default_n_samples")]
    pub n_samples: usize,
    #[serde(default = "default_embedding_dim")]
    pub embedding_dim: usize,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub generator_dim: Option<Vec<usize>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub discriminator_dim: Option<Vec<usize>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub compress_dims: Option<Vec<usize>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub decompress_dims: Option<Vec<usize>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub generator_lr: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub discriminator_lr: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pac: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub discriminator_steps: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub l2scale: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub loss_factor: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub discrete_columns: Option<Vec<String>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub training_data: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cuda: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fit_seed: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sample_seeds: Option<Vec<u64>>,
}

impl Default for NeuralGeneratorConfig {
    fn default() -> Self {
        Self {
            model_type: NeuralModelType::Ctgan,
            epochs: default_epochs(),
            batch_size: default_batch_size(),
            seed: default_seed(),
            n_samples: default_n_samples(),
            embedding_dim: default_embedding_dim(),
            generator_dim: None,
            discriminator_dim: None,
            compress_dims: None,
            decompress_dims: None,
            generator_lr: None,
            discriminator_lr: None,
            pac: None,
            discriminator_steps: None,
            l2scale: None,
            loss_factor: None,
            discrete_columns: None,
            training_data: None,
            cuda: None,
            fit_seed: None,
            sample_seeds: None,
        }
    }
}

impl NeuralGeneratorConfig {
    /// Validate basic hyperparameter invariants before spawning Python.
    pub fn validate(&self) -> Result<(), Error> {
        if self.epochs == 0 {
            return Err(Error::Input("neural generator epochs must be > 0".into()));
        }
        if self.batch_size == 0 {
            return Err(Error::Input(
                "neural generator batch_size must be > 0".into(),
            ));
        }
        if self.n_samples == 0 {
            return Err(Error::Input(
                "neural generator n_samples must be > 0".into(),
            ));
        }
        if self.embedding_dim == 0 {
            return Err(Error::Input(
                "neural generator embedding_dim must be > 0".into(),
            ));
        }
        if let Some(sample_seeds) = &self.sample_seeds {
            if sample_seeds.is_empty() {
                return Err(Error::Input(
                    "neural generator sample_seeds must not be empty when specified".into(),
                ));
            }
        }
        if self.model_type == NeuralModelType::Ctgan {
            let pac = self.pac.unwrap_or(10);
            if pac == 0 || !self.batch_size.is_multiple_of(pac) {
                return Err(Error::Input(format!(
                    "CTGAN requires pac > 0 and batch_size ({}) divisible by pac ({pac})",
                    self.batch_size
                )));
            }
        }
        Ok(())
    }
}

/// Metadata emitted directly by `scripts/neural_bridge.py`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct NeuralBridgeManifest {
    pub schema_version: String,
    pub model_type: NeuralModelType,
    pub hyperparameters: serde_json::Value,
    pub epochs: usize,
    pub seed: u64,
    pub n_samples: usize,
    pub training_dataset_fingerprint: String,
    pub training_rows: usize,
    pub training_columns: Vec<String>,
    pub discrete_columns: Vec<String>,
    pub samples_fingerprint: String,
    pub timestamp: String,
    pub python_version: String,
    pub ctgan_version: String,
    pub torch_version: String,
    pub artifacts: Vec<String>,
}

/// Complete Janus manifest for a neural generator run (`manifest.json`).
///
/// Captures hyperparameters, epochs, training dataset fingerprint, seed,
/// timestamp, software environment, and reproducibility limitations.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct NeuralGeneratorManifest {
    pub schema_version: String,
    pub generator_kind: GeneratorKind,
    pub model_type: NeuralModelType,
    pub hyperparameters: serde_json::Value,
    pub epochs: usize,
    pub seed: u64,
    pub n_samples: usize,
    pub training_dataset_fingerprint: String,
    pub training_rows: usize,
    pub training_columns: Vec<String>,
    pub discrete_columns: Vec<String>,
    pub samples_fingerprint: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub study_fingerprint: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub config_fingerprint: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fixture_fingerprint: Option<String>,
    pub timestamp: String,
    pub python_version: String,
    pub ctgan_version: String,
    pub torch_version: String,
    pub software_version: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dependency_lock_fingerprint: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub operating_system: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub architecture: Option<String>,
    pub artifacts: Vec<String>,
    pub reproducibility_limits: String,
}

impl NeuralGeneratorManifest {
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }

    pub fn from_json(json: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(json)
    }
}

/// Locate `scripts/neural_bridge.py` and its enclosing `uv` project root.
///
/// Resolution order:
/// 1. Explicit path argument
/// 2. `JANUS_NEURAL_BRIDGE` environment variable
/// 3. `./scripts/neural_bridge.py` relative to current working directory
/// 4. Ancestors of the current executable
/// 5. Compile-time workspace root fallback (`CARGO_MANIFEST_DIR/../../scripts/neural_bridge.py`)
pub fn resolve_bridge_script(explicit: Option<&Path>) -> Result<(PathBuf, PathBuf), Error> {
    let mut candidates = Vec::new();
    if let Some(p) = explicit {
        candidates.push(p.to_path_buf());
    }
    if let Ok(env_path) = std::env::var("JANUS_NEURAL_BRIDGE") {
        candidates.push(PathBuf::from(env_path));
    }
    candidates.push(PathBuf::from("scripts/neural_bridge.py"));
    if let Ok(exe) = std::env::current_exe() {
        for ancestor in exe.ancestors() {
            candidates.push(ancestor.join("scripts/neural_bridge.py"));
        }
    }
    let workspace_fallback =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../scripts/neural_bridge.py");
    candidates.push(workspace_fallback);

    for candidate in candidates {
        if candidate.is_file() {
            let canonical = candidate.canonicalize()?;
            let project_root = canonical
                .parent()
                .and_then(|p| p.parent())
                .ok_or_else(|| {
                    Error::Input(format!(
                        "cannot determine uv project root from bridge path {}",
                        canonical.display()
                    ))
                })?
                .to_path_buf();
            return Ok((canonical, project_root));
        }
    }

    Err(Error::Input(
        "unable to locate scripts/neural_bridge.py; pass --bridge-script or set JANUS_NEURAL_BRIDGE"
            .into(),
    ))
}

/// Invoke `scripts/neural_bridge.py` via `uv run` in `work_dir` and parse its manifest.
///
/// Writes `config.json`, `samples.csv`, `model.pkl`, and `neural_manifest.json`
/// inside `work_dir`.
pub fn run_neural_bridge(
    input_data: &Path,
    config: &NeuralGeneratorConfig,
    work_dir: &Path,
    bridge_script: Option<&Path>,
) -> Result<NeuralBridgeManifest, Error> {
    config.validate()?;
    if !input_data.is_file() {
        return Err(Error::Input(format!(
            "training dataset not found: {}",
            input_data.display()
        )));
    }

    let (script_path, project_root) = resolve_bridge_script(bridge_script)?;
    fs::create_dir_all(work_dir)?;

    let config_path = work_dir.join("config.json");
    let samples_path = work_dir.join("samples.csv");
    let model_path = work_dir.join("model.pkl");
    let manifest_path = work_dir.join("neural_manifest.json");

    let config_json = serde_json::to_string_pretty(config)
        .map_err(|e| Error::Input(format!("failed to serialize neural config: {e}")))?;
    fs::write(&config_path, config_json)?;

    let output = Command::new("uv")
        .arg("run")
        .arg("--project")
        .arg(&project_root)
        .arg(&script_path)
        .arg("--input")
        .arg(input_data)
        .arg("--config")
        .arg(&config_path)
        .arg("--output-samples")
        .arg(&samples_path)
        .arg("--output-model")
        .arg(&model_path)
        .arg("--output-manifest")
        .arg(&manifest_path)
        .output()
        .map_err(|e| Error::Input(format!("failed to execute uv neural bridge: {e}")))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        let stdout = String::from_utf8_lossy(&output.stdout);
        return Err(Error::Input(format!(
            "neural bridge exited with {}: {}{}",
            output.status,
            stderr.trim(),
            if stdout.trim().is_empty() {
                String::new()
            } else {
                format!("\nstdout: {}", stdout.trim())
            }
        )));
    }

    let manifest_raw = fs::read_to_string(&manifest_path)?;
    serde_json::from_str(&manifest_raw)
        .map_err(|e| Error::Input(format!("invalid neural bridge manifest JSON: {e}")))
}

#[derive(Debug, Deserialize)]
struct StdioBridgeResponse {
    manifest: NeuralBridgeManifest,
    draws: BTreeMap<String, Vec<BTreeMap<String, String>>>,
}

fn validate_stdio_training_rows(training_rows: &[Row]) -> Result<Vec<String>, Error> {
    if training_rows.is_empty() {
        return Err(Error::Input(
            "neural training_rows must not be empty".into(),
        ));
    }
    let columns: Vec<String> = training_rows[0].fields.keys().cloned().collect();
    if columns.is_empty() {
        return Err(Error::Input(
            "neural training_rows must contain at least one variable".into(),
        ));
    }
    for (idx, row) in training_rows.iter().enumerate().skip(1) {
        if row.fields.len() != columns.len() || !row.fields.keys().eq(columns.iter()) {
            return Err(Error::Input(format!(
                "neural training_rows[{idx}] schema does not match training_rows[0]"
            )));
        }
    }
    Ok(columns)
}

fn parse_stdio_draws(
    response: &StdioBridgeResponse,
    columns: &[String],
    training_rows_len: usize,
    fit_seed: u64,
    n_samples: usize,
    sample_seeds: &[u64],
) -> Result<BTreeMap<u64, Vec<Row>>, Error> {
    if !response.manifest.artifacts.is_empty() {
        return Err(Error::Input(
            "in-memory neural bridge must not emit disk artifacts".into(),
        ));
    }
    if response.manifest.training_columns != columns {
        return Err(Error::Input(
            "neural bridge manifest training_columns do not match training schema".into(),
        ));
    }
    if response.manifest.training_rows != training_rows_len {
        return Err(Error::Input(format!(
            "neural bridge manifest training_rows {} != expected {training_rows_len}",
            response.manifest.training_rows
        )));
    }
    if response.manifest.n_samples != n_samples {
        return Err(Error::Input(format!(
            "neural bridge manifest n_samples {} != expected {n_samples}",
            response.manifest.n_samples
        )));
    }
    if response.manifest.seed != fit_seed {
        return Err(Error::Input(format!(
            "neural bridge manifest seed {} != expected fit_seed {fit_seed}",
            response.manifest.seed
        )));
    }

    let mut draws = BTreeMap::new();
    for (seed_key, raw_rows) in &response.draws {
        let seed_val = seed_key.parse::<u64>().map_err(|e| {
            Error::Input(format!(
                "neural bridge draw key '{seed_key}' is not a valid u64 seed: {e}"
            ))
        })?;
        if raw_rows.len() != n_samples {
            return Err(Error::Input(format!(
                "neural bridge draw for seed {seed_val} returned {} rows, expected {n_samples}",
                raw_rows.len()
            )));
        }
        let mut parsed_rows = Vec::with_capacity(raw_rows.len());
        for (row_idx, row_map) in raw_rows.iter().enumerate() {
            if row_map.len() != columns.len() || !row_map.keys().eq(columns.iter()) {
                return Err(Error::Input(format!(
                    "neural bridge draw for seed {seed_val} row {row_idx} columns do not match expected schema"
                )));
            }
            parsed_rows.push(Row::new(row_map.clone()));
        }
        if draws.insert(seed_val, parsed_rows).is_some() {
            return Err(Error::Input(format!(
                "duplicate sample seed key {seed_val} in neural bridge response"
            )));
        }
    }

    for &expected_seed in sample_seeds {
        if !draws.contains_key(&expected_seed) {
            return Err(Error::Input(format!(
                "neural bridge response missing requested sample_seed {expected_seed}"
            )));
        }
    }

    Ok(draws)
}

/// Invoke `scripts/neural_bridge.py --stdio` strictly in memory with zero disk I/O.
///
/// Trains the neural model once using `config.fit_seed.unwrap_or(config.seed)` and
/// samples `config.n_samples` rows for every seed in `sample_seeds`.
pub fn run_neural_bridge_in_memory(
    training_rows: &[Row],
    config: &NeuralGeneratorConfig,
    sample_seeds: &[u64],
    bridge_script: Option<&Path>,
) -> Result<(NeuralBridgeManifest, BTreeMap<u64, Vec<Row>>), Error> {
    config.validate()?;
    if sample_seeds.is_empty() {
        return Err(Error::Input(
            "neural sample_seeds must not be empty".into(),
        ));
    }
    let columns = validate_stdio_training_rows(training_rows)?;
    let (script_path, project_root) = resolve_bridge_script(bridge_script)?;
    let fit_seed = config.fit_seed.unwrap_or(config.seed);

    let rows_payload: Vec<Vec<&str>> = training_rows
        .iter()
        .map(|row| {
            columns
                .iter()
                .map(|col| row.fields.get(col).map(String::as_str).unwrap_or(""))
                .collect()
        })
        .collect();
    let request = serde_json::json!({
        "config": config,
        "fit_seed": fit_seed,
        "sample_seeds": sample_seeds,
        "n_samples": config.n_samples,
        "columns": columns,
        "rows": rows_payload,
    });

    let mut child = Command::new("uv")
        .arg("run")
        .arg("--offline")
        .arg("--project")
        .arg(&project_root)
        .arg(&script_path)
        .arg("--stdio")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| Error::Input(format!("failed to execute uv neural bridge: {e}")))?;

    let write_result = {
        let mut stdin = child
            .stdin
            .take()
            .ok_or_else(|| Error::Input("failed to open stdin pipe for neural bridge".into()))?;
        serde_json::to_writer(&mut stdin, &request)
            .map_err(|e| Error::Input(format!("failed to serialize neural bridge request: {e}")))
            .and_then(|()| {
                stdin
                    .flush()
                    .map_err(|e| Error::Input(format!("failed to flush neural bridge stdin: {e}")))
            })
    };

    let output = child
        .wait_with_output()
        .map_err(|e| Error::Input(format!("failed to wait for neural bridge: {e}")))?;

    if let Err(write_err) = write_result {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(Error::Input(format!(
            "{write_err}{}",
            if stderr.trim().is_empty() {
                String::new()
            } else {
                format!(" (stderr: {})", stderr.trim())
            }
        )));
    }

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        let stdout = String::from_utf8_lossy(&output.stdout);
        return Err(Error::Input(format!(
            "neural bridge exited with {}: {}{}",
            output.status,
            stderr.trim(),
            if stdout.trim().is_empty() {
                String::new()
            } else {
                format!("\nstdout: {}", stdout.trim())
            }
        )));
    }

    let response: StdioBridgeResponse = serde_json::from_slice(&output.stdout).map_err(|e| {
        let stdout = String::from_utf8_lossy(&output.stdout);
        Error::Input(format!(
            "invalid neural bridge stdio JSON response: {e} (stdout: {})",
            stdout.trim()
        ))
    })?;

    let draws = parse_stdio_draws(
        &response,
        &columns,
        training_rows.len(),
        fit_seed,
        config.n_samples,
        sample_seeds,
    )?;

    Ok((response.manifest, draws))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generator_kind_serde_roundtrip() {
        let yaml_neural = serde_yaml::to_string(&GeneratorKind::Neural).unwrap();
        assert_eq!(yaml_neural.trim(), "neural-ctgan");
        let parsed: GeneratorKind = serde_yaml::from_str("neural-ctgan").unwrap();
        assert_eq!(parsed, GeneratorKind::Neural);
        let parsed_alias: GeneratorKind = serde_yaml::from_str("neural").unwrap();
        assert_eq!(parsed_alias, GeneratorKind::Neural);
        let parsed_default: GeneratorKind = serde_yaml::from_str("configured").unwrap();
        assert_eq!(parsed_default, GeneratorKind::Configured);
    }

    #[test]
    fn neural_config_validation_rejects_invalid_batch_size_and_pac() {
        let mut cfg = NeuralGeneratorConfig {
            epochs: 5,
            batch_size: 15,
            pac: Some(10),
            ..Default::default()
        };
        assert!(cfg.validate().is_err());

        cfg.pac = Some(5);
        assert!(cfg.validate().is_ok());

        cfg.epochs = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn neural_config_fit_seed_and_sample_seeds_serde_and_validation() {
        let cfg = NeuralGeneratorConfig {
            epochs: 4,
            batch_size: 10,
            pac: Some(2),
            seed: 42,
            fit_seed: Some(99),
            sample_seeds: Some(vec![101, 202]),
            ..Default::default()
        };
        assert!(cfg.validate().is_ok());
        let json = serde_json::to_string(&cfg).unwrap();
        let decoded: NeuralGeneratorConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(cfg, decoded);
        assert_eq!(decoded.fit_seed, Some(99));
        assert_eq!(decoded.sample_seeds, Some(vec![101, 202]));

        // Backwards compatibility when fit_seed / sample_seeds are omitted.
        let minimal: NeuralGeneratorConfig =
            serde_json::from_str(r#"{"epochs": 4, "batch_size": 10, "pac": 2}"#).unwrap();
        assert_eq!(minimal.fit_seed, None);
        assert_eq!(minimal.sample_seeds, None);
        assert!(minimal.validate().is_ok());

        let invalid = NeuralGeneratorConfig {
            epochs: 4,
            batch_size: 10,
            pac: Some(2),
            sample_seeds: Some(Vec::new()),
            ..Default::default()
        };
        assert!(invalid.validate().is_err());
    }

    #[test]
    fn neural_manifest_json_roundtrip() {
        let manifest = NeuralGeneratorManifest {
            schema_version: "1".into(),
            generator_kind: GeneratorKind::Neural,
            model_type: NeuralModelType::Ctgan,
            hyperparameters: serde_json::json!({"epochs": 5, "batch_size": 10, "pac": 2}),
            epochs: 5,
            seed: 42,
            n_samples: 20,
            training_dataset_fingerprint: "sha256:abc123".into(),
            training_rows: 30,
            training_columns: vec!["age".into(), "group".into()],
            discrete_columns: vec!["group".into()],
            samples_fingerprint: "sha256:def456".into(),
            study_fingerprint: Some("fnv1a64:1111".into()),
            config_fingerprint: Some("fnv1a64:2222".into()),
            fixture_fingerprint: None,
            timestamp: "2026-11-13T10:00:00Z".into(),
            python_version: "3.13.18".into(),
            ctgan_version: "0.11.0".into(),
            torch_version: "2.7.0".into(),
            software_version: "0.1.0".into(),
            dependency_lock_fingerprint: Some("fnv1a64-yaml:3333".into()),
            operating_system: Some("linux".into()),
            architecture: Some("x86_64".into()),
            artifacts: vec![
                "samples.csv".into(),
                "model.pkl".into(),
                "manifest.json".into(),
            ],
            reproducibility_limits: "Deterministic seed tested on single environment only.".into(),
        };

        let json = manifest.to_json().unwrap();
        let decoded = NeuralGeneratorManifest::from_json(&json).unwrap();
        assert_eq!(manifest, decoded);
    }

    #[test]
    fn run_neural_bridge_in_memory_rejects_invalid_inputs() {
        let cfg = NeuralGeneratorConfig {
            epochs: 2,
            batch_size: 10,
            pac: Some(2),
            n_samples: 5,
            ..Default::default()
        };

        let empty_rows: Vec<Row> = Vec::new();
        assert!(run_neural_bridge_in_memory(&empty_rows, &cfg, &[101], None).is_err());

        let valid_row = Row::from_pairs([("a", "1"), ("b", "x")]);
        assert!(
            run_neural_bridge_in_memory(std::slice::from_ref(&valid_row), &cfg, &[], None).is_err()
        );

        let mismatched_row = Row::from_pairs([("a", "2"), ("c", "y")]);
        assert!(
            run_neural_bridge_in_memory(&[valid_row, mismatched_row], &cfg, &[101], None).is_err()
        );
    }

    #[test]
    fn run_neural_bridge_in_memory_deterministic_multiseed_and_missingness() {
        let mut rows = Vec::new();
        for i in 0..20 {
            let age = if i % 5 == 0 {
                "NA".to_string()
            } else {
                (30 + i).to_string()
            };
            let group = if i % 4 == 0 {
                "".to_string()
            } else if i % 2 == 0 {
                "control".to_string()
            } else {
                "active".to_string()
            };
            let score = if i % 6 == 0 {
                "na".to_string()
            } else {
                format!("{:.1}", 1.5 + (i as f64) * 0.25)
            };
            rows.push(Row::from_pairs([
                ("age", age.as_str()),
                ("group", group.as_str()),
                ("score", score.as_str()),
                ("all_missing", "NA"),
            ]));
        }

        let cfg = NeuralGeneratorConfig {
            model_type: NeuralModelType::Ctgan,
            epochs: 2,
            batch_size: 10,
            pac: Some(2),
            embedding_dim: 8,
            generator_dim: Some(vec![16, 16]),
            discriminator_dim: Some(vec![16, 16]),
            seed: 42,
            fit_seed: Some(77),
            n_samples: 8,
            ..Default::default()
        };

        let (m1, draws1) = run_neural_bridge_in_memory(&rows, &cfg, &[101, 202], None).unwrap();
        let (m2, draws2) = run_neural_bridge_in_memory(&rows, &cfg, &[101, 202], None).unwrap();

        assert!(m1.artifacts.is_empty());
        assert_eq!(m1.seed, 77);
        assert_eq!(m1.n_samples, 8);
        assert_eq!(m1.training_rows, 20);
        assert_eq!(
            m1.training_columns,
            vec!["age", "all_missing", "group", "score"]
        );
        assert_eq!(m1.samples_fingerprint, m2.samples_fingerprint);
        assert_eq!(
            m1.training_dataset_fingerprint,
            m2.training_dataset_fingerprint
        );
        assert_eq!(draws1, draws2);
        assert_eq!(draws1.len(), 2);
        assert_eq!(draws1[&101].len(), 8);
        assert_eq!(draws1[&202].len(), 8);
        assert_ne!(draws1[&101], draws1[&202]);

        for row in draws1[&101].iter().chain(draws1[&202].iter()) {
            assert_eq!(row.get("all_missing"), Some("NA"));
            assert!(!row.fields.contains_key("__janus_missing_age"));
            assert!(!row.fields.contains_key("__janus_missing_score"));
        }
    }
}
