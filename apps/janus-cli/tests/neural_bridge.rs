use janus::{
    candidate::Row,
    neural::{
        run_neural_bridge_in_memory, GeneratorKind, NeuralGeneratorConfig, NeuralGeneratorManifest,
        NeuralModelType,
    },
};
use std::{
    fs,
    path::Path,
    process::Command,
    sync::atomic::{AtomicU64, Ordering},
};

static COUNTER: AtomicU64 = AtomicU64::new(0);

fn unique_temp_dir(prefix: &str) -> std::path::PathBuf {
    let id = COUNTER.fetch_add(1, Ordering::Relaxed);
    let dir = std::env::temp_dir().join(format!("{prefix}-{}-{id}", std::process::id()));
    fs::create_dir_all(&dir).unwrap();
    dir
}

fn write_fictional_study(path: &Path) {
    fs::write(
        path,
        r#"schema_version: "2"
name: fictional_neural_test
unit: episode
variables:
  - {name: exposure, kind: boolean, role: index_context}
  - {name: age, kind: integer, role: baseline}
  - {name: group, kind: string, role: index_context}
  - {name: outcome, kind: boolean, role: follow_up_outcome}
index: exposure
eligibility: []
follow_up_days: 30
"#,
    )
    .unwrap();
}

fn write_fictional_training_csv(path: &Path) {
    let mut csv = String::from("exposure,age,group,outcome\n");
    for i in 0..20 {
        let exposure = if i % 2 == 0 { "true" } else { "false" };
        let age = 30 + (i * 2) % 40;
        let group = if i % 3 == 0 { "control" } else { "active" };
        let outcome = if i % 4 == 0 { "true" } else { "false" };
        csv.push_str(&format!("{exposure},{age},{group},{outcome}\n"));
    }
    fs::write(path, csv).unwrap();
}

#[test]
fn neural_ctgan_fast_epoch_mock_run_and_deterministic_replay() {
    let dir = unique_temp_dir("janus-neural-ctgan-test");
    let study_path = dir.join("study.yaml");
    let train_path = dir.join("train.csv");
    let config_path = dir.join("ctgan.json");
    let out1 = dir.join("out1");
    let out2 = dir.join("out2");

    write_fictional_study(&study_path);
    write_fictional_training_csv(&train_path);
    fs::write(
        &config_path,
        r#"{
  "model_type": "ctgan",
  "epochs": 5,
  "batch_size": 10,
  "pac": 2,
  "embedding_dim": 16,
  "generator_dim": [32, 32],
  "discriminator_dim": [32, 32],
  "seed": 123,
  "n_samples": 15
}"#,
    )
    .unwrap();

    for out in [&out1, &out2] {
        let output = Command::new(env!("CARGO_BIN_EXE_janus-cli"))
            .args([
                "generate",
                study_path.to_str().unwrap(),
                config_path.to_str().unwrap(),
                out.to_str().unwrap(),
                "--generator",
                "neural-ctgan",
                "--training-data",
                train_path.to_str().unwrap(),
            ])
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "janus-cli generate --generator neural-ctgan failed:\nstdout: {}\nstderr: {}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
    }

    for artifact in [
        "samples.csv",
        "model.pkl",
        "manifest.json",
        "neural_manifest.json",
        "config.json",
        "study.yaml",
    ] {
        assert!(out1.join(artifact).is_file(), "missing artifact {artifact}");
    }

    let samples1 = fs::read_to_string(out1.join("samples.csv")).unwrap();
    let samples2 = fs::read_to_string(out2.join("samples.csv")).unwrap();
    assert_eq!(
        samples1, samples2,
        "deterministic seed must produce identical synthetic samples"
    );
    assert_eq!(samples1.lines().count(), 16, "header + 15 sampled rows");

    let manifest_raw = fs::read_to_string(out1.join("manifest.json")).unwrap();
    let manifest = NeuralGeneratorManifest::from_json(&manifest_raw).unwrap();
    assert_eq!(manifest.generator_kind, GeneratorKind::Neural);
    assert_eq!(manifest.model_type, NeuralModelType::Ctgan);
    assert_eq!(manifest.epochs, 5);
    assert_eq!(manifest.seed, 123);
    assert_eq!(manifest.n_samples, 15);
    assert_eq!(manifest.training_rows, 20);
    assert!(manifest.training_dataset_fingerprint.starts_with("sha256:"));
    assert!(manifest.samples_fingerprint.starts_with("sha256:"));
    assert!(manifest.timestamp.ends_with('Z'));
    assert!(manifest.dependency_lock_fingerprint.is_some());
    assert_eq!(
        manifest.operating_system.as_deref(),
        Some(std::env::consts::OS)
    );
    assert_eq!(
        manifest.architecture.as_deref(),
        Some(std::env::consts::ARCH)
    );

    let audit_output = Command::new(env!("CARGO_BIN_EXE_janus-cli"))
        .args([
            "audit-reproducibility",
            "--manifest",
            out1.join("manifest.json").to_str().unwrap(),
            "--strict",
        ])
        .output()
        .unwrap();
    assert!(
        audit_output.status.success(),
        "audit-reproducibility --strict failed on neural manifest:\nstdout: {}\nstderr: {}",
        String::from_utf8_lossy(&audit_output.stdout),
        String::from_utf8_lossy(&audit_output.stderr)
    );

    let reserialized = manifest.to_json().unwrap();
    let roundtrip = NeuralGeneratorManifest::from_json(&reserialized).unwrap();
    assert_eq!(manifest, roundtrip);

    let manifest2_raw = fs::read_to_string(out2.join("manifest.json")).unwrap();
    let manifest2 = NeuralGeneratorManifest::from_json(&manifest2_raw).unwrap();
    assert_eq!(manifest.samples_fingerprint, manifest2.samples_fingerprint);
    assert_eq!(
        manifest.training_dataset_fingerprint,
        manifest2.training_dataset_fingerprint
    );

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn neural_tvae_fast_epoch_mock_run() {
    let dir = unique_temp_dir("janus-neural-tvae-test");
    let study_path = dir.join("study.yaml");
    let train_path = dir.join("train.csv");
    let config_path = dir.join("tvae.json");
    let out = dir.join("out");

    write_fictional_study(&study_path);
    write_fictional_training_csv(&train_path);
    fs::write(
        &config_path,
        r#"{
  "model_type": "tvae",
  "epochs": 5,
  "batch_size": 10,
  "embedding_dim": 16,
  "compress_dims": [16, 16],
  "decompress_dims": [16, 16],
  "seed": 77,
  "n_samples": 10
}"#,
    )
    .unwrap();

    let output = Command::new(env!("CARGO_BIN_EXE_janus-cli"))
        .args([
            "generate",
            study_path.to_str().unwrap(),
            config_path.to_str().unwrap(),
            out.to_str().unwrap(),
            "--generator",
            "neural-ctgan",
            "--training-data",
            train_path.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "janus-cli generate (TVAE) failed:\nstdout: {}\nstderr: {}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );

    let samples = fs::read_to_string(out.join("samples.csv")).unwrap();
    assert_eq!(samples.lines().count(), 11, "header + 10 sampled rows");

    let manifest_raw = fs::read_to_string(out.join("manifest.json")).unwrap();
    let manifest = NeuralGeneratorManifest::from_json(&manifest_raw).unwrap();
    assert_eq!(manifest.model_type, NeuralModelType::Tvae);
    assert_eq!(manifest.epochs, 5);
    assert_eq!(manifest.seed, 77);
    assert_eq!(manifest.n_samples, 10);

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn neural_in_memory_multiseed_replay_and_missingness_without_disk_io() {
    let mut rows = Vec::new();
    for i in 0..20 {
        let age = if i % 4 == 0 {
            "NA".to_string()
        } else {
            (30 + (i % 8) * 3).to_string()
        };
        let group = if i % 5 == 0 {
            "".to_string()
        } else if i % 2 == 0 {
            "control".to_string()
        } else {
            "active".to_string()
        };
        let score = if i % 3 == 0 {
            "na".to_string()
        } else {
            format!("{:.2}", 10.5 + ((i % 5) as f64) * 1.25)
        };
        rows.push(Row::from_pairs([
            ("age", age.as_str()),
            ("group", group.as_str()),
            ("score", score.as_str()),
            ("unobserved", ""),
        ]));
    }

    let ctgan_cfg = NeuralGeneratorConfig {
        model_type: NeuralModelType::Ctgan,
        epochs: 2,
        batch_size: 10,
        pac: Some(2),
        embedding_dim: 8,
        generator_dim: Some(vec![16, 16]),
        discriminator_dim: Some(vec![16, 16]),
        seed: 42,
        fit_seed: Some(111),
        sample_seeds: Some(vec![101, 202]),
        n_samples: 10,
        ..Default::default()
    };

    let (m1, draws1) = run_neural_bridge_in_memory(&rows, &ctgan_cfg, &[101, 202], None).unwrap();
    let (m2, draws2) = run_neural_bridge_in_memory(&rows, &ctgan_cfg, &[101, 202], None).unwrap();

    assert!(m1.artifacts.is_empty());
    assert_eq!(m1.seed, 111);
    assert_eq!(m1.samples_fingerprint, m2.samples_fingerprint);
    assert_eq!(
        m1.training_dataset_fingerprint,
        m2.training_dataset_fingerprint
    );
    assert_eq!(draws1, draws2);
    assert_eq!(draws1.len(), 2);
    assert_ne!(draws1[&101], draws1[&202]);

    for row in draws1[&101].iter().chain(draws1[&202].iter()) {
        assert_eq!(row.get("unobserved"), Some("NA"));
        let group = row.get("group").unwrap();
        assert!(matches!(group, "control" | "active" | "NA"));
        for numeric_col in ["age", "score"] {
            let val = row.get(numeric_col).unwrap();
            assert!(!val.is_empty());
            if val != "NA" {
                let parsed: f64 = val.parse().unwrap();
                assert_ne!(parsed, 0.0);
            }
        }
        assert!(!row.fields.contains_key("__janus_missing_age"));
        assert!(!row.fields.contains_key("__janus_missing_score"));
    }

    let tvae_cfg = NeuralGeneratorConfig {
        model_type: NeuralModelType::Tvae,
        epochs: 2,
        batch_size: 10,
        embedding_dim: 8,
        compress_dims: Some(vec![16, 16]),
        decompress_dims: Some(vec![16, 16]),
        seed: 77,
        n_samples: 6,
        ..Default::default()
    };
    let (tm1, tdraws1) = run_neural_bridge_in_memory(&rows, &tvae_cfg, &[101, 202], None).unwrap();
    let (tm2, tdraws2) = run_neural_bridge_in_memory(&rows, &tvae_cfg, &[101, 202], None).unwrap();
    assert!(tm1.artifacts.is_empty());
    assert_eq!(tm1.samples_fingerprint, tm2.samples_fingerprint);
    assert_eq!(tdraws1, tdraws2);
    assert_eq!(tdraws1[&101].len(), 6);
    assert_eq!(tdraws1[&202].len(), 6);
}

#[test]
fn neural_generate_in_memory_aggregate_writes_no_samples_or_model() {
    let dir = unique_temp_dir("janus-neural-in-memory-agg");
    let study_path = dir.join("study.yaml");
    let train_path = dir.join("train.csv");
    let config_path = dir.join("ctgan.json");
    let out = dir.join("out");

    write_fictional_study(&study_path);
    write_fictional_training_csv(&train_path);
    fs::write(
        &config_path,
        r#"{
  "model_type": "ctgan",
  "epochs": 2,
  "batch_size": 10,
  "pac": 2,
  "embedding_dim": 8,
  "generator_dim": [16, 16],
  "discriminator_dim": [16, 16],
  "seed": 42,
  "fit_seed": 77,
  "sample_seeds": [101, 202],
  "n_samples": 10
}"#,
    )
    .unwrap();

    let output = Command::new(env!("CARGO_BIN_EXE_janus-cli"))
        .args([
            "generate",
            study_path.to_str().unwrap(),
            config_path.to_str().unwrap(),
            out.to_str().unwrap(),
            "--generator",
            "neural-ctgan",
            "--training-data",
            train_path.to_str().unwrap(),
            "--in-memory-aggregate",
        ])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "generate --in-memory-aggregate failed:\nstdout: {}\nstderr: {}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );

    for expected in [
        "study.yaml",
        "config.json",
        "aggregate_summary.json",
        "manifest.json",
    ] {
        assert!(out.join(expected).is_file(), "missing artifact {expected}");
    }
    for forbidden in [
        "samples.csv",
        "model.pkl",
        "neural_manifest.json",
        "fixture.yaml",
    ] {
        assert!(
            !out.join(forbidden).exists(),
            "forbidden artifact written in aggregate-only mode: {forbidden}"
        );
    }

    let manifest =
        NeuralGeneratorManifest::from_json(&fs::read_to_string(out.join("manifest.json")).unwrap())
            .unwrap();
    assert_eq!(manifest.fixture_fingerprint, None);
    assert_eq!(manifest.seed, 77);
    assert_eq!(manifest.n_samples, 10);
    assert_eq!(
        manifest.artifacts,
        vec![
            "study.yaml",
            "config.json",
            "aggregate_summary.json",
            "manifest.json",
        ]
    );

    let summary: serde_json::Value =
        serde_json::from_str(&fs::read_to_string(out.join("aggregate_summary.json")).unwrap())
            .unwrap();
    assert_eq!(summary["training_rows"], 20);
    assert_eq!(summary["rows_generated_per_seed"], 10);
    assert_eq!(summary["sample_seeds"], serde_json::json!([101, 202]));
    let exp_obs = summary["variable_counts_by_seed"]["101"]["exposure"]["observed"]
        .as_u64()
        .unwrap();
    let exp_miss = summary["variable_counts_by_seed"]["101"]["exposure"]["missing"]
        .as_u64()
        .unwrap();
    assert_eq!(exp_obs + exp_miss, 10);

    let audit_output = Command::new(env!("CARGO_BIN_EXE_janus-cli"))
        .args([
            "audit-reproducibility",
            "--manifest",
            out.join("manifest.json").to_str().unwrap(),
            "--strict",
        ])
        .output()
        .unwrap();
    assert!(
        audit_output.status.success(),
        "audit-reproducibility --strict failed:\nstdout: {}\nstderr: {}",
        String::from_utf8_lossy(&audit_output.stdout),
        String::from_utf8_lossy(&audit_output.stderr)
    );

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn neural_generate_emits_evaluable_fixture_for_evaluate_pipeline() {
    let dir = unique_temp_dir("janus-neural-eval-pipeline");
    let study_path = dir.join("study.yaml");
    let train_path = dir.join("train.csv");
    let config_path = dir.join("ctgan.json");
    let neural_out = dir.join("neural_out");
    let ref_path = dir.join("reference.yaml");
    let eval_cfg_path = dir.join("evaluation.yaml");
    let eval_out = dir.join("eval_out");

    write_fictional_study(&study_path);
    write_fictional_training_csv(&train_path);
    fs::write(
        &config_path,
        r#"{
  "model_type": "ctgan",
  "epochs": 2,
  "batch_size": 10,
  "pac": 2,
  "embedding_dim": 8,
  "generator_dim": [16, 16],
  "discriminator_dim": [16, 16],
  "seed": 42,
  "n_samples": 12
}"#,
    )
    .unwrap();

    let gen_out = Command::new(env!("CARGO_BIN_EXE_janus-cli"))
        .args([
            "generate",
            study_path.to_str().unwrap(),
            config_path.to_str().unwrap(),
            neural_out.to_str().unwrap(),
            "--generator",
            "neural-ctgan",
            "--training-data",
            train_path.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(
        gen_out.status.success(),
        "generate --generator neural-ctgan failed:\nstderr: {}",
        String::from_utf8_lossy(&gen_out.stderr)
    );

    assert!(neural_out.join("fixture.yaml").is_file());
    assert!(neural_out.join("cohort.yaml").is_file());
    assert!(neural_out.join("summary.yaml").is_file());
    let manifest = NeuralGeneratorManifest::from_json(
        &fs::read_to_string(neural_out.join("manifest.json")).unwrap(),
    )
    .unwrap();
    assert!(
        manifest.fixture_fingerprint.is_some(),
        "expected fixture_fingerprint on neural manifest"
    );

    let mut events = Vec::new();
    for idx in 0..8 {
        let pid = format!("ref-p{idx:03}");
        let eid = format!("ref-e{idx:03}");
        let exposure = if idx % 2 == 0 { "true" } else { "false" };
        let age = (35 + idx * 3).to_string();
        let group = if idx % 2 == 0 { "active" } else { "control" };
        let outcome = if idx % 3 == 0 { "true" } else { "false" };
        for (v_idx, (var, val, day)) in [
            ("exposure", exposure, 0),
            ("age", age.as_str(), 0),
            ("group", group, 0),
            ("outcome", outcome, 1),
        ]
        .into_iter()
        .enumerate()
        {
            events.push(janus::Event {
                patient_id: pid.clone(),
                episode_id: eid.clone(),
                id: format!("ref-ev-{idx:03}-{v_idx:02}"),
                variable: var.into(),
                value: Some(val.into()),
                missing: None,
                event_day: day,
                recording_day: day,
                provenance: "fictional_reference".into(),
            });
        }
    }
    let reference = janus::Fixture {
        schema_version: "1".into(),
        events,
        observations: Vec::new(),
    };
    fs::write(&ref_path, serde_yaml::to_string(&reference).unwrap()).unwrap();
    fs::write(
        &eval_cfg_path,
        "schema_version: \"1\"\njoint:\n  exposure: exposure\n  outcome: outcome\n",
    )
    .unwrap();

    let eval_cmd = Command::new(env!("CARGO_BIN_EXE_janus-cli"))
        .args([
            "evaluate",
            study_path.to_str().unwrap(),
            neural_out.join("fixture.yaml").to_str().unwrap(),
            ref_path.to_str().unwrap(),
            eval_cfg_path.to_str().unwrap(),
            eval_out.to_str().unwrap(),
            "--synthetic-manifest",
            neural_out.join("manifest.json").to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(
        eval_cmd.status.success(),
        "evaluate with neural fixture failed:\nstdout: {}\nstderr: {}",
        String::from_utf8_lossy(&eval_cmd.stdout),
        String::from_utf8_lossy(&eval_cmd.stderr)
    );
    assert!(eval_out.join("evaluation.yaml").is_file());
    assert!(eval_out.join("manifest.yaml").is_file());

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn compare_candidates_cli_runs_forest_dag_and_neural_in_memory() {
    let dir = unique_temp_dir("janus-compare-candidates-cli");
    let contract_path = dir.join("contract.yaml");
    let cmp_cfg_path = dir.join("comparison.yaml");
    let neural_cfg_path = dir.join("neural.json");
    let train_csv = dir.join("train.csv");
    let ref_csv = dir.join("reference.csv");
    let out = dir.join("out");

    fs::write(
        &contract_path,
        r#"schema_version: "1"
contract_id: fictional-compare-v1
status: fictional_test_only
study: FictionalStudy
patient_unit: patient
variables:
  - name: ha_wfns
    role: admission_context
    meaning: fictional grade
    timing: admission
    units: grade
    allowed_values: ["1", "2", "3", "4", "5"]
    missing_is_structural_absence: false
  - name: intervention
    role: index_context
    meaning: fictional intervention
    timing: index
    units: category
    allowed_values: ["evd", "lumbar_drain"]
    missing_is_structural_absence: false
  - name: fup_mrs
    role: follow_up_outcome
    meaning: fictional outcome
    timing: followup
    units: grade
    allowed_values: ["0", "1", "2", "3", "5"]
    missing_is_structural_absence: false
limitations:
  - Fictional unit test fixture only.
unresolved_assumptions: []
"#,
    )
    .unwrap();

    fs::write(
        &cmp_cfg_path,
        r#"schema_version: "1"
protocol_id: fictional-cli-compare-v1
contract_id: fictional-compare-v1
preprocessing: shared_in_memory_csv_missing_preserved
candidate_status: fictional_benchmark
populations: [fictional_train, fictional_reference, same_seed_synthetic]
candidate_names: [forest, dag, interaction-dag, neural-ctgan]
split_fingerprint: fnv1a64-split:fictional-cli-compare-v1
sample_count_per_seed: 20
seeds: [11, 22]
metrics:
  marginal_tvd: true
  pairwise_tvd: true
  joint_tvd: true
  utility_brier:
    task_name: fictional_mrs_prediction
    predictor: ha_wfns
    additional_predictors: [intervention]
    target: fup_mrs
    positive_values: ["3", "4", "5", "6"]
    negative_values: ["0", "1", "2"]
    alpha: 1.0
    predictor_available_before_target: true
"#,
    )
    .unwrap();

    fs::write(
        &neural_cfg_path,
        r#"{
  "model_type": "ctgan",
  "epochs": 2,
  "batch_size": 10,
  "pac": 2,
  "embedding_dim": 8,
  "generator_dim": [16, 16],
  "discriminator_dim": [16, 16],
  "seed": 42,
  "n_samples": 20
}"#,
    )
    .unwrap();

    let mut train_body = String::from("ha_wfns,intervention,fup_mrs\n");
    let mut ref_body = String::from("ha_wfns,intervention,fup_mrs\n");
    for i in 0..24 {
        let wfns = (i % 5) + 1;
        let intervention = if i % 2 == 0 { "evd" } else { "lumbar_drain" };
        let mrs = if i % 11 == 0 {
            "NA"
        } else if i % 13 == 0 {
            ""
        } else if wfns <= 2 {
            "1"
        } else {
            "5"
        };
        train_body.push_str(&format!("{wfns},{intervention},{mrs}\n"));
        let ref_mrs = if i % 9 == 0 {
            "NA"
        } else if wfns <= 2 {
            "2"
        } else {
            "3"
        };
        ref_body.push_str(&format!("{wfns},{intervention},{ref_mrs}\n"));
    }
    fs::write(&train_csv, train_body).unwrap();
    fs::write(&ref_csv, ref_body).unwrap();

    let cmp_out = Command::new(env!("CARGO_BIN_EXE_janus-cli"))
        .args([
            "compare-candidates",
            "--config",
            cmp_cfg_path.to_str().unwrap(),
            "--training",
            train_csv.to_str().unwrap(),
            "--reference",
            ref_csv.to_str().unwrap(),
            "--contract",
            contract_path.to_str().unwrap(),
            "--neural-config",
            neural_cfg_path.to_str().unwrap(),
            "--output",
            out.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(
        cmp_out.status.success(),
        "compare-candidates failed:\nstdout: {}\nstderr: {}",
        String::from_utf8_lossy(&cmp_out.stdout),
        String::from_utf8_lossy(&cmp_out.stderr)
    );

    assert!(out.join("comparison_manifest.json").is_file());
    assert!(out.join("forensic_manifests.json").is_file());
    assert_eq!(fs::read_dir(&out).unwrap().count(), 2);
    assert!(!out.join("samples.csv").exists());
    assert!(!out.join("model.pkl").exists());

    let comparison_manifest: janus::AggregateComparisonManifest = serde_json::from_str(
        &fs::read_to_string(out.join("comparison_manifest.json")).unwrap(),
    )
    .unwrap();
    comparison_manifest.validate().unwrap();
    assert_eq!(comparison_manifest.candidate_summaries.len(), 4);

    let forensic_entries: Vec<serde_json::Value> = serde_json::from_str(
        &fs::read_to_string(out.join("forensic_manifests.json")).unwrap(),
    )
    .unwrap();
    assert_eq!(forensic_entries.len(), 8);
    for entry in &forensic_entries {
        let fm: janus::ForensicManifest =
            serde_json::from_value(entry["manifest"].clone()).unwrap();
        fm.validate().unwrap();
    }

    // Mismatched contract_id must fail preflight before creating output
    let bad_contract = dir.join("bad_contract.yaml");
    fs::write(
        &bad_contract,
        fs::read_to_string(&contract_path)
            .unwrap()
            .replace("fictional-compare-v1", "different-contract-v2"),
    )
    .unwrap();
    let bad_out = dir.join("bad_out");
    let rejected = Command::new(env!("CARGO_BIN_EXE_janus-cli"))
        .args([
            "compare-candidates",
            "--config",
            cmp_cfg_path.to_str().unwrap(),
            "--training",
            train_csv.to_str().unwrap(),
            "--reference",
            ref_csv.to_str().unwrap(),
            "--contract",
            bad_contract.to_str().unwrap(),
            "--neural-config",
            neural_cfg_path.to_str().unwrap(),
            "--output",
            bad_out.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(!rejected.status.success());
    assert!(!bad_out.exists());

    fs::remove_dir_all(dir).unwrap();
}
