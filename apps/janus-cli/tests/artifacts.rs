use janus::Fixture;
use std::{fs, path::Path, process::Command};

fn cli(args: &[&str]) -> bool {
    Command::new(env!("CARGO_BIN_EXE_janus-cli"))
        .args(args)
        .output()
        .unwrap()
        .status
        .success()
}

#[test]
fn numeric_smoothing_replays_and_reports_aggregate_counts() {
    let root =
        std::env::temp_dir().join(format!("janus-flat-smoothing-cli-{}", std::process::id()));
    let data = root.join("datasets");
    fs::create_dir_all(&data).unwrap();
    let source = data.join("fictional.csv");
    let config = data.join("config.yaml");
    let mut input = String::from("record,site,value\n");
    for i in 0..40 {
        input.push_str(&format!("p{i},one,{}\n", i % 8 + 1));
    }
    fs::write(&source, input).unwrap();
    fs::write(&config, "schema_version: '1'\nrow_key_column: record\nallocation_column: site\nallocation_values: [one]\ngenerator: regularized_binned_forest\ndevelopment_proportion: 0.5\ndevelopment_validation_proportion: 0.25\nnumeric_smoothing: [{variable: value, bandwidth: 0.25, minimum: 0.0, maximum: 10.0}]\nbootstrap_replicates: 100\nthresholds: {max_missingness_rate_difference: 1.0, max_categorical_tvd: 1.0, max_numeric_quantile_difference: 10.0}\nvariables:\n  - {column: value, kind: numeric}\n").unwrap();
    for name in ["first", "replay"] {
        assert!(cli(&[
            "describe-flat",
            source.to_str().unwrap(),
            config.to_str().unwrap(),
            data.join(name).to_str().unwrap()
        ]));
    }
    for artifact in ["report.yaml", "manifest.yaml"] {
        assert_eq!(
            fs::read(data.join("first").join(artifact)).unwrap(),
            fs::read(data.join("replay").join(artifact)).unwrap()
        );
    }
    let report: serde_yaml::Value =
        serde_yaml::from_slice(&fs::read(data.join("first/report.yaml")).unwrap()).unwrap();
    assert!(report["numeric_smoothing"][0]["changed"].as_u64().unwrap() > 0);
    assert!(
        report["development_validation"][0]["numeric_smoothing"][0]["changed"]
            .as_u64()
            .unwrap()
            > 0
    );
    assert_eq!(fs::read_dir(data.join("first")).unwrap().count(), 2);
    let interpolation_config = data.join("interpolation.yaml");
    let yaml = fs::read_to_string(&config).unwrap().replace(
        "numeric_smoothing: [{variable: value, bandwidth: 0.25, minimum: 0.0, maximum: 10.0}]",
        "numeric_interpolation: [value]",
    );
    fs::write(&interpolation_config, yaml).unwrap();
    for name in ["interpolated", "interpolated-replay"] {
        assert!(cli(&[
            "describe-flat",
            source.to_str().unwrap(),
            interpolation_config.to_str().unwrap(),
            data.join(name).to_str().unwrap()
        ]));
    }
    for artifact in ["report.yaml", "manifest.yaml"] {
        assert_eq!(
            fs::read(data.join("interpolated").join(artifact)).unwrap(),
            fs::read(data.join("interpolated-replay").join(artifact)).unwrap()
        );
    }
    let interpolated: serde_yaml::Value =
        serde_yaml::from_slice(&fs::read(data.join("interpolated/report.yaml")).unwrap()).unwrap();
    assert!(
        interpolated["numeric_interpolation"][0]["changed"]
            .as_u64()
            .unwrap()
            > 0
    );
    assert_eq!(fs::read_dir(data.join("interpolated")).unwrap().count(), 2);
    fs::remove_dir_all(root).unwrap();
}

#[test]
fn learned_flat_generator_replays_with_configured_row_key() {
    let root = std::env::temp_dir().join(format!("janus-flat-forest-cli-{}", std::process::id()));
    let data = root.join("datasets");
    fs::create_dir_all(&data).unwrap();
    let source = data.join("fictional.csv");
    let config = data.join("config.yaml");
    let mut input = String::from("record,site,baseline,outcome\n");
    for i in 0..40 {
        let (baseline, outcome) = if i % 2 == 0 {
            ("low", "a")
        } else {
            ("high", "b")
        };
        input.push_str(&format!("p{i},one,{baseline},{outcome}\n"));
    }
    fs::write(&source, input).unwrap();
    fs::write(&config, "schema_version: '1'\nrow_key_column: record\nallocation_column: site\nallocation_values: [one]\ngenerator: learned_categorical_forest\ndevelopment_proportion: 0.5\nbootstrap_replicates: 100\nthresholds: {max_missingness_rate_difference: 1.0, max_categorical_tvd: 1.0, max_numeric_quantile_difference: 1.0}\nvariables:\n  - {column: baseline, kind: categorical, allowed_values: [low, high]}\n  - {column: outcome, kind: categorical, allowed_values: [a, b]}\n").unwrap();
    for name in ["first", "replay"] {
        assert!(cli(&[
            "describe-flat",
            source.to_str().unwrap(),
            config.to_str().unwrap(),
            data.join(name).to_str().unwrap()
        ]));
    }
    for artifact in ["report.yaml", "manifest.yaml"] {
        assert_eq!(
            fs::read(data.join("first").join(artifact)).unwrap(),
            fs::read(data.join("replay").join(artifact)).unwrap()
        );
    }
    let report: serde_yaml::Value =
        serde_yaml::from_slice(&fs::read(data.join("first/report.yaml")).unwrap()).unwrap();
    assert_eq!(
        report["learned_dependencies"].as_sequence().unwrap().len(),
        1
    );
    assert_eq!(
        report["joint_generated_vs_holdout"]
            .as_sequence()
            .unwrap()
            .len(),
        1
    );
    fs::remove_dir_all(root).unwrap();
}

#[test]
fn uncertainty_workflow_replays_and_rejects_invalid_configuration() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let dir = std::env::temp_dir().join(format!("janus-uncertainty-test-{}", std::process::id()));
    fs::create_dir_all(&dir).unwrap();
    let study = root.join("examples/study.yaml");
    let split = dir.join("split");
    let generated = dir.join("generated");
    assert!(cli(&[
        "split",
        study.to_str().unwrap(),
        root.join("examples/fixture.yaml").to_str().unwrap(),
        root.join("examples/split-for-fit.yaml").to_str().unwrap(),
        split.to_str().unwrap()
    ]));
    assert!(cli(&[
        "generate",
        study.to_str().unwrap(),
        root.join("examples/generator.yaml").to_str().unwrap(),
        generated.to_str().unwrap()
    ]));
    for name in ["a", "b"] {
        assert!(cli(&[
            "evaluate",
            study.to_str().unwrap(),
            generated.join("fixture.yaml").to_str().unwrap(),
            split.join("holdout.yaml").to_str().unwrap(),
            root.join("examples/evaluation-uncertainty.yaml")
                .to_str()
                .unwrap(),
            dir.join(name).to_str().unwrap(),
            "--split-manifest",
            split.join("manifest.yaml").to_str().unwrap()
        ]));
    }
    for name in ["evaluation.yaml", "manifest.yaml"] {
        assert_eq!(
            fs::read(dir.join("a").join(name)).unwrap(),
            fs::read(dir.join("b").join(name)).unwrap()
        );
    }
    let report: serde_yaml::Value =
        serde_yaml::from_slice(&fs::read(dir.join("a/evaluation.yaml")).unwrap()).unwrap();
    assert_eq!(
        report["uncertainty"]["configuration"]["replicates"].as_u64(),
        Some(1000)
    );
    assert!(
        report["pairs"]["severity_outcome"]["categories"][0]["comparison"]["uncertainty"]
            .is_mapping()
    );
    let manifest: serde_yaml::Value =
        serde_yaml::from_slice(&fs::read(dir.join("a/manifest.yaml")).unwrap()).unwrap();
    assert!(manifest["uncertainty"]["method"]
        .as_str()
        .unwrap()
        .contains("patient-level"));
    let invalid = dir.join("invalid.yaml");
    fs::write(
        &invalid,
        "schema_version: '1'\nuncertainty: {seed: 1, replicates: 0, confidence_level: 0.95}\n",
    )
    .unwrap();
    let rejected = dir.join("rejected");
    assert!(!cli(&[
        "evaluate",
        study.to_str().unwrap(),
        generated.join("fixture.yaml").to_str().unwrap(),
        split.join("holdout.yaml").to_str().unwrap(),
        invalid.to_str().unwrap(),
        rejected.to_str().unwrap()
    ]));
    assert!(!rejected.exists());
    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn evidence_workflow_replays_and_checks_registry_identity() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let dir = std::env::temp_dir().join(format!("janus-evidence-test-{}", std::process::id()));
    fs::create_dir_all(&dir).unwrap();
    let study = root.join("examples/study-evidence.yaml");
    let generator = root.join("examples/generator-repeated-evidence.yaml");
    let registry = root.join("examples/evidence.yaml");
    let split = dir.join("split");
    let fitted = dir.join("fit");
    let generated = dir.join("generated");
    let s = study.to_str().unwrap();
    let g = generator.to_str().unwrap();
    let r = registry.to_str().unwrap();
    for name in ["generated", "replay"] {
        assert!(cli(&[
            "generate",
            s,
            g,
            dir.join(name).to_str().unwrap(),
            "--evidence-registry",
            r
        ]));
    }
    for file in ["fixture.yaml", "manifest.yaml"] {
        assert_eq!(
            fs::read(generated.join(file)).unwrap(),
            fs::read(dir.join("replay").join(file)).unwrap()
        );
    }
    assert!(cli(&[
        "split",
        s,
        root.join("examples/fixture.yaml").to_str().unwrap(),
        root.join("examples/split-for-fit.yaml").to_str().unwrap(),
        split.to_str().unwrap(),
        "--evidence-registry",
        r
    ]));
    assert!(cli(&[
        "fit",
        s,
        split.join("development.yaml").to_str().unwrap(),
        root.join("examples/fit.yaml").to_str().unwrap(),
        fitted.to_str().unwrap(),
        "--evidence-registry",
        r,
        "--split-manifest",
        split.join("manifest.yaml").to_str().unwrap()
    ]));
    let eval = dir.join("evaluation");
    assert!(cli(&[
        "evaluate",
        s,
        generated.join("fixture.yaml").to_str().unwrap(),
        split.join("holdout.yaml").to_str().unwrap(),
        root.join("examples/evaluation.yaml").to_str().unwrap(),
        eval.to_str().unwrap(),
        "--evidence-registry",
        r,
        "--split-manifest",
        split.join("manifest.yaml").to_str().unwrap(),
        "--synthetic-manifest",
        generated.join("manifest.yaml").to_str().unwrap()
    ]));
    let manifest = |path: &Path| -> serde_yaml::Value {
        serde_yaml::from_slice(&fs::read(path).unwrap()).unwrap()
    };
    let gen_manifest = manifest(&generated.join("manifest.yaml"));
    let fp = &gen_manifest["evidence"]["registry_fingerprint"];
    for path in [
        split.join("manifest.yaml"),
        fitted.join("manifest.yaml"),
        eval.join("manifest.yaml"),
    ] {
        assert_eq!(&manifest(&path)["evidence"]["registry_fingerprint"], fp);
    }
    assert_eq!(
        manifest(&eval.join("manifest.yaml"))["evidence"]["referenced_ids"]
            .as_sequence()
            .unwrap()
            .len(),
        5
    );
    let changed = dir.join("changed.yaml");
    fs::write(
        &changed,
        fs::read_to_string(&registry).unwrap().replace(
            "Illustrative protocol duration.",
            "Changed fictional duration rationale.",
        ),
    )
    .unwrap();
    assert!(cli(&[
        "generate",
        s,
        g,
        dir.join("changed-generated").to_str().unwrap(),
        "--evidence-registry",
        changed.to_str().unwrap()
    ]));
    assert_eq!(
        fs::read(generated.join("fixture.yaml")).unwrap(),
        fs::read(dir.join("changed-generated/fixture.yaml")).unwrap()
    );
    assert_ne!(
        &manifest(&dir.join("changed-generated/manifest.yaml"))["evidence"]["registry_fingerprint"],
        fp
    );
    let dangling = dir.join("dangling.yaml");
    fs::write(
        &dangling,
        fs::read_to_string(&registry).unwrap().replace(
            "generator.repeated.outcome.count",
            "generator.repeated.unknown.count",
        ),
    )
    .unwrap();
    let rejected = dir.join("rejected");
    assert!(!cli(&[
        "generate",
        s,
        g,
        rejected.to_str().unwrap(),
        "--evidence-registry",
        dangling.to_str().unwrap()
    ]));
    assert!(!rejected.exists());
    assert!(cli(&[
        "generate",
        s,
        fitted.join("generator.yaml").to_str().unwrap(),
        dir.join("fitted-generated").to_str().unwrap(),
        "--evidence-registry",
        r
    ]));
    let mismatched_eval = dir.join("mismatched-evaluation");
    assert!(!cli(&[
        "evaluate",
        s,
        generated.join("fixture.yaml").to_str().unwrap(),
        split.join("holdout.yaml").to_str().unwrap(),
        root.join("examples/evaluation.yaml").to_str().unwrap(),
        mismatched_eval.to_str().unwrap(),
        "--evidence-registry",
        changed.to_str().unwrap(),
        "--split-manifest",
        split.join("manifest.yaml").to_str().unwrap(),
        "--synthetic-manifest",
        generated.join("manifest.yaml").to_str().unwrap()
    ]));
    assert!(!mismatched_eval.exists());
    for (name, registry_arg) in [("missing", None), ("changed-fit", Some(changed.as_path()))] {
        let output = dir.join(name);
        let mut args = vec![
            "generate".to_string(),
            s.to_string(),
            fitted.join("generator.yaml").to_str().unwrap().to_owned(),
            output.to_str().unwrap().to_owned(),
        ];
        if let Some(path) = registry_arg {
            args.push("--evidence-registry".into());
            args.push(path.to_str().unwrap().into());
        }
        let args: Vec<&str> = args.iter().map(String::as_str).collect();
        assert!(!cli(&args));
        assert!(!output.exists());
    }
    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn longitudinal_workflow_replays() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let dir = std::env::temp_dir().join(format!("janus-longitudinal-test-{}", std::process::id()));
    fs::create_dir_all(&dir).unwrap();
    let study = root.join("examples/study.yaml");
    let split = dir.join("split");
    let generated = dir.join("generated");
    assert!(cli(&[
        "split",
        study.to_str().unwrap(),
        root.join("examples/fixture-longitudinal.yaml")
            .to_str()
            .unwrap(),
        root.join("examples/split-for-fit.yaml").to_str().unwrap(),
        split.to_str().unwrap()
    ]));
    assert!(cli(&[
        "generate",
        study.to_str().unwrap(),
        root.join("examples/generator-repeated-zero.yaml")
            .to_str()
            .unwrap(),
        dir.join("timed-gen").to_str().unwrap()
    ]));
    assert!(cli(&[
        "evaluate",
        study.to_str().unwrap(),
        dir.join("timed-gen/fixture.yaml").to_str().unwrap(),
        split.join("holdout.yaml").to_str().unwrap(),
        root.join("examples/evaluation-longitudinal.yaml")
            .to_str()
            .unwrap(),
        dir.join("timed-eval").to_str().unwrap(),
        "--split-manifest",
        split.join("manifest.yaml").to_str().unwrap()
    ]));
    assert!(cli(&[
        "generate",
        study.to_str().unwrap(),
        root.join("examples/generator.yaml").to_str().unwrap(),
        generated.to_str().unwrap()
    ]));
    for name in ["a", "b"] {
        assert!(cli(&[
            "evaluate",
            study.to_str().unwrap(),
            generated.join("fixture.yaml").to_str().unwrap(),
            split.join("holdout.yaml").to_str().unwrap(),
            root.join("examples/evaluation-longitudinal.yaml")
                .to_str()
                .unwrap(),
            dir.join(name).to_str().unwrap(),
            "--split-manifest",
            split.join("manifest.yaml").to_str().unwrap()
        ]));
    }
    for file in ["evaluation.yaml", "manifest.yaml"] {
        assert_eq!(
            fs::read(dir.join("a").join(file)).unwrap(),
            fs::read(dir.join("b").join(file)).unwrap()
        );
    }
    let report: serde_yaml::Value =
        serde_yaml::from_slice(&fs::read(dir.join("a/evaluation.yaml")).unwrap()).unwrap();
    assert!(report["longitudinal"]["variable_counts"]["outcome"].is_mapping());
    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn repeated_event_generator_replays_and_evaluates() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let dir = std::env::temp_dir().join(format!("janus-repeated-test-{}", std::process::id()));
    fs::create_dir_all(&dir).unwrap();
    let study = root.join("examples/study.yaml");
    let split = dir.join("split");
    assert!(cli(&[
        "split",
        study.to_str().unwrap(),
        root.join("examples/fixture-longitudinal.yaml")
            .to_str()
            .unwrap(),
        root.join("examples/split-for-fit.yaml").to_str().unwrap(),
        split.to_str().unwrap()
    ]));
    assert!(cli(&[
        "generate",
        study.to_str().unwrap(),
        root.join("examples/generator-repeated-zero.yaml")
            .to_str()
            .unwrap(),
        dir.join("timed-gen").to_str().unwrap()
    ]));
    assert!(cli(&[
        "evaluate",
        study.to_str().unwrap(),
        dir.join("timed-gen/fixture.yaml").to_str().unwrap(),
        split.join("holdout.yaml").to_str().unwrap(),
        root.join("examples/evaluation-longitudinal.yaml")
            .to_str()
            .unwrap(),
        dir.join("timed-eval").to_str().unwrap(),
        "--split-manifest",
        split.join("manifest.yaml").to_str().unwrap()
    ]));
    for name in ["a", "b"] {
        assert!(cli(&[
            "generate",
            study.to_str().unwrap(),
            root.join("examples/generator-repeated.yaml")
                .to_str()
                .unwrap(),
            dir.join(format!("{name}-gen")).to_str().unwrap()
        ]));
        assert!(cli(&[
            "evaluate",
            study.to_str().unwrap(),
            dir.join(format!("{name}-gen/fixture.yaml"))
                .to_str()
                .unwrap(),
            split.join("holdout.yaml").to_str().unwrap(),
            root.join("examples/evaluation-longitudinal.yaml")
                .to_str()
                .unwrap(),
            dir.join(format!("{name}-eval")).to_str().unwrap(),
            "--split-manifest",
            split.join("manifest.yaml").to_str().unwrap()
        ]));
    }
    for name in ["fixture.yaml", "manifest.yaml"] {
        assert_eq!(
            fs::read(dir.join("a-gen").join(name)).unwrap(),
            fs::read(dir.join("b-gen").join(name)).unwrap()
        );
    }
    let fixture: Fixture =
        serde_yaml::from_slice(&fs::read(dir.join("a-gen/fixture.yaml")).unwrap()).unwrap();
    assert_eq!(fixture.observations.len(), 100);
    assert!(fixture
        .events
        .iter()
        .filter(|e| e.variable == "outcome")
        .all(|e| (0..=365).contains(&e.event_day) && e.recording_day == e.event_day));
    let mut counts = std::collections::BTreeMap::new();
    let mut previous: std::collections::BTreeMap<String, (i64, String)> =
        std::collections::BTreeMap::new();
    let mut ids = std::collections::HashSet::new();
    for event in fixture.events.iter().filter(|e| e.variable == "outcome") {
        *counts.entry(event.episode_id.clone()).or_insert(0usize) += 1;
        assert!(ids.insert(event.id.clone()));
        if let Some((day, id)) = previous.get(&event.episode_id) {
            assert!((*day, id.as_str()) <= (event.event_day, event.id.as_str()));
        }
        previous.insert(
            event.episode_id.clone(),
            (event.event_day, event.id.clone()),
        );
    }
    assert!(counts.values().all(|count| [0, 1, 100].contains(count)));
    let report: serde_yaml::Value =
        serde_yaml::from_slice(&fs::read(dir.join("a-eval/evaluation.yaml")).unwrap()).unwrap();
    assert!(report["longitudinal"]["event_count_distributions"]["outcome"].is_mapping());
    let shifted: serde_yaml::Value =
        serde_yaml::from_slice(&fs::read(dir.join("timed-eval/evaluation.yaml")).unwrap()).unwrap();
    assert_ne!(
        report["longitudinal"]["variable_days"]["outcome"],
        shifted["longitudinal"]["variable_days"]["outcome"]
    );
    let manifest: serde_yaml::Value =
        serde_yaml::from_slice(&fs::read(dir.join("a-eval/manifest.yaml")).unwrap()).unwrap();
    assert_eq!(
        manifest["holdout_provenance"],
        "reference matched recorded holdout fixture fingerprint"
    );
    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn outcome_timing_workflow_replays() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let dir = std::env::temp_dir().join(format!("janus-timing-test-{}", std::process::id()));
    fs::create_dir_all(&dir).unwrap();
    let study = root.join("examples/study.yaml");
    for name in ["a", "b"] {
        let split = dir.join(format!("{name}-split"));
        let fit = dir.join(format!("{name}-fit"));
        let generated = dir.join(format!("{name}-generated"));
        let eval = dir.join(format!("{name}-eval"));
        assert!(cli(&[
            "split",
            study.to_str().unwrap(),
            root.join("examples/fixture-conditional.yaml")
                .to_str()
                .unwrap(),
            root.join("examples/split-for-fit.yaml").to_str().unwrap(),
            split.to_str().unwrap()
        ]));
        assert!(cli(&[
            "fit",
            study.to_str().unwrap(),
            split.join("development.yaml").to_str().unwrap(),
            root.join("examples/fit-timing.yaml").to_str().unwrap(),
            fit.to_str().unwrap()
        ]));
        assert!(cli(&[
            "generate",
            study.to_str().unwrap(),
            fit.join("generator.yaml").to_str().unwrap(),
            generated.to_str().unwrap()
        ]));
        assert!(cli(&[
            "evaluate",
            study.to_str().unwrap(),
            generated.join("fixture.yaml").to_str().unwrap(),
            split.join("holdout.yaml").to_str().unwrap(),
            root.join("examples/evaluation-timing.yaml")
                .to_str()
                .unwrap(),
            eval.to_str().unwrap(),
            "--split-manifest",
            split.join("manifest.yaml").to_str().unwrap()
        ]));
        let report: serde_yaml::Value =
            serde_yaml::from_slice(&fs::read(eval.join("evaluation.yaml")).unwrap()).unwrap();
        assert!(
            report["outcome_timing"]["synthetic"]["observed"]
                .as_u64()
                .unwrap()
                > 0
        );
    }
    for (stage, files) in [
        (
            "split",
            vec!["development.yaml", "holdout.yaml", "manifest.yaml"],
        ),
        ("fit", vec!["generator.yaml", "manifest.yaml"]),
        (
            "generated",
            vec!["fixture.yaml", "cohort.yaml", "manifest.yaml"],
        ),
        ("eval", vec!["evaluation.yaml", "manifest.yaml"]),
    ] {
        for file in files {
            assert_eq!(
                fs::read(dir.join(format!("a-{stage}")).join(file)).unwrap(),
                fs::read(dir.join(format!("b-{stage}")).join(file)).unwrap()
            );
        }
    }
    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn fitting_artifacts_repeat_and_generator_is_accepted() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let dir = std::env::temp_dir().join(format!("janus-fit-{}", std::process::id()));
    fs::create_dir_all(&dir).unwrap();
    let study = root.join("examples/study.yaml");
    let fixture = root.join("examples/fixture.yaml");
    let fit_config = root.join("examples/fit.yaml");
    let split = dir.join("split");
    let usable_split_config = dir.join("split-config.yaml");
    fs::write(
        &usable_split_config,
        "schema_version: '1'\nseed: 5\ndevelopment_proportion: 0.5\n",
    )
    .unwrap();
    assert!(cli(&[
        "split",
        study.to_str().unwrap(),
        fixture.to_str().unwrap(),
        usable_split_config.to_str().unwrap(),
        split.to_str().unwrap()
    ]));
    let first = dir.join("fit-a");
    let second = dir.join("fit-b");
    let run_fit = |output: &Path| {
        let development = split.join("development.yaml");
        cli(&[
            "fit",
            study.to_str().unwrap(),
            development.to_str().unwrap(),
            fit_config.to_str().unwrap(),
            output.to_str().unwrap(),
        ])
    };
    assert!(run_fit(&first));
    assert!(run_fit(&second));
    assert!(!run_fit(&first));
    for name in ["generator.yaml", "manifest.yaml"] {
        assert_eq!(
            fs::read(first.join(name)).unwrap(),
            fs::read(second.join(name)).unwrap()
        );
    }
    let synthetic = dir.join("synthetic");
    assert!(cli(&[
        "generate",
        study.to_str().unwrap(),
        first.join("generator.yaml").to_str().unwrap(),
        synthetic.to_str().unwrap()
    ]));
    assert!(cli(&[
        "validate",
        study.to_str().unwrap(),
        synthetic.join("fixture.yaml").to_str().unwrap()
    ]));
    let evaluation = root.join("examples/evaluation.yaml");
    let evaluation_out = dir.join("evaluation");
    assert!(cli(&[
        "evaluate",
        study.to_str().unwrap(),
        synthetic.join("fixture.yaml").to_str().unwrap(),
        split.join("holdout.yaml").to_str().unwrap(),
        evaluation.to_str().unwrap(),
        evaluation_out.to_str().unwrap(),
        "--split-manifest",
        split.join("manifest.yaml").to_str().unwrap()
    ]));
    let report: serde_yaml::Value =
        serde_yaml::from_slice(&fs::read(evaluation_out.join("evaluation.yaml")).unwrap()).unwrap();
    assert!(report["temporal"]["synthetic"]["variables"].is_mapping());
    let malformed = dir.join("malformed.yaml");
    fs::write(&malformed, "schema_version: [\n").unwrap();
    let bad = dir.join("bad-fit");
    assert!(!cli(&[
        "fit",
        study.to_str().unwrap(),
        split.join("development.yaml").to_str().unwrap(),
        malformed.to_str().unwrap(),
        bad.to_str().unwrap()
    ]));
    assert!(!bad.exists());
    assert!(!dir.join("bad-fit.janus-staging").exists());
    for (name, event_day, recording_day) in [
        ("early-recording", "1", "0"),
        ("overflow", "9223372036854775807", "9223372036854775807"),
    ] {
        let config = dir.join(format!("{name}.yaml"));
        fs::write(&config, format!("schema_version: '1'\nseed: 1\nepisode_count: 2\nevent_day: {event_day}\nrecording_day: {recording_day}\n")).unwrap();
        let output = dir.join(name);
        assert!(!cli(&[
            "fit",
            study.to_str().unwrap(),
            split.join("development.yaml").to_str().unwrap(),
            config.to_str().unwrap(),
            output.to_str().unwrap()
        ]));
        assert!(!output.exists());
        assert!(!dir.join(format!("{name}.janus-staging")).exists());
    }
    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn evaluation_artifacts_repeat_and_reject_invalid_inputs() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let study = root.join("examples/study.yaml");
    let synthetic = root.join("examples/fixture.yaml");
    let reference = root.join("examples/reference.yaml");
    let config = root.join("examples/evaluation.yaml");
    let dir = std::env::temp_dir().join(format!("janus-evaluation-{}", std::process::id()));
    fs::create_dir_all(&dir).unwrap();
    let first = dir.join("first");
    let second = dir.join("second");
    let run = |reference: &Path, config: &Path, output: &Path| {
        cli(&[
            "evaluate",
            study.to_str().unwrap(),
            synthetic.to_str().unwrap(),
            reference.to_str().unwrap(),
            config.to_str().unwrap(),
            output.to_str().unwrap(),
        ])
    };
    assert!(run(&reference, &config, &first));
    assert!(!run(&reference, &config, &first));
    assert!(run(&reference, &config, &second));
    let manifest: serde_yaml::Value =
        serde_yaml::from_slice(&fs::read(first.join("manifest.yaml")).unwrap()).unwrap();
    let names = ["evaluation.yaml", "manifest.yaml"];
    assert_eq!(
        manifest["artifacts"].as_sequence().unwrap().len(),
        names.len()
    );
    assert_eq!(fs::read_dir(&first).unwrap().count(), names.len());
    for name in names {
        assert!(manifest["artifacts"]
            .as_sequence()
            .unwrap()
            .iter()
            .any(|v| v.as_str() == Some(name)));
        assert_eq!(
            fs::read(first.join(name)).unwrap(),
            fs::read(second.join(name)).unwrap()
        );
    }
    let malformed = dir.join("malformed.yaml");
    fs::write(&malformed, "schema_version: [\n").unwrap();
    let invalid = dir.join("invalid.yaml");
    fs::write(
        &invalid,
        "schema_version: '1'\njoint: {exposure: exposure, outcome: unknown}\n",
    )
    .unwrap();
    let invalid_temporal = dir.join("invalid-temporal.yaml");
    fs::write(
        &invalid_temporal,
        "schema_version: '1'\ntemporal:\n  windows:\n    - {name: bad, start_day: 4, end_day: 3}\n",
    )
    .unwrap();
    for (r, c, name) in [
        (&malformed, &config, "bad-reference"),
        (&reference, &malformed, "bad-config"),
        (&reference, &invalid, "invalid-selection"),
        (&reference, &invalid_temporal, "invalid-temporal"),
    ] {
        let output = dir.join(name);
        assert!(!run(r, c, &output));
        assert!(!output.exists());
        assert!(!dir.join(format!("{name}.janus-staging")).exists());
    }
    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn artifacts_are_complete_repeatable_and_invalid_inputs_leave_no_output() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let study = root.join("examples/study.yaml");
    let fixture = root.join("examples/fixture.yaml");
    let dir = std::env::temp_dir().join(format!("janus-artifacts-{}", std::process::id()));
    fs::create_dir_all(&dir).unwrap();
    let first = dir.join("first");
    let second = dir.join("second");
    let study = study.to_str().unwrap();
    let fixture = fixture.to_str().unwrap();
    let first_str = first.to_str().unwrap();
    assert!(cli(&["validate", study, fixture]));
    assert!(cli(&["run", study, fixture, first_str]));
    assert!(!cli(&["run", study, fixture, first_str]));
    assert!(cli(&["run", study, fixture, second.to_str().unwrap()]));
    let manifest: serde_yaml::Value =
        serde_yaml::from_slice(&fs::read(first.join("manifest.yaml")).unwrap()).unwrap();
    let names = [
        "cohort.yaml",
        "exclusions.yaml",
        "summary.yaml",
        "integrity.yaml",
        "diagnostics.yaml",
        "manifest.yaml",
    ];
    let listed = manifest["artifacts"].as_sequence().unwrap();
    assert_eq!(listed.len(), names.len());
    for name in names {
        assert!(listed.iter().any(|v| v.as_str() == Some(name)));
        assert_eq!(
            fs::read(first.join(name)).unwrap(),
            fs::read(second.join(name)).unwrap()
        );
    }
    assert_eq!(fs::read_dir(&first).unwrap().count(), names.len());
    let malformed = dir.join("malformed.yaml");
    fs::write(&malformed, "schema_version: '['\n").unwrap();
    for (s, f, name) in [
        (malformed.to_str().unwrap(), fixture, "bad-study"),
        (study, malformed.to_str().unwrap(), "bad-fixture"),
    ] {
        let output = dir.join(name);
        assert!(!cli(&["run", s, f, output.to_str().unwrap()]));
        assert!(!output.exists());
    }
    let invalid_value = dir.join("invalid-value.yaml");
    let original = fs::read_to_string(fixture).unwrap();
    fs::write(
        &invalid_value,
        original.replacen("value: \"true\"", "value: \"invalid\"", 1),
    )
    .unwrap();
    let output = dir.join("invalid-value-output");
    assert!(!cli(&["validate", study, invalid_value.to_str().unwrap()]));
    assert!(!cli(&[
        "run",
        study,
        invalid_value.to_str().unwrap(),
        output.to_str().unwrap()
    ]));
    assert!(!output.exists());
    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn generated_artifacts_are_repeatable_and_invalid_configs_leave_no_output() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let example_study = root.join("examples/study.yaml");
    let generator = root.join("examples/generator.yaml");
    let dir = std::env::temp_dir().join(format!("janus-generated-{}", std::process::id()));
    fs::create_dir_all(&dir).unwrap();
    let study = dir.join("input-study.yaml");
    fs::copy(example_study, &study).unwrap();
    let study = study.to_str().unwrap();
    let generator = generator.to_str().unwrap();
    let first = dir.join("first");
    let second = dir.join("second");
    for output in [&first, &second] {
        assert!(cli(&[
            "generate",
            study,
            generator,
            output.to_str().unwrap()
        ]));
    }
    assert!(!cli(&[
        "generate",
        study,
        generator,
        first.to_str().unwrap()
    ]));
    assert!(cli(&[
        "validate",
        study,
        first.join("fixture.yaml").to_str().unwrap()
    ]));
    let manifest: serde_yaml::Value =
        serde_yaml::from_slice(&fs::read(first.join("manifest.yaml")).unwrap()).unwrap();
    let names = [
        "study.yaml",
        "fixture.yaml",
        "cohort.yaml",
        "exclusions.yaml",
        "summary.yaml",
        "integrity.yaml",
        "diagnostics.yaml",
        "manifest.yaml",
    ];
    let listed = manifest["artifacts"].as_sequence().unwrap();
    assert_eq!(listed.len(), names.len());
    assert_eq!(fs::read_dir(&first).unwrap().count(), names.len());
    for name in names {
        assert!(listed.iter().any(|v| v.as_str() == Some(name)));
        assert_eq!(
            fs::read(first.join(name)).unwrap(),
            fs::read(second.join(name)).unwrap()
        );
    }
    assert_eq!(
        manifest["generator"]["configuration"]["seed"].as_u64(),
        Some(42)
    );
    assert!(manifest["generator"]["rng"]
        .as_str()
        .unwrap()
        .contains("SplitMix64"));
    assert!(!manifest["generator"]["software_version"]
        .as_str()
        .unwrap()
        .is_empty());
    let saved_study: serde_yaml::Value =
        serde_yaml::from_slice(&fs::read(first.join("study.yaml")).unwrap()).unwrap();
    let input_study: serde_yaml::Value = serde_yaml::from_slice(&fs::read(study).unwrap()).unwrap();
    assert_eq!(saved_study, input_study);
    fs::write(study, "changed after generation").unwrap();
    let replay_config = dir.join("replay-generator.yaml");
    fs::write(
        &replay_config,
        serde_yaml::to_string(&manifest["generator"]["configuration"]).unwrap(),
    )
    .unwrap();
    let replay = dir.join("replay");
    assert!(cli(&[
        "generate",
        first.join("study.yaml").to_str().unwrap(),
        replay_config.to_str().unwrap(),
        replay.to_str().unwrap(),
    ]));
    for name in names {
        assert_eq!(
            fs::read(first.join(name)).unwrap(),
            fs::read(replay.join(name)).unwrap()
        );
    }
    fs::write(study, serde_yaml::to_string(&input_study).unwrap()).unwrap();
    let original = fs::read_to_string(generator).unwrap();
    let variants = [
        original.replace("seed: 42", "seed: 43"),
        original.replace("probability: 0.5", "probability: -0.5"),
        original.replace("probability: 0.5", "probability: 0.4"),
        original.replace("value: \"true\"", "value: \"invalid\""),
        original.replace("recording_day: 0", "recording_day: -1"),
        original.replace("name: exposure", "name: unknown"),
        format!(
            "schema_version: \"1\"\nseed: 42\nepisode_count: 1\nvariables:\n{}",
            original
                .split("  - name: outcome")
                .nth(1)
                .map(|tail| format!("  - name: outcome{tail}"))
                .unwrap()
        ),
        original.replace("episode_count: 100", "episode_count: 0"),
        original.replace("schema_version: \"1\"", "schema_version: \"3\""),
        "schema_version: [".into(),
    ];
    for (i, contents) in variants.iter().enumerate() {
        let config = dir.join(format!("config-{i}.yaml"));
        let output = dir.join(format!("output-{i}"));
        fs::write(&config, contents).unwrap();
        let success = cli(&[
            "generate",
            study,
            config.to_str().unwrap(),
            output.to_str().unwrap(),
        ]);
        if i == 0 {
            assert!(success);
            assert_ne!(
                fs::read(first.join("fixture.yaml")).unwrap(),
                fs::read(output.join("fixture.yaml")).unwrap()
            );
        } else {
            assert!(!success, "variant {i}");
            assert!(!output.exists());
            assert!(!dir.join(format!("output-{i}.janus-staging")).exists());
        }
    }
    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn conditional_generation_replays_and_cleans_up_failures() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let study = root.join("examples/study.yaml");
    let config = root.join("examples/generator-conditional.yaml");
    let dir = std::env::temp_dir().join(format!("janus-conditional-{}", std::process::id()));
    fs::create_dir_all(&dir).unwrap();
    let first = dir.join("first");
    let second = dir.join("second");
    for output in [&first, &second] {
        assert!(cli(&[
            "generate",
            study.to_str().unwrap(),
            config.to_str().unwrap(),
            output.to_str().unwrap()
        ]));
    }
    assert!(!cli(&[
        "generate",
        study.to_str().unwrap(),
        config.to_str().unwrap(),
        first.to_str().unwrap()
    ]));
    assert!(cli(&[
        "validate",
        first.join("study.yaml").to_str().unwrap(),
        first.join("fixture.yaml").to_str().unwrap()
    ]));
    let manifest: serde_yaml::Value =
        serde_yaml::from_slice(&fs::read(first.join("manifest.yaml")).unwrap()).unwrap();
    let input: serde_yaml::Value = serde_yaml::from_slice(&fs::read(&config).unwrap()).unwrap();
    assert_eq!(manifest["generator"]["configuration"], input);
    let replay_config = dir.join("replay.yaml");
    fs::write(
        &replay_config,
        serde_yaml::to_string(&manifest["generator"]["configuration"]).unwrap(),
    )
    .unwrap();
    let replay = dir.join("replay");
    assert!(cli(&[
        "generate",
        first.join("study.yaml").to_str().unwrap(),
        replay_config.to_str().unwrap(),
        replay.to_str().unwrap()
    ]));
    let names = [
        "study.yaml",
        "fixture.yaml",
        "cohort.yaml",
        "exclusions.yaml",
        "summary.yaml",
        "integrity.yaml",
        "diagnostics.yaml",
        "manifest.yaml",
    ];
    let listed = manifest["artifacts"].as_sequence().unwrap();
    assert_eq!(listed.len(), names.len());
    assert_eq!(fs::read_dir(&first).unwrap().count(), names.len());
    for name in names {
        assert!(listed.iter().any(|v| v.as_str() == Some(name)));
        let contents = fs::read(first.join(name)).unwrap();
        assert_eq!(contents, fs::read(second.join(name)).unwrap());
        assert_eq!(contents, fs::read(replay.join(name)).unwrap());
    }
    let changed = dir.join("changed.yaml");
    let text = fs::read_to_string(&config).unwrap();
    fs::write(&changed, text.replace("seed: 42", "seed: 43")).unwrap();
    let changed_output = dir.join("changed");
    assert!(cli(&[
        "generate",
        study.to_str().unwrap(),
        changed.to_str().unwrap(),
        changed_output.to_str().unwrap()
    ]));
    assert_ne!(
        fs::read(first.join("fixture.yaml")).unwrap(),
        fs::read(changed_output.join("fixture.yaml")).unwrap()
    );
    fs::write(
        &changed,
        text.replace("when: {exposure: \"false\"}", "when: {exposure: \"true\"}"),
    )
    .unwrap();
    let failed = dir.join("failed");
    assert!(!cli(&[
        "generate",
        study.to_str().unwrap(),
        changed.to_str().unwrap(),
        failed.to_str().unwrap()
    ]));
    assert!(!failed.exists());
    assert!(!dir.join("failed.janus-staging").exists());
    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn patient_split_and_holdout_provenance_workflow() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let study = root.join("examples/study.yaml");
    let fixture = root.join("examples/fixture.yaml");
    let config = root.join("examples/split.yaml");
    let evaluation = root.join("examples/evaluation.yaml");
    let synthetic = root.join("examples/reference.yaml");
    let dir = std::env::temp_dir().join(format!("janus-split-test-{}", std::process::id()));
    fs::create_dir_all(&dir).unwrap();
    let split = dir.join("split");
    let replay = dir.join("replay");
    for out in [&split, &replay] {
        assert!(cli(&[
            "split",
            study.to_str().unwrap(),
            fixture.to_str().unwrap(),
            config.to_str().unwrap(),
            out.to_str().unwrap()
        ]));
    }
    let dev: Fixture =
        serde_yaml::from_slice(&fs::read(split.join("development.yaml")).unwrap()).unwrap();
    let hold: Fixture =
        serde_yaml::from_slice(&fs::read(split.join("holdout.yaml")).unwrap()).unwrap();
    assert!(dev
        .events
        .iter()
        .all(|e| !hold.events.iter().any(|h| h.patient_id == e.patient_id)));
    assert_eq!(
        fs::read(split.join("holdout.yaml")).unwrap(),
        fs::read(replay.join("holdout.yaml")).unwrap()
    );
    let split_manifest: serde_yaml::Value =
        serde_yaml::from_slice(&fs::read(split.join("manifest.yaml")).unwrap()).unwrap();
    assert_eq!(
        split_manifest["development_patients"].as_u64().unwrap()
            + split_manifest["holdout_patients"].as_u64().unwrap(),
        2
    );
    assert_eq!(
        split_manifest["development_episodes"].as_u64().unwrap()
            + split_manifest["holdout_episodes"].as_u64().unwrap(),
        3
    );
    for name in ["development.yaml", "holdout.yaml", "manifest.yaml"] {
        assert_eq!(
            fs::read(split.join(name)).unwrap(),
            fs::read(replay.join(name)).unwrap()
        );
    }
    let manifest = split.join("manifest.yaml");
    let eval_out = dir.join("evaluation");
    assert!(cli(&[
        "evaluate",
        study.to_str().unwrap(),
        synthetic.to_str().unwrap(),
        split.join("holdout.yaml").to_str().unwrap(),
        evaluation.to_str().unwrap(),
        eval_out.to_str().unwrap(),
        "--split-manifest",
        manifest.to_str().unwrap()
    ]));
    let tampered = dir.join("tampered.yaml");
    let mut modified = hold.clone();
    if let Some(event) = modified.events.first_mut() {
        event.provenance.push_str("_modified");
    } else {
        modified.schema_version = "1".into();
    }
    fs::write(&tampered, serde_yaml::to_string(&modified).unwrap()).unwrap();
    assert!(!cli(&[
        "evaluate",
        study.to_str().unwrap(),
        synthetic.to_str().unwrap(),
        tampered.to_str().unwrap(),
        evaluation.to_str().unwrap(),
        dir.join("bad-eval").to_str().unwrap(),
        "--split-manifest",
        manifest.to_str().unwrap()
    ]));
    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn csv_parquet_round_trip_and_existing_workflows() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let dir =
        std::env::temp_dir().join(format!("janus-tabular-integration-{}", std::process::id()));
    fs::create_dir_all(&dir).unwrap();
    let study = root.join("examples/study.yaml");
    let mapping = root.join("examples/mapping.yaml");
    let source = root.join("examples/source.csv");
    assert!(cli(&[
        "validate-csv",
        source.to_str().unwrap(),
        mapping.to_str().unwrap()
    ]));
    for name in ["a", "b"] {
        assert!(cli(&[
            "import-csv",
            source.to_str().unwrap(),
            mapping.to_str().unwrap(),
            dir.join(name).to_str().unwrap()
        ]));
    }
    for name in [
        "data.parquet",
        "schema.yaml",
        "validation.yaml",
        "manifest.yaml",
    ] {
        assert_eq!(
            fs::read(dir.join("a").join(name)).unwrap(),
            fs::read(dir.join("b").join(name)).unwrap()
        );
    }
    let parquet = dir.join("a/data.parquet");
    assert!(cli(&[
        "inspect",
        parquet.to_str().unwrap(),
        dir.join("inspection").to_str().unwrap()
    ]));
    assert!(cli(&[
        "inspect",
        dir.join("b/data.parquet").to_str().unwrap(),
        dir.join("inspection-b").to_str().unwrap()
    ]));
    for name in ["inspection.yaml", "manifest.yaml"] {
        assert_eq!(
            fs::read(dir.join("inspection").join(name)).unwrap(),
            fs::read(dir.join("inspection-b").join(name)).unwrap()
        );
    }
    assert!(cli(&[
        "run",
        study.to_str().unwrap(),
        parquet.to_str().unwrap(),
        dir.join("run").to_str().unwrap()
    ]));
    assert!(cli(&[
        "split",
        study.to_str().unwrap(),
        parquet.to_str().unwrap(),
        root.join("examples/split-for-fit.yaml").to_str().unwrap(),
        dir.join("split").to_str().unwrap()
    ]));
    assert!(cli(&[
        "fit",
        study.to_str().unwrap(),
        parquet.to_str().unwrap(),
        root.join("examples/fit.yaml").to_str().unwrap(),
        dir.join("fit").to_str().unwrap()
    ]));
    assert!(cli(&[
        "evaluate",
        study.to_str().unwrap(),
        parquet.to_str().unwrap(),
        parquet.to_str().unwrap(),
        root.join("examples/evaluation.yaml").to_str().unwrap(),
        dir.join("evaluation").to_str().unwrap()
    ]));
    let exported = dir.join("export.csv");
    assert!(cli(&[
        "export-csv",
        parquet.to_str().unwrap(),
        exported.to_str().unwrap()
    ]));
    assert!(cli(&[
        "import-csv",
        exported.to_str().unwrap(),
        mapping.to_str().unwrap(),
        dir.join("replayed").to_str().unwrap()
    ]));
    assert_eq!(
        fs::read(&parquet).unwrap(),
        fs::read(dir.join("replayed/data.parquet")).unwrap()
    );
    let report = fs::read_to_string(dir.join("inspection/inspection.yaml")).unwrap();
    assert!(!report.contains("P001"));
    assert!(!cli(&[
        "import-csv",
        source.to_str().unwrap(),
        mapping.to_str().unwrap(),
        dir.join("a").to_str().unwrap()
    ]));
    let invalid = dir.join("invalid.csv");
    fs::write(&invalid, "patient_id\nfoo\n").unwrap();
    let rejected = dir.join("rejected");
    assert!(!cli(&[
        "import-csv",
        invalid.to_str().unwrap(),
        mapping.to_str().unwrap(),
        rejected.to_str().unwrap()
    ]));
    assert!(!rejected.exists());
    let mut altered_schema: serde_yaml::Value =
        serde_yaml::from_slice(&fs::read(dir.join("a/schema.yaml")).unwrap()).unwrap();
    altered_schema["study_fingerprint"] = "wrong".into();
    fs::write(
        dir.join("a/schema.yaml"),
        serde_yaml::to_string(&altered_schema).unwrap(),
    )
    .unwrap();
    assert!(!cli(&[
        "validate",
        study.to_str().unwrap(),
        parquet.to_str().unwrap()
    ]));
    for field in ["mapping_fingerprint", "schema_version"] {
        let mut altered: serde_yaml::Value =
            serde_yaml::from_slice(&fs::read(dir.join("b/schema.yaml")).unwrap()).unwrap();
        altered[field] = "wrong".into();
        fs::write(
            dir.join("a/schema.yaml"),
            serde_yaml::to_string(&altered).unwrap(),
        )
        .unwrap();
        assert!(!cli(&[
            "validate",
            study.to_str().unwrap(),
            parquet.to_str().unwrap()
        ]));
    }
    fs::copy(dir.join("b/schema.yaml"), dir.join("a/schema.yaml")).unwrap();
    let mut bytes = fs::read(&parquet).unwrap();
    bytes[10] ^= 1;
    fs::write(&parquet, bytes).unwrap();
    assert!(!cli(&[
        "validate",
        study.to_str().unwrap(),
        parquet.to_str().unwrap()
    ]));
    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn flat_descriptive_requires_source_config_and_output_under_datasets() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let data_root = root.join("datasets");
    let source = data_root.join("flat-cli-test.csv");
    let config = data_root.join("flat-cli-test.yaml");
    let output = data_root.join("results/flat-cli-test-output");
    fs::create_dir_all(output.parent().unwrap()).unwrap();
    if output.exists() {
        fs::remove_dir_all(&output).unwrap();
    }
    fs::write(&source, "no,arm,x\na,LD,1\nb,LD,2\nc,NoLD,3\n").unwrap();
    fs::write(&config, "schema_version: '1'\nallocation_column: arm\nallocation_values: [LD, NoLD]\nsplit_seed: 3\ndevelopment_proportion: 0.67\nbootstrap_replicates: 100\nbootstrap_confidence_level: 0.95\nthresholds: {max_missingness_rate_difference: 1.0, max_categorical_tvd: 1.0, max_numeric_quantile_difference: 1000.0}\nvariables:\n  - {column: x, kind: numeric}\n").unwrap();
    assert!(cli(&[
        "describe-flat",
        source.to_str().unwrap(),
        config.to_str().unwrap(),
        output.to_str().unwrap()
    ]));
    assert!(!cli(&[
        "describe-flat",
        source.to_str().unwrap(),
        root.join("examples/mapping.yaml").to_str().unwrap(),
        data_root
            .join("results/flat-cli-test-rejected")
            .to_str()
            .unwrap()
    ]));
    fs::remove_dir_all(output).unwrap();
    fs::remove_file(source).unwrap();
    fs::remove_file(config).unwrap();
}

#[test]
fn audit_reproducibility_cli_verifies_lockfile_platform_splits_and_modes() {
    let dir = std::env::temp_dir().join(format!(
        "janus-audit-reproducibility-test-{}",
        std::process::id()
    ));
    fs::create_dir_all(&dir).unwrap();

    let lockfile_hash = janus::coherence::cargo_lock_sha256();
    let holdout_hash =
        janus::coherence::sha256_fingerprint(b"fictional-holdout-fixture-allocation-v1");

    let valid_manifest_path = dir.join("run_manifest.json");
    let splits_path = dir.join("split_manifest.json");
    let mismatched_lock_path = dir.join("mismatched_lock_manifest.json");
    let platform_diff_path = dir.join("platform_diff_manifest.json");

    let valid_manifest = format!(
        r#"{{
  "schema_version": "1",
  "lockfile_hash": "{lockfile_hash}",
  "platform": {{
    "os": "{}",
    "arch": "{}"
  }},
  "fingerprints": {{
    "holdout_fingerprint": "{holdout_hash}"
  }},
  "software": {{
    "package_version": "{}",
    "dependencies": {{
      "serde": "1",
      "sha2": "0.10"
    }}
  }}
}}"#,
        std::env::consts::OS,
        std::env::consts::ARCH,
        env!("CARGO_PKG_VERSION")
    );
    fs::write(&valid_manifest_path, valid_manifest).unwrap();

    let splits_manifest = format!(
        r#"{{
  "schema_version": "1",
  "holdout_fingerprint": "{holdout_hash}"
}}"#
    );
    fs::write(&splits_path, splits_manifest).unwrap();

    // 1. Valid fictional manifest + splits in strict mode -> exit 0, reproducible: true
    let out_valid = Command::new(env!("CARGO_BIN_EXE_janus-cli"))
        .args([
            "audit-reproducibility",
            "--manifest",
            valid_manifest_path.to_str().unwrap(),
            "--splits",
            splits_path.to_str().unwrap(),
            "--strict",
        ])
        .output()
        .unwrap();
    assert!(out_valid.status.success());
    let report_valid: janus::coherence::ReproducibilityAuditReport =
        serde_yaml::from_slice(&out_valid.stdout).unwrap();
    assert!(report_valid.reproducible);
    assert!(report_valid.warnings.is_empty());
    assert_eq!(report_valid.checks.len(), 4);
    assert!(report_valid.checks.iter().all(|c| c.passed));

    // 2. Mismatched lockfile hash in strict mode -> exit 1 with specific mismatch details
    let mismatched_lock = format!(
        r#"{{
  "schema_version": "1",
  "lockfile_hash": "sha256:deadbeefdeadbeefdeadbeefdeadbeefdeadbeefdeadbeefdeadbeefdeadbeef",
  "platform": {{
    "os": "{}",
    "arch": "{}"
  }}
}}"#,
        std::env::consts::OS,
        std::env::consts::ARCH
    );
    fs::write(&mismatched_lock_path, mismatched_lock).unwrap();

    let out_bad_lock = Command::new(env!("CARGO_BIN_EXE_janus-cli"))
        .args([
            "audit-reproducibility",
            "--manifest",
            mismatched_lock_path.to_str().unwrap(),
            "--strict",
        ])
        .output()
        .unwrap();
    assert_eq!(out_bad_lock.status.code(), Some(1));
    let report_bad_lock: janus::coherence::ReproducibilityAuditReport =
        serde_yaml::from_slice(&out_bad_lock.stdout).unwrap();
    assert!(!report_bad_lock.reproducible);
    assert!(report_bad_lock
        .warnings
        .iter()
        .any(|w| w.contains("lockfile_fingerprint mismatch")));
    let stderr_bad_lock = String::from_utf8_lossy(&out_bad_lock.stderr);
    assert!(stderr_bad_lock.contains("lockfile_fingerprint mismatch"));

    // 3. Warn-only mode with platform differences -> exit 0, warnings populated
    let platform_diff = format!(
        r#"{{
  "schema_version": "1",
  "lockfile_hash": "{lockfile_hash}",
  "platform": {{
    "os": "fictional-os",
    "arch": "fictional-arch"
  }}
}}"#
    );
    fs::write(&platform_diff_path, platform_diff).unwrap();

    let out_warn = Command::new(env!("CARGO_BIN_EXE_janus-cli"))
        .args([
            "audit-reproducibility",
            "--manifest",
            platform_diff_path.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(out_warn.status.success());
    let report_warn: janus::coherence::ReproducibilityAuditReport =
        serde_yaml::from_slice(&out_warn.stdout).unwrap();
    assert!(report_warn.reproducible);
    assert_eq!(report_warn.warnings.len(), 1);
    assert!(report_warn.warnings[0].contains("platform_metadata mismatch"));

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn audit_reproducibility_accepts_real_janus_manifests() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let dir = std::env::temp_dir().join(format!(
        "janus-audit-real-manifests-test-{}",
        std::process::id()
    ));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();

    let study = root.join("examples/study.yaml");
    let fixture = root.join("examples/fixture.yaml");
    let split_cfg = root.join("examples/split-for-fit.yaml");
    let gen_cfg = root.join("examples/generator.yaml");
    let eval_cfg = root.join("examples/evaluation.yaml");

    let split_dir = dir.join("split");
    let run_dir = dir.join("run");
    let gen_dir = dir.join("generated");
    let eval_dir = dir.join("eval");

    assert!(cli(&[
        "split",
        study.to_str().unwrap(),
        fixture.to_str().unwrap(),
        split_cfg.to_str().unwrap(),
        split_dir.to_str().unwrap(),
    ]));
    assert!(cli(&[
        "run",
        study.to_str().unwrap(),
        split_dir.join("development.yaml").to_str().unwrap(),
        run_dir.to_str().unwrap(),
    ]));
    assert!(cli(&[
        "generate",
        study.to_str().unwrap(),
        gen_cfg.to_str().unwrap(),
        gen_dir.to_str().unwrap(),
    ]));
    assert!(cli(&[
        "evaluate",
        study.to_str().unwrap(),
        gen_dir.join("fixture.yaml").to_str().unwrap(),
        split_dir.join("holdout.yaml").to_str().unwrap(),
        eval_cfg.to_str().unwrap(),
        eval_dir.to_str().unwrap(),
        "--split-manifest",
        split_dir.join("manifest.yaml").to_str().unwrap(),
    ]));

    let split_manifest = split_dir.join("manifest.yaml");
    for (manifest_path, splits_opt, expected_checks) in [
        (split_manifest.as_path(), None, 3usize),
        (run_dir.join("manifest.yaml").as_path(), None, 3usize),
        (gen_dir.join("manifest.yaml").as_path(), None, 3usize),
        (
            eval_dir.join("manifest.yaml").as_path(),
            Some(split_manifest.as_path()),
            4usize,
        ),
    ] {
        let mut args = vec![
            "audit-reproducibility",
            "--manifest",
            manifest_path.to_str().unwrap(),
            "--strict",
        ];
        if let Some(splits) = splits_opt {
            args.push("--splits");
            args.push(splits.to_str().unwrap());
        }
        let output = Command::new(env!("CARGO_BIN_EXE_janus-cli"))
            .args(&args)
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "audit-reproducibility --strict failed for {}:\nstdout: {}\nstderr: {}",
            manifest_path.display(),
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr),
        );
        let report: janus::ReproducibilityAuditReport =
            serde_yaml::from_slice(&output.stdout).unwrap();
        assert!(report.reproducible);
        assert!(
            report.warnings.is_empty(),
            "unexpected warnings for {}: {:?}",
            manifest_path.display(),
            report.warnings
        );
        assert_eq!(report.checks.len(), expected_checks);
        assert!(report.checks.iter().all(|c| c.passed));
        assert!(report.checks[0].actual.starts_with("fnv1a64-yaml:"));
    }

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn forensic_eval_suite_cli_handles_mixed_cohorts_and_small_cell_suppression() {
    let dir = std::env::temp_dir().join(format!(
        "janus-forensic-suite-cli-test-{}",
        std::process::id()
    ));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();

    let train_csv = dir.join("train.csv");
    let real_csv = dir.join("real.csv");
    let synth_csv = dir.join("synth.csv");
    let suite_out = dir.join("suite_manifest.json");
    let legacy_out = dir.join("legacy_report.json");

    let mut train_body = String::from("age,biomarker,severity,status\n");
    let mut real_body = String::from("age,biomarker,severity,status\n");
    let mut synth_body = String::from("age,biomarker,severity,status\n");

    for i in 0..24 {
        let age_r = 45 + (i % 10);
        let bio_r = if i % 6 == 0 {
            "NA".to_string()
        } else {
            format!("{:.2}", 1.2 + (i % 5) as f64 * 0.3)
        };
        let sev_r = if i % 4 == 0 { "severe" } else { "mild" };
        let st_r = if i % 5 == 0 {
            ""
        } else if i % 3 == 0 {
            "dead"
        } else {
            "alive"
        };
        train_body.push_str(&format!("{age_r},{bio_r},{sev_r},{st_r}\n"));
        real_body.push_str(&format!("{age_r},{bio_r},{sev_r},{st_r}\n"));

        // Deliberately shifted numeric and categorical distributions in synthetic cohort.
        let age_s = 72 + (i % 10);
        let bio_s = if i % 8 == 0 {
            "0.0".to_string()
        } else {
            format!("{:.2}", 4.5 + (i % 5) as f64 * 0.4)
        };
        let sev_s = if i % 4 == 0 { "mild" } else { "severe" };
        let st_s = if i % 2 == 0 { "dead" } else { "alive" };
        synth_body.push_str(&format!("{age_s},{bio_s},{sev_s},{st_s}\n"));
    }

    fs::write(&train_csv, train_body).unwrap();
    fs::write(&real_csv, real_body).unwrap();
    fs::write(&synth_csv, synth_body).unwrap();

    // 1. Multi-seed forensic suite on >= 10 row mixed numeric + categorical cohorts
    let suite_cmd = Command::new(env!("CARGO_BIN_EXE_janus-cli"))
        .args([
            "forensic-eval",
            "--real",
            real_csv.to_str().unwrap(),
            "--synthetic",
            synth_csv.to_str().unwrap(),
            "--training",
            train_csv.to_str().unwrap(),
            "--suite",
            "--seeds",
            "11,22",
            "--output",
            suite_out.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(
        suite_cmd.status.success(),
        "forensic-eval --suite failed:\nstdout: {}\nstderr: {}",
        String::from_utf8_lossy(&suite_cmd.stdout),
        String::from_utf8_lossy(&suite_cmd.stderr)
    );

    let suite_json = fs::read_to_string(&suite_out).unwrap();
    assert!(!suite_json.contains("reference-row-"));
    assert!(!suite_json.contains("synthetic-row-"));
    let manifest: janus::ForensicManifest = serde_json::from_str(&suite_json).unwrap();
    manifest.validate().unwrap();
    assert!(!manifest.aggregate_metrics.suppressed);
    assert_eq!(manifest.seeds, vec![11, 22]);
    assert_eq!(manifest.aggregate_metrics.training_rows, 24);
    assert_eq!(manifest.aggregate_metrics.reference_rows, 24);
    assert_eq!(manifest.aggregate_metrics.generated_rows, 24);
    assert_eq!(manifest.aggregate_metrics.per_seed_metrics.len(), 2);
    assert_eq!(manifest.aggregate_metrics.chi_squared_tests.len(), 2);
    assert_eq!(manifest.aggregate_metrics.ks_tests.len(), 2);
    assert_eq!(manifest.outcome, janus::ForensicOutcome::Discriminable);

    // 2. Small-cell suppression (< 10 rows) without --training (clones reference as training)
    let small_real = dir.join("small_real.csv");
    let small_synth = dir.join("small_synth.csv");
    let suppressed_out = dir.join("suppressed.json");
    let mut small_r = String::from("age,severity\n");
    let mut small_s = String::from("age,severity\n");
    for i in 0..8 {
        small_r.push_str(&format!("{},{}\n", 40 + i, if i % 2 == 0 { "mild" } else { "severe" }));
        small_s.push_str(&format!("{},{}\n", 70 + i, "severe"));
    }
    fs::write(&small_real, small_r).unwrap();
    fs::write(&small_synth, small_s).unwrap();

    let supp_cmd = Command::new(env!("CARGO_BIN_EXE_janus-cli"))
        .args([
            "forensic-eval",
            "--real",
            small_real.to_str().unwrap(),
            "--synthetic",
            small_synth.to_str().unwrap(),
            "--suite",
            "--seeds",
            "11,22",
            "--output",
            suppressed_out.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(
        supp_cmd.status.success(),
        "forensic-eval --suite small-cell failed:\nstderr: {}",
        String::from_utf8_lossy(&supp_cmd.stderr)
    );
    let supp_manifest: janus::ForensicManifest =
        serde_json::from_str(&fs::read_to_string(&suppressed_out).unwrap()).unwrap();
    supp_manifest.validate().unwrap();
    assert!(supp_manifest.aggregate_metrics.suppressed);
    assert_eq!(supp_manifest.outcome, janus::ForensicOutcome::Insufficient);
    assert!(supp_manifest.aggregate_metrics.chi_squared_tests.is_empty());
    assert!(supp_manifest.aggregate_metrics.ks_tests.is_empty());
    assert!(supp_manifest.aggregate_metrics.per_seed_metrics.is_empty());

    // 3. Legacy single-seed mode on mixed cohort includes both KS and chi-squared tests
    let legacy_cmd = Command::new(env!("CARGO_BIN_EXE_janus-cli"))
        .args([
            "forensic-eval",
            "--real",
            real_csv.to_str().unwrap(),
            "--synthetic",
            synth_csv.to_str().unwrap(),
            "--seed",
            "42",
            "--output",
            legacy_out.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(
        legacy_cmd.status.success(),
        "legacy forensic-eval on mixed cohort failed:\nstderr: {}",
        String::from_utf8_lossy(&legacy_cmd.stderr)
    );
    let legacy_report: janus::forensic::ForensicReport =
        serde_json::from_str(&fs::read_to_string(&legacy_out).unwrap()).unwrap();
    assert_eq!(legacy_report.distributional_tests.len(), 4);
    assert_eq!(
        legacy_report
            .distributional_tests
            .iter()
            .filter(|t| t.test_type == "kolmogorov_smirnov")
            .count(),
        2
    );
    assert_eq!(
        legacy_report
            .distributional_tests
            .iter()
            .filter(|t| t.test_type == "chi_squared")
            .count(),
        2
    );

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn validate_contract_cli_runs_neurosurgical_validators_and_kaplan_meier() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let dir = std::env::temp_dir().join(format!(
        "janus-validate-contract-phase3-test-{}",
        std::process::id()
    ));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();

    let contract = root.join("examples/coherence-contract.yaml");
    let manifest = root.join("examples/protocol-manifest.yaml");
    let comparison = root.join("examples/candidate-comparison.yaml");

    // 1. Preserve exact stdout output when no Phase 3 flags are supplied
    let plain = Command::new(env!("CARGO_BIN_EXE_janus-cli"))
        .args([
            "validate-contract",
            contract.to_str().unwrap(),
            "--manifest",
            manifest.to_str().unwrap(),
            "--comparison",
            comparison.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(plain.status.success());
    assert_eq!(
        String::from_utf8_lossy(&plain.stdout).trim(),
        "valid: coherence contract and supplied protocol manifests"
    );

    // 2. CSV neurosurgical cohort + Kaplan-Meier survival comparison
    let neuro_csv = dir.join("neuro.csv");
    fs::write(
        &neuro_csv,
        concat!(
            "record_id,admission_time_hours,surgery_time_hours,discharge_time_hours,followup_90d_time_hours,",
            "initial_wfns,subsequent_wfns,initial_hunt_hess,subsequent_hunt_hess,documented_acute_intervention,",
            "discharge_mrs,followup_90d_mrs,documented_recovery_intervention\n",
            "p1,0,12,120,2160,2,2,2,3,false,2,2,false\n",
            "p2,0,NA,,2160,,NA,1,1,false,NA,,false\n",
            "p3,0,80,150,2160,2,2,2,2,false,6,3,false\n",
        ),
    )
    .unwrap();

    let surv_real_csv = dir.join("surv_real.csv");
    let surv_synth_csv = dir.join("surv_synth.csv");
    fs::write(
        &surv_real_csv,
        concat!(
            "record_id,time,event_occurred\n",
            "r1,10,1\n",
            "r2,20,0\n",
            "r3,30,1\n",
            "r4,40,0\n",
            "r5,NA,1\n",
        ),
    )
    .unwrap();
    fs::write(
        &surv_synth_csv,
        concat!(
            "record_id,time,event_occurred\n",
            "s1,5,1\n",
            "s2,8,1\n",
            "s3,12,1\n",
            "s4,15,0\n",
            "s5,25,\n",
        ),
    )
    .unwrap();

    let report_path = dir.join("contract_report.json");
    let phase3_out = Command::new(env!("CARGO_BIN_EXE_janus-cli"))
        .args([
            "validate-contract",
            contract.to_str().unwrap(),
            "--neurosurgical-cohort",
            neuro_csv.to_str().unwrap(),
            "--survival-real",
            surv_real_csv.to_str().unwrap(),
            "--survival-synthetic",
            surv_synth_csv.to_str().unwrap(),
            "--output",
            report_path.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(
        phase3_out.status.success(),
        "validate-contract Phase 3 failed:\nstderr: {}",
        String::from_utf8_lossy(&phase3_out.stderr)
    );

    let report_json: serde_json::Value =
        serde_json::from_str(&fs::read_to_string(&report_path).unwrap()).unwrap();
    assert_eq!(report_json["valid"], false);
    assert_eq!(
        report_json["contract_id"],
        "hemoval-development-coherence-v1-draft"
    );

    let nv: janus::ValidationReport =
        serde_json::from_value(report_json["neurosurgical_validation"].clone()).unwrap();
    assert_eq!(nv.total_records, 3);
    assert_eq!(nv.valid_records, 2);
    assert_eq!(nv.rejected_records, 1);
    assert_eq!(nv.missing_timestamp_count, 2);
    assert_eq!(nv.missing_grade_count, 4);
    assert_eq!(nv.integrity_failure_count, 1);
    assert_eq!(nv.plausibility_failure_count, 1);

    let sc: janus::SurvivalCurveComparison =
        serde_json::from_value(report_json["survival_comparison"].clone()).unwrap();
    assert_eq!(sc.real_curve.total_records, 5);
    assert_eq!(sc.real_curve.evaluable_records, 4);
    assert_eq!(sc.real_curve.missing_records, 1);
    assert_eq!(sc.synthetic_curve.total_records, 5);
    assert_eq!(sc.synthetic_curve.evaluable_records, 4);
    assert_eq!(sc.synthetic_curve.missing_records, 1);
    assert!(sc.log_rank_statistic.is_finite());
    assert!((0.0..=1.0).contains(&sc.log_rank_pvalue));
    assert!(sc.rmst_difference.is_finite());

    // 3. Passing only --survival-real without --survival-synthetic must fail
    assert!(!cli(&[
        "validate-contract",
        contract.to_str().unwrap(),
        "--survival-real",
        surv_real_csv.to_str().unwrap(),
    ]));

    // 4. JSON Vec<NeurosurgicalCoherenceRecord> input works too
    let neuro_json = dir.join("neuro.json");
    let valid_records = vec![janus::NeurosurgicalCoherenceRecord {
        temporal: janus::TemporalTransitionRecord {
            record_id: "j1".into(),
            admission_time_hours: Some(0.0),
            surgery_time_hours: Some(10.0),
            discharge_time_hours: Some(100.0),
            followup_90d_time_hours: Some(2160.0),
        },
        initial_wfns: Some(2),
        subsequent_wfns: Some(2),
        initial_hunt_hess: Some(2),
        subsequent_hunt_hess: Some(2),
        documented_acute_intervention: false,
        discharge_mrs: Some(1),
        followup_90d_mrs: Some(1),
        documented_recovery_intervention: false,
    }];
    fs::write(&neuro_json, serde_json::to_string(&valid_records).unwrap()).unwrap();
    let json_cmd = Command::new(env!("CARGO_BIN_EXE_janus-cli"))
        .args([
            "validate-contract",
            contract.to_str().unwrap(),
            "--neurosurgical-cohort",
            neuro_json.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(json_cmd.status.success());
    let json_out: serde_json::Value = serde_json::from_slice(&json_cmd.stdout).unwrap();
    assert_eq!(json_out["valid"], true);

    fs::remove_dir_all(dir).unwrap();
}
