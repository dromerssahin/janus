use clap::{Parser, Subcommand};
use janus::*;
use std::{
    collections::{BTreeMap, BTreeSet, HashMap, HashSet},
    fs,
    path::{Path, PathBuf},
};
fn write_artifacts(
    output: &Path,
    artifacts: &[(&str, String)],
) -> Result<(), Box<dyn std::error::Error>> {
    if output.symlink_metadata().is_ok() {
        return Err("output directory already exists".into());
    }
    let parent = output
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    fs::create_dir_all(parent)?;
    let mut staging_name = output
        .file_name()
        .ok_or("output must have a file name")?
        .to_os_string();
    staging_name.push(".janus-staging");
    let staging = parent.join(staging_name);
    fs::create_dir(&staging)?;
    let write_result = (|| -> Result<(), std::io::Error> {
        for (name, contents) in artifacts {
            fs::write(staging.join(name), contents)?;
        }
        fs::rename(&staging, output)
    })();
    if let Err(error) = write_result {
        fs::remove_dir_all(&staging).ok();
        return Err(error.into());
    }
    Ok(())
}
fn load_fixture(path: &Path, study: &Study) -> Result<Fixture, Box<dyn std::error::Error>> {
    if path.extension().is_some_and(|ext| ext == "parquet") {
        janus::tabular::load_dataset(path, study)
    } else {
        Ok(load(path)?)
    }
}
#[derive(Clone, Copy, Debug, Default, clap::ValueEnum)]
enum CliGeneratorKind {
    #[default]
    Configured,
    #[value(name = "neural-ctgan")]
    Neural,
}

#[derive(Parser)]
struct Cli {
    #[command(subcommand)]
    command: Cmd,
}
#[derive(Subcommand)]
enum Cmd {
    ValidateCsv {
        source: PathBuf,
        mapping: PathBuf,
    },
    ExportCsv {
        dataset: PathBuf,
        output: PathBuf,
    },
    Inspect {
        dataset: PathBuf,
        output: PathBuf,
    },
    ImportCsv {
        source: PathBuf,
        mapping: PathBuf,
        output: PathBuf,
    },
    DescribeFlat {
        source: PathBuf,
        config: PathBuf,
        output: PathBuf,
    },
    Evaluate {
        study: PathBuf,
        synthetic: PathBuf,
        reference: PathBuf,
        config: PathBuf,
        output: PathBuf,
        #[arg(long)]
        split_manifest: Option<PathBuf>,
        #[arg(long)]
        evidence_registry: Option<PathBuf>,
        #[arg(long)]
        synthetic_manifest: Option<PathBuf>,
    },
    Validate {
        study: PathBuf,
        fixture: PathBuf,
    },
    ValidateContract {
        contract: PathBuf,
        #[arg(long)]
        manifest: Option<PathBuf>,
        #[arg(long)]
        comparison: Option<PathBuf>,
        /// Optional neurosurgical cohort records (YAML/JSON Vec<NeurosurgicalCoherenceRecord> or CSV)
        #[arg(long)]
        neurosurgical_cohort: Option<PathBuf>,
        /// Optional real survival observations (YAML/JSON Vec<SurvivalObservation> or CSV)
        #[arg(long)]
        survival_real: Option<PathBuf>,
        /// Optional synthetic survival observations (YAML/JSON Vec<SurvivalObservation> or CSV)
        #[arg(long)]
        survival_synthetic: Option<PathBuf>,
        /// Optional JSON output path when running neurosurgical or survival validators
        #[arg(long)]
        output: Option<PathBuf>,
    },
    /// Verify replay prerequisites and audit manifest reproducibility
    AuditReproducibility {
        /// Path to run manifest JSON with fingerprints, lockfile hash, platform metadata
        #[arg(long)]
        manifest: PathBuf,
        /// Optional split manifest with patient allocation fingerprint
        #[arg(long)]
        splits: Option<PathBuf>,
        /// Fail on any mismatch vs warn-only mode
        #[arg(long, default_value_t = false)]
        strict: bool,
    },
    /// Run forensic evaluation comparing real and synthetic datasets
    ForensicEval {
        /// Path to real dataset (CSV or Parquet)
        #[arg(long)]
        real: PathBuf,
        /// Path to synthetic dataset (CSV or Parquet)
        #[arg(long)]
        synthetic: PathBuf,
        /// Optional training dataset (CSV or Parquet) for multi-seed forensic suite mode
        #[arg(long)]
        training: Option<PathBuf>,
        /// Random seed for discriminator
        #[arg(long, default_value_t = 42)]
        seed: u64,
        /// Optional comma-separated list of seeds for multi-seed forensic suite mode
        #[arg(long, value_delimiter = ',')]
        seeds: Option<Vec<u64>>,
        /// Run full forensic suite (Chi-squared, KS, Logistic Regression, Random Forest, small-cell suppression)
        #[arg(long, default_value_t = false)]
        suite: bool,
        /// Optional JSON output path (defaults to stdout)
        #[arg(long)]
        output: Option<PathBuf>,
    },
    /// Compare multiple candidate generators in memory and emit aggregate manifests only
    CompareCandidates {
        /// Path to CandidateComparisonConfig (YAML or JSON)
        #[arg(long)]
        config: PathBuf,
        /// Path to training dataset (CSV or Parquet)
        #[arg(long)]
        training: PathBuf,
        /// Path to reference dataset (CSV or Parquet)
        #[arg(long)]
        reference: PathBuf,
        /// Optional CoherenceContract (YAML or JSON) for contract_id preflight check
        #[arg(long)]
        contract: Option<PathBuf>,
        /// Optional NeuralGeneratorConfig (JSON or YAML) for neural candidates
        #[arg(long)]
        neural_config: Option<PathBuf>,
        /// Optional override for scripts/neural_bridge.py path
        #[arg(long)]
        bridge_script: Option<PathBuf>,
        /// Output directory for aggregate comparison and forensic manifests
        #[arg(long)]
        output: PathBuf,
    },
    Run {
        study: PathBuf,
        fixture: PathBuf,
        output: PathBuf,
    },
    Generate {
        study: PathBuf,
        generator_config: PathBuf,
        output: PathBuf,
        #[arg(long)]
        evidence_registry: Option<PathBuf>,
        /// Generator backend: 'configured' (default YAML generator) or 'neural-ctgan'
        #[arg(long = "generator", value_enum, default_value_t = CliGeneratorKind::Configured)]
        generator_kind: CliGeneratorKind,
        /// Training dataset (.csv or .parquet) for neural-ctgan generation
        #[arg(long)]
        training_data: Option<PathBuf>,
        /// Optional override for scripts/neural_bridge.py path
        #[arg(long)]
        bridge_script: Option<PathBuf>,
        /// Keep synthetic rows and fitted model in memory and emit aggregate summary + manifest only (neural-ctgan only)
        #[arg(long, default_value_t = false)]
        in_memory_aggregate: bool,
    },
    Split {
        study: PathBuf,
        fixture: PathBuf,
        config: PathBuf,
        output: PathBuf,
        #[arg(long)]
        evidence_registry: Option<PathBuf>,
    },
    Fit {
        study: PathBuf,
        development: PathBuf,
        config: PathBuf,
        output: PathBuf,
        #[arg(long)]
        evidence_registry: Option<PathBuf>,
        #[arg(long)]
        split_manifest: Option<PathBuf>,
    },
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    match Cli::parse().command {
        Cmd::ValidateCsv { source, mapping } => {
            let mapping: janus::tabular::CsvMapping = load(&mapping)?;
            let fixture = janus::tabular::read_csv(&source, &mapping)?;
            println!("valid: {} source rows", fixture.events.len());
        }
        Cmd::ExportCsv { dataset, output } => {
            let schema = janus::tabular::read_dataset_schema(&dataset)?;
            let fixture = janus::tabular::load_dataset(&dataset, &schema.mapping.study)?;
            janus::tabular::export_mapped_csv(&fixture, &schema.mapping, &output)?;
        }
        Cmd::Inspect { dataset, output } => {
            let schema = janus::tabular::read_dataset_schema(&dataset)?;
            janus::tabular::load_dataset(&dataset, &schema.mapping.study)?;
            let report = janus::tabular::inspect(&dataset, &schema.mapping.study)?;
            let mut report = serde_yaml::to_value(&report)?;
            let fields = report.as_mapping_mut().ok_or("invalid inspection report")?;
            fields.insert("dataset_schema".into(), serde_yaml::to_value(&schema)?);
            fields.insert(
                "validation_failures".into(),
                serde_yaml::to_value(Vec::<String>::new())?,
            );
            let mut manifest = serde_yaml::Mapping::new();
            manifest.insert("schema_version".into(), "1".into());
            manifest.insert(
                "artifacts".into(),
                serde_yaml::to_value(["inspection.yaml", "schema.yaml", "manifest.yaml"])?,
            );
            manifest.insert("software".into(), schema.creation_software.clone());
            manifest.insert(
                "study_fingerprint".into(),
                schema.study_fingerprint.clone().into(),
            );
            manifest.insert(
                "mapping_fingerprint".into(),
                schema.mapping_fingerprint.clone().into(),
            );
            manifest.insert(
                "data_fingerprint".into(),
                schema.data_fingerprint.clone().into(),
            );
            manifest.insert(
                "metrics".into(),
                serde_yaml::to_value([
                    "source_rows",
                    "distinct_patients",
                    "distinct_episodes",
                    "variable_missingness",
                    "numeric_moments_and_quantiles",
                    "categorical_frequencies",
                    "event_and_recording_timing",
                    "duplicate_event_ids",
                ])?,
            );
            write_artifacts(
                &output,
                &[
                    ("inspection.yaml", serde_yaml::to_string(&report)?),
                    ("schema.yaml", serde_yaml::to_string(&schema)?),
                    ("manifest.yaml", serde_yaml::to_string(&manifest)?),
                ],
            )?;
        }

        Cmd::ImportCsv {
            source,
            mapping,
            output,
        } => {
            let mapping: janus::tabular::CsvMapping = load(&mapping)?;
            let fixture = janus::tabular::read_csv(&source, &mapping)?;
            let mut schema = janus::tabular::dataset_schema(&mapping, fixture.events.len())?;
            if output.symlink_metadata().is_ok() {
                return Err("output directory already exists".into());
            }
            let parent = output
                .parent()
                .filter(|p| !p.as_os_str().is_empty())
                .unwrap_or(Path::new("."));
            fs::create_dir_all(parent)?;
            let mut staging_name = output
                .file_name()
                .ok_or("output must have a file name")?
                .to_os_string();
            staging_name.push(".janus-staging");
            let staging = parent.join(staging_name);
            fs::create_dir(&staging)?;
            let result = (|| -> Result<(), Box<dyn std::error::Error>> {
                janus::tabular::write_parquet(
                    &fixture,
                    &mapping.study,
                    &staging.join("data.parquet"),
                )?;
                schema.data_fingerprint =
                    janus::tabular::data_fingerprint(&staging.join("data.parquet"))?;
                fs::write(staging.join("schema.yaml"), serde_yaml::to_string(&schema)?)?;
                fs::write(
                    staging.join("validation.yaml"),
                    format!(
                        "schema_version: '1'\nvalid: true\nsource_rows: {}\nfailures: []\nchecks: [mapping, direct_identifier_headers, study_types, event_identifiers, timing, provenance, missingness, observation_metadata]\nlimitations:\n  - Header checks do not establish deidentification or governance approval.\n",
                        fixture.events.len()
                    ),
                )?;
                let mut manifest = serde_yaml::to_value(&schema)?;
                let fields = manifest
                    .as_mapping_mut()
                    .ok_or("invalid dataset manifest")?;
                fields.insert(
                    "artifacts".into(),
                    serde_yaml::to_value([
                        "data.parquet",
                        "schema.yaml",
                        "validation.yaml",
                        "manifest.yaml",
                    ])?,
                );
                fields.insert("limitations".into(), serde_yaml::to_value(["Declarations and header checks do not establish governance approval, anonymization, clinical validity, privacy, or external validity.", "Fingerprints are non-cryptographic identity checks and software and environment changes can affect reproducibility."])?);
                fs::write(
                    staging.join("manifest.yaml"),
                    serde_yaml::to_string(&manifest)?,
                )?;
                fs::rename(&staging, &output)?;
                Ok(())
            })();
            if result.is_err() {
                fs::remove_dir_all(&staging).ok();
            }
            result?;
            println!("wrote {}", output.display());
        }

        Cmd::DescribeFlat {
            source,
            config,
            output,
        } => {
            let source = source.canonicalize()?;
            let root = source
                .ancestors()
                .find(|ancestor| ancestor.file_name().is_some_and(|name| name == "datasets"))
                .ok_or("flat cohort source must be inside a datasets directory")?
                .to_path_buf();
            let config_path = config.canonicalize()?;
            let output_parent = output.parent().unwrap_or(Path::new("."));
            let output_parent = output_parent.canonicalize()?;
            let config: janus::tabular::FlatCohortConfig = load(&config_path)?;
            if !source.starts_with(&root)
                || !config_path.starts_with(&root)
                || !output_parent.starts_with(&root)
            {
                return Err("flat cohort input and output must remain under /datasets".into());
            }
            let report = janus::tabular::describe_flat_csv(&source, &config)?;
            let report_yaml = serde_yaml::to_string(&report)?;
            let manifest = janus::tabular::FlatCohortManifest {
                schema_version: "1".into(),
                source_fingerprint: janus::tabular::flat_fingerprint(&source)?,
                config_fingerprint: janus::yaml_fingerprint(&config)?,
                report_fingerprint: janus::yaml_fingerprint(&report)?,
                artifacts: vec!["report.yaml".into(), "manifest.yaml".into()],
                limitations: vec!["Fingerprints are non-cryptographic identity checks, not privacy guarantees.".into(), "Report contains aggregate cohort-derived data and must remain under approved storage terms.".into()],
            };
            write_artifacts(
                &output,
                &[
                    ("report.yaml", report_yaml),
                    ("manifest.yaml", serde_yaml::to_string(&manifest)?),
                ],
            )?;
            println!("wrote {}", output.display());
        }

        Cmd::Evaluate {
            study,
            synthetic,
            reference,
            config,
            output,
            split_manifest,
            evidence_registry,
            synthetic_manifest,
        } => {
            let config: EvaluationConfig = load(&config)?;
            let study: Study = load(&study)?;
            let reference_fixture = load_fixture(&reference, &study)?;
            validate_evidence_links(&study, None)?;
            let registry: Option<EvidenceRegistry> =
                evidence_registry.as_ref().map(|p| load(p)).transpose()?;
            if let Some(registry) = &registry {
                validate_registry_links(registry, &study, None)?;
            }
            let mut evidence = evidence_provenance(registry.as_ref(), &[&study.evidence])?;
            let holdout_provenance = if let Some(path) = split_manifest {
                let manifest: SplitManifest = load(&path)?;
                if manifest.evidence != evidence {
                    return Err("evidence does not match split manifest".into());
                }
                if let Some(fp) = &manifest.study_fingerprint {
                    if fp != &yaml_fingerprint(&study)? {
                        return Err("study does not match split manifest".into());
                    }
                }
                let expected = manifest
                    .holdout_fingerprint
                    .ok_or("split manifest has no holdout fingerprint")?;
                if fixture_fingerprint(&reference_fixture)? != expected {
                    return Err(
                        "reference does not match split manifest holdout fingerprint".into(),
                    );
                }
                Some("reference matched recorded holdout fixture fingerprint".to_string())
            } else {
                None
            };
            let synthetic_fixture = load_fixture(&synthetic, &study)?;
            if let Some(path) = synthetic_manifest {
                let manifest: serde_yaml::Value = load(&path)?;
                let recorded_fixture = manifest
                    .get("fixture_fingerprint")
                    .and_then(|v| v.as_str())
                    .ok_or("synthetic manifest has no fixture fingerprint")?;
                if recorded_fixture != fixture_fingerprint(&synthetic_fixture)? {
                    return Err("synthetic fixture does not match manifest".into());
                }
                let recorded_evidence: Option<EvidenceProvenance> = manifest
                    .get("evidence")
                    .cloned()
                    .map(serde_yaml::from_value)
                    .transpose()?;
                if recorded_evidence.as_ref().map(|e| &e.registry_fingerprint)
                    != evidence.as_ref().map(|e| &e.registry_fingerprint)
                {
                    return Err("evidence registry does not match synthetic manifest".into());
                }
                if let (Some(current), Some(recorded)) = (&mut evidence, recorded_evidence) {
                    if recorded.referenced_ids.iter().any(|id| {
                        !registry
                            .as_ref()
                            .is_some_and(|r| r.records.iter().any(|record| &record.id == id))
                    }) {
                        return Err("synthetic manifest references unknown evidence".into());
                    }
                    current.referenced_ids.extend(recorded.referenced_ids);
                    current.referenced_ids.sort();
                    current.referenced_ids.dedup();
                    current.records = current
                        .referenced_ids
                        .iter()
                        .map(|id| {
                            registry
                                .as_ref()
                                .unwrap()
                                .records
                                .iter()
                                .find(|record| &record.id == id)
                                .unwrap()
                                .clone()
                        })
                        .collect();
                }
            }
            let report = evaluate(&study, &synthetic_fixture, &reference_fixture, &config)?;
            let manifest = EvaluationManifest {
                schema_version: "1".into(),
                configuration: config.clone(),
                uncertainty: report.uncertainty.clone(),
                configuration_fingerprint: yaml_fingerprint(&config)?,
                artifacts: vec!["evaluation.yaml".into(), "manifest.yaml".into()],
                software_version: env!("CARGO_PKG_VERSION").into(),
                reproducibility_limits: "Repeatability depends on software, dependencies and environment; code revision and compiler flags are not recorded. Fingerprints identify serialized inputs only.".into(),
                study_name: study.name.clone(),
                study_schema_version: study.schema_version.clone(),
                study_fingerprint: yaml_fingerprint(&study)?,
                synthetic_fingerprint: fixture_fingerprint(&synthetic_fixture)?,
                reference_fingerprint: fixture_fingerprint(&reference_fixture)?,
                metric_definitions: vec!["Categorical marginals use observed included episodes; joints use complete observed pairs.".into(), "Temporal event distributions use all source events, including excluded episodes; index-relative distributions use events in indexed episodes. Episode spans use all source events. Episodes per patient use all source episodes.".into(), "Windows are inclusive relative to index event day; the latest event in the window recorded by its end determines observed, missing or absent, with indexed source episodes as denominator.".into(), "Last observed follow-up excludes the index event and includes observed values on days 0 through follow_up_days recorded by follow-up end; denominator is indexed episodes with such an event.".into(), "Temporal-order denominator is all source events; validation rejects recording before event.".into()],
                unperformed_checks: vec![if config.uncertainty.is_some() { "Clinical plausibility, privacy attacks, forensic detection, causal validity, external validation, uncertainty for temporal and longitudinal metrics, observation completeness and censoring assessment." } else { "Clinical plausibility, privacy attacks, forensic detection, causal validity, external validation, uncertainty, observation completeness and censoring assessment." }.into()],
                limitations: report.limitations.clone(),
                holdout_provenance,
                evidence,
                software: Some(software_identity()?),
            };
            write_artifacts(
                &output,
                &[
                    ("evaluation.yaml", serde_yaml::to_string(&report)?),
                    ("manifest.yaml", serde_yaml::to_string(&manifest)?),
                ],
            )?;
            println!("wrote {}", output.display());
        }
        Cmd::Generate {
            study,
            generator_config,
            output,
            evidence_registry,
            generator_kind,
            training_data,
            bridge_script,
            in_memory_aggregate,
        } => match generator_kind {
            CliGeneratorKind::Configured => {
                if in_memory_aggregate {
                    return Err(
                        "--in-memory-aggregate is only supported with --generator neural-ctgan"
                            .into(),
                    );
                }
                let study: Study = load(&study)?;
                let config: GeneratorConfig = load(&generator_config)?;
                validate_evidence_links(&study, Some(&config))?;
                let registry: Option<EvidenceRegistry> =
                    evidence_registry.as_ref().map(|p| load(p)).transpose()?;
                if let Some(registry) = &registry {
                    validate_registry_links(registry, &study, Some(&config))?;
                }
                let evidence =
                    evidence_provenance(registry.as_ref(), &[&study.evidence, &config.evidence])?;
                if let Some(expected) = &config.registry_fingerprint {
                    if evidence.as_ref().map(|e| &e.registry_fingerprint) != Some(expected) {
                        return Err("generator registry fingerprint does not match".into());
                    }
                }
                let fixture = generate(&study, &config)?;
                let mut r = run(&study, &fixture)?;
                r.manifest.evidence = evidence;
                r.manifest.software = Some(software_identity()?);
                r.manifest.study_fingerprint = Some(yaml_fingerprint(&study)?);
                r.manifest.generator_fingerprint = Some(yaml_fingerprint(&config)?);
                r.manifest.fixture_fingerprint = Some(fixture_fingerprint(&fixture)?);
                r.manifest.artifacts.push("fixture.yaml".into());
                r.manifest.artifacts.push("study.yaml".into());
                r.manifest.generator = Some(GeneratorManifest {
                    configuration: config.clone(),
                    rng: "SplitMix64; upper 53 bits mapped to [0,1)".into(),
                    software_version: env!("CARGO_PKG_VERSION").into(),
                    reproducibility_limits: if config.schema_version == "4" {
                        "Repeatability is tested only for the same software, dependencies and environment. Repeated counts and inclusive timing ranges are configured assumptions; complete observation is declared only when configured. This does not establish clinical plausibility, privacy, causal, censoring, or external validity.".into()
                    } else {
                        "Repeatability is tested only for the same software, dependencies and environment; floating-point and serialization behavior may differ across environments. Configured associations do not establish clinical, causal, privacy or external validity.".into()
                    },
                });
                let artifacts = [
                    ("study.yaml", serde_yaml::to_string(&study)?),
                    ("fixture.yaml", serde_yaml::to_string(&fixture)?),
                    ("cohort.yaml", serde_yaml::to_string(&r.cohort)?),
                    ("exclusions.yaml", serde_yaml::to_string(&r.exclusions)?),
                    ("summary.yaml", serde_yaml::to_string(&r.summary)?),
                    ("integrity.yaml", serde_yaml::to_string(&r.integrity)?),
                    ("diagnostics.yaml", serde_yaml::to_string(&r.diagnostics)?),
                    ("manifest.yaml", serde_yaml::to_string(&r.manifest)?),
                ];
                write_artifacts(&output, &artifacts)?;
                println!("wrote {}", output.display());
            }
            CliGeneratorKind::Neural => {
                run_neural_generate(
                    &study,
                    &generator_config,
                    &output,
                    training_data.as_deref(),
                    bridge_script.as_deref(),
                    in_memory_aggregate,
                )?;
                println!("wrote {}", output.display());
            }
        },
        Cmd::Split {
            study,
            fixture,
            config,
            output,
            evidence_registry,
        } => {
            let study: Study = load(&study)?;
            validate_evidence_links(&study, None)?;
            let registry: Option<EvidenceRegistry> =
                evidence_registry.as_ref().map(|p| load(p)).transpose()?;
            if let Some(registry) = &registry {
                validate_registry_links(registry, &study, None)?;
            }
            let evidence = evidence_provenance(registry.as_ref(), &[&study.evidence])?;
            let mut result =
                split_fixture(&study, &load_fixture(&fixture, &study)?, &load(&config)?)?;
            result.manifest.evidence = evidence;
            result.manifest.software = Some(software_identity()?);
            write_artifacts(
                &output,
                &[
                    (
                        "development.yaml",
                        serde_yaml::to_string(&result.development)?,
                    ),
                    ("holdout.yaml", serde_yaml::to_string(&result.holdout)?),
                    ("manifest.yaml", serde_yaml::to_string(&result.manifest)?),
                ],
            )?;
            println!("wrote {}", output.display());
        }
        Cmd::Fit {
            study,
            development,
            config,
            output,
            evidence_registry,
            split_manifest,
        } => {
            let study: Study = load(&study)?;
            validate_evidence_links(&study, None)?;
            let registry: Option<EvidenceRegistry> =
                evidence_registry.as_ref().map(|p| load(p)).transpose()?;
            if let Some(registry) = &registry {
                validate_registry_links(registry, &study, None)?;
            }
            let evidence = evidence_provenance(registry.as_ref(), &[&study.evidence])?;
            let development = load_fixture(&development, &study)?;
            if let Some(path) = split_manifest {
                let manifest: SplitManifest = load(&path)?;
                let study_fingerprint = yaml_fingerprint(&study)?;
                if manifest.evidence != evidence
                    || manifest
                        .study_fingerprint
                        .as_ref()
                        .is_some_and(|fp| fp != &study_fingerprint)
                    || manifest.development_fingerprint.as_ref()
                        != Some(&fixture_fingerprint(&development)?)
                {
                    return Err(
                        "development fixture or evidence does not match split manifest".into(),
                    );
                }
            }
            let mut result = fit(&study, &development, &load(&config)?)?;
            result.generator.registry_fingerprint =
                evidence.as_ref().map(|e| e.registry_fingerprint.clone());
            result.manifest.evidence = evidence;
            result.manifest.software = Some(software_identity()?);
            write_artifacts(
                &output,
                &[
                    ("generator.yaml", serde_yaml::to_string(&result.generator)?),
                    ("manifest.yaml", serde_yaml::to_string(&result.manifest)?),
                ],
            )?;
            println!("wrote {}", output.display());
        }
        Cmd::Validate { study, fixture } => {
            let study: Study = load(&study)?;
            validate(&study, &load_fixture(&fixture, &study)?)?;
            println!("valid")
        }
        Cmd::ValidateContract {
            contract,
            manifest,
            comparison,
            neurosurgical_cohort,
            survival_real,
            survival_synthetic,
            output,
        } => {
            run_validate_contract(ValidateContractArgs {
                contract: &contract,
                manifest: manifest.as_deref(),
                comparison: comparison.as_deref(),
                neurosurgical_cohort: neurosurgical_cohort.as_deref(),
                survival_real: survival_real.as_deref(),
                survival_synthetic: survival_synthetic.as_deref(),
                output: output.as_deref(),
            })?;
        }
        Cmd::Run {
            study,
            fixture,
            output,
        } => {
            if output.exists() {
                return Err("output directory already exists".into());
            }
            let study: Study = load(&study)?;
            let mut r = run(&study, &load_fixture(&fixture, &study)?)?;
            r.manifest.software = Some(software_identity()?);
            let artifacts = [
                ("cohort.yaml", serde_yaml::to_string(&r.cohort)?),
                ("exclusions.yaml", serde_yaml::to_string(&r.exclusions)?),
                ("summary.yaml", serde_yaml::to_string(&r.summary)?),
                ("integrity.yaml", serde_yaml::to_string(&r.integrity)?),
                ("diagnostics.yaml", serde_yaml::to_string(&r.diagnostics)?),
                ("manifest.yaml", serde_yaml::to_string(&r.manifest)?),
            ];
            write_artifacts(&output, &artifacts)?;
            println!("wrote {}", output.display())
        }
        Cmd::ForensicEval {
            real,
            synthetic,
            training,
            seed,
            seeds,
            suite,
            output,
        } => {
            run_forensic_eval_with_options(ForensicEvalOptions {
                real_path: &real,
                synthetic_path: &synthetic,
                training_path: training.as_deref(),
                seed,
                seeds: seeds.as_deref(),
                suite,
                output: output.as_deref(),
            })?;
        }
        Cmd::CompareCandidates {
            config,
            training,
            reference,
            contract,
            neural_config,
            bridge_script,
            output,
        } => {
            run_compare_candidates(CompareCandidatesArgs {
                config_path: &config,
                training_path: &training,
                reference_path: &reference,
                contract_path: contract.as_deref(),
                neural_config_path: neural_config.as_deref(),
                bridge_script: bridge_script.as_deref(),
                output: &output,
            })?;
            println!("wrote {}", output.display());
        }
        Cmd::AuditReproducibility {
            manifest,
            splits,
            strict,
        } => {
            let report = run_audit_reproducibility(&manifest, splits.as_deref(), strict)?;
            let json = serde_json::to_string_pretty(&report)?;
            println!("{json}");
            if strict
                && (!report.reproducible
                    || !report.warnings.is_empty()
                    || report.checks.iter().any(|c| !c.passed))
            {
                eprintln!(
                    "reproducibility audit failed in strict mode: {}",
                    report.warnings.join("; ")
                );
                std::process::exit(1);
            }
        }
    }
    Ok(())
}

fn load_json_or_yaml_value(path: &Path) -> Result<serde_json::Value, Box<dyn std::error::Error>> {
    let raw = fs::read_to_string(path)?;
    if let Ok(val) = serde_json::from_str::<serde_json::Value>(&raw) {
        return Ok(val);
    }
    let val: serde_json::Value = serde_yaml::from_str(&raw)?;
    Ok(val)
}

fn run_audit_reproducibility(
    manifest_path: &Path,
    splits_path: Option<&Path>,
    strict: bool,
) -> Result<janus::coherence::ReproducibilityAuditReport, Box<dyn std::error::Error>> {
    let manifest_val = load_json_or_yaml_value(manifest_path)?;
    let splits_val = match splits_path {
        Some(path) => Some(load_json_or_yaml_value(path)?),
        None => None,
    };
    Ok(janus::coherence::audit_reproducibility(
        &manifest_val,
        splits_val.as_ref(),
        strict,
    )?)
}

fn schema_fingerprint(columns: &[String], nrows: usize) -> String {
    let mut hash = 0xcbf29ce484222325u64;
    for col in columns {
        for byte in col.as_bytes() {
            hash ^= u64::from(*byte);
            hash = hash.wrapping_mul(0x100000001b3);
        }
        hash ^= 0u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    for byte in nrows.to_le_bytes() {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    format!("fnv1a64-schema:{hash:016x}")
}

fn utc_timestamp_now() -> String {
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    let days = (secs / 86_400) as i64;
    let rem = secs % 86_400;
    let hour = rem / 3_600;
    let minute = (rem % 3_600) / 60;
    let second = rem % 60;

    // Howard Hinnant's civil_from_days algorithm
    let z = days + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = (z - era * 146_097) as u64;
    let yoe = (doe - doe / 1_460 + doe / 36_524 - doe / 146_096) / 365;
    let y = (yoe as i64) + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };

    format!("{y:04}-{m:02}-{d:02}T{hour:02}:{minute:02}:{second:02}Z")
}

struct ForensicEvalOptions<'a> {
    real_path: &'a Path,
    synthetic_path: &'a Path,
    training_path: Option<&'a Path>,
    seed: u64,
    seeds: Option<&'a [u64]>,
    suite: bool,
    output: Option<&'a Path>,
}

struct CompareCandidatesArgs<'a> {
    config_path: &'a Path,
    training_path: &'a Path,
    reference_path: &'a Path,
    contract_path: Option<&'a Path>,
    neural_config_path: Option<&'a Path>,
    bridge_script: Option<&'a Path>,
    output: &'a Path,
}

struct ValidateContractArgs<'a> {
    contract: &'a Path,
    manifest: Option<&'a Path>,
    comparison: Option<&'a Path>,
    neurosurgical_cohort: Option<&'a Path>,
    survival_real: Option<&'a Path>,
    survival_synthetic: Option<&'a Path>,
    output: Option<&'a Path>,
}

fn write_json_output(
    json: &str,
    output: Option<&Path>,
) -> Result<(), Box<dyn std::error::Error>> {
    if let Some(out_path) = output {
        if let Some(parent) = out_path.parent().filter(|p| !p.as_os_str().is_empty()) {
            fs::create_dir_all(parent)?;
        }
        fs::write(out_path, json)?;
    } else {
        println!("{json}");
    }
    Ok(())
}

fn load_candidate_rows(path: &Path) -> Result<Vec<janus::Row>, Box<dyn std::error::Error>> {
    if !path.is_file() {
        return Err(format!("dataset file not found: {}", path.display()).into());
    }
    if path
        .extension()
        .is_some_and(|ext| ext.eq_ignore_ascii_case("parquet"))
    {
        let cohort = janus::tabular::load_forensic_cohort(path, "candidate")?;
        if cohort.rows.is_empty() {
            return Err(format!("dataset is empty (0 rows): {}", path.display()).into());
        }
        let rows = cohort
            .rows
            .into_iter()
            .map(|fr| {
                let mut fields = BTreeMap::new();
                for (k, val_opt) in fr.numeric_fields {
                    let token = match val_opt {
                        Some(v) if v.is_finite() => {
                            if v == 0.0 {
                                "0".to_string()
                            } else {
                                v.to_string()
                            }
                        }
                        _ => "NA".to_string(),
                    };
                    fields.insert(k, token);
                }
                for (k, val_opt) in fr.categorical_fields {
                    fields.insert(k, val_opt.unwrap_or_else(|| "NA".to_string()));
                }
                janus::Row::new(fields)
            })
            .collect();
        return Ok(rows);
    }

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

    let mut rows = Vec::new();
    for record in reader.records() {
        let record = record?;
        let mut fields = BTreeMap::new();
        for (idx, header) in headers.iter().enumerate() {
            let raw = record.get(idx).unwrap_or("").trim().to_string();
            fields.insert(header.clone(), raw);
        }
        rows.push(janus::Row::new(fields));
    }
    if rows.is_empty() {
        return Err(format!("dataset is empty (0 rows): {}", path.display()).into());
    }
    Ok(rows)
}

fn normalize_fixture_token(kind: &str, raw: &str) -> Option<String> {
    match kind {
        "string" => Some(raw.to_string()),
        "boolean" => {
            if raw.eq_ignore_ascii_case("true") {
                Some("true".to_string())
            } else if raw.eq_ignore_ascii_case("false") {
                Some("false".to_string())
            } else {
                None
            }
        }
        "integer" => {
            if raw.parse::<i64>().is_ok() {
                Some(raw.to_string())
            } else if let Ok(f) = raw.parse::<f64>() {
                if f.is_finite()
                    && f.fract() == 0.0
                    && f >= (i64::MIN as f64)
                    && f < 9_223_372_036_854_775_808.0
                {
                    Some((f as i64).to_string())
                } else {
                    None
                }
            } else {
                None
            }
        }
        "number" => raw
            .parse::<f64>()
            .is_ok_and(f64::is_finite)
            .then(|| raw.to_string()),
        _ => None,
    }
}

fn samples_csv_to_fixture(
    samples_path: &Path,
    study: &Study,
) -> Result<Option<Fixture>, Box<dyn std::error::Error>> {
    let mut reader = csv::Reader::from_path(samples_path)?;
    let headers: Vec<String> = reader
        .headers()?
        .iter()
        .map(|h| h.trim().to_string())
        .collect();
    if headers.len() != study.variables.len() {
        return Ok(None);
    }
    let header_indices: HashMap<&str, usize> = headers
        .iter()
        .enumerate()
        .map(|(idx, name)| (name.as_str(), idx))
        .collect();
    if header_indices.len() != study.variables.len()
        || study
            .variables
            .iter()
            .any(|v| !header_indices.contains_key(v.name.as_str()))
    {
        return Ok(None);
    }

    let mut events = Vec::new();
    let mut row_count = 0usize;
    for record in reader.records() {
        let record = record?;
        let patient_id = format!("neural-p{row_count:06}");
        let episode_id = format!("neural-e{row_count:06}");
        for (var_idx, var) in study.variables.iter().enumerate() {
            let col_idx = header_indices[var.name.as_str()];
            let raw = record.get(col_idx).unwrap_or("").trim();
            let day = match var.role {
                Some(VariableRole::FollowUpOutcome) => study.follow_up_days.min(1),
                _ => 0,
            };
            let (value, missing) =
                if janus::token_is_missing(raw) || raw.eq_ignore_ascii_case("nan") {
                    (None, Some("missing".to_string()))
                } else {
                    match normalize_fixture_token(&var.kind, raw) {
                        Some(normalized) => (Some(normalized), None),
                        None => return Ok(None),
                    }
                };
            events.push(Event {
                patient_id: patient_id.clone(),
                episode_id: episode_id.clone(),
                id: format!("neural-ev-{row_count:06}-{var_idx:03}"),
                variable: var.name.clone(),
                value,
                missing,
                event_day: day,
                recording_day: day,
                provenance: "neural_bridge_sample".into(),
            });
        }
        row_count += 1;
    }

    if row_count == 0 {
        return Ok(None);
    }

    let fixture = Fixture {
        schema_version: "1".into(),
        events,
        observations: Vec::new(),
    };
    if janus::validate(study, &fixture).is_ok() {
        Ok(Some(fixture))
    } else {
        Ok(None)
    }
}

fn run_neural_generate_in_memory(
    study: &Study,
    config: &janus::neural::NeuralGeneratorConfig,
    training_path: &Path,
    output: &Path,
    bridge_script: Option<&Path>,
) -> Result<(), Box<dyn std::error::Error>> {
    if output.symlink_metadata().is_ok() {
        return Err("output directory already exists".into());
    }
    let training_rows = load_candidate_rows(training_path)?;
    let sample_seeds = config
        .sample_seeds
        .clone()
        .unwrap_or_else(|| vec![config.seed]);
    let (bridge_manifest, draws) = janus::neural::run_neural_bridge_in_memory(
        &training_rows,
        config,
        &sample_seeds,
        bridge_script,
    )?;

    let mut variable_counts_by_seed = BTreeMap::new();
    for (&seed_val, rows) in &draws {
        let mut col_counts = BTreeMap::new();
        for col in &bridge_manifest.training_columns {
            let mut observed = 0usize;
            let mut missing = 0usize;
            for row in rows {
                let raw = row.get(col).unwrap_or("");
                if janus::token_is_missing(raw) {
                    missing += 1;
                } else {
                    observed += 1;
                }
            }
            col_counts.insert(
                col.clone(),
                serde_json::json!({
                    "observed": observed,
                    "missing": missing,
                }),
            );
        }
        variable_counts_by_seed.insert(seed_val.to_string(), col_counts);
    }

    let summary = serde_json::json!({
        "schema_version": "1",
        "generator_kind": janus::GeneratorKind::Neural,
        "model_type": bridge_manifest.model_type,
        "fit_seed": bridge_manifest.seed,
        "sample_seeds": sample_seeds,
        "training_rows": bridge_manifest.training_rows,
        "rows_generated_per_seed": bridge_manifest.n_samples,
        "columns": bridge_manifest.training_columns,
        "variable_counts_by_seed": variable_counts_by_seed,
        "training_dataset_fingerprint": bridge_manifest.training_dataset_fingerprint,
        "samples_fingerprint": bridge_manifest.samples_fingerprint,
        "limitations": [
            "Aggregate-only in-memory execution: patient rows, synthetic rows, and fitted neural weights were never written to disk.",
            "Observed/missing counts summarize generated column availability only and do not establish clinical plausibility, causal validity, or privacy safety.",
        ],
    });

    let sw = janus::software_identity()?;
    let manifest = janus::neural::NeuralGeneratorManifest {
        schema_version: "1".into(),
        generator_kind: janus::GeneratorKind::Neural,
        model_type: bridge_manifest.model_type,
        hyperparameters: bridge_manifest.hyperparameters,
        epochs: bridge_manifest.epochs,
        seed: bridge_manifest.seed,
        n_samples: bridge_manifest.n_samples,
        training_dataset_fingerprint: bridge_manifest.training_dataset_fingerprint,
        training_rows: bridge_manifest.training_rows,
        training_columns: bridge_manifest.training_columns,
        discrete_columns: bridge_manifest.discrete_columns,
        samples_fingerprint: bridge_manifest.samples_fingerprint,
        study_fingerprint: Some(yaml_fingerprint(study)?),
        config_fingerprint: Some(yaml_fingerprint(config)?),
        fixture_fingerprint: None,
        timestamp: bridge_manifest.timestamp,
        python_version: bridge_manifest.python_version,
        ctgan_version: bridge_manifest.ctgan_version,
        torch_version: bridge_manifest.torch_version,
        software_version: sw.package_version,
        dependency_lock_fingerprint: Some(sw.dependency_lock_fingerprint),
        operating_system: Some(sw.operating_system),
        architecture: Some(sw.architecture),
        artifacts: vec![
            "study.yaml".into(),
            "config.json".into(),
            "aggregate_summary.json".into(),
            "manifest.json".into(),
        ],
        reproducibility_limits: "In-memory aggregate-only execution kept synthetic rows and model weights off disk. Deterministic seeding depends on Python, PyTorch, and ctgan versions and hardware backend; identical seeds are verified only within the recorded environment and do not establish clinical, causal, or privacy validity.".into(),
    };

    write_artifacts(
        output,
        &[
            ("study.yaml", serde_yaml::to_string(study)?),
            ("config.json", serde_json::to_string_pretty(config)?),
            (
                "aggregate_summary.json",
                serde_json::to_string_pretty(&summary)?,
            ),
            ("manifest.json", manifest.to_json()?),
        ],
    )?;
    Ok(())
}

fn run_neural_generate(
    study_path: &Path,
    config_path: &Path,
    output: &Path,
    training_data: Option<&Path>,
    bridge_script: Option<&Path>,
    in_memory_aggregate: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    let study: Study = load(study_path)?;
    let config: janus::neural::NeuralGeneratorConfig = load(config_path)?;
    config.validate()?;

    let training_path = match training_data {
        Some(p) => p.to_path_buf(),
        None => match &config.training_data {
            Some(p) => PathBuf::from(p),
            None => {
                return Err(
                    "--training-data is required when using --generator neural-ctgan".into(),
                )
            }
        },
    };

    if in_memory_aggregate {
        return run_neural_generate_in_memory(
            &study,
            &config,
            &training_path,
            output,
            bridge_script,
        );
    }

    if output.symlink_metadata().is_ok() {
        return Err("output directory already exists".into());
    }
    let parent = output
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    fs::create_dir_all(parent)?;
    let mut staging_name = output
        .file_name()
        .ok_or("output must have a file name")?
        .to_os_string();
    staging_name.push(".janus-staging");
    let staging = parent.join(staging_name);
    fs::create_dir(&staging)?;

    let result = (|| -> Result<(), Box<dyn std::error::Error>> {
        let bridge_manifest =
            janus::neural::run_neural_bridge(&training_path, &config, &staging, bridge_script)?;
        let sw = janus::software_identity()?;
        let mut artifacts = vec![
            "study.yaml".into(),
            "config.json".into(),
            "samples.csv".into(),
            "model.pkl".into(),
            "neural_manifest.json".into(),
            "manifest.json".into(),
        ];
        let mut fixture_fp = None;
        if let Some(fixture) = samples_csv_to_fixture(&staging.join("samples.csv"), &study)? {
            if let Ok(run_report) = janus::run(&study, &fixture) {
                fs::write(staging.join("fixture.yaml"), serde_yaml::to_string(&fixture)?)?;
                fs::write(
                    staging.join("cohort.yaml"),
                    serde_yaml::to_string(&run_report.cohort)?,
                )?;
                fs::write(
                    staging.join("exclusions.yaml"),
                    serde_yaml::to_string(&run_report.exclusions)?,
                )?;
                fs::write(
                    staging.join("summary.yaml"),
                    serde_yaml::to_string(&run_report.summary)?,
                )?;
                fs::write(
                    staging.join("integrity.yaml"),
                    serde_yaml::to_string(&run_report.integrity)?,
                )?;
                fs::write(
                    staging.join("diagnostics.yaml"),
                    serde_yaml::to_string(&run_report.diagnostics)?,
                )?;
                fixture_fp = Some(janus::fixture_fingerprint(&fixture)?);
                artifacts.extend([
                    "fixture.yaml".into(),
                    "cohort.yaml".into(),
                    "exclusions.yaml".into(),
                    "summary.yaml".into(),
                    "integrity.yaml".into(),
                    "diagnostics.yaml".into(),
                ]);
            }
        }
        let manifest = janus::neural::NeuralGeneratorManifest {
            schema_version: "1".into(),
            generator_kind: janus::GeneratorKind::Neural,
            model_type: bridge_manifest.model_type,
            hyperparameters: bridge_manifest.hyperparameters,
            epochs: bridge_manifest.epochs,
            seed: bridge_manifest.seed,
            n_samples: bridge_manifest.n_samples,
            training_dataset_fingerprint: bridge_manifest.training_dataset_fingerprint,
            training_rows: bridge_manifest.training_rows,
            training_columns: bridge_manifest.training_columns,
            discrete_columns: bridge_manifest.discrete_columns,
            samples_fingerprint: bridge_manifest.samples_fingerprint,
            study_fingerprint: Some(yaml_fingerprint(&study)?),
            config_fingerprint: Some(yaml_fingerprint(&config)?),
            fixture_fingerprint: fixture_fp,
            timestamp: bridge_manifest.timestamp,
            python_version: bridge_manifest.python_version,
            ctgan_version: bridge_manifest.ctgan_version,
            torch_version: bridge_manifest.torch_version,
            software_version: sw.package_version,
            dependency_lock_fingerprint: Some(sw.dependency_lock_fingerprint),
            operating_system: Some(sw.operating_system),
            architecture: Some(sw.architecture),
            artifacts,
            reproducibility_limits: "Deterministic seeding depends on Python, PyTorch, and ctgan versions and hardware backend; identical seeds are verified only within the recorded environment and do not establish clinical, causal, or privacy validity.".into(),
        };
        fs::write(staging.join("study.yaml"), serde_yaml::to_string(&study)?)?;
        fs::write(staging.join("manifest.json"), manifest.to_json()?)?;
        fs::rename(&staging, output)?;
        Ok(())
    })();

    if result.is_err() {
        fs::remove_dir_all(&staging).ok();
    }
    result
}

fn cohort_column_set(cohort: &janus::ForensicCohort) -> BTreeSet<String> {
    let mut cols = BTreeSet::new();
    for row in &cohort.rows {
        cols.extend(row.numeric_fields.keys().cloned());
        cols.extend(row.categorical_fields.keys().cloned());
    }
    cols
}

fn run_forensic_eval_with_options(
    opts: ForensicEvalOptions<'_>,
) -> Result<(), Box<dyn std::error::Error>> {
    if opts.suite || opts.training_path.is_some() || opts.seeds.is_some() {
        let effective_seeds: Vec<u64> = match opts.seeds {
            Some(s) => {
                if s.is_empty() {
                    return Err("--seeds must contain at least one seed".into());
                }
                s.to_vec()
            }
            None => vec![opts.seed],
        };
        let reference = janus::tabular::load_forensic_cohort(opts.real_path, "reference")?;
        let generated = janus::tabular::load_forensic_cohort(opts.synthetic_path, "synthetic")?;
        let training_cohort = match opts.training_path {
            Some(path) => janus::tabular::load_forensic_cohort(path, "training")?,
            None => {
                let mut cloned = reference.clone();
                cloned.cohort_id = "training".into();
                cloned
            }
        };

        let ref_cols = cohort_column_set(&reference);
        let gen_cols = cohort_column_set(&generated);
        let train_cols = cohort_column_set(&training_cohort);
        if ref_cols != gen_cols || ref_cols != train_cols {
            return Err(format!(
                "incompatible schemas across training {:?}, reference {:?}, and synthetic {:?}",
                train_cols, ref_cols, gen_cols
            )
            .into());
        }

        let manifest = janus::orchestrate_forensic_suite(
            &training_cohort,
            &generated,
            &reference,
            &effective_seeds,
        )?;
        let json = manifest.to_json()?;
        write_json_output(&json, opts.output)?;
        return Ok(());
    }

    let real_table = janus::tabular::load_numeric_table(opts.real_path)?;
    let synth_table = janus::tabular::load_numeric_table(opts.synthetic_path)?;

    if real_table.columns != synth_table.columns {
        return Err(format!(
            "incompatible schemas: real numeric columns {:?} do not match synthetic numeric columns {:?}",
            real_table.columns, synth_table.columns
        )
        .into());
    }

    let real_cohort = janus::tabular::load_forensic_cohort(opts.real_path, "real")?;
    let synth_cohort = janus::tabular::load_forensic_cohort(opts.synthetic_path, "synthetic")?;
    let real_cat_cols: BTreeSet<String> = real_cohort
        .rows
        .first()
        .map(|r| r.categorical_fields.keys().cloned().collect())
        .unwrap_or_default();
    let synth_cat_cols: BTreeSet<String> = synth_cohort
        .rows
        .first()
        .map(|r| r.categorical_fields.keys().cloned().collect())
        .unwrap_or_default();
    if real_cat_cols != synth_cat_cols {
        return Err(format!(
            "incompatible schemas: real categorical columns {:?} do not match synthetic categorical columns {:?}",
            real_cat_cols, synth_cat_cols
        )
        .into());
    }

    let nrows_real = real_table.nrows();
    let nrows_synth = synth_table.nrows();
    let mut distributional_tests =
        Vec::with_capacity(real_table.columns.len() + real_cat_cols.len());
    for (col_idx, col_name) in real_table.columns.iter().enumerate() {
        let real_col = real_table.observed_column(col_idx);
        let synth_col = synth_table.observed_column(col_idx);
        if real_col.is_empty() || synth_col.is_empty() {
            let side = match (real_col.is_empty(), synth_col.is_empty()) {
                (true, true) => "both real and synthetic datasets",
                (true, false) => "real dataset",
                (false, true) => "synthetic dataset",
                (false, false) => unreachable!(),
            };
            return Err(janus::Error::Input(format!(
                "column '{col_name}' has zero observed finite values in {side}"
            ))
            .into());
        }
        distributional_tests.push(janus::forensic::kolmogorov_smirnov_test(
            col_name, &real_col, &synth_col,
        )?);
    }

    for col_name in &real_cat_cols {
        let collect_vals = |cohort: &janus::ForensicCohort| -> Vec<String> {
            cohort
                .rows
                .iter()
                .filter_map(|r| r.categorical_fields.get(col_name).and_then(|v| v.as_ref()))
                .map(|s| s.trim().to_string())
                .filter(|s| !janus::token_is_missing(s))
                .collect()
        };
        let real_vals = collect_vals(&real_cohort);
        let synth_vals = collect_vals(&synth_cohort);
        if !real_vals.is_empty() && !synth_vals.is_empty() {
            distributional_tests.push(janus::forensic::chi_squared_test(
                col_name,
                &real_vals,
                &synth_vals,
            )?);
        }
    }

    let correlation_distance =
        janus::forensic::correlation_distance_optional(&real_table.data, &synth_table.data)?;
    let (real_features, synth_features) =
        janus::tabular::NumericTable::build_standardized_features_with_indicators(
            &real_table,
            &synth_table,
        )?;
    let discriminator =
        janus::forensic::train_discriminator(&real_features, &synth_features, opts.seed)?;

    let report = janus::forensic::ForensicReport {
        dataset_fingerprint_real: schema_fingerprint(&real_table.columns, nrows_real),
        dataset_fingerprint_synthetic: schema_fingerprint(&synth_table.columns, nrows_synth),
        seed: opts.seed,
        distributional_tests,
        correlation_distance,
        temporal_patterns: None,
        discriminator,
        timestamp: utc_timestamp_now(),
    };

    let json = report.to_json()?;
    write_json_output(&json, opts.output)?;
    Ok(())
}

#[cfg(test)]
fn run_forensic_eval(
    real_path: &Path,
    synthetic_path: &Path,
    seed: u64,
    output: Option<&Path>,
) -> Result<(), Box<dyn std::error::Error>> {
    run_forensic_eval_with_options(ForensicEvalOptions {
        real_path,
        synthetic_path,
        training_path: None,
        seed,
        seeds: None,
        suite: false,
        output,
    })
}

fn build_candidate_generator(
    name: &str,
    neural_config: Option<&janus::neural::NeuralGeneratorConfig>,
    bridge_script: Option<&Path>,
) -> Result<Box<dyn janus::CohortGenerator>, Box<dyn std::error::Error>> {
    match name {
        "forest" | "learned_binned_forest" => Ok(Box::new(janus::ForestCandidate)),
        "dag" | "learned_binned_dag" => Ok(Box::new(janus::DagCandidate)),
        "interaction-dag" | "interaction_binned_dag" => {
            Ok(Box::new(janus::InteractionDagCandidate))
        }
        "neural" | "neural-ctgan" | "neural-tvae" | "experimental_neural" => {
            let mut cfg = neural_config.cloned().unwrap_or_default();
            if name == "neural-tvae" {
                cfg.model_type = janus::neural::NeuralModelType::Tvae;
            }
            cfg.validate()?;
            Ok(Box::new(janus::NeuralCandidate {
                config: cfg,
                bridge_script: bridge_script.map(Path::to_path_buf),
            }))
        }
        other => Err(format!(
            "unknown candidate generator '{other}'; expected one of: forest, learned_binned_forest, dag, learned_binned_dag, interaction-dag, interaction_binned_dag, neural, neural-ctgan, neural-tvae, experimental_neural"
        )
        .into()),
    }
}

fn run_compare_candidates(args: CompareCandidatesArgs<'_>) -> Result<(), Box<dyn std::error::Error>> {
    if args.output.symlink_metadata().is_ok() {
        return Err("output directory already exists".into());
    }
    let comparison_cfg: janus::CandidateComparisonConfig = load(args.config_path)?;
    comparison_cfg.validate()?;

    if let Some(contract_path) = args.contract_path {
        let contract: janus::coherence::CoherenceContract = load(contract_path)?;
        contract.validate()?;
        if comparison_cfg.contract_id != contract.contract_id {
            return Err("candidate comparison contract_id does not match coherence contract".into());
        }
    }

    let neural_cfg: Option<janus::neural::NeuralGeneratorConfig> = args
        .neural_config_path
        .map(load::<janus::neural::NeuralGeneratorConfig>)
        .transpose()?;

    let training_rows = load_candidate_rows(args.training_path)?;
    let reference_rows = load_candidate_rows(args.reference_path)?;

    let mut generators: Vec<Box<dyn janus::CohortGenerator>> =
        Vec::with_capacity(comparison_cfg.candidate_names.len());
    for name in &comparison_cfg.candidate_names {
        generators.push(build_candidate_generator(
            name,
            neural_cfg.as_ref(),
            args.bridge_script,
        )?);
    }
    let candidate_refs: Vec<&dyn janus::CohortGenerator> =
        generators.iter().map(|g| g.as_ref()).collect();

    let comparison_manifest = janus::compare_candidates(
        &training_rows,
        &reference_rows,
        &candidate_refs,
        &comparison_cfg,
    )?;

    let training_cohort = janus::ForensicCohort::from_candidate_rows("training", &training_rows)?;
    let reference_cohort =
        janus::ForensicCohort::from_candidate_rows("reference", &reference_rows)?;

    let mut forensic_entries = Vec::with_capacity(
        comparison_cfg.candidate_names.len() * comparison_cfg.seeds.len(),
    );
    for (candidate_name, generator) in comparison_cfg
        .candidate_names
        .iter()
        .zip(generators.iter())
    {
        let fitted = generator.fit(&training_rows)?;
        let draws = generator.sample_batch(
            &fitted,
            comparison_cfg.sample_count_per_seed,
            &comparison_cfg.seeds,
        )?;
        for &sample_seed in &comparison_cfg.seeds {
            let drawn_rows = draws.get(&sample_seed).ok_or_else(|| {
                format!(
                    "candidate '{candidate_name}' did not return sampled rows for seed {sample_seed}"
                )
            })?;
            let generated_cohort = janus::ForensicCohort::from_candidate_rows(
                format!("{candidate_name}-seed-{sample_seed}"),
                drawn_rows,
            )?;
            let manifest = janus::orchestrate_forensic_suite(
                &training_cohort,
                &generated_cohort,
                &reference_cohort,
                &comparison_cfg.seeds,
            )?;
            forensic_entries.push(serde_json::json!({
                "candidate_name": candidate_name,
                "sample_seed": sample_seed,
                "manifest": manifest,
            }));
        }
    }

    write_artifacts(
        args.output,
        &[
            (
                "comparison_manifest.json",
                serde_json::to_string_pretty(&comparison_manifest)?,
            ),
            (
                "forensic_manifests.json",
                serde_json::to_string_pretty(&forensic_entries)?,
            ),
        ],
    )?;
    Ok(())
}

fn parse_bool_token(raw: &str, field_name: &str) -> Result<bool, Box<dyn std::error::Error>> {
    match raw.trim().to_ascii_lowercase().as_str() {
        "true" | "1" => Ok(true),
        "false" | "0" => Ok(false),
        other => Err(format!("invalid boolean token '{other}' for field '{field_name}'").into()),
    }
}

fn csv_header_index(
    headers: &[String],
    required: &[&str],
    path: &Path,
) -> Result<HashMap<String, usize>, Box<dyn std::error::Error>> {
    if headers.iter().collect::<HashSet<_>>().len() != headers.len() {
        return Err(format!("CSV contains duplicate headers: {}", path.display()).into());
    }
    let map: HashMap<String, usize> = headers
        .iter()
        .enumerate()
        .map(|(idx, h)| (h.clone(), idx))
        .collect();
    for &col in required {
        if !map.contains_key(col) {
            return Err(format!(
                "CSV {} is missing required column '{col}'",
                path.display()
            )
            .into());
        }
    }
    Ok(map)
}

fn load_neurosurgical_records(
    path: &Path,
) -> Result<Vec<janus::NeurosurgicalCoherenceRecord>, Box<dyn std::error::Error>> {
    if !path
        .extension()
        .is_some_and(|ext| ext.eq_ignore_ascii_case("csv"))
    {
        return Ok(load(path)?);
    }

    const REQUIRED: &[&str] = &[
        "record_id",
        "admission_time_hours",
        "surgery_time_hours",
        "discharge_time_hours",
        "followup_90d_time_hours",
        "initial_wfns",
        "subsequent_wfns",
        "initial_hunt_hess",
        "subsequent_hunt_hess",
        "documented_acute_intervention",
        "discharge_mrs",
        "followup_90d_mrs",
        "documented_recovery_intervention",
    ];
    let mut reader = csv::Reader::from_path(path)?;
    let headers: Vec<String> = reader
        .headers()?
        .iter()
        .map(|h| h.trim().to_string())
        .collect();
    let idx = csv_header_index(&headers, REQUIRED, path)?;
    let mut records = Vec::new();
    for record in reader.records() {
        let record = record?;
        let get = |col: &str| record.get(idx[col]).unwrap_or("").trim();
        let temporal = janus::TemporalTransitionRecord::from_raw_strings(
            get("record_id"),
            get("admission_time_hours"),
            get("surgery_time_hours"),
            get("discharge_time_hours"),
            get("followup_90d_time_hours"),
        )
        .map_err(|failures| {
            failures
                .into_iter()
                .map(|f| format!("{}: {}", f.rule, f.message))
                .collect::<Vec<_>>()
                .join("; ")
        })?;
        let parse_u8 = |col: &str| -> Result<Option<u8>, Box<dyn std::error::Error>> {
            Ok(janus::parse_optional_u8(get(col), col)?.as_option().copied())
        };
        records.push(janus::NeurosurgicalCoherenceRecord {
            temporal,
            initial_wfns: parse_u8("initial_wfns")?,
            subsequent_wfns: parse_u8("subsequent_wfns")?,
            initial_hunt_hess: parse_u8("initial_hunt_hess")?,
            subsequent_hunt_hess: parse_u8("subsequent_hunt_hess")?,
            documented_acute_intervention: parse_bool_token(
                get("documented_acute_intervention"),
                "documented_acute_intervention",
            )?,
            discharge_mrs: parse_u8("discharge_mrs")?,
            followup_90d_mrs: parse_u8("followup_90d_mrs")?,
            documented_recovery_intervention: parse_bool_token(
                get("documented_recovery_intervention"),
                "documented_recovery_intervention",
            )?,
        });
    }
    if records.is_empty() {
        return Err(format!("neurosurgical cohort CSV is empty: {}", path.display()).into());
    }
    Ok(records)
}

fn load_survival_observations(
    path: &Path,
) -> Result<Vec<janus::SurvivalObservation>, Box<dyn std::error::Error>> {
    if !path
        .extension()
        .is_some_and(|ext| ext.eq_ignore_ascii_case("csv"))
    {
        return Ok(load(path)?);
    }

    const REQUIRED: &[&str] = &["record_id", "time", "event_occurred"];
    let mut reader = csv::Reader::from_path(path)?;
    let headers: Vec<String> = reader
        .headers()?
        .iter()
        .map(|h| h.trim().to_string())
        .collect();
    let idx = csv_header_index(&headers, REQUIRED, path)?;
    let mut observations = Vec::new();
    for record in reader.records() {
        let record = record?;
        let get = |col: &str| record.get(idx[col]).unwrap_or("").trim();
        let obs = janus::SurvivalObservation::from_raw_strings(
            get("record_id"),
            get("time"),
            get("event_occurred"),
        )
        .map_err(|failures| {
            failures
                .into_iter()
                .map(|f| format!("{}: {}", f.rule, f.message))
                .collect::<Vec<_>>()
                .join("; ")
        })?;
        observations.push(obs);
    }
    if observations.is_empty() {
        return Err(format!("survival observations CSV is empty: {}", path.display()).into());
    }
    Ok(observations)
}

fn run_validate_contract(
    args: ValidateContractArgs<'_>,
) -> Result<(), Box<dyn std::error::Error>> {
    let contract: janus::coherence::CoherenceContract = load(args.contract)?;
    contract.validate()?;
    if let Some(path) = args.manifest {
        let manifest: janus::coherence::ProtocolManifest = load(path)?;
        manifest.validate()?;
        if manifest.contract_id != contract.contract_id {
            return Err("protocol manifest contract_id does not match contract".into());
        }
    }
    if let Some(path) = args.comparison {
        let comparison: janus::coherence::CandidateComparisonManifest = load(path)?;
        comparison.validate()?;
        if comparison.contract_id != contract.contract_id {
            return Err("candidate comparison contract_id does not match contract".into());
        }
    }

    let survival_comparison = match (args.survival_real, args.survival_synthetic) {
        (Some(real_path), Some(synth_path)) => {
            let real_obs = load_survival_observations(real_path)?;
            let synth_obs = load_survival_observations(synth_path)?;
            Some(janus::compare_survival_curves(&real_obs, &synth_obs)?)
        }
        (None, None) => None,
        _ => {
            return Err("--survival-real and --survival-synthetic must be passed together".into());
        }
    };

    let neurosurgical_validation = match args.neurosurgical_cohort {
        Some(cohort_path) => {
            let records = load_neurosurgical_records(cohort_path)?;
            Some(janus::validate_neurosurgical_cohort(&records))
        }
        None => None,
    };

    if neurosurgical_validation.is_some() || survival_comparison.is_some() {
        let valid = neurosurgical_validation
            .as_ref()
            .is_none_or(janus::ValidationReport::is_valid);
        let mut report = serde_json::Map::new();
        report.insert("valid".into(), serde_json::Value::Bool(valid));
        report.insert(
            "contract_id".into(),
            serde_json::Value::String(contract.contract_id),
        );
        if let Some(nv) = neurosurgical_validation {
            report.insert(
                "neurosurgical_validation".into(),
                serde_json::to_value(nv)?,
            );
        }
        if let Some(sc) = survival_comparison {
            report.insert("survival_comparison".into(), serde_json::to_value(sc)?);
        }
        let json = serde_json::to_string_pretty(&serde_json::Value::Object(report))?;
        write_json_output(&json, args.output)?;
    } else {
        if args.output.is_some() {
            return Err(
                "--output requires --neurosurgical-cohort or --survival-real/--survival-synthetic"
                    .into(),
            );
        }
        println!("valid: coherence contract and supplied protocol manifests");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn failed_write_cleans_staging_and_allows_retry() {
        let dir = std::env::temp_dir().join(format!("janus-cli-test-{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        let output = dir.join("output");
        assert!(write_artifacts(&output, &[("missing/file", "x".into())]).is_err());
        assert!(!output.exists());
        assert!(!dir.join("output.janus-staging").exists());
        write_artifacts(&output, &[("ok", "x".into())]).unwrap();
        fs::remove_dir_all(dir).unwrap();
    }
    #[test]
    fn staging_names_keep_output_extensions() {
        let dir =
            std::env::temp_dir().join(format!("janus-stage-extension-{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        fs::create_dir(dir.join("run.a.janus-staging")).unwrap();
        assert!(write_artifacts(&dir.join("run.a"), &[("ok", "a".into())]).is_err());
        write_artifacts(&dir.join("run.b"), &[("ok", "b".into())]).unwrap();
        assert_eq!(fs::read_to_string(dir.join("run.b/ok")).unwrap(), "b");
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn forensic_eval_generates_aggregate_json_report() {
        let dir = std::env::temp_dir().join(format!("janus-forensic-test-{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        let real_path = dir.join("real.csv");
        let synth_path = dir.join("synth.csv");
        let out_path = dir.join("report.json");

        let real_csv = "age,score\n20,1.0\n30,2.0\n40,3.0\n50,4.0\n60,5.0\n70,6.0\n";
        let synth_csv = "age,score\n22,1.2\n32,2.1\n42,3.1\n52,4.2\n62,5.1\n72,6.2\n";
        fs::write(&real_path, real_csv).unwrap();
        fs::write(&synth_path, synth_csv).unwrap();

        run_forensic_eval(&real_path, &synth_path, 42, Some(&out_path)).unwrap();
        let json = fs::read_to_string(&out_path).unwrap();
        let report: janus::forensic::ForensicReport = serde_yaml::from_str(&json).unwrap();
        assert_eq!(report.seed, 42);
        assert_eq!(report.distributional_tests.len(), 2);
        assert_eq!(report.correlation_distance.n_variables, 2);
        assert!(report.timestamp.ends_with('Z'));
        assert!(report
            .dataset_fingerprint_real
            .starts_with("fnv1a64-schema:"));

        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn forensic_eval_rejects_incompatible_schemas_and_missing_files() {
        let dir = std::env::temp_dir().join(format!("janus-forensic-err-{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        let real_path = dir.join("real.csv");
        let synth_path = dir.join("synth.csv");
        let empty_path = dir.join("empty.csv");

        fs::write(&real_path, "age,score\n20,1.0\n30,2.0\n").unwrap();
        fs::write(&synth_path, "age,other\n22,1.2\n32,2.1\n").unwrap();
        fs::write(&empty_path, "age,score\n").unwrap();

        assert!(run_forensic_eval(&real_path, &synth_path, 42, None).is_err());
        assert!(run_forensic_eval(&dir.join("missing.csv"), &synth_path, 42, None).is_err());
        assert!(run_forensic_eval(&empty_path, &real_path, 42, None).is_err());

        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn forensic_eval_handles_missing_cells_without_zero_imputation() {
        let dir = std::env::temp_dir().join(format!(
            "janus-forensic-missing-test-{}",
            std::process::id()
        ));
        fs::create_dir_all(&dir).unwrap();
        let real_path = dir.join("real.csv");
        let synth_path = dir.join("synth.csv");
        let out_path = dir.join("report.json");

        // Real has observed 0.0 in `score`; synthetic has 1 observed 0.0 and 29 NA/blank rows.
        // Without 0.0 imputation, observed `score` matches (KS stat == 0.0), while the explicit
        // `is_missing` indicator makes the discriminator detect the missingness difference.
        let mut real_csv = String::from("age,score\n");
        let mut synth_csv = String::from("age,score\n");
        for i in 0..30 {
            let age = 30.0 + (i as f64);
            real_csv.push_str(&format!("{age},0.0\n"));
            if i == 0 {
                synth_csv.push_str(&format!("{age},0.0\n"));
            } else if i % 2 == 0 {
                synth_csv.push_str(&format!("{age},NA\n"));
            } else {
                synth_csv.push_str(&format!("{age},\n"));
            }
        }
        fs::write(&real_path, real_csv).unwrap();
        fs::write(&synth_path, synth_csv).unwrap();

        run_forensic_eval(&real_path, &synth_path, 42, Some(&out_path)).unwrap();
        let json = fs::read_to_string(&out_path).unwrap();
        let report: janus::forensic::ForensicReport = serde_json::from_str(&json).unwrap();

        let score_ks = report
            .distributional_tests
            .iter()
            .find(|t| t.variable_name == "score")
            .unwrap();
        assert_eq!(score_ks.statistic, 0.0);
        assert!(
            report.discriminator.auc_roc > 0.7,
            "expected discriminator AUC > 0.7 from explicit missingness indicator, got {}",
            report.discriminator.auc_roc
        );

        // Also verify KS test does not invent 0.0 for missing cells in a shifted column
        let real_path_2 = dir.join("real2.csv");
        let synth_path_2 = dir.join("synth2.csv");
        let out_path_2 = dir.join("report2.json");
        fs::write(
            &real_path_2,
            "age,score\n20,10.0\n30,20.0\n40,30.0\n50,40.0\n",
        )
        .unwrap();
        fs::write(
            &synth_path_2,
            "age,score\n20,NA\n30,20.0\n40,30.0\n50,40.0\n",
        )
        .unwrap();
        run_forensic_eval(&real_path_2, &synth_path_2, 42, Some(&out_path_2)).unwrap();
        let report2: janus::forensic::ForensicReport =
            serde_json::from_str(&fs::read_to_string(&out_path_2).unwrap()).unwrap();
        let score_ks2 = report2
            .distributional_tests
            .iter()
            .find(|t| t.variable_name == "score")
            .unwrap();
        // Observed values are [10,20,30,40] vs [20,30,40]: max CDF diff at 10 is 0.25 (not 0.0/0.25 from fake 0.0 at 0)
        // Wait: if 0.0 were imputed, CDF diff at 0..10 would be 0.25, and at 10 would be 0.25 - 0.25 = 0.0!
        // Specifically: observed [10,20,30,40] vs [20,30,40] has CDF1(10)=0.25, CDF2(10)=0.0, CDF1(20)=0.50, CDF2(20)=1/3 => diff = 0.25.
        // More decisively, compare [10, 20, 30] vs [NA, 10, 20, 30] -> observed distributions are IDENTICAL -> KS stat MUST be 0.0!
        assert!((score_ks2.statistic - 0.25).abs() < 1e-12);

        fs::write(
            &real_path_2,
            "age,score\n20,10.0\n30,20.0\n40,30.0\n",
        )
        .unwrap();
        fs::write(
            &synth_path_2,
            "age,score\n10,NA\n20,10.0\n30,20.0\n40,30.0\n",
        )
        .unwrap();
        run_forensic_eval(&real_path_2, &synth_path_2, 42, Some(&out_path_2)).unwrap();
        let report3: janus::forensic::ForensicReport =
            serde_json::from_str(&fs::read_to_string(&out_path_2).unwrap()).unwrap();
        let score_ks3 = report3
            .distributional_tests
            .iter()
            .find(|t| t.variable_name == "score")
            .unwrap();
        assert_eq!(score_ks3.statistic, 0.0);

        fs::remove_dir_all(dir).unwrap();
    }
}
