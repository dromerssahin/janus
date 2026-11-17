//! In-memory external cohort evaluation. Input and output may be sensitive.
use janus::tabular::{
    evaluate_flat_external, FlatFittedBinnedForest, FlatSparsePolicy, FlatUtilityTask, FlatVariable,
};
use serde::{Deserialize, Serialize};
use std::io::{self, Read};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Input {
    variables: Vec<FlatVariable>,
    task: FlatUtilityTask,
    training: Vec<Vec<String>>,
    reference: Vec<Vec<String>>,
    seeds: Vec<u64>,
    synthetic_training_size: Option<usize>,
    real_training_sets: Option<Vec<Vec<Vec<String>>>>,
}

#[derive(Serialize)]
struct Output {
    model: String,
    seed: u64,
    full: janus::tabular::FlatExternalEvaluation,
    sparse_backoffs: Vec<janus::tabular::FlatBackoffReport>,
    matched_real: Vec<janus::tabular::FlatUtilityScore>,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut input = String::new();
    io::stdin().read_to_string(&mut input)?;
    let input: Input = serde_yaml::from_str(&input)?;
    let mut reports = Vec::new();
    for model in ["greedy", "interaction"] {
        let fitted = if model == "greedy" {
            FlatFittedBinnedForest::fit_dag(&input.training, &input.variables, "pooled")?
        } else {
            FlatFittedBinnedForest::fit_interaction_dag(
                &input.training,
                &input.variables,
                "pooled",
            )?
        };
        for &seed in &input.seeds {
            let sample = fitted.sample_with_policy(
                seed,
                input
                    .synthetic_training_size
                    .unwrap_or(input.reference.len()),
                FlatSparsePolicy::DropLastParent,
            )?;
            let evaluation = evaluate_flat_external(
                &input.training,
                &sample.rows,
                &input.reference,
                &input.variables,
                &input.task,
            )?;
            let matched_real = input
                .real_training_sets
                .as_ref()
                .map(|sets| {
                    sets.iter()
                        .map(|rows| {
                            Ok::<_, Box<dyn std::error::Error>>(
                                evaluate_flat_external(
                                    rows,
                                    &sample.rows,
                                    &input.reference,
                                    &input.variables,
                                    &input.task,
                                )?
                                .utility
                                .real_training,
                            )
                        })
                        .collect::<Result<Vec<_>, _>>()
                })
                .transpose()?
                .unwrap_or_default();
            reports.push(Output {
                model: model.into(),
                seed,
                full: evaluation,
                sparse_backoffs: sample.sparse_backoffs,
                matched_real,
            });
        }
    }
    print!("{}", serde_yaml::to_string(&reports)?);
    Ok(())
}
