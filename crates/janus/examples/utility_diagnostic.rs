//! Aggregate-only diagnostic; all input, sampled rows and predictions remain in memory.
use janus::tabular::{
    evaluate_flat_external, FlatFittedBinnedForest, FlatSparsePolicy, FlatUtilityTask, FlatVariable,
};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::io::{self, Read};

#[derive(Deserialize)]
struct Input {
    variables: Vec<FlatVariable>,
    task: FlatUtilityTask,
    training: Vec<Vec<String>>,
    reference: Vec<Vec<String>>,
    seeds: Vec<u64>,
}
#[derive(Serialize)]
struct Counts {
    total: usize,
    labeled: usize,
    positive: usize,
    missing: usize,
}
#[derive(Serialize)]
struct State {
    training_real: Counts,
    training_synthetic: Counts,
    validation: Counts,
    observed_positive_fraction: Option<f64>,
    real_probability: Option<f64>,
    synthetic_probability: Option<f64>,
    real_fallback: bool,
    synthetic_fallback: bool,
}
#[derive(Serialize)]
struct Report {
    model: String,
    seed: u64,
    evaluation: janus::tabular::FlatExternalEvaluation,
    sparse_backoffs: Vec<janus::tabular::FlatBackoffReport>,
    states: BTreeMap<String, State>,
    outcome_counts: BTreeMap<String, BTreeMap<String, usize>>,
}
fn outcome_counts(rows: &[Vec<String>]) -> BTreeMap<String, usize> {
    ["0", "1", "2", "3", "4", "5", "6", "NA"]
        .iter()
        .map(|state| {
            (
                (*state).to_string(),
                rows.iter().filter(|r| r[1] == *state).count(),
            )
        })
        .collect()
}
fn label(value: &str) -> Option<bool> {
    match value {
        "0" | "1" | "2" => Some(false),
        "3" | "4" | "5" | "6" => Some(true),
        _ => None,
    }
}
fn counts(rows: &[Vec<String>], state: Option<&str>) -> Counts {
    let mut c = Counts {
        total: 0,
        labeled: 0,
        positive: 0,
        missing: 0,
    };
    for row in rows.iter().filter(|row| state.is_none_or(|s| row[0] == s)) {
        c.total += 1;
        match label(&row[1]) {
            Some(y) => {
                c.labeled += 1;
                c.positive += usize::from(y);
            }
            None => c.missing += 1,
        }
    }
    c
}
fn probability(rows: &[Vec<String>], state: &str) -> (Option<f64>, bool) {
    let specific = counts(rows, Some(state));
    let fallback = specific.labeled == 0;
    let c = if fallback {
        counts(rows, None)
    } else {
        specific
    };
    (
        (c.labeled > 0).then(|| (c.positive as f64 + 1.0) / (c.labeled as f64 + 2.0)),
        fallback,
    )
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut input = String::new();
    io::stdin().read_to_string(&mut input)?;
    let input: Input = serde_yaml::from_str(&input)?;
    if input.variables.len() != 2
        || input.variables[0].column != "wfns"
        || input.variables[1].column != "mrs"
        || input.task.predictor != "wfns"
        || input.task.target != "mrs"
        || !input.task.additional_predictors.is_empty()
        || input.task.alpha != 1.0
        || input.task.negative_values != ["0", "1", "2"]
        || input.task.positive_values != ["3", "4", "5", "6"]
    {
        return Err("utility diagnostic requires the declared WFNS/mRS task".into());
    }
    let mut reports = Vec::new();
    for model in ["greedy", "interaction"] {
        let fit = if model == "greedy" {
            FlatFittedBinnedForest::fit_dag(&input.training, &input.variables, "pooled")?
        } else {
            FlatFittedBinnedForest::fit_interaction_dag(
                &input.training,
                &input.variables,
                "pooled",
            )?
        };
        for &seed in &input.seeds {
            let sample = fit.sample_with_policy(
                seed,
                input.training.len(),
                FlatSparsePolicy::DropLastParent,
            )?;
            let evaluation = evaluate_flat_external(
                &input.training,
                &sample.rows,
                &input.reference,
                &input.variables,
                &input.task,
            )?;
            let mut states = BTreeMap::new();
            for state in ["1", "2", "3", "4", "5", "NA"] {
                let v = counts(&input.reference, Some(state));
                let (rp, rf) = probability(&input.training, state);
                let (sp, sf) = probability(&sample.rows, state);
                let fraction = (v.labeled > 0).then(|| v.positive as f64 / v.labeled as f64);
                states.insert(
                    state.to_string(),
                    State {
                        training_real: counts(&input.training, Some(state)),
                        training_synthetic: counts(&sample.rows, Some(state)),
                        validation: v,
                        observed_positive_fraction: fraction,
                        real_probability: rp,
                        synthetic_probability: sp,
                        real_fallback: rf,
                        synthetic_fallback: sf,
                    },
                );
            }
            reports.push(Report {
                model: model.into(),
                seed,
                evaluation,
                sparse_backoffs: sample.sparse_backoffs,
                states,
                outcome_counts: BTreeMap::from([
                    ("training_real".into(), outcome_counts(&input.training)),
                    ("training_synthetic".into(), outcome_counts(&sample.rows)),
                    ("validation".into(), outcome_counts(&input.reference)),
                ]),
            });
        }
    }
    print!("{}", serde_yaml::to_string(&reports)?);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    fn rows(values: &[(&str, &str)]) -> Vec<Vec<String>> {
        values
            .iter()
            .map(|(a, b)| vec![(*a).into(), (*b).into()])
            .collect()
    }
    #[test]
    fn calibration_uses_labeled_states_and_overall_fallback() {
        let data = rows(&[("1", "0"), ("1", "4"), ("2", "NA"), ("NA", "6")]);
        let one = counts(&data, Some("1"));
        assert_eq!(
            (one.total, one.labeled, one.positive, one.missing),
            (2, 2, 1, 0)
        );
        assert_eq!(probability(&data, "1"), (Some(0.5), false));
        assert_eq!(probability(&data, "2"), (Some(0.6), true));
        assert_eq!(probability(&data, "3"), (Some(0.6), true));
        assert_eq!(probability(&rows(&[("1", "NA")]), "1"), (None, true));
    }
}
