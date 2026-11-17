use janus::tabular::{FlatFittedBinnedForest, FlatFittedStratifiedForest, FlatVariable};
use std::collections::BTreeMap;

#[test]
fn fitted_forest_is_reusable_and_generates_in_memory() {
    let variables: Vec<FlatVariable> = serde_yaml::from_str(
        "- {column: category, kind: categorical, allowed_values: [a, b]}\n- {column: value, kind: numeric}\n",
    )
    .unwrap();
    // Entirely fictional records, with no identifiers.
    let training: Vec<Vec<String>> = (0..200)
        .map(|i| {
            if i % 2 == 0 {
                vec!["a".into(), "1".into()]
            } else {
                vec!["b".into(), "10".into()]
            }
        })
        .collect();
    let model = FlatFittedBinnedForest::fit(&training, &variables, "fictional", true).unwrap();
    assert_eq!(model.variables()[1].column, "value");
    assert_eq!(model.learned_dependencies().len(), 1);
    let first = model.sample(41, 100).unwrap();
    assert_eq!(first, model.sample(41, 100).unwrap());
    assert_ne!(first, model.sample(42, 100).unwrap());
    assert!(first
        .iter()
        .all(|r| (r[0] == "a" && r[1] == "1") || (r[0] == "b" && r[1] == "10")));
    assert!(model.sample(41, 0).unwrap().is_empty());
    assert!(model.sample(41, 1_000_001).is_err());
}

#[test]
fn fitted_forest_rejects_invalid_training_inputs() {
    let variables: Vec<FlatVariable> = serde_yaml::from_str(
        "- {column: category, kind: categorical, allowed_values: [a, b]}\n- {column: value, kind: numeric}\n",
    )
    .unwrap();
    assert!(FlatFittedBinnedForest::fit(&[], &variables, "g", false).is_err());
    assert!(FlatFittedBinnedForest::fit(&[vec!["a".into()]], &variables, "g", false).is_err());
    assert!(
        FlatFittedBinnedForest::fit(&[vec!["a".into(), "NaN".into()]], &variables, "g", false)
            .is_err()
    );
    assert!(
        FlatFittedBinnedForest::fit(&[vec!["c".into(), "2".into()]], &variables, "g", false)
            .is_err()
    );
    assert!(
        FlatFittedBinnedForest::fit(&[vec![" a".into(), "2".into()]], &variables, "g", false)
            .is_err()
    );
}

#[test]
fn fitted_forest_can_interpolate_continuous_columns_in_memory() {
    let variables: Vec<FlatVariable> = serde_yaml::from_str(
        "- {column: category, kind: categorical, allowed_values: [a, b]}\n- {column: value, kind: numeric}\n",
    ).unwrap();
    let training: Vec<Vec<String>> = (0..200)
        .map(|i| {
            let offset = (i / 2) % 8;
            if i % 2 == 0 {
                vec!["a".into(), (1 + offset).to_string()]
            } else {
                vec!["b".into(), (51 + offset).to_string()]
            }
        })
        .collect();
    let model = FlatFittedBinnedForest::fit(&training, &variables, "fictional", true).unwrap();
    let columns = vec!["value".to_owned()];
    let sample = model.sample_interpolated(41, 100, &columns).unwrap();
    let replay = model.sample_interpolated(41, 100, &columns).unwrap();
    assert_eq!(sample.rows, replay.rows);
    assert!(sample.interpolation[0].changed > 0);
    assert!(sample
        .rows
        .iter()
        .any(|r| r[1].parse::<f64>().unwrap().fract() != 0.0));
    assert!(sample.rows.iter().all(|r| {
        let value: f64 = r[1].parse().unwrap();
        (r[0] == "a" && (1.0..=8.0).contains(&value))
            || (r[0] == "b" && (51.0..=58.0).contains(&value))
    }));
    assert!(model
        .sample_interpolated(41, 1, &["category".into()])
        .is_err());
    assert!(model
        .sample_interpolated(41, 1, &["value".into(), "value".into()])
        .is_err());
}

#[test]
fn stratified_forest_fits_and_samples_requested_groups() {
    let variables: Vec<FlatVariable> = serde_yaml::from_str(
        "- {column: category, kind: categorical, allowed_values: [a, b]}\n- {column: value, kind: numeric}\n",
    ).unwrap();
    let training: BTreeMap<String, Vec<Vec<String>>> = [
        (
            "north".into(),
            (0..80)
                .map(|i| vec!["a".into(), (1 + i % 8).to_string()])
                .collect(),
        ),
        (
            "south".into(),
            (0..80)
                .map(|i| vec!["b".into(), (51 + i % 8).to_string()])
                .collect(),
        ),
    ]
    .into_iter()
    .collect();
    let model = FlatFittedStratifiedForest::fit(&training, &variables, true).unwrap();
    assert!(model.group("north").is_some());
    assert!(model.group("unknown").is_none());
    let allocation = BTreeMap::from([("north".into(), 30), ("south".into(), 20)]);
    let sample = model.sample(41, &allocation).unwrap();
    assert_eq!(sample, model.sample(41, &allocation).unwrap());
    assert_eq!(sample["north"].len(), 30);
    assert_eq!(sample["south"].len(), 20);
    assert!(sample["north"].iter().all(|r| r[0] == "a"));
    assert!(sample["south"].iter().all(|r| r[0] == "b"));
    let changed_allocation = BTreeMap::from([("north".into(), 4), ("south".into(), 20)]);
    assert_eq!(
        sample["south"],
        model.sample(41, &changed_allocation).unwrap()["south"]
    );
    let interpolated = model
        .sample_interpolated(41, &allocation, &["value".into()])
        .unwrap();
    let policy_sample = model
        .sample_interpolated_with_policy(
            41,
            &allocation,
            &["value".into()],
            janus::tabular::FlatSparsePolicy::DropLastParent,
        )
        .unwrap();
    assert_eq!(policy_sample["north"].rows, interpolated["north"].rows);
    assert_eq!(policy_sample["south"].rows, interpolated["south"].rows);
    assert!(policy_sample
        .values()
        .all(|group| group.sparse_backoffs.is_empty()));
    assert!(interpolated["north"].interpolation[0].changed > 0);
    assert!(interpolated["south"].interpolation[0].changed > 0);
    assert!(interpolated["north"]
        .rows
        .iter()
        .all(|r| (1.0..=8.0).contains(&r[1].parse::<f64>().unwrap())));
    assert!(interpolated["south"]
        .rows
        .iter()
        .all(|r| (51.0..=58.0).contains(&r[1].parse::<f64>().unwrap())));
    assert!(model
        .sample(41, &BTreeMap::from([("north".into(), 1)]))
        .is_err());
    assert!(model
        .sample(
            41,
            &BTreeMap::from([("north".into(), 1), ("other".into(), 1)])
        )
        .is_err());
    assert!(model
        .sample(
            41,
            &BTreeMap::from([("north".into(), usize::MAX), ("south".into(), 1)])
        )
        .is_err());
    let zero = BTreeMap::from([("north".into(), 0), ("south".into(), 0)]);
    assert!(model.sample(41, &zero).unwrap().values().all(Vec::is_empty));
    assert!(FlatFittedStratifiedForest::fit(&BTreeMap::new(), &variables, false).is_err());
    let empty = BTreeMap::from([("north".into(), Vec::new())]);
    assert!(FlatFittedStratifiedForest::fit(&empty, &variables, false).is_err());
}

#[test]
fn fitted_dag_learns_and_samples_multiple_parents() {
    use janus::tabular::FlatSparsePolicy;
    let variables: Vec<FlatVariable> = serde_yaml::from_str(
        "- {column: a, kind: categorical, allowed_values: ['0', '1']}\n- {column: b, kind: categorical, allowed_values: ['0', '1']}\n- {column: outcome, kind: categorical, allowed_values: ['0', '1']}\n",
    ).unwrap();
    let mut training = Vec::new();
    for _ in 0..100 {
        for a in 0..2 {
            for b in 0..2 {
                training.push(vec![a.to_string(), b.to_string(), (a | b).to_string()]);
            }
        }
    }
    let model = FlatFittedBinnedForest::fit_dag(&training, &variables, "fictional").unwrap();
    assert_eq!(
        model
            .learned_dependencies()
            .iter()
            .filter(|edge| edge.child == "outcome")
            .count(),
        2
    );
    let sample = model
        .sample_with_policy(41, 100, FlatSparsePolicy::DropLastParent)
        .unwrap();
    assert_eq!(
        sample.rows,
        model
            .sample_with_policy(41, 100, FlatSparsePolicy::DropLastParent)
            .unwrap()
            .rows
    );
    assert!(sample.sparse_backoffs.is_empty());
    assert!(sample
        .rows
        .iter()
        .all(|r| r[2] == if r[0] == "1" || r[1] == "1" { "1" } else { "0" }));
}

#[test]
fn interaction_model_samples_a_pure_two_parent_pattern() {
    let variables: Vec<FlatVariable> = serde_yaml::from_str(
        "- {column: a, kind: categorical, allowed_values: ['0', '1']}\n- {column: b, kind: categorical, allowed_values: ['0', '1']}\n- {column: outcome, kind: categorical, allowed_values: ['0', '1']}\n",
    ).unwrap();
    let mut training = Vec::new();
    for _ in 0..100 {
        for a in 0..2 {
            for b in 0..2 {
                training.push(vec![a.to_string(), b.to_string(), (a ^ b).to_string()]);
            }
        }
    }
    let model =
        FlatFittedBinnedForest::fit_interaction_dag(&training, &variables, "fictional").unwrap();
    assert_eq!(
        model
            .learned_dependencies()
            .iter()
            .filter(|e| e.child == "outcome" && e.joint_selection_gain_nats.is_some())
            .count(),
        2
    );
    let sample = model.sample(41, 100).unwrap();
    assert_eq!(sample, model.sample(41, 100).unwrap());
    assert!(sample
        .iter()
        .all(|r| r[2] == if r[0] == r[1] { "0" } else { "1" }));
}

#[test]
fn stratified_interaction_models_sample_groups_independently() {
    use janus::tabular::FlatSparsePolicy;
    let variables: Vec<FlatVariable> = serde_yaml::from_str(
        "- {column: a, kind: categorical, allowed_values: ['0', '1']}\n- {column: b, kind: categorical, allowed_values: ['0', '1']}\n- {column: outcome, kind: categorical, allowed_values: ['0', '1']}\n",
    ).unwrap();
    let mut xor_rows = Vec::new();
    let mut or_rows = Vec::new();
    for _ in 0..100 {
        for a in 0..2 {
            for b in 0..2 {
                xor_rows.push(vec![a.to_string(), b.to_string(), (a ^ b).to_string()]);
                or_rows.push(vec![a.to_string(), b.to_string(), (a | b).to_string()]);
            }
        }
    }
    let training = BTreeMap::from([("xor".into(), xor_rows), ("or".into(), or_rows)]);
    let model = FlatFittedStratifiedForest::fit_interaction_dag(&training, &variables).unwrap();
    assert!(model
        .group("xor")
        .unwrap()
        .learned_dependencies()
        .iter()
        .any(|edge| edge.joint_selection_gain_nats.is_some()));
    let allocation = BTreeMap::from([("xor".into(), 80), ("or".into(), 70)]);
    let sample = model
        .sample_with_policy(42, &allocation, FlatSparsePolicy::DropLastParent)
        .unwrap();
    assert_eq!(sample["xor"].rows.len(), 80);
    assert_eq!(sample["or"].rows.len(), 70);
    assert!(sample
        .values()
        .all(|group| group.sparse_backoffs.is_empty()));
    assert!(sample["xor"]
        .rows
        .iter()
        .all(|r| r[2] == if r[0] == r[1] { "0" } else { "1" }));
    assert!(sample["or"]
        .rows
        .iter()
        .all(|r| r[2] == if r[0] == "1" || r[1] == "1" { "1" } else { "0" }));
    let changed = model
        .sample_with_policy(
            42,
            &BTreeMap::from([("xor".into(), 0), ("or".into(), 70)]),
            FlatSparsePolicy::DropLastParent,
        )
        .unwrap();
    assert!(changed["xor"].rows.is_empty());
    assert_eq!(sample["or"].rows, changed["or"].rows);
    assert!(model
        .sample_with_policy(
            42,
            &BTreeMap::from([("xor".into(), 1)]),
            FlatSparsePolicy::Fail
        )
        .is_err());
    assert!(model
        .sample_with_policy(
            42,
            &BTreeMap::from([("xor".into(), 1_000_000), ("or".into(), 1)]),
            FlatSparsePolicy::Fail
        )
        .is_err());
    let greedy = FlatFittedStratifiedForest::fit_dag(&training, &variables).unwrap();
    assert!(!greedy
        .group("xor")
        .unwrap()
        .learned_dependencies()
        .iter()
        .any(|edge| edge.child == "outcome"));
    assert!(FlatFittedStratifiedForest::fit_interaction_dag(&BTreeMap::new(), &variables).is_err());
}
