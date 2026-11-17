//! Transient stdin/stdout bridge for aggregate-only experiments. Never logs rows.
use janus_neural::{Column, ConditionSampling, Config, Model, Value};
use std::io::{self, Read};
fn main() {
    if let Err(e) = run() {
        eprintln!("comparison failed: {e}");
        std::process::exit(1);
    }
}
fn run() -> Result<(), String> {
    let sampling = match std::env::args().nth(1).as_deref() {
        None | Some("log") => ConditionSampling::LogFrequency,
        Some("empirical") => ConditionSampling::EmpiricalFrequency,
        Some(_) => return Err("usage: comparison [log|empirical]".into()),
    };
    let mut input = String::new();
    io::stdin()
        .read_to_string(&mut input)
        .map_err(|e| e.to_string())?;
    let mut lines = input.lines();
    let kinds: Vec<_> = lines.next().ok_or("missing schema")?.split('\t').collect();
    let raw: Vec<Vec<_>> = lines.map(|l| l.split('\t').collect()).collect();
    if raw.is_empty() || raw.iter().any(|r| r.len() != kinds.len()) {
        return Err("invalid input dimensions".into());
    }
    let mut schema = Vec::new();
    for (i, k) in kinds.iter().enumerate() {
        schema.push(match *k {
            "N" => Column::Numeric,
            "C" => {
                let categories: Vec<String> = raw
                    .iter()
                    .filter_map(|r| {
                        if r[i] == "NA" {
                            None
                        } else {
                            Some(r[i].to_owned())
                        }
                    })
                    .collect::<std::collections::BTreeSet<_>>()
                    .into_iter()
                    .collect();
                if categories.is_empty() {
                    return Err("categorical field has no observed state".into());
                }
                Column::Categorical { categories }
            }
            _ => return Err("invalid column type".into()),
        });
    }
    let mut rows = Vec::new();
    for r in raw {
        let mut row = Vec::new();
        for (s, k) in r.iter().zip(&kinds) {
            row.push(if *s == "NA" {
                Value::Missing
            } else if *k == "N" {
                Value::Number(s.parse().map_err(|_| "invalid numeric")?)
            } else {
                Value::Category((*s).to_string())
            });
        }
        rows.push(row);
    }
    tch::set_num_threads(1);
    tch::set_num_interop_threads(1);
    for fit_seed in [11, 22, 33, 44, 55] {
        let model = Model::fit(
            &rows,
            &schema,
            Config {
                condition_sampling: sampling,
                ..Config::default()
            },
            fit_seed,
        )?;
        for sample_seed in [101, 202] {
            println!("draw\t{fit_seed}\t{sample_seed}");
            for row in model.sample(rows.len(), sample_seed)? {
                let values: Vec<_> = row
                    .into_iter()
                    .map(|v| match v {
                        Value::Missing => "NA".to_string(),
                        Value::Number(x) => x.to_string(),
                        Value::Category(s) => s,
                        Value::StructuralAbsence => unreachable!(),
                    })
                    .collect();
                println!("{}", values.join("\t"));
            }
        }
    }
    Ok(())
}
