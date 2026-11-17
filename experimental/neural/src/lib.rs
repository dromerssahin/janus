//! Experimental, CPU-only conditional WGAN for mixed tabular data.
//!
//! Numerical columns are scaled to [-1, 1] and carry a separate observed /
//! missing indicator. Categorical columns include an explicit missing state.
//! The critic receives a condition identifying one discrete state; training
//! draws states proportional to log(1 + count), and draws matching real rows.
//! Generator loss combines the Wasserstein loss and conditional cross entropy;
//! the critic uses a gradient penalty on interpolated rows. Categorical outputs
//! use softmax during training and argmax when decoded. This is CTGAN-style,
//! not a reproduction of CTGAN's mode-specific normalization or architecture.
//! All preprocessing and weights remain in memory. Global LibTorch RNG access
//! is serialized within this crate; other users of LibTorch must coordinate.

use std::sync::{Mutex, OnceLock};
use tch::{nn, nn::OptimizerConfig, Device, Kind, Tensor};

#[derive(Clone, Debug, PartialEq)]
pub enum Value {
    Number(f64),
    Category(String),
    Missing,
    StructuralAbsence,
}
#[derive(Clone, Debug)]
pub enum Column {
    Numeric,
    Categorical { categories: Vec<String> },
}
#[derive(Clone, Debug)]
pub struct Config {
    pub steps: usize,
    pub batch_size: usize,
    pub hidden: i64,
    pub noise: i64,
    pub learning_rate: f64,
    pub gradient_penalty: f64,
    pub condition_sampling: ConditionSampling,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ConditionSampling {
    LogFrequency,
    EmpiricalFrequency,
}
impl Default for Config {
    fn default() -> Self {
        Self {
            steps: 300,
            batch_size: 64,
            hidden: 64,
            noise: 32,
            learning_rate: 0.0002,
            gradient_penalty: 10.0,
            condition_sampling: ConditionSampling::LogFrequency,
        }
    }
}
#[derive(Clone, Debug)]
struct Span {
    start: i64,
    len: i64,
    condition: i64,
    rows: Vec<Vec<i64>>,
}
#[derive(Clone, Debug)]
struct Encoding {
    numeric: bool,
    min: f64,
    max: f64,
    categories: Vec<String>,
    value_offset: i64,
    span: Span,
}
pub struct Model {
    _weights: nn::VarStore,
    generator: nn::Sequential,
    encoding: Vec<Encoding>,
    spans: Vec<Span>,
    width: i64,
    conditions: i64,
    noise: i64,
    condition_sampling: ConditionSampling,
    pub generator_change: f64,
    pub critic_change: f64,
    pub last_losses: (f64, f64),
}
fn lock() -> std::sync::MutexGuard<'static, ()> {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| Mutex::new(()))
        .lock()
        .unwrap_or_else(|e| e.into_inner())
}
fn bad(s: &str) -> String {
    s.to_owned()
}

impl Model {
    pub fn fit(
        rows: &[Vec<Value>],
        schema: &[Column],
        config: Config,
        seed: i64,
    ) -> Result<Self, String> {
        if rows.is_empty() || schema.is_empty() || rows.iter().any(|r| r.len() != schema.len()) {
            return Err(bad("empty data/schema or inconsistent row width"));
        }
        if config.steps == 0
            || config.batch_size == 0
            || config.hidden <= 0
            || config.noise <= 0
            || !config.learning_rate.is_finite()
            || config.learning_rate <= 0.0
            || !config.gradient_penalty.is_finite()
            || config.gradient_penalty <= 0.0
        {
            return Err(bad("invalid training configuration"));
        }
        let mut encoding = Vec::new();
        let mut width = 0;
        let mut conditions = 0;
        for (col, ty) in schema.iter().enumerate() {
            let (numeric, categories) = match ty {
                Column::Numeric => (true, vec![]),
                Column::Categorical { categories } => {
                    let unique: std::collections::BTreeSet<_> = categories.iter().collect();
                    if categories.is_empty()
                        || unique.len() != categories.len()
                        || categories.iter().any(|s| s.is_empty())
                    {
                        return Err(bad("invalid categorical domain"));
                    }
                    (false, categories.clone())
                }
            };
            let mut min = f64::INFINITY;
            let mut max = f64::NEG_INFINITY;
            let len = if numeric {
                2
            } else {
                categories.len() as i64 + 1
            };
            let mut members = vec![Vec::new(); len as usize];
            for (i, row) in rows.iter().enumerate() {
                let state = match &row[col] {
                    Value::Missing => {
                        if numeric {
                            1
                        } else {
                            categories.len()
                        }
                    }
                    Value::Number(x) if numeric && x.is_finite() => {
                        min = min.min(*x);
                        max = max.max(*x);
                        0
                    }
                    Value::Category(s) if !numeric => categories
                        .iter()
                        .position(|x| x == s)
                        .ok_or_else(|| bad("category outside domain"))?,
                    _ => {
                        return Err(bad(
                            "invalid value, type, or unsupported structural absence",
                        ))
                    }
                };
                members[state].push(i as i64);
            }
            if numeric && !min.is_finite() {
                min = 0.0;
                max = 0.0;
            }
            let value_offset = width;
            if numeric {
                width += 1;
            }
            let span = Span {
                start: width,
                len,
                condition: conditions,
                rows: members,
            };
            width += len;
            conditions += len;
            encoding.push(Encoding {
                numeric,
                min,
                max,
                categories,
                value_offset,
                span,
            });
        }
        let mut data = vec![0f32; rows.len() * width as usize];
        for (i, row) in rows.iter().enumerate() {
            for (j, e) in encoding.iter().enumerate() {
                let base = i * width as usize;
                let state = match &row[j] {
                    Value::Missing => {
                        if e.numeric {
                            1
                        } else {
                            e.categories.len()
                        }
                    }
                    Value::Number(x) => {
                        let scaled = if e.max == e.min {
                            0.0
                        } else {
                            2.0 * ((x - e.min) / (e.max - e.min)) - 1.0
                        };
                        if !scaled.is_finite() {
                            return Err(bad("numeric range overflow"));
                        }
                        data[base + e.value_offset as usize] = scaled as f32;
                        0
                    }
                    Value::Category(s) => e.categories.iter().position(|v| v == s).unwrap(),
                    _ => unreachable!(),
                };
                data[base + e.span.start as usize + state] = 1.0;
            }
        }
        let _guard = lock();
        tch::manual_seed(seed);
        let real = Tensor::from_slice(&data).reshape([rows.len() as i64, width]);
        let spans: Vec<_> = encoding.iter().map(|e| e.span.clone()).collect();
        let gv = nn::VarStore::new(Device::Cpu);
        let dv = nn::VarStore::new(Device::Cpu);
        let generator = nn::seq()
            .add(nn::linear(
                &gv.root() / "a",
                config.noise + conditions,
                config.hidden,
                Default::default(),
            ))
            .add_fn(|x| x.relu())
            .add(nn::linear(
                &gv.root() / "b",
                config.hidden,
                width,
                Default::default(),
            ));
        let critic = nn::seq()
            .add(nn::linear(
                &dv.root() / "a",
                width + conditions,
                config.hidden,
                Default::default(),
            ))
            .add_fn(|x| x.leaky_relu())
            .add(nn::linear(
                &dv.root() / "b",
                config.hidden,
                1,
                Default::default(),
            ));
        let mut go = nn::Adam::default()
            .build(&gv, config.learning_rate)
            .map_err(|e| e.to_string())?;
        let mut dopt = nn::Adam::default()
            .build(&dv, config.learning_rate)
            .map_err(|e| e.to_string())?;
        let initial = gv.trainable_variables()[0].detach().copy();
        let critic_initial = dv.trainable_variables()[0].detach().copy();
        let mut losses = (0., 0.);
        let batch = config.batch_size as i64;
        for _ in 0..config.steps {
            let (cond, indices, targets) =
                conditions_batch(&spans, conditions, batch, config.condition_sampling);
            let r = real.index_select(0, &indices);
            let logits = Tensor::cat(
                &[
                    Tensor::randn([batch, config.noise], (Kind::Float, Device::Cpu)),
                    cond.shallow_clone(),
                ],
                1,
            )
            .apply(&generator);
            let fake = activate(&logits, &encoding).detach();
            let alpha = Tensor::rand([batch, 1], (Kind::Float, Device::Cpu));
            let mixed = (&alpha * &r + (1.0f64 - &alpha) * &fake).set_requires_grad(true);
            let score =
                Tensor::cat(&[mixed.shallow_clone(), cond.shallow_clone()], 1).apply(&critic);
            let grad =
                Tensor::run_backward(&[score.sum(Kind::Float)], &[&mixed], true, true).remove(0);
            let norm = (grad
                .square()
                .sum_dim_intlist(&[1i64][..], false, Kind::Float)
                + 1e-12)
                .sqrt();
            let dl = Tensor::cat(&[fake, cond.shallow_clone()], 1)
                .apply(&critic)
                .mean(Kind::Float)
                - Tensor::cat(&[r, cond.shallow_clone()], 1)
                    .apply(&critic)
                    .mean(Kind::Float)
                + (norm - 1.0).square().mean(Kind::Float) * config.gradient_penalty;
            losses.0 = f64::try_from(&dl).map_err(|e| e.to_string())?;
            if !losses.0.is_finite() {
                return Err(bad("nonfinite critic loss"));
            }
            dopt.backward_step(&dl);
            let logits = Tensor::cat(
                &[
                    Tensor::randn([batch, config.noise], (Kind::Float, Device::Cpu)),
                    cond.shallow_clone(),
                ],
                1,
            )
            .apply(&generator);
            let activated = activate(&logits, &encoding);
            let mut ce = Tensor::zeros([], (Kind::Float, Device::Cpu));
            for (k, span) in spans.iter().enumerate() {
                let mask = targets.narrow(1, k as i64, 1);
                let logp = logits
                    .narrow(1, span.start, span.len)
                    .log_softmax(-1, Kind::Float);
                ce -= (&logp * cond.narrow(1, span.condition, span.len) * mask).sum(Kind::Float)
                    / (batch as f64);
            }
            let gl = -Tensor::cat(&[activated, cond], 1)
                .apply(&critic)
                .mean(Kind::Float)
                + ce;
            losses.1 = f64::try_from(&gl).map_err(|e| e.to_string())?;
            if !losses.1.is_finite() {
                return Err(bad("nonfinite generator loss"));
            }
            go.backward_step(&gl);
        }
        let change = f64::try_from(
            (gv.trainable_variables()[0].detach() - initial)
                .abs()
                .sum(Kind::Float),
        )
        .map_err(|e| e.to_string())?;
        if !change.is_finite() || change == 0.0 {
            return Err(bad("generator weights did not update"));
        }
        let critic_change = f64::try_from(
            (dv.trainable_variables()[0].detach() - critic_initial)
                .abs()
                .sum(Kind::Float),
        )
        .map_err(|e| e.to_string())?;
        if !critic_change.is_finite() || critic_change == 0.0 {
            return Err(bad("critic weights did not update"));
        }
        Ok(Self {
            _weights: gv,
            generator,
            encoding,
            spans,
            width,
            conditions,
            noise: config.noise,
            condition_sampling: config.condition_sampling,
            generator_change: change,
            critic_change,
            last_losses: losses,
        })
    }
    pub fn sample(&self, count: usize, seed: i64) -> Result<Vec<Vec<Value>>, String> {
        if count == 0 {
            return Ok(vec![]);
        }
        if count > i64::MAX as usize {
            return Err(bad("sample count too large"));
        }
        let _guard = lock();
        tch::manual_seed(seed);
        let n = count as i64;
        let (cond, _, _) = conditions_batch(
            &self.spans,
            self.conditions,
            n,
            self.condition_sampling,
        );
        let output = tch::no_grad(|| {
            activate(
                &Tensor::cat(
                    &[
                        Tensor::randn([n, self.noise], (Kind::Float, Device::Cpu)),
                        cond,
                    ],
                    1,
                )
                .apply(&self.generator),
                &self.encoding,
            )
        });
        let values = Vec::<f32>::try_from(output.reshape([-1])).map_err(|e| e.to_string())?;
        if values.iter().any(|x| !x.is_finite()) {
            return Err(bad("nonfinite output"));
        }
        let mut result = Vec::with_capacity(count);
        for row in values.chunks(self.width as usize) {
            let mut out = Vec::new();
            for e in &self.encoding {
                let slice = &row[e.span.start as usize..(e.span.start + e.span.len) as usize];
                let state = slice
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.total_cmp(b.1))
                    .unwrap()
                    .0;
                out.push(if e.numeric {
                    if state == 1 {
                        Value::Missing
                    } else {
                        Value::Number(if e.max == e.min {
                            e.min
                        } else {
                            e.min
                                + (row[e.value_offset as usize] as f64 + 1.0)
                                    * 0.5
                                    * (e.max - e.min)
                        })
                    }
                } else if state == e.categories.len() {
                    Value::Missing
                } else {
                    Value::Category(e.categories[state].clone())
                });
            }
            result.push(out);
        }
        Ok(result)
    }
}
fn activate(logits: &Tensor, encoding: &[Encoding]) -> Tensor {
    let mut parts = Vec::new();
    for e in encoding {
        if e.numeric {
            parts.push(logits.narrow(1, e.value_offset, 1).tanh());
        }
        let mask: Vec<f32> = e
            .span
            .rows
            .iter()
            .map(|r| if r.is_empty() { -1e9 } else { 0. })
            .collect();
        parts.push(
            (logits.narrow(1, e.span.start, e.span.len) + Tensor::from_slice(&mask))
                .softmax(-1, Kind::Float),
        );
    }
    Tensor::cat(&parts, 1)
}
fn conditions_batch(
    spans: &[Span],
    width: i64,
    batch: i64,
    sampling: ConditionSampling,
) -> (Tensor, Tensor, Tensor) {
    let mut cond = vec![0f32; (batch * width) as usize];
    let mut masks = vec![0f32; batch as usize * spans.len()];
    let mut indices = Vec::with_capacity(batch as usize);
    for i in 0..batch {
        let k = i64::try_from(Tensor::randint(
            spans.len() as i64,
            [],
            (Kind::Int64, Device::Cpu),
        ))
        .unwrap() as usize;
        let s = &spans[k];
        let weights: Vec<f32> = s
            .rows
            .iter()
            .map(|r| match sampling {
                ConditionSampling::LogFrequency => (1.0 + r.len() as f32).ln(),
                ConditionSampling::EmpiricalFrequency => r.len() as f32,
            })
            .collect();
        let state =
            i64::try_from(Tensor::from_slice(&weights).multinomial(1, true)).unwrap() as usize;
        let candidates = &s.rows[state];
        let idx = i64::try_from(Tensor::randint(
            candidates.len() as i64,
            [],
            (Kind::Int64, Device::Cpu),
        ))
        .unwrap() as usize;
        indices.push(candidates[idx]);
        cond[(i * width + s.condition + state as i64) as usize] = 1.0;
        masks[i as usize * spans.len() + k] = 1.0;
    }
    (
        Tensor::from_slice(&cond).reshape([batch, width]),
        Tensor::from_slice(&indices),
        Tensor::from_slice(&masks).reshape([batch, spans.len() as i64]),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    fn fixture(n: usize) -> (Vec<Vec<Value>>, Vec<Column>) {
        let schema = vec![
            Column::Numeric,
            Column::Categorical {
                categories: vec!["common".into(), "rare".into()],
            },
            Column::Numeric,
        ];
        let rows = (0..n)
            .map(|i| {
                vec![
                    if i % 5 == 0 {
                        Value::Missing
                    } else {
                        Value::Number((i % 17) as f64)
                    },
                    Value::Category(if i % 31 == 0 { "rare" } else { "common" }.into()),
                    if i % 5 == 0 {
                        Value::Missing
                    } else {
                        Value::Number(4.)
                    },
                ]
            })
            .collect();
        (rows, schema)
    }
    #[test]
    fn fictional_sizes_replay_and_variation() {
        tch::set_num_threads(1);
        for n in [153, 229, 298] {
            let (rows, schema) = fixture(n);
            let cfg = Config {
                steps: 30,
                batch_size: 32,
                ..Default::default()
            };
            let a = Model::fit(&rows, &schema, cfg.clone(), 42).unwrap();
            let b = Model::fit(&rows, &schema, cfg.clone(), 42).unwrap();
            let c = Model::fit(&rows, &schema, cfg, 43).unwrap();
            assert!(
                a.generator_change > 0.
                    && a.critic_change > 0.
                    && a.last_losses.0.is_finite()
                    && a.last_losses.1.is_finite()
            );
            let x = a.sample(200, 77).unwrap();
            assert_eq!(x, b.sample(200, 77).unwrap());
            assert_ne!(x, a.sample(200, 78).unwrap());
            assert_ne!(x, c.sample(200, 77).unwrap());
            assert!(x
                .iter()
                .all(|r| matches!(r[2], Value::Missing | Value::Number(4.))));
            assert!(
                x.iter().any(|r| r[1] == Value::Category("rare".into())),
                "rare state absent for size {n}"
            );
        }
    }
    #[test]
    fn validation_and_missing_only() {
        let cfg = Config {
            steps: 2,
            batch_size: 4,
            ..Default::default()
        };
        assert!(Model::fit(&[], &[Column::Numeric], cfg.clone(), 1).is_err());
        assert!(Model::fit(
            &[vec![Value::StructuralAbsence]],
            &[Column::Numeric],
            cfg.clone(),
            1
        )
        .is_err());
        assert!(Model::fit(
            &[vec![Value::Number(f64::NAN)]],
            &[Column::Numeric],
            cfg.clone(),
            1
        )
        .is_err());
        let model = Model::fit(&[vec![Value::Missing]], &[Column::Numeric], cfg, 1).unwrap();
        assert!(model
            .sample(10, 2)
            .unwrap()
            .iter()
            .all(|r| r[0] == Value::Missing));
    }

    #[test]
    fn conditioning_policy_is_explicit_and_bounded() {
        let (rows, schema) = fixture(229);
        let model_log = Model::fit(
            &rows,
            &schema,
            Config {
                steps: 2,
                condition_sampling: ConditionSampling::LogFrequency,
                ..Default::default()
            },
            9,
        )
        .unwrap();
        let model_empirical = Model::fit(
            &rows,
            &schema,
            Config {
                steps: 2,
                condition_sampling: ConditionSampling::EmpiricalFrequency,
                ..Default::default()
            },
            9,
        )
        .unwrap();
        assert_eq!(model_log.condition_sampling, ConditionSampling::LogFrequency);
        assert_eq!(
            model_empirical.condition_sampling,
            ConditionSampling::EmpiricalFrequency
        );
        let log = model_log.sample(300, 123).unwrap();
        let empirical = model_empirical.sample(300, 123).unwrap();
        assert_eq!(log.len(), empirical.len());
        assert!(log.iter().all(|r| r.len() == schema.len()));
        assert!(empirical.iter().all(|r| r.len() == schema.len()));
    }
}
