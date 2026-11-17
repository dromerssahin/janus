# Experimental conditional neural generator

This standalone Rust crate is optional and is outside the default workspace. It
requires `tch` 0.20.0 and compatible LibTorch 2.7.0; do not bypass the version
check. `Model::fit` accepts rows, an explicit schema, configuration and fit
seed; `sample` accepts a count and a separate seed. Values, preprocessing and
weights remain in memory. Structural absence is rejected. Numeric values are
bounded by the training range, constants remain constant, and unobserved
categorical and missing states are masked. The encoder does not distinguish
integer or ordinal numeric values, so generated numeric scores can be
fractional. Specify such domains as categorical if appropriate; the model does
not infer clinical rules.

The implementation is a conditional WGAN with a gradient penalty and a
conditional cross-entropy term. One discrete column and observed state is
sampled with log-frequency weighting for each example, and a matching real row
is selected. `ConditionSampling::EmpiricalFrequency` is available for bounded
diagnostics of unconditional marginal fidelity; the selected policy is retained
by the fitted model and used during sampling. Missingness is represented as a
discrete state for every column.
Categorical probabilities use softmax during training and argmax at sampling.
It does not implement CTGAN's mode-specific normalization, PacGAN critic,
or an established clinical data model. Training can collapse or fail to learn
dependencies; it is an experimental candidate, not a validated generator.

LibTorch's global random stream is locked across calls within this crate, but
other code using LibTorch must coordinate separately. Replay is demonstrated
only for the tested CPU environment and thread settings. The comparison binary
uses stdin and stdout for transient rows and is only appropriate when both ends
obey the source's data-handling policy. Never persist its streams for restricted
cohorts.

For the locally tested CPU installation, the build and fictional tests use:

```sh
env CARGO_HOME=/tmp/janus-tch-cargo LIBTORCH_USE_PYTORCH=1 PYTHONPATH=/tmp/janus-torch-270 LD_LIBRARY_PATH=/tmp/janus-torch-270/torch/lib PATH=/home/omer/.local/share/uv/python/cpython-3.13-linux-x86_64-gnu/bin:/usr/bin:/bin cargo test --offline --manifest-path experimental/neural/Cargo.toml --target-dir /tmp/janus-tch-target -- --test-threads=1
```

These temporary dependencies are not durable. Install the pinned versions in
your own environment for replay. The default workspace does not require them.
