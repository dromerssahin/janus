#!/usr/bin/env bash
set -euo pipefail
out=${1:?output directory required}
test ! -e "$out"
cargo run --offline -q -p janus-cli -- validate examples/study.yaml examples/fixture.yaml
cargo run --offline -q -p janus-cli -- split examples/study.yaml examples/fixture.yaml examples/split-for-fit.yaml "$out/split"
cargo run --offline -q -p janus-cli -- fit examples/study.yaml "$out/split/development.yaml" examples/fit.yaml "$out/fit" --split-manifest "$out/split/manifest.yaml"
cargo run --offline -q -p janus-cli -- generate examples/study.yaml "$out/fit/generator.yaml" "$out/synthetic"
cargo run --offline -q -p janus-cli -- evaluate examples/study.yaml "$out/synthetic/fixture.yaml" "$out/split/holdout.yaml" examples/evaluation.yaml "$out/evaluation" --split-manifest "$out/split/manifest.yaml" --synthetic-manifest "$out/synthetic/manifest.yaml"
cargo run --offline -q -p janus-cli -- evaluate examples/study.yaml "$out/synthetic/fixture.yaml" examples/reference.yaml examples/evaluation.yaml "$out/reference-evaluation" --synthetic-manifest "$out/synthetic/manifest.yaml"
cp examples/benchmark.yaml "$out/benchmark.yaml"
printf 'benchmark: concrete-descriptive-study\nartifacts: [split, fit, synthetic, evaluation]\n' > "$out/manifest.yaml"
