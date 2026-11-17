#!/usr/bin/env bash
set -euo pipefail

source=${1:-datasets/ed.csv}
config=${2:-datasets/early-drain-descriptive.yaml}
output=${3:?provide a fresh output directory under datasets/}
replay="${output}.replay-$$"
cleanup() {
    rm -rf "$replay"
}
trap cleanup EXIT

cargo run --offline -q -p janus-cli -- describe-flat "$source" "$config" "$output"
cargo run --offline -q -p janus-cli -- describe-flat "$source" "$config" "$replay"
cmp "$output/report.yaml" "$replay/report.yaml"
cmp "$output/manifest.yaml" "$replay/manifest.yaml"

python3 - "$source" "$config" "$output/report.yaml" "$output/manifest.yaml" <<'PY'
import csv
import sys
import yaml

source_path, config_path, report_path, manifest_path = sys.argv[1:]
with open(report_path) as stream:
    report = yaml.safe_load(stream)
with open(config_path) as stream:
    config = yaml.safe_load(stream)
with open(manifest_path) as stream:
    manifest = yaml.safe_load(stream)

def fnv1a(path):
    value = 0xcbf29ce484222325
    with open(path, "rb") as stream:
        for byte in stream.read():
            value = ((value ^ byte) * 0x100000001b3) & 0xffffffffffffffff
    return f"fnv1a64:{value:016x}"

assert manifest["source_fingerprint"] == fnv1a(source_path)
assert manifest["config_fingerprint"].startswith("fnv1a64-yaml:")
assert manifest["report_fingerprint"].startswith("fnv1a64-yaml:")
with open(source_path, newline="") as stream:
    headers = next(csv.reader(stream))
assert set(headers) == set(config["expected_source_columns"])

assert report["total_patients"] == sum(group["patients"] for group in report["groups"])
assert report["split"]["development_patients"] + report["split"]["holdout_patients"] == report["total_patients"]
assert sum(group["patients"] for group in report["generated"]) == report["split"]["holdout_patients"]
assert len(report["generated_vs_holdout"]) == sum(len(group["variables"]) for group in report["generated"])
for group in report["groups"]:
    for variable in group["variables"]:
        assert variable["observed"] + variable["missing"] + variable["structurally_absent"] == group["patients"]
for partition in report["partitions"]:
    for group in partition["groups"]:
        for variable in group["variables"]:
            assert variable["observed"] + variable["missing"] + variable["structurally_absent"] == group["patients"]
print("aggregate report checks passed")
PY
echo "deterministic replay checks passed"
