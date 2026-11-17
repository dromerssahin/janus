# Neural Operations

Janus isolates experimental neural tabular generation (CTGAN/TVAE) behind a narrow CLI boundary (`crates/janus/src/neural.rs`) invoking `scripts/neural_bridge.py` via `uv run`. This keeps the Rust core free of Python and LibTorch runtime dependencies.

## GPU Setup and Version Pinning

### PyTorch Version Pinning

- **Validated**: PyTorch `2.7.0` is validated for both the Python bridge and the experimental Rust `tch 0.20.0` probe.
- **Rejected**: PyTorch `2.7.1` is rejected by `tch 0.20.0` version checks. Do not bypass `tch` version checks or upgrade PyTorch to `2.7.1` in environments shared with `tch` probes.

### CUDA Availability Checks

In `scripts/neural_bridge.py:92-93`, CUDA is enabled only when both requested in config (`cuda: true`) and reported available by PyTorch:

```python
cuda_requested = bool(cfg.get("cuda", False))
use_cuda = cuda_requested and torch.cuda.is_available()
```

Verify CUDA availability in the `uv` environment before launching GPU runs:

```bash
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

## CPU vs. GPU Reproducibility Limits

- **CPU**: Deterministic seeding produces reproducible results on the same host, software versions, and thread configuration.
- **GPU (CUDA)**: Even with deterministic flags enabled, CUDA kernels, cuDNN algorithm selection, and floating-point reduction order can vary across GPU architectures, driver versions, and CUDA toolkit releases.
- **Cross-platform**: A seed alone does not guarantee identical outputs across CPU and GPU or across different hardware/OS environments. Always record `software_version`, `environment`, and `cuda` status in the manifest.

## Seed Handling

`set_deterministic_seeds(seed)` (`scripts/neural_bridge.py:33-45`) configures all relevant RNGs and determinism flags before model initialization (`scripts/neural_bridge.py:214`, `scripts/neural_bridge.py:235`):

- `os.environ["PYTHONHASHSEED"] = str(seed)` (`scripts/neural_bridge.py:35`)
- `random.seed(seed)` (`scripts/neural_bridge.py:36`)
- `np.random.seed(seed)` (`scripts/neural_bridge.py:37`)
- `torch.manual_seed(seed)` (`scripts/neural_bridge.py:38`)
- `torch.cuda.manual_seed_all(seed)` when `torch.cuda.is_available()` (`scripts/neural_bridge.py:39-40`)
- `torch.backends.cudnn.deterministic = True` and `torch.backends.cudnn.benchmark = False` (`scripts/neural_bridge.py:41-42`)
- `torch.use_deterministic_algorithms(True, warn_only=True)` (`scripts/neural_bridge.py:45`) — `warn_only=True` avoids crashing on PyTorch ops that lack deterministic kernels while enforcing determinism wherever supported.

## Conditioning Policies

For the experimental neural candidate:
- **Log-frequency (default)**: Preserves the historical comparator baseline. Use this as the default for comparability across runs.
- **Empirical-frequency**: Use only for bounded diagnostics when comparing conditioning behavior. Keep fit and sample policies aligned, and record the active policy alongside seeds and environment metadata in aggregate reports.
- Do not treat changes in conditioning, ordinal encoding, or softmax decoding as clinical validation.

## In-Memory Neural IPC and Aggregate-Only Flows (Phase 5)

Phase 5 adds a strictly in-memory JSON-over-stdio protocol (`scripts/neural_bridge.py --stdio`, invoked by `crates/janus/src/neural.rs::run_neural_bridge_in_memory`) for restricted-source and multi-candidate workflows:

- **Zero patient/weight disk I/O**: Training rows (`Row`), sampled draws across `sample_seeds`, and fitted model weights stay strictly in memory across the Rust–Python stdio boundary; neither `samples.csv` nor `model.pkl` is written to disk.
- **Aggregate-only CLI generation (`--in-memory-aggregate`)**:
  ```bash
  janus-cli generate <study.yaml> <neural_config.json> <output_dir> \
    --generator neural-ctgan --training-data <train.csv> --in-memory-aggregate
  ```
  Writes only `study.yaml`, `config.json`, `aggregate_summary.json` (observed/missing counts per variable and seed), and `manifest.json` (`NeuralGeneratorManifest`).
- **Unified candidate comparison (`compare-candidates`)**:
  ```bash
  janus-cli compare-candidates --config <comparison.yaml> \
    --training <train.csv> --reference <ref.csv> \
    --neural-config <neural_config.json> --output <output_dir>
  ```
  Evaluates configured candidates (`ForestCandidate`, `DagCandidate`, `InteractionDagCandidate`, and `NeuralCandidate` in `crates/janus/src/candidate.rs`) under identical splits, seeds, sample counts, and denominators, plus multi-seed in-memory forensic evaluation (`orchestrate_forensic_suite`), writing only `comparison_manifest.json` and `forensic_manifests.json`.
- **Standard fictional disk mode**: Running `janus-cli generate --generator neural-ctgan` *without* `--in-memory-aggregate` writes `samples.csv`, `model.pkl`, `neural_manifest.json`, `config.json`, `study.yaml`, `manifest.json`, and (when `samples.csv` matches the `Study` schema) `fixture.yaml` + `fixture_fingerprint` for `janus-cli evaluate --synthetic-manifest`. Use standard mode only on fictional or explicitly permitted local directories, never on aggregate-only restricted sources.
- **Reproducibility auditing**: `janus-cli audit-reproducibility` verifies `manifest.json` against both `fnv1a64-yaml` and `sha256` `Cargo.lock` fingerprints alongside OS/architecture metadata and optional split fingerprints.
- **Bounded scope**: `experimental/neural/` remains untouched. Passing in-memory neural IPC, candidate comparison, or reproducibility audits does not establish clinical plausibility, independent validation, causality, privacy safety, or GPU reproducibility.

## Troubleshooting

- **Bridge script not found**: `resolve_bridge_script` (`crates/janus/src/neural.rs:220-265`) searches `--bridge-script`, `JANUS_NEURAL_BRIDGE`, `./scripts/neural_bridge.py`, ancestor directories, and the compile-time workspace root. Pass `--bridge-script` or set `JANUS_NEURAL_BRIDGE` if running from outside the repository.
- **`uv` environment issues**: Ensure `uv` is installed and on `PATH`. Run `uv sync` in the project root containing `pyproject.toml` if dependencies (`torch`, `ctgan`, `pandas`, `numpy`) are missing.
- **LibTorch unavailable / `tch` version mismatch**: Keep `PyTorch == 2.7.0` when running `tch 0.20.0` probes. Offline environments without cached `tch`/LibTorch wheels cannot build or run the standalone `tch` probe; use `scripts/neural_bridge.py` via `uv` instead.
- **CTGAN `pac` divisibility error**: `scripts/neural_bridge.py:107-110` requires `pac > 0` and `batch_size % pac == 0` (defaults: `batch_size=500`, `pac=10`). Adjust `batch_size` or `pac` in `NeuralGeneratorConfig` (`crates/janus/src/neural.rs:59-98`).
