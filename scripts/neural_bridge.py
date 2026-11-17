#!/usr/bin/env python3
"""Thin Python bridge for training and sampling CTGAN/TVAE models in Janus.

Decision & Rationale:
- Keeps neural model execution (PyTorch/ctgan) behind a narrow, study-agnostic
  CLI boundary invoked by `janus-cli generate --generator neural-ctgan`.
- Preserves Janus's Rust core free of Python/LibTorch runtime dependencies.
- Enforces deterministic seeding across Python, NumPy, PyTorch, and CTGAN/TVAE,
  and emits a complete JSON manifest with hyperparameters, epochs, seed,
  dataset fingerprint, and environment versions.
"""

from __future__ import annotations

import argparse
import contextlib
import datetime
import hashlib
import json
import math
import os
import platform
import random
import re
import statistics
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import ctgan
from ctgan import CTGAN, TVAE

_INT_LITERAL_RE = re.compile(r"^[+-]?\d+$")
_INT64_MIN = -(1 << 63)
_INT64_MAX = (1 << 63) - 1


def set_deterministic_seeds(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch RNGs for deterministic execution."""
    if seed < 0:
        raise ValueError(f"seed must be >= 0, got {seed}")
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed & 0xFFFFFFFF)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # warn_only=True avoids crashing on ops without deterministic kernels while
    # still enforcing deterministic algorithms wherever PyTorch supports them.
    torch.use_deterministic_algorithms(True, warn_only=True)


def _set_model_random_state(model: CTGAN | TVAE, seed: int) -> None:
    """Apply a deterministic random state to a CTGAN/TVAE synthesizer."""
    if not hasattr(model, "set_random_state"):
        return
    if seed < (1 << 32):
        model.set_random_state(seed)
    else:
        model.set_random_state(
            (np.random.RandomState(seed=seed & 0xFFFFFFFF), torch.Generator().manual_seed(seed))
        )


def sha256_file(path: Path) -> str:
    """Compute a prefixed SHA-256 fingerprint of a file's raw bytes."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return f"sha256:{h.hexdigest()}"


def sha256_canonical_json(value: Any) -> str:
    """Compute a prefixed SHA-256 fingerprint over canonical JSON."""
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def is_missing_token(raw: Any) -> bool:
    """Return True if raw represents a missing value (None, blank, or case-insensitive 'NA')."""
    if raw is None:
        return True
    if isinstance(raw, float) and math.isnan(raw):
        return True
    trimmed = str(raw).strip()
    return trimmed == "" or trimmed.upper() == "NA"


def _validate_seed(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a non-negative integer, got {value!r}")
    return int(value)


def _scalar_to_token(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"Non-finite float value in input rows: {value!r}")
        return str(value)
    if isinstance(value, str):
        return value
    raise ValueError(f"Unsupported cell value type in input rows: {type(value).__name__}")


def _try_parse_finite_float(token: str) -> float | None:
    s = token.strip()
    if not s or "_" in s:
        return None
    try:
        val = float(s)
    except ValueError:
        return None
    if not math.isfinite(val):
        return None
    return val


def _unique_indicator_name(col: str, existing: set[str]) -> str:
    candidate = f"__janus_missing_{col}"
    suffix = 1
    while candidate in existing:
        candidate = f"__janus_missing_{col}_{suffix}"
        suffix += 1
    existing.add(candidate)
    return candidate


def _parse_stdio_table(
    raw_columns: Any,
    raw_rows: Any,
) -> tuple[list[str], list[list[str]]]:
    if not isinstance(raw_rows, list) or len(raw_rows) == 0:
        raise ValueError("'rows' must be a non-empty list")

    first = raw_rows[0]
    if isinstance(first, dict):
        if raw_columns is None:
            columns = [str(k) for k in first.keys()]
        else:
            if not isinstance(raw_columns, list) or not all(
                isinstance(c, str) and c != "" for c in raw_columns
            ):
                raise ValueError("'columns' must be a non-empty list of non-empty strings")
            columns = list(raw_columns)
        if not columns or len(set(columns)) != len(columns):
            raise ValueError("'columns' must be non-empty and contain unique names")
        expected_keys = set(columns)
        matrix: list[list[str]] = []
        for idx, row in enumerate(raw_rows):
            if not isinstance(row, dict):
                raise ValueError(f"rows[{idx}] must be an object matching row 0")
            if set(row.keys()) != expected_keys:
                raise ValueError(f"rows[{idx}] keys do not match expected columns {columns}")
            matrix.append([_scalar_to_token(row[c]) for c in columns])
        return columns, matrix

    if isinstance(first, list):
        if not isinstance(raw_columns, list) or not raw_columns or not all(
            isinstance(c, str) and c != "" for c in raw_columns
        ):
            raise ValueError("'columns' is required as a list of non-empty strings when rows are lists")
        columns = list(raw_columns)
        if len(set(columns)) != len(columns):
            raise ValueError("'columns' must contain unique names")
        matrix = []
        for idx, row in enumerate(raw_rows):
            if not isinstance(row, list):
                raise ValueError(f"rows[{idx}] must be a list matching row 0")
            if len(row) != len(columns):
                raise ValueError(
                    f"rows[{idx}] length {len(row)} does not match columns length {len(columns)}"
                )
            matrix.append([_scalar_to_token(cell) for cell in row])
        return columns, matrix

    raise ValueError("'rows' must be a list of lists or a list of objects")


def _prepare_stdio_dataframe(
    columns: list[str],
    matrix: list[list[str]],
    configured_discrete: list[str] | None,
) -> tuple[pd.DataFrame, list[str], list[str], dict[str, str]]:
    """Build an in-memory DataFrame for CTGAN/TVAE preserving missingness semantics.

    - Discrete columns preserve missing tokens ("" or case-insensitive "NA") as explicit "NA"
      with object dtype so pandas never coerces "NA" to NaN.
    - All-missing columns are treated as discrete with category ["NA"].
    - Numeric columns with both observed numeric values and missing values use a companion
      discrete indicator column `__janus_missing_<col>` ("1"/"0") and fill missing positions
      internally with the training median only for model fitting.
    """
    if configured_discrete is not None:
        if not isinstance(configured_discrete, list) or not all(
            isinstance(c, str) for c in configured_discrete
        ):
            raise ValueError("discrete_columns must be a list of column names")
        unknown = [c for c in configured_discrete if c not in columns]
        if unknown:
            raise ValueError(f"Configured discrete_columns not found in dataset: {unknown}")
        configured_set = set(configured_discrete)
    else:
        configured_set = None

    used_names = set(columns)
    series_map: dict[str, pd.Series] = {}
    public_discrete: list[str] = []
    fit_discrete: list[str] = []
    indicator_for_col: dict[str, str] = {}

    for col_idx, col in enumerate(columns):
        raw_col = [row[col_idx] for row in matrix]
        missing_flags = [is_missing_token(val) for val in raw_col]
        observed = [val.strip() for val, is_miss in zip(raw_col, missing_flags) if not is_miss]

        if not observed:
            series_map[col] = pd.Series(["NA"] * len(raw_col), dtype=object)
            public_discrete.append(col)
            fit_discrete.append(col)
            continue

        if configured_set is not None and col in configured_set:
            discrete_vals = ["NA" if is_miss else val for val, is_miss in zip(raw_col, missing_flags)]
            series_map[col] = pd.Series(discrete_vals, dtype=object)
            public_discrete.append(col)
            fit_discrete.append(col)
            continue

        parsed_floats = [_try_parse_finite_float(tok) for tok in observed]
        is_numeric = all(v is not None for v in parsed_floats)

        if not is_numeric:
            if configured_set is not None:
                raise ValueError(
                    f"Non-discrete column '{col}' contains non-numeric observed values"
                )
            discrete_vals = ["NA" if is_miss else val for val, is_miss in zip(raw_col, missing_flags)]
            series_map[col] = pd.Series(discrete_vals, dtype=object)
            public_discrete.append(col)
            fit_discrete.append(col)
            continue

        all_int = all(_INT_LITERAL_RE.fullmatch(tok) is not None for tok in observed)
        if all_int:
            observed_ints = [int(tok) for tok in observed]
            if not all(_INT64_MIN <= v <= _INT64_MAX for v in observed_ints):
                all_int = False

        if all_int:
            observed_ints = [int(tok) for tok in observed]
            fill_int = int(round(statistics.median(observed_ints)))
            numeric_vals: list[int] = []
            for val, is_miss in zip(raw_col, missing_flags):
                numeric_vals.append(fill_int if is_miss else int(val.strip()))
            series_map[col] = pd.Series(numeric_vals, dtype="int64")
        else:
            observed_floats = [float(v) for v in parsed_floats if v is not None]
            fill_float = float(statistics.median(observed_floats))
            float_vals: list[float] = []
            for val, is_miss in zip(raw_col, missing_flags):
                if is_miss:
                    float_vals.append(fill_float)
                else:
                    parsed = _try_parse_finite_float(val)
                    assert parsed is not None
                    float_vals.append(parsed)
            series_map[col] = pd.Series(float_vals, dtype="float64")

        if any(missing_flags):
            ind_col = _unique_indicator_name(col, used_names)
            indicator_for_col[col] = ind_col
            series_map[ind_col] = pd.Series(
                ["1" if is_miss else "0" for is_miss in missing_flags],
                dtype=object,
            )
            fit_discrete.append(ind_col)

    return pd.DataFrame(series_map), public_discrete, fit_discrete, indicator_for_col


def _decode_sampled_rows(
    synthetic_df: pd.DataFrame,
    columns: list[str],
    public_discrete_set: set[str],
    indicator_for_col: dict[str, str],
) -> list[dict[str, str]]:
    """Convert sampled DataFrame rows back to string dicts, restoring 'NA' via indicators."""
    records: list[dict[str, str]] = []
    n_rows = len(synthetic_df)
    for row_idx in range(n_rows):
        row_out: dict[str, str] = {}
        for col in columns:
            ind_col = indicator_for_col.get(col)
            if ind_col is not None and str(synthetic_df.at[row_idx, ind_col]).strip() == "1":
                row_out[col] = "NA"
                continue

            val = synthetic_df.at[row_idx, col]
            if val is None or pd.isna(val):
                row_out[col] = "NA"
            elif col in public_discrete_set:
                text = str(val)
                row_out[col] = "NA" if is_missing_token(text) else text
            elif isinstance(val, (bool, np.bool_)):
                row_out[col] = "true" if bool(val) else "false"
            elif isinstance(val, (int, np.integer)):
                row_out[col] = str(int(val))
            elif isinstance(val, (float, np.floating)):
                fval = float(val)
                row_out[col] = str(fval) if math.isfinite(fval) else "NA"
            else:
                row_out[col] = str(val)
        records.append(row_out)
    return records


def load_tabular(path: Path) -> pd.DataFrame:
    """Load a CSV or Parquet tabular dataset without study-specific assumptions."""
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported input format '{path.suffix}'; expected .csv or .parquet")


def resolve_discrete_columns(df: pd.DataFrame, configured: list[str] | None) -> list[str]:
    """Use configured discrete columns or auto-detect non-numeric columns."""
    if configured is not None:
        missing = [c for c in configured if c not in df.columns]
        if missing:
            raise ValueError(f"Configured discrete_columns not found in dataset: {missing}")
        return list(configured)
    discrete: list[str] = []
    for col in df.columns:
        dtype = df[col].dtype
        if (
            pd.api.types.is_bool_dtype(dtype)
            or pd.api.types.is_object_dtype(dtype)
            or pd.api.types.is_string_dtype(dtype)
            or isinstance(dtype, pd.CategoricalDtype)
        ):
            discrete.append(str(col))
    return discrete


def build_model(model_type: str, cfg: dict[str, Any], seed: int) -> tuple[CTGAN | TVAE, dict[str, Any]]:
    """Instantiate CTGAN or TVAE with explicit hyperparameters and deterministic seed."""
    epochs = int(cfg.get("epochs", 300))
    batch_size = int(cfg.get("batch_size", 500))
    embedding_dim = int(cfg.get("embedding_dim", 128))
    cuda_requested = bool(cfg.get("cuda", False))
    use_cuda = cuda_requested and torch.cuda.is_available()

    if epochs <= 0:
        raise ValueError(f"epochs must be > 0, got {epochs}")
    if batch_size <= 0:
        raise ValueError(f"batch_size must be > 0, got {batch_size}")

    if model_type == "ctgan":
        generator_dim = tuple(int(x) for x in cfg.get("generator_dim", [256, 256]))
        discriminator_dim = tuple(int(x) for x in cfg.get("discriminator_dim", [256, 256]))
        generator_lr = float(cfg.get("generator_lr", 2e-4))
        discriminator_lr = float(cfg.get("discriminator_lr", 2e-4))
        discriminator_steps = int(cfg.get("discriminator_steps", 1))
        pac = int(cfg.get("pac", 10))
        if pac <= 0 or batch_size % pac != 0:
            raise ValueError(
                f"CTGAN requires pac > 0 and batch_size ({batch_size}) divisible by pac ({pac})"
            )

        model = CTGAN(
            embedding_dim=embedding_dim,
            generator_dim=generator_dim,
            discriminator_dim=discriminator_dim,
            generator_lr=generator_lr,
            discriminator_lr=discriminator_lr,
            batch_size=batch_size,
            discriminator_steps=discriminator_steps,
            epochs=epochs,
            pac=pac,
            cuda=use_cuda,
            verbose=False,
        )
        effective_params = {
            "model_type": "ctgan",
            "epochs": epochs,
            "batch_size": batch_size,
            "embedding_dim": embedding_dim,
            "generator_dim": list(generator_dim),
            "discriminator_dim": list(discriminator_dim),
            "generator_lr": generator_lr,
            "discriminator_lr": discriminator_lr,
            "discriminator_steps": discriminator_steps,
            "pac": pac,
            "cuda": use_cuda,
        }
    elif model_type == "tvae":
        compress_dims = tuple(int(x) for x in cfg.get("compress_dims", [128, 128]))
        decompress_dims = tuple(int(x) for x in cfg.get("decompress_dims", [128, 128]))
        l2scale = float(cfg.get("l2scale", 1e-5))
        loss_factor = int(cfg.get("loss_factor", 2))

        model = TVAE(
            embedding_dim=embedding_dim,
            compress_dims=compress_dims,
            decompress_dims=decompress_dims,
            l2scale=l2scale,
            batch_size=batch_size,
            epochs=epochs,
            loss_factor=loss_factor,
            cuda=use_cuda,
            verbose=False,
        )
        effective_params = {
            "model_type": "tvae",
            "epochs": epochs,
            "batch_size": batch_size,
            "embedding_dim": embedding_dim,
            "compress_dims": list(compress_dims),
            "decompress_dims": list(decompress_dims),
            "l2scale": l2scale,
            "loss_factor": loss_factor,
            "cuda": use_cuda,
        }
    else:
        raise ValueError(f"Unsupported model_type '{model_type}'; expected 'ctgan' or 'tvae'")

    _set_model_random_state(model, seed)

    return model, effective_params


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Janus neural generator bridge (CTGAN / TVAE via ctgan)."
    )
    parser.add_argument(
        "--stdio",
        action="store_true",
        help="Read JSON request from stdin and emit manifest + multi-seed draws as JSON to stdout without writing files",
    )
    parser.add_argument("--input", required=False, default=None, type=Path, help="Training dataset (.csv or .parquet)")
    parser.add_argument("--config", type=Path, default=None, help="Path to JSON hyperparameter config")
    parser.add_argument("--output-samples", required=False, default=None, type=Path, help="Output CSV path for synthetic samples")
    parser.add_argument("--output-model", type=Path, default=None, help="Optional output path for saved model (.pkl)")
    parser.add_argument("--output-manifest", required=False, default=None, type=Path, help="Output JSON path for run manifest")
    parser.add_argument("--model-type", choices=["ctgan", "tvae"], default=None, help="Override model_type")
    parser.add_argument("--epochs", type=int, default=None, help="Override training epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch_size")
    parser.add_argument("--seed", type=int, default=None, help="Override deterministic seed")
    parser.add_argument("--n-samples", type=int, default=None, help="Override number of synthetic rows to sample")
    args = parser.parse_args(argv)
    if args.stdio:
        forbidden = [
            flag
            for flag, val in (
                ("--input", args.input),
                ("--output-samples", args.output_samples),
                ("--output-model", args.output_model),
                ("--output-manifest", args.output_manifest),
            )
            if val is not None
        ]
        if forbidden:
            parser.error(f"cannot combine --stdio with file arguments: {', '.join(forbidden)}")
    else:
        missing = [
            flag
            for flag, val in (
                ("--input", args.input),
                ("--output-samples", args.output_samples),
                ("--output-manifest", args.output_manifest),
            )
            if val is None
        ]
        if missing:
            parser.error(
                f"the following arguments are required unless --stdio is set: {', '.join(missing)}"
            )
    return args


def _apply_cli_overrides(cfg: dict[str, Any], args: argparse.Namespace) -> None:
    if args.model_type is not None:
        cfg["model_type"] = args.model_type
    if args.epochs is not None:
        cfg["epochs"] = args.epochs
    if args.batch_size is not None:
        cfg["batch_size"] = args.batch_size
    if args.seed is not None:
        cfg["seed"] = args.seed
    if args.n_samples is not None:
        cfg["n_samples"] = args.n_samples


def _build_manifest_payload(
    *,
    model_type: str,
    effective_params: dict[str, Any],
    seed: int,
    n_samples: int,
    dataset_fingerprint: str,
    training_rows: int,
    training_columns: list[str],
    discrete_columns: list[str],
    samples_fingerprint: str,
    artifacts: list[str],
) -> dict[str, Any]:
    return {
        "schema_version": "1",
        "model_type": model_type,
        "hyperparameters": effective_params,
        "epochs": effective_params["epochs"],
        "seed": seed,
        "n_samples": n_samples,
        "training_dataset_fingerprint": dataset_fingerprint,
        "training_rows": training_rows,
        "training_columns": training_columns,
        "discrete_columns": discrete_columns,
        "samples_fingerprint": samples_fingerprint,
        "timestamp": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "python_version": platform.python_version(),
        "ctgan_version": getattr(ctgan, "__version__", "unknown"),
        "torch_version": torch.__version__,
        "artifacts": artifacts,
    }


def _run_stdio_mode(args: argparse.Namespace) -> int:
    payload = json.load(sys.stdin)
    if not isinstance(payload, dict):
        raise ValueError("stdin payload must be a JSON object")

    cfg: dict[str, Any] = {}
    if args.config is not None:
        with args.config.open("r", encoding="utf-8") as f:
            file_cfg = json.load(f)
        if not isinstance(file_cfg, dict):
            raise ValueError("--config file must contain a JSON object")
        cfg.update(file_cfg)

    raw_cfg = payload.get("config")
    if raw_cfg is not None:
        if not isinstance(raw_cfg, dict):
            raise ValueError("'config' in stdin payload must be a JSON object")
        cfg.update(raw_cfg)

    _apply_cli_overrides(cfg, args)

    columns, matrix = _parse_stdio_table(payload.get("columns"), payload.get("rows"))

    raw_fit_seed = payload.get("fit_seed")
    if raw_fit_seed is None:
        raw_fit_seed = cfg.get("fit_seed")
    if raw_fit_seed is None:
        raw_fit_seed = cfg.get("seed")
    if raw_fit_seed is None:
        raw_fit_seed = 42
    fit_seed = _validate_seed(raw_fit_seed, "fit_seed")

    raw_sample_seeds = payload.get("sample_seeds")
    if raw_sample_seeds is None:
        raw_sample_seeds = cfg.get("sample_seeds")
    if raw_sample_seeds is None:
        raw_sample_seeds = [fit_seed]
    if not isinstance(raw_sample_seeds, list) or len(raw_sample_seeds) == 0:
        raise ValueError("sample_seeds must be a non-empty list of integers")
    sample_seeds = [_validate_seed(s, "sample_seeds[]") for s in raw_sample_seeds]

    raw_n_samples = payload.get("n_samples")
    if raw_n_samples is None:
        raw_n_samples = cfg.get("n_samples")
    if raw_n_samples is None:
        raw_n_samples = len(matrix)
    if isinstance(raw_n_samples, bool) or not isinstance(raw_n_samples, int) or raw_n_samples <= 0:
        raise ValueError(f"n_samples must be > 0, got {raw_n_samples!r}")
    n_samples = int(raw_n_samples)

    model_type = str(cfg.get("model_type") or "ctgan").lower()

    fit_df, public_discrete, fit_discrete, indicator_for_col = _prepare_stdio_dataframe(
        columns, matrix, cfg.get("discrete_columns")
    )
    public_discrete_set = set(public_discrete)
    dataset_fingerprint = sha256_canonical_json({"columns": columns, "rows": matrix})

    draws: dict[str, list[dict[str, str]]] = {}
    with contextlib.redirect_stdout(sys.stderr):
        set_deterministic_seeds(fit_seed)
        model, effective_params = build_model(model_type, cfg, fit_seed)
        effective_params["seed"] = fit_seed
        effective_params["fit_seed"] = fit_seed
        effective_params["sample_seeds"] = sample_seeds
        effective_params["n_samples"] = n_samples
        effective_params["discrete_columns"] = public_discrete

        model.fit(fit_df, discrete_columns=fit_discrete)

        for s_seed in sample_seeds:
            set_deterministic_seeds(s_seed)
            _set_model_random_state(model, s_seed)
            synthetic_df = model.sample(n_samples)
            draws[str(s_seed)] = _decode_sampled_rows(
                synthetic_df,
                columns,
                public_discrete_set,
                indicator_for_col,
            )

    samples_fingerprint = sha256_canonical_json(draws)
    manifest = _build_manifest_payload(
        model_type=model_type,
        effective_params=effective_params,
        seed=fit_seed,
        n_samples=n_samples,
        dataset_fingerprint=dataset_fingerprint,
        training_rows=len(matrix),
        training_columns=columns,
        discrete_columns=public_discrete,
        samples_fingerprint=samples_fingerprint,
        artifacts=[],
    )

    sys.stdout.write(json.dumps({"manifest": manifest, "draws": draws}))
    sys.stdout.write("\n")
    sys.stdout.flush()
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.stdio:
        return _run_stdio_mode(args)

    cfg: dict[str, Any] = {}
    if args.config is not None:
        with args.config.open("r", encoding="utf-8") as f:
            cfg = json.load(f)

    _apply_cli_overrides(cfg, args)

    model_type = str(cfg.get("model_type", "ctgan")).lower()
    seed = int(cfg.get("seed", 42))

    set_deterministic_seeds(seed)

    df = load_tabular(args.input)
    if df.empty:
        raise ValueError("Training dataset is empty")

    n_samples = int(cfg.get("n_samples", len(df)))
    if n_samples <= 0:
        raise ValueError(f"n_samples must be > 0, got {n_samples}")

    discrete_columns = resolve_discrete_columns(df, cfg.get("discrete_columns"))
    dataset_fingerprint = sha256_file(args.input)

    model, effective_params = build_model(model_type, cfg, seed)
    effective_params["seed"] = seed
    effective_params["n_samples"] = n_samples
    effective_params["discrete_columns"] = discrete_columns

    model.fit(df, discrete_columns=discrete_columns)

    # Re-seed before sampling so sampling is deterministic given the trained model and seed.
    set_deterministic_seeds(seed)
    _set_model_random_state(model, seed)
    synthetic_df = model.sample(n_samples)

    args.output_samples.parent.mkdir(parents=True, exist_ok=True)
    synthetic_df.to_csv(args.output_samples, index=False)
    samples_fingerprint = sha256_file(args.output_samples)

    artifacts = [args.output_samples.name, args.output_manifest.name]
    if args.output_model is not None:
        args.output_model.parent.mkdir(parents=True, exist_ok=True)
        model.save(str(args.output_model))
        artifacts.append(args.output_model.name)

    manifest = _build_manifest_payload(
        model_type=model_type,
        effective_params=effective_params,
        seed=seed,
        n_samples=n_samples,
        dataset_fingerprint=dataset_fingerprint,
        training_rows=int(len(df)),
        training_columns=[str(c) for c in df.columns],
        discrete_columns=discrete_columns,
        samples_fingerprint=samples_fingerprint,
        artifacts=artifacts,
    )

    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    with args.output_manifest.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
        f.write("\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
