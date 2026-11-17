"""Unit and fast deterministic replay tests for scripts/neural_bridge.py."""

from __future__ import annotations

import io
import json
import os
import random
import subprocess
import sys
import tempfile
import unittest
import unittest.mock
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from ctgan import CTGAN, TVAE

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import neural_bridge  # noqa: E402


def _write_fictional_csv(path: Path, n_rows: int = 20) -> None:
    """Write a small synthetic tabular CSV for fast neural bridge tests."""
    rows = ["exposure,age,group,outcome\n"]
    for i in range(n_rows):
        exposure = "true" if i % 2 == 0 else "false"
        age = 30 + (i * 2) % 40
        group = "control" if i % 3 == 0 else "active"
        outcome = "true" if i % 4 == 0 else "false"
        rows.append(f"{exposure},{age},{group},{outcome}\n")
    path.write_text("".join(rows), encoding="utf-8")


class NeuralBridgeTests(unittest.TestCase):
    """Verify seeding, column resolution, model construction, and replay."""

    def setUp(self) -> None:
        self._orig_pythonhashseed = os.environ.get("PYTHONHASHSEED")
        self._orig_det_algos = torch.are_deterministic_algorithms_enabled()

    def tearDown(self) -> None:
        if self._orig_pythonhashseed is None:
            os.environ.pop("PYTHONHASHSEED", None)
        else:
            os.environ["PYTHONHASHSEED"] = self._orig_pythonhashseed
        torch.use_deterministic_algorithms(self._orig_det_algos, warn_only=True)

    def test_set_deterministic_seeds_replays_rngs(self) -> None:
        neural_bridge.set_deterministic_seeds(123)
        self.assertEqual(os.environ.get("PYTHONHASHSEED"), "123")
        self.assertTrue(torch.are_deterministic_algorithms_enabled())

        py_a = [random.random() for _ in range(4)]
        np_a = np.random.rand(4)
        torch_a = torch.rand(4)

        neural_bridge.set_deterministic_seeds(123)
        py_b = [random.random() for _ in range(4)]
        np_b = np.random.rand(4)
        torch_b = torch.rand(4)

        self.assertEqual(py_a, py_b)
        np.testing.assert_array_equal(np_a, np_b)
        self.assertTrue(torch.equal(torch_a, torch_b))

        neural_bridge.set_deterministic_seeds(456)
        torch_other = torch.rand(4)
        self.assertFalse(torch.equal(torch_a, torch_other))

    def test_resolve_discrete_columns_explicit_auto_and_missing(self) -> None:
        df = pd.DataFrame(
            {
                "flag": [True, False, True],
                "label": ["a", "b", "a"],
                "cat": pd.Categorical(["x", "y", "x"]),
                "age": [35, 42, 51],
                "score": [1.5, 2.5, 3.5],
            }
        )

        self.assertEqual(
            neural_bridge.resolve_discrete_columns(df, None),
            ["flag", "label", "cat"],
        )
        self.assertEqual(
            neural_bridge.resolve_discrete_columns(df, ["age", "label"]),
            ["age", "label"],
        )
        self.assertEqual(neural_bridge.resolve_discrete_columns(df, []), [])

        with self.assertRaisesRegex(ValueError, "not found in dataset"):
            neural_bridge.resolve_discrete_columns(df, ["label", "missing_col"])

    def test_build_model_ctgan_and_tvae_parameters(self) -> None:
        ctgan_cfg = {
            "epochs": 2,
            "batch_size": 10,
            "embedding_dim": 16,
            "generator_dim": [32, 32],
            "discriminator_dim": [24, 24],
            "generator_lr": 1e-3,
            "discriminator_lr": 5e-4,
            "discriminator_steps": 2,
            "pac": 2,
            "cuda": False,
        }
        ctgan_model, ctgan_params = neural_bridge.build_model("ctgan", ctgan_cfg, seed=11)
        self.assertIsInstance(ctgan_model, CTGAN)
        self.assertEqual(
            ctgan_params,
            {
                "model_type": "ctgan",
                "epochs": 2,
                "batch_size": 10,
                "embedding_dim": 16,
                "generator_dim": [32, 32],
                "discriminator_dim": [24, 24],
                "generator_lr": 1e-3,
                "discriminator_lr": 5e-4,
                "discriminator_steps": 2,
                "pac": 2,
                "cuda": False,
            },
        )

        tvae_cfg = {
            "epochs": 2,
            "batch_size": 10,
            "embedding_dim": 16,
            "compress_dims": [16, 16],
            "decompress_dims": [20, 20],
            "l2scale": 1e-4,
            "loss_factor": 3,
            "cuda": False,
        }
        tvae_model, tvae_params = neural_bridge.build_model("tvae", tvae_cfg, seed=22)
        self.assertIsInstance(tvae_model, TVAE)
        self.assertEqual(
            tvae_params,
            {
                "model_type": "tvae",
                "epochs": 2,
                "batch_size": 10,
                "embedding_dim": 16,
                "compress_dims": [16, 16],
                "decompress_dims": [20, 20],
                "l2scale": 1e-4,
                "loss_factor": 3,
                "cuda": False,
            },
        )

    def test_build_model_validation_errors(self) -> None:
        with self.assertRaisesRegex(ValueError, "divisible by pac"):
            neural_bridge.build_model("ctgan", {"epochs": 2, "batch_size": 15, "pac": 10}, seed=1)

        with self.assertRaisesRegex(ValueError, "pac > 0"):
            neural_bridge.build_model("ctgan", {"epochs": 2, "batch_size": 10, "pac": 0}, seed=1)

        for bad_epochs in (0, -3):
            with self.assertRaisesRegex(ValueError, "epochs must be > 0"):
                neural_bridge.build_model("ctgan", {"epochs": bad_epochs, "batch_size": 10}, seed=1)
            with self.assertRaisesRegex(ValueError, "epochs must be > 0"):
                neural_bridge.build_model("tvae", {"epochs": bad_epochs, "batch_size": 10}, seed=1)

        with self.assertRaisesRegex(ValueError, "batch_size must be > 0"):
            neural_bridge.build_model("tvae", {"epochs": 2, "batch_size": 0}, seed=1)

        with self.assertRaisesRegex(ValueError, "Unsupported model_type"):
            neural_bridge.build_model("unknown", {"epochs": 2, "batch_size": 10}, seed=1)

    def _run_main_replay(self, model_type: str, cfg: dict[str, object], n_samples: int) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_csv = root / "fictional_train.csv"
            config_json = root / "config.json"
            _write_fictional_csv(input_csv, n_rows=20)
            config_json.write_text(json.dumps(cfg), encoding="utf-8")

            outputs: list[tuple[Path, Path, Path]] = []
            for run_name in ("run1", "run2"):
                run_dir = root / run_name
                samples_path = run_dir / "samples.csv"
                model_path = run_dir / "model.pkl"
                manifest_path = run_dir / "neural_manifest.json"
                rc = neural_bridge.main(
                    [
                        "--input",
                        str(input_csv),
                        "--config",
                        str(config_json),
                        "--output-samples",
                        str(samples_path),
                        "--output-model",
                        str(model_path),
                        "--output-manifest",
                        str(manifest_path),
                    ]
                )
                self.assertEqual(rc, 0)
                self.assertTrue(samples_path.is_file())
                self.assertTrue(model_path.is_file())
                self.assertTrue(manifest_path.is_file())
                outputs.append((samples_path, model_path, manifest_path))

            samples1 = outputs[0][0].read_text(encoding="utf-8")
            samples2 = outputs[1][0].read_text(encoding="utf-8")
            self.assertEqual(samples1, samples2)

            synth_df = pd.read_csv(outputs[0][0])
            self.assertEqual(len(synth_df), n_samples)
            self.assertEqual(list(synth_df.columns), ["exposure", "age", "group", "outcome"])

            manifest1 = json.loads(outputs[0][2].read_text(encoding="utf-8"))
            manifest2 = json.loads(outputs[1][2].read_text(encoding="utf-8"))
            self.assertEqual(manifest1["schema_version"], "1")
            self.assertEqual(manifest1["model_type"], model_type)
            self.assertEqual(manifest1["epochs"], 2)
            self.assertEqual(manifest1["seed"], cfg["seed"])
            self.assertEqual(manifest1["n_samples"], n_samples)
            self.assertEqual(manifest1["training_rows"], 20)
            self.assertEqual(manifest1["training_columns"], ["exposure", "age", "group", "outcome"])
            self.assertEqual(manifest1["discrete_columns"], ["exposure", "group", "outcome"])
            self.assertTrue(manifest1["training_dataset_fingerprint"].startswith("sha256:"))
            self.assertEqual(manifest1["samples_fingerprint"], manifest2["samples_fingerprint"])
            self.assertEqual(
                manifest1["artifacts"],
                ["samples.csv", "neural_manifest.json", "model.pkl"],
            )

    def test_main_ctgan_fast_deterministic_replay(self) -> None:
        self._run_main_replay(
            "ctgan",
            {
                "model_type": "ctgan",
                "epochs": 2,
                "batch_size": 10,
                "pac": 2,
                "embedding_dim": 8,
                "generator_dim": [16, 16],
                "discriminator_dim": [16, 16],
                "seed": 123,
                "n_samples": 12,
            },
            n_samples=12,
        )

    def test_main_tvae_fast_deterministic_replay(self) -> None:
        self._run_main_replay(
            "tvae",
            {
                "model_type": "tvae",
                "epochs": 2,
                "batch_size": 10,
                "embedding_dim": 8,
                "compress_dims": [16, 16],
                "decompress_dims": [16, 16],
                "seed": 77,
                "n_samples": 8,
            },
            n_samples=8,
        )

    def test_parse_args_stdio_vs_file_mode_validation(self) -> None:
        with unittest.mock.patch("sys.stderr", new_callable=io.StringIO):
            with self.assertRaises(SystemExit):
                neural_bridge.parse_args([])
            with self.assertRaises(SystemExit):
                neural_bridge.parse_args(["--stdio", "--output-samples", "out.csv"])
        parsed_stdio = neural_bridge.parse_args(["--stdio"])
        self.assertTrue(parsed_stdio.stdio)

    def test_stdio_zero_files_written_and_multiseed_deterministic_replay(self) -> None:
        columns = ["exposure", "age", "group", "outcome"]
        list_rows = [
            [
                "true" if i % 2 == 0 else "false",
                str(30 + (i * 2) % 20),
                "control" if i % 3 == 0 else "active",
                "true" if i % 4 == 0 else "false",
            ]
            for i in range(20)
        ]
        dict_rows = [dict(zip(columns, row)) for row in list_rows]

        base_cfg = {
            "model_type": "ctgan",
            "epochs": 2,
            "batch_size": 10,
            "pac": 2,
            "embedding_dim": 8,
            "generator_dim": [16, 16],
            "discriminator_dim": [16, 16],
        }
        req_list = {
            "config": base_cfg,
            "fit_seed": 42,
            "sample_seeds": [101, 202],
            "n_samples": 10,
            "columns": columns,
            "rows": list_rows,
        }
        req_dict = {
            "config": base_cfg,
            "fit_seed": 42,
            "sample_seeds": [101, 202],
            "n_samples": 10,
            "rows": dict_rows,
        }

        bridge_path = SCRIPTS_DIR / "neural_bridge.py"
        with tempfile.TemporaryDirectory() as tmp:
            cwd = Path(tmp)
            proc1 = subprocess.run(
                [sys.executable, str(bridge_path), "--stdio"],
                input=json.dumps(req_list),
                text=True,
                capture_output=True,
                cwd=cwd,
                check=False,
            )
            self.assertEqual(proc1.returncode, 0, msg=proc1.stderr)
            self.assertEqual(list(cwd.iterdir()), [])

            proc2 = subprocess.run(
                [sys.executable, str(bridge_path), "--stdio"],
                input=json.dumps(req_dict),
                text=True,
                capture_output=True,
                cwd=cwd,
                check=False,
            )
            self.assertEqual(proc2.returncode, 0, msg=proc2.stderr)
            self.assertEqual(list(cwd.iterdir()), [])

        out1 = json.loads(proc1.stdout)
        out2 = json.loads(proc2.stdout)

        self.assertEqual(out1["manifest"]["artifacts"], [])
        self.assertEqual(out1["manifest"]["seed"], 42)
        self.assertEqual(out1["manifest"]["n_samples"], 10)
        self.assertEqual(out1["manifest"]["training_rows"], 20)
        self.assertEqual(out1["manifest"]["training_columns"], columns)
        self.assertEqual(out1["manifest"]["discrete_columns"], ["exposure", "group", "outcome"])
        self.assertTrue(out1["manifest"]["training_dataset_fingerprint"].startswith("sha256:"))
        self.assertTrue(out1["manifest"]["samples_fingerprint"].startswith("sha256:"))
        self.assertEqual(
            out1["manifest"]["training_dataset_fingerprint"],
            out2["manifest"]["training_dataset_fingerprint"],
        )
        self.assertEqual(
            out1["manifest"]["samples_fingerprint"],
            out2["manifest"]["samples_fingerprint"],
        )
        self.assertEqual(out1["draws"], out2["draws"])
        self.assertEqual(set(out1["draws"].keys()), {"101", "202"})
        self.assertEqual(len(out1["draws"]["101"]), 10)
        self.assertEqual(len(out1["draws"]["202"]), 10)
        self.assertNotEqual(out1["draws"]["101"], out1["draws"]["202"])

    def test_stdio_missingness_preserved_without_zero_leakage(self) -> None:
        columns = ["cat", "age", "score", "all_na"]
        rows = []
        for i in range(20):
            cat = "NA" if i % 4 == 0 else ("" if i % 5 == 0 else ("a" if i % 2 == 0 else "b"))
            age = "NA" if i % 3 == 0 else str(30 + (i % 10) * 2)
            score = "" if i % 4 == 1 else ("na" if i % 7 == 0 else f"{10.5 + (i % 5) * 1.25:.2f}")
            all_na = "" if i % 2 == 0 else "NA"
            rows.append([cat, age, score, all_na])

        for model_type, model_cfg in (
            (
                "ctgan",
                {
                    "model_type": "ctgan",
                    "epochs": 2,
                    "batch_size": 10,
                    "pac": 2,
                    "embedding_dim": 8,
                    "generator_dim": [16, 16],
                    "discriminator_dim": [16, 16],
                },
            ),
            (
                "tvae",
                {
                    "model_type": "tvae",
                    "epochs": 2,
                    "batch_size": 10,
                    "embedding_dim": 8,
                    "compress_dims": [16, 16],
                    "decompress_dims": [16, 16],
                },
            ),
        ):
            payload = {
                "config": model_cfg,
                "fit_seed": 17,
                "sample_seeds": [101, 202],
                "n_samples": 12,
                "columns": columns,
                "rows": rows,
            }
            bridge_path = SCRIPTS_DIR / "neural_bridge.py"
            with tempfile.TemporaryDirectory() as tmp:
                proc = subprocess.run(
                    [sys.executable, str(bridge_path), "--stdio"],
                    input=json.dumps(payload),
                    text=True,
                    capture_output=True,
                    cwd=tmp,
                    check=False,
                )
                self.assertEqual(proc.returncode, 0, msg=f"{model_type} failed: {proc.stderr}")
                self.assertEqual(list(Path(tmp).iterdir()), [])

            result = json.loads(proc.stdout)
            manifest = result["manifest"]
            self.assertEqual(manifest["model_type"], model_type)
            self.assertEqual(manifest["training_columns"], columns)
            self.assertEqual(manifest["discrete_columns"], ["cat", "all_na"])
            self.assertEqual(manifest["artifacts"], [])

            for seed_key in ("101", "202"):
                draw_rows = result["draws"][seed_key]
                self.assertEqual(len(draw_rows), 12)
                for row in draw_rows:
                    self.assertEqual(list(row.keys()), columns)
                    self.assertEqual(row["all_na"], "NA")
                    self.assertIn(row["cat"], {"a", "b", "NA"})
                    for numeric_col in ("age", "score"):
                        val = row[numeric_col]
                        self.assertNotEqual(val, "")
                        if val != "NA":
                            parsed = float(val)
                            self.assertNotEqual(parsed, 0.0)


if __name__ == "__main__":
    unittest.main()
