"""Synthetic orchestration tests; no real model fits or downloaded data."""

import copy
import csv
import gzip
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.vkr_model import attempts, horizons
from src.vkr_model import train as training
from tests.test_controls import FakeControlRegressor


class HorizonsTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.dataset = self.root / "dataset"
        self.dataset.mkdir()
        (self.dataset / "market.sqlite3").write_bytes(b"immutable synthetic database")
        self.dataset_config = self.root / "dataset.json"
        self.dataset_config.write_text("{}")
        self.ledger = self.root / "training/attempts.jsonl"
        self.ledger.parent.mkdir()
        for name in ("legacy4", "legacy6", "control1", "control2", "control3"):
            for status in ("STARTED", "SUCCESS"):
                attempts.append(self.ledger, "older", name, 4, status, "older-exp")
        self.ledger_prefix = self.ledger.read_bytes()
        self.config = {"experiment_id": "0007_horizon_control", "dataset_config": str(self.dataset_config),
                       "dataset_dir": str(self.dataset), "expected_snapshot_id": "a" * 64,
                       "expected_dataset_sha256": "b" * 64, "artifact_dir": str(self.root / "horizons"),
                       "budget_ledger": str(self.ledger), "horizons": [4, 24], "params": dict(horizons.PARAMS),
                       **{role + "_" + key: value for role, pair in horizons.BOUNDARIES.items()
                          for key, value in zip(("start", "end"), pair)}}
        self.data = {role: [] for role in horizons.BOUNDARIES}
        for role, (start, _) in horizons.BOUNDARIES.items():
            for index in range(3):
                decision = training._ms(start) + index * training.HOUR
                execution = decision + 120_000
                self.data[role].append({"decision_time": decision, "signal_time": decision + 60_000,
                    "execution_time": execution, "split": role, "source_split": "train",
                    **{name: 10.0 + index + column for column, name in enumerate(training.FEATURE_COLUMNS)},
                    "targets": {key: {"label_end": execution + hours * training.HOUR,
                                      "y": (index + 1) * (0.01 if key == "h4" else 0.04)
                                      * (1 if role == "inner_train" else 3)}
                                for key, hours in horizons.HORIZONS.items()}})
        self.verification = {"verified": True, "snapshot_id": "a" * 64,
                             "dataset_sha256": "b" * 64, "source_sha256": "c" * 64}
        self.selection = {"rows": {role: len(rows) for role, rows in self.data.items()},
                          "selected_rows_sha256": "d" * 64, "common_keys_sha256": "e" * 64}
        FakeControlRegressor.fits = []
        FakeControlRegressor.fail_index = None
        FakeControlRegressor.wrong_tree_count = False
        FakeControlRegressor.reload_shift = 0.0
        for owner, name, replacement in (
                (horizons, "_load_data", lambda config: (copy.deepcopy(self.data), dict(self.verification), copy.deepcopy(self.selection))),
                (horizons, "_pool", lambda x, y=None: {"x": x, "y": y, "names": list(training.FEATURE_COLUMNS)}),
                (training, "_backend", lambda: FakeControlRegressor),
                (training, "_runtime", lambda: {"python": "3.12.test", "catboost": "1.2.10"})):
            mock = patch.object(owner, name, replacement)
            mock.start()
            self.addCleanup(mock.stop)

    def latest_report(self):
        return json.loads(sorted((self.root / "horizons").glob("run-*/report.json"))[-1].read_text())

    def test_two_fits_same_features_keys_and_training_only_targets(self):
        report = horizons.run_horizons(self.config)
        self.assertEqual(report["status"], "SUCCESS")
        self.assertEqual(report["accounting"]["fit_calls"], 2)
        self.assertEqual(report["accounting"]["ledger_after"]["started_attempts"], 7)
        self.assertEqual([item["variant"] for item in report["attempts"]], ["h4", "h24"])
        self.assertEqual([item["tree_count"] for item in report["attempts"]], [300, 300])
        self.assertFalse(report["validation_2024_used"])
        self.assertFalse(report["final_split_used"])
        self.assertEqual(len(FakeControlRegressor.fits), 2)
        self.assertEqual(FakeControlRegressor.fits[0]["pool"]["x"], FakeControlRegressor.fits[1]["pool"]["x"])
        self.assertEqual(FakeControlRegressor.fits[0]["pool"]["y"], [0.01, 0.02, 0.03])
        self.assertEqual(FakeControlRegressor.fits[1]["pool"]["y"], [0.04, 0.08, 0.12])
        for fit in FakeControlRegressor.fits:
            self.assertEqual(fit["pool"]["names"], list(training.FEATURE_COLUMNS))
            self.assertEqual(fit["params"], horizons.PARAMS)
        self.assertEqual(report["baseline_train_mean"], {"h4": 0.02, "h24": 0.08})
        self.assertTrue(self.ledger.read_bytes().startswith(self.ledger_prefix))
        self.assertTrue(report["dataset_immutable"])
        self.assertNotIn(str(self.root), (Path(report["run_dir"]) / "report.json").read_text())
        with gzip.open(Path(report["run_dir"]) / "predictions.csv.gz", "rt") as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), 36)
        self.assertEqual({row["horizon"] for row in rows}, {"h4", "h24"})

    def test_verify_reloads_models_reproduces_metrics_and_never_fits(self):
        report = horizons.run_horizons(self.config)
        before = {p: p.read_bytes() for p in Path(report["run_dir"]).rglob("*") if p.is_file()}
        ledger_before = self.ledger.read_bytes()
        verified = horizons.verify_horizons(self.config, report["run_dir"])
        self.assertTrue(verified["verified"])
        self.assertTrue(verified["metrics_reproduced"])
        self.assertEqual(verified["fit_calls"], 0)
        self.assertEqual(len(FakeControlRegressor.fits), 2)
        self.assertEqual(self.ledger.read_bytes(), ledger_before)
        self.assertTrue(all(path.read_bytes() == value for path, value in before.items()))

    def test_failed_fit_is_durable_counted_and_not_retried(self):
        FakeControlRegressor.fail_index = 1
        with self.assertRaisesRegex(RuntimeError, "synthetic control failure"):
            horizons.run_horizons(self.config)
        report = self.latest_report()
        self.assertEqual(report["status"], "FAILED")
        self.assertEqual(report["accounting"]["started_in_run"], 1)
        self.assertEqual(report["accounting"]["ledger_after"]["started_attempts"], 6)
        self.assertEqual(len(FakeControlRegressor.fits), 1)
        self.assertEqual([row["status"] for row in attempts.load(self.ledger)[-2:]], ["STARTED", "FAILED"])

    def test_changed_params_or_horizons_rejected_before_outputs(self):
        for key, value in (("horizons", [24]), ("params", {**horizons.PARAMS, "depth": 6}),
                           ("assessment_end", "2025-01-01T00:00:00Z")):
            modified = {**self.config, key: value}
            with self.assertRaises(ValueError):
                horizons.run_horizons(modified)
        self.assertEqual(self.ledger.read_bytes(), self.ledger_prefix)
        self.assertFalse((self.root / "horizons").exists())

    def test_crossing_label_or_forbidden_year_stops_before_fit(self):
        self.data["assessment"][0]["targets"]["h24"]["label_end"] = training._ms("2024-01-01T00:00:00Z")
        with self.assertRaisesRegex(RuntimeError, "boundary"):
            horizons.run_horizons(self.config)
        self.assertEqual(FakeControlRegressor.fits, [])
        self.assertEqual(self.ledger.read_bytes(), self.ledger_prefix)

    def test_verify_guard_blocks_hidden_fit(self):
        report = horizons.run_horizons(self.config)
        def unsafe_loader(config):
            FakeControlRegressor().fit({})
        with patch.object(horizons, "_load_data", unsafe_loader):
            with self.assertRaisesRegex(RuntimeError, "fit is forbidden"):
                horizons.verify_horizons(self.config, report["run_dir"])
        self.assertEqual(len(FakeControlRegressor.fits), 2)

    def test_artifact_tampering_stops_verify(self):
        report = horizons.run_horizons(self.config)
        path = Path(report["run_dir"]) / "h24.cbm"
        path.write_bytes(path.read_bytes() + b" ")
        with self.assertRaisesRegex(ValueError, "artifact integrity"):
            horizons.verify_horizons(self.config, report["run_dir"])

    def test_model_tree_count_and_reload_predictions_are_enforced(self):
        FakeControlRegressor.wrong_tree_count = True
        with self.assertRaisesRegex(RuntimeError, "300 trees"):
            horizons.run_horizons(self.config)
        self.assertEqual(len(FakeControlRegressor.fits), 1)


if __name__ == "__main__":
    unittest.main()
