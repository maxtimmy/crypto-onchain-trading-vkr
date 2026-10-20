"""Offline 2022 -> 2023 control checks with no real research fits."""

import copy
import json
import math
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.vkr_model import attempts, controls
from src.vkr_model import train as training


class FakeControlRegressor:
    fits = []
    fail_index = None
    wrong_tree_count = False
    reload_shift = 0.0

    def __init__(self, **params):
        self.params = dict(params)
        self.loaded = False

    def fit(self, pool):
        type(self).fits.append(copy.deepcopy({"pool": pool, "params": self.params}))
        if self.fail_index == len(self.fits):
            raise ValueError("synthetic control failure")
        self.feature_names_ = list(pool["names"])
        self.tree_count_ = 299 if self.wrong_tree_count else self.params["iterations"]
        self.mean = math.fsum(pool["y"]) / len(pool["y"])

    def predict(self, pool, thread_count=1):
        if thread_count != 1 or list(pool["names"]) != self.feature_names_:
            raise AssertionError("Prediction threads or named feature order changed")
        return [self.mean + row[0] / (100 * self.params["l2_leaf_reg"])
                + (self.reload_shift if self.loaded else 0.0) for row in pool["x"]]

    def save_model(self, path, **kwargs):
        Path(path).write_text(json.dumps({"params": self.params, "mean": self.mean,
                                         "columns": self.feature_names_, "trees": self.tree_count_}))

    def load_model(self, path, **kwargs):
        saved = json.loads(Path(path).read_text())
        self.params, self.mean = saved["params"], saved["mean"]
        self.feature_names_, self.tree_count_ = saved["columns"], saved["trees"]
        self.loaded = True

    def get_all_params(self):
        result = dict(self.params)
        result["learning_rate"] = 0.05000000075 if self.loaded else 0.05000000074505806
        return result


class ControlsTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.dataset_dir = self.root / "dataset"
        self.dataset_dir.mkdir()
        self.database = self.dataset_dir / "market.sqlite3"
        self.dataset_config_path = self.root / "dataset.json"
        self.dataset_config_path.write_text(json.dumps({
            "symbol": "BTCUSDT", "start": "2021-12-01T00:00:00Z", "end": "2026-01-01T00:00:00Z",
            "dataset_dir": str(self.dataset_dir), "feature_interval": "1h", "horizon_hours": 4,
            "feature_delay_seconds": 60, "execution_delay_seconds": 120,
            "feature_windows_hours": [24, 168, 720],
            "splits": [{"name": name, "start": start, "end": end}
                       for name, (start, end) in training.SPLIT_BOUNDARIES.items()] + [
                           {"name": "final", "start": "2025-01-01T00:00:00Z", "end": "2026-01-01T00:00:00Z"}],
        }))
        self.verification = {"verified": True, "snapshot_id": "a" * 64,
                             "dataset_sha256": "b" * 64, "source_sha256": "c" * 64}
        self.build_report = {**self.verification, "feature_columns": list(training.FEATURE_COLUMNS),
                             "splits": {"train": {"rows": 12}}}
        with self.connect() as connection:
            connection.execute("CREATE TABLE metadata(key TEXT PRIMARY KEY,value TEXT)")
            connection.execute("INSERT INTO metadata VALUES ('build_report',?)", (json.dumps(self.build_report),))
            connection.execute("CREATE TABLE dataset_rows (decision_time INTEGER PRIMARY KEY,"
                               "execution_time INTEGER,label_end INTEGER,split TEXT,y REAL,"
                               + ",".join(name + " REAL" for name in training.FEATURE_COLUMNS) + ")")
            for year in (2022, 2023):
                start = training._ms(str(year) + "-01-01T00:00:00Z")
                for index in reversed(range(4)):
                    y = (index + 1) / (100 if year == 2022 else 10)
                    self.insert(connection, start + index * training.HOUR, "train", y, index + 1)
            boundary = training._ms("2023-01-01T00:00:00Z")
            for index in range(1, 5):
                self.insert(connection, boundary - index * training.HOUR, "train", 999.0, 999.0)
            for year, split in ((2024, "validation"), (2025, "final")):
                self.insert(connection, training._ms(str(year) + "-01-01T00:00:00Z"),
                            split, float("inf"), float("inf"))
        self.ledger_path = self.root / "training/attempts.jsonl"
        self.ledger_path.parent.mkdir()
        self.ledger_path.write_text("".join(json.dumps({"run_id": "legacy", "depth": depth,
                                                       "status": status}) + "\n"
                                            for depth in (4, 6) for status in ("STARTED", "SUCCESS")))
        self.legacy_bytes = self.ledger_path.read_bytes()
        self.config = {"experiment_id": "0005_catboost_controls",
                       "dataset_config": str(self.dataset_config_path), "dataset_dir": str(self.dataset_dir),
                       "artifact_dir": str(self.root / "artifacts"), "budget_ledger": str(self.ledger_path),
                       "expected_snapshot_id": "a" * 64, "expected_dataset_sha256": "b" * 64,
                       "params": dict(controls.COMMON_PARAMS), "variants": copy.deepcopy(controls.VARIANTS),
                       **{role + "_" + name: value for role, pair in controls.BOUNDARIES.items()
                          for name, value in zip(("start", "end"), pair)}}
        FakeControlRegressor.fits = []
        FakeControlRegressor.fail_index = None
        FakeControlRegressor.wrong_tree_count = False
        FakeControlRegressor.reload_shift = 0.0
        for owner, target, replacement in (
                (controls, "verify_build", lambda config: dict(self.verification)),
                (training, "_backend", lambda: FakeControlRegressor),
                (training, "_runtime", lambda: {"python": "3.12.test", "catboost": "1.2.10"}),
                (controls, "_pool", lambda x, columns, y=None: {"x": x, "y": y, "names": list(columns)})):
            mock = patch.object(owner, target, replacement)
            mock.start()
            self.addCleanup(mock.stop)

    def connect(self):
        return sqlite3.connect(self.database)

    def insert(self, connection, decision, split, y, first_feature):
        connection.execute("INSERT INTO dataset_rows VALUES (" + ",".join("?" for _ in range(23)) + ")",
                           [decision, decision + 120_000, decision + 120_000 + 4 * training.HOUR,
                            split, y] + [float(first_feature + index) for index in range(18)])

    def latest_report(self):
        path = sorted((self.root / "artifacts").glob("run-*/report.json"))[-1]
        return json.loads(path.read_text())

    def test_fixed_three_controls_only_2022_fit_and_correct_baseline(self):
        result = controls.run_controls(self.config)
        self.assertEqual(result["status"], "SUCCESS")
        self.assertEqual(result["rows"], {"inner_train": 4, "assessment": 4})
        self.assertEqual(result["selection"]["source_rows"], 12)
        self.assertEqual(result["selection"]["purged_inner_boundary_rows"], 4)
        self.assertEqual(len(result["selection"]["purged_rows"]), 4)
        self.assertAlmostEqual(result["baseline_train_mean"], 0.025)
        self.assertEqual(len(FakeControlRegressor.fits), 3)
        self.assertEqual([fit["params"]["l2_leaf_reg"] for fit in FakeControlRegressor.fits], [3, 30, 3])
        self.assertEqual([len(fit["pool"]["names"]) for fit in FakeControlRegressor.fits], [18, 18, 11])
        for fit in FakeControlRegressor.fits:
            self.assertEqual(fit["pool"]["y"], [0.01, 0.02, 0.03, 0.04])
            self.assertEqual([row[0] for row in fit["pool"]["x"]], [1, 2, 3, 4])
            self.assertEqual(fit["params"]["thread_count"], 1)
            self.assertEqual(fit["params"]["iterations"], 300)
            self.assertFalse(fit["params"]["use_best_model"])
        self.assertEqual(set(FakeControlRegressor.fits[-1]["pool"]["names"]),
                         set(training.FEATURE_COLUMNS) - controls.VOLUME_COLUMNS)
        self.assertEqual(result["accounting"]["ledger_before"]["started_attempts"], 2)
        self.assertEqual(result["accounting"]["ledger_after"]["started_attempts"], 5)
        self.assertEqual(result["accounting"]["started_in_run"], 3)
        self.assertNotIn("limit", result["accounting"])
        self.assertTrue(self.ledger_path.read_bytes().startswith(self.legacy_bytes))
        self.assertFalse(result["validation_2024_used"])
        self.assertFalse(result["final_split_used"])
        self.assertFalse(result["automatic_selection"])
        self.assertEqual(set(result["metrics"]), {"inner_train", "assessment"})
        self.assertEqual(set(result["metrics"]["assessment"]["methods"]["zero"]["quarterly"]), {"2023-Q1"})
        verified = controls.verify_controls(self.config, result["run_dir"])
        self.assertTrue(verified["verified"])
        self.assertEqual(verified["fit_calls"], 0)
        self.assertEqual(len(FakeControlRegressor.fits), 3)

    def test_purge_boundary_excludes_exact_crossing_label(self):
        data, _, selection = controls._load_data(self.config)
        boundary = training._ms("2023-01-01T00:00:00Z")
        self.assertTrue(all(row["label_end"] < boundary for row in data["inner_train"]))
        self.assertTrue(all(row["execution_time"] >= boundary for row in data["assessment"]))
        self.assertEqual(selection["purged_inner_boundary_rows"], 4)
        self.assertNotIn(999.0, [row["y"] for rows in data.values() for row in rows])

    def test_no_validation_loader_and_no_final_arrays(self):
        with patch.object(training, "_load_data", side_effect=AssertionError("2024 loader forbidden")):
            data, _, _ = controls._load_data(self.config)
        self.assertEqual(set(data), {"inner_train", "assessment"})
        self.assertTrue(all(row["source_split"] == "train" for rows in data.values() for row in rows))
        self.assertTrue(all(row["label_end"] < training._ms("2024-01-01T00:00:00Z")
                            for rows in data.values() for row in rows))

    def test_allowlist_params_and_temporal_changes_are_rejected(self):
        original = copy.deepcopy(self.config)
        for mutate in (lambda: self.config["variants"][0].update({"feature_set": "y"}),
                       lambda: self.config["params"].update({"early_stopping_rounds": 5}),
                       lambda: self.config.update({"assessment_end": "2025-01-01T00:00:00Z"})):
            self.config = copy.deepcopy(original)
            mutate()
            with self.assertRaises(ValueError):
                controls.run_controls(self.config)
        self.assertEqual(FakeControlRegressor.fits, [])

    def test_invalid_output_directory_is_rejected_before_any_mutation(self):
        forbidden = self.dataset_dir / "controls-must-not-exist"
        self.config["artifact_dir"] = str(forbidden)
        with self.assertRaisesRegex(ValueError, "inside the dataset"):
            controls.run_controls(self.config)
        self.assertFalse(forbidden.exists())

    def test_constructor_failure_counts_attempt_without_claiming_fit_call(self):
        class BrokenConstructor(FakeControlRegressor):
            def __init__(self, **params):
                raise ValueError("synthetic constructor failure")
        with patch.object(training, "_backend", lambda: BrokenConstructor):
            with self.assertRaisesRegex(RuntimeError, "constructor failure"):
                controls.run_controls(self.config)
        report = self.latest_report()
        self.assertEqual(report["accounting"]["fit_calls"], 0)
        self.assertEqual(report["accounting"]["started_in_run"], 1)
        self.assertTrue(report["dataset_immutable"])

    def test_tampered_feature_report_or_snapshot_fail_before_fit(self):
        self.build_report["feature_columns"][-1] = "y"
        with self.connect() as connection:
            connection.execute("UPDATE metadata SET value=?", (json.dumps(self.build_report),))
        with self.assertRaisesRegex(RuntimeError, "allowlist"):
            controls.run_controls(self.config)
        self.assertEqual(FakeControlRegressor.fits, [])

    def test_failed_fit_consumes_one_attempt_and_stops_without_retry(self):
        FakeControlRegressor.fail_index = 2
        with self.assertRaisesRegex(RuntimeError, "synthetic control failure"):
            controls.run_controls(self.config)
        self.assertEqual(len(FakeControlRegressor.fits), 2)
        report = self.latest_report()
        self.assertEqual(report["status"], "FAILED")
        self.assertEqual([item["status"] for item in report["attempts"]], ["SUCCESS", "FAILED"])
        self.assertEqual(report["accounting"]["started_in_run"], 2)
        self.assertEqual(attempts.state(self.ledger_path)["started_attempts"], 4)
        self.assertEqual([record["status"] for record in attempts.load(self.ledger_path)[-2:]], ["STARTED", "FAILED"])

    def test_tree_count_and_reload_predictions_are_enforced(self):
        FakeControlRegressor.wrong_tree_count = True
        with self.assertRaisesRegex(RuntimeError, "300 trees"):
            controls.run_controls(self.config)
        FakeControlRegressor.wrong_tree_count = False
        FakeControlRegressor.reload_shift = 0.1
        with self.assertRaisesRegex(RuntimeError, "reproduce predictions"):
            controls.run_controls(self.config)

    def test_verify_detects_artifact_tampering_without_fit(self):
        result = controls.run_controls(self.config)
        path = Path(result["run_dir"]) / "full_l2_3.cbm"
        path.write_bytes(path.read_bytes() + b" ")
        with self.assertRaisesRegex(ValueError, "artifact integrity"):
            controls.verify_controls(self.config, result["run_dir"])
        self.assertEqual(len(FakeControlRegressor.fits), 3)

    def test_verify_allows_later_ledger_attempts_and_does_not_mutate_it(self):
        result = controls.run_controls(self.config)
        attempts.append(self.ledger_path, "later-run", "later-variant", 4, "STARTED", "later-experiment")
        before = self.ledger_path.read_bytes()
        verified = controls.verify_controls(self.config, result["run_dir"])
        self.assertTrue(verified["verified"])
        self.assertEqual(verified["ledger_before"]["started_attempts"], 6)
        self.assertEqual(self.ledger_path.read_bytes(), before)

    def test_repeated_runs_preserve_previous_artifacts_and_same_predictions(self):
        first = controls.run_controls(self.config)
        original = (Path(first["run_dir"]) / "report.json").read_bytes()
        second = controls.run_controls(self.config)
        self.assertNotEqual(first["run_dir"], second["run_dir"])
        self.assertEqual(first["metrics"], second["metrics"])
        self.assertEqual(first["artifact_sha256"]["predictions.csv.gz"],
                         second["artifact_sha256"]["predictions.csv.gz"])
        self.assertEqual((Path(first["run_dir"]) / "report.json").read_bytes(), original)
        self.assertEqual(second["accounting"]["ledger_after"]["started_attempts"], 8)


if __name__ == "__main__":
    unittest.main()
