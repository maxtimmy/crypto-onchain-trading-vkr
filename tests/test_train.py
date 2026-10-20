"""Offline trainer checks with a fake backend; no research fits are consumed."""

import copy
import gzip
import json
import math
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.vkr_model import train as trainer


class FakeRegressor:
    fits = []
    fail_depth = None
    wrong_tree_count = False
    reload_shift = 0.0
    round_loaded_params = False
    loaded_params_override = None

    def __init__(self, **params):
        self.params = params
        self.tree_count_ = 0
        self.feature_names_ = []
        self.loaded = False

    def fit(self, pool):
        x, y, names = pool
        type(self).fits.append({"x": copy.deepcopy(x), "y": list(y),
                                "params": dict(self.params), "feature_names": names})
        if self.params["depth"] == self.fail_depth:
            raise ValueError("synthetic failed fit")
        self.tree_count_ = 299 if self.wrong_tree_count else self.params["iterations"]
        self.feature_names_ = names
        self.mean = math.fsum(y) / len(y)
        return self

    def predict(self, x, thread_count=1):
        if thread_count != 1:
            raise AssertionError("Predictions must use one thread")
        return [self.mean + row[0] * self.params["depth"] / 1000
                + (self.reload_shift if self.loaded else 0.0) for row in x]

    def save_model(self, path, **kwargs):
        Path(path).write_text(json.dumps({"params": self.params, "mean": self.mean,
                                         "feature_names": self.feature_names_,
                                         "tree_count": self.tree_count_}), encoding="utf-8")

    def load_model(self, path, **kwargs):
        record = json.loads(Path(path).read_text(encoding="utf-8"))
        self.params, self.mean = record["params"], record["mean"]
        self.feature_names_, self.tree_count_ = record["feature_names"], record["tree_count"]
        self.loaded = True
        return self

    def get_all_params(self):
        result = dict(self.params)
        # CatBoost exposes the fitted float32 value before serializing decimal
        # metadata into CBM; the test fixture mirrors the real observed values.
        result["learning_rate"] = 0.05000000074505806
        result["bayesian_matrix_reg"] = 0.10000000149011612
        result["subsample"] = 0.800000011920929
        if self.loaded and self.round_loaded_params:
            result = {key: float(format(value, ".10g")) if isinstance(value, float) else value
                      for key, value in result.items()}
        if self.loaded and self.loaded_params_override:
            result.update(self.loaded_params_override)
        return result


class TrainTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.dataset = self.root / "dataset"
        self.dataset.mkdir()
        self.database = self.dataset / "market.sqlite3"
        self.dataset_config = self.root / "dataset.json"
        self.dataset_config.write_text(json.dumps({
            "symbol": "BTCUSDT", "start": "2021-12-01T00:00:00Z", "end": "2026-01-01T00:00:00Z",
            "dataset_dir": str(self.dataset), "feature_interval": "1h", "horizon_hours": 4,
            "feature_delay_seconds": 60, "execution_delay_seconds": 120,
            "feature_windows_hours": [24, 168, 720],
            "splits": [{"name": name, "start": start, "end": end}
                       for name, (start, end) in trainer.SPLIT_BOUNDARIES.items()] + [
                           {"name": "final", "start": "2025-01-01T00:00:00Z", "end": "2026-01-01T00:00:00Z"}],
        }), encoding="utf-8")
        self.verification = {"verified": True, "snapshot_id": "a" * 64,
                             "dataset_sha256": "b" * 64, "source_sha256": "c" * 64}
        self.build_report = {**self.verification, "feature_columns": list(trainer.FEATURE_COLUMNS),
                             "splits": {"train": {"rows": 4}, "validation": {"rows": 4}}}
        with self.connect() as connection:
            connection.execute("CREATE TABLE metadata(key TEXT PRIMARY KEY,value TEXT)")
            connection.execute("INSERT INTO metadata VALUES ('build_report',?)", (json.dumps(self.build_report),))
            connection.execute("CREATE TABLE dataset_rows (decision_time INTEGER PRIMARY KEY,"
                               "execution_time INTEGER,label_end INTEGER,split TEXT,y REAL,"
                               + ",".join(name + " REAL" for name in trainer.FEATURE_COLUMNS) + ")")
            for split, (start, _) in trainer.SPLIT_BOUNDARIES.items():
                base = trainer._ms(start)
                # Insert in reverse order to exercise explicit chronological sort.
                for index in reversed(range(4)):
                    decision = base + index * trainer.HOUR
                    y = (index + 1) / 100 if split == "train" else (index + 1) / 10
                    connection.execute("INSERT INTO dataset_rows VALUES (" + ",".join("?" for _ in range(23)) + ")",
                                       [decision, decision + 120_000, decision + 120_000 + 4 * trainer.HOUR,
                                        split, y] + [float(index + feature + 1) for feature in range(18)])
            decision = trainer._ms("2025-01-01T00:00:00Z")
            # A sentinel nonfinite final row would fail input validation if read.
            connection.execute("INSERT INTO dataset_rows VALUES (" + ",".join("?" for _ in range(23)) + ")",
                               [decision, decision + 120_000, decision + 120_000 + 4 * trainer.HOUR,
                                "final", float("inf")] + [float("inf")] * 18)
        self.config = {
            "experiment_id": "0003_btcusdt_1h_4h", "dataset_config": str(self.dataset_config),
            "dataset_dir": str(self.dataset), "artifact_dir": str(self.root / "artifacts"),
            "expected_snapshot_id": "a" * 64, "expected_dataset_sha256": "b" * 64,
            "depths": [4, 6], "params": dict(trainer.FIXED_PARAMS),
            "train_split": "train", "validation_split": "validation",
            "train_start": trainer.SPLIT_BOUNDARIES["train"][0],
            "train_end": trainer.SPLIT_BOUNDARIES["train"][1],
            "validation_start": trainer.SPLIT_BOUNDARIES["validation"][0],
            "validation_end": trainer.SPLIT_BOUNDARIES["validation"][1],
        }
        FakeRegressor.fits = []
        FakeRegressor.fail_depth = None
        FakeRegressor.wrong_tree_count = False
        FakeRegressor.reload_shift = 0.0
        FakeRegressor.round_loaded_params = False
        FakeRegressor.loaded_params_override = None
        for target, replacement in (
                ("verify_build", lambda config: dict(self.verification)),
                ("_runtime", lambda: {"python": "3.12.test", "packages": {"catboost": "1.2.10"}}),
                ("_backend", lambda: FakeRegressor),
                ("_pool", lambda x, y: (x, y, list(trainer.FEATURE_COLUMNS)))):
            mock = patch.object(trainer, target, replacement)
            mock.start()
            self.addCleanup(mock.stop)

    def connect(self):
        return sqlite3.connect(self.database)

    def update_report(self):
        with self.connect() as connection:
            connection.execute("UPDATE metadata SET value=? WHERE key='build_report'",
                               (json.dumps(self.build_report),))

    def latest_report(self):
        paths = sorted((self.root / "artifacts").glob("run-*/report.json"))
        return json.loads(paths[-1].read_text(encoding="utf-8"))

    def test_exact_features_train_only_mean_two_fits_and_saved_reload(self):
        result = trainer.train(self.config)
        self.assertEqual(result["status"], "SUCCESS")
        self.assertEqual(result["rows"], {"train": 4, "validation": 4})
        self.assertAlmostEqual(result["baseline_train_mean"], 0.025)
        self.assertEqual([fit["params"]["depth"] for fit in FakeRegressor.fits], [4, 6])
        for fit in FakeRegressor.fits:
            self.assertEqual(fit["y"], [0.01, 0.02, 0.03, 0.04])
            self.assertEqual(len(fit["x"][0]), 18)
            self.assertEqual([row[0] for row in fit["x"]], [1, 2, 3, 4])
            self.assertEqual(fit["feature_names"], list(trainer.FEATURE_COLUMNS))
            self.assertEqual({key: value for key, value in fit["params"].items() if key != "depth"},
                             trainer.FIXED_PARAMS)
        self.assertEqual(result["metrics"]["zero"]["validation"]["overall"]["n"], 4)
        self.assertEqual(set(result["metrics"]["depth4"]["validation"]["quarterly"]), {"2024-Q1"})
        verification = trainer.verify_run(self.config, result["run_dir"])
        self.assertTrue(verification["verified"])
        self.assertEqual(verification["fit_calls"], 0)
        self.assertEqual(len(FakeRegressor.fits), 2)
        with gzip.open(Path(result["run_dir"]) / "predictions.csv.gz", "rt") as stream:
            text = stream.read()
        self.assertNotIn(",final,", text)
        self.assertEqual(len(text.splitlines()), 33)
        self.assertNotIn(str(self.root), (Path(result["run_dir"]) / "report.json").read_text())

    def test_tampered_feature_whitelist_cannot_include_target_or_time(self):
        for column in ("y", "execution_time", "signal_time", "unexpected_feature"):
            with self.subTest(column=column):
                self.build_report["feature_columns"] = list(trainer.FEATURE_COLUMNS[:-1]) + [column]
                self.update_report()
                with self.assertRaisesRegex(RuntimeError, "allowlist"):
                    trainer.train(self.config)
        self.assertEqual(FakeRegressor.fits, [])

    def test_temporal_split_and_label_end_boundaries_are_enforced(self):
        with self.connect() as connection:
            connection.execute("UPDATE dataset_rows SET label_end=? WHERE split='train' AND y=0.01",
                               (trainer._ms("2024-01-01T00:00:00Z"),))
        with self.assertRaisesRegex(RuntimeError, "Temporal boundary"):
            trainer.train(self.config)
        self.assertEqual(FakeRegressor.fits, [])

    def test_snapshot_is_verified_before_any_fit(self):
        self.verification["snapshot_id"] = "d" * 64
        with self.assertRaisesRegex(RuntimeError, "pinned"):
            trainer.train(self.config)
        self.assertEqual(FakeRegressor.fits, [])

    def test_no_early_stopping_or_unplanned_params(self):
        self.config["params"]["early_stopping_rounds"] = 20
        with self.assertRaisesRegex(RuntimeError, "predeclared"):
            trainer.train(self.config)
        self.assertEqual(FakeRegressor.fits, [])
        self.assertEqual(self.latest_report()["status"], "FAILED")

    def test_failed_fit_is_recorded_and_stops_the_run(self):
        FakeRegressor.fail_depth = 4
        with self.assertRaisesRegex(RuntimeError, "synthetic failed fit"):
            trainer.train(self.config)
        self.assertEqual(len(FakeRegressor.fits), 1)
        report = self.latest_report()
        self.assertEqual(report["status"], "FAILED")
        self.assertEqual(report["attempts"][0]["status"], "FAILED")
        records = [json.loads(line) for line in (self.root / "artifacts/attempts.jsonl").read_text().splitlines()]
        self.assertEqual([row["status"] for row in records], ["STARTED", "FAILED"])
        self.assertFalse(list((self.root / "artifacts").glob("run-*/manifest.json")))

    def test_tree_count_must_be_300(self):
        FakeRegressor.wrong_tree_count = True
        with self.assertRaisesRegex(RuntimeError, "exactly 300"):
            trainer.train(self.config)
        self.assertEqual(len(FakeRegressor.fits), 1)

    def test_reloaded_model_must_match_predictions(self):
        FakeRegressor.reload_shift = 0.1
        with self.assertRaisesRegex(RuntimeError, "reproduce predictions"):
            trainer.train(self.config)
        self.assertEqual(len(FakeRegressor.fits), 1)

    def test_nonfinite_train_input_is_rejected(self):
        with self.connect() as connection:
            connection.execute("UPDATE dataset_rows SET last_return=? WHERE split='train' AND y=0.01",
                               (float("inf"),))
        with self.assertRaisesRegex(RuntimeError, "finite"):
            trainer.train(self.config)
        self.assertEqual(FakeRegressor.fits, [])

    def test_repeated_training_never_overwrites_previous_artifacts(self):
        first = trainer.train(self.config)
        original = (Path(first["run_dir"]) / "report.json").read_bytes()
        second = trainer.train(self.config)
        self.assertNotEqual(first["run_dir"], second["run_dir"])
        self.assertEqual((Path(first["run_dir"]) / "report.json").read_bytes(), original)
        self.assertEqual(second["budget"]["attempts_before_run"], 2)
        self.assertTrue(trainer.verify_run(self.config)["verified"])

    def test_artifact_and_report_tampering_are_detected(self):
        result = trainer.train(self.config)
        run_dir = Path(result["run_dir"])
        model_path = run_dir / "depth4.cbm"
        original = model_path.read_bytes()
        model_path.write_bytes(original + b" ")
        with self.assertRaisesRegex(ValueError, "artifact integrity"):
            trainer.verify_run(self.config, run_dir)
        model_path.write_bytes(original)
        report_path = run_dir / "report.json"
        report_path.write_bytes(report_path.read_bytes() + b" ")
        with self.assertRaisesRegex(ValueError, "report integrity"):
            trainer.verify_run(self.config, run_dir)

    def test_verify_detects_changed_loaded_rows(self):
        result = trainer.train(self.config)
        with self.connect() as connection:
            connection.execute("UPDATE dataset_rows SET last_return=last_return+1 WHERE split='validation'")
        with self.assertRaisesRegex(ValueError, "row identity"):
            trainer.verify_run(self.config, result["run_dir"])

    def test_verify_accepts_cbm_decimal_parameter_rounding(self):
        result = trainer.train(self.config)
        self.assertTrue(trainer._same_params(
            {"learning_rate": 0.05000000074505806,
             "nested": [{"subsample": 0.800000011920929}]},
            {"learning_rate": 0.05000000075,
             "nested": [{"subsample": 0.8000000119}]}))
        FakeRegressor.round_loaded_params = True
        self.assertTrue(trainer.verify_run(self.config, result["run_dir"])["verified"])
        self.assertEqual(len(FakeRegressor.fits), 2)

    def test_verify_rejects_changed_parameter_values_and_types(self):
        result = trainer.train(self.config)
        for override in ({"learning_rate": 0.06}, {"depth": 8}, {"depth": 4.0},
                         {"use_best_model": 0}):
            with self.subTest(override=override):
                FakeRegressor.loaded_params_override = override
                with self.assertRaisesRegex(ValueError, "parameters or structure"):
                    trainer.verify_run(self.config, result["run_dir"])
        self.assertEqual(len(FakeRegressor.fits), 2)
        self.assertFalse(trainer._same_params({"a": [1.0]}, {"a": [1.0], "b": 2}))
        self.assertFalse(trainer._same_params({"a": [1.0]}, {"a": [1.0, 2.0]}))
        self.assertFalse(trainer._same_params({"a": float("inf")}, {"a": float("inf")}))

    def test_shared_budget_counts_failed_started_attempts_once(self):
        ledger = self.root / "shared/attempts.jsonl"
        ledger.parent.mkdir()
        records = []
        for index in range(19):
            records += [{"run_id": "old" + str(index), "depth": 4, "status": "STARTED"},
                        {"run_id": "old" + str(index), "depth": 4, "status": "FAILED"}]
        ledger.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")
        self.config["budget_ledger"] = str(ledger)
        with self.assertRaisesRegex(RuntimeError, "budget"):
            trainer.train(self.config)
        self.assertEqual(FakeRegressor.fits, [])
        self.assertEqual(len(ledger.read_text().splitlines()), 38)


if __name__ == "__main__":
    unittest.main()
