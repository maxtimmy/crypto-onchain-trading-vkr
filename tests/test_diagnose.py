"""Offline diagnostics orchestration checks; no real fitting or API requests."""

import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.vkr_model import diagnose
from src.vkr_model import train as training


class DummyRegressor:
    def fit(self, *args, **kwargs):
        raise AssertionError("The fit guard must intercept this call")


class DiagnoseTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.training_dir = self.root / "training/run-fixture"
        self.training_dir.mkdir(parents=True)
        self.dataset_dir = self.root / "dataset"
        self.dataset_dir.mkdir()
        self.database = self.dataset_dir / "market.sqlite3"
        self.database.write_bytes(b"unchanged dataset fixture")
        self.ledger = self.root / "training/attempts.jsonl"
        self.ledger.write_text("".join(json.dumps({"run_id": "run-fixture", "depth": depth,
                                                  "status": status}) + "\n"
                                       for depth in (4, 6) for status in ("STARTED", "SUCCESS")),
                               encoding="utf-8")
        self.training_config_path = self.root / "training.json"
        self.training_config = {"dataset_config": str(self.root / "dataset.json"),
                                "dataset_dir": str(self.dataset_dir),
                                "artifact_dir": str(self.training_dir.parent),
                                "budget_ledger": str(self.ledger)}
        self.training_config_path.write_text(json.dumps(self.training_config), encoding="utf-8")
        self.source_report = {"status": "SUCCESS", "run_id": self.training_dir.name,
                              "final_split_used": False, "selected_rows_sha256": "d" * 64,
                              "baseline_train_mean": 0.015}
        artifacts = {}
        for name in ("depth4.cbm", "depth6.cbm", "predictions.csv.gz"):
            path = self.training_dir / name
            path.write_bytes(name.encode("utf-8"))
            artifacts[name] = training._sha_file(path)
        self.source_report["artifact_sha256"] = artifacts
        training._atomic_json(self.training_dir / "report.json", self.source_report)
        self.source_manifest = {"status": "SUCCESS", "run_id": self.training_dir.name,
                                "report_sha256": training._sha_file(self.training_dir / "report.json"),
                                "artifact_sha256": artifacts}
        training._atomic_json(self.training_dir / "manifest.json", self.source_manifest)
        self.config = {"experiment_id": "0004_catboost_diagnostics",
                       "training_config": str(self.training_config_path),
                       "training_run": str(self.training_dir),
                       "expected_report_sha256": training._sha_file(self.training_dir / "report.json"),
                       "expected_manifest_sha256": training._sha_file(self.training_dir / "manifest.json"),
                       "artifact_dir": str(self.root / "diagnostics")}
        self.data = {}
        for split, (start, _) in training.SPLIT_BOUNDARIES.items():
            self.data[split] = []
            for index in range(2):
                decision = training._ms(start) + index * training.HOUR
                self.data[split].append({"decision_time": decision, "execution_time": decision + 120_000,
                                         "label_end": decision + 120_000 + 4 * training.HOUR,
                                         "split": split, "y": (index + 1) * 0.01,
                                         **{name: 0.01 for name in training.FEATURE_COLUMNS}})
        self.predictions = {method: {split: [0.0, 0.0] for split in self.data}
                            for method in training.METHODS}
        self.verification = {"verified": True, "run_id": self.training_dir.name,
                             "run_dir": str(self.training_dir), "snapshot_id": "a" * 64,
                             "dataset_sha256": "b" * 64, "selected_rows_sha256": "d" * 64,
                             "rows": {"train": 2, "validation": 2}, "fit_calls": 0,
                             "final_split_used": False}
        self.dataset_verification = {"snapshot_id": "a" * 64, "dataset_sha256": "b" * 64,
                                     "source_sha256": "c" * 64}
        for target, replacement in (
                ("_backend", lambda: DummyRegressor),
                ("_runtime", lambda: {"python": "3.12.test", "packages": {"catboost": "1.2.10"}}),
                ("verify_run", lambda *args, **kwargs: copy.deepcopy(self.verification)),
                ("_load_data", lambda *args: (copy.deepcopy(self.data), dict(self.dataset_verification), "d" * 64)),
                ("_read_predictions", lambda *args: copy.deepcopy(self.predictions))):
            mock = patch.object(training, target, replacement)
            mock.start()
            self.addCleanup(mock.stop)
        for target, replacement in (("_analysis", self.analysis), ("_plots", self.plots),
                                    ("_snapshot_source", self.source)):
            mock = patch.object(diagnose, target, replacement)
            mock.start()
            self.addCleanup(mock.stop)

    def analysis(self, data, predictions, baseline_train_mean):
        self.assertEqual(set(data), {"train", "validation"})
        self.assertEqual(set(predictions), set(training.METHODS))
        self.assertEqual(baseline_train_mean, 0.015)
        return {"schema_version": "fixture-v1", "rows": {split: len(rows) for split, rows in data.items()},
                "deterministic_value": 0.015}

    def plots(self, summary, output_dir):
        self.assertNotIn("run_dir", summary)
        names = ["monthly_errors.png", "forecast_quantiles.png"]
        for name in names:
            (output_dir / name).write_bytes(b"offline plot fixture")
        return names

    def source(self, run_dir, config, training_config):
        (run_dir / "source").mkdir()
        path = run_dir / "source/fixture.py"
        path.write_text("# source fixture\n", encoding="utf-8")
        training._atomic_json(run_dir / "config.json", diagnose._portable_config(config))
        return {"fixture.py": training._sha_file(path)}

    def latest_report(self):
        path = sorted((self.root / "diagnostics").glob("run-*/report.json"))[-1]
        return json.loads(path.read_text(encoding="utf-8"))

    def test_success_pins_provenance_zero_fits_and_inputs_unchanged(self):
        before = diagnose._input_state(self.config, self.training_config)
        result = diagnose.run_diagnostics(self.config)
        self.assertEqual(result["status"], "SUCCESS")
        self.assertEqual(result["fit_calls"], 0)
        self.assertFalse(result["final_split_used"])
        self.assertTrue(result["input_immutability_verified"])
        self.assertEqual(result["inputs_before"], before)
        self.assertEqual(result["inputs_after"], before)
        self.assertEqual(result["inputs_before"]["budget_ledger"]["started_attempts"], 2)
        self.assertEqual(result["dataset"]["selected_rows_sha256"], "d" * 64)
        self.assertEqual(set(result["training_artifact_sha256"]), {
            "report.json", "manifest.json", "depth4.cbm", "depth6.cbm", "predictions.csv.gz"})
        run_dir = Path(result["run_dir"])
        self.assertNotIn(str(self.root), (run_dir / "report.json").read_text())
        manifest = json.loads((run_dir / "manifest.json").read_text())
        self.assertEqual(manifest["report_sha256"], training._sha_file(run_dir / "report.json"))
        for relative, expected in manifest["artifact_sha256"].items():
            self.assertEqual(training._sha_file(run_dir / relative), expected)
        self.assertEqual(diagnose._input_state(self.config, self.training_config), before)

    def test_report_manifest_and_model_tamper_fail_before_verification(self):
        for name in ("report.json", "manifest.json", "depth4.cbm"):
            with self.subTest(name=name):
                path = self.training_dir / name
                original = path.read_bytes()
                path.write_bytes(original + b" ")
                with patch.object(training, "verify_run") as verify:
                    with self.assertRaisesRegex(RuntimeError, "integrity mismatch"):
                        diagnose.run_diagnostics(self.config)
                    verify.assert_not_called()
                self.assertEqual(self.latest_report()["status"], "FAILED")
                path.write_bytes(original)

    def test_unknown_config_keys_and_unpinned_hash_are_rejected(self):
        self.config["new_fit"] = True
        with self.assertRaisesRegex(ValueError, "configuration keys"):
            diagnose.run_diagnostics(self.config)
        self.config.pop("new_fit")
        self.config["expected_report_sha256"] = "not pinned"
        with self.assertRaisesRegex(ValueError, "pinned SHA"):
            diagnose.run_diagnostics(self.config)

    def test_fit_guard_blocks_attempt_and_failure_is_durable(self):
        def accidental_fit(*args, **kwargs):
            DummyRegressor().fit([[0.0]], [0.0])
        with patch.object(training, "verify_run", accidental_fit):
            with self.assertRaisesRegex(RuntimeError, "fit is forbidden"):
                diagnose.run_diagnostics(self.config)
        result = self.latest_report()
        self.assertEqual(result["status"], "FAILED")
        self.assertEqual(result["fit_calls"], 1)
        self.assertTrue(result["input_immutability_verified"])
        self.assertEqual(result["inputs_after"]["budget_ledger"]["started_attempts"], 2)

    def test_final_rows_and_extra_prediction_methods_are_rejected(self):
        self.data["final"] = [self.data["validation"][0]]
        with self.assertRaisesRegex(RuntimeError, "train/validation"):
            diagnose.run_diagnostics(self.config)
        self.data.pop("final")
        self.data["validation"][0]["split"] = "final"
        with self.assertRaisesRegex(RuntimeError, "Forbidden final"):
            diagnose.run_diagnostics(self.config)
        self.data["validation"][0]["split"] = "validation"
        self.predictions["new_model"] = self.predictions["depth4"]
        with self.assertRaisesRegex(RuntimeError, "four fixed methods"):
            diagnose.run_diagnostics(self.config)

    def test_selected_row_identity_must_match_verified_run(self):
        self.verification["selected_rows_sha256"] = "e" * 64
        with self.assertRaisesRegex(RuntimeError, "row identity"):
            diagnose.run_diagnostics(self.config)

    def test_missing_required_training_manifest_key_is_rejected(self):
        self.source_manifest["artifact_sha256"].pop("depth6.cbm")
        self.source_report["artifact_sha256"] = self.source_manifest["artifact_sha256"]
        training._atomic_json(self.training_dir / "report.json", self.source_report)
        self.source_manifest["report_sha256"] = training._sha_file(self.training_dir / "report.json")
        training._atomic_json(self.training_dir / "manifest.json", self.source_manifest)
        self.config["expected_report_sha256"] = training._sha_file(self.training_dir / "report.json")
        self.config["expected_manifest_sha256"] = training._sha_file(self.training_dir / "manifest.json")
        with self.assertRaisesRegex(RuntimeError, "missing required model"):
            diagnose.run_diagnostics(self.config)

    def test_plots_cannot_change_summary_or_escape_run_directory(self):
        def mutating_plot(summary, output_dir):
            summary["unexpected"] = True
            return self.plots(summary, output_dir)
        with patch.object(diagnose, "_plots", mutating_plot):
            with self.assertRaisesRegex(RuntimeError, "changed the diagnostic summary"):
                diagnose.run_diagnostics(self.config)
        with patch.object(diagnose, "_plots", lambda *args: [str(self.database)]):
            with self.assertRaisesRegex(RuntimeError, "inside the diagnostic run"):
                diagnose.run_diagnostics(self.config)

    def test_repeat_is_identical_and_never_overwrites_previous_run(self):
        first = diagnose.run_diagnostics(self.config)
        original_files = diagnose._tree_hashes(Path(first["run_dir"]))
        second = diagnose.run_diagnostics(self.config)
        self.assertNotEqual(first["run_dir"], second["run_dir"])
        self.assertEqual(first["summary_sha256"], second["summary_sha256"])
        self.assertEqual(second["reproducibility"]["previous_matching_runs"], 1)
        self.assertTrue(second["reproducibility"]["summary_identity_verified"])
        self.assertEqual(diagnose._tree_hashes(Path(first["run_dir"])), original_files)

    def test_repeat_with_changed_summary_is_rejected(self):
        first = diagnose.run_diagnostics(self.config)
        with patch.object(diagnose, "_analysis", lambda *args: {"changed": True}):
            with self.assertRaisesRegex(RuntimeError, "summary differs"):
                diagnose.run_diagnostics(self.config)
        self.assertTrue((Path(first["run_dir"]) / "manifest.json").is_file())
        self.assertEqual(self.latest_report()["status"], "FAILED")

    def test_input_database_or_ledger_mutation_is_detected(self):
        original_db, original_ledger = self.database.read_bytes(), self.ledger.read_bytes()
        for path in (self.database, self.ledger):
            with self.subTest(path=path.name):
                def unsafe_analysis(*args):
                    if path == self.ledger:
                        path.write_bytes(original_ledger + b'\n')
                    else:
                        path.write_bytes(b"changed input fixture")
                    return {"fixture": True}
                with patch.object(diagnose, "_analysis", unsafe_analysis):
                    with self.assertRaisesRegex(RuntimeError, "changed an input"):
                        diagnose.run_diagnostics(self.config)
                result = self.latest_report()
                self.assertFalse(result["input_immutability_verified"])
                self.database.write_bytes(original_db)
                self.ledger.write_bytes(original_ledger)

    def test_output_cannot_be_inside_training_or_dataset_inputs(self):
        for directory in (self.training_dir, self.dataset_dir / "diagnostics"):
            with self.subTest(directory=directory.name):
                self.config["artifact_dir"] = str(directory)
                with self.assertRaisesRegex(ValueError, "must not be inside"):
                    diagnose.run_diagnostics(self.config)


if __name__ == "__main__":
    unittest.main()
