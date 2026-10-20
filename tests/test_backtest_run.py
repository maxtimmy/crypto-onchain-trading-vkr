"""Offline backtest orchestration; engines and quotes are small synthetic fixtures."""

import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.vkr_backtest import run as backtest
from src.vkr_model import attempts
from src.vkr_model import train as training


class DummyRegressor:
    def fit(self, *args, **kwargs):
        raise AssertionError("Backtest fit guard must intercept fit")


class BacktestRunTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.training_dir = self.root / "training/run-fixture"
        self.training_dir.mkdir(parents=True)
        self.dataset_dir = self.root / "dataset"
        self.dataset_dir.mkdir()
        self.database = self.dataset_dir / "market.sqlite3"
        self.database.write_bytes(b"offline market snapshot fixture")
        self.dataset_config = self.root / "dataset.json"
        self.dataset_config.write_text("{}")
        self.ledger = self.root / "training/attempts.jsonl"
        self.ledger.write_text("".join(json.dumps({"run_id": "legacy", "depth": depth,
                                                 "status": status}) + "\n"
                                      for depth in (4, 6) for status in ("STARTED", "SUCCESS")))
        for name in ("full_l2_3", "full_l2_30", "no_volume_l2_3"):
            for status in ("STARTED", "SUCCESS"):
                attempts.append(self.ledger, "controls-fixture", name, 4, status, "0005")
        source_artifacts = {}
        for name in ("depth4.cbm", "depth6.cbm", "predictions.csv.gz"):
            path = self.training_dir / name
            path.write_bytes(name.encode("utf-8"))
            source_artifacts[name] = training._sha_file(path)
        source_report = {"run_id": self.training_dir.name, "status": "SUCCESS", "experiment_id": "0003_btcusdt_1h_4h",
                         "snapshot_id": "a" * 64, "dataset_sha256": "b" * 64,
                         "feature_columns": list(training.FEATURE_COLUMNS), "final_split_used": False,
                         "artifact_sha256": source_artifacts}
        training._atomic_json(self.training_dir / "report.json", source_report)
        source_manifest = {"run_id": self.training_dir.name, "status": "SUCCESS",
                           "report_sha256": training._sha_file(self.training_dir / "report.json"),
                           "artifact_sha256": source_artifacts}
        training._atomic_json(self.training_dir / "manifest.json", source_manifest)
        self.config = {"experiment_id": "0006_btcusdt_backtest", "dataset_config": str(self.dataset_config),
                       "dataset_dir": str(self.dataset_dir), "training_run": str(self.training_dir),
                       "expected_report_sha256": training._sha_file(self.training_dir / "report.json"),
                       "expected_manifest_sha256": training._sha_file(self.training_dir / "manifest.json"),
                       "expected_snapshot_id": "a" * 64, "expected_dataset_sha256": "b" * 64,
                       "artifact_dir": str(self.root / "backtests"), "budget_ledger": str(self.ledger),
                       "start": backtest.START, "end": backtest.END, "initial_cash": "10000",
                       "holding_minutes": 240, "allocation": "1", "qty_step": "0.000001",
                       "min_notional": "10", "entry_threshold": backtest.ENTRY_THRESHOLD,
                       "scenarios": copy.deepcopy(backtest.SCENARIOS)}
        start = training._ms(backtest.START)
        self.inputs = {"quotes": {start + 120_000: "42000"},
                       "signals": {method: [{"decision_time": start, "signal_time": start + 60_000,
                                             "execution_time": start + 120_000, "enter": method == "depth4",
                                             "score": 0.003 if method == "depth4" else 0.0}]
                                   for method in backtest.SIGNAL_METHODS},
                       "metadata": {"fixture": True, "model_fit_calls": 0, "final_2025_used": False}}
        self.simulations = []
        self.engine_status = "COMPLETE"
        for owner, target, replacement in (
                (training, "_backend", lambda: DummyRegressor),
                (training, "_runtime", lambda: {"python": "3.12.test", "catboost": "1.2.10"}),
                (backtest, "_load_inputs", lambda config: copy.deepcopy(self.inputs)),
                (backtest, "_simulate", self.simulate), (backtest, "_plots", self.plots),
                (backtest, "_source_snapshot", self.source)):
            mock = patch.object(owner, target, replacement)
            mock.start()
            self.addCleanup(mock.stop)

    def simulate(self, signals, quotes, config, costs, mode):
        self.simulations.append(copy.deepcopy({"signals": signals, "quotes": quotes,
                                               "config": config, "costs": costs, "mode": mode}))
        method_index = (len(self.simulations) - 1) % 7
        equity = {"time": config["terminal_time"], "equity": "10000", "cash": "10000",
                  "quantity": "0", "reference_price": "42000", "stale": False, "kind": "terminal"}
        return {"status": self.engine_status,
                "metrics": {"net_return": method_index / 100, "final_equity": "10000"}
                if self.engine_status == "COMPLETE" else None,
                "trades": [], "equity_hourly": [equity], "equity_daily": [equity],
                "counters": {"signals_total": len(signals)}, "final_state": {"cash": "10000", "quantity": "0"},
                "accounting": {"fee_bps": costs["fee_bps"], "mode": mode}}

    def plots(self, summary, details, output_dir):
        self.assertEqual(set(details), {"base", "stress"})
        self.assertTrue(all(len(methods) == 7 for methods in details.values()))
        (output_dir / "equity.png").write_bytes(b"offline PNG fixture")
        return ["equity.png"]

    def source(self, run_dir, config):
        (run_dir / "source").mkdir()
        path = run_dir / "source/fixture.py"
        path.write_text("# backtest source fixture\n")
        training._atomic_json(run_dir / "config.json", backtest._portable(config))
        return {"fixture.py": training._sha_file(path)}

    def latest_report(self):
        path = sorted((self.root / "backtests").glob("run-*/report.json"))[-1]
        return json.loads(path.read_text())

    def test_exact_fourteen_simulations_correct_modes_and_immutable_inputs(self):
        before = backtest._input_state(self.config)
        report = backtest.run_backtest(self.config)
        self.assertEqual(report["status"], "SUCCESS")
        self.assertEqual(len(self.simulations), 14)
        self.assertEqual([item["mode"] for item in self.simulations[:7]],
                         ["fixed_horizon"] * 5 + ["cash", "buy_hold"])
        for item in self.simulations:
            self.assertIsInstance(item["config"]["start_time"], int)
            self.assertEqual(item["config"]["terminal_time"], training._ms(backtest.END) - 60_000)
            self.assertLess(item["config"]["terminal_time"], training._ms(backtest.END))
        self.assertEqual(self.simulations[6]["signals"][0]["execution_time"],
                         training._ms(backtest.START) + 120_000)
        self.assertEqual([item["signals"] for item in self.simulations[:5]],
                         [item["signals"] for item in self.simulations[7:12]])
        self.assertEqual(report["fit_calls"], 0)
        self.assertEqual(report["simulation_calls"], 14)
        self.assertFalse(report["final_2025_used"])
        self.assertFalse(report["selection_performed"])
        self.assertTrue(report["input_immutability_verified"])
        self.assertEqual(before, report["inputs_after"])
        self.assertEqual(before["budget_ledger"]["started_attempts"], 5)
        self.assertNotIn(str(self.root), (Path(report["run_dir"]) / "report.json").read_text())
        detail_paths = {relative for files in report["detail_files"].values() for relative in files.values()}
        self.assertEqual(len(detail_paths), 14)
        self.assertTrue(detail_paths.issubset(report["artifact_sha256"]))
        self.assertTrue({"summary.json", "inputs_metadata.json", "config.json", "equity.png", "source/fixture.py"}
                        .issubset(report["artifact_sha256"]))
        for scenario, files in report["detail_files"].items():
            for method, relative in files.items():
                detail = backtest._read_gzip_json(Path(report["run_dir"]) / relative)
                self.assertEqual(backtest._semantic_hash(detail), report["detail_semantic_sha256"][scenario][method])

    def test_verify_repeats_fourteen_results_without_fitting_or_mutations(self):
        report = backtest.run_backtest(self.config)
        original = backtest._tree_hashes(Path(report["run_dir"]))
        verified = backtest.verify_backtest(self.config, report["run_dir"])
        self.assertTrue(verified["verified"])
        self.assertTrue(verified["comparison_complete"])
        self.assertEqual(verified["simulations_repeated"], 14)
        self.assertEqual(verified["fit_calls"], 0)
        self.assertEqual(len(self.simulations), 28)
        self.assertEqual(backtest._tree_hashes(Path(report["run_dir"])), original)
        self.assertEqual(verified["inputs_before"], verified["inputs_after"])

    def test_incomplete_engine_results_are_saved_and_remain_incomplete(self):
        self.engine_status = "INCOMPLETE"
        report = backtest.run_backtest(self.config)
        self.assertEqual(report["status"], "INCOMPLETE")
        self.assertFalse(report["comparison_complete"])
        self.assertTrue((Path(report["run_dir"]) / "manifest.json").exists())
        summary = json.loads((Path(report["run_dir"]) / "summary.json").read_text())
        self.assertIsNone(summary["scenarios"]["base"]["depth4"]["metrics"])
        self.assertTrue(backtest.verify_backtest(self.config, report["run_dir"])["verified"])

    def test_source_report_manifest_and_model_tampering_fail_before_loader(self):
        for name in ("report.json", "manifest.json", "depth4.cbm"):
            with self.subTest(name=name):
                path = self.training_dir / name
                original = path.read_bytes()
                path.write_bytes(original + b" ")
                with patch.object(backtest, "_load_inputs") as loader:
                    with self.assertRaisesRegex(RuntimeError, "integrity mismatch"):
                        backtest.run_backtest(self.config)
                    loader.assert_not_called()
                path.write_bytes(original)
        self.assertEqual(self.simulations, [])

    def test_fit_guard_blocks_attempt_and_records_failure(self):
        def unsafe_loader(config):
            DummyRegressor().fit([[0]], [0])
        with patch.object(backtest, "_load_inputs", unsafe_loader):
            with self.assertRaisesRegex(RuntimeError, "fit is forbidden"):
                backtest.run_backtest(self.config)
        report = self.latest_report()
        self.assertEqual(report["status"], "FAILED")
        self.assertEqual(report["fit_calls"], 1)
        self.assertTrue(report["input_immutability_verified"])
        self.assertEqual(self.simulations, [])

    def test_invalid_fixed_settings_rejected_before_output_mutation(self):
        original = copy.deepcopy(self.config)
        for mutate in (lambda: self.config.update({"end": "2026-01-01T00:00:00Z"}),
                       lambda: self.config.update({"entry_threshold": "0"}),
                       lambda: self.config["scenarios"]["stress"].update({"fee_bps": 0}),
                       lambda: self.config.update({"artifact_dir": str(self.dataset_dir / "forbidden")})):
            self.config = copy.deepcopy(original)
            mutate()
            with self.assertRaises(ValueError):
                backtest.run_backtest(self.config)
        self.assertFalse((self.root / "backtests").exists())
        self.assertFalse((self.dataset_dir / "forbidden").exists())

    def test_2025_signal_rejected_before_any_simulation(self):
        self.inputs["signals"]["depth4"][0]["execution_time"] = training._ms(backtest.END)
        with self.assertRaisesRegex(RuntimeError, "Forbidden period"):
            backtest.run_backtest(self.config)
        self.assertEqual(self.simulations, [])

    def test_mismatched_signal_time_keys_or_threshold_entries_are_rejected(self):
        original = copy.deepcopy(self.inputs)
        self.inputs["signals"]["depth4"] = []
        with self.assertRaisesRegex(RuntimeError, "identical nonempty"):
            backtest.run_backtest(self.config)
        self.inputs = copy.deepcopy(original)
        self.inputs["signals"]["zero"][0]["enter"] = True
        with self.assertRaisesRegex(RuntimeError, "fixed base-cost threshold"):
            backtest.run_backtest(self.config)
        self.assertEqual(self.simulations, [])

    def test_verify_rejects_changed_loader_metadata(self):
        report = backtest.run_backtest(self.config)
        self.inputs["metadata"]["fixture"] = "changed"
        with self.assertRaisesRegex(ValueError, "loader metadata differs"):
            backtest.verify_backtest(self.config, report["run_dir"])
        self.assertEqual(len(self.simulations), 14)

    def test_repeated_runs_have_equal_semantic_summary_and_preserve_previous_run(self):
        first = backtest.run_backtest(self.config)
        original = backtest._tree_hashes(Path(first["run_dir"]))
        second = backtest.run_backtest(self.config)
        self.assertNotEqual(first["run_dir"], second["run_dir"])
        self.assertEqual(first["summary_sha256"], second["summary_sha256"])
        self.assertEqual(first["detail_semantic_sha256"], second["detail_semantic_sha256"])
        self.assertEqual(backtest._tree_hashes(Path(first["run_dir"])), original)

    def test_verify_detects_tampered_details_before_resimulation(self):
        report = backtest.run_backtest(self.config)
        path = Path(report["run_dir"]) / report["detail_files"]["base"]["depth4"]
        path.write_bytes(path.read_bytes() + b" ")
        with self.assertRaisesRegex(ValueError, "artifact integrity"):
            backtest.verify_backtest(self.config, report["run_dir"])
        self.assertEqual(len(self.simulations), 14)

    def test_engine_failure_is_durable_and_simulation_attempts_are_counted(self):
        def failed_engine(*args):
            raise ValueError("synthetic simulation error")
        with patch.object(backtest, "_simulate", failed_engine):
            with self.assertRaisesRegex(RuntimeError, "synthetic simulation error"):
                backtest.run_backtest(self.config)
        report = self.latest_report()
        self.assertEqual(report["status"], "FAILED")
        self.assertEqual(report["simulation_calls"], 1)
        self.assertTrue(report["input_immutability_verified"])

    def test_changed_ledger_is_detected_and_later_legal_records_can_be_verified(self):
        report = backtest.run_backtest(self.config)
        attempts.append(self.ledger, "later", "later-variant", 4, "STARTED", "later-exp")
        before = self.ledger.read_bytes()
        self.assertTrue(backtest.verify_backtest(self.config, report["run_dir"])["verified"])
        self.assertEqual(self.ledger.read_bytes(), before)
        def mutating_loader(config):
            self.ledger.write_bytes(self.ledger.read_bytes() + b"\n")
            return copy.deepcopy(self.inputs)
        with patch.object(backtest, "_load_inputs", mutating_loader):
            with self.assertRaisesRegex(RuntimeError, "changed an input"):
                backtest.run_backtest(self.config)
        self.assertFalse(self.latest_report()["input_immutability_verified"])


if __name__ == "__main__":
    unittest.main()
