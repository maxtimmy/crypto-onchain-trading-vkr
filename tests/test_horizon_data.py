"""Offline paired-label checks with no research fits or network calls."""

import json
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.vkr_model import horizon_data as data
from src.vkr_model import train as training


class HorizonDataTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.database = self.root / "market.sqlite3"
        self.verification = {"verified": True, "snapshot_id": "a" * 64,
                             "dataset_sha256": "b" * 64, "source_sha256": "c" * 64}
        self.config = {"dataset_config": str(self.root / "dataset.json"), "dataset_dir": str(self.root),
                       "expected_snapshot_id": "a" * 64, "expected_dataset_sha256": "b" * 64}
        dataset_config = json.loads(Path("configs/btcusdt_1h_4h.json").read_text())
        self.root.joinpath("dataset.json").write_text(json.dumps(dataset_config))
        report = {**self.verification, "feature_columns": list(training.FEATURE_COLUMNS),
                  "splits": {"train": {"rows": 8}}}
        with self.connect() as connection:
            connection.execute("CREATE TABLE metadata(key TEXT PRIMARY KEY,value TEXT)")
            connection.execute("INSERT INTO metadata VALUES ('build_report',?)", (json.dumps(report),))
            connection.execute("CREATE TABLE dataset_rows (decision_time INTEGER PRIMARY KEY,"
                "signal_time INTEGER,execution_time INTEGER,label_end INTEGER,split TEXT,y REAL,"
                + ",".join(name + " REAL" for name in training.FEATURE_COLUMNS) + ")")
            connection.execute("CREATE TABLE candles (symbol TEXT,interval TEXT,open_time INTEGER PRIMARY KEY,"
                "open TEXT,high TEXT,low TEXT,close TEXT,volume TEXT,close_time INTEGER,quote_volume TEXT,"
                "trade_count INTEGER,taker_base_volume TEXT,taker_quote_volume TEXT,quality_flags TEXT)")
            for year in (2022, 2023):
                start = training._ms(str(year) + "-01-01T00:00:00Z")
                for decision in (start, start + training.HOUR,
                                 training._ms(str(year) + "-12-31T00:00:00Z"),
                                 training._ms(str(year) + "-12-31T19:00:00Z")):
                    self.insert_source(connection, decision)
            for year, split in ((2024, "validation"), (2025, "final")):
                decision = training._ms(str(year) + "-01-01T00:00:00Z")
                self.insert_source(connection, decision, split=split, bad=True)
        mock = patch.object(data.builder, "verify_build", lambda config: dict(self.verification))
        mock.start()
        self.addCleanup(mock.stop)

    def connect(self):
        return sqlite3.connect(self.database)

    @staticmethod
    def price(timestamp):
        return 100 + timestamp // training.HOUR % 100

    def insert_source(self, connection, decision, split="train", bad=False):
        execution = decision + 120_000
        target = self.price(execution + 4 * training.HOUR) / self.price(execution) - 1
        connection.execute("INSERT INTO dataset_rows VALUES (" + ",".join("?" * 24) + ")",
            [decision, decision + 60_000, execution, execution + 4 * training.HOUR,
             split, float("inf") if bad else target] + [float("inf") if bad else index + 1.0 for index in range(18)])
        for offset in (0, 4, 24):
            timestamp = execution + offset * training.HOUR
            price = self.price(timestamp)
            connection.execute("INSERT OR IGNORE INTO candles VALUES (" + ",".join("?" * 14) + ")",
                ["BTCUSDT", "1m", timestamp, str(price), str(price + 2), str(price - 2), str(price),
                 "1", timestamp + 59_999, "100", 1, "0.5", "50", "[]"])

    def test_common_masks_target_formula_and_delays(self):
        selected, verification, selection = data.load_horizon_data(self.config)
        self.assertEqual(selection["rows"], {"inner_train": 2, "assessment": 2})
        self.assertEqual(selection["source_rows"], 8)
        self.assertEqual(selection["exclusion_count"], 4)
        self.assertEqual(len(selection["calendar_exclusions"]), 4)
        self.assertEqual(selection["label_exclusions"], [])
        self.assertEqual(verification, self.verification)
        for role, rows in selected.items():
            for row in rows:
                execution = row["execution_time"]
                self.assertEqual(row["signal_time"], row["decision_time"] + 60_000)
                self.assertEqual(row["execution_time"], row["decision_time"] + 120_000)
                self.assertEqual(row["split"], role)
                for horizon, hours in (("h4", 4), ("h24", 24)):
                    target = row["targets"][horizon]
                    self.assertEqual(target["label_end"], execution + hours * training.HOUR)
                    self.assertEqual(target["y"], self.price(target["label_end"]) / self.price(execution) - 1)
        self.assertLess(selection["price_requests"]["last_requested_time"], training._ms("2024-01-01T00:00:00Z"))

    def test_missing_24h_price_removes_both_targets_without_interpolation(self):
        timestamp = training._ms("2023-01-01T00:02:00Z") + 24 * training.HOUR
        with self.connect() as connection:
            connection.execute("DELETE FROM candles WHERE open_time=?", (timestamp,))
        selected, _, selection = data.load_horizon_data(self.config)
        self.assertEqual(len(selected["assessment"]), 1)
        self.assertEqual(selection["label_exclusions"][0]["endpoints"], [
            {"endpoint": "h24", "open_time": timestamp, "error": "missing_minute"}])

    def test_invalid_24h_candle_excludes_common_row(self):
        timestamp = training._ms("2023-01-01T00:02:00Z") + 24 * training.HOUR
        with self.connect() as connection:
            connection.execute("UPDATE candles SET close='NaN' WHERE open_time=?", (timestamp,))
        _, _, selection = data.load_horizon_data(self.config)
        self.assertEqual(selection["rows"]["assessment"], 1)
        self.assertEqual(selection["label_exclusions"][0]["endpoints"][0]["error"], "nonfinite_numeric_field")

    def test_original_target_mismatch_stops(self):
        with self.connect() as connection:
            connection.execute("UPDATE dataset_rows SET y=42 WHERE decision_time=?",
                               (training._ms("2023-01-01T00:00:00Z"),))
        with self.assertRaisesRegex(ValueError, "4h target differs"):
            data.load_horizon_data(self.config)

    def test_repeated_selection_same_hashes_and_database_bytes(self):
        before = self.database.read_bytes()
        first = data.load_horizon_data(self.config)
        second = data.load_horizon_data(self.config)
        self.assertEqual(first, second)
        self.assertEqual(before, self.database.read_bytes())

    def test_pin_and_source_allowlist_are_checked(self):
        self.verification["snapshot_id"] = "d" * 64
        with self.assertRaisesRegex(ValueError, "pinned"):
            data.load_horizon_data(self.config)
        self.verification["snapshot_id"] = "a" * 64
        with self.connect() as connection:
            raw = connection.execute("SELECT value FROM metadata").fetchone()[0]
            report = json.loads(raw)
            report["feature_columns"] = list(reversed(report["feature_columns"]))
            connection.execute("UPDATE metadata SET value=?", (json.dumps(report),))
        with self.assertRaisesRegex(ValueError, "allowlist"):
            data.load_horizon_data(self.config)

    def test_exact_boundary_is_purged_before_price_requests(self):
        end = training._ms("2023-01-01T00:00:00Z")
        row = {"decision_time": end - 24 * training.HOUR - 120_000,
               "execution_time": end - 24 * training.HOUR,
               "label_end": end - 20 * training.HOUR}
        candidates, exclusions = data._calendar_candidates([row])
        self.assertFalse(candidates["inner_train"])
        self.assertEqual(exclusions[0]["reason"], "common_24h_boundary")

    def test_later_split_poison_is_not_loaded(self):
        selected, _, _ = data.load_horizon_data(self.config)
        self.assertEqual({row["source_split"] for rows in selected.values() for row in rows}, {"train"})
        self.assertEqual({row["last_return"] for rows in selected.values() for row in rows}, {1.0})


if __name__ == "__main__":
    unittest.main()
