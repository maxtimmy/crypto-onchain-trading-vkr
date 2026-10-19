"""Offline checks for label timing, feature availability and snapshot integrity."""

import json
import math
import sqlite3
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

from src.vkr_data.build import HOUR, MINUTE, build, verify_build


def utc(timestamp):
    return datetime.fromtimestamp(timestamp / 1000, timezone.utc).isoformat().replace("+00:00", "Z")


class BuildTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "market.sqlite3"
        self.t = int(datetime(2022, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)
        self.start, self.end = self.t - 6 * HOUR, self.t + 14 * HOUR
        self.config = {
            "dataset_dir": self.directory.name, "symbol": "BTCUSDT",
            "start": utc(self.start), "end": utc(self.end), "feature_interval": "1h",
            "horizon_hours": 4, "feature_delay_seconds": 60,
            "execution_delay_seconds": 120, "feature_windows_hours": [2, 3],
            "splits": [
                {"name": "train", "start": utc(self.t), "end": utc(self.t + 8 * HOUR)},
                {"name": "validation", "start": utc(self.t + 8 * HOUR), "end": utc(self.end)},
            ], "require_complete": False,
        }
        with self.connect() as connection:
            connection.execute("""CREATE TABLE candles (
                symbol TEXT, interval TEXT, open_time INTEGER,
                open TEXT, high TEXT, low TEXT, close TEXT, volume TEXT,
                close_time INTEGER, quote_volume TEXT, trade_count INTEGER,
                taker_base_volume TEXT, taker_quote_volume TEXT, quality_flags TEXT,
                PRIMARY KEY(symbol,interval,open_time))""")
            connection.execute("CREATE TABLE metadata(key TEXT PRIMARY KEY,value TEXT)")
            records = []
            for index, timestamp in enumerate(range(self.start, self.end, HOUR)):
                price, volume = 100 + index, 10 + index
                records.append(("BTCUSDT", "1h", timestamp, str(price), str(price + 3),
                                str(price - 1), str(price + 1), str(volume), timestamp + HOUR - 1,
                                str(volume * 100), 1, str(volume / 2), str(volume * 50), "[]"))
            for index, timestamp in enumerate(range(self.start, self.end, MINUTE)):
                price = 1000 + index
                records.append(("BTCUSDT", "1m", timestamp, str(price), str(price + 1),
                                str(price - 1), str(price + 0.5), "2", timestamp + MINUTE - 1,
                                "200", 1, "1", "100", "[]"))
            connection.executemany("INSERT INTO candles VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)", records)

    def connect(self):
        connection = sqlite3.connect(str(self.path))
        connection.row_factory = sqlite3.Row
        return connection

    def row(self, table="dataset_rows", decision=None):
        with self.connect() as connection:
            return connection.execute("SELECT * FROM %s WHERE decision_time=?" % table,
                                      (self.t if decision is None else decision,)).fetchone()

    def test_explicit_feature_and_target_formula(self):
        report = build(self.config)
        row = self.row()
        self.assertEqual(row["decision_time"], self.t)
        self.assertEqual(row["signal_time"], self.t + MINUTE)
        self.assertEqual(row["execution_time"], self.t + 2 * MINUTE)
        self.assertEqual(row["label_end"], self.t + 4 * HOUR + 2 * MINUTE)
        self.assertAlmostEqual(row["y"], 1602 / 1362 - 1)
        self.assertAlmostEqual(row["last_return"], 106 / 105 - 1)
        self.assertAlmostEqual(row["last_range"], 108 / 104 - 1)
        self.assertEqual(row["last_base_volume"], 15)
        self.assertAlmostEqual(row["return_3h"], 106 / 103 - 1)
        self.assertAlmostEqual(row["realized_volatility_3h"], math.sqrt(
            math.log(104 / 103) ** 2 + math.log(105 / 104) ** 2 + math.log(106 / 105) ** 2))
        self.assertEqual(row["base_volume_3h"], 42)
        self.assertEqual(row["quote_volume_3h"], 4200)
        self.assertAlmostEqual(row["range_3h"], 108 / 102 - 1)
        self.assertEqual(report["rows"], 6)
        self.assertEqual(report["candidate_rows"], 14)
        self.assertEqual(report["excluded_rows"], 8)
        self.assertEqual(report["exclusion_reasons"], {"split_boundary": 8})
        self.assertTrue(verify_build(self.config)["verified"])

    def test_future_hour_cannot_change_features_at_earlier_decision(self):
        first = build(self.config)
        columns = first["feature_columns"]
        before = [self.row()[column] for column in columns]
        with self.connect() as connection:
            connection.execute("""UPDATE candles SET
                open=CAST(open AS REAL)*10, high=CAST(high AS REAL)*10,
                low=CAST(low AS REAL)*10, close=CAST(close AS REAL)*10,
                volume=CAST(volume AS REAL)*7
                WHERE interval='1h' AND open_time>=?""", (self.t,))
        second = build(self.config)
        after = [self.row()[column] for column in columns]
        self.assertEqual(before, after)
        self.assertNotEqual(first["source_sha256"], second["source_sha256"])
        self.assertNotEqual(first["snapshot_id"], second["snapshot_id"])

    def test_missing_denominator_bar_invalidates_return_window(self):
        # The 3-hour return also needs the close of the preceding fourth bar.
        with self.connect() as connection:
            connection.execute("DELETE FROM candles WHERE interval='1h' AND open_time=?",
                               (self.t - 4 * HOUR,))
        report = build(self.config)
        self.assertIsNone(self.row())
        exclusion = self.row("dataset_exclusions")
        self.assertEqual(exclusion["reason"], "missing_hour")
        self.assertEqual(json.loads(exclusion["details"])["first_5_open_times"], [self.t - 4 * HOUR])
        self.assertIsNotNone(self.row(decision=self.t + HOUR))
        self.assertEqual(report["coverage"]["1h"]["missing_rows"], 1)

    def test_bad_hour_is_excluded_even_without_source_flag(self):
        with self.connect() as connection:
            connection.execute("UPDATE candles SET close_time=close_time-1 WHERE interval='1h' AND open_time=?",
                               (self.t - 4 * HOUR,))
        report = build(self.config)
        self.assertEqual(self.row("dataset_exclusions")["reason"], "invalid_hour")
        self.assertEqual(report["coverage"]["1h"]["nonstandard_duration_rows"], 1)

    def test_hour_flags_and_nonfinite_values_are_rejected(self):
        with self.connect() as connection:
            connection.execute("UPDATE candles SET quality_flags=? WHERE interval='1h' AND open_time=?",
                               ('["unfinished"]', self.t - 4 * HOUR))
        build(self.config)
        self.assertEqual(self.row("dataset_exclusions")["reason"], "invalid_hour")
        with self.connect() as connection:
            connection.execute("UPDATE candles SET quality_flags='[]',close='NaN' WHERE interval='1h' AND open_time=?",
                               (self.t - 4 * HOUR,))
        build(self.config)
        self.assertEqual(self.row("dataset_exclusions")["reason"], "invalid_hour")

    def test_missing_execution_minute_does_not_use_nearest_price(self):
        with self.connect() as connection:
            connection.execute("DELETE FROM candles WHERE interval='1m' AND open_time=?",
                               (self.t + 2 * MINUTE,))
        report = build(self.config)
        self.assertIsNone(self.row())
        exclusion = self.row("dataset_exclusions")
        self.assertEqual(exclusion["reason"], "missing_minute")
        self.assertEqual(json.loads(exclusion["details"])["role"], "execution")
        self.assertEqual(report["coverage"]["1m"]["missing_rows"], 1)

    def test_missing_label_minute_excludes_row(self):
        with self.connect() as connection:
            connection.execute("DELETE FROM candles WHERE interval='1m' AND open_time=?",
                               (self.t + 4 * HOUR + 2 * MINUTE,))
        build(self.config)
        exclusion = self.row("dataset_exclusions")
        self.assertEqual(exclusion["reason"], "missing_minute")
        self.assertEqual(json.loads(exclusion["details"])["role"], "label_end")

    def test_flagged_execution_minute_excludes_row(self):
        with self.connect() as connection:
            connection.execute("UPDATE candles SET quality_flags=? WHERE interval='1m' AND open_time=?",
                               ('["nonstandard_interval"]', self.t + 2 * MINUTE))
        build(self.config)
        self.assertEqual(self.row("dataset_exclusions")["reason"], "invalid_minute")

    def test_half_open_split_and_exact_label_end_purge(self):
        self.config["splits"] = [{"name": "train", "start": utc(self.t + 2 * MINUTE),
                                  "end": utc(self.t + 8 * HOUR + 2 * MINUTE)}]
        report = build(self.config)
        self.assertIsNotNone(self.row())  # execution exactly at split start is inside.
        self.assertEqual(self.row("dataset_exclusions", self.t + 4 * HOUR)["reason"], "split_boundary")
        with self.connect() as connection:
            self.assertIsNone(connection.execute("SELECT * FROM dataset_exclusions WHERE decision_time=?",
                                                 (self.t + 8 * HOUR,)).fetchone())
            for row in connection.execute("SELECT execution_time,label_end FROM dataset_rows"):
                self.assertGreaterEqual(row[0], self.t + 2 * MINUTE)
                self.assertLess(row[1], self.t + 8 * HOUR + 2 * MINUTE)
        self.assertEqual(report["candidate_rows"], 8)
        self.assertEqual(report["rows"], 4)

    def test_rebuild_has_identical_semantic_report_and_no_duplicates(self):
        first = build(self.config)
        second = build(dict(self.config))
        self.assertEqual(first, second)
        with self.connect() as connection:
            count, unique = connection.execute("SELECT count(*),count(DISTINCT decision_time) FROM dataset_rows").fetchone()
        self.assertEqual(count, first["rows"])
        self.assertEqual(count, unique)
        self.assertTrue(verify_build(self.config)["verified"])

    def test_failed_rebuild_preserves_previous_tables_and_metadata(self):
        previous = build(self.config)
        changed = dict(self.config, feature_windows_hours=[1, 2])
        with patch("src.vkr_data.build._features", side_effect=RuntimeError("interrupted build")):
            with self.assertRaisesRegex(RuntimeError, "interrupted"):
                build(changed)
        verified = verify_build(self.config)
        self.assertEqual(verified["snapshot_id"], previous["snapshot_id"])
        self.assertIn("return_3h", self.row().keys())

    def test_verify_detects_changed_source(self):
        build(self.config)
        with self.connect() as connection:
            connection.execute("UPDATE candles SET volume='99' WHERE interval='1h' AND open_time=?",
                               (self.t - HOUR,))
        with self.assertRaisesRegex(ValueError, "source_sha256"):
            verify_build(self.config)

    def test_verify_detects_changed_dataset(self):
        build(self.config)
        with self.connect() as connection:
            connection.execute("UPDATE dataset_rows SET y=y+0.001 WHERE decision_time=?", (self.t,))
        with self.assertRaisesRegex(ValueError, "dataset_sha256"):
            verify_build(self.config)

    def test_complete_download_is_required_by_default(self):
        config = dict(self.config)
        del config["require_complete"]
        with self.assertRaisesRegex(ValueError, "not complete"):
            build(config)
        with self.connect() as connection:
            connection.execute("INSERT INTO metadata VALUES ('download_complete','false')")
        with self.assertRaisesRegex(ValueError, "not complete"):
            build(config)
        with self.connect() as connection:
            connection.execute("UPDATE metadata SET value='true' WHERE key='download_complete'")
        self.assertEqual(build(config)["rows"], 6)

    def test_bad_timing_and_overlapping_splits_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "minute boundary"):
            build(dict(self.config, execution_delay_seconds=121))
        with self.assertRaisesRegex(ValueError, "must not overlap"):
            build(dict(self.config, splits=[
                {"name": "one", "start": utc(self.t), "end": utc(self.t + 8 * HOUR)},
                {"name": "two", "start": utc(self.t + 7 * HOUR), "end": utc(self.end)},
            ]))


if __name__ == "__main__":
    unittest.main()
