"""Synthetic causal signal tests without network access or model fitting."""

import csv
import gzip
import sqlite3
import tempfile
import unittest
from pathlib import Path

from src.vkr_backtest import data


class BacktestDataTests(unittest.TestCase):
    def setUp(self):
        self.connection = sqlite3.connect(":memory:")
        self.addCleanup(self.connection.close)
        self.start = data.builder._ms("2024-01-01T00:00:00Z")
        self.end = self.start + 8 * data.HOUR
        self.connection.execute("CREATE TABLE candles (symbol TEXT,interval TEXT,open_time INTEGER,"
            "open TEXT,high TEXT,low TEXT,close TEXT,volume TEXT,close_time INTEGER,quote_volume TEXT,"
            "trade_count TEXT,taker_base_volume TEXT,taker_quote_volume TEXT,quality_flags TEXT)")
        for timestamp in range(self.start - 721 * data.HOUR, self.end, data.HOUR):
            self.insert(timestamp, "1h")

    def insert(self, timestamp, interval="1m", price="100", close="100", flags="[]"):
        duration = data.HOUR if interval == "1h" else data.MINUTE
        self.connection.execute("INSERT INTO candles VALUES (" + ",".join("?" * 14) + ")",
            ("BTCUSDT", interval, timestamp, price, "110", "90", close, "1",
             timestamp + duration - 1, "100", "1", "0.5", "50", flags))

    def test_causal_signals_ignore_future_exit_and_supervised_tables(self):
        self.insert(self.start + 2 * data.MINUTE)
        # No exit minute exists. No dataset_rows or future labels even exist.
        rows, counts = data._causal_rows(self.connection, self.start, self.end, 240)
        self.assertEqual(counts, {"calendar_horizon_exclusions": 4,
                                 "causal_signal_rows": 4, "hourly_decisions": 8})
        self.assertEqual(rows[0]["execution_time"], self.start + 2 * data.MINUTE)
        self.assertEqual(rows[0]["signal_time"], self.start + data.MINUTE)
        self.assertEqual(len(rows[0]["features"]), 18)
        self.assertFalse(rows[0]["sma_enter"])

    def test_minute_open_does_not_use_future_ohlcv_or_quality_flags(self):
        self.insert(self.start, close="not-yet-known", flags='["invalid_ohlc"]')
        quotes, metadata = data._read_quotes(self.connection, self.start, self.end)
        self.assertEqual(quotes, {self.start: "100"})
        self.assertEqual(metadata["valid_open_prices"], 1)
        self.assertEqual(metadata["unavailable_minutes"], 479)

    def test_open_only_validation_rejects_invalid_price_and_grid(self):
        for raw in ("NaN", "Infinity", "-1", "0", "bad"):
            self.assertIsNone(data._minute_open(self.start, raw))
        self.assertIsNone(data._minute_open(self.start + 1, "100"))
        self.assertIsNone(data._minute_open(True, "100"))
        self.assertEqual(data._minute_open(self.start, "0.00000001"), "1E-8")

    def test_past_missing_or_invalid_hour_blocks_only_affected_windows(self):
        self.connection.execute("UPDATE candles SET quality_flags='[\"bad\"]' WHERE open_time=?",
                                (self.start - data.HOUR,))
        rows, counts = data._causal_rows(self.connection, self.start, self.end, 240)
        self.assertEqual(rows, [])
        self.assertEqual(counts["past_feature_exclusions:invalid_hour"], 4)

    def test_future_hour_change_cannot_change_first_decision(self):
        before, _ = data._causal_rows(self.connection, self.start, self.end, 240)
        self.connection.execute("UPDATE candles SET close='109' WHERE open_time>=?", (self.start,))
        after, _ = data._causal_rows(self.connection, self.start, self.end, 240)
        self.assertEqual(before[0], after[0])
        self.assertNotEqual(before[1]["features"], after[1]["features"])

    def test_unlabelled_causal_row_is_recovered_without_fit(self):
        rows, _ = data._causal_rows(self.connection, self.start, self.end, 240)
        archived = {method: {} for method in data.FORECAST_METHODS}
        predicted = {"depth4": [0.01] * 4, "depth6": [0.02] * 4}
        signals, metadata = data._assemble(rows, archived, predicted, 0.001, "0.0024")
        self.assertEqual(metadata["prediction_values_recovered_without_labels"], 16)
        self.assertTrue(signals["depth4"][0]["enter"])
        self.assertFalse(signals["train_mean"][0]["enter"])
        self.assertEqual(len(signals["zero"]), 4)

    def test_recovered_signal_with_missing_future_exit_becomes_pending_trade(self):
        from src.vkr_backtest.engine import simulate
        rows, _ = data._causal_rows(self.connection, self.start, self.end, 240)
        archived = {method: {} for method in data.FORECAST_METHODS}
        signals, _ = data._assemble(rows, archived,
            {"depth4": [0.01] * 4, "depth6": [0.02] * 4}, 0.001, "0.0024")
        entry = self.start + 2 * data.MINUTE
        delayed_exit = self.start + 5 * data.HOUR
        result = simulate(signals["depth4"], {entry: "100", delayed_exit: "110"}, {
            "start_time": self.start, "end_time": self.end, "terminal_time": self.end - data.MINUTE,
            "holding_minutes": 240, "initial_cash": "10000", "allocation": "1",
            "qty_step": "0.000001", "min_notional": "10"},
            {"fee_bps": 0, "half_spread_bps": 0, "slippage_bps": 0})
        self.assertEqual(result["status"], "COMPLETE")
        self.assertEqual(len(result["trades"]), 1)
        self.assertEqual(result["trades"][0]["entry_time"], entry)
        self.assertEqual(result["trades"][0]["exit_time"], delayed_exit)
        self.assertTrue(result["trades"][0]["exit_delayed"])

    def test_archived_prediction_mismatch_stops_analysis(self):
        rows, _ = data._causal_rows(self.connection, self.start, self.end, 240)
        archived = {method: {self.start: 0.9} for method in data.FORECAST_METHODS}
        with self.assertRaisesRegex(ValueError, "differ"):
            data._assemble(rows, archived, {"depth4": [0.01] * 4, "depth6": [0.02] * 4}, 0.001, "0.0024")

    def test_prediction_keys_are_aligned_but_future_answer_not_parsed(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "predictions.csv.gz"
            with gzip.open(path, "wt", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(["decision_time", "execution_time", "label_end", "split", "y_true", "y_pred", "method"])
                for method in data.FORECAST_METHODS:
                    writer.writerow([self.start, self.start + 2 * data.MINUTE,
                                     self.start + 2 * data.MINUTE + 4 * data.HOUR,
                                     "validation", "MUST NOT PARSE THIS", "0.01", method])
            result = data._read_predictions(path, self.start, self.end)
            self.assertEqual(result["depth4"], {self.start: 0.01})
            with gzip.open(path, "at", newline="") as stream:
                csv.writer(stream).writerow([self.start, self.start + 2 * data.MINUTE,
                    self.start + 2 * data.MINUTE + 4 * data.HOUR, "validation", "unused", "0.01", "depth4"])
            with self.assertRaisesRegex(ValueError, "duplicate"):
                data._read_predictions(path, self.start, self.end)


if __name__ == "__main__":
    unittest.main()
