"""Offline regressions for errors that could make a source audit misleading."""

import unittest
import json
import tempfile
from http.client import IncompleteRead
from pathlib import Path
from unittest.mock import MagicMock, patch
from datetime import timedelta
from scripts.probe_sources import (audit_klines, audit_metrics, grid_profile,
                                  ms, parse_time, Probe)


class SourceProbeTests(unittest.TestCase):
    def test_duplicate_cannot_hide_boundary_gap(self):
        profile = grid_profile([0, 1, 1], 0, 3, 1)
        self.assertEqual(profile["rows"], profile["expected_rows"])
        self.assertEqual(profile["duplicate_keys"], 1)
        self.assertEqual(profile["missing_rows"], 1)
        self.assertEqual(profile["missing_first_5_ms"], [2])

    def test_half_open_range_and_unexpected_timestamp(self):
        profile = grid_profile([2, 0, 3], 0, 3, 1)
        self.assertEqual(profile["unexpected_rows"], 1)
        self.assertEqual(profile["missing_rows"], 1)
        self.assertTrue(profile["out_of_order"])

    def test_valid_and_unfinished_candles(self):
        start = parse_time("2022-01-01")
        row = [ms(start), "10", "12", "9", "11", "20",
               ms(start) + 3_600_000 - 1, "200", 5, "10", "100", "0"]
        profile = audit_klines([row], start, start + timedelta(hours=1),
                               "1h", start + timedelta(minutes=30))
        self.assertEqual(profile["invalid_rows"], 0)
        self.assertEqual(profile["unfinished_rows"], 1)
        self.assertEqual(profile["missing_rows"], 0)

    def test_invalid_ohlc_and_nonfinite_value(self):
        start = parse_time("2022-01-01")
        row = [ms(start), "10", "9", "8", "11", "20",
               ms(start) + 3_600_000 - 1, "200", 5, "10", "100", "0"]
        end = start + timedelta(hours=1)
        self.assertEqual(audit_klines([row], start, end, "1h", end)["invalid_rows"], 1)
        row[2] = "NaN"
        self.assertEqual(audit_klines([row], start, end, "1h", end)["invalid_rows"], 1)

    def test_present_day_with_absent_metric_is_not_complete(self):
        start = parse_time("2022-01-01")
        profile = audit_metrics([{"asset": "btc", "time": "2022-01-01T00:00:00.000000000Z"}],
                                start, start + timedelta(days=1), "TxCnt")
        self.assertEqual(profile["missing_rows"], 0)
        self.assertEqual(profile["missing_metric_values"], 1)

    def test_completion_timestamp_is_measured_from_day_end(self):
        start = parse_time("2022-01-01")
        rows = [{"asset": "btc", "time": "2022-01-01T00:00:00Z",
                 "AssetEODCompletionTime": str(int(start.timestamp()) + 86400 + 300)}]
        profile = audit_metrics(rows, start, start + timedelta(days=1), "AssetEODCompletionTime")
        self.assertEqual(profile["completion_lag_after_day_end_seconds"]["min"], 300)

    def test_timezone_is_normalized_to_utc(self):
        self.assertEqual(parse_time("2022-01-01T03:00:00+03:00"), parse_time("2022-01-01"))

    def check_read_failure(self, error, expected_body):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            (output / "raw").mkdir()
            response = MagicMock()
            response.code = 200
            response.headers = {"Content-Type": "application/json"}
            response.__enter__.return_value = response
            response.read.side_effect = error
            with patch("scripts.probe_sources.urlopen", return_value=response):
                payload, entry = Probe(output).get("failed_read", "https://example.org", "/data")
            self.assertIsNone(payload)
            self.assertFalse(entry["body_complete"])
            self.assertIn("body_read_error", entry)
            self.assertEqual((output / entry["raw_file"]).read_bytes(), expected_body)
            self.assertEqual(json.loads((output / "manifest.json").read_text())[0], entry)

    def test_read_timeout_preserves_failed_request(self):
        self.check_read_failure(TimeoutError("test timeout"), b"")

    def test_incomplete_read_preserves_partial_body(self):
        self.check_read_failure(IncompleteRead(b'{"data":', 10), b'{"data":')

    def test_reused_response_keeps_original_acquisition_time(self):
        with tempfile.TemporaryDirectory() as directory:
            old, new = Path(directory) / "old", Path(directory) / "new"
            (old / "raw").mkdir(parents=True)
            (new / "raw").mkdir(parents=True)
            response = MagicMock()
            response.code = 200
            response.headers = {"Content-Type": "application/json"}
            response.__enter__.return_value = response
            response.read.return_value = b'{"data":[]}'
            with patch("scripts.probe_sources.urlopen", return_value=response):
                _, original = Probe(old).get("old", "https://example.org", "/data")
            with patch("scripts.probe_sources.urlopen", side_effect=AssertionError("unexpected network")):
                payload, reused = Probe(new, old).get("new", "https://example.org", "/data")
            self.assertEqual(payload, {"data": []})
            self.assertEqual(reused["received_at"], original["received_at"])
            self.assertEqual(reused["sha256"], original["sha256"])
            (old / original["raw_file"]).write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "integrity"):
                Probe(new, old).get("tampered", "https://example.org", "/data")


if __name__ == "__main__":
    unittest.main()
