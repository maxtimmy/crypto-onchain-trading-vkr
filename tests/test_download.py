"""Offline tests for snapshot integrity, atomic resume and HTTP failure handling."""

import gzip
import json
import sqlite3
import sys
import tempfile
import unittest
from http.client import IncompleteRead
from pathlib import Path
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.parse import parse_qs, urlparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vkr_data.download import (BinanceBan, DownloadError, IntegrityError,
                               SourceConflict, download, parse_ms, validate)
from vkr_data.store import connect, get_metadata


def candle(stamp, step=60_000):
    return [stamp, "10", "12", "9", "11", "20", stamp + step - 1,
            "200", 5, "10", "100", "0"]


class Response:
    def __init__(self, body, status=200, headers=None, error=None):
        self.body, self.code, self.headers, self.error = body, status, headers or {}, error

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def read(self):
        if self.error:
            raise self.error
        return self.body


class DownloadTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.config = {"symbol": "BTCUSDT", "start": "2022-01-01T00:00:00Z",
                       "end": "2022-01-01T00:06:00Z", "intervals": ["1m"],
                       "dataset_dir": self.directory.name, "workers": 2,
                       "page_limit": 2, "timeout_seconds": 1, "max_retries": 1,
                       "base_url": "https://example.test", "request_spacing_seconds": 0}
        self.start = parse_ms(self.config["start"])

    def full_response(self, request, timeout):
        params = parse_qs(urlparse(request.full_url).query)
        start, end = int(params["startTime"][0]), int(params["endTime"][0]) + 1
        rows = [candle(stamp) for stamp in range(start, end, 60_000)]
        return Response(json.dumps(rows).encode())

    def test_interruption_and_resume_do_not_redownload_or_duplicate(self):
        with patch("vkr_data.download.urlopen", side_effect=self.full_response) as network:
            first = download(self.config, max_pages=1)
            self.assertFalse(first["complete"])
            self.assertEqual(network.call_count, 1)
            resumed = download(self.config)
            self.assertTrue(resumed["complete"])
            self.assertEqual(network.call_count, 3)
        with patch("vkr_data.download.urlopen", side_effect=AssertionError("network")):
            self.assertEqual(download(self.config)["processed_pages"], 0)
        profile = validate(self.config)
        self.assertEqual(profile["intervals"]["1m"]["actual_rows"], 6)
        self.assertEqual(profile["intervals"]["1m"]["duplicate_keys"], 0)

    def test_crash_after_raw_save_before_database_commit_reuses_snapshot(self):
        with patch("vkr_data.download.urlopen", side_effect=self.full_response), patch(
                "vkr_data.download._import_page", side_effect=RuntimeError("simulated crash")):
            with self.assertRaisesRegex(RuntimeError, "simulated crash"):
                download(self.config, max_pages=1)
        with patch("vkr_data.download.urlopen", side_effect=AssertionError("unexpected network")):
            result = download(self.config, max_pages=1)
        self.assertEqual(result["reused_pages"], 1)
        self.assertEqual(validate(self.config)["intervals"]["1m"]["actual_rows"], 2)

    def test_import_transaction_rolls_back_conflict_then_recovers(self):
        rows = [candle(self.start), candle(self.start)]
        rows[1][4] = "10.5"
        with patch("vkr_data.download.urlopen", return_value=Response(json.dumps(rows).encode())):
            with self.assertRaises(SourceConflict):
                download(self.config, max_pages=1)
        with connect(self.config) as connection:
            self.assertEqual(connection.execute("SELECT count(*) FROM candles").fetchone()[0], 0)
            self.assertEqual(connection.execute("SELECT count(*) FROM pages").fetchone()[0], 0)

    def test_identical_duplicates_are_counted_without_hiding_gaps(self):
        rows = [candle(self.start), candle(self.start)]
        with patch("vkr_data.download.urlopen", return_value=Response(json.dumps(rows).encode())):
            download(self.config, max_pages=1)
        profile = validate(self.config)["intervals"]["1m"]
        self.assertEqual(profile["duplicate_source_rows"], 1)
        self.assertEqual(profile["actual_rows"], 1)
        self.assertEqual(profile["missing_rows"], 5)

    def test_short_and_empty_windows_do_not_end_history(self):
        def source(request, timeout):
            params = parse_qs(urlparse(request.full_url).query)
            start = int(params["startTime"][0])
            if start == self.start:
                rows = [candle(start)]
            elif start == self.start + 120_000:
                rows = []
            else:
                row = candle(start)
                row[6] -= 1000
                rows = [row, candle(start + 60_000)]
            return Response(json.dumps(rows).encode())
        with patch("vkr_data.download.urlopen", side_effect=source):
            self.assertTrue(download(self.config)["complete"])
        profile = validate(self.config)["intervals"]["1m"]
        self.assertEqual(profile["missing_rows"], 3)
        self.assertEqual(profile["gaps"], [{"start_ms": self.start + 60_000,
                                            "end_ms": self.start + 240_000, "rows": 3}])
        self.assertEqual(profile["quality_flags"], {"duration": 1})

    def test_boundary_gaps_include_first_and_last_candle(self):
        self.config["page_limit"] = 1000
        with patch("vkr_data.download.urlopen", return_value=Response(json.dumps(
                [candle(self.start + 60_000), candle(self.start + 120_000)]).encode())):
            download(self.config)
        gaps = validate(self.config)["intervals"]["1m"]["gaps"]
        self.assertEqual(gaps[0]["start_ms"], self.start)
        self.assertEqual(gaps[-1]["end_ms"], self.start + 360_000)

    def test_timeout_and_incomplete_body_are_saved_and_retried(self):
        for first in (TimeoutError("timeout"), Response(b"", error=IncompleteRead(b"[", 10))):
            with self.subTest(first=type(first).__name__):
                with tempfile.TemporaryDirectory() as directory:
                    config = dict(self.config, dataset_dir=directory)
                    with patch("vkr_data.download.urlopen", side_effect=[first, self.full_response(
                            type("R", (), {"full_url": "https://x/?startTime={}&endTime={}".format(
                                self.start, self.start + 120_000 - 1)})(), 1)]), patch(
                                "vkr_data.download._retry_after", return_value=0):
                        self.assertEqual(download(config, max_pages=1)["processed_pages"], 1)
                    attempts = list((Path(directory) / "attempts").glob("*.json"))
                    self.assertEqual(len(attempts), 1)
                    self.assertFalse(json.loads(attempts[0].read_text())["body_complete"])

    def test_retry_exhaustion_never_advances_checkpoint(self):
        with patch("vkr_data.download.urlopen", side_effect=TimeoutError("timeout")) as network, patch(
                "vkr_data.download._retry_after", return_value=0):
            with self.assertRaises(DownloadError):
                download(self.config, max_pages=1)
        self.assertEqual(network.call_count, 2)
        self.assertEqual(validate(self.config)["pages_downloaded"], 0)

    def test_429_coordinates_retry_after_and_418_stops_without_retry(self):
        success = Response(json.dumps([candle(self.start)]).encode())
        with patch("vkr_data.download.urlopen", side_effect=[Response(b"{}", 429, {"Retry-After": "0"}), success]), patch(
                "vkr_data.download._RequestGate.defer") as defer:
            download(self.config, max_pages=1)
            defer.assert_called_once_with(0.0)
        with patch("vkr_data.download.urlopen", return_value=Response(b"{}", 418)) as network:
            with self.assertRaises(BinanceBan):
                download(self.config, max_pages=1)
        self.assertEqual(network.call_count, 1)

    def test_wire_gzip_is_decoded_and_hashed_as_json(self):
        body = json.dumps([candle(self.start)]).encode()
        wire = gzip.compress(body)
        with patch("vkr_data.download.urlopen", return_value=Response(
                wire, headers={"Content-Encoding": "gzip", "Content-Length": str(len(wire))})):
            download(self.config, max_pages=1)
        reference = next((Path(self.directory.name) / "refs").glob("*.json"))
        meta = json.loads(reference.read_text())
        self.assertEqual(meta["bytes"], len(body))
        self.assertEqual(meta["wire_bytes"], len(wire))
        self.assertEqual(gzip.decompress((Path(self.directory.name) / meta["raw_path"]).read_bytes()), body)

    def test_truncated_and_corrupt_gzip_are_saved_then_retried(self):
        correct = gzip.compress(json.dumps([candle(self.start)]).encode())
        corrupt = bytearray(correct)
        corrupt[10] ^= 255  # Invalid deflate block -> zlib.error, not HTTP read error.
        for failed in (correct[:-4], bytes(corrupt)):
            with self.subTest(failed=failed[:12]):
                with tempfile.TemporaryDirectory() as directory:
                    config = dict(self.config, dataset_dir=directory)
                    with patch("vkr_data.download.urlopen", side_effect=[
                            Response(failed, headers={"Content-Encoding": "gzip"}),
                            Response(correct, headers={"Content-Encoding": "gzip"})]) as network, patch(
                                "vkr_data.download._retry_after", return_value=0):
                        result = download(config, max_pages=1)
                    self.assertEqual(network.call_count, 2)
                    self.assertEqual(result["processed_pages"], 1)
                    attempt = next((Path(directory) / "attempts").glob("*.json"))
                    meta = json.loads(attempt.read_text())
                    self.assertFalse(meta["body_complete"])
                    self.assertEqual(gzip.decompress((Path(directory) / meta["raw_path"]).read_bytes()), failed)

    def test_only_public_provenance_and_rate_limit_headers_are_saved(self):
        headers = {"Date": "Mon, 19 Oct 2026 18:00:00 GMT", "Set-Cookie": "secret-cookie",
                   "X-Custom-Token": "secret-token", "X-MBX-USED-WEIGHT-1M": "2"}
        with patch("vkr_data.download.urlopen", return_value=Response(
                json.dumps([candle(self.start)]).encode(), headers=headers)):
            download(self.config, max_pages=1)
        reference = next((Path(self.directory.name) / "refs").glob("*.json"))
        saved = json.loads(reference.read_text())["headers"]
        self.assertEqual(saved, {"Date": headers["Date"], "X-MBX-USED-WEIGHT-1M": "2"})

    def test_tampered_snapshot_fails_before_network_even_when_complete(self):
        with patch("vkr_data.download.urlopen", side_effect=self.full_response):
            download(self.config)
        raw = next((Path(self.directory.name) / "raw").glob("*.gz"))
        raw.write_bytes(gzip.compress(b"[]"))
        with patch("vkr_data.download.urlopen", side_effect=AssertionError("unexpected network")):
            with self.assertRaises(IntegrityError):
                download(self.config)
            with self.assertRaises(IntegrityError):
                validate(self.config)

    def test_valid_price_change_in_sql_is_rejected_against_raw(self):
        with patch("vkr_data.download.urlopen", side_effect=self.full_response):
            download(self.config)
        with connect(self.config) as connection:
            connection.execute("UPDATE candles SET close='10.5' WHERE open_time=?", (self.start,))
        with self.assertRaisesRegex(IntegrityError, "Raw-to-SQL candle mismatch"):
            validate(self.config, verify_raw=True)

    def test_missing_sql_candle_is_rejected_against_raw(self):
        with patch("vkr_data.download.urlopen", side_effect=self.full_response):
            download(self.config)
        with connect(self.config) as connection:
            connection.execute("DELETE FROM candles WHERE open_time=?", (self.start,))
        with self.assertRaisesRegex(IntegrityError, "Raw-to-SQL candle mismatch"):
            validate(self.config, verify_raw=True)

    def test_extra_sql_candle_is_rejected_against_raw(self):
        with patch("vkr_data.download.urlopen", side_effect=self.full_response):
            download(self.config)
        with connect(self.config) as connection:
            connection.execute("INSERT INTO candles SELECT symbol,interval,open_time-60000,"
                               "open,high,low,close,volume,close_time-60000,quote_volume,"
                               "trade_count,taker_base_volume,taker_quote_volume,quality_flags "
                               "FROM candles WHERE open_time=?", (self.start,))
        with self.assertRaisesRegex(IntegrityError, "Unexpected SQL candle"):
            validate(self.config, verify_raw=True)

    def test_rejected_rows_are_verified_against_their_original_payload(self):
        rows = [candle(self.start), ["malformed", "row"]]
        with patch("vkr_data.download.urlopen", return_value=Response(json.dumps(rows).encode())):
            download(self.config, max_pages=1)
        profile = validate(self.config, verify_raw=True)
        self.assertEqual(profile["intervals"]["1m"]["rejected_rows"], 1)
        with connect(self.config) as connection:
            connection.execute("UPDATE rejected_rows SET reason='different_reason'")
        with self.assertRaisesRegex(IntegrityError, "Raw-to-SQL rejected row mismatch"):
            validate(self.config, verify_raw=True)

    def test_replay_from_saved_snapshot_has_identical_values(self):
        with patch("vkr_data.download.urlopen", side_effect=self.full_response):
            download(self.config)
        with connect(self.config) as connection:
            before = [tuple(row) for row in connection.execute("SELECT * FROM candles ORDER BY open_time")]
            connection.execute("DELETE FROM pages")
            connection.execute("DELETE FROM candles")
            connection.execute("UPDATE jobs SET status='pending'")
        with patch("vkr_data.download.urlopen", side_effect=AssertionError("unexpected network")):
            result = download(self.config)
        with connect(self.config) as connection:
            after = [tuple(row) for row in connection.execute("SELECT * FROM candles ORDER BY open_time")]
        self.assertEqual(before, after)
        self.assertEqual(result["reused_pages"], 3)


if __name__ == "__main__":
    unittest.main()
