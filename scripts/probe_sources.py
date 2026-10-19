#!/usr/bin/env python3
"""Bounded, unauthenticated API probes; not an incremental data loader.

Run from the repository root with Python >=3.9. Raw HTTP responses stay in
ignored data/probes/. No trading endpoints, API keys or external packages.
"""

import argparse
import hashlib
import json
import re
import time
from datetime import datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
from http.client import IncompleteRead
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


UTC = timezone.utc
ROOT = Path(__file__).resolve().parents[1]
RESEARCH_START = "2021-12-01"
RESEARCH_END = "2026-01-01"
METRICS = ("TxCnt", "AdrActCnt", "TxTfrValNtv", "TxTfrValAdjNtv",
           "AssetEODCompletionTime")
INTERVAL_MS = {"1m": 60_000, "1h": 3_600_000, "4h": 14_400_000,
               "1d": 86_400_000}


def iso(value):
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def parse_time(value):
    # Coin Metrics emits nanoseconds; Python 3.9 accepts up to microseconds.
    normalized = re.sub(r"(\.\d{6})\d+(?=Z|[+-]\d{2}:\d{2}|$)", r"\1", value)
    parsed = datetime.fromisoformat(normalized.replace("Z", "+00:00"))
    return parsed.replace(tzinfo=UTC) if parsed.tzinfo is None else parsed.astimezone(UTC)


def ms(value):
    return int(value.timestamp() * 1000)


def grid_profile(timestamps, start, end, step):
    """Compare the full expected half-open grid, including boundary gaps."""
    expected = set(range(start, end, step))
    actual = set(timestamps)
    missing = sorted(expected - actual)
    unexpected = sorted(actual - expected)
    return {"rows": len(timestamps), "expected_rows": len(expected),
            "duplicate_keys": len(timestamps) - len(actual),
            "missing_rows": len(missing), "unexpected_rows": len(unexpected),
            "missing_fraction": len(missing) / len(expected) if expected else None,
            "missing_first_5_ms": missing[:5],
            "unexpected_first_5_ms": unexpected[:5],
            "out_of_order": timestamps != sorted(timestamps),
            "first_ms": min(timestamps) if timestamps else None,
            "last_ms": max(timestamps) if timestamps else None}


def audit_klines(rows, start, end, interval, received_at):
    timestamps, malformed, invalid, unfinished = [], 0, 0, 0
    for row in rows:
        try:
            if len(row) != 12:
                raise ValueError("expected 12 kline fields")
            opened, closed = int(row[0]), int(row[6])
            timestamps.append(opened)
            values = [Decimal(str(row[i])) for i in (1, 2, 3, 4, 5, 7, 9, 10)]
            op, high, low, close, volume, quote, taker, taker_quote = values
            if (not all(v.is_finite() for v in values)
                    or not (0 < low <= min(op, close) <= max(op, close) <= high)
                    or min(volume, quote, taker, taker_quote) < 0
                    or taker > volume or taker_quote > quote or int(row[8]) < 0
                    or closed != opened + INTERVAL_MS[interval] - 1):
                invalid += 1
            unfinished += closed >= ms(received_at)
        except (TypeError, ValueError, IndexError, InvalidOperation):
            malformed += 1
    result = grid_profile(timestamps, ms(start), ms(end), INTERVAL_MS[interval])
    result.update(malformed_rows=malformed, invalid_rows=invalid,
                  unfinished_rows=unfinished, input_rows=len(rows))
    return result


def audit_metrics(rows, start, end, metric):
    timestamps, missing, invalid, completion_lags = [], 0, 0, []
    for row in rows:
        try:
            event = parse_time(row["time"])
            timestamps.append(ms(event))
            if row.get(metric) is None:
                missing += 1
                continue
            value = Decimal(str(row[metric]))
            if not value.is_finite() or value < 0 or row.get("asset") != "btc":
                invalid += 1
            elif metric == "AssetEODCompletionTime":
                completion_lags.append(float(value) - event.timestamp() - 86400)
        except (KeyError, TypeError, ValueError, InvalidOperation):
            invalid += 1
    result = grid_profile(timestamps, ms(start), ms(end), INTERVAL_MS["1d"])
    result.update(missing_metric_values=missing, invalid_rows=invalid, input_rows=len(rows),
                  fields=sorted({key for row in rows for key in row}))
    if completion_lags:
        result["completion_lag_after_day_end_seconds"] = {
            "min": min(completion_lags), "max": max(completion_lags),
            "negative_count": sum(value < 0 for value in completion_lags)}
    return result


class Probe:
    def __init__(self, output, reuse_run=None):
        self.output = output
        self.requests = []
        self.last_coinmetrics = 0.0
        self.reuse_run = reuse_run
        self.cached = {}
        if reuse_run is not None:
            for entry in json.loads((reuse_run / "manifest.json").read_text()):
                if (entry.get("http_status") == 200 and entry.get("raw_file")
                        and entry.get("body_complete", True)
                        and not entry.get("parse_error")):
                    self.cached[entry["url"]] = entry

    def get(self, name, base, path, params=None):
        url = base + path + ("?" + urlencode(params) if params else "")
        if url in self.cached:
            previous = self.cached[url]
            body = (self.reuse_run / previous["raw_file"]).read_bytes()
            if (len(body) != previous["bytes"] or
                    hashlib.sha256(body).hexdigest() != previous["sha256"]):
                raise ValueError("Cached response failed integrity check: " + previous["id"])
            payload = json.loads(body)
            entry = dict(previous, id=name, raw_file="raw/" + name + ".json",
                         reused_from=str(self.reuse_run), body_complete=True)
            (self.output / entry["raw_file"]).write_bytes(body)
            self.requests.append(entry)
            self.save_manifest()
            return payload, entry
        # Stay below the documented Community ceiling, irrespective of the
        # higher plan value sometimes observed in response headers.
        if "coinmetrics.io" in base:
            time.sleep(max(0, 0.7 - (time.monotonic() - self.last_coinmetrics)))
            self.last_coinmetrics = time.monotonic()
        sent = datetime.now(UTC)
        request = Request(url, headers={"User-Agent": "vkr-source-probe/1.0"})
        try:
            response = urlopen(request, timeout=20)
        except HTTPError as error:
            response = error
        except (URLError, TimeoutError, OSError) as error:
            entry = {"id": name, "url": url, "sent_at": iso(sent),
                     "received_at": iso(datetime.now(UTC)), "http_status": None,
                     "network_error": str(error)}
            self.requests.append(entry)
            self.save_manifest()
            return None, entry
        status = response.code
        headers = {key.lower(): value for key, value in response.headers.items()
                   if key.lower() in ("date", "content-type", "retry-after")
                   or key.lower().startswith(("x-mbx-used-weight", "x-ratelimit"))}
        read_error = None
        try:
            with response:
                body = response.read()
        except (TimeoutError, OSError, IncompleteRead) as error:
            body = error.partial if isinstance(error, IncompleteRead) else b""
            read_error = str(error)
        received = datetime.now(UTC)
        relative = "raw/" + name + ".json"
        (self.output / relative).write_bytes(body)
        entry = {"id": name, "url": url, "sent_at": iso(sent),
                 "received_at": iso(received), "http_status": status,
                 "elapsed_seconds": round((received - sent).total_seconds(), 3),
                 "response_headers": headers, "raw_file": relative,
                 "bytes": len(body), "sha256": hashlib.sha256(body).hexdigest(),
                 "body_complete": read_error is None}
        if read_error is not None:
            entry["body_read_error"] = read_error
            self.requests.append(entry)
            self.save_manifest()
            if status in (418, 429):
                raise RuntimeError("Rate limit: incomplete response; stop before any further request")
            return None, entry
        try:
            payload = json.loads(body)
        except (ValueError, UnicodeError):
            payload = None
            entry["parse_error"] = "response is not JSON"
        if isinstance(payload, dict) and "error" in payload:
            entry["api_error"] = payload["error"]
        self.requests.append(entry)
        self.save_manifest()
        # Never hammer a throttled/banned endpoint, even if Retry-After is long.
        if status in (418, 429):
            raise RuntimeError("Rate limit: saved response; stop and honor Retry-After")
        return payload if status == 200 else None, entry

    def save_manifest(self):
        (self.output / "manifest.json").write_text(
            json.dumps(self.requests, ensure_ascii=False, indent=2) + "\n")

    def binance_rows(self, name, start, end, interval):
        rows, cursor = [], ms(start)
        for page in range(100):
            payload, entry = self.get(
                name + "_p" + str(page + 1), "https://data-api.binance.vision",
                "/api/v3/klines", {"symbol": "BTCUSDT", "interval": interval,
                "timeZone": "0", "startTime": cursor, "endTime": ms(end) - 1,
                "limit": 1000})
            if not isinstance(payload, list):
                return {"status": "failed", "request_id": entry["id"]}
            rows.extend(payload)
            if not payload or len(payload) < 1000:
                break
            next_cursor = int(payload[-1][0]) + INTERVAL_MS[interval]
            if next_cursor <= cursor:
                raise RuntimeError("Non-advancing Binance pagination")
            cursor = next_cursor
            if cursor >= ms(end):
                break
        else:
            raise RuntimeError("Probe page cap exceeded")
        return {"status": "ok", "start": iso(start), "end_exclusive": iso(end),
                "interval": interval, "pages": page + 1,
                "profile": audit_klines(rows, start, end, interval, datetime.now(UTC))}

    def coinmetrics_rows(self, name, start, end, metric):
        rows = []
        params = {"assets": "btc", "metrics": metric, "frequency": "1d",
                  "start_time": iso(start),
                  "end_time": iso(end - timedelta(seconds=1)), "page_size": 1000}
        for page in range(10):
            payload, entry = self.get(name + "_p" + str(page + 1),
                "https://community-api.coinmetrics.io", "/v4/timeseries/asset-metrics", params)
            if not isinstance(payload, dict) or "data" not in payload:
                return {"status": "unavailable", "request_id": entry["id"],
                        "http_status": entry["http_status"], "error": entry.get("api_error")}
            rows.extend(payload["data"])
            token = payload.get("next_page_token")
            if not token:
                break
            if params.get("next_page_token") == token:
                raise RuntimeError("Non-advancing Coin Metrics pagination")
            params["next_page_token"] = token
        else:
            raise RuntimeError("Probe page cap exceeded")
        return {"status": "ok", "start": iso(start), "end_exclusive": iso(end),
                "metric": metric, "frequency": "1d", "pages": page + 1,
                "profile": audit_metrics(rows, start, end, metric)}


def run(output, as_of, reuse_run=None):
    output.mkdir(parents=True, exist_ok=False)
    (output / "raw").mkdir()
    (output / "probe_sources.py").write_bytes(Path(__file__).read_bytes())
    probe = Probe(output, reuse_run)
    summary = {"schema_version": 1, "started_at": iso(datetime.now(UTC)),
               "as_of_date_utc": as_of.date().isoformat(),
               "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               "research_start": RESEARCH_START, "research_end_exclusive": RESEARCH_END,
               "binance": {}, "coinmetrics": {},
               "scope": "Raw snapshot quality only; no model, trading results or PIT certification."}
    exchange, _ = probe.get("binance_exchange_info", "https://data-api.binance.vision",
                            "/api/v3/exchangeInfo", {"symbol": "BTCUSDT"})
    if isinstance(exchange, dict):
        summary["binance"]["rate_limits"] = exchange.get("rateLimits")
        summary["binance"]["current_symbol_rules"] = exchange.get("symbols")
    server, _ = probe.get("binance_time", "https://data-api.binance.vision", "/api/v3/time")
    summary["binance"]["server_time_snapshot"] = server
    earliest, _ = probe.get("binance_earliest", "https://data-api.binance.vision",
                           "/api/v3/klines", {"symbol": "BTCUSDT", "interval": "1h",
                           "startTime": 0, "limit": 1})
    if isinstance(earliest, list) and earliest:
        summary["binance"]["earliest_returned_open_utc"] = iso(
            datetime.fromtimestamp(earliest[0][0] / 1000, UTC))
    start, end = parse_time(RESEARCH_START), parse_time(RESEARCH_END)
    # Audit the whole hourly grid, without calculating returns or inspecting
    # final-test predictive performance. Daily bars are an independent check.
    for interval in ("1h", "1d"):
        name = "research_" + interval
        summary["binance"][name] = probe.binance_rows("binance_" + name, start, end, interval)
    old = parse_time("2022-01-01")
    recent = as_of - timedelta(days=14)
    for name, window_start, window_end in (("old", old, old + timedelta(days=7)),
                                         ("recent", recent, as_of)):
        for interval in ("1h", "4h", "1d"):
            key = name + "_" + interval
            summary["binance"][key] = probe.binance_rows(
                "binance_" + key, window_start, window_end, interval)
        key = name + "_1m"
        summary["binance"][key] = probe.binance_rows(
            "binance_" + key, window_start, window_start + timedelta(hours=2), "1m")
    # Separate requests prevent an unavailable metric from hiding available ones.
    for metric in METRICS:
        history = probe.coinmetrics_rows("cm_history_" + metric, start, end, metric)
        summary["coinmetrics"][metric] = {"history": history}
        if history["status"] == "ok":
            summary["coinmetrics"][metric]["recent"] = probe.coinmetrics_rows(
                "cm_recent_" + metric, recent, as_of, metric)
    catalog, _ = probe.get("cm_catalog", "https://community-api.coinmetrics.io",
        "/v4/catalog-v2/asset-metrics", {"assets": "btc", "metrics": ",".join(METRICS),
        "page_size": 100})
    summary["coinmetrics_catalog"] = catalog
    status_sample, _ = probe.get("cm_status_all", "https://community-api.coinmetrics.io",
        "/v4/timeseries/asset-metrics", {"assets": "btc", "metrics": "TxCnt,AdrActCnt,AssetEODCompletionTime",
        "frequency": "1d", "start_time": "2022-01-01", "end_time": "2022-01-03", "page_size": 3,
        "status": "all"})
    summary["coinmetrics_status_sample"] = status_sample
    latest, entry = probe.get("cm_latest", "https://community-api.coinmetrics.io",
        "/v4/timeseries/asset-metrics", {"assets": "btc", "metrics": "TxCnt,AdrActCnt,AssetEODCompletionTime",
        "frequency": "1d", "limit_per_asset": 3, "paging_from": "end", "page_size": 3})
    summary["coinmetrics_latest"] = latest
    summary["completed_at"] = iso(datetime.now(UTC))
    summary["requests"] = len(probe.requests)
    summary["http_status_counts"] = {str(code): sum(
        entry["http_status"] == code for entry in probe.requests)
        for code in {entry["http_status"] for entry in probe.requests}}
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"output": str(output), "requests": len(probe.requests),
                      "http_status_counts": summary["http_status_counts"]}, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="New directory; existing runs are never overwritten")
    parser.add_argument("--as-of", help="UTC date; only controls recent probes (YYYY-MM-DD)")
    parser.add_argument("--reuse-run", type=Path,
                        help="Reuse matching successful responses after SHA-256 verification; preserve acquisition timestamps")
    args = parser.parse_args()
    now = datetime.now(UTC)
    as_of = parse_time(args.as_of) if args.as_of else now.replace(hour=0, minute=0, second=0, microsecond=0)
    if as_of > now or as_of.hour or as_of.minute or as_of.second:
        parser.error("--as-of must be a UTC midnight no later than today")
    run(args.output or ROOT / "data" / "probes" / now.strftime("%Y%m%dT%H%M%S%fZ"), as_of,
        args.reuse_run)
