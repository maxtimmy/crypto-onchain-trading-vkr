"""Resumable Binance kline snapshots, with immutable raw-response evidence.

Each request has a fixed, half-open window containing at most ``page_limit``
expected candles. A short/empty result therefore only describes that window;
it never ends subsequent history. Raw bodies are content-addressed gzip files.
Their request references are durable before the candles/checkpoint transaction.
"""

import gzip
import hashlib
import json
import os
import sqlite3
import sys
import threading
import time
import uuid
import zlib
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from email.utils import parsedate_to_datetime
from http.client import IncompleteRead
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from .store import connect, get_metadata, set_metadata


UTC = timezone.utc
INTERVAL_MS = {"1m": 60_000, "1h": 3_600_000}
FIELDS = ("symbol", "interval", "open_time", "open", "high", "low", "close",
          "volume", "close_time", "quote_volume", "trade_count",
          "taker_base_volume", "taker_quote_volume", "quality_flags")


class DownloadError(RuntimeError):
    pass


class IntegrityError(DownloadError):
    pass


class SourceConflict(DownloadError):
    pass


class BinanceBan(DownloadError):
    pass


def parse_ms(value):
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return int(parsed.timestamp() * 1000)


def utc_now():
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _sha(body):
    return hashlib.sha256(body).hexdigest()


def _settings(config):
    result = dict(config)
    result.setdefault("base_url", "https://data-api.binance.vision")
    result.setdefault("workers", 8)
    result.setdefault("page_limit", 1000)
    result.setdefault("timeout_seconds", 30)
    result.setdefault("max_retries", 4)
    result.setdefault("request_spacing_seconds", 0.1)
    result["base_url"] = result["base_url"].rstrip("/")
    result["start_ms"], result["end_ms"] = parse_ms(result["start"]), parse_ms(result["end"])
    if (not result.get("symbol") or not result.get("intervals")
            or len(set(result["intervals"])) != len(result["intervals"])
            or result["start_ms"] >= result["end_ms"]):
        raise ValueError("symbol, unique intervals and start < end are required")
    if (not 1 <= result["page_limit"] <= 1000 or result["workers"] < 1
            or result["max_retries"] < 0 or result["timeout_seconds"] <= 0
            or result["request_spacing_seconds"] < 0):
        raise ValueError("Invalid download limits")
    for interval in result["intervals"]:
        if interval not in INTERVAL_MS:
            raise ValueError("Unsupported interval: " + str(interval))
        if result["start_ms"] % INTERVAL_MS[interval] or result["end_ms"] % INTERVAL_MS[interval]:
            raise ValueError("Range boundaries must align with each interval")
    return result


def _config_hash(config):
    fields = {key: config[key] for key in ("symbol", "start_ms", "end_ms", "base_url")}
    fields["intervals"] = sorted(config["intervals"])
    return _sha(_canonical(fields).encode())


def _jobs(config):
    for interval in config["intervals"]:
        width = INTERVAL_MS[interval] * config["page_limit"]
        for start in range(config["start_ms"], config["end_ms"], width):
            end = min(config["end_ms"], start + width)
            yield {"job_id": "{}_{}_{}".format(interval, start, end),
                   "interval": interval, "start_ms": start, "end_ms": end}


def _atomic_write(path, body):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp-" + uuid.uuid4().hex)
    try:
        with temporary.open("xb") as stream:
            stream.write(body)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(str(temporary), str(path))
        # Persist the directory entry before updating a SQLite checkpoint.
        descriptor = os.open(str(path.parent), os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        if temporary.exists():
            temporary.unlink()


def _save_response(directory, body, metadata, reference):
    digest = _sha(body)
    relative = "raw/{}.json.gz".format(digest)
    raw = directory / relative
    if raw.exists():
        try:
            previous = gzip.decompress(raw.read_bytes())
        except (OSError, EOFError) as error:
            raise IntegrityError("Unreadable raw body: " + relative) from error
        if _sha(previous) != digest:
            raise IntegrityError("Raw SHA256 mismatch: " + relative)
    else:
        _atomic_write(raw, gzip.compress(body, mtime=0))
    metadata = dict(metadata, raw_path=relative, sha256=digest, bytes=len(body))
    target = directory / reference
    _atomic_write(target, (_canonical(metadata) + "\n").encode())
    return metadata


def _read_reference(directory, reference, expected_url=None):
    try:
        metadata = json.loads((directory / reference).read_text())
        body = gzip.decompress((directory / metadata["raw_path"]).read_bytes())
        if len(body) != metadata["bytes"] or _sha(body) != metadata["sha256"]:
            raise IntegrityError("Raw SHA256 mismatch: " + reference)
        if expected_url is not None and metadata["url"] != expected_url:
            raise IntegrityError("Cached request URL differs: " + reference)
        payload = json.loads(body)
        if (metadata["http_status"] != 200 or not metadata["body_complete"]
                or not isinstance(payload, list)):
            raise IntegrityError("Cached response is not a complete kline page: " + reference)
        return metadata, payload
    except IntegrityError:
        raise
    except (OSError, EOFError, ValueError, KeyError, TypeError) as error:
        raise IntegrityError("Cannot verify saved response: " + reference) from error


class _RequestGate:
    def __init__(self, spacing):
        self.spacing, self.next_at = spacing, 0.0
        self.lock, self.abort = threading.Lock(), threading.Event()

    def wait(self):
        while True:
            if self.abort.is_set():
                raise BinanceBan("Download stopped after HTTP 418")
            with self.lock:
                delay = self.next_at - time.monotonic()
                if delay <= 0:
                    self.next_at = time.monotonic() + self.spacing
                    return
            self.abort.wait(min(delay, 0.25))

    def defer(self, seconds):
        with self.lock:
            self.next_at = max(self.next_at, time.monotonic() + seconds)


def _retry_after(headers, attempt):
    value = next((value for key, value in headers.items() if key.lower() == "retry-after"), None)
    if value:
        try:
            return max(0.0, float(value))
        except ValueError:
            try:
                return max(0.0, parsedate_to_datetime(value).timestamp() - time.time())
            except (ValueError, TypeError, OverflowError):
                pass
    return min(2 ** attempt, 30)


def _request(config, job, gate):
    directory = Path(config["dataset_dir"])
    params = {"symbol": config["symbol"], "interval": job["interval"],
              "startTime": job["start_ms"], "endTime": job["end_ms"] - 1,
              "limit": config["page_limit"]}
    url = config["base_url"] + "/api/v3/klines?" + urlencode(params)
    reference = "refs/{}.json".format(job["job_id"])
    if (directory / reference).exists():
        metadata, payload = _read_reference(directory, reference, url)
        return job, reference, metadata, payload, True
    last_error = None
    for attempt in range(config["max_retries"] + 1):
        gate.wait()
        metadata = {"job_id": job["job_id"], "url": url, "params": params,
                    "sent_at": utc_now(), "http_status": None,
                    "body_complete": False, "attempt": attempt + 1}
        body, headers, status, payload = b"", {}, None, None
        try:
            request = Request(url, headers={"User-Agent": "vkr-dataset-loader/1.0",
                                            "Accept-Encoding": "gzip"})
            try:
                response = urlopen(request, timeout=config["timeout_seconds"])
            except HTTPError as error:
                response = error
            with response:
                status = response.code
                # Persist only provenance/rate-limit headers. Never keep cookies
                # or arbitrary response headers in a shareable request record.
                allowed = {"date", "content-type", "content-encoding", "content-length", "retry-after"}
                headers = {key: value for key, value in response.headers.items()
                           if key.lower() in allowed or key.lower().startswith("x-mbx-used-weight")}
                metadata["http_status"] = status
                metadata["headers"] = headers
                body = response.read()
                content_length = response.headers.get("Content-Length")
                if content_length is not None and len(body) != int(content_length):
                    raise IncompleteRead(body, max(0, int(content_length) - len(body)))
                metadata["wire_bytes"] = len(body)
                metadata["wire_sha256"] = _sha(body)
                metadata["wire_content_encoding"] = response.headers.get("Content-Encoding", "identity")
                if metadata["wire_content_encoding"].lower() == "gzip":
                    body = gzip.decompress(body)
                metadata["body_complete"] = True
            if status == 200:
                payload = json.loads(body)
                if not isinstance(payload, list):
                    raise ValueError("Expected a JSON list of klines")
            elif status == 418:
                gate.abort.set()
                last_error = BinanceBan("Binance HTTP 418; stop and wait for the ban to expire")
            elif status == 429 or status >= 500:
                last_error = DownloadError("Retryable HTTP {}".format(status))
            else:
                last_error = DownloadError("Non-retryable HTTP {}".format(status))
        except IncompleteRead as error:
            body = error.partial
            metadata["body_complete"] = False
            metadata["error"] = repr(error)
            last_error = error
        except (URLError, TimeoutError, OSError, ValueError, EOFError, zlib.error) as error:
            metadata["body_complete"] = False
            metadata["error"] = repr(error)
            last_error = error
        metadata["fetched_at"] = utc_now()
        if payload is not None and status == 200 and metadata["body_complete"]:
            metadata = _save_response(directory, body, metadata, reference)
            return job, reference, metadata, payload, False
        failed_ref = "attempts/{}_{}_{}.json".format(job["job_id"], attempt + 1, uuid.uuid4().hex)
        _save_response(directory, body, metadata, failed_ref)
        if status == 418:
            raise last_error
        if status is not None and status not in (200, 429) and status < 500:
            raise last_error
        if attempt < config["max_retries"]:
            gate.defer(_retry_after(headers, attempt))
    raise DownloadError("Request exhausted retries: {}: {}".format(url, last_error)) from last_error


def _candle(row, config, job, fetched_at):
    if not isinstance(row, list) or len(row) != 12:
        return None, "expected_12_fields"
    if isinstance(row[0], bool) or not isinstance(row[0], int):
        return None, "open_time_type"
    flags = []
    opened = row[0]
    if opened < job["start_ms"] or opened >= job["end_ms"]:
        flags.append("outside_request")
    if opened % INTERVAL_MS[job["interval"]]:
        flags.append("off_grid")
    closed = row[6] if isinstance(row[6], int) and not isinstance(row[6], bool) else None
    count = row[8] if isinstance(row[8], int) and not isinstance(row[8], bool) else None
    if closed is None or count is None or not all(isinstance(row[i], str) for i in (1,2,3,4,5,7,9,10)):
        flags.append("type")
    if closed != opened + INTERVAL_MS[job["interval"]] - 1:
        flags.append("duration")
    if closed is not None and closed >= parse_ms(fetched_at):
        flags.append("unfinished")
    try:
        values = [Decimal(str(row[i])) for i in (1,2,3,4,5,7,9,10)]
        op, high, low, close, volume, quote, taker, taker_quote = values
        if not all(value.is_finite() for value in values):
            flags.append("nonfinite")
        else:
            if not (0 < low <= min(op, close) <= max(op, close) <= high):
                flags.append("ohlc")
            if min(volume, quote, taker, taker_quote) < 0 or taker > volume or taker_quote > quote:
                flags.append("volume")
        if count is not None and count < 0:
            flags.append("trade_count")
    except (InvalidOperation, TypeError, ValueError):
        flags.append("numeric")
    return (config["symbol"], job["interval"], opened, str(row[1]), str(row[2]),
            str(row[3]), str(row[4]), str(row[5]), closed, str(row[7]), count,
            str(row[9]), str(row[10]), _canonical(sorted(set(flags)))), None


def _import_page(connection, config, result):
    job, reference, metadata, payload, reused = result
    duplicates, rejected = 0, 0
    with connection:
        for index, row in enumerate(payload):
            candle, reason = _candle(row, config, job, metadata["fetched_at"])
            if candle is None:
                connection.execute("INSERT INTO rejected_rows VALUES (?,?,?,?)",
                                   (job["job_id"], index, _canonical(row), reason))
                rejected += 1
                continue
            existing = connection.execute("SELECT * FROM candles WHERE symbol=? AND interval=? AND open_time=?",
                                          candle[:3]).fetchone()
            if existing is not None:
                if tuple(existing) != candle:
                    raise SourceConflict("Conflicting candle {} {} {}".format(*candle[:3]))
                duplicates += 1
            else:
                connection.execute("INSERT INTO candles VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)", candle)
        connection.execute("INSERT INTO pages VALUES (?,?,?,?,?,?,?,?)",
                           (job["job_id"], metadata["raw_path"], reference,
                            metadata["sha256"], metadata["fetched_at"], len(payload), duplicates, rejected))
        connection.execute("UPDATE jobs SET status='complete' WHERE job_id=?", (job["job_id"],))
    return reused


def _verify_committed(connection, config):
    """Prove both raw integrity and the exact raw-to-SQL import, without writes."""
    expected_keys = {interval: set() for interval in config["intervals"]}
    expected_rejected_count, duplicate_count, stored_duplicate_count = 0, 0, 0
    pages = connection.execute(
        "SELECT pages.*,jobs.interval,jobs.start_ms,jobs.end_ms,jobs.status "
        "FROM pages JOIN jobs USING(job_id)").fetchall()
    if len(pages) != connection.execute("SELECT count(*) FROM pages").fetchone()[0]:
        raise IntegrityError("A committed page has no matching job")
    if connection.execute(
            "SELECT count(*) FROM jobs LEFT JOIN pages USING(job_id) "
            "WHERE jobs.status='complete' AND pages.job_id IS NULL").fetchone()[0]:
        raise IntegrityError("A completed job has no committed page")
    for page in pages:
        if page["status"] != "complete" or page["interval"] not in expected_keys:
            raise IntegrityError("Unexpected committed job: " + page["job_id"])
        metadata, payload = _read_reference(Path(config["dataset_dir"]), page["meta_path"])
        if (metadata["sha256"] != page["sha256"] or metadata["raw_path"] != page["raw_path"]
                or metadata["fetched_at"] != page["fetched_at"]
                or len(payload) != page["row_count"]):
            raise IntegrityError("SQLite and request reference differ: " + page["job_id"])
        expected_rows, expected_rejected = {}, {}
        for index, row in enumerate(payload):
            candle, reason = _candle(row, config, page, metadata["fetched_at"])
            if candle is None:
                expected_rejected[index] = (_canonical(row), reason)
                continue
            stamp = candle[2]
            if stamp in expected_rows and expected_rows[stamp] != candle:
                raise IntegrityError("Conflicting raw candles: " + page["job_id"])
            duplicate_count += int(stamp in expected_keys[page["interval"]])
            expected_keys[page["interval"]].add(stamp)
            expected_rows[stamp] = candle
        stamps, stored_rows = list(expected_rows), {}
        for offset in range(0, len(stamps), 500):
            batch = stamps[offset:offset + 500]
            query = "SELECT {} FROM candles WHERE symbol=? AND interval=? AND open_time IN ({})".format(
                ",".join(FIELDS), ",".join("?" for _ in batch))
            for row in connection.execute(query, [config["symbol"], page["interval"]] + batch):
                stored_rows[row["open_time"]] = tuple(row)
        for stamp, candle in expected_rows.items():
            if stored_rows.get(stamp) != candle:
                raise IntegrityError("Raw-to-SQL candle mismatch: {} {} {}".format(
                    config["symbol"], page["interval"], stamp))
        stored_rejected = {row["row_index"]: (row["raw_json"], row["reason"])
                           for row in connection.execute(
                               "SELECT row_index,raw_json,reason FROM rejected_rows WHERE job_id=?",
                               (page["job_id"],))}
        if (stored_rejected != expected_rejected
                or len(expected_rejected) != page["rejected_rows"]):
            raise IntegrityError("Raw-to-SQL rejected row mismatch: " + page["job_id"])
        expected_rejected_count += len(expected_rejected)
        stored_duplicate_count += page["duplicate_rows"]
    if duplicate_count != stored_duplicate_count:
        raise IntegrityError("Raw-to-SQL duplicate count mismatch")
    if expected_rejected_count != connection.execute("SELECT count(*) FROM rejected_rows").fetchone()[0]:
        raise IntegrityError("Unexpected rejected rows without a committed raw page")
    # Inspect every SQL key, including rows outside the requested time grid,
    # another symbol, or another interval. Counts cannot hide replacement keys.
    actual_count = 0
    for row in connection.execute("SELECT symbol,interval,open_time FROM candles"):
        if (row["symbol"] != config["symbol"] or row["interval"] not in expected_keys
                or row["open_time"] not in expected_keys[row["interval"]]):
            raise IntegrityError("Unexpected SQL candle without a raw source: {} {} {}".format(*row))
        actual_count += 1
    if actual_count != sum(len(keys) for keys in expected_keys.values()):
        raise IntegrityError("Raw-to-SQL candle key count mismatch")


def download(config, max_pages=None):
    """Acquire at most max_pages successful request windows, resuming atomically."""
    config = _settings(config)
    if max_pages is not None and max_pages < 0:
        raise ValueError("max_pages must be nonnegative")
    directory = Path(config["dataset_dir"])
    directory.mkdir(parents=True, exist_ok=True)
    connection = connect(config)
    try:
        fingerprint = _config_hash(config)
        previous = get_metadata(connection, "download_config_hash")
        if previous is not None and previous != fingerprint:
            raise IntegrityError("Snapshot configuration differs; use a new dataset_dir")
        previous_limit = get_metadata(connection, "download_page_limit")
        if previous_limit is not None and previous_limit != config["page_limit"]:
            raise IntegrityError("Existing snapshot uses another page_limit")
        with connection:
            set_metadata(connection, "download_config_hash", fingerprint)
            set_metadata(connection, "download_page_limit", config["page_limit"])
            for job in _jobs(config):
                connection.execute("INSERT OR IGNORE INTO jobs(job_id,interval,start_ms,end_ms) VALUES (?,?,?,?)",
                                   (job["job_id"], job["interval"], job["start_ms"], job["end_ms"]))
        _verify_committed(connection, config)
        pending = [dict(row) for row in connection.execute(
            "SELECT * FROM jobs WHERE status!='complete' ORDER BY interval,start_ms")]
        if max_pages is not None:
            pending = pending[:max_pages]
        gate, acquired, reused = _RequestGate(config["request_spacing_seconds"]), 0, 0
        if pending:
            with connection:
                set_metadata(connection, "download_complete", False)
            # Keep at most workers requests in flight; all SQLite writes belong
            # to the coordinator. The bounded mode is sequential for exact stops.
            workers = 1 if max_pages is not None else config["workers"]
            with ThreadPoolExecutor(max_workers=workers) as pool:
                iterator = iter(pending)
                inflight = {}
                for _ in range(min(workers, len(pending))):
                    job = next(iterator)
                    inflight[pool.submit(_request, config, job, gate)] = job
                try:
                    while inflight:
                        done, _ = wait(inflight, return_when=FIRST_COMPLETED)
                        for future in done:
                            del inflight[future]
                            reused += int(_import_page(connection, config, future.result()))
                            acquired += 1
                            if acquired % 20 == 0:
                                print(_canonical({"event": "download_progress", "processed_pages": acquired,
                                                  "scheduled_pages": len(pending)}), file=sys.stderr, flush=True)
                            next_job = next(iterator, None)
                            if next_job is not None:
                                inflight[pool.submit(_request, config, next_job, gate)] = next_job
                except BaseException:
                    gate.abort.set()
                    for future in inflight:
                        future.cancel()
                    raise
        incomplete = connection.execute("SELECT count(*) FROM jobs WHERE status!='complete'").fetchone()[0]
        with connection:
            set_metadata(connection, "download_complete", incomplete == 0)
        return {"complete": incomplete == 0, "processed_pages": acquired,
                "reused_pages": reused, "remaining_pages": incomplete,
                "database": str(directory / "market.sqlite3")}
    finally:
        connection.close()


def validate(config, verify_raw=True):
    """Audit coverage and optionally prove every raw digest and imported field."""
    config = _settings(config)
    path = Path(config["dataset_dir"]) / "market.sqlite3"
    if not path.is_file():
        raise FileNotFoundError("Download candles before validating: " + str(path))
    connection = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True, timeout=30)
    connection.row_factory = sqlite3.Row
    try:
        connection.execute("BEGIN")
        fingerprint = get_metadata(connection, "download_config_hash")
        if fingerprint is not None and fingerprint != _config_hash(config):
            raise IntegrityError("Snapshot configuration differs")
        if verify_raw:
            _verify_committed(connection, config)
        intervals = {}
        for interval in config["intervals"]:
            step = INTERVAL_MS[interval]
            times, invalid, flags = [], 0, {}
            for row in connection.execute(
                    "SELECT open_time,quality_flags FROM candles WHERE symbol=? AND interval=? ORDER BY open_time",
                    (config["symbol"], interval)):
                times.append(row[0])
                row_flags = json.loads(row[1])
                invalid += int(bool(row_flags))
                for flag in row_flags:
                    flags[flag] = flags.get(flag, 0) + 1
            expected = (config["end_ms"] - config["start_ms"]) // step
            present = set(times)
            missing, gaps, gap_start, gap_end = 0, [], None, None
            for stamp in range(config["start_ms"], config["end_ms"], step):
                if stamp not in present:
                    missing += 1
                    if gap_start is None:
                        gap_start = stamp
                    gap_end = stamp + step
                elif gap_start is not None:
                    gaps.append({"start_ms": gap_start, "end_ms": gap_end,
                                 "rows": (gap_end-gap_start)//step})
                    gap_start = None
            if gap_start is not None:
                gaps.append({"start_ms": gap_start, "end_ms": gap_end,
                             "rows": (gap_end-gap_start)//step})
            unexpected = sum(bool(stamp < config["start_ms"] or stamp >= config["end_ms"] or stamp % step)
                             for stamp in times)
            jobs = connection.execute("SELECT count(*),sum(status='complete') FROM jobs WHERE interval=?",
                                      (interval,)).fetchone()
            stats = connection.execute(
                "SELECT coalesce(sum(duplicate_rows),0),coalesce(sum(rejected_rows),0) FROM pages JOIN jobs USING(job_id) WHERE interval=?",
                (interval,)).fetchone()
            intervals[interval] = {"expected_rows": expected, "actual_rows": len(times),
                                   "missing_rows": missing, "gaps": gaps,
                                   "unexpected_rows": int(unexpected),
                                   "duplicate_keys": len(times)-len(present),
                                   "duplicate_source_rows": stats[0], "rejected_rows": stats[1],
                                   "invalid_rows": invalid, "quality_flags": flags,
                                   "first_ms": times[0] if times else None,
                                   "last_ms": times[-1] if times else None,
                                   "pages_expected": jobs[0], "pages_downloaded": jobs[1] or 0,
                                   "coverage_fraction": (expected-missing)/expected if expected else None}
        return {"complete": get_metadata(connection, "download_complete", False),
                "raw_verified": bool(verify_raw), "intervals": intervals,
                "download_config_hash": fingerprint,
                "pages_downloaded": connection.execute("SELECT count(*) FROM pages").fetchone()[0]}
    finally:
        connection.close()
