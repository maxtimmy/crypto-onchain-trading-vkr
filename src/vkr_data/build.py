"""Build the first hourly market dataset from an immutable local candle snapshot.

All times in SQLite are UTC Unix milliseconds.  This module deliberately does
not train a model, simulate trading, or summarise the distribution of labels.
"""

import hashlib
import json
import math
import sqlite3
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


BUILDER_VERSION = "market-hourly-v1"
HOUR = 3_600_000
MINUTE = 60_000
RAW_COLUMNS = (
    "symbol", "interval", "open_time", "open", "high", "low", "close",
    "volume", "close_time", "quote_volume", "trade_count", "taker_base_volume",
    "taker_quote_volume", "quality_flags",
)
TIME_COLUMNS = ("decision_time", "signal_time", "execution_time", "label_end")


def _json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False)


def _sha(value):
    return hashlib.sha256(_json(value).encode("utf-8")).hexdigest()


def _ms(value):
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("Times must include an explicit UTC offset")
    return int(parsed.astimezone(timezone.utc).timestamp() * 1000)


def _utc(value):
    return datetime.fromtimestamp(value / 1000, timezone.utc).isoformat().replace("+00:00", "Z")


def _integer(value, name, minimum=1):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError("%s must be an integer >= %s" % (name, minimum))
    return value


def _configuration(config):
    """Only semantic settings participate in the build identity."""
    start, end = _ms(config["start"]), _ms(config["end"])
    if start >= end or start % HOUR or end % HOUR:
        raise ValueError("The download range must have increasing hourly boundaries")
    if config.get("feature_interval", "1h") != "1h":
        raise ValueError("This dataset version supports feature_interval=1h only")
    horizon = _integer(config.get("horizon_hours", 4), "horizon_hours")
    feature_delay = _integer(config.get("feature_delay_seconds", 60),
                             "feature_delay_seconds", 0)
    execution_delay = _integer(config.get("execution_delay_seconds", 120),
                               "execution_delay_seconds", 0)
    if feature_delay > execution_delay or execution_delay % 60:
        raise ValueError("Execution must be on a minute boundary after feature availability")
    windows = sorted(set(_integer(w, "feature window")
                         for w in config.get("feature_windows_hours", [24, 168, 720])))
    if not windows:
        raise ValueError("At least one feature window is required")
    splits, names = [], set()
    for split in config["splits"]:
        name = split["name"]
        left, right = _ms(split["start"]), _ms(split["end"])
        if not isinstance(name, str) or not name or name in names:
            raise ValueError("Split names must be nonempty and unique")
        if left < start or right > end or left >= right:
            raise ValueError("Every split must be inside the download range")
        names.add(name)
        splits.append({"name": name, "start": _utc(left), "end": _utc(right)})
    splits.sort(key=lambda split: _ms(split["start"]))
    if not splits or any(_ms(a["end"]) > _ms(b["start"])
                         for a, b in zip(splits, splits[1:])):
        raise ValueError("Splits must be nonempty and must not overlap")
    return {
        "symbol": config["symbol"], "start": _utc(start), "end": _utc(end),
        "feature_interval": "1h", "horizon_hours": horizon,
        "feature_delay_seconds": feature_delay,
        "execution_delay_seconds": execution_delay,
        "feature_windows_hours": windows, "splits": splits,
    }


def _connect(config):
    path = Path(config["dataset_dir"]) / "market.sqlite3"
    if not path.is_file():
        raise FileNotFoundError("Download candles before building: %s" % path)
    connection = sqlite3.connect(str(path), timeout=60, isolation_level=None)
    connection.row_factory = sqlite3.Row
    return connection


def _metadata(connection, key):
    try:
        row = connection.execute("SELECT value FROM metadata WHERE key=?", (key,)).fetchone()
    except sqlite3.OperationalError:
        return None
    return json.loads(row[0]) if row else None


def _require_complete(connection, config):
    if config.get("require_complete", True) and _metadata(connection, "download_complete") is not True:
        raise ValueError("Download is not complete; finish both candle intervals before building")


def _flags(raw):
    try:
        flags = json.loads(raw)
    except (TypeError, ValueError):
        return ["invalid_quality_flags"]
    return flags if isinstance(flags, list) else ["invalid_quality_flags"]


def _candle(row, duration):
    """Validate all source fields even when only the minute open is used."""
    flags = _flags(row["quality_flags"])
    if flags:
        return None, "quality_flags:%s" % _json(flags)
    if row["open_time"] % duration or row["close_time"] != row["open_time"] + duration - 1:
        return None, "nonstandard_interval"
    try:
        values = {name: float(row[name]) for name in (
            "open", "high", "low", "close", "volume", "quote_volume",
            "trade_count", "taker_base_volume", "taker_quote_volume")}
    except (TypeError, ValueError, OverflowError):
        return None, "invalid_numeric_field"
    if not all(math.isfinite(value) for value in values.values()):
        return None, "nonfinite_numeric_field"
    if (min(values[name] for name in ("open", "high", "low", "close")) <= 0
            or any(values[name] < 0 for name in ("volume", "quote_volume", "trade_count",
                                                "taker_base_volume", "taker_quote_volume"))
            or values["trade_count"] != int(values["trade_count"])):
        return None, "invalid_numeric_range"
    if (values["high"] < max(values["open"], values["close"], values["low"])
            or values["low"] > min(values["open"], values["close"], values["high"])):
        return None, "invalid_ohlc"
    return values, None


def _scan_source(connection, semantic):
    """Hash source values in key order without loading minute history into RAM."""
    digest = hashlib.sha256()
    digest.update((_json({"symbol": semantic["symbol"], "start": semantic["start"],
                          "end": semantic["end"]}) + "\n").encode("utf-8"))
    coverage = {}
    start, end = _ms(semantic["start"]), _ms(semantic["end"])
    for interval, step in (("1h", HOUR), ("1m", MINUTE)):
        counts = {"expected_rows": (end - start) // step, "rows": 0,
                  "on_grid_rows": 0, "flagged_rows": 0, "nonstandard_duration_rows": 0}
        query = "SELECT %s FROM candles WHERE symbol=? AND interval=? AND open_time>=? AND open_time<? ORDER BY open_time" % ",".join(RAW_COLUMNS)
        for row in connection.execute(query, (semantic["symbol"], interval, start, end)):
            digest.update((_json(list(row)) + "\n").encode("utf-8"))
            counts["rows"] += 1
            counts["on_grid_rows"] += int(row["open_time"] % step == 0)
            counts["flagged_rows"] += int(bool(_flags(row["quality_flags"])))
            counts["nonstandard_duration_rows"] += int(row["close_time"] != row["open_time"] + step - 1)
        counts["missing_rows"] = counts["expected_rows"] - counts["on_grid_rows"]
        counts["off_grid_rows"] = counts["rows"] - counts["on_grid_rows"]
        coverage[interval] = counts
    return digest.hexdigest(), coverage


def _feature_columns(windows):
    columns = ["last_return", "last_range", "last_base_volume"]
    for window in windows:
        columns.extend("%s_%sh" % (name, window) for name in (
            "return", "realized_volatility", "base_volume", "quote_volume", "range"))
    return columns


def _features(hours, decision, windows):
    needed = [decision - offset * HOUR for offset in range(max(windows) + 1, 0, -1)]
    missing = [timestamp for timestamp in needed if timestamp not in hours]
    if missing:
        return None, "missing_hour", {"count": len(missing), "first_5_open_times": missing[:5]}
    invalid = [(timestamp, hours[timestamp][1]) for timestamp in needed if hours[timestamp][1]]
    if invalid:
        return None, "invalid_hour", {"count": len(invalid), "first_5": invalid[:5]}
    bars = [hours[timestamp][0] for timestamp in needed]
    last = bars[-1]
    features = [last["close"] / last["open"] - 1,
                last["high"] / last["low"] - 1, last["volume"]]
    for window in windows:
        selected = bars[-window:]
        closes = [bar["close"] for bar in bars[-window - 1:]]
        features.extend([
            closes[-1] / closes[0] - 1,
            math.sqrt(math.fsum(math.log(right / left) ** 2
                                for left, right in zip(closes, closes[1:]))),
            math.fsum(bar["volume"] for bar in selected),
            math.fsum(bar["quote_volume"] for bar in selected),
            max(bar["high"] for bar in selected) / min(bar["low"] for bar in selected) - 1,
        ])
    if not all(math.isfinite(value) for value in features):
        return None, "nonfinite_feature", {}
    return features, None, None


def _candidates(semantic):
    delay = semantic["execution_delay_seconds"] * 1000
    for split in semantic["splits"]:
        left, right = _ms(split["start"]), _ms(split["end"])
        decision = ((left - delay + HOUR - 1) // HOUR) * HOUR
        while decision + delay < right:
            yield decision, decision + delay, split["name"], right
            decision += HOUR


def _table_digest(connection, table, columns):
    digest, count = hashlib.sha256(), 0
    query = "SELECT %s FROM %s ORDER BY decision_time" % (",".join(columns), table)
    for row in connection.execute(query):
        digest.update((_json(list(row)) + "\n").encode("utf-8"))
        count += 1
    return digest.hexdigest(), count


def _snapshot(semantic_hash, source_hash, dataset_hash, exclusions_hash):
    return _sha({"builder_version": BUILDER_VERSION, "config_sha256": semantic_hash,
                 "source_sha256": source_hash, "dataset_sha256": dataset_hash,
                 "exclusions_sha256": exclusions_hash})


def build(config):
    """Atomically replace a dataset, returning its deterministic coverage report."""
    semantic = _configuration(config)
    feature_columns = _feature_columns(semantic["feature_windows_hours"])
    row_columns = list(TIME_COLUMNS) + ["split", "y"] + feature_columns
    exclusion_columns = ["decision_time", "execution_time", "label_end", "split", "reason", "details"]
    connection = _connect(config)
    try:
        connection.execute("BEGIN IMMEDIATE")
        _require_complete(connection, config)
        source_hash, coverage = _scan_source(connection, semantic)
        hours = {}
        query = "SELECT * FROM candles WHERE symbol=? AND interval='1h' AND open_time>=? AND open_time<? ORDER BY open_time"
        for row in connection.execute(query, (semantic["symbol"], _ms(semantic["start"]), _ms(semantic["end"]))):
            hours[row["open_time"]] = _candle(row, HOUR)
        connection.execute("DROP TABLE IF EXISTS dataset_rows")
        connection.execute("DROP TABLE IF EXISTS dataset_exclusions")
        connection.execute("CREATE TABLE dataset_rows (decision_time INTEGER PRIMARY KEY, signal_time INTEGER NOT NULL, execution_time INTEGER NOT NULL, label_end INTEGER NOT NULL, split TEXT NOT NULL, y REAL NOT NULL,%s)" %
                           ",".join('"%s" REAL NOT NULL' % column for column in feature_columns))
        connection.execute("CREATE TABLE dataset_exclusions (decision_time INTEGER PRIMARY KEY, execution_time INTEGER NOT NULL, label_end INTEGER NOT NULL, split TEXT NOT NULL, reason TEXT NOT NULL, details TEXT NOT NULL)")
        connection.execute("CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY,value TEXT NOT NULL)")
        insert = "INSERT INTO dataset_rows (%s) VALUES (%s)" % (
            ",".join(row_columns), ",".join("?" for _ in row_columns))
        minute_query = "SELECT * FROM candles WHERE symbol=? AND interval='1m' AND open_time=?"
        candidates, included, excluded, reasons = Counter(), Counter(), Counter(), Counter()
        per_split_reasons = {split["name"]: Counter() for split in semantic["splits"]}
        for decision, execution, split, split_end in _candidates(semantic):
            candidates[split] += 1
            label_end = execution + semantic["horizon_hours"] * HOUR
            features, reason, details = None, None, None
            if label_end >= split_end:
                reason, details = "split_boundary", {"split_end": split_end}
            else:
                features, reason, details = _features(hours, decision, semantic["feature_windows_hours"])
            minute_prices = []
            if reason is None:
                for role, timestamp in (("execution", execution), ("label_end", label_end)):
                    raw = connection.execute(minute_query, (semantic["symbol"], timestamp)).fetchone()
                    if raw is None:
                        reason, details = "missing_minute", {"role": role, "open_time": timestamp}
                        break
                    values, error = _candle(raw, MINUTE)
                    if error:
                        reason, details = "invalid_minute", {"role": role, "open_time": timestamp, "error": error}
                        break
                    minute_prices.append(values["open"])
            if reason is None:
                target = minute_prices[1] / minute_prices[0] - 1
                if not math.isfinite(target):
                    reason, details = "nonfinite_label", {}
            if reason is not None:
                connection.execute("INSERT INTO dataset_exclusions VALUES (?,?,?,?,?,?)",
                                   (decision, execution, label_end, split, reason, _json(details)))
                excluded[split] += 1
                reasons[reason] += 1
                per_split_reasons[split][reason] += 1
                continue
            connection.execute(insert, [decision, decision + semantic["feature_delay_seconds"] * 1000,
                                        execution, label_end, split, target] + features)
            included[split] += 1
        dataset_hash, row_count = _table_digest(connection, "dataset_rows", row_columns)
        exclusions_hash, exclusion_count = _table_digest(connection, "dataset_exclusions", exclusion_columns)
        config_hash = _sha(semantic)
        snapshot = _snapshot(config_hash, source_hash, dataset_hash, exclusions_hash)
        report = {
            "builder_version": BUILDER_VERSION, "config_sha256": config_hash,
            "source_sha256": source_hash, "dataset_sha256": dataset_hash,
            "exclusions_sha256": exclusions_hash, "snapshot_id": snapshot,
            "symbol": semantic["symbol"], "feature_columns": feature_columns,
            "time_unit": "UTC Unix milliseconds", "row_columns": row_columns,
            "coverage": coverage, "candidate_rows": sum(candidates.values()),
            "rows": row_count, "excluded_rows": exclusion_count,
            "exclusion_reasons": dict(sorted(reasons.items())),
            "splits": {split["name"]: {
                "candidate_rows": candidates[split["name"]],
                "rows": included[split["name"]], "excluded_rows": excluded[split["name"]],
                "exclusion_reasons": dict(sorted(per_split_reasons[split["name"]].items())),
            } for split in semantic["splits"]},
            "availability": "Completed market bars are assumed available at decision_time + feature_delay_seconds; historical first publication is not verified.",
            "onchain_status": "Excluded: historical point-in-time versions have not been established.",
            "target": "open_1m(label_end) / open_1m(execution_time) - 1, before costs",
            "exclusion_policy": "One primary reason per candidate: split boundary, hourly input, then exact minute prices. No interpolation or nearest-price lookup.",
        }
        metadata = {"builder_version": BUILDER_VERSION, "build_config": semantic,
                    "build_config_hash": config_hash, "source_sha256": source_hash,
                    "dataset_sha256": dataset_hash, "exclusions_sha256": exclusions_hash,
                    "snapshot_id": snapshot, "build_report": report,
                    "build_report_sha256": _sha(report)}
        connection.executemany("INSERT OR REPLACE INTO metadata(key,value) VALUES (?,?)",
                               [(key, _json(value)) for key, value in metadata.items()])
        connection.commit()
        return report
    except BaseException:
        connection.rollback()
        raise
    finally:
        connection.close()


def verify_build(config):
    """Read-only verification of source, table contents and saved build settings."""
    semantic = _configuration(config)
    connection = _connect(config)
    try:
        connection.execute("PRAGMA query_only=ON")
        connection.execute("BEGIN")
        _require_complete(connection, config)
        report = _metadata(connection, "build_report")
        if not report:
            raise ValueError("No saved dataset build to verify")
        errors = []
        if _metadata(connection, "builder_version") != BUILDER_VERSION:
            errors.append("builder_version")
        if _metadata(connection, "build_config") != semantic:
            errors.append("build_config")
        if _metadata(connection, "build_config_hash") != _sha(semantic):
            errors.append("build_config_hash")
        if _sha(report) != _metadata(connection, "build_report_sha256"):
            errors.append("build_report")
        source_hash, coverage = _scan_source(connection, semantic)
        row_columns = list(TIME_COLUMNS) + ["split", "y"] + _feature_columns(semantic["feature_windows_hours"])
        dataset_hash, rows = _table_digest(connection, "dataset_rows", row_columns)
        exclusions_hash, exclusions = _table_digest(connection, "dataset_exclusions",
            ["decision_time", "execution_time", "label_end", "split", "reason", "details"])
        for key, actual in (("source_sha256", source_hash), ("dataset_sha256", dataset_hash),
                            ("exclusions_sha256", exclusions_hash)):
            if actual != _metadata(connection, key) or actual != report.get(key):
                errors.append(key)
        snapshot = _snapshot(_sha(semantic), source_hash, dataset_hash, exclusions_hash)
        if snapshot != _metadata(connection, "snapshot_id") or snapshot != report.get("snapshot_id"):
            errors.append("snapshot_id")
        if rows != report.get("rows") or exclusions != report.get("excluded_rows") or coverage != report.get("coverage"):
            errors.append("counts_or_coverage")
        if errors:
            raise ValueError("Saved dataset integrity mismatch: " + ", ".join(errors))
        return {"verified": True, "snapshot_id": snapshot, "dataset_sha256": dataset_hash,
                "source_sha256": source_hash, "rows": rows, "excluded_rows": exclusions}
    finally:
        connection.close()
