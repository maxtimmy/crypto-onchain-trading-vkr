"""Paired 4/24-hour labels on the same frozen 2022/2023 feature keys."""

import hashlib
import json
import math
import sqlite3
from collections import Counter

from src.vkr_data import build as builder
from . import train as training


ROLES = {
    "inner_train": ("2022-01-01T00:00:00Z", "2023-01-01T00:00:00Z"),
    "assessment": ("2023-01-01T00:00:00Z", "2024-01-01T00:00:00Z"),
}
HORIZONS = {"h4": 4, "h24": 24}
SOURCE_COLUMNS = builder.TIME_COLUMNS + ("split", "y") + training.FEATURE_COLUMNS


def _hash_update(digest, value):
    digest.update((training._json(value) + "\n").encode("utf-8"))


def _source_rows(connection, expected_rows):
    """Read only the original training split; do not extract 2024/2025 arrays."""
    left = builder._ms(ROLES["inner_train"][0])
    right = builder._ms(ROLES["assessment"][1])
    digest, previous, result = hashlib.sha256(), None, []
    query = ("SELECT " + ",".join('"' + key + '"' for key in SOURCE_COLUMNS)
             + " FROM dataset_rows WHERE split='train' ORDER BY execution_time,decision_time")
    for raw in connection.execute(query):
        row = dict(raw)
        decision, available, execution, label_end = (row[key] for key in builder.TIME_COLUMNS)
        if (any(isinstance(value, bool) or not isinstance(value, int)
                for value in (decision, available, execution, label_end))
                or decision % builder.HOUR or available != decision + builder.MINUTE
                or execution != decision + 2 * builder.MINUTE
                or label_end != execution + 4 * builder.HOUR
                or not left <= decision < right or label_end >= right
                or row["split"] != "train" or (previous is not None and execution <= previous)):
            raise ValueError("Forbidden source year, ordering, or target timestamp")
        values = [row["y"]] + [row[name] for name in training.FEATURE_COLUMNS]
        if any(isinstance(value, bool) or not isinstance(value, (int, float))
               or not math.isfinite(value) for value in values):
            raise ValueError("Source features and target must be finite numeric values")
        _hash_update(digest, list(raw))
        previous = execution
        result.append(row)
    if len(result) != expected_rows or not result:
        raise ValueError("Original training row count differs from the pinned build report")
    return result, digest.hexdigest()


def _calendar_candidates(source):
    """Purge both target horizons before requesting any future opening prices."""
    boundary = builder._ms(ROLES["assessment"][0])
    data = {role: [] for role in ROLES}
    excluded = []
    for row in source:
        role = "inner_train" if row["execution_time"] < boundary else "assessment"
        end = builder._ms(ROLES[role][1])
        if row["execution_time"] + 24 * builder.HOUR >= end:
            excluded.append({"decision_time": row["decision_time"], "role": role,
                             "reason": "common_24h_boundary",
                             "h4_also_crosses": row["label_end"] >= end})
        else:
            data[role].append(row)
    return data, excluded


def _label_prices(connection, candidates):
    needed = sorted({row["execution_time"] + offset * builder.HOUR
                     for rows in candidates.values() for row in rows for offset in (0, 4, 24)})
    if any(timestamp >= builder._ms(ROLES["assessment"][1]) for timestamp in needed):
        raise ValueError("Label requests cannot reach 2024 or 2025")
    result = {}
    for offset in range(0, len(needed), 800):
        chunk = needed[offset:offset + 800]
        query = ("SELECT " + ",".join(builder.RAW_COLUMNS) + " FROM candles "
                 "WHERE symbol='BTCUSDT' AND interval='1m' AND open_time IN ("
                 + ",".join("?" for _ in chunk) + ") ORDER BY open_time")
        for raw in connection.execute(query, chunk):
            timestamp = raw["open_time"]
            if timestamp in result:
                raise ValueError("Duplicate exact minute price")
            values, reason = builder._candle(raw, builder.MINUTE)
            result[timestamp] = (values["open"] if values else None, reason)
    return result, {"requested_exact_minutes": len(needed), "returned_exact_minutes": len(result),
                    "first_requested_time": needed[0] if needed else None,
                    "last_requested_time": needed[-1] if needed else None,
                    "missing_exact_minutes": len(needed) - len(result)}


def _paired_rows(candidates, prices):
    data = {role: [] for role in ROLES}
    exclusions = []
    for role, rows in candidates.items():
        for row in rows:
            execution = row["execution_time"]
            endpoints = [("entry", execution), ("h4", execution + 4 * builder.HOUR),
                         ("h24", execution + 24 * builder.HOUR)]
            bad = [(name, timestamp, "missing_minute" if timestamp not in prices else prices[timestamp][1])
                   for name, timestamp in endpoints
                   if timestamp not in prices or prices[timestamp][1] is not None]
            if bad:
                exclusions.append({"decision_time": row["decision_time"], "role": role,
                    "reason": "common_label_price", "endpoints": [
                        {"endpoint": name, "open_time": stamp, "error": reason}
                        for name, stamp, reason in bad]})
                continue
            entry = prices[execution][0]
            targets = {name: {"label_end": execution + horizon * builder.HOUR,
                              "y": prices[execution + horizon * builder.HOUR][0] / entry - 1}
                       for name, horizon in HORIZONS.items()}
            if not all(math.isfinite(target["y"]) for target in targets.values()):
                raise ValueError("A reconstructed target is not finite")
            if targets["h4"]["y"] != row["y"]:
                raise ValueError("Reconstructed 4h target differs from the frozen source")
            data[role].append({**{name: row[name] for name in (
                "decision_time", "signal_time", "execution_time", *training.FEATURE_COLUMNS)},
                "split": role, "source_split": "train", "targets": targets})
    return data, exclusions


def load_horizon_data(config):
    """Read-only paired selection; full-snapshot integrity is checked technically."""
    dataset_config = training._dataset_config(config)
    if (dataset_config["symbol"] != "BTCUSDT" or dataset_config["horizon_hours"] != 4
            or dataset_config["feature_delay_seconds"] != 60
            or dataset_config["execution_delay_seconds"] != 120
            or dataset_config["feature_windows_hours"] != [24, 168, 720]):
        raise ValueError("Use the unchanged market feature and timing protocol")
    verification = builder.verify_build(dataset_config)
    if (verification.get("verified") is not True
            or verification["snapshot_id"] != config["expected_snapshot_id"]
            or verification["dataset_sha256"] != config["expected_dataset_sha256"]):
        raise ValueError("The paired control requires the pinned verified snapshot")
    database = training._path(dataset_config["dataset_dir"]) / "market.sqlite3"
    connection = sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        connection.execute("PRAGMA query_only=ON")
        connection.execute("BEGIN")
        raw_report = connection.execute("SELECT value FROM metadata WHERE key='build_report'").fetchone()
        if raw_report is None:
            raise ValueError("Missing source build report")
        report = json.loads(raw_report[0])
        if (report.get("feature_columns") != list(training.FEATURE_COLUMNS)
                or report.get("snapshot_id") != verification["snapshot_id"]
                or report.get("dataset_sha256") != verification["dataset_sha256"]):
            raise ValueError("Source feature allowlist or identity differs")
        source, source_hash = _source_rows(connection, report["splits"]["train"]["rows"])
        candidates, calendar_exclusions = _calendar_candidates(source)
        prices, price_metadata = _label_prices(connection, candidates)
        data, label_exclusions = _paired_rows(candidates, prices)
    finally:
        connection.close()
    if not all(data.values()):
        raise ValueError("Both common training and assessment parts must be nonempty")
    selected_digest, keys_digest = hashlib.sha256(), hashlib.sha256()
    for role, rows in data.items():
        for row in rows:
            _hash_update(selected_digest, row)
            _hash_update(keys_digest, [role, row["decision_time"], row["execution_time"]])
    exclusions = calendar_exclusions + label_exclusions
    selection = {
        "source_split": "train", "source_rows": len(source),
        "rows": {role: len(rows) for role, rows in data.items()},
        "exclusion_count": len(exclusions), "calendar_exclusions": calendar_exclusions,
        "label_exclusions": label_exclusions,
        "exclusions_by_role": {role: dict(sorted(Counter(
            item["reason"] for item in exclusions if item["role"] == role).items())) for role in ROLES},
        "price_requests": price_metadata, "source_training_rows_sha256": source_hash,
        "selected_rows_sha256": selected_digest.hexdigest(), "common_keys_sha256": keys_digest.hexdigest(),
        "selection_policy": "Intersection of available 4h source features and valid 4h/24h labels inside each part; not a trading admission mask",
        "target_formula": "open_1m(execution_time+horizon)/open_1m(execution_time)-1",
        "validation_2024_values_loaded": False, "final_2025_values_loaded": False}
    if len(source) != sum(selection["rows"].values()) + len(exclusions):
        raise ValueError("Paired selection accounting does not reconcile")
    return data, verification, selection
