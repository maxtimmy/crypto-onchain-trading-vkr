"""Reconstruct signals from past candles, never from future label availability."""

import csv
import gzip
import hashlib
import json
import math
import sqlite3
from collections import Counter
from decimal import Decimal, InvalidOperation

from src.vkr_data import build as builder
from src.vkr_model import train as training
from src.vkr_model.diagnose import _pin_training


FORECAST_METHODS = ("zero", "train_mean", "depth4", "depth6")
HOUR, MINUTE = builder.HOUR, builder.MINUTE


def _hash(value):
    return hashlib.sha256(training._json(value).encode("utf-8")).hexdigest()


def _minute_open(timestamp, raw):
    # Nothing from the rest of this minute is known at its opening instant.
    if isinstance(timestamp, bool) or not isinstance(timestamp, int) or timestamp % MINUTE:
        return None
    try:
        value = Decimal(raw)
    except (InvalidOperation, TypeError, ValueError):
        return None
    return str(value) if value.is_finite() and value > 0 else None


def _read_quotes(connection, start, end):
    quotes, invalid = {}, 0
    for timestamp, raw in connection.execute(
            "SELECT open_time,open FROM candles WHERE symbol='BTCUSDT' AND interval='1m' "
            "AND open_time>=? AND open_time<? ORDER BY open_time", (start, end)):
        value = _minute_open(timestamp, raw)
        if value is None:
            invalid += 1
        elif timestamp in quotes:
            raise ValueError("Duplicate minute opening price")
        else:
            quotes[timestamp] = value
    expected = (end - start) // MINUTE
    return quotes, {"expected_minutes": expected, "valid_open_prices": len(quotes),
                    "invalid_open_prices": invalid, "unavailable_minutes": expected - len(quotes),
                    "open_prices_sha256": _hash(list(quotes.items()))}


def _causal_rows(connection, start, end, holding_minutes):
    """No SELECT of dataset_rows, y, exclusions, or future minute prices here."""
    windows = [24, 168, 720]
    hours = {}
    connection.row_factory = sqlite3.Row
    for row in connection.execute(
            "SELECT " + ",".join(builder.RAW_COLUMNS) + " FROM candles "
            "WHERE symbol='BTCUSDT' AND interval='1h' AND open_time>=? AND open_time<? "
            "ORDER BY open_time", (start - 721 * HOUR, end)):
        if row["open_time"] in hours:
            raise ValueError("Duplicate completed hourly candle")
        hours[row["open_time"]] = builder._candle(row, HOUR)
    rows, counts = [], Counter()
    terminal = end - MINUTE
    for decision in range(start, end, HOUR):
        counts["hourly_decisions"] += 1
        execution = decision + 2 * MINUTE
        # This exclusion is a calendar rule, knowable before any trade.
        if execution + holding_minutes * MINUTE > terminal:
            counts["calendar_horizon_exclusions"] += 1
            continue
        features, reason, _ = builder._features(hours, decision, windows)
        if reason:
            counts["past_feature_exclusions:" + reason] += 1
            continue
        sma24 = math.fsum(hours[decision - i * HOUR][0]["close"] for i in range(1, 25)) / 24
        sma168 = math.fsum(hours[decision - i * HOUR][0]["close"] for i in range(1, 169)) / 168
        rows.append({"decision_time": decision, "signal_time": decision + MINUTE,
                     "execution_time": execution, "features": features,
                     "sma_score": sma24 / sma168 - 1, "sma_enter": sma24 > sma168})
        counts["causal_signal_rows"] += 1
    return rows, dict(sorted(counts.items()))


def _read_predictions(path, start, end):
    result = {method: {} for method in FORECAST_METHODS}
    with gzip.open(path, "rt", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ["decision_time", "execution_time", "label_end", "split",
                                 "y_true", "y_pred", "method"]:
            raise ValueError("Unexpected frozen prediction schema")
        for raw in reader:
            if raw["split"] == "train":
                continue
            if raw["split"] != "validation" or raw["method"] not in result:
                raise ValueError("Forbidden split or method in frozen predictions")
            decision, execution, label_end = (int(raw[key]) for key in (
                "decision_time", "execution_time", "label_end"))
            if (not start <= decision < end or decision % HOUR
                    or execution != decision + 2 * MINUTE
                    or label_end != execution + 4 * HOUR or label_end >= end):
                raise ValueError("Invalid frozen prediction time key")
            value = float(raw["y_pred"])
            if not math.isfinite(value) or decision in result[raw["method"]]:
                raise ValueError("Nonfinite or duplicate frozen prediction")
            result[raw["method"]][decision] = value
            # y_true is intentionally neither converted nor used.
    keys = [set(values) for values in result.values()]
    if not keys[0] or any(key != keys[0] for key in keys[1:]):
        raise ValueError("Frozen forecast methods have different time keys")
    return result


def _model_predictions(run_dir, report, rows):
    from catboost import Pool
    pool = Pool([row["features"] for row in rows],
                feature_names=list(training.FEATURE_COLUMNS), thread_count=1)
    result = {}
    for method in ("depth4", "depth6"):
        model = training._backend()()
        model.load_model(str(run_dir / (method + ".cbm")))
        attempt = next(item for item in report["attempts"] if item["model_file"] == method + ".cbm")
        if (model.tree_count_ != 300 or model.feature_names_ != list(training.FEATURE_COLUMNS)
                or not training._same_params(model.get_all_params(), attempt["params"])):
            raise ValueError("Frozen model structure or parameters changed")
        values = [float(value) for value in model.predict(pool, thread_count=1)]
        if len(values) != len(rows) or not all(math.isfinite(value) for value in values):
            raise ValueError("Invalid frozen model predictions")
        result[method] = values
    return result


def _assemble(rows, archived, predicted, baseline, threshold):
    result = {method: [] for method in (*FORECAST_METHODS, "sma24_168")}
    threshold = Decimal(threshold)
    matched, recovered = 0, 0
    for index, row in enumerate(rows):
        decision = row["decision_time"]
        scores = {"zero": 0.0, "train_mean": baseline,
                  **{method: predicted[method][index] for method in ("depth4", "depth6")}}
        for method in FORECAST_METHODS:
            old = archived[method].get(decision)
            if old is not None:
                if not training._same_predictions([scores[method]], [old]):
                    raise ValueError("Reconstructed features differ from frozen forecast: " + method)
                scores[method] = old
                matched += 1
            else:
                # Future label gaps may have excluded an otherwise tradable signal.
                recovered += 1
            result[method].append({key: row[key] for key in (
                "decision_time", "signal_time", "execution_time")} | {
                    "score": scores[method], "enter": Decimal(str(scores[method])) > threshold})
        result["sma24_168"].append({key: row[key] for key in (
            "decision_time", "signal_time", "execution_time")} | {
                "score": row["sma_score"], "enter": row["sma_enter"]})
    return result, {"archived_prediction_values_verified": matched,
                    "prediction_values_recovered_without_labels": recovered,
                    "signals_sha256": _hash(result)}


def load_inputs(config):
    """Pinned 0003 models, causal 2024 features, and opening quotes only."""
    start, end = builder._ms(config["start"]), builder._ms(config["end"])
    if ((start, end) != (builder._ms("2024-01-01T00:00:00Z"),
                         builder._ms("2025-01-01T00:00:00Z"))
            or config["holding_minutes"] != 240):
        raise ValueError("The first technical simulation is fixed to 2024 and four hours")
    report, manifest = _pin_training(config)
    dataset_config = json.loads(training._path(config["dataset_config"]).read_text(encoding="utf-8"))
    dataset_config["dataset_dir"] = str(training._path(config["dataset_dir"]))
    if (dataset_config["symbol"] != "BTCUSDT"
            or dataset_config["feature_windows_hours"] != [24, 168, 720]
            or dataset_config["feature_delay_seconds"] != 60
            or dataset_config["execution_delay_seconds"] != 120
            or dataset_config["horizon_hours"] != 4):
        raise ValueError("Dataset timing or feature protocol mismatch")
    verification = builder.verify_build(dataset_config)
    for key in ("snapshot_id", "dataset_sha256"):
        if verification[key] != config["expected_" + key] or verification[key] != report[key]:
            raise ValueError("Pinned dataset identity mismatch: " + key)
    if (verification["source_sha256"] != report["source_sha256"]
            or report["feature_columns"] != list(training.FEATURE_COLUMNS)):
        raise ValueError("Frozen feature source identity mismatch")
    database = training._path(config["dataset_dir"]) / "market.sqlite3"
    with sqlite3.connect(database.as_uri() + "?mode=ro", uri=True) as connection:
        quotes, quote_metadata = _read_quotes(connection, start, end)
        rows, row_metadata = _causal_rows(connection, start, end, config["holding_minutes"])
    if not rows:
        raise ValueError("No causal signals available")
    run_dir = training._path(config["training_run"])
    archived = _read_predictions(run_dir / "predictions.csv.gz", start, end)
    predicted = _model_predictions(run_dir, report, rows)
    signals, prediction_metadata = _assemble(
        rows, archived, predicted, report["baseline_train_mean"], config["entry_threshold"])
    return {"quotes": quotes, "signals": signals, "metadata": {
        "dataset": verification, "training_report_sha256": config["expected_report_sha256"],
        "training_manifest_sha256": config["expected_manifest_sha256"],
        "frozen_artifact_sha256": {key: manifest["artifact_sha256"][key] for key in (
            "depth4.cbm", "depth6.cbm", "predictions.csv.gz")},
        "quotes": quote_metadata, "schedule": row_metadata, "predictions": prediction_metadata,
        "feature_columns": list(training.FEATURE_COLUMNS),
        "supervised_row_mask_used": False, "future_label_values_used": False,
        "final_split_used": False, "fit_calls": 0,
        "entry_quote_policy": "exact_open_or_cancel",
        "exit_quote_policy": "first_available_open_at_or_after_due_before_fixed_terminal"}}
