"""Train the two predeclared CatBoost variants without opening the final split.

Integrity verification may stream the complete snapshot, including final rows.
Only the explicit train/validation SELECT below supplies arrays to the model or
to metrics.  Verification reloads saved models and never calls fit.
"""

import csv
import fcntl
import gzip
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import shutil
import sqlite3
import sys
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from src.vkr_data.build import HOUR, verify_build
from . import attempts as attempt_tracking


ROOT = Path(__file__).resolve().parents[2]
TRAINER_VERSION = "catboost-market-v1"
FEATURE_COLUMNS = (
    "last_return", "last_range", "last_base_volume",
    "return_24h", "realized_volatility_24h", "base_volume_24h",
    "quote_volume_24h", "range_24h", "return_168h",
    "realized_volatility_168h", "base_volume_168h", "quote_volume_168h",
    "range_168h", "return_720h", "realized_volatility_720h",
    "base_volume_720h", "quote_volume_720h", "range_720h",
)
FIXED_PARAMS = {
    "loss_function": "RMSE", "iterations": 300, "learning_rate": 0.05,
    "l2_leaf_reg": 3, "random_seed": 42, "task_type": "CPU",
    "thread_count": 1, "has_time": True, "use_best_model": False,
    "allow_writing_files": False, "verbose": False,
}
SPLIT_BOUNDARIES = {
    "train": ("2022-01-01T00:00:00Z", "2024-01-01T00:00:00Z"),
    "validation": ("2024-01-01T00:00:00Z", "2025-01-01T00:00:00Z"),
}
PREDICTION_COLUMNS = (
    "decision_time", "execution_time", "label_end", "split",
    "y_true", "y_pred", "method",
)
METHODS = ("zero", "train_mean", "depth4", "depth6")


def _json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False)


def _utc_now():
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _ms(value):
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("All split boundaries must be timezone-aware")
    return int(parsed.timestamp() * 1000)


def _path(value):
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _sha_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp-" + uuid.uuid4().hex)
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            stream.write(_json(value) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _portable(value):
    """Record useful relative file identifiers without copying absolute paths."""
    if isinstance(value, dict):
        return {key: (_display_path(item) if key in (
            "dataset_dir", "artifact_dir", "dataset_config", "config_path", "budget_ledger") else _portable(item))
                for key, item in value.items()}
    if isinstance(value, list):
        return [_portable(item) for item in value]
    return value


def _display_path(value):
    path = _path(value).resolve()
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.name


def _backend():
    from catboost import CatBoostRegressor
    return CatBoostRegressor


def _pool(x, y):
    from catboost import Pool
    return Pool(x, label=y, feature_names=list(FEATURE_COLUMNS), thread_count=1)


def _runtime():
    versions = {name: importlib.metadata.version(name) for name in (
        "catboost", "numpy", "scipy", "pandas", "matplotlib", "plotly", "six")}
    if sys.version_info[:2] != (3, 12) or versions["catboost"] != "1.2.10":
        raise ValueError("This experiment requires Python 3.12 and CatBoost 1.2.10")
    return {"python": platform.python_version(), "implementation": platform.python_implementation(),
            "system": platform.system(), "machine": platform.machine(),
            "packages": versions}


def _validate_config(config):
    if config.get("depths") != [4, 6]:
        raise ValueError("This experiment permits exactly depths [4, 6]")
    if config.get("params") != FIXED_PARAMS:
        raise ValueError("Use exactly the predeclared CatBoost parameters; no early stopping")
    for split, (start, end) in SPLIT_BOUNDARIES.items():
        if (config.get(split + "_split") != split
                or config.get(split + "_start") != start
                or config.get(split + "_end") != end):
            raise ValueError("Unexpected temporal boundary for " + split)
    for name in ("expected_snapshot_id", "expected_dataset_sha256"):
        value = config.get(name, "")
        if (not isinstance(value, str) or len(value) != 64
                or any(char not in "0123456789abcdef" for char in value)):
            raise ValueError("A pinned SHA-256 is required: " + name)
    if not isinstance(config.get("experiment_id"), str) or not config["experiment_id"]:
        raise ValueError("An experiment_id is required")


def _dataset_config(config):
    result = json.loads(_path(config["dataset_config"]).read_text(encoding="utf-8"))
    result["dataset_dir"] = str(_path(config["dataset_dir"]))
    splits = {item["name"]: item for item in result["splits"]}
    for split, (start, end) in SPLIT_BOUNDARIES.items():
        if split not in splits or (splits[split]["start"], splits[split]["end"]) != (start, end):
            raise ValueError("Dataset and training split boundaries differ")
    if (result.get("feature_windows_hours") != [24, 168, 720]
            or result.get("horizon_hours") != 4
            or result.get("execution_delay_seconds") != 120
            or result.get("feature_delay_seconds") != 60
            or result.get("symbol") != "BTCUSDT"):
        raise ValueError("Dataset settings differ from the approved hourly experiment")
    return result


def _load_data(config):
    """Verify the snapshot, then read only the two allowed splits into memory."""
    dataset_config = _dataset_config(config)
    verification = verify_build(dataset_config)
    if (verification.get("verified") is not True
            or verification["snapshot_id"] != config["expected_snapshot_id"]
            or verification["dataset_sha256"] != config["expected_dataset_sha256"]):
        raise ValueError("The dataset does not match the pinned verified snapshot")
    database = Path(dataset_config["dataset_dir"]) / "market.sqlite3"
    connection = sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        connection.execute("PRAGMA query_only=ON")
        connection.execute("BEGIN")
        row = connection.execute("SELECT value FROM metadata WHERE key='build_report'").fetchone()
        if row is None:
            raise ValueError("The dataset has no build report")
        report = json.loads(row[0])
        if report.get("feature_columns") != list(FEATURE_COLUMNS):
            raise ValueError("The exact 18-feature allowlist does not match the build report")
        if (report.get("snapshot_id") != verification["snapshot_id"]
                or report.get("dataset_sha256") != verification["dataset_sha256"]):
            raise ValueError("Dataset metadata changed during verification")
        # Never replace this projection with SELECT * or remove the split filter.
        columns = ("decision_time", "execution_time", "label_end", "split", "y") + FEATURE_COLUMNS
        query = ("SELECT " + ",".join('"' + name + '"' for name in columns)
                 + " FROM dataset_rows WHERE split IN (?,?) ORDER BY execution_time,decision_time")
        data = {"train": [], "validation": []}
        selected_digest = hashlib.sha256()
        previous_execution = None
        decisions = set()
        for item in connection.execute(query, ("train", "validation")):
            split = item["split"]
            if split not in data:
                raise ValueError("Forbidden split supplied to the model")
            decision, execution, label_end = (item[key] for key in columns[:3])
            if any(isinstance(value, bool) or not isinstance(value, int)
                   for value in (decision, execution, label_end)):
                raise ValueError("Dataset times must be integer UTC milliseconds")
            left, right = map(_ms, SPLIT_BOUNDARIES[split])
            if (decision % HOUR or execution != decision + 120_000
                    or label_end != execution + 4 * HOUR
                    or execution < left or label_end >= right
                    or (previous_execution is not None and execution <= previous_execution)
                    or decision in decisions):
                raise ValueError("Temporal boundary, order or label-overlap violation")
            numeric = [item["y"]] + [item[name] for name in FEATURE_COLUMNS]
            if any(isinstance(value, bool) or not isinstance(value, (int, float))
                   or not math.isfinite(value) for value in numeric):
                raise ValueError("Inputs and targets must be finite numeric values")
            previous_execution = execution
            decisions.add(decision)
            data[split].append(dict(item))
            selected_digest.update((_json(list(item)) + "\n").encode("utf-8"))
        for split, rows in data.items():
            if not rows or len(rows) != report["splits"][split]["rows"]:
                raise ValueError("Unexpected number of allowed rows for " + split)
        return data, verification, selected_digest.hexdigest()
    finally:
        connection.close()


def _arrays(data):
    return {split: {
        "x": [[row[name] for name in FEATURE_COLUMNS] for row in rows],
        "y": [row["y"] for row in rows],
        "times": [row["execution_time"] for row in rows],
    } for split, rows in data.items()}


def _predictions(model, inputs):
    values = [float(value) for value in model.predict(inputs, thread_count=1)]
    if len(values) != len(inputs) or not all(math.isfinite(value) for value in values):
        raise ValueError("Model predictions must be finite and match the input row count")
    return values


def _same_predictions(left, right):
    return len(left) == len(right) and all(
        math.isclose(a, b, rel_tol=0, abs_tol=1e-12) for a, b in zip(left, right))


def _same_params(left, right):
    """Allow CBM's decimal rounding of floats, preserving types and structure.

    CatBoost 1.2.10 serializes some full-precision parameter floats with ten
    significant digits. Model-file hashes remain exact; only the comparison
    between fitted and reloaded parameter metadata needs this tolerance.
    """
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(
            _same_params(value, right[key]) for key, value in left.items())
    if isinstance(left, (list, tuple)):
        return len(left) == len(right) and all(
            _same_params(a, b) for a, b in zip(left, right))
    if isinstance(left, float):
        return (math.isfinite(left) and math.isfinite(right)
                and math.isclose(left, right, rel_tol=1e-9, abs_tol=1e-12))
    return left == right


def _metrics(arrays, predictions):
    from .metrics import evaluate
    return {method: {split: evaluate(values["y"], predictions[method][split], values["times"])
                     for split, values in arrays.items()}
            for method in predictions}


def _write_predictions(path, data, predictions):
    # mtime=0 removes a wall-clock dependency from gzip; the run path is unique.
    import io
    with Path(path).open("xb") as raw:
        with gzip.GzipFile(filename="", fileobj=raw, mode="wb", mtime=0) as compressed:
            with io.TextIOWrapper(compressed, encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=PREDICTION_COLUMNS)
                writer.writeheader()
                for method in METHODS:
                    for split, rows in data.items():
                        for row, prediction in zip(rows, predictions[method][split]):
                            writer.writerow({**{name: row[name] for name in PREDICTION_COLUMNS[:4]},
                                             "y_true": repr(row["y"]), "y_pred": repr(prediction),
                                             "method": method})


def _read_predictions(path, data):
    result = {method: {split: [] for split in data} for method in METHODS}
    expected_order = ((method, split, row) for method in METHODS
                      for split, rows in data.items() for row in rows)
    with gzip.open(path, "rt", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or []) != PREDICTION_COLUMNS:
            raise ValueError("Unexpected prediction columns")
        for stored, expected in zip(reader, expected_order, strict=True):
            method, split, row = expected
            if (stored["method"] != method or stored["split"] != split
                    or any(int(stored[name]) != row[name] for name in PREDICTION_COLUMNS[:3])
                    or float(stored["y_true"]) != row["y"]):
                raise ValueError("Predictions do not match the allowed dataset rows")
            prediction = float(stored["y_pred"])
            if not math.isfinite(prediction):
                raise ValueError("Stored prediction is not finite")
            result[method][split].append(prediction)
    return result


def _source_snapshot(run_dir, config):
    candidates = list((ROOT / "src/vkr_model").glob("*.py"))
    candidates += [ROOT / "src/vkr_data/build.py", _path(config["dataset_config"])]
    if config.get("config_path"):
        candidates.append(_path(config["config_path"]))
    candidates += [ROOT / name for name in (
        "scripts/train_model.py", "requirements-model.lock", "uv.lock", "pyproject.toml", ".python-version")
        if (ROOT / name).is_file()]
    result = {}
    for original in sorted(set(path.resolve() for path in candidates)):
        relative = _display_path(original)
        destination = run_dir / "source" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(original, destination)
        result[relative] = _sha_file(destination)
    _atomic_json(run_dir / "config.json", _portable(config))
    return result


@contextmanager
def _training_lock(directory):
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".train.lock").open("a", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def _ledger(path):
    return attempt_tracking.load(path)


def _record_attempt(path, run_id, depth, status):
    record = {"run_id": run_id, "depth": depth, "status": status, "recorded_at_utc": _utc_now()}
    with path.open("a", encoding="utf-8") as stream:
        stream.write(_json(record) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _error(error, config):
    # Avoid absolute source/data paths in a failure report; never capture locals.
    message = str(error)
    for path in sorted((str(ROOT), str(_path(config["dataset_dir"])),
                        str(_path(config["artifact_dir"]))), key=len, reverse=True):
        message = message.replace(path, "<local>")
    return {"type": type(error).__name__, "message": message[:2000]}


def train(config):
    """Run exactly the two configured fits; failures are durable and re-raised."""
    artifact_dir = _path(config["artifact_dir"])
    ledger_path = _path(config["budget_ledger"]) if config.get("budget_ledger") else artifact_dir / "attempts.jsonl"
    with _training_lock(ledger_path.parent):
        artifact_dir.mkdir(parents=True, exist_ok=True)
        run_id = "run-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + "-" + uuid.uuid4().hex[:8]
        run_dir = artifact_dir / run_id
        run_dir.mkdir(exist_ok=False)
        report = {"trainer_version": TRAINER_VERSION, "experiment_id": config.get("experiment_id"),
                  "run_id": run_id, "status": "STARTED", "started_at_utc": _utc_now(),
                  "config": _portable(config), "feature_columns": list(FEATURE_COLUMNS),
                  "attempts": [], "metrics": {}, "final_split_used": False}
        _atomic_json(run_dir / "report.json", report)
        try:
            _validate_config(config)
            started = attempt_tracking.started_count(_ledger(ledger_path))
            report["runtime"] = _runtime()
            data, verification, selected_hash = _load_data(config)
            report.update({"snapshot_id": verification["snapshot_id"],
                           "dataset_sha256": verification["dataset_sha256"],
                           "source_sha256": verification["source_sha256"],
                           "selected_rows_sha256": selected_hash,
                           "rows": {split: len(rows) for split, rows in data.items()},
                           "implementation_sha256": _source_snapshot(run_dir, config)})
            arrays = _arrays(data)
            mean = math.fsum(arrays["train"]["y"]) / len(arrays["train"]["y"])
            report["baseline_train_mean"] = mean
            predictions = {
                "zero": {split: [0.0] * len(rows) for split, rows in data.items()},
                "train_mean": {split: [mean] * len(rows) for split, rows in data.items()},
            }
            report["metrics"] = _metrics(arrays, predictions)
            _atomic_json(run_dir / "report.json", report)
            backend = _backend()
            train_pool = _pool(arrays["train"]["x"], arrays["train"]["y"])
            for depth in config["depths"]:
                attempt = {"depth": depth, "status": "STARTED", "started_at_utc": _utc_now()}
                report["attempts"].append(attempt)
                _atomic_json(run_dir / "report.json", report)
                _record_attempt(ledger_path, run_id, depth, "STARTED")
                try:
                    model = backend(**{**config["params"], "depth": depth})
                    model.fit(train_pool)
                    if model.tree_count_ != 300:
                        raise ValueError("The fitted model must contain exactly 300 trees")
                    prediction = {split: _predictions(model, values["x"])
                                  for split, values in arrays.items()}
                    filename = "depth%d.cbm" % depth
                    model.save_model(str(run_dir / filename), format="cbm")
                    loaded = backend()
                    loaded.load_model(str(run_dir / filename), format="cbm")
                    if loaded.tree_count_ != 300 or list(loaded.feature_names_) != list(FEATURE_COLUMNS):
                        raise ValueError("Reloaded model structure differs from the declared model")
                    for split, values in arrays.items():
                        if not _same_predictions(prediction[split], _predictions(loaded, values["x"])):
                            raise ValueError("Reloaded model does not reproduce predictions")
                    method = "depth%d" % depth
                    predictions[method] = prediction
                    report["metrics"].update(_metrics(arrays, {method: prediction}))
                    attempt.update({"status": "SUCCESS", "finished_at_utc": _utc_now(),
                                    "model_file": filename, "tree_count": model.tree_count_,
                                    "params": model.get_all_params(),
                                    "reload_predictions_match": True})
                    _record_attempt(ledger_path, run_id, depth, "SUCCESS")
                    _atomic_json(run_dir / "report.json", report)
                except BaseException as error:
                    attempt.update({"status": "FAILED", "finished_at_utc": _utc_now(),
                                    "error": _error(error, config)})
                    _record_attempt(ledger_path, run_id, depth, "FAILED")
                    _atomic_json(run_dir / "report.json", report)
                    raise
            _write_predictions(run_dir / "predictions.csv.gz", data, predictions)
            _atomic_json(run_dir / "metrics.json", report["metrics"])
            report["status"] = "SUCCESS"
            report["finished_at_utc"] = _utc_now()
            # Keep the historical report key while dropping the arbitrary cap.
            report["budget"] = {"comparative_limit": None,
                                "attempts_before_run": started, "attempts_in_run": 2,
                                "attempts_after_run": started + 2,
                                "final_refits_in_run": 0}
            report["artifact_sha256"] = {
                path.relative_to(run_dir).as_posix(): _sha_file(path)
                for path in sorted(run_dir.rglob("*")) if path.is_file()
                and path.name not in ("report.json", "manifest.json")}
            _atomic_json(run_dir / "report.json", report)
            _atomic_json(run_dir / "manifest.json", {
                "run_id": run_id, "status": "SUCCESS", "report_sha256": _sha_file(run_dir / "report.json"),
                "artifact_sha256": report["artifact_sha256"]})
            return {**report, "run_dir": str(run_dir)}
        except BaseException as error:
            report.update({"status": "FAILED", "finished_at_utc": _utc_now(),
                           "error": _error(error, config)})
            _atomic_json(run_dir / "report.json", report)
            if isinstance(error, (KeyboardInterrupt, SystemExit)):
                raise
            raise RuntimeError("Training failed; see " + run_id + "/report.json: "
                               + report["error"]["type"] + ": " + report["error"]["message"]) from error


def verify_run(config, run_dir=None):
    """Check hashes, row identity, saved predictions and loaded models; no fits."""
    _validate_config(config)
    artifact_dir = _path(config["artifact_dir"]).resolve()
    if run_dir is None:
        candidates = sorted(path.parent for path in artifact_dir.glob("run-*/manifest.json"))
        if not candidates:
            raise ValueError("No completed training run to verify")
        run_dir = candidates[-1]
    run_dir = Path(run_dir).resolve()
    if run_dir.parent != artifact_dir:
        raise ValueError("The run must belong to the configured artifact directory")
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("status") != "SUCCESS" or manifest.get("run_id") != run_dir.name:
        raise ValueError("No successful saved run")
    if manifest.get("report_sha256") != _sha_file(run_dir / "report.json"):
        raise ValueError("Training report integrity mismatch")
    report = json.loads((run_dir / "report.json").read_text(encoding="utf-8"))
    if (report.get("status") != "SUCCESS" or report.get("config") != _portable(config)
            or report.get("artifact_sha256") != manifest.get("artifact_sha256")
            or report.get("feature_columns") != list(FEATURE_COLUMNS)
            or report.get("final_split_used") is not False):
        raise ValueError("Saved run configuration or feature allowlist mismatch")
    for name, expected in manifest["artifact_sha256"].items():
        path = (run_dir / name).resolve()
        if not path.is_relative_to(run_dir) or not path.is_file() or _sha_file(path) != expected:
            raise ValueError("Training artifact integrity mismatch: " + name)
    runtime = _runtime()
    if runtime != report.get("runtime"):
        raise ValueError("Verify the run in its pinned runtime")
    data, verification, selected_hash = _load_data(config)
    if (verification["snapshot_id"] != report["snapshot_id"]
            or selected_hash != report.get("selected_rows_sha256")
            or report.get("rows") != {split: len(rows) for split, rows in data.items()}):
        raise ValueError("Run and current dataset row identity differ")
    arrays = _arrays(data)
    predictions = _read_predictions(run_dir / "predictions.csv.gz", data)
    mean = math.fsum(arrays["train"]["y"]) / len(arrays["train"]["y"])
    if mean != report["baseline_train_mean"]:
        raise ValueError("Training-only baseline mismatch")
    for split, rows in data.items():
        if (predictions["zero"][split] != [0.0] * len(rows)
                or predictions["train_mean"][split] != [mean] * len(rows)):
            raise ValueError("Saved baseline predictions mismatch")
    attempts = report.get("attempts", [])
    if [item["depth"] for item in attempts] != [4, 6] or any(
            item["status"] != "SUCCESS" or item["tree_count"] != 300 for item in attempts):
        raise ValueError("The run must contain exactly two successful 300-tree fits")
    backend = _backend()
    for attempt in attempts:
        model = backend()
        model.load_model(str(run_dir / attempt["model_file"]), format="cbm")
        if (model.tree_count_ != 300 or list(model.feature_names_) != list(FEATURE_COLUMNS)
                or not _same_params(model.get_all_params(), attempt["params"])):
            raise ValueError("Reloaded model parameters or structure differ")
        for split, values in arrays.items():
            if not _same_predictions(predictions["depth%d" % attempt["depth"]][split],
                                     _predictions(model, values["x"])):
                raise ValueError("Saved predictions differ from loaded model predictions")
    metrics = _metrics(arrays, predictions)
    if (metrics != report["metrics"]
            or metrics != json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))):
        raise ValueError("Saved metrics cannot be reproduced")
    return {"verified": True, "run_id": run_dir.name, "run_dir": str(run_dir),
            "snapshot_id": report["snapshot_id"], "dataset_sha256": report["dataset_sha256"],
            "selected_rows_sha256": selected_hash, "rows": report["rows"],
            "models_reloaded": ["depth4", "depth6"], "metrics_reproduced": True,
            "final_split_used": False, "fit_calls": 0}
