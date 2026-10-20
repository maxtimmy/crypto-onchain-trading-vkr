"""Paired four-hour and 24-hour controls on identical 2022/2023 rows."""

import csv
import gzip
import io
import json
import math
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

from . import attempts
from . import train as training


ROOT = Path(__file__).resolve().parents[2]
HORIZON_VERSION = "paired-horizons-v1"
HORIZONS = {"h4": 4, "h24": 24}
METHODS = ("zero", "train_mean", "catboost")
PARAMS = {**training.FIXED_PARAMS, "depth": 4}
BOUNDARIES = {
    "inner_train": ("2022-01-01T00:00:00Z", "2023-01-01T00:00:00Z"),
    "assessment": ("2023-01-01T00:00:00Z", "2024-01-01T00:00:00Z"),
}
PREDICTION_COLUMNS = ("horizon", "method", "decision_time", "execution_time", "label_end",
                      "split", "y_true", "y_pred")


def _validate_config(config):
    required = {"experiment_id", "dataset_config", "dataset_dir", "expected_snapshot_id", "expected_dataset_sha256",
                "artifact_dir", "budget_ledger", "inner_train_start", "inner_train_end", "assessment_start",
                "assessment_end", "horizons", "params"}
    if required - set(config) or set(config) - (required | {"config_path"}):
        raise ValueError("Unexpected or missing paired-horizon configuration keys")
    if (config["experiment_id"] != "0007_horizon_control" or config["horizons"] != [4, 24]
            or training._json(config["params"]) != training._json(PARAMS)):
        raise ValueError("Use exactly the two fixed horizons and predeclared CatBoost parameters")
    for role, (start, end) in BOUNDARIES.items():
        if (config[role + "_start"], config[role + "_end"]) != (start, end):
            raise ValueError("Horizon controls use only inner 2022 -> 2023 periods")
    for key in ("expected_snapshot_id", "expected_dataset_sha256"):
        value = config[key]
        if (not isinstance(value, str) or len(value) != 64
                or any(char not in "0123456789abcdef" for char in value)):
            raise ValueError("A pinned SHA-256 is required: " + key)
    output, dataset = (training._path(config[key]).resolve() for key in ("artifact_dir", "dataset_dir"))
    if output == dataset or output.is_relative_to(dataset):
        raise ValueError("Horizon artifacts must not be created inside the dataset")


def _database_state(config):
    path = training._path(config["dataset_dir"]) / "market.sqlite3"
    result = {"market.sqlite3": training._sha_file(path)}
    wal = path.with_name(path.name + "-wal")
    if wal.exists():
        result["market.sqlite3-wal"] = training._sha_file(wal)
    return result


def _load_data(config):
    from .horizon_data import load_horizon_data
    return load_horizon_data(config)


def _validate_data(data, verification, selection, config):
    if (verification.get("verified") is not True
            or verification.get("snapshot_id") != config["expected_snapshot_id"]
            or verification.get("dataset_sha256") != config["expected_dataset_sha256"]):
        raise ValueError("Horizon dataset differs from the pinned verified snapshot")
    if set(data) != set(BOUNDARIES) or not all(data.values()):
        raise ValueError("Horizon arrays may contain only nonempty inner_train and assessment periods")
    if selection.get("rows") != {role: len(rows) for role, rows in data.items()}:
        raise ValueError("Horizon row selection counts do not match the data")
    if not isinstance(selection.get("selected_rows_sha256"), str) or len(selection["selected_rows_sha256"]) != 64:
        raise ValueError("Common row selection must have a SHA-256 identity")
    for role, rows in data.items():
        start, end = map(training._ms, BOUNDARIES[role])
        previous = None
        for row in rows:
            decision, signal, execution = (row[name] for name in ("decision_time", "signal_time", "execution_time"))
            if (row["split"] != role or row.get("source_split") != "train"
                    or any(isinstance(value, bool) or not isinstance(value, int)
                           for value in (decision, signal, execution))
                    or decision % training.HOUR or signal != decision + 60_000
                    or execution != decision + 120_000 or execution < start or execution >= end
                    or (previous is not None and execution <= previous)
                    or set(row["targets"]) != set(HORIZONS)):
                raise ValueError("Forbidden year, row ordering or target keys in horizon data")
            features = [row[name] for name in training.FEATURE_COLUMNS]
            if any(isinstance(value, bool) or not isinstance(value, (int, float))
                   or not math.isfinite(value) for value in features):
                raise ValueError("The exact 18 horizon features must be finite numeric values")
            for horizon, hours in HORIZONS.items():
                target = row["targets"][horizon]
                if (set(target) != {"y", "label_end"} or isinstance(target["y"], bool)
                        or not isinstance(target["y"], (int, float)) or not math.isfinite(target["y"])
                        or isinstance(target["label_end"], bool) or not isinstance(target["label_end"], int)
                        or target["label_end"] != execution + hours * training.HOUR
                        or target["label_end"] >= end):
                    raise ValueError("Horizon target crosses its role boundary or has invalid timing")
            previous = execution


def _pool(x, y=None):
    from catboost import Pool
    return Pool(x, label=y, feature_names=list(training.FEATURE_COLUMNS), thread_count=1)


def _features(rows):
    return [[row[name] for name in training.FEATURE_COLUMNS] for row in rows]


def _predict(model, x):
    predicted = [float(value) for value in model.predict(_pool(x), thread_count=1)]
    if len(predicted) != len(x) or not all(math.isfinite(value) for value in predicted):
        raise ValueError("Horizon predictions must be finite and align with the common rows")
    return predicted


def _metrics(data, predictions):
    from .horizon_metrics import summarize_horizons
    return summarize_horizons(data, predictions)


def _write_predictions(path, data, predictions):
    with path.open("xb") as raw:
        with gzip.GzipFile(filename="", fileobj=raw, mode="wb", mtime=0) as compressed:
            with io.TextIOWrapper(compressed, encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=PREDICTION_COLUMNS)
                writer.writeheader()
                for horizon in HORIZONS:
                    for method in METHODS:
                        for role, rows in data.items():
                            for row, predicted in zip(rows, predictions[horizon][method][role], strict=True):
                                target = row["targets"][horizon]
                                writer.writerow({"horizon": horizon, "method": method,
                                                 "decision_time": row["decision_time"], "execution_time": row["execution_time"],
                                                 "label_end": target["label_end"], "split": role,
                                                 "y_true": repr(target["y"]), "y_pred": repr(predicted)})


def _read_predictions(path, data):
    result = {horizon: {method: {role: [] for role in data} for method in METHODS} for horizon in HORIZONS}
    expected_rows = ((horizon, method, role, row) for horizon in HORIZONS for method in METHODS
                     for role, rows in data.items() for row in rows)
    with gzip.open(path, "rt", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or []) != PREDICTION_COLUMNS:
            raise ValueError("Unexpected paired-horizon prediction columns")
        for saved, (horizon, method, role, row) in zip(reader, expected_rows, strict=True):
            target = row["targets"][horizon]
            if (saved["horizon"] != horizon or saved["method"] != method or saved["split"] != role
                    or int(saved["decision_time"]) != row["decision_time"]
                    or int(saved["execution_time"]) != row["execution_time"]
                    or int(saved["label_end"]) != target["label_end"] or float(saved["y_true"]) != target["y"]):
                raise ValueError("Horizon predictions differ from their exact common row/target keys")
            value = float(saved["y_pred"])
            if not math.isfinite(value):
                raise ValueError("Saved horizon prediction is not finite")
            result[horizon][method][role].append(value)
    return result


def _source_snapshot(run_dir, config):
    originals = list((ROOT / "src/vkr_model").glob("*.py"))
    originals += [ROOT / "src/vkr_data/build.py", ROOT / "scripts/compare_horizons.py",
                  training._path(config["dataset_config"])]
    if config.get("config_path"):
        originals.append(training._path(config["config_path"]))
    originals += [ROOT / name for name in ("uv.lock", "pyproject.toml", ".python-version")]
    result = {}
    for original in sorted(set(path.resolve() for path in originals)):
        relative = training._display_path(original)
        destination = run_dir / "source" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(original, destination)
        result[relative] = training._sha_file(destination)
    training._atomic_json(run_dir / "config.json", training._portable(config))
    return result


def _accounting(report, ledger):
    own = [record for record in attempts.load(ledger) if record.get("run_id") == report["run_id"]]
    report["accounting"].update({"ledger_after": attempts.state(ledger),
                                 "started_in_run": attempts.started_count(own)})


def run_horizons(config):
    """Refit exactly h4 and h24 on their common sample, without model selection."""
    _validate_config(config)
    directory, ledger = (training._path(config[key]) for key in ("artifact_dir", "budget_ledger"))
    with training._training_lock(ledger.parent):
        directory.mkdir(parents=True, exist_ok=True)
        run_id = "run-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + "-" + uuid.uuid4().hex[:8]
        run_dir = directory / run_id
        run_dir.mkdir(exist_ok=False)
        report = {"horizon_version": HORIZON_VERSION, "experiment_id": config["experiment_id"],
                  "run_id": run_id, "status": "STARTED", "started_at_utc": training._utc_now(),
                  "config": training._portable(config), "feature_columns": list(training.FEATURE_COLUMNS),
                  "attempts": [], "metrics": {}, "validation_2024_used": False, "final_split_used": False,
                  "automatic_selection": False, "final_refits": 0,
                  "interpretation": "Retrospective horizon control on common 2022/2023 keys; "
                  "both horizons are refitted. Assessment 2023 is not an untouched holdout.",
                  "accounting": {"planned_fits": 2, "ledger_before": attempts.state(ledger),
                                 "started_in_run": 0, "fit_calls": 0}}
        training._atomic_json(run_dir / "report.json", report)
        try:
            report["runtime"] = training._runtime()
            report["database_before"] = _database_state(config)
            data, verification, selection = _load_data(config)
            _validate_data(data, verification, selection, config)
            report.update({"snapshot_id": verification["snapshot_id"], "dataset_sha256": verification["dataset_sha256"],
                           "source_sha256": verification["source_sha256"], "selection": selection,
                           "selected_rows_sha256": selection["selected_rows_sha256"], "rows": selection["rows"],
                           "implementation_sha256": _source_snapshot(run_dir, config)})
            x = {role: _features(rows) for role, rows in data.items()}
            predictions, means = {}, {}
            for horizon in HORIZONS:
                truth = [row["targets"][horizon]["y"] for row in data["inner_train"]]
                means[horizon] = math.fsum(truth) / len(truth)
                predictions[horizon] = {
                    "zero": {role: [0.0] * len(rows) for role, rows in data.items()},
                    "train_mean": {role: [means[horizon]] * len(rows) for role, rows in data.items()}}
            report["baseline_train_mean"] = means
            training._atomic_json(run_dir / "report.json", report)
            backend = training._backend()
            for horizon, hours in HORIZONS.items():
                truth = [row["targets"][horizon]["y"] for row in data["inner_train"]]
                pool = _pool(x["inner_train"], truth)
                attempt = {"attempt_id": run_id + "/" + horizon, "variant": horizon, "horizon_hours": hours,
                           "depth": 4, "status": "STARTED", "started_at_utc": training._utc_now(),
                           "feature_columns": list(training.FEATURE_COLUMNS), "requested_params": config["params"]}
                report["attempts"].append(attempt)
                training._atomic_json(run_dir / "report.json", report)
                attempts.append(ledger, run_id, horizon, 4, "STARTED", config["experiment_id"])
                try:
                    model = backend(**config["params"])
                    report["accounting"]["fit_calls"] += 1
                    training._atomic_json(run_dir / "report.json", report)
                    model.fit(pool)
                    if model.tree_count_ != 300 or list(model.feature_names_) != list(training.FEATURE_COLUMNS):
                        raise ValueError("Each horizon model must have 300 trees and the exact 18-feature allowlist")
                    predicted = {role: _predict(model, values) for role, values in x.items()}
                    filename = horizon + ".cbm"
                    model.save_model(str(run_dir / filename), format="cbm")
                    loaded = backend()
                    loaded.load_model(str(run_dir / filename), format="cbm")
                    full_params = model.get_all_params()
                    if (loaded.tree_count_ != 300 or list(loaded.feature_names_) != list(training.FEATURE_COLUMNS)
                            or not training._same_params(loaded.get_all_params(), full_params)):
                        raise ValueError("Reloaded horizon structure or parameters differ")
                    for role, values in x.items():
                        if not training._same_predictions(predicted[role], _predict(loaded, values)):
                            raise ValueError("Reloaded horizon model does not reproduce predictions")
                    predictions[horizon]["catboost"] = predicted
                    attempt.update({"status": "SUCCESS", "finished_at_utc": training._utc_now(), "model_file": filename,
                                    "tree_count": 300, "params": full_params, "reload_predictions_match": True})
                    attempts.append(ledger, run_id, horizon, 4, "SUCCESS", config["experiment_id"])
                    _accounting(report, ledger)
                    training._atomic_json(run_dir / "report.json", report)
                except BaseException as error:
                    attempt.update({"status": "FAILED", "finished_at_utc": training._utc_now(),
                                    "error": training._error(error, config)})
                    attempts.append(ledger, run_id, horizon, 4, "FAILED", config["experiment_id"])
                    _accounting(report, ledger)
                    training._atomic_json(run_dir / "report.json", report)
                    raise
            report["metrics"] = _metrics(data, predictions)
            _write_predictions(run_dir / "predictions.csv.gz", data, predictions)
            training._atomic_json(run_dir / "metrics.json", report["metrics"])
            training._atomic_json(run_dir / "params.json", {item["variant"]: item["params"] for item in report["attempts"]})
            report["database_after"] = _database_state(config)
            report["dataset_immutable"] = report["database_before"] == report["database_after"]
            _accounting(report, ledger)
            if (not report["dataset_immutable"] or report["accounting"]["started_in_run"] != 2
                    or report["accounting"]["fit_calls"] != 2
                    or report["accounting"]["ledger_after"]["started_attempts"]
                    != report["accounting"]["ledger_before"]["started_attempts"] + 2):
                raise ValueError("Paired-horizon input immutability or attempt accounting mismatch")
            report.update({"status": "SUCCESS", "finished_at_utc": training._utc_now()})
            report["artifact_sha256"] = {path.relative_to(run_dir).as_posix(): training._sha_file(path)
                for path in sorted(run_dir.rglob("*")) if path.is_file() and path.name not in ("report.json", "manifest.json")}
            training._atomic_json(run_dir / "report.json", report)
            training._atomic_json(run_dir / "manifest.json", {"run_id": run_id, "status": "SUCCESS",
                "report_sha256": training._sha_file(run_dir / "report.json"), "artifact_sha256": report["artifact_sha256"]})
            return {**report, "run_dir": str(run_dir)}
        except BaseException as error:
            report.update({"status": "FAILED", "finished_at_utc": training._utc_now(), "error": training._error(error, config)})
            _accounting(report, ledger)
            if "database_before" in report:
                try:
                    report["database_after"] = _database_state(config)
                    report["dataset_immutable"] = report["database_before"] == report["database_after"]
                except OSError:
                    report["dataset_immutable"] = False
            training._atomic_json(run_dir / "report.json", report)
            if isinstance(error, (KeyboardInterrupt, SystemExit)):
                raise
            raise RuntimeError("Horizon controls failed; see " + run_id + "/report.json: "
                               + report["error"]["type"] + ": " + report["error"]["message"]) from error


def verify_horizons(config, run_dir=None):
    """Reload both models and reproduce paired metrics, without additional fits."""
    _validate_config(config)
    directory, ledger = (training._path(config[key]).resolve() for key in ("artifact_dir", "budget_ledger"))
    if run_dir is None:
        candidates = sorted(path.parent for path in directory.glob("run-*/manifest.json"))
        if not candidates:
            raise ValueError("No completed paired-horizon run to verify")
        run_dir = candidates[-1]
    run_dir = Path(run_dir).resolve()
    if run_dir.parent != directory:
        raise ValueError("The horizon run must belong to the configured artifact directory")
    with training._training_lock(ledger.parent):
        ledger_before, database_before = attempts.state(ledger), _database_state(config)
        backend = training._backend()
        with patch.object(backend, "fit", side_effect=RuntimeError("fit is forbidden during horizon verification")) as guard:
            manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
            if (manifest.get("status") != "SUCCESS" or manifest.get("run_id") != run_dir.name
                    or manifest.get("report_sha256") != training._sha_file(run_dir / "report.json")):
                raise ValueError("Paired-horizon manifest or report integrity mismatch")
            report = json.loads((run_dir / "report.json").read_text(encoding="utf-8"))
            if (report.get("config") != training._portable(config) or report.get("status") != "SUCCESS"
                    or report.get("artifact_sha256") != manifest.get("artifact_sha256")
                    or report.get("feature_columns") != list(training.FEATURE_COLUMNS)
                    or report.get("validation_2024_used") is not False or report.get("final_split_used") is not False
                    or report.get("automatic_selection") is not False or report.get("final_refits") != 0):
                raise ValueError("Saved horizon configuration, feature allowlist or scope mismatch")
            for relative, expected in manifest["artifact_sha256"].items():
                path = (run_dir / relative).resolve()
                if not path.is_relative_to(run_dir) or not path.is_file() or training._sha_file(path) != expected:
                    raise ValueError("Paired-horizon artifact integrity mismatch: " + relative)
            if training._runtime() != report["runtime"]:
                raise ValueError("Verify paired horizons in their pinned runtime")
            data, verification, selection = _load_data(config)
            _validate_data(data, verification, selection, config)
            if (verification["snapshot_id"] != report["snapshot_id"] or verification["dataset_sha256"] != report["dataset_sha256"]
                    or selection != report["selection"] or report["rows"] != selection["rows"]):
                raise ValueError("Paired-horizon dataset or common row selection changed")
            predictions = _read_predictions(run_dir / "predictions.csv.gz", data)
            x = {role: _features(rows) for role, rows in data.items()}
            saved_attempts = report["attempts"]
            if [item["variant"] for item in saved_attempts] != list(HORIZONS):
                raise ValueError("The run must contain exactly h4 and h24 refits")
            records = attempts.load(ledger)
            for item, (horizon, hours) in zip(saved_attempts, HORIZONS.items(), strict=True):
                mean = math.fsum(row["targets"][horizon]["y"] for row in data["inner_train"]) / len(data["inner_train"])
                if mean != report["baseline_train_mean"][horizon]:
                    raise ValueError("Horizon baseline does not use only the inner 2022 sample")
                for role, rows in data.items():
                    if (predictions[horizon]["zero"][role] != [0.0] * len(rows)
                            or predictions[horizon]["train_mean"][role] != [mean] * len(rows)):
                        raise ValueError("Horizon baseline predictions differ")
                expected_id = run_dir.name + "/" + horizon
                own = [record for record in records if record.get("attempt_id") == expected_id]
                if (item["status"] != "SUCCESS" or item["attempt_id"] != expected_id or item["horizon_hours"] != hours
                        or item["tree_count"] != 300 or item["requested_params"] != PARAMS
                        or item["feature_columns"] != list(training.FEATURE_COLUMNS)
                        or [record["status"] for record in own] != ["STARTED", "SUCCESS"]
                        or any(record.get("run_id") != run_dir.name or record.get("variant") != horizon
                               or record.get("depth") != 4 or record.get("experiment_id") != config["experiment_id"] for record in own)):
                    raise ValueError("Horizon attempt identity, parameters or ledger mismatch")
                model = backend()
                model_path = (run_dir / item["model_file"]).resolve()
                if not model_path.is_relative_to(run_dir):
                    raise ValueError("Horizon model path escapes the run")
                model.load_model(str(model_path), format="cbm")
                if (model.tree_count_ != 300 or list(model.feature_names_) != list(training.FEATURE_COLUMNS)
                        or not training._same_params(model.get_all_params(), item["params"])):
                    raise ValueError("Reloaded horizon model parameters or structure differ")
                for role, values in x.items():
                    if not training._same_predictions(predictions[horizon]["catboost"][role], _predict(model, values)):
                        raise ValueError("Reloaded horizon predictions differ")
            if json.loads((run_dir / "params.json").read_text(encoding="utf-8")) != {
                    item["variant"]: item["params"] for item in saved_attempts}:
                raise ValueError("Saved full horizon parameters differ")
            metrics = _metrics(data, predictions)
            if metrics != report["metrics"] or metrics != json.loads((run_dir / "metrics.json").read_text(encoding="utf-8")):
                raise ValueError("Paired-horizon metrics cannot be reproduced")
            ledger_after, database_after = attempts.state(ledger), _database_state(config)
            if (ledger_before != ledger_after or database_before != database_after
                    or database_after != report["database_after"] or guard.call_count):
                raise ValueError("Horizon verification changed an input or attempted model fitting")
            return {"verified": True, "run_id": run_dir.name, "run_dir": str(run_dir),
                    "snapshot_id": report["snapshot_id"], "dataset_sha256": report["dataset_sha256"],
                    "selected_rows_sha256": selection["selected_rows_sha256"], "rows": report["rows"],
                    "models_reloaded": list(HORIZONS), "metrics_reproduced": True, "fit_calls": 0,
                    "ledger_before": ledger_before, "ledger_after": ledger_after, "dataset_immutable": True,
                    "validation_2024_used": False, "final_split_used": False}
