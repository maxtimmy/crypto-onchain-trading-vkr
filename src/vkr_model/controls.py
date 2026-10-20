"""Three fixed retrospective controls on 2022 -> 2023, without 2024 arrays."""

import csv
import gzip
import hashlib
import io
import json
import math
import shutil
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

from . import attempts
from . import train as training
from src.vkr_data.build import verify_build


ROOT = Path(__file__).resolve().parents[2]
CONTROL_VERSION = "catboost-inner-controls-v1"
VOLUME_COLUMNS = frozenset(("last_base_volume",) + tuple(
    "%s_%sh" % (name, window) for window in (24, 168, 720)
    for name in ("base_volume", "quote_volume")))
NO_VOLUME_COLUMNS = tuple(name for name in training.FEATURE_COLUMNS if name not in VOLUME_COLUMNS)
COMMON_PARAMS = {key: value for key, value in training.FIXED_PARAMS.items() if key != "l2_leaf_reg"}
COMMON_PARAMS["depth"] = 4
VARIANTS = [
    {"name": "full_l2_3", "l2_leaf_reg": 3, "feature_set": "full"},
    {"name": "full_l2_30", "l2_leaf_reg": 30, "feature_set": "full"},
    {"name": "no_volume_l2_3", "l2_leaf_reg": 3, "feature_set": "no_volume"},
]
METHODS = ("zero", "train_mean") + tuple(variant["name"] for variant in VARIANTS)
BOUNDARIES = {
    "inner_train": ("2022-01-01T00:00:00Z", "2023-01-01T00:00:00Z"),
    "assessment": ("2023-01-01T00:00:00Z", "2024-01-01T00:00:00Z"),
}
FEATURE_SETS = {"full": training.FEATURE_COLUMNS, "no_volume": NO_VOLUME_COLUMNS}
PREDICTION_COLUMNS = training.PREDICTION_COLUMNS


def _validate_config(config):
    required = {"experiment_id", "dataset_config", "dataset_dir", "artifact_dir", "budget_ledger",
                "expected_snapshot_id", "expected_dataset_sha256", "params", "variants",
                "inner_train_start", "inner_train_end", "assessment_start", "assessment_end"}
    if required - set(config) or set(config) - (required | {"config_path"}):
        raise ValueError("Unexpected or missing fixed-control configuration keys")
    if config["experiment_id"] != "0005_catboost_controls":
        raise ValueError("Use the fixed 0005 experiment identity")
    if (training._json(config["params"]) != training._json(COMMON_PARAMS)
            or training._json(config["variants"]) != training._json(VARIANTS)):
        raise ValueError("Use exactly the three predeclared control variants and parameters")
    for role, (start, end) in BOUNDARIES.items():
        if (config[role + "_start"], config[role + "_end"]) != (start, end):
            raise ValueError("The controls use only inner 2022 -> 2023 boundaries")
    for key in ("expected_snapshot_id", "expected_dataset_sha256"):
        value = config[key]
        if (not isinstance(value, str) or len(value) != 64
                or any(char not in "0123456789abcdef" for char in value)):
            raise ValueError("A pinned SHA-256 is required: " + key)
    output = training._path(config["artifact_dir"]).resolve()
    dataset = training._path(config["dataset_dir"]).resolve()
    if output == dataset or output.is_relative_to(dataset):
        raise ValueError("Control artifacts must not be written inside the dataset")


def _database_state(config):
    database = training._path(config["dataset_dir"]) / "market.sqlite3"
    result = {"market.sqlite3": training._sha_file(database)}
    wal = database.with_name(database.name + "-wal")
    if wal.exists():
        result["market.sqlite3-wal"] = training._sha_file(wal)
    return result


def _load_data(config):
    """Integrity scans are technical; model inputs SELECT only original train."""
    dataset_config = training._dataset_config(config)
    verification = verify_build(dataset_config)
    if (verification.get("verified") is not True
            or verification["snapshot_id"] != config["expected_snapshot_id"]
            or verification["dataset_sha256"] != config["expected_dataset_sha256"]):
        raise ValueError("The dataset differs from the pinned verified snapshot")
    database = Path(dataset_config["dataset_dir"]) / "market.sqlite3"
    connection = sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        connection.execute("PRAGMA query_only=ON")
        connection.execute("BEGIN")
        metadata = connection.execute("SELECT value FROM metadata WHERE key='build_report'").fetchone()
        if metadata is None:
            raise ValueError("Dataset has no build report")
        report = json.loads(metadata[0])
        if report.get("feature_columns") != list(training.FEATURE_COLUMNS):
            raise ValueError("The exact 18-feature source allowlist is required")
        if (report.get("snapshot_id") != verification["snapshot_id"]
                or report.get("dataset_sha256") != verification["dataset_sha256"]):
            raise ValueError("Dataset metadata changed during verification")
        columns = ("decision_time", "execution_time", "label_end", "split", "y") + training.FEATURE_COLUMNS
        # Do not reuse training._load_data: it also loads validation year 2024.
        query = ("SELECT " + ",".join('"' + column + '"' for column in columns)
                 + " FROM dataset_rows WHERE split='train' ORDER BY execution_time,decision_time")
        data = {"inner_train": [], "assessment": []}
        source_hash, selected_hash = hashlib.sha256(), hashlib.sha256()
        previous = None
        source_count = 0
        purged = []
        left = training._ms(BOUNDARIES["inner_train"][0])
        boundary = training._ms(BOUNDARIES["assessment"][0])
        end = training._ms(BOUNDARIES["assessment"][1])
        for item in connection.execute(query):
            decision, execution, label_end = (item[name] for name in columns[:3])
            if (item["split"] != "train"
                    or any(isinstance(value, bool) or not isinstance(value, int)
                           for value in (decision, execution, label_end))
                    or decision % training.HOUR or execution != decision + 120_000
                    or label_end != execution + 4 * training.HOUR
                    or execution < left or label_end >= end
                    or (previous is not None and execution <= previous)):
                raise ValueError("Forbidden year, ordering or temporal boundary in source train")
            values = [item["y"]] + [item[name] for name in training.FEATURE_COLUMNS]
            if any(isinstance(value, bool) or not isinstance(value, (int, float))
                   or not math.isfinite(value) for value in values):
                raise ValueError("Control inputs and targets must be finite numeric values")
            previous = execution
            source_count += 1
            source_hash.update((training._json(list(item)) + "\n").encode("utf-8"))
            if execution < boundary and label_end >= boundary:
                purged.append({name: item[name] for name in columns[:3]})
                continue
            role = "inner_train" if execution < boundary else "assessment"
            row = dict(item)
            row["source_split"], row["split"] = "train", role
            data[role].append(row)
            selected_hash.update((training._json([role] + list(item)) + "\n").encode("utf-8"))
        if source_count != report["splits"]["train"]["rows"] or not all(data.values()):
            raise ValueError("Control source counts do not match the verified training split")
        selection = {"source_split": "train", "source_rows": source_count,
                     "rows": {role: len(rows) for role, rows in data.items()},
                     "purged_inner_boundary_rows": len(purged), "purged_rows": purged,
                     "purge_rule": "execution_time < 2023-01-01 and label_end >= 2023-01-01",
                     "source_training_rows_sha256": source_hash.hexdigest(),
                     "selected_rows_sha256": selected_hash.hexdigest()}
        if source_count != sum(selection["rows"].values()) + len(purged):
            raise ValueError("Control selection accounting does not reconcile")
        return data, verification, selection
    finally:
        connection.close()


def _pool(x, columns, y=None):
    from catboost import Pool
    return Pool(x, label=y, feature_names=list(columns), thread_count=1)


def _x(rows, columns):
    return [[row[name] for name in columns] for row in rows]


def _predict(model, rows, columns):
    values = [float(value) for value in model.predict(_pool(_x(rows, columns), columns), thread_count=1)]
    if len(values) != len(rows) or not all(math.isfinite(value) for value in values):
        raise ValueError("Control predictions must be finite and align with the allowed rows")
    return values


def _metrics(data, predictions):
    from .control_metrics import summarize_controls
    return {role: summarize_controls(
        [row["y"] for row in rows], {method: predictions[method][role] for method in METHODS},
        [row["execution_time"] for row in rows]) for role, rows in data.items()}


def _write_predictions(path, data, predictions):
    with path.open("xb") as raw:
        with gzip.GzipFile(filename="", fileobj=raw, mode="wb", mtime=0) as compressed:
            with io.TextIOWrapper(compressed, encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=PREDICTION_COLUMNS)
                writer.writeheader()
                for method in METHODS:
                    for role, rows in data.items():
                        for row, predicted in zip(rows, predictions[method][role], strict=True):
                            writer.writerow({**{name: row[name] for name in PREDICTION_COLUMNS[:4]},
                                             "y_true": repr(row["y"]), "y_pred": repr(predicted),
                                             "method": method})


def _read_predictions(path, data):
    result = {method: {role: [] for role in data} for method in METHODS}
    expected_rows = ((method, role, row) for method in METHODS
                     for role, rows in data.items() for row in rows)
    with gzip.open(path, "rt", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or []) != PREDICTION_COLUMNS:
            raise ValueError("Unexpected control prediction columns")
        for saved, (method, role, row) in zip(reader, expected_rows, strict=True):
            if (saved["method"] != method or saved["split"] != role
                    or any(int(saved[name]) != row[name] for name in PREDICTION_COLUMNS[:3])
                    or float(saved["y_true"]) != row["y"]):
                raise ValueError("Saved control predictions differ from the allowed source rows")
            value = float(saved["y_pred"])
            if not math.isfinite(value):
                raise ValueError("Saved control prediction is not finite")
            result[method][role].append(value)
    return result


def _snapshot_source(run_dir, config):
    originals = list((ROOT / "src/vkr_model").glob("*.py"))
    originals += [ROOT / "src/vkr_data/build.py", ROOT / "scripts/control_models.py",
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


def _update_accounting(report, ledger_path):
    records = attempts.load(ledger_path)
    own = [record for record in records if record.get("run_id") == report["run_id"]]
    report["accounting"].update({"ledger_after": attempts.state(ledger_path),
                                 "started_in_run": attempts.started_count(own)})


def run_controls(config):
    """Fit only the three predeclared controls; failures stop and consume attempts."""
    _validate_config(config)
    directory = training._path(config["artifact_dir"])
    ledger_path = training._path(config["budget_ledger"])
    with training._training_lock(ledger_path.parent):
        directory.mkdir(parents=True, exist_ok=True)
        run_id = "run-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + "-" + uuid.uuid4().hex[:8]
        run_dir = directory / run_id
        run_dir.mkdir(exist_ok=False)
        report = {"control_version": CONTROL_VERSION, "experiment_id": config.get("experiment_id"),
                  "run_id": run_id, "status": "STARTED", "started_at_utc": training._utc_now(),
                  "config": training._portable(config), "attempts": [], "metrics": {},
                  "validation_2024_used": False, "final_split_used": False,
                  "automatic_selection": False, "final_refits": 0,
                  "interpretation": "Retrospective development controls; hypotheses were formed after examining 2024. "
                  "Assessment 2023 was part of the earlier 0003 training sample and is not an untouched holdout.",
                  "accounting": {"planned_fits": 3, "ledger_before": attempts.state(ledger_path),
                                 "started_in_run": 0, "fit_calls": 0}}
        training._atomic_json(run_dir / "report.json", report)
        try:
            report["runtime"] = training._runtime()
            report["database_before"] = _database_state(config)
            data, verification, selection = _load_data(config)
            report.update({"snapshot_id": verification["snapshot_id"],
                           "dataset_sha256": verification["dataset_sha256"],
                           "source_sha256": verification["source_sha256"], "selection": selection,
                           "selected_rows_sha256": selection["selected_rows_sha256"],
                           "rows": selection["rows"],
                           "implementation_sha256": _snapshot_source(run_dir, config)})
            truth = [row["y"] for row in data["inner_train"]]
            mean = math.fsum(truth) / len(truth)
            report["baseline_train_mean"] = mean
            predictions = {
                "zero": {role: [0.0] * len(rows) for role, rows in data.items()},
                "train_mean": {role: [mean] * len(rows) for role, rows in data.items()},
            }
            training._atomic_json(run_dir / "report.json", report)
            backend = training._backend()
            for variant in config["variants"]:
                name = variant["name"]
                columns = FEATURE_SETS[variant["feature_set"]]
                params = {**config["params"], "l2_leaf_reg": variant["l2_leaf_reg"]}
                train_pool = _pool(_x(data["inner_train"], columns), columns, truth)
                attempt = {"attempt_id": run_id + "/" + name, "variant": name, "depth": 4,
                           "status": "STARTED", "started_at_utc": training._utc_now(),
                           "feature_columns": list(columns), "requested_params": params}
                report["attempts"].append(attempt)
                training._atomic_json(run_dir / "report.json", report)
                attempts.append(ledger_path, run_id, name, 4, "STARTED", config["experiment_id"])
                try:
                    model = backend(**params)
                    report["accounting"]["fit_calls"] += 1
                    training._atomic_json(run_dir / "report.json", report)
                    model.fit(train_pool)
                    if model.tree_count_ != 300 or list(model.feature_names_) != list(columns):
                        raise ValueError("A control model must have 300 trees and its exact feature allowlist")
                    predicted = {role: _predict(model, rows, columns) for role, rows in data.items()}
                    filename = name + ".cbm"
                    model.save_model(str(run_dir / filename), format="cbm")
                    loaded = backend()
                    loaded.load_model(str(run_dir / filename), format="cbm")
                    full_params = model.get_all_params()
                    if (loaded.tree_count_ != 300 or list(loaded.feature_names_) != list(columns)
                            or not training._same_params(loaded.get_all_params(), full_params)):
                        raise ValueError("Reloaded control model structure or parameters differ")
                    for role, rows in data.items():
                        if not training._same_predictions(predicted[role], _predict(loaded, rows, columns)):
                            raise ValueError("Reloaded control model does not reproduce predictions")
                    predictions[name] = predicted
                    attempt.update({"status": "SUCCESS", "finished_at_utc": training._utc_now(),
                                    "model_file": filename, "tree_count": 300, "params": full_params,
                                    "reload_predictions_match": True})
                    attempts.append(ledger_path, run_id, name, 4, "SUCCESS", config["experiment_id"])
                    _update_accounting(report, ledger_path)
                    training._atomic_json(run_dir / "report.json", report)
                except BaseException as error:
                    attempt.update({"status": "FAILED", "finished_at_utc": training._utc_now(),
                                    "error": training._error(error, config)})
                    attempts.append(ledger_path, run_id, name, 4, "FAILED", config["experiment_id"])
                    _update_accounting(report, ledger_path)
                    training._atomic_json(run_dir / "report.json", report)
                    raise
            report["metrics"] = _metrics(data, predictions)
            _write_predictions(run_dir / "predictions.csv.gz", data, predictions)
            training._atomic_json(run_dir / "metrics.json", report["metrics"])
            training._atomic_json(run_dir / "params.json", {
                item["variant"]: item["params"] for item in report["attempts"]})
            report["database_after"] = _database_state(config)
            report["dataset_immutable"] = report["database_before"] == report["database_after"]
            _update_accounting(report, ledger_path)
            if (not report["dataset_immutable"] or report["accounting"]["started_in_run"] != 3
                    or report["accounting"]["ledger_after"]["started_attempts"]
                    != report["accounting"]["ledger_before"]["started_attempts"] + 3):
                raise ValueError("Control input immutability or attempt accounting mismatch")
            report.update({"status": "SUCCESS", "finished_at_utc": training._utc_now()})
            report["artifact_sha256"] = {
                path.relative_to(run_dir).as_posix(): training._sha_file(path)
                for path in sorted(run_dir.rglob("*")) if path.is_file()
                and path.name not in ("report.json", "manifest.json")}
            training._atomic_json(run_dir / "report.json", report)
            training._atomic_json(run_dir / "manifest.json", {
                "run_id": run_id, "status": "SUCCESS", "report_sha256": training._sha_file(run_dir / "report.json"),
                "artifact_sha256": report["artifact_sha256"]})
            return {**report, "run_dir": str(run_dir)}
        except BaseException as error:
            report.update({"status": "FAILED", "finished_at_utc": training._utc_now(),
                           "error": training._error(error, config)})
            _update_accounting(report, ledger_path)
            if "database_before" in report:
                try:
                    report["database_after"] = _database_state(config)
                    report["dataset_immutable"] = report["database_before"] == report["database_after"]
                except OSError:
                    report["dataset_immutable"] = False
            training._atomic_json(run_dir / "report.json", report)
            if isinstance(error, (KeyboardInterrupt, SystemExit)):
                raise
            raise RuntimeError("Controls failed; see " + run_id + "/report.json: "
                               + report["error"]["type"] + ": " + report["error"]["message"]) from error


def verify_controls(config, run_dir=None):
    """Reload the three control models and reproduce their 2022/2023 metrics."""
    _validate_config(config)
    directory = training._path(config["artifact_dir"]).resolve()
    ledger_path = training._path(config["budget_ledger"])
    if run_dir is None:
        candidates = sorted(path.parent for path in directory.glob("run-*/manifest.json"))
        if not candidates:
            raise ValueError("No completed controls run to verify")
        run_dir = candidates[-1]
    run_dir = Path(run_dir).resolve()
    if run_dir.parent != directory:
        raise ValueError("The controls run must belong to the configured artifact directory")
    with training._training_lock(ledger_path.parent):
        ledger_before = attempts.state(ledger_path)
        database_before = _database_state(config)
        backend = training._backend()
        with patch.object(backend, "fit", side_effect=RuntimeError("fit is forbidden during controls verification")) as guard:
            manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
            if (manifest.get("status") != "SUCCESS" or manifest.get("run_id") != run_dir.name
                    or manifest.get("report_sha256") != training._sha_file(run_dir / "report.json")):
                raise ValueError("Control manifest or report integrity mismatch")
            report = json.loads((run_dir / "report.json").read_text(encoding="utf-8"))
            if (report.get("status") != "SUCCESS" or report.get("config") != training._portable(config)
                    or report.get("artifact_sha256") != manifest.get("artifact_sha256")
                    or report.get("validation_2024_used") is not False
                    or report.get("final_split_used") is not False
                    or report.get("automatic_selection") is not False or report.get("final_refits") != 0):
                raise ValueError("Saved control configuration or scope differs")
            for relative, expected in manifest["artifact_sha256"].items():
                path = (run_dir / relative).resolve()
                if not path.is_relative_to(run_dir) or not path.is_file() or training._sha_file(path) != expected:
                    raise ValueError("Control artifact integrity mismatch: " + relative)
            if training._runtime() != report.get("runtime"):
                raise ValueError("Verify controls in their pinned runtime")
            data, verification, selection = _load_data(config)
            if (verification["snapshot_id"] != report["snapshot_id"]
                    or verification["dataset_sha256"] != report["dataset_sha256"]
                    or selection != report["selection"]):
                raise ValueError("Control dataset or row selection changed")
            predicted = _read_predictions(run_dir / "predictions.csv.gz", data)
            mean = math.fsum(row["y"] for row in data["inner_train"]) / len(data["inner_train"])
            if mean != report["baseline_train_mean"]:
                raise ValueError("Control baseline is not the inner 2022 training mean")
            for role, rows in data.items():
                if (predicted["zero"][role] != [0.0] * len(rows)
                        or predicted["train_mean"][role] != [mean] * len(rows)):
                    raise ValueError("Saved inner-training baseline predictions differ")
            saved_attempts = report["attempts"]
            if [item["variant"] for item in saved_attempts] != [item["name"] for item in VARIANTS]:
                raise ValueError("The controls must contain exactly the three fixed variants")
            ledger_records = attempts.load(ledger_path)
            for item, variant in zip(saved_attempts, VARIANTS, strict=True):
                columns = FEATURE_SETS[variant["feature_set"]]
                requested = {**COMMON_PARAMS, "l2_leaf_reg": variant["l2_leaf_reg"]}
                expected_id = run_dir.name + "/" + variant["name"]
                own_records = [record for record in ledger_records if record.get("attempt_id") == expected_id]
                statuses = [record["status"] for record in own_records]
                if (item["status"] != "SUCCESS" or item["tree_count"] != 300
                        or item["attempt_id"] != expected_id or statuses != ["STARTED", "SUCCESS"]
                        or item["requested_params"] != requested or item["feature_columns"] != list(columns)
                        or any(record.get("run_id") != run_dir.name or record.get("variant") != variant["name"]
                               or record.get("depth") != 4 or record.get("experiment_id") != config["experiment_id"]
                               for record in own_records)):
                    raise ValueError("Control attempt identity, feature allowlist or ledger mismatch")
                model = backend()
                model_path = (run_dir / item["model_file"]).resolve()
                if not model_path.is_relative_to(run_dir):
                    raise ValueError("Control model path escapes the run")
                model.load_model(str(model_path), format="cbm")
                if (model.tree_count_ != 300 or list(model.feature_names_) != list(columns)
                        or not training._same_params(model.get_all_params(), item["params"])):
                    raise ValueError("Reloaded control model parameters or structure differ")
                for role, rows in data.items():
                    if not training._same_predictions(predicted[variant["name"]][role], _predict(model, rows, columns)):
                        raise ValueError("Reloaded control model predictions differ")
            params = json.loads((run_dir / "params.json").read_text(encoding="utf-8"))
            if params != {item["variant"]: item["params"] for item in saved_attempts}:
                raise ValueError("Saved full control parameters differ")
            metrics = _metrics(data, predicted)
            if (metrics != report["metrics"]
                    or metrics != json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))):
                raise ValueError("Saved control metrics cannot be reproduced")
            ledger_after, database_after = attempts.state(ledger_path), _database_state(config)
            if (ledger_before != ledger_after or database_before != database_after
                    or database_after != report["database_after"] or guard.call_count):
                raise ValueError("Verification changed an input or attempted model fitting")
            return {"verified": True, "run_id": run_dir.name, "run_dir": str(run_dir),
                    "snapshot_id": report["snapshot_id"], "dataset_sha256": report["dataset_sha256"],
                    "selected_rows_sha256": selection["selected_rows_sha256"], "rows": report["rows"],
                    "purged_inner_boundary_rows": selection["purged_inner_boundary_rows"],
                    "models_reloaded": [variant["name"] for variant in VARIANTS],
                    "metrics_reproduced": True, "fit_calls": 0,
                    "ledger_before": ledger_before, "ledger_after": ledger_after,
                    "dataset_immutable": True, "validation_2024_used": False, "final_split_used": False}
