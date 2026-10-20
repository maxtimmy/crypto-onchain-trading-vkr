"""Read-only diagnostics of the pinned first CatBoost run; fitting is forbidden."""

import hashlib
import json
import shutil
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

from . import train as training


ROOT = Path(__file__).resolve().parents[2]
DIAGNOSTIC_VERSION = "catboost-diagnostics-v1"
PATH_KEYS = {"training_config", "training_run", "artifact_dir", "config_path"}
REQUIRED_KEYS = {"experiment_id", "training_config", "training_run", "artifact_dir",
                 "expected_report_sha256", "expected_manifest_sha256"}


def _path(value):
    path = Path(value)
    return (path if path.is_absolute() else ROOT / path).resolve()


def _display_path(value):
    path = _path(value)
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.name


def _portable_config(config):
    return {key: _display_path(value) if key in PATH_KEYS else value
            for key, value in config.items()}


def _read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _semantic_hash(value):
    return hashlib.sha256(training._json(value).encode("utf-8")).hexdigest()


def _validate_config(config):
    if set(config) - (REQUIRED_KEYS | {"config_path"}) or REQUIRED_KEYS - set(config):
        raise ValueError("Unexpected or missing diagnostic configuration keys")
    if config["experiment_id"] != "0004_catboost_diagnostics":
        raise ValueError("Use the fixed 0004 diagnostics experiment")
    for name in ("expected_report_sha256", "expected_manifest_sha256"):
        value = config[name]
        if (not isinstance(value, str) or len(value) != 64
                or any(char not in "0123456789abcdef" for char in value)):
            raise ValueError("A pinned SHA-256 is required: " + name)


def _load_training_config(path):
    result = _read_json(path)
    for key in ("dataset_config", "dataset_dir", "artifact_dir", "budget_ledger"):
        result[key] = str(_path(result[key]))
    result["config_path"] = str(_path(path))
    return result


@contextmanager
def _fit_guard():
    # Resolves to catboost.CatBoostRegressor in production. Tests substitute a
    # dummy backend so they require neither model fitting nor downloaded data.
    with patch.object(training._backend(), "fit", side_effect=RuntimeError(
            "CatBoost fit is forbidden during diagnostics")) as guard:
        yield guard


def _pin_training(config):
    run_dir = _path(config["training_run"])
    report_path, manifest_path = run_dir / "report.json", run_dir / "manifest.json"
    if (training._sha_file(report_path) != config["expected_report_sha256"]
            or training._sha_file(manifest_path) != config["expected_manifest_sha256"]):
        raise ValueError("Pinned training report or manifest integrity mismatch")
    report, manifest = _read_json(report_path), _read_json(manifest_path)
    if (manifest.get("status") != "SUCCESS" or report.get("status") != "SUCCESS"
            or manifest.get("report_sha256") != config["expected_report_sha256"]
            or manifest.get("artifact_sha256") != report.get("artifact_sha256")
            or report.get("run_id") != run_dir.name or manifest.get("run_id") != run_dir.name
            or report.get("final_split_used") is not False):
        raise ValueError("Training run is not an intact successful market-only run")
    required = {"depth4.cbm", "depth6.cbm", "predictions.csv.gz"}
    artifacts = manifest.get("artifact_sha256", {})
    if not required.issubset(artifacts):
        raise ValueError("Training manifest is missing required model or prediction hashes")
    for relative, expected in artifacts.items():
        path = (run_dir / relative).resolve()
        if (not path.is_relative_to(run_dir) or not path.is_file()
                or training._sha_file(path) != expected):
            raise ValueError("Pinned training artifact integrity mismatch: " + relative)
    return report, manifest


def _tree_hashes(directory):
    return {path.relative_to(directory).as_posix(): training._sha_file(path)
            for path in sorted(directory.rglob("*")) if path.is_file()}


def _ledger_state(path):
    if not path.exists():
        return {"exists": False, "sha256": None, "records": 0, "started_attempts": 0}
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    started = {(item["run_id"], item["depth"]) for item in records if item["status"] == "STARTED"}
    return {"exists": True, "sha256": training._sha_file(path),
            "records": len(records), "started_attempts": len(started)}


def _input_state(config, training_config):
    database = _path(training_config["dataset_dir"]) / "market.sqlite3"
    database_files = {"market.sqlite3": training._sha_file(database)}
    wal = database.with_name(database.name + "-wal")
    if wal.exists():
        database_files["market.sqlite3-wal"] = training._sha_file(wal)
    return {"training_files_sha256": _tree_hashes(_path(config["training_run"])),
            "database_files_sha256": database_files,
            "budget_ledger": _ledger_state(_path(training_config["budget_ledger"]))}


def _assert_allowed_inputs(data, predictions, verification, loaded_hash):
    if set(data) != {"train", "validation"} or set(predictions) != set(training.METHODS):
        raise ValueError("Diagnostics only accept train/validation and the four fixed methods")
    if (verification.get("verified") is not True or verification.get("final_split_used") is not False
            or verification.get("fit_calls") != 0
            or verification.get("selected_rows_sha256") != loaded_hash):
        raise ValueError("Training verification or selected row identity mismatch")
    for split, rows in data.items():
        if not rows or len(rows) != verification["rows"][split]:
            raise ValueError("Diagnostic rows differ from verified model rows")
        start, end = map(training._ms, training.SPLIT_BOUNDARIES[split])
        for row in rows:
            if (row["split"] != split or row["execution_time"] < start
                    or row["label_end"] >= end or row["label_end"] <= row["execution_time"]):
                raise ValueError("Forbidden final rows or invalid temporal boundary in diagnostics")
        for method in training.METHODS:
            if set(predictions[method]) != set(data) or len(predictions[method][split]) != len(rows):
                raise ValueError("Diagnostic predictions do not align with the allowed rows")


def _snapshot_source(run_dir, config, training_config):
    originals = list((ROOT / "src/vkr_model").glob("*.py"))
    originals += [ROOT / "src/vkr_data/build.py", ROOT / "scripts/diagnose_model.py",
                  ROOT / "scripts/train_model.py", _path(config["training_config"]),
                  _path(training_config["dataset_config"])]
    if config.get("config_path"):
        originals.append(_path(config["config_path"]))
    originals += [ROOT / name for name in ("uv.lock", "pyproject.toml", ".python-version")]
    result = {}
    for original in sorted(set(path.resolve() for path in originals)):
        if not original.is_file():
            raise FileNotFoundError("Missing reproducibility source: " + _display_path(original))
        relative = _display_path(original)
        destination = run_dir / "source" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(original, destination)
        result[relative] = training._sha_file(destination)
    training._atomic_json(run_dir / "config.json", _portable_config(config))
    return result


def _repeated_summary(artifact_dir, run_dir, config, semantic_hash):
    previous = []
    for manifest_path in sorted(artifact_dir.glob("run-*/manifest.json")):
        previous_dir = manifest_path.parent
        if previous_dir == run_dir:
            continue
        manifest = _read_json(manifest_path)
        report_path = previous_dir / "report.json"
        if manifest.get("report_sha256") != training._sha_file(report_path):
            raise ValueError("Previous diagnostic report integrity mismatch")
        report = _read_json(report_path)
        if report.get("config") != _portable_config(config):
            continue
        summary_path = previous_dir / "summary.json"
        if (report.get("status") != "SUCCESS"
                or manifest.get("artifact_sha256", {}).get("summary.json") != training._sha_file(summary_path)
                or _semantic_hash(_read_json(summary_path)) != report.get("summary_sha256")):
            raise ValueError("Previous diagnostic summary integrity mismatch")
        if report["summary_sha256"] != semantic_hash:
            raise ValueError("Repeated diagnostics summary differs for the same pinned inputs")
        previous.append(previous_dir.name)
    return {"previous_matching_runs": len(previous), "previous_run_ids": previous,
            "summary_identity_verified": bool(previous), "all_summary_hashes_equal": True}


def _analysis(data, predictions, baseline_train_mean):
    from .diagnostics import analyze
    return analyze(data, predictions, baseline_train_mean)


def _plots(summary, output_dir):
    from .diagnostic_plots import plot_diagnostics
    return plot_diagnostics(summary, output_dir)


def _error(error, config, training_config):
    message = str(error)
    paths = [str(ROOT)] + [str(_path(config[key])) for key in PATH_KEYS if key in config]
    if training_config:
        paths.extend(str(_path(training_config[key])) for key in (
            "dataset_config", "dataset_dir", "artifact_dir", "budget_ledger", "config_path"))
    for path in sorted(paths, key=len, reverse=True):
        message = message.replace(path, "<local>")
    return {"type": type(error).__name__, "message": message[:2000]}


def run_diagnostics(config):
    """Save one immutable diagnostic run without fitting or opening final data."""
    _validate_config(config)
    artifact_dir = _path(config["artifact_dir"])
    training_dir = _path(config["training_run"])
    if artifact_dir == training_dir or artifact_dir.is_relative_to(training_dir):
        raise ValueError("Diagnostic output must not be inside the training run")
    training_config = _load_training_config(_path(config["training_config"]))
    dataset_dir = _path(training_config["dataset_dir"])
    if artifact_dir == dataset_dir or artifact_dir.is_relative_to(dataset_dir):
        raise ValueError("Diagnostic output must not be inside the dataset")
    artifact_dir.mkdir(parents=True, exist_ok=True)
    run_id = "run-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + "-" + uuid.uuid4().hex[:8]
    run_dir = artifact_dir / run_id
    run_dir.mkdir(exist_ok=False)
    report = {"diagnostic_version": DIAGNOSTIC_VERSION, "experiment_id": config["experiment_id"],
              "run_id": run_id, "status": "STARTED", "started_at_utc": training._utc_now(),
              "config": _portable_config(config), "fit_calls": 0, "final_split_used": False}
    training._atomic_json(run_dir / "report.json", report)
    inputs_before = None
    guard = None
    try:
        with _fit_guard() as guard:
            training_report, training_manifest = _pin_training(config)
            inputs_before = _input_state(config, training_config)
            report["inputs_before"] = inputs_before
            report["runtime"] = training._runtime()
            report["implementation_sha256"] = _snapshot_source(run_dir, config, training_config)
            report["training_artifact_sha256"] = {
                "report.json": config["expected_report_sha256"],
                "manifest.json": config["expected_manifest_sha256"],
                **{name: training_manifest["artifact_sha256"][name] for name in (
                    "depth4.cbm", "depth6.cbm", "predictions.csv.gz")}}
            training._atomic_json(run_dir / "report.json", report)
            verification = training.verify_run(training_config, run_dir=training_dir)
            data, dataset_verification, loaded_hash = training._load_data(training_config)
            predictions = training._read_predictions(training_dir / "predictions.csv.gz", data)
            _assert_allowed_inputs(data, predictions, verification, loaded_hash)
            if (dataset_verification["snapshot_id"] != verification["snapshot_id"]
                    or dataset_verification["dataset_sha256"] != verification["dataset_sha256"]
                    or training_report["selected_rows_sha256"] != loaded_hash):
                raise ValueError("Dataset snapshot changed after training verification")
            report["dataset"] = {"snapshot_id": verification["snapshot_id"],
                                 "dataset_sha256": verification["dataset_sha256"],
                                 "source_sha256": dataset_verification["source_sha256"],
                                 "selected_rows_sha256": loaded_hash, "rows": verification["rows"]}
            report["training_verification"] = {
                key: value for key, value in verification.items() if key != "run_dir"}
            summary = _analysis(data, predictions, training_report["baseline_train_mean"])
            semantic_hash = _semantic_hash(summary)
            training._atomic_json(run_dir / "summary.json", summary)
            report["summary_sha256"] = semantic_hash
            report["summary_file"] = "summary.json"
            plots = _plots(summary, run_dir)
            if _semantic_hash(summary) != semantic_hash:
                raise ValueError("Plotting changed the diagnostic summary")
            if not isinstance(plots, list) or not plots or len(set(plots)) != len(plots):
                raise ValueError("Diagnostics must produce a unique nonempty plot list")
            for relative in plots:
                path = (run_dir / relative).resolve()
                if (Path(relative).is_absolute() or not path.is_relative_to(run_dir)
                        or not path.is_file() or path.suffix.lower() != ".png"):
                    raise ValueError("Plot output must be a file inside the diagnostic run")
            report["plots"] = plots
            report["reproducibility"] = _repeated_summary(artifact_dir, run_dir, config, semantic_hash)
            report["inputs_after"] = _input_state(config, training_config)
            report["input_immutability_verified"] = report["inputs_after"] == inputs_before
            report["fit_calls"] = guard.call_count
            if not report["input_immutability_verified"] or report["fit_calls"]:
                raise ValueError("Diagnostics changed an input or attempted model fitting")
            report["status"] = "SUCCESS"
            report["finished_at_utc"] = training._utc_now()
            report["artifact_sha256"] = {
                path.relative_to(run_dir).as_posix(): training._sha_file(path)
                for path in sorted(run_dir.rglob("*")) if path.is_file()
                and path.name not in ("report.json", "manifest.json")}
            training._atomic_json(run_dir / "report.json", report)
            training._atomic_json(run_dir / "manifest.json", {
                "run_id": run_id, "status": "SUCCESS", "summary_sha256": semantic_hash,
                "report_sha256": training._sha_file(run_dir / "report.json"),
                "artifact_sha256": report["artifact_sha256"]})
            return {**report, "run_dir": str(run_dir)}
    except BaseException as error:
        report.update({"status": "FAILED", "finished_at_utc": training._utc_now(),
                       "fit_calls": guard.call_count if guard is not None else 0,
                       "error": _error(error, config, training_config)})
        if inputs_before is not None:
            try:
                report["inputs_after"] = _input_state(config, training_config)
                report["input_immutability_verified"] = report["inputs_after"] == inputs_before
            except (OSError, ValueError, KeyError) as state_error:
                report["input_immutability_verified"] = False
                report["input_state_error"] = _error(state_error, config, training_config)
        training._atomic_json(run_dir / "report.json", report)
        if isinstance(error, (KeyboardInterrupt, SystemExit)):
            raise
        raise RuntimeError("Diagnostics failed; see " + run_id + "/report.json: "
                           + report["error"]["type"] + ": " + report["error"]["message"]) from error
