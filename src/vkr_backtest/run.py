"""Immutable orchestration of the fixed 2024 technical backtest, without fits."""

import gzip
import hashlib
import json
import math
import shutil
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from unittest.mock import patch

from src.vkr_model import attempts
from src.vkr_model import train as training


ROOT = Path(__file__).resolve().parents[2]
RUNNER_VERSION = "technical-backtest-v1"
START = "2024-01-01T00:00:00Z"
END = "2025-01-01T00:00:00Z"
ENTRY_THRESHOLD = "0.0024028829789982020428109645953214666957415507125449113846793382700564136851394"
SCENARIOS = {
    "base": {"fee_bps": 10, "half_spread_bps": 1, "slippage_bps": 1},
    "stress": {"fee_bps": 10, "half_spread_bps": 2, "slippage_bps": 3},
}
SIGNAL_METHODS = ("zero", "train_mean", "depth4", "depth6", "sma24_168")
METHODS = SIGNAL_METHODS + ("cash", "buy_hold")
PATH_KEYS = {"dataset_config", "dataset_dir", "training_run", "artifact_dir", "budget_ledger", "config_path"}


def _path(value):
    return training._path(value).resolve()


def _portable(config):
    return {key: training._display_path(value) if key in PATH_KEYS else value
            for key, value in config.items()}


def _read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _semantic_hash(value):
    return hashlib.sha256(training._json(value).encode("utf-8")).hexdigest()


def _validate_config(config):
    required = {"experiment_id", "dataset_config", "dataset_dir", "training_run", "artifact_dir", "budget_ledger",
                "expected_report_sha256", "expected_manifest_sha256", "expected_snapshot_id",
                "expected_dataset_sha256", "start", "end", "initial_cash", "holding_minutes",
                "allocation", "qty_step", "min_notional", "entry_threshold", "scenarios"}
    if required - set(config) or set(config) - (required | {"config_path"}):
        raise ValueError("Unexpected or missing technical-backtest configuration keys")
    fixed = {"experiment_id": "0006_btcusdt_backtest", "start": START, "end": END,
             "initial_cash": "10000", "holding_minutes": 240, "allocation": "1",
             "qty_step": "0.000001", "min_notional": "10", "entry_threshold": ENTRY_THRESHOLD,
             "scenarios": SCENARIOS}
    if training._json({key: config[key] for key in fixed}) != training._json(fixed):
        raise ValueError("Use the frozen 2024 backtest settings and both fixed cost scenarios")
    for key in ("expected_report_sha256", "expected_manifest_sha256", "expected_snapshot_id", "expected_dataset_sha256"):
        value = config[key]
        if (not isinstance(value, str) or len(value) != 64
                or any(char not in "0123456789abcdef" for char in value)):
            raise ValueError("A pinned SHA-256 is required: " + key)
    output = _path(config["artifact_dir"])
    for key in ("dataset_dir", "training_run"):
        source = _path(config[key])
        if output == source or output.is_relative_to(source):
            raise ValueError("Backtest artifacts must not be written inside source inputs")


def _pin_training(config):
    directory = _path(config["training_run"])
    report_path, manifest_path = directory / "report.json", directory / "manifest.json"
    if (training._sha_file(report_path) != config["expected_report_sha256"]
            or training._sha_file(manifest_path) != config["expected_manifest_sha256"]):
        raise ValueError("Pinned 0003 report or manifest integrity mismatch")
    report, manifest = _read_json(report_path), _read_json(manifest_path)
    if (report.get("status") != "SUCCESS" or manifest.get("status") != "SUCCESS"
            or report.get("experiment_id") != "0003_btcusdt_1h_4h"
            or report.get("run_id") != directory.name or manifest.get("run_id") != directory.name
            or manifest.get("report_sha256") != config["expected_report_sha256"]
            or manifest.get("artifact_sha256") != report.get("artifact_sha256")
            or report.get("snapshot_id") != config["expected_snapshot_id"]
            or report.get("dataset_sha256") != config["expected_dataset_sha256"]
            or report.get("feature_columns") != list(training.FEATURE_COLUMNS)
            or report.get("final_split_used") is not False):
        raise ValueError("Source must be the intact pinned 0003 market-model run")
    artifacts = manifest.get("artifact_sha256", {})
    if not {"depth4.cbm", "depth6.cbm", "predictions.csv.gz"}.issubset(artifacts):
        raise ValueError("Source manifest is missing model or prediction hashes")
    for relative, expected in artifacts.items():
        path = (directory / relative).resolve()
        if not path.is_relative_to(directory) or not path.is_file() or training._sha_file(path) != expected:
            raise ValueError("Pinned source artifact integrity mismatch: " + relative)
    return report, manifest


def _tree_hashes(directory):
    return {path.relative_to(directory).as_posix(): training._sha_file(path)
            for path in sorted(directory.rglob("*")) if path.is_file()}


def _input_state(config):
    database = _path(config["dataset_dir"]) / "market.sqlite3"
    database_files = {"market.sqlite3": training._sha_file(database)}
    wal = database.with_name(database.name + "-wal")
    if wal.exists():
        database_files["market.sqlite3-wal"] = training._sha_file(wal)
    return {"training_files_sha256": _tree_hashes(_path(config["training_run"])),
            "database_files_sha256": database_files,
            "budget_ledger": attempts.state(_path(config["budget_ledger"]))}


@contextmanager
def _fit_guard():
    with patch.object(training._backend(), "fit", side_effect=RuntimeError(
            "CatBoost fit is forbidden during backtesting")) as guard:
        yield guard


def _load_inputs(config):
    from .data import load_inputs
    return load_inputs(config)


def _simulate(signals, quotes, config, costs, mode):
    from .engine import simulate
    return simulate(signals, quotes, config, costs, mode=mode)


def _plots(summary, details, output_dir):
    from .plots import plot_backtest
    return plot_backtest(summary, details, output_dir)


def _engine_config(config):
    start, end = training._ms(config["start"]), training._ms(config["end"])
    return {"start_time": start, "end_time": end, "terminal_time": end - 60_000,
            **{key: config[key] for key in ("initial_cash", "holding_minutes", "allocation", "qty_step", "min_notional")}}


def _validate_inputs(inputs, config):
    if set(inputs) != {"quotes", "signals", "metadata"}:
        raise ValueError("Backtest inputs must contain exactly quotes, signals and metadata")
    if set(inputs["signals"]) != set(SIGNAL_METHODS) or not isinstance(inputs["metadata"], dict):
        raise ValueError("Backtest inputs must contain the five fixed signal methods")
    start, end = training._ms(config["start"]), training._ms(config["end"])
    terminal = end - 60_000
    common_times = None
    for method, signals in inputs["signals"].items():
        previous = None
        method_times = []
        for signal in signals:
            decision, available, execution = (signal[key] for key in (
                "decision_time", "signal_time", "execution_time"))
            if (any(isinstance(value, bool) or not isinstance(value, int)
                    for value in (decision, available, execution))
                    or not start <= decision < end or decision % 3_600_000
                    or not start <= execution < end or available != decision + 60_000
                    or execution != decision + 120_000
                    or execution + config["holding_minutes"] * 60_000 > terminal
                    or (previous is not None and execution <= previous)
                    or not isinstance(signal["enter"], bool)):
                raise ValueError("Forbidden period or invalid chronology in " + method + " signals")
            score = signal.get("score")
            if score is not None and (isinstance(score, bool) or not isinstance(score, (int, float))
                                      or not math.isfinite(score)):
                raise ValueError("Signal scores must be finite numbers or None")
            if method != "sma24_168":
                should_enter = score is not None and Decimal(str(score)) > Decimal(config["entry_threshold"])
                if signal["enter"] != should_enter:
                    raise ValueError("Forecast entry differs from the fixed base-cost threshold")
            previous = execution
            method_times.append((decision, available, execution))
        if not method_times or (common_times is not None and method_times != common_times):
            raise ValueError("Signal methods must use identical nonempty causal time keys")
        common_times = method_times
    for time in inputs["quotes"]:
        if isinstance(time, bool) or not isinstance(time, int) or time % 60_000 or not start <= time < end:
            raise ValueError("Quotes must contain only 2024 minute-opening timestamps")
    # This serializes metadata strictly, without sending it to the simulator.
    training._json(inputs["metadata"])


def _details(config, inputs, progress=None):
    engine_config = _engine_config(config)
    details = {}
    for scenario, costs in config["scenarios"].items():
        details[scenario] = {}
        for method in METHODS:
            if method == "cash":
                signals, mode = [], "cash"
            elif method == "buy_hold":
                start = engine_config["start_time"]
                signals, mode = [{"decision_time": start, "signal_time": start + 60_000,
                                  "execution_time": start + 120_000, "enter": True, "score": None}], "buy_hold"
            else:
                signals, mode = inputs["signals"][method], "fixed_horizon"
            if progress is not None:
                progress()
            result = _simulate(signals, inputs["quotes"], engine_config, costs, mode)
            if result.get("status") not in ("COMPLETE", "INCOMPLETE"):
                raise ValueError("Unexpected engine result status")
            required = {"metrics", "trades", "equity_hourly", "equity_daily", "counters", "final_state", "accounting"}
            if not required.issubset(result) or (result["status"] == "INCOMPLETE" and result["metrics"] is not None):
                raise ValueError("Engine result is missing audit data or reports incomplete metrics")
            details[scenario][method] = result
    return details


def _summary(config, details):
    complete = all(result["status"] == "COMPLETE" for results in details.values() for result in results.values())
    terminal = _engine_config(config)["terminal_time"]
    return {"experiment_id": config["experiment_id"], "status": "COMPLETE" if complete else "INCOMPLETE",
            "comparison_complete": complete,
            "period": {"start": config["start"], "end": config["end"], "terminal_time":
                       datetime.fromtimestamp(terminal / 1000, timezone.utc).isoformat().replace("+00:00", "Z")},
            "entry_threshold": config["entry_threshold"], "scenario_costs": config["scenarios"],
            "methods": list(METHODS),
            "assumptions": {"quantity_filters": "scenario_only_not_historical_binance_filters",
                            "entry_threshold_fixed_across_scenarios": True, "no_parameter_selection": True},
            "scenarios": {scenario: {method: {key: result[key] for key in (
                "status", "metrics", "counters", "accounting", "final_state")}
                for method, result in results.items()} for scenario, results in details.items()}}


def _write_gzip_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as raw:
        with gzip.GzipFile(filename="", fileobj=raw, mode="wb", mtime=0) as stream:
            stream.write((training._json(value) + "\n").encode("utf-8"))


def _read_gzip_json(path):
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        return json.load(stream)


def _source_snapshot(run_dir, config):
    originals = list((ROOT / "src/vkr_backtest").glob("*.py"))
    originals += list((ROOT / "src/vkr_model").glob("*.py"))
    originals += [ROOT / "src/vkr_data/build.py", ROOT / "scripts/backtest.py", _path(config["dataset_config"])]
    if config.get("config_path"):
        originals.append(_path(config["config_path"]))
    originals += [ROOT / name for name in ("uv.lock", "pyproject.toml", ".python-version")]
    result = {}
    for original in sorted(set(path.resolve() for path in originals)):
        relative = training._display_path(original)
        destination = run_dir / "source" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(original, destination)
        result[relative] = training._sha_file(destination)
    training._atomic_json(run_dir / "config.json", _portable(config))
    return result


def _error(error, config):
    message = str(error)
    for path in sorted([str(ROOT)] + [str(_path(config[key])) for key in PATH_KEYS if key in config],
                       key=len, reverse=True):
        message = message.replace(path, "<local>")
    return {"type": type(error).__name__, "message": message[:2000]}


def run_backtest(config):
    """Run exactly fourteen fixed simulations and preserve incomplete results."""
    _validate_config(config)
    directory = _path(config["artifact_dir"])
    directory.mkdir(parents=True, exist_ok=True)
    run_id = "run-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + "-" + uuid.uuid4().hex[:8]
    run_dir = directory / run_id
    run_dir.mkdir(exist_ok=False)
    report = {"runner_version": RUNNER_VERSION, "experiment_id": config["experiment_id"],
              "run_id": run_id, "status": "STARTED", "started_at_utc": training._utc_now(),
              "config": _portable(config), "fit_calls": 0, "final_2025_used": False,
              "selection_performed": False, "simulation_calls": 0}
    training._atomic_json(run_dir / "report.json", report)
    before, guard = None, None
    try:
        with _fit_guard() as guard:
            source_report, source_manifest = _pin_training(config)
            before = _input_state(config)
            report.update({"inputs_before": before, "runtime": training._runtime(),
                           "implementation_sha256": _source_snapshot(run_dir, config),
                           "source_artifact_sha256": {"report.json": config["expected_report_sha256"],
                             "manifest.json": config["expected_manifest_sha256"],
                             **{key: source_manifest["artifact_sha256"][key] for key in (
                                 "depth4.cbm", "depth6.cbm", "predictions.csv.gz")}},
                           "snapshot_id": source_report["snapshot_id"],
                           "dataset_sha256": source_report["dataset_sha256"]})
            training._atomic_json(run_dir / "report.json", report)
            inputs = _load_inputs(config)
            _validate_inputs(inputs, config)
            report["loader_metadata"] = inputs["metadata"]
            training._atomic_json(run_dir / "inputs_metadata.json", inputs["metadata"])
            def progress():
                report["simulation_calls"] += 1
                training._atomic_json(run_dir / "report.json", report)
            details = _details(config, inputs, progress)
            summary = _summary(config, details)
            summary_hash = _semantic_hash(summary)
            report["summary_sha256"] = summary_hash
            report["detail_semantic_sha256"], report["detail_files"] = {}, {}
            for scenario, results in details.items():
                report["detail_semantic_sha256"][scenario], report["detail_files"][scenario] = {}, {}
                for method, result in results.items():
                    relative = "details/" + scenario + "/" + method + ".json.gz"
                    _write_gzip_json(run_dir / relative, result)
                    report["detail_semantic_sha256"][scenario][method] = _semantic_hash(result)
                    report["detail_files"][scenario][method] = relative
            training._atomic_json(run_dir / "summary.json", summary)
            report["plots"] = _plots(summary, details, run_dir)
            if _semantic_hash(summary) != summary_hash:
                raise ValueError("Plotting changed the deterministic backtest summary")
            if any(_semantic_hash(result) != report["detail_semantic_sha256"][scenario][method]
                   for scenario, results in details.items() for method, result in results.items()):
                raise ValueError("Plotting changed deterministic backtest details")
            if not isinstance(report["plots"], list) or not report["plots"] or len(set(report["plots"])) != len(report["plots"]):
                raise ValueError("Backtest must produce unique plot filenames")
            for relative in report["plots"]:
                path = (run_dir / relative).resolve()
                if (Path(relative).is_absolute() or not path.is_relative_to(run_dir)
                        or not path.is_file() or path.suffix.lower() != ".png"):
                    raise ValueError("Backtest plot must be a PNG inside the new run")
            report["inputs_after"] = _input_state(config)
            report["input_immutability_verified"] = before == report["inputs_after"]
            report["fit_calls"] = guard.call_count
            if not report["input_immutability_verified"] or guard.call_count:
                raise ValueError("Backtesting changed an input or attempted model fitting")
            report.update({"status": "SUCCESS" if summary["comparison_complete"] else "INCOMPLETE",
                           "comparison_complete": summary["comparison_complete"], "finished_at_utc": training._utc_now()})
            report["artifact_sha256"] = {
                path.relative_to(run_dir).as_posix(): training._sha_file(path)
                for path in sorted(run_dir.rglob("*")) if path.is_file()
                and path.name not in ("report.json", "manifest.json")}
            training._atomic_json(run_dir / "report.json", report)
            training._atomic_json(run_dir / "manifest.json", {
                "run_id": run_id, "status": report["status"], "summary_sha256": summary_hash,
                "report_sha256": training._sha_file(run_dir / "report.json"), "artifact_sha256": report["artifact_sha256"]})
            return {**report, "run_dir": str(run_dir)}
    except BaseException as error:
        report.update({"status": "FAILED", "finished_at_utc": training._utc_now(),
                       "fit_calls": guard.call_count if guard is not None else 0, "error": _error(error, config)})
        if before is not None:
            try:
                report["inputs_after"] = _input_state(config)
                report["input_immutability_verified"] = before == report["inputs_after"]
            except (OSError, ValueError, KeyError) as state_error:
                report["input_immutability_verified"] = False
                report["input_state_error"] = _error(state_error, config)
        training._atomic_json(run_dir / "report.json", report)
        if isinstance(error, (KeyboardInterrupt, SystemExit)):
            raise
        raise RuntimeError("Backtest failed; see " + run_id + "/report.json: "
                           + report["error"]["type"] + ": " + report["error"]["message"]) from error


def verify_backtest(config, run_dir=None):
    """Verify hashes and deterministically repeat the fourteen saved simulations."""
    _validate_config(config)
    directory = _path(config["artifact_dir"])
    if run_dir is None:
        candidates = sorted(path.parent for path in directory.glob("run-*/manifest.json"))
        if not candidates:
            raise ValueError("No saved backtest run to verify")
        run_dir = candidates[-1]
    run_dir = Path(run_dir).resolve()
    if run_dir.parent != directory:
        raise ValueError("Backtest run must belong to the configured artifact directory")
    with _fit_guard() as guard:
        _pin_training(config)
        before = _input_state(config)
        manifest = _read_json(run_dir / "manifest.json")
        if (manifest.get("status") not in ("SUCCESS", "INCOMPLETE")
                or manifest.get("run_id") != run_dir.name
                or manifest.get("report_sha256") != training._sha_file(run_dir / "report.json")):
            raise ValueError("Backtest manifest or report integrity mismatch")
        report = _read_json(run_dir / "report.json")
        if (report.get("config") != _portable(config) or report.get("status") != manifest["status"]
                or report.get("artifact_sha256") != manifest.get("artifact_sha256")
                or report.get("fit_calls") != 0 or report.get("final_2025_used") is not False
                or report.get("selection_performed") is not False or report.get("simulation_calls") != 14):
            raise ValueError("Saved backtest configuration or scope mismatch")
        for relative, expected in manifest["artifact_sha256"].items():
            path = (run_dir / relative).resolve()
            if not path.is_relative_to(run_dir) or not path.is_file() or training._sha_file(path) != expected:
                raise ValueError("Backtest artifact integrity mismatch: " + relative)
        if training._runtime() != report["runtime"]:
            raise ValueError("Verify backtest in its pinned runtime")
        if (before["training_files_sha256"] != report["inputs_before"]["training_files_sha256"]
                or before["database_files_sha256"] != report["inputs_before"]["database_files_sha256"]):
            raise ValueError("Source training run or database differs from the saved backtest inputs")
        inputs = _load_inputs(config)
        _validate_inputs(inputs, config)
        if (inputs["metadata"] != report["loader_metadata"]
                or inputs["metadata"] != _read_json(run_dir / "inputs_metadata.json")):
            raise ValueError("Repeated loader metadata differs from the saved backtest inputs")
        details = _details(config, inputs)
        summary = _summary(config, details)
        if (_semantic_hash(summary) != report["summary_sha256"]
                or report["summary_sha256"] != manifest.get("summary_sha256")
                or summary != _read_json(run_dir / "summary.json")):
            raise ValueError("Repeated backtest summary differs from the saved run")
        for scenario, results in details.items():
            for method, result in results.items():
                relative = report["detail_files"][scenario][method]
                path = (run_dir / relative).resolve()
                if not path.is_relative_to(run_dir):
                    raise ValueError("Backtest detail path escapes the run")
                expected = report["detail_semantic_sha256"][scenario][method]
                if _semantic_hash(result) != expected or _semantic_hash(_read_gzip_json(path)) != expected:
                    raise ValueError("Repeated backtest detail differs: " + scenario + "/" + method)
        after = _input_state(config)
        if before != after or guard.call_count:
            raise ValueError("Backtest verification changed an input or attempted fitting")
        return {"verified": True, "run_id": run_dir.name, "run_dir": str(run_dir),
                "comparison_complete": summary["comparison_complete"], "summary_sha256": report["summary_sha256"],
                "simulations_repeated": 14, "details_reproduced": True, "fit_calls": 0,
                "final_2025_used": False, "inputs_before": before, "inputs_after": after,
                "input_immutability_verified": True}
