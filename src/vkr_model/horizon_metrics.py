"""Paired descriptive errors of fixed 4-hour and 24-hour model experiments.

Both horizons use the exact same supplied execution rows. Each horizon has its
own training-only mean baseline. Raw error scales never select a horizon here.
"""

import math
from collections.abc import Mapping
from datetime import datetime, timezone
from numbers import Integral

from .control_metrics import _moments, _pair
from .metrics import _finite_vector


SCHEMA_VERSION = "paired-horizon-metrics-v1"
HORIZONS = {"h4": 4, "h24": 24}
METHODS = ("zero", "train_mean", "catboost")
ROLE_YEARS = {"inner_train": 2022, "assessment": 2023}
HOUR = 3_600_000


def _keys(value, expected, name):
    if not isinstance(value, Mapping) or set(value) != set(expected):
        raise ValueError(name + " must contain exactly " + ", ".join(expected))


def _timestamp(value, name):
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(name + " must be integer UTC Unix milliseconds")
    return int(value)


def _validate(data, predictions):
    _keys(data, ROLE_YEARS, "Data roles (validation/final are forbidden)")
    _keys(predictions, HORIZONS, "Prediction horizons")
    times, targets = {}, {horizon: {} for horizon in HORIZONS}
    for role, year in ROLE_YEARS.items():
        if not isinstance(data[role], (list, tuple)) or not data[role]:
            raise ValueError("Each role must contain nonempty paired rows")
        times[role] = []
        for horizon in HORIZONS:
            targets[horizon][role] = []
        left = int(datetime(year, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)
        right = int(datetime(year + 1, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)
        for row in data[role]:
            if not isinstance(row, Mapping) or "execution_time" not in row or "targets" not in row:
                raise ValueError("Each paired row requires execution_time and both horizon targets")
            time = _timestamp(row["execution_time"], "execution_time")
            if not left <= time < right:
                raise ValueError("Execution time is outside its predeclared role")
            _keys(row["targets"], HORIZONS, "Paired target horizons")
            times[role].append(time)
            for horizon, hours in HORIZONS.items():
                target = row["targets"][horizon]
                if not isinstance(target, Mapping) or "y" not in target or "label_end" not in target:
                    raise ValueError("Each horizon target requires y and label_end")
                label_end = _timestamp(target["label_end"], horizon + " label_end")
                if label_end != time + hours * HOUR or label_end >= right:
                    raise ValueError("Horizon label_end must match its horizon and stay inside the role")
                targets[horizon][role].append(target["y"])
        if len(set(times[role])) != len(times[role]):
            raise ValueError("Paired execution keys must be unique within a role")
        for horizon in HORIZONS:
            targets[horizon][role] = _finite_vector(targets[horizon][role], horizon + " targets")
    clean, means = {}, {}
    for horizon in HORIZONS:
        _keys(predictions[horizon], METHODS, horizon + " prediction methods")
        train = targets[horizon]["inner_train"]
        means[horizon] = math.fsum(train) / len(train)
        clean[horizon] = {}
        for method in METHODS:
            _keys(predictions[horizon][method], ROLE_YEARS, horizon + "/" + method + " prediction roles")
            clean[horizon][method] = {}
            for role in ROLE_YEARS:
                vector = _finite_vector(predictions[horizon][method][role], horizon + "/" + method + "/" + role)
                if len(vector) != len(times[role]):
                    raise ValueError("All horizon methods must use the same paired execution rows")
                if method == "zero" and any(value != 0 for value in vector):
                    raise ValueError("The zero baseline must be constant zero")
                if method == "train_mean" and any(value != means[horizon] for value in vector):
                    raise ValueError("Each horizon train_mean must use its own inner training targets only")
                clean[horizon][method][role] = vector
    return targets, times, clean, means


def _slice(truth, predictions, indices):
    count = len(indices)
    target = [truth[index] for index in indices]
    target_moments = _moments(target) if count else {"mean": None, "std": None}
    result = {"n": count, "share_of_role": count / len(truth), "methods": {},
              "comparisons": {"catboost": {}}}
    for method in METHODS:
        selected = [predictions[method][index] for index in indices]
        errors = [predicted - actual for predicted, actual in zip(selected, target)]
        mse = math.fsum(error * error for error in errors) / count if count else None
        rmse = math.sqrt(mse) if count else None
        result["methods"][method] = {
            "mse": mse, "rmse": rmse,
            "mae": math.fsum(abs(error) for error in errors) / count if count else None,
            "normalized_rmse": rmse / target_moments["std"] if count and target_moments["std"] else None,
            "target": dict(target_moments),
            "prediction": _moments(selected) if count else {"mean": None, "std": None},
            "residual": _moments(errors) if count else {"mean": None, "std": None},
        }
    for baseline in METHODS[:2]:
        result["comparisons"]["catboost"][baseline] = _pair(
            truth, predictions["catboost"], predictions[baseline], indices) if count else {
                "n": 0, "excess_mse": None, "relative_mse_pct": None,
                "excess_mae": None, "relative_mae_pct": None,
            }
    return result


def _reconcile(overall, quarterly):
    result = {"methods": {}, "comparisons": {}}
    for method in METHODS:
        identity = {}
        for metric in ("mse", "mae"):
            weighted = math.fsum(slice_["share_of_role"] * slice_["methods"][method][metric]
                                 for slice_ in quarterly.values() if slice_["n"])
            aggregate = overall["methods"][method][metric]
            identity.update({"weighted_" + metric: weighted, "overall_" + metric: aggregate,
                             metric + "_residual": weighted - aggregate})
        result["methods"][method] = identity
    for baseline in METHODS[:2]:
        identity = {}
        for metric in ("mse", "mae"):
            weighted = math.fsum(slice_["share_of_role"] * slice_["comparisons"]["catboost"][baseline]["excess_" + metric]
                                 for slice_ in quarterly.values() if slice_["n"])
            aggregate = overall["comparisons"]["catboost"][baseline]["excess_" + metric]
            identity.update({"weighted_excess_" + metric: weighted, "overall_excess_" + metric: aggregate,
                             metric + "_residual": weighted - aggregate})
        result["comparisons"][baseline] = identity
    return result


def summarize_horizons(data, predictions):
    """Summarize two horizons on shared 2022/2023 rows, without fitting/ranking.

    Residuals are prediction minus target, standard deviations use ddof=0, and
    relative error differences divide paired aggregate excess by baseline error.
    All four UTC quarters are retained, with None metrics for empty quarters.
    """
    targets, times, predictions, means = _validate(data, predictions)
    result = {
        "schema_version": SCHEMA_VERSION,
        "metadata": {"residual_definition": "prediction_minus_target", "std_ddof": 0,
                     "labels_overlap": True, "significance_calculated": False,
                     "cross_horizon_raw_rmse_ranking": False},
        "rows": {role: len(values) for role, values in times.items()}, "horizons": {},
    }
    for horizon, hours in HORIZONS.items():
        horizon_result = {"horizon_hours": hours, "baseline_train_mean": means[horizon], "roles": {}}
        for role, year in ROLE_YEARS.items():
            truth = targets[horizon][role]
            role_predictions = {method: predictions[horizon][method][role] for method in METHODS}
            overall = _slice(truth, role_predictions, list(range(len(truth))))
            groups = {"{}-Q{}".format(year, quarter): [] for quarter in range(1, 5)}
            for index, milliseconds in enumerate(times[role]):
                stamp = datetime.fromtimestamp(milliseconds / 1000, timezone.utc)
                groups["{}-Q{}".format(stamp.year, (stamp.month - 1) // 3 + 1)].append(index)
            quarterly = {quarter: _slice(truth, role_predictions, indices) for quarter, indices in groups.items()}
            horizon_result["roles"][role] = {
                "n": len(truth), "overall": overall, "quarterly": quarterly,
                "quarterly_weighted_identity": _reconcile(overall, quarterly),
            }
        result["horizons"][horizon] = horizon_result
    _finite_output(result)
    return result


def _finite_output(value):
    if isinstance(value, Mapping):
        for item in value.values():
            _finite_output(item)
    elif isinstance(value, float) and not math.isfinite(value):
        raise ValueError("Horizon summary contains a nonfinite computed value")
