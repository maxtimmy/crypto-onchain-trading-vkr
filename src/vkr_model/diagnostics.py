"""Pure descriptive diagnostics of saved predictions; never fits a model.

Excess errors are paired model-minus-baseline errors, so negative values favour
the model. Label-based tails are retrospective descriptions, not trading rules.
"""

import math
from collections.abc import Mapping
from datetime import datetime, timezone
from numbers import Integral, Real

import numpy as np

from .train import FEATURE_COLUMNS, METHODS, SPLIT_BOUNDARIES


SCHEMA_VERSION = "market-diagnostics-v1"
MODELS = ("depth4", "depth6")
BASELINES = ("zero", "train_mean")


def _number(value, name):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(name + " must be a real number")
    try:
        result = float(value)
    except (OverflowError, ValueError) as error:
        raise ValueError(name + " must be finite") from error
    if not math.isfinite(result):
        raise ValueError(name + " must be finite")
    return result


def _keys(value, expected, name):
    if not isinstance(value, Mapping) or set(value) != set(expected):
        raise ValueError(name + " must contain exactly " + ", ".join(expected))


def _sequence(value, name):
    if isinstance(value, (str, bytes, Mapping)):
        raise ValueError(name + " must be a one-dimensional sequence")
    try:
        return list(value)
    except TypeError as error:
        raise ValueError(name + " must be a one-dimensional sequence") from error


def _mean(values):
    return math.fsum(values) / len(values)


def _moments(values):
    if not values:
        return {"mean": None, "std": None}
    mean = _mean(values)
    return {"mean": mean, "std": math.sqrt(_mean([(value - mean) ** 2 for value in values]))}


def _stats(values):
    q01, q25, median, q75, q99 = np.quantile(values, [.01, .25, .5, .75, .99], method="linear")
    return {"min": min(values), "max": max(values), "median": float(median),
            "iqr": float(q75 - q25), "q01": float(q01), "q99": float(q99)}


def _validate(data, predictions, baseline_train_mean):
    splits = ("train", "validation")
    _keys(data, splits, "data splits (final is forbidden)")
    _keys(predictions, METHODS, "prediction methods")
    rows, values, times = {}, {}, {}
    for split in splits:
        rows[split] = _sequence(data[split], "data[" + split + "]")
        if not rows[split]:
            raise ValueError("Each split must contain at least one row")
        values[split], times[split] = [], []
        left, right = [int(datetime.fromisoformat(item.replace("Z", "+00:00")).timestamp() * 1000)
                       for item in SPLIT_BOUNDARIES[split]]
        for row in rows[split]:
            if not isinstance(row, Mapping):
                raise ValueError("Each data row must be a mapping")
            required = ("execution_time", "y") + FEATURE_COLUMNS
            if any(key not in row for key in required):
                raise ValueError("Each row requires execution_time, y and all 18 features")
            if "split" in row and row["split"] != split:
                raise ValueError("Row split does not match its allowed part")
            time = row["execution_time"]
            if isinstance(time, bool) or not isinstance(time, Integral):
                raise ValueError("execution_time must be integer UTC Unix milliseconds")
            time = int(time)
            if not left <= time < right:
                raise ValueError("execution_time is outside the allowed split; final is forbidden")
            times[split].append(time)
            values[split].append({name: _number(row[name], name) for name in ("y",) + FEATURE_COLUMNS})
        if len(set(times[split])) != len(times[split]):
            raise ValueError("execution_time must be unique within each split")
    mean = _number(baseline_train_mean, "baseline_train_mean")
    actual_mean = _mean([row["y"] for row in values["train"]])
    if mean != actual_mean:
        raise ValueError("baseline_train_mean must be calculated from training targets only")
    clean_predictions = {}
    for method in METHODS:
        _keys(predictions[method], splits, "prediction splits (final is forbidden)")
        clean_predictions[method] = {}
        for split in splits:
            vector = _sequence(predictions[method][split], method + "/" + split)
            if len(vector) != len(rows[split]):
                raise ValueError("All prediction methods must use the same paired rows")
            vector = [_number(value, method + " prediction") for value in vector]
            if method in BASELINES:
                expected = 0.0 if method == "zero" else mean
                if any(value != expected for value in vector):
                    raise ValueError("Constant baseline predictions do not match training-only baseline")
            clean_predictions[method][split] = vector
    return values, times, clean_predictions


def _slice(indices, split, values, predictions):
    count = len(indices)
    target = [values[split][index]["y"] for index in indices]
    target_moments = _moments(target)
    result = {"n": count, "fraction_of_split": count / len(values[split]),
              "methods": {}, "comparisons": {}}
    selected, errors = {}, {}
    for method in METHODS:
        selected[method] = [predictions[method][split][index] for index in indices]
        residuals = [predicted - actual for predicted, actual in zip(selected[method], target)]
        errors[method] = residuals
        mse = _mean([error * error for error in residuals]) if count else None
        result["methods"][method] = {
            "rmse": math.sqrt(mse) if count else None,
            "mae": _mean([abs(error) for error in residuals]) if count else None,
            "mse": mse, "target": dict(target_moments),
            "prediction": _moments(selected[method]), "residual": _moments(residuals),
        }
    for model in MODELS:
        result["comparisons"][model] = {}
        for baseline in BASELINES:
            excess_mse = _mean([a * a - b * b for a, b in zip(errors[model], errors[baseline])]) if count else None
            excess_mae = _mean([abs(a) - abs(b) for a, b in zip(errors[model], errors[baseline])]) if count else None
            distance = [p - b for p, b in zip(selected[model], selected[baseline])]
            distance_mse = _mean([value * value for value in distance]) if count else None
            cross = 2 * _mean([(y - b) * delta for y, b, delta in
                               zip(target, selected[baseline], distance)]) if count else None
            reconstructed = distance_mse - cross if count else None
            baseline_metrics = result["methods"][baseline]
            result["comparisons"][model][baseline] = {
                "excess_mse": excess_mse, "excess_mae": excess_mae,
                "relative_mse_pct": 100 * excess_mse / baseline_metrics["mse"]
                if count and baseline_metrics["mse"] != 0 else None,
                "relative_mae_pct": 100 * excess_mae / baseline_metrics["mae"]
                if count and baseline_metrics["mae"] != 0 else None,
                "decomposition": {"prediction_distance_mse": distance_mse,
                                  "alignment_cross_term": cross,
                                  "reconstructed_excess_mse": reconstructed,
                                  "identity_residual": excess_mse - reconstructed if count else None},
            }
    return result


def analyze(data, predictions, baseline_train_mean):
    """Describe fixed train/2024 predictions with no fitting or state changes.

    Input follows ``train._load_data`` and its four saved prediction methods.
    All numeric outputs are JSON-native; undefined ratios and empty-slice
    metrics are ``None``. Quantiles use NumPy's explicit linear convention.
    """
    values, times, predictions = _validate(data, predictions, baseline_train_mean)
    train = values["train"]
    validation = values["validation"]
    count = len(validation)
    all_indices = {split: list(range(len(rows))) for split, rows in values.items()}
    summarize = lambda indices, split="validation": _slice(indices, split, values, predictions)
    cut1, cut2 = [float(value) for value in np.quantile(
        [row["realized_volatility_24h"] for row in train], [1 / 3, 2 / 3], method="linear")]
    absolute_cut = float(np.quantile([abs(row["y"]) for row in train], .95, method="linear"))
    result = {
        "schema_version": SCHEMA_VERSION, "residual_definition": "prediction_minus_target",
        "rows": {split: len(rows) for split, rows in values.items()},
        "thresholds": {"quantile_method": "linear",
                       "volatility24": {"feature": "realized_volatility_24h", "fit_split": "train",
                                        "quantiles": [1 / 3, 2 / 3], "cut1": cut1, "cut2": cut2},
                       "absolute_target": {"fit_split": "train", "quantile": .95,
                                           "cut": absolute_cut, "retrospective": True}},
        "overall": {split: summarize(indices, split) for split, indices in all_indices.items()},
        "validation": {}, "drift": {}, "monthly_weighted_identity": {}, "distributions": {},
    }
    monthly = {"2024-{:02d}".format(month): [] for month in range(1, 13)}
    quarterly = {"2024-Q{}".format(quarter): [] for quarter in range(1, 5)}
    volatility = {"low": [], "middle": [], "high": []}
    absolute_target = {"ordinary": [], "tail": []}
    for index, (row, time) in enumerate(zip(validation, times["validation"])):
        stamp = datetime.fromtimestamp(time / 1000, timezone.utc)
        monthly[stamp.strftime("%Y-%m")].append(index)
        quarterly["{}-Q{}".format(stamp.year, (stamp.month - 1) // 3 + 1)].append(index)
        value = row["realized_volatility_24h"]
        volatility["low" if value <= cut1 else "middle" if value <= cut2 else "high"].append(index)
        absolute_target["ordinary" if abs(row["y"]) <= absolute_cut else "tail"].append(index)
    for name, groups in (("monthly", monthly), ("quarterly", quarterly),
                         ("volatility24", volatility), ("absolute_target", absolute_target)):
        result["validation"][name] = {name: summarize(indices) for name, indices in groups.items()}

    ranking = sorted(range(count), key=lambda index: (-abs(validation[index]["y"]), times["validation"][index]))
    tails = {}
    for name, fraction in (("top_1pct", .01), ("top_5pct", .05)):
        selected = ranking[:math.ceil(count * fraction)]
        entry = {"n": len(selected), "fraction_of_split": len(selected) / count,
                 "execution_times": [times["validation"][index] for index in selected], "methods": {}}
        for method in METHODS:
            squared = [(p - row["y"]) ** 2 for p, row in zip(predictions[method]["validation"], validation)]
            total = math.fsum(squared)
            selected_sse = math.fsum(squared[index] for index in selected)
            entry["methods"][method] = {"selected_sse": selected_sse, "total_sse": total,
                                        "sse_share": selected_sse / total if total else None}
        tails[name] = entry
    result["validation"]["tail_concentration"] = tails

    for feature in FEATURE_COLUMNS:
        a = _stats([row[feature] for row in train])
        b = _stats([row[feature] for row in validation])
        outside = sum(row[feature] < a["min"] or row[feature] > a["max"] for row in validation)
        result["drift"][feature] = {
            "train": a, "validation": b, "validation_outside_train_range_fraction": outside / count,
            "median_shift_over_train_iqr": (b["median"] - a["median"]) / a["iqr"] if a["iqr"] else None,
        }
    for model in MODELS:
        result["monthly_weighted_identity"][model] = {}
        for baseline in BASELINES:
            identity = {}
            comparison = result["overall"]["validation"]["comparisons"][model][baseline]
            for metric in ("mse", "mae"):
                weighted = math.fsum(month["fraction_of_split"] * month["comparisons"][model][baseline]["excess_" + metric]
                                     for month in result["validation"]["monthly"].values() if month["n"])
                overall = comparison["excess_" + metric]
                identity.update({"weighted_excess_" + metric: weighted, "overall_excess_" + metric: overall,
                                 metric + "_residual": weighted - overall})
            result["monthly_weighted_identity"][model][baseline] = identity

    probabilities = [index / 100 for index in range(101)]
    for split, rows in values.items():
        result["distributions"][split] = {
            "probabilities": list(probabilities),
            "target_quantiles": [float(value) for value in np.quantile(
                [row["y"] for row in rows], probabilities, method="linear")],
            "prediction_quantiles": {method: [float(value) for value in np.quantile(
                predictions[method][split], probabilities, method="linear")] for method in METHODS},
        }
    _check_finite_output(result)
    return result


def _check_finite_output(value):
    if isinstance(value, dict):
        for item in value.values():
            _check_finite_output(item)
    elif isinstance(value, list):
        for item in value:
            _check_finite_output(item)
    elif isinstance(value, float) and not math.isfinite(value):
        raise ValueError("Diagnostics cannot represent a nonfinite computed result")
