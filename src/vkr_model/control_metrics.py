"""Paired errors of the predeclared development controls, without selection."""

import math
from datetime import datetime, timezone

from .metrics import _finite_vector, evaluate


METHODS = ("zero", "train_mean", "full_l2_3", "full_l2_30", "no_volume_l2_3")
MODELS = METHODS[2:]


def _moments(values):
    mean = math.fsum(values) / len(values)
    return {"mean": mean, "std": math.sqrt(math.fsum((value - mean) ** 2 for value in values) / len(values))}


def _pair(truth, model, baseline, indices):
    a = [model[index] - truth[index] for index in indices]
    b = [baseline[index] - truth[index] for index in indices]
    count = len(indices)
    mse = math.fsum(value * value for value in b) / count
    mae = math.fsum(abs(value) for value in b) / count
    excess_mse = math.fsum(left * left - right * right for left, right in zip(a, b)) / count
    excess_mae = math.fsum(abs(left) - abs(right) for left, right in zip(a, b)) / count
    return {"n": count, "excess_mse": excess_mse, "excess_mae": excess_mae,
            "relative_mse_pct": 100 * excess_mse / mse if mse else None,
            "relative_mae_pct": 100 * excess_mae / mae if mae else None}


def summarize_controls(y_true, predictions, execution_times):
    """Summarize all five methods on the same supplied rows; never fit or rank.

    The caller supplies the training-only constant in ``train_mean`` for both
    parts. This function checks it is constant, but never recomputes it from an
    assessment target. Residuals are prediction minus target, std uses ddof=0.
    """
    truth = _finite_vector(y_true, "y_true")
    times = _finite_vector(execution_times, "execution_times")
    if not isinstance(predictions, dict) or set(predictions) != set(METHODS):
        raise ValueError("Controls require exactly the five predeclared methods")
    clean = {method: _finite_vector(predictions[method], method) for method in METHODS}
    # evaluate validates lengths, emptiness, finite residuals and UTC times.
    methods = {method: evaluate(truth, values, times) for method, values in clean.items()}
    if any(value != 0 for value in clean["zero"]):
        raise ValueError("The zero baseline must be constant zero")
    if any(value != clean["train_mean"][0] for value in clean["train_mean"]):
        raise ValueError("The training-mean baseline must stay constant")
    for method, values in clean.items():
        result = methods[method]["overall"]
        result["mse"] = math.fsum((prediction - target) ** 2
                                  for prediction, target in zip(values, truth)) / len(truth)
        result["target"] = _moments(truth)
        result["prediction"] = _moments(values)
        result["residual"] = _moments([prediction - target for prediction, target in zip(values, truth)])
    quarters = {}
    for index, milliseconds in enumerate(times):
        stamp = datetime.fromtimestamp(milliseconds / 1000, timezone.utc)
        quarter = "{:04d}-Q{}".format(stamp.year, (stamp.month - 1) // 3 + 1)
        quarters.setdefault(quarter, []).append(index)
    all_indices = list(range(len(truth)))

    def compare(model, baseline):
        return {"overall": _pair(truth, clean[model], clean[baseline], all_indices),
                "quarterly": {quarter: _pair(truth, clean[model], clean[baseline], indices)
                              for quarter, indices in sorted(quarters.items())}}

    result = {"methods": methods,
              "comparisons": {model: {baseline: compare(model, baseline) for baseline in METHODS[:2]}
                              for model in MODELS},
              "controls_vs_reference": {model: compare(model, "full_l2_3")
                                        for model in ("full_l2_30", "no_volume_l2_3")}}
    # JSON reports must fail rather than silently serializing an overflow.
    def finite(value):
        if isinstance(value, dict):
            for item in value.values():
                finite(item)
        elif isinstance(value, float) and not math.isfinite(value):
            raise ValueError("Control summary contains a nonfinite computed value")
    finite(result)
    return result
