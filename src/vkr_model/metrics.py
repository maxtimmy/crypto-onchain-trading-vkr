"""Descriptive prediction errors for the complete sample and UTC quarters.

The four-hour labels may overlap. These summaries do not treat observations as
independent and do not calculate confidence intervals or significance tests.
"""

import math
from collections.abc import Mapping
from datetime import datetime, timezone
from numbers import Real


def _finite_vector(values, name):
    """Return a one-dimensional list without silently coercing text or bools."""
    if isinstance(values, (str, bytes, Mapping)):
        raise ValueError("{} must be a one-dimensional numeric sequence".format(name))
    try:
        items = list(values)
    except TypeError as error:
        raise ValueError("{} must be a one-dimensional numeric sequence".format(name)) from error
    result = []
    for value in items:
        if isinstance(value, bool) or not isinstance(value, Real):
            raise ValueError("{} must contain only real numbers".format(name))
        try:
            number = float(value)
        except (OverflowError, ValueError) as error:
            raise ValueError("{} must contain only finite numbers".format(name)) from error
        if not math.isfinite(number):
            raise ValueError("{} must contain only finite numbers".format(name))
        result.append(number)
    return result


def _summarize(errors):
    """Scale before summation so large finite errors need not be squared."""
    count = len(errors)
    scale = max(abs(error) for error in errors)
    if scale == 0:
        return {"n": count, "rmse": 0.0, "mae": 0.0}
    normalized = [error / scale for error in errors]
    return {
        "n": count,
        "rmse": scale * math.sqrt(math.fsum(error * error for error in normalized) / count),
        "mae": scale * (math.fsum(abs(error) for error in normalized) / count),
    }


def evaluate(y_true, y_pred, execution_times):
    """Return overall and quarterly RMSE/MAE on exactly the supplied rows.

    Inputs must be nonempty, equally sized, one-dimensional numeric sequences
    (including NumPy arrays). Execution times are UTC Unix milliseconds; quarter
    boundaries are based on execution time, irrespective of the host timezone.
    Rows need not be sorted. Empty quarters are omitted. Invalid input raises
    ``ValueError`` rather than dropping observations from the comparison.
    """
    truth = _finite_vector(y_true, "y_true")
    predictions = _finite_vector(y_pred, "y_pred")
    times = _finite_vector(execution_times, "execution_times")
    if not truth:
        raise ValueError("Evaluation requires at least one row")
    if len(truth) != len(predictions) or len(truth) != len(times):
        raise ValueError("Evaluation inputs must have the same length")

    errors = []
    by_quarter = {}
    for actual, predicted, milliseconds in zip(truth, predictions, times):
        error = actual - predicted
        if not math.isfinite(error):
            raise ValueError("Prediction errors must be finite")
        try:
            timestamp = datetime.fromtimestamp(milliseconds / 1000, timezone.utc)
        except (OverflowError, OSError, ValueError) as invalid_time:
            raise ValueError("execution_times contains an unsupported Unix timestamp") from invalid_time
        quarter = "{:04d}-Q{}".format(timestamp.year, (timestamp.month - 1) // 3 + 1)
        errors.append(error)
        by_quarter.setdefault(quarter, []).append(error)

    return {
        "overall": _summarize(errors),
        "quarterly": {quarter: _summarize(by_quarter[quarter]) for quarter in sorted(by_quarter)},
    }
