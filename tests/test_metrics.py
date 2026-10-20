"""Offline, hand-calculated checks for descriptive prediction errors."""

import math
import unittest
from datetime import datetime, timezone

from src.vkr_model.metrics import evaluate

try:
    import numpy as np
except ImportError:
    np = None


def utc_ms(value):
    return int(datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp() * 1000)


class MetricsTests(unittest.TestCase):
    def setUp(self):
        self.t = int(datetime(2024, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)

    def test_hand_calculated_overall_errors(self):
        result = evaluate([3, 5, 7, 9], [1, 2, 7, 10], [self.t] * 4)
        self.assertEqual(result["overall"]["n"], 4)
        self.assertAlmostEqual(result["overall"]["rmse"], math.sqrt(14 / 4))
        self.assertEqual(result["overall"]["mae"], 6 / 4)
        self.assertEqual(result["quarterly"], {"2024-Q1": result["overall"]})

    def test_irregular_unsorted_rows_and_exact_utc_quarter_boundaries(self):
        times = [
            "2025-01-01T00:00:00Z", "2024-03-31T23:59:59.999Z",
            "2024-04-01T00:00:00Z", "2024-12-31T23:59:59.999Z",
            "2023-12-31T23:59:59.999Z", "2024-07-01T00:00:00Z",
            "2024-06-30T23:59:59.999Z", "2024-09-30T23:59:59.999Z",
            "2024-10-01T00:00:00Z",
        ]
        result = evaluate([9, 1, 2, 8, 0, 4, 3, 6, 7], [0] * len(times),
                          [utc_ms(value) for value in times])
        expected = {
            "2023-Q4": (1, 0.0, 0.0), "2024-Q1": (1, 1.0, 1.0),
            "2024-Q2": (2, math.sqrt(13 / 2), 2.5),
            "2024-Q3": (2, math.sqrt(52 / 2), 5.0),
            "2024-Q4": (2, math.sqrt(113 / 2), 7.5),
            "2025-Q1": (1, 9.0, 9.0),
        }
        self.assertEqual(list(result["quarterly"]), list(expected))
        for quarter, (count, rmse, mae) in expected.items():
            with self.subTest(quarter=quarter):
                self.assertEqual(result["quarterly"][quarter]["n"], count)
                self.assertAlmostEqual(result["quarterly"][quarter]["rmse"], rmse)
                self.assertAlmostEqual(result["quarterly"][quarter]["mae"], mae)
        self.assertEqual(result["overall"]["n"], sum(value[0] for value in expected.values()))

    def test_constant_and_perfect_predictions(self):
        for prediction, expected in [([2, 2, 2], 0.0), ([0, 0, 0], 2.0)]:
            with self.subTest(prediction=prediction):
                result = evaluate([2, 2, 2], prediction, [self.t] * 3)
                self.assertEqual(result["overall"], {"n": 3, "rmse": expected, "mae": expected})

    def test_single_row_has_absolute_error_for_both_metrics(self):
        result = evaluate([-0.25], [0.5], [self.t])
        self.assertEqual(result["overall"], {"n": 1, "rmse": 0.75, "mae": 0.75})

    def test_large_finite_errors_do_not_overflow_when_squared(self):
        result = evaluate([1e200, -1e200], [0, 0], [self.t] * 2)
        self.assertEqual(result["overall"], {"n": 2, "rmse": 1e200, "mae": 1e200})

    def test_empty_and_mismatched_lengths_are_rejected(self):
        for truth, prediction, times in [([], [], []), ([1], [], [self.t]),
                                         ([1], [1], []), ([1, 2], [1], [self.t])]:
            with self.subTest(truth=truth, prediction=prediction, times=times):
                with self.assertRaises(ValueError):
                    evaluate(truth, prediction, times)

    def test_nonfinite_values_are_rejected_in_every_input(self):
        for value in [float("nan"), float("inf"), -float("inf")]:
            for position in range(3):
                inputs = [[1], [0], [self.t]]
                inputs[position] = [value]
                with self.subTest(value=value, position=position):
                    with self.assertRaisesRegex(ValueError, "finite"):
                        evaluate(*inputs)

    def test_wrong_shapes_and_non_numeric_values_are_rejected(self):
        invalid = [1, "1", b"1", {1: 2}, [[1]], [(1,)], ["1"], [True], [1 + 0j], [None]]
        for value in invalid:
            for position in range(3):
                inputs = [[1], [0], [self.t]]
                inputs[position] = value
                with self.subTest(value=value, position=position):
                    with self.assertRaises(ValueError):
                        evaluate(*inputs)

    def test_unrepresentable_timestamps_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "timestamp"):
            evaluate([1], [0], [1e200])

    def test_overflowed_residuals_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "errors must be finite"):
            evaluate([1e308], [-1e308], [self.t])

    @unittest.skipIf(np is None, "NumPy is available in the model training environment")
    def test_one_dimensional_numpy_arrays_are_supported(self):
        result = evaluate(np.array([1, 3], dtype=np.float32), np.array([0, 0]),
                          np.array([self.t, self.t], dtype=np.int64))
        self.assertEqual(result["overall"]["n"], 2)
        self.assertAlmostEqual(result["overall"]["rmse"], math.sqrt(5))
        self.assertEqual(result["overall"]["mae"], 2.0)

    @unittest.skipIf(np is None, "NumPy is available in the model training environment")
    def test_numpy_scalar_and_two_dimensional_arrays_are_rejected(self):
        for value in [np.array(1), np.array([[1]]), np.array([[1], [2]])]:
            with self.subTest(shape=value.shape):
                with self.assertRaises(ValueError):
                    evaluate(value, [0], [self.t])


if __name__ == "__main__":
    unittest.main()
