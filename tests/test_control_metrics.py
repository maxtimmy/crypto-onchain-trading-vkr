"""Manual error and aggregate-weight checks for the fixed control comparison."""

import copy
import math
import unittest
from datetime import datetime

from src.vkr_model.control_metrics import METHODS, summarize_controls


def ms(value):
    return int(datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp() * 1000)


class ControlMetricsTests(unittest.TestCase):
    def setUp(self):
        self.truth = [1.0, -1.0, 2.0]
        self.predictions = {"zero": [0.0] * 3, "train_mean": [0.5] * 3,
                            "full_l2_3": [1.0, 0.0, 1.0], "full_l2_30": [0.0] * 3,
                            "no_volume_l2_3": [1.0, -1.0, 2.0]}
        self.times = [ms("2023-01-01T00:02:00Z"), ms("2023-03-31T23:59:59.999Z"),
                      ms("2023-04-01T00:00:00Z")]

    def test_hand_errors_signed_moments_and_control_comparison(self):
        r = summarize_controls(self.truth, self.predictions, self.times)
        reference = r["methods"]["full_l2_3"]["overall"]
        self.assertAlmostEqual(reference["mse"], 2 / 3)
        self.assertAlmostEqual(reference["rmse"], math.sqrt(2 / 3))
        self.assertAlmostEqual(reference["mae"], 2 / 3)
        self.assertAlmostEqual(reference["residual"]["mean"], 0)
        self.assertAlmostEqual(reference["residual"]["std"], math.sqrt(2 / 3))
        better = r["controls_vs_reference"]["no_volume_l2_3"]["overall"]
        self.assertEqual(better["n"], 3)
        self.assertAlmostEqual(better["excess_mse"], -2 / 3)
        self.assertAlmostEqual(better["relative_mse_pct"], -100)
        self.assertAlmostEqual(better["relative_mae_pct"], -100)
        self.assertNotIn("winner", r)
        self.assertEqual(r["methods"]["train_mean"]["overall"]["prediction"]["mean"], 0.5)

    def test_utc_quarters_and_count_weighted_excess(self):
        r = summarize_controls(self.truth, self.predictions, self.times)
        for model in METHODS[2:]:
            for baseline in METHODS[:2]:
                groups = r["comparisons"][model][baseline]
                quarters = groups["quarterly"]
                self.assertEqual(quarters["2023-Q1"]["n"], 2)
                self.assertEqual(quarters["2023-Q2"]["n"], 1)
                for metric in ("excess_mse", "excess_mae"):
                    weighted = sum(v["n"] * v[metric] for v in quarters.values()) / 3
                    self.assertAlmostEqual(weighted, groups["overall"][metric])

    def test_zero_baseline_error_gives_undefined_relative_percentage(self):
        p = {method: [0.0] * 3 for method in METHODS}
        r = summarize_controls([0.0] * 3, p, self.times)
        for model in METHODS[2:]:
            value = r["comparisons"][model]["zero"]["overall"]
            self.assertEqual(value["excess_mse"], 0)
            self.assertIsNone(value["relative_mse_pct"])
            self.assertIsNone(value["relative_mae_pct"])

    def test_validation_does_not_refit_the_training_constant_or_mutate_inputs(self):
        before = copy.deepcopy(self.predictions)
        r = summarize_controls([100.0, 200.0, 300.0], self.predictions, self.times)
        self.assertEqual(r["methods"]["train_mean"]["overall"]["prediction"]["mean"], 0.5)
        self.assertEqual(self.predictions, before)

    def test_nonconstant_baselines_missing_rows_methods_and_nonfinite_rejected(self):
        for mutate in (
            lambda p: p.pop("full_l2_30"),
            lambda p: p["full_l2_3"].pop(),
            lambda p: p["zero"].__setitem__(0, 0.1),
            lambda p: p["train_mean"].__setitem__(0, 0.1),
            lambda p: p["full_l2_3"].__setitem__(0, float("nan")),
        ):
            p = copy.deepcopy(self.predictions)
            mutate(p)
            with self.subTest(mutate=mutate), self.assertRaises(ValueError):
                summarize_controls(self.truth, p, self.times)


if __name__ == "__main__":
    unittest.main()
