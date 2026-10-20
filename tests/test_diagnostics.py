"""Hand-calculated diagnostics, train-only thresholds and immutable inputs."""

import copy
import json
import math
import unittest
from datetime import datetime

import numpy as np

from src.vkr_model.diagnostics import analyze
from src.vkr_model.train import FEATURE_COLUMNS, METHODS


def utc_ms(value):
    return int(datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp() * 1000)


class DiagnosticsTests(unittest.TestCase):
    def setUp(self):
        self.mean = 1.0
        self.data = {"train": [], "validation": []}
        targets = {"train": [-2.0, 0.0, 2.0, 4.0], "validation": [1.0, -2.0, 4.0, -8.0]}
        dates = {"train": ["2022-01-01", "2022-07-01", "2023-01-01", "2023-12-31"],
                 "validation": ["2024-01-01", "2024-01-31", "2024-04-01", "2024-12-31"]}
        for split in self.data:
            for index, target in enumerate(targets[split]):
                row = {feature: float(index + offset) for offset, feature in enumerate(FEATURE_COLUMNS)}
                if split == "validation":
                    row.update({feature: float([-1, 2, 3, 6][index] + offset)
                                for offset, feature in enumerate(FEATURE_COLUMNS)})
                row.update({"split": split, "execution_time": utc_ms(dates[split][index] + "T00:02:00Z"), "y": target})
                row["realized_volatility_24h"] = {"train": [0, 3, 6, 9], "validation": [3, 6, 6.1, -1]}[split][index]
                self.data[split].append(row)
        self.predictions = {
            "zero": {split: [0.0] * 4 for split in self.data},
            "train_mean": {split: [self.mean] * 4 for split in self.data},
            "depth4": {"train": list(targets["train"]), "validation": [1.0, 0.0, 3.0, -4.0]},
            "depth6": {"train": [-1.0, 0.0, 1.0, 3.0], "validation": [2.0, -1.0, 2.0, -7.0]},
        }

    def analyze(self):
        return analyze(self.data, self.predictions, self.mean)

    def test_hand_calculated_metrics_and_population_moments(self):
        result = self.analyze()
        training = result["overall"]["train"]["methods"]["depth4"]
        self.assertEqual(training["mse"], 0)
        self.assertEqual(training["mae"], 0)
        self.assertEqual(training["target"]["mean"], 1)
        self.assertAlmostEqual(training["target"]["std"], math.sqrt(5))
        validation = result["overall"]["validation"]["methods"]["depth4"]
        self.assertEqual(validation["mse"], 21 / 4)
        self.assertAlmostEqual(validation["rmse"], math.sqrt(21 / 4))
        self.assertEqual(validation["mae"], 7 / 4)
        self.assertEqual(validation["residual"]["mean"], 5 / 4)
        self.assertAlmostEqual(validation["residual"]["std"], math.sqrt(59 / 16))
        self.assertEqual(validation["prediction"]["mean"], 0)
        self.assertAlmostEqual(validation["prediction"]["std"], math.sqrt(26 / 4))
        self.assertEqual(result["residual_definition"], "prediction_minus_target")

    def test_paired_excess_decomposition_and_relative_aggregate(self):
        result = self.analyze()["overall"]["validation"]["comparisons"]["depth4"]
        zero = result["zero"]
        self.assertEqual(zero["excess_mse"], -16)
        self.assertEqual(zero["excess_mae"], -2)
        self.assertAlmostEqual(zero["relative_mse_pct"], 100 * -16 / (85 / 4))
        self.assertAlmostEqual(zero["relative_mae_pct"], 100 * -2 / (15 / 4))
        self.assertEqual(zero["decomposition"], {
            "prediction_distance_mse": 6.5, "alignment_cross_term": 22.5,
            "reconstructed_excess_mse": -16, "identity_residual": 0,
        })
        self.assertEqual(result["train_mean"]["excess_mse"], -19.5)
        self.assertEqual(result["train_mean"]["decomposition"]["prediction_distance_mse"], 7.5)
        self.assertEqual(result["train_mean"]["decomposition"]["alignment_cross_term"], 27)
        self.assertEqual(result["train_mean"]["decomposition"]["identity_residual"], 0)

    def test_monthly_and_quarterly_use_utc_boundaries_and_split_fractions(self):
        self.data["validation"][0]["execution_time"] = utc_ms("2024-03-31T23:59:59.999Z")
        self.data["validation"][1]["execution_time"] = utc_ms("2024-04-01T00:00:00Z")
        self.data["validation"][2]["execution_time"] = utc_ms("2024-09-30T23:59:59.999Z")
        self.data["validation"][3]["execution_time"] = utc_ms("2024-10-01T00:00:00Z")
        result = self.analyze()["validation"]
        self.assertEqual(len(result["monthly"]), 12)
        self.assertEqual(len(result["quarterly"]), 4)
        self.assertEqual(result["monthly"]["2024-03"]["n"], 1)
        self.assertEqual(result["monthly"]["2024-04"]["n"], 1)
        self.assertEqual(result["monthly"]["2024-09"]["n"], 1)
        self.assertEqual(result["monthly"]["2024-10"]["n"], 1)
        for quarter in result["quarterly"].values():
            self.assertEqual(quarter["n"], 1)
            self.assertEqual(quarter["fraction_of_split"], .25)
        empty = result["monthly"]["2024-01"]
        self.assertEqual(empty["n"], 0)
        self.assertEqual(empty["fraction_of_split"], 0)
        for method in empty["methods"].values():
            self.assertIsNone(method["mse"])
            self.assertIsNone(method["target"]["mean"])
        self.assertIsNone(empty["comparisons"]["depth4"]["zero"]["excess_mse"])

    def test_weighted_monthly_excess_matches_overall(self):
        result = self.analyze()
        # Unequal counts (January=2, April=1, December=1) prevent an unweighted
        # average of monthly errors from accidentally passing this identity.
        self.assertEqual(result["validation"]["monthly"]["2024-01"]["fraction_of_split"], .5)
        for model, baselines in result["monthly_weighted_identity"].items():
            for baseline, identity in baselines.items():
                with self.subTest(model=model, baseline=baseline):
                    self.assertAlmostEqual(identity["mse_residual"], 0)
                    self.assertAlmostEqual(identity["mae_residual"], 0)
                    self.assertAlmostEqual(identity["weighted_excess_mse"], identity["overall_excess_mse"])
                    self.assertAlmostEqual(identity["weighted_excess_mae"], identity["overall_excess_mae"])

    def test_train_only_linear_quantiles_and_group_equality_rules(self):
        result = self.analyze()
        cuts = result["thresholds"]
        self.assertEqual(cuts["quantile_method"], "linear")
        self.assertEqual(cuts["volatility24"]["cut1"], 3)
        self.assertEqual(cuts["volatility24"]["cut2"], 6)
        self.assertAlmostEqual(cuts["absolute_target"]["cut"], 3.7)
        self.assertTrue(cuts["absolute_target"]["retrospective"])
        self.assertEqual({group: item["n"] for group, item in result["validation"]["volatility24"].items()},
                         {"low": 2, "middle": 1, "high": 1})
        self.assertEqual(result["validation"]["absolute_target"]["ordinary"]["n"], 2)
        self.assertEqual(result["validation"]["absolute_target"]["tail"]["n"], 2)
        for row in self.data["validation"]:
            row["y"] *= 1000
            row["realized_volatility_24h"] *= 1000
        self.assertEqual(self.analyze()["thresholds"], cuts)

    def test_target_equal_to_train_tail_cut_stays_ordinary(self):
        cut = self.analyze()["thresholds"]["absolute_target"]["cut"]
        self.data["validation"][0]["y"] = cut
        result = self.analyze()["validation"]["absolute_target"]
        self.assertEqual(result["ordinary"]["n"], 2)
        self.assertEqual(result["tail"]["n"], 2)

    def test_empty_groups_and_zero_denominators_return_none(self):
        self.mean = 0.0
        for split, rows in self.data.items():
            for row in rows:
                row["y"] = 0.0
                row["realized_volatility_24h"] = 0.0
            for method in METHODS:
                self.predictions[method][split] = [0.0] * len(rows)
        result = self.analyze()
        for group in ("middle", "high"):
            empty = result["validation"]["volatility24"][group]
            self.assertEqual(empty["n"], 0)
            self.assertEqual(empty["fraction_of_split"], 0)
            self.assertIsNone(empty["methods"]["zero"]["rmse"])
        self.assertEqual(result["validation"]["absolute_target"]["tail"]["n"], 0)
        zero = result["overall"]["validation"]["comparisons"]["depth4"]["zero"]
        self.assertEqual(zero["excess_mse"], 0)
        self.assertIsNone(zero["relative_mse_pct"])
        self.assertIsNone(zero["relative_mae_pct"])
        self.assertIsNone(result["validation"]["tail_concentration"]["top_1pct"]["methods"]["zero"]["sse_share"])

    def test_tail_counts_ceil_and_common_tie_breaking_mask(self):
        template = self.data["validation"][0]
        start = utc_ms("2024-05-01T00:00:00Z")
        self.data["validation"] = [dict(template, execution_time=start + (20 - i) * 60_000,
                                       y=2.0 if i % 2 == 0 else -2.0) for i in range(21)]
        for method in METHODS:
            self.predictions[method]["validation"] = [self.mean if method == "train_mean" else 0.0] * 21
        self.predictions["depth6"]["validation"] = [row["y"] for row in self.data["validation"]]
        tails = self.analyze()["validation"]["tail_concentration"]
        self.assertEqual(tails["top_1pct"]["n"], 1)
        self.assertEqual(tails["top_5pct"]["n"], 2)
        self.assertEqual(tails["top_1pct"]["execution_times"], [start])
        self.assertEqual(tails["top_5pct"]["execution_times"], [start, start + 60_000])
        self.assertEqual(tails["top_5pct"]["fraction_of_split"], 2 / 21)
        zero = tails["top_5pct"]["methods"]["zero"]
        self.assertEqual(zero["selected_sse"], 8)
        self.assertEqual(zero["total_sse"], 84)
        self.assertEqual(zero["sse_share"], 2 / 21)
        self.assertEqual(tails["top_5pct"]["methods"]["depth6"]["selected_sse"], 0)
        self.assertIsNone(tails["top_5pct"]["methods"]["depth6"]["sse_share"])

    def test_drift_linear_quantiles_outside_range_and_zero_iqr(self):
        result = self.analyze()["drift"]
        self.assertEqual(set(result), set(FEATURE_COLUMNS))
        feature = result["last_return"]
        self.assertEqual(feature["train"]["min"], 0)
        self.assertEqual(feature["train"]["max"], 3)
        self.assertEqual(feature["train"]["median"], 1.5)
        self.assertEqual(feature["train"]["iqr"], 1.5)
        self.assertAlmostEqual(feature["train"]["q01"], .03)
        self.assertAlmostEqual(feature["train"]["q99"], 2.97)
        self.assertEqual(feature["validation_outside_train_range_fraction"], .5)
        self.assertEqual(feature["median_shift_over_train_iqr"], 2 / 3)
        for row in self.data["train"]:
            row["last_return"] = 1
        constant = self.analyze()["drift"]["last_return"]
        self.assertEqual(constant["train"]["iqr"], 0)
        self.assertIsNone(constant["median_shift_over_train_iqr"])

    def test_quantile_distributions_are_explicit_101_point_linear_grid(self):
        result = self.analyze()["distributions"]
        for split in self.data:
            self.assertEqual(result[split]["probabilities"], [index / 100 for index in range(101)])
            self.assertEqual(len(result[split]["target_quantiles"]), 101)
            np.testing.assert_array_equal(result[split]["target_quantiles"], np.quantile(
                [row["y"] for row in self.data[split]], [index / 100 for index in range(101)], method="linear"))
            for method in METHODS:
                self.assertEqual(len(result[split]["prediction_quantiles"][method]), 101)
        self.assertEqual(result["train"]["prediction_quantiles"]["train_mean"], [self.mean] * 101)

    def test_json_native_output_and_no_input_mutation(self):
        before_data, before_predictions = copy.deepcopy(self.data), copy.deepcopy(self.predictions)
        result = self.analyze()
        encoded = json.dumps(result, allow_nan=False)
        self.assertEqual(json.loads(encoded), result)
        self.assertEqual(self.data, before_data)
        self.assertEqual(self.predictions, before_predictions)

    def test_nonfinite_inputs_are_rejected(self):
        for field in ("y", "last_return"):
            for value in (float("nan"), float("inf"), -float("inf")):
                data = copy.deepcopy(self.data)
                data["validation"][0][field] = value
                with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                    analyze(data, self.predictions, self.mean)
        for value in (float("nan"), float("inf")):
            predictions = copy.deepcopy(self.predictions)
            predictions["depth4"]["validation"][0] = value
            with self.assertRaises(ValueError):
                analyze(self.data, predictions, self.mean)
            with self.assertRaises(ValueError):
                analyze(self.data, self.predictions, value)

    def test_final_data_and_prediction_splits_are_rejected(self):
        data = dict(self.data, final=self.data["validation"])
        with self.assertRaisesRegex(ValueError, "final"):
            analyze(data, self.predictions, self.mean)
        self.data["validation"][0]["execution_time"] = utc_ms("2025-01-01T00:00:00Z")
        with self.assertRaisesRegex(ValueError, "final"):
            self.analyze()
        self.setUp()
        self.predictions["depth4"]["final"] = [1]
        with self.assertRaisesRegex(ValueError, "final"):
            self.analyze()

    def test_missing_methods_rows_features_and_non_numeric_shapes_are_rejected(self):
        for mutate in (
            lambda data, predictions: predictions.pop("depth6"),
            lambda data, predictions: predictions["depth4"]["validation"].pop(),
            lambda data, predictions: data["train"].clear(),
            lambda data, predictions: data["train"][0].pop("range_720h"),
            lambda data, predictions: data["train"][0].update(y="1"),
            lambda data, predictions: data["train"][0].update(last_return=True),
            lambda data, predictions: data["validation"][0].update(split="final"),
            lambda data, predictions: data["validation"][0].update(execution_time=True),
            lambda data, predictions: predictions["depth4"].update(validation=[[1]] * 4),
            lambda data, predictions: data["validation"][0].update(execution_time=data["validation"][1]["execution_time"]),
        ):
            data, predictions = copy.deepcopy(self.data), copy.deepcopy(self.predictions)
            mutate(data, predictions)
            with self.subTest(mutate=mutate), self.assertRaises(ValueError):
                analyze(data, predictions, self.mean)

    def test_baseline_must_be_training_mean_and_constant_on_both_splits(self):
        with self.assertRaisesRegex(ValueError, "training targets"):
            analyze(self.data, self.predictions, -1.25)  # validation-only target mean
        for method, split in (("zero", "train"), ("zero", "validation"),
                              ("train_mean", "train"), ("train_mean", "validation")):
            predictions = copy.deepcopy(self.predictions)
            predictions[method][split][0] += .01
            with self.subTest(method=method, split=split), self.assertRaisesRegex(ValueError, "baseline"):
                analyze(self.data, predictions, self.mean)


if __name__ == "__main__":
    unittest.main()
