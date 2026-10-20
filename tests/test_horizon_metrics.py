"""Paired horizon errors, UTC quarters and training-only per-horizon baselines."""

import copy
import json
import math
import unittest
from datetime import datetime

from src.vkr_model.horizon_metrics import HORIZONS, HOUR, METHODS, summarize_horizons


def utc_ms(value):
    return int(datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp() * 1000)


class HorizonMetricsTests(unittest.TestCase):
    def setUp(self):
        targets = {"h4": {"inner_train": [-2., 0., 2., 4.], "assessment": [1., -2., 4., -8.]},
                   "h24": {"inner_train": [0., 2., 4., 6.], "assessment": [2., -4., 8., -16.]}}
        self.data = {"inner_train": [], "assessment": []}
        for role, year in (("inner_train", 2022), ("assessment", 2023)):
            for index, date in enumerate(("01-01", "01-31", "04-01", "10-01")):
                time = utc_ms("{}-{}T00:02:00Z".format(year, date))
                self.data[role].append({"execution_time": time, "targets": {
                    horizon: {"y": targets[horizon][role][index], "label_end": time + hours * HOUR}
                    for horizon, hours in HORIZONS.items()}, "ignored_provenance": "not a model input"})
        self.predictions = {}
        for horizon in HORIZONS:
            mean = math.fsum(targets[horizon]["inner_train"]) / 4
            self.predictions[horizon] = {
                "zero": {role: [0.] * 4 for role in self.data},
                "train_mean": {role: [mean] * 4 for role in self.data},
                "catboost": {"inner_train": list(targets[horizon]["inner_train"]),
                             "assessment": [1., 0., 3., -4.] if horizon == "h4" else [2., 0., 6., -8.]},
            }

    def summary(self):
        return summarize_horizons(self.data, self.predictions)

    def test_hand_calculated_mse_mae_rmse_population_moments(self):
        result = self.summary()["horizons"]["h4"]["roles"]
        train = result["inner_train"]["overall"]["methods"]["catboost"]
        self.assertEqual(train["mse"], 0)
        self.assertEqual(train["target"]["mean"], 1)
        self.assertAlmostEqual(train["target"]["std"], math.sqrt(5))
        assessment = result["assessment"]["overall"]["methods"]["catboost"]
        self.assertEqual(assessment["mse"], 21 / 4)
        self.assertAlmostEqual(assessment["rmse"], math.sqrt(21 / 4))
        self.assertEqual(assessment["mae"], 7 / 4)
        self.assertEqual(assessment["residual"]["mean"], 5 / 4)
        self.assertAlmostEqual(assessment["residual"]["std"], math.sqrt(59 / 16))
        self.assertAlmostEqual(assessment["normalized_rmse"], math.sqrt((21 / 4) / (315 / 16)))

    def test_paired_relative_errors_are_divided_after_aggregation(self):
        comparison = self.summary()["horizons"]["h4"]["roles"]["assessment"]["overall"]["comparisons"]["catboost"]
        self.assertEqual(comparison["zero"]["excess_mse"], -16)
        self.assertEqual(comparison["zero"]["excess_mae"], -2)
        self.assertAlmostEqual(comparison["zero"]["relative_mse_pct"], 100 * -16 / (85 / 4))
        self.assertAlmostEqual(comparison["zero"]["relative_mae_pct"], 100 * -2 / (15 / 4))
        self.assertEqual(comparison["train_mean"]["excess_mse"], -19.5)
        self.assertNotEqual(comparison["zero"]["relative_mse_pct"], sum([0, -100, -93.75, -75]) / 4)

    def test_each_horizon_uses_its_own_inner_train_mean_on_both_roles(self):
        result = self.summary()
        self.assertEqual(result["horizons"]["h4"]["baseline_train_mean"], 1)
        self.assertEqual(result["horizons"]["h24"]["baseline_train_mean"], 3)
        for horizon, mean in (("h4", 1), ("h24", 3)):
            for role in self.data:
                self.assertEqual(result["horizons"][horizon]["roles"][role]["overall"]["methods"]["train_mean"]["prediction"],
                                 {"mean": mean, "std": 0})
        self.predictions["h24"]["train_mean"]["assessment"] = [1.] * 4
        with self.assertRaisesRegex(ValueError, "own inner training"):
            self.summary()

    def test_assessment_changes_do_not_change_training_baselines(self):
        before = self.summary()
        for row in self.data["assessment"]:
            for horizon in HORIZONS:
                row["targets"][horizon]["y"] *= 100
        after = self.summary()
        for horizon in HORIZONS:
            self.assertEqual(after["horizons"][horizon]["baseline_train_mean"], before["horizons"][horizon]["baseline_train_mean"])
            self.assertEqual(after["horizons"][horizon]["roles"]["inner_train"], before["horizons"][horizon]["roles"]["inner_train"])

    def test_unequal_quarter_sizes_reconcile_weighted_mse_mae_and_excess(self):
        result = self.summary()
        for horizon in HORIZONS:
            for role in self.data:
                summary = result["horizons"][horizon]["roles"][role]
                year = 2022 if role == "inner_train" else 2023
                self.assertEqual(summary["quarterly"][str(year) + "-Q1"]["n"], 2)
                self.assertEqual(summary["quarterly"][str(year) + "-Q1"]["share_of_role"], .5)
                for category in summary["quarterly_weighted_identity"].values():
                    for identity in category.values():
                        self.assertAlmostEqual(identity["mse_residual"], 0)
                        self.assertAlmostEqual(identity["mae_residual"], 0)

    def test_quarter_boundaries_are_utc_and_unsorted_input_keeps_alignment(self):
        dates = ["2023-07-01T00:00:00Z", "2023-03-31T23:59:59.999Z",
                 "2023-04-01T00:00:00Z", "2023-09-30T23:59:59.999Z"]
        for row, date in zip(self.data["assessment"], dates):
            time = row["execution_time"] = utc_ms(date)
            for horizon, hours in HORIZONS.items():
                row["targets"][horizon]["label_end"] = time + hours * HOUR
        result = self.summary()["horizons"]["h4"]["roles"]["assessment"]["quarterly"]
        self.assertEqual(result["2023-Q1"]["n"], 1)
        self.assertEqual(result["2023-Q2"]["n"], 1)
        self.assertEqual(result["2023-Q3"]["n"], 2)
        self.assertEqual(result["2023-Q4"]["n"], 0)
        self.assertEqual(result["2023-Q1"]["methods"]["catboost"]["mse"], 4)

    def test_empty_quarter_metrics_remain_none_and_never_fake_zero(self):
        summary = self.summary()["horizons"]["h4"]["roles"]["assessment"]["quarterly"]["2023-Q3"]
        self.assertEqual(summary["n"], 0)
        self.assertEqual(summary["share_of_role"], 0)
        for method in METHODS:
            self.assertEqual(summary["methods"][method], {
                "mse": None, "rmse": None, "mae": None, "normalized_rmse": None,
                "target": {"mean": None, "std": None}, "prediction": {"mean": None, "std": None},
                "residual": {"mean": None, "std": None}})
        self.assertIsNone(summary["comparisons"]["catboost"]["zero"]["relative_mse_pct"])

    def test_zero_targets_make_undefined_ratios_and_normalization_none(self):
        for rows in self.data.values():
            for row in rows:
                for target in row["targets"].values():
                    target["y"] = 0.
        for horizon in HORIZONS:
            for method in METHODS:
                for role in self.data:
                    self.predictions[horizon][method][role] = [0.] * 4
        result = self.summary()
        for horizon in HORIZONS:
            overall = result["horizons"][horizon]["roles"]["assessment"]["overall"]
            self.assertEqual(overall["methods"]["catboost"]["mse"], 0)
            self.assertIsNone(overall["methods"]["catboost"]["normalized_rmse"])
            self.assertIsNone(overall["comparisons"]["catboost"]["zero"]["relative_mse_pct"])
            self.assertIsNone(overall["comparisons"]["catboost"]["train_mean"]["relative_mae_pct"])

    def test_shared_keysets_lengths_and_horizon_label_alignment_are_enforced(self):
        mutations = [
            lambda data, predictions: data["assessment"][0]["targets"].pop("h24"),
            lambda data, predictions: predictions.pop("h24"),
            lambda data, predictions: predictions["h4"].pop("catboost"),
            lambda data, predictions: predictions["h4"]["catboost"]["assessment"].pop(),
            lambda data, predictions: data["assessment"][0]["targets"]["h24"].update(label_end=data["assessment"][0]["execution_time"] + 4 * HOUR),
            lambda data, predictions: data["assessment"][0].update(execution_time=data["assessment"][1]["execution_time"]),
        ]
        for mutation in mutations:
            data, predictions = copy.deepcopy(self.data), copy.deepcopy(self.predictions)
            mutation(data, predictions)
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                summarize_horizons(data, predictions)

    def test_final_roles_cross_year_labels_and_nonfinite_targets_predictions_fail(self):
        data = dict(self.data, final=self.data["assessment"])
        with self.assertRaises(ValueError):
            summarize_horizons(data, self.predictions)
        for value in (float("nan"), float("inf"), -float("inf")):
            for where in ("target", "prediction"):
                data, predictions = copy.deepcopy(self.data), copy.deepcopy(self.predictions)
                if where == "target":
                    data["assessment"][0]["targets"]["h4"]["y"] = value
                else:
                    predictions["h24"]["catboost"]["assessment"][0] = value
                with self.subTest(value=value, where=where), self.assertRaises(ValueError):
                    summarize_horizons(data, predictions)
        time = self.data["assessment"][0]["execution_time"] = utc_ms("2023-12-31T12:00:00Z")
        for horizon, hours in HORIZONS.items():
            self.data["assessment"][0]["targets"][horizon]["label_end"] = time + hours * HOUR
        with self.assertRaisesRegex(ValueError, "inside the role"):
            self.summary()

    def test_zero_baseline_is_verified_and_inputs_are_immutable(self):
        before = copy.deepcopy((self.data, self.predictions))
        first = self.summary()
        self.assertEqual(first, self.summary())
        self.assertEqual((self.data, self.predictions), before)
        self.assertEqual(json.loads(json.dumps(first, allow_nan=False)), first)
        self.predictions["h4"]["zero"]["assessment"][0] = 1
        with self.assertRaisesRegex(ValueError, "constant zero"):
            self.summary()

    def test_metadata_describes_dependence_without_significance_or_ranking(self):
        result = self.summary()
        self.assertEqual(result["rows"], {"inner_train": 4, "assessment": 4})
        self.assertTrue(result["metadata"]["labels_overlap"])
        self.assertFalse(result["metadata"]["significance_calculated"])
        self.assertFalse(result["metadata"]["cross_horizon_raw_rmse_ranking"])
        self.assertEqual(result["metadata"]["std_ddof"], 0)


if __name__ == "__main__":
    unittest.main()
