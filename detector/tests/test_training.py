"""CPU tests for detector split validation, threshold calibration and ablations."""

import argparse
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import joblib
import numpy as np
import pandas as pd

from detector import BASE_FEATURE_COLUMNS, FEATURE_COLUMNS, FEATURE_PROTOCOL, HEAD_MODE
from detector import train_detector as training
from detector.common import save_json, sha256_file


def make_metadata():
    return {
        "feature_protocol": FEATURE_PROTOCOL,
        "head_mode": HEAD_MODE,
        "head_seed": 12,
        "label_mode": "predicted",
        "checkpoint_sha256": "a" * 64,
        "inversion_sha256": "b" * 64,
        "origin_role": "paired_benchmark",
        "feature_columns": list(FEATURE_COLUMNS),
    }


def write_features(table, path):
    table.to_csv(path, index=False)
    metadata = make_metadata()
    metadata["feature_columns"] = training.feature_sets(table)["all"]
    metadata["features_sha256"] = sha256_file(path)
    save_json(metadata, path.with_suffix(".metadata.json"))


def make_feature_table():
    rng = np.random.default_rng(42)
    rows = []
    original_index = 0
    for split in ("train", "validation", "test"):
        for image in range(20):
            for view, label in (("clean", 0), ("poison", 1), ("random_control", -1)):
                row = {
                    "original_index": original_index,
                    "source_index": 1000 + original_index,
                    "class_id": image % 10,
                    "split": split,
                    "view": view,
                    "detector_label": label,
                    "feature_protocol": FEATURE_PROTOCOL,
                    "head_mode": HEAD_MODE,
                    "head_seed": 12,
                    "attack_seed": 0,
                    "label_mode": "predicted",
                }
                for index, column in enumerate(FEATURE_COLUMNS):
                    row[column] = float(
                        rng.uniform(0.1, 0.5) + (label == 1) * (0.1 + index * 0.04)
                    )
                rows.append(row)
            original_index += 1
    return pd.DataFrame(rows)


class FeatureValidationTests(unittest.TestCase):
    def setUp(self):
        self.features = make_feature_table()

    def test_complete_table(self):
        training.validate_feature_table(self.features)

    def test_dynamic_features_are_discovered_without_duplicate_task_summaries(self):
        table = self.features.assign(
            **{
                "grad_norm_param__layer1.conv1.weight": 0.4,
                "grad_norm_layer__layer1.conv1": 0.5,
                "grad_cosine_task_0": 0.2,
            }
        )
        groups = training.feature_sets(table)
        self.assertEqual(groups["parameters"], ["grad_norm_param__layer1.conv1.weight"])
        self.assertEqual(groups["layers"], ["grad_norm_layer__layer1.conv1"])
        self.assertEqual(len(groups["all"]), len(FEATURE_COLUMNS) + 3)
        self.assertEqual(len(groups["all"]), len(set(groups["all"])))
        training.validate_feature_table(table)

    def test_missing_metadata_never_disappears_in_groupby(self):
        for column in training.METADATA_COLUMNS:
            with self.subTest(column=column):
                table = self.features.copy()
                # Object dtype avoids pandas warnings when assigning None to integer metadata.
                table[column] = table[column].astype(object)
                table.loc[0, column] = None
                with self.assertRaisesRegex(RuntimeError, "missing"):
                    training.validate_feature_table(table)

    def test_integral_metadata_and_finite_features(self):
        for column in training.INTEGER_COLUMNS:
            with self.subTest(column=column):
                table = self.features.copy()
                table[column] = table[column].astype(float)
                table.loc[0, column] += 0.5
                with self.assertRaisesRegex(RuntimeError, "integers"):
                    training.validate_feature_table(table)
        for value in (np.nan, np.inf, -np.inf):
            with self.subTest(value=value):
                table = self.features.copy()
                table.loc[0, FEATURE_COLUMNS[0]] = value
                with self.assertRaises(RuntimeError):
                    training.validate_feature_table(table)

    def test_rejects_mixed_seeds(self):
        for column in ("head_seed", "attack_seed"):
            with self.subTest(column=column):
                table = self.features.copy()
                table.loc[0, column] += 1
                with self.assertRaisesRegex(RuntimeError, "requires one"):
                    training.validate_feature_table(table)

    def test_pair_metadata_must_agree(self):
        for column, replacement in (
            ("split", "test"),
            ("source_index", 999),
            ("class_id", 9),
        ):
            with self.subTest(column=column):
                table = self.features.copy()
                table.loc[0, column] = replacement
                with self.assertRaisesRegex(RuntimeError, "disagree"):
                    training.validate_feature_table(table)

    def test_attack_source_mapping_is_one_to_one(self):
        table = self.features.copy()
        table.loc[table["original_index"] == 1, "source_index"] = 1000
        with self.assertRaisesRegex(RuntimeError, "share"):
            training.validate_feature_table(table)

    def test_each_original_has_all_views_and_correct_labels(self):
        with self.assertRaisesRegex(RuntimeError, "exactly three"):
            training.validate_feature_table(self.features.drop(index=0))
        table = self.features.copy()
        table.loc[0, "detector_label"] = 1
        with self.assertRaisesRegex(RuntimeError, "Incorrect detector labels"):
            training.validate_feature_table(table)


class ThresholdTests(unittest.TestCase):
    def test_ties_are_conservative_under_greater_equal(self):
        scores = np.array([0.1] * 8 + [0.9] * 2)
        threshold = training.select_clean_threshold(scores, 0.1)
        self.assertGreater(threshold, 0.9)
        self.assertEqual(np.mean(scores >= threshold), 0.0)

    def test_small_sample_and_zero_target(self):
        scores = np.array([0.2, 0.5, 1.0])
        for target in (0.0, 0.05):
            threshold = training.select_clean_threshold(scores, target)
            self.assertTrue(np.isfinite(threshold))
            self.assertGreater(threshold, 1.0)
            self.assertEqual(np.sum(scores >= threshold), 0)

    def test_empirical_cap_for_varied_budgets_and_ties(self):
        rng = np.random.default_rng(23)
        for size in (1, 7, 20, 100, 500):
            scores = rng.integers(0, 11, size=size) / 10.0
            for target in (0.0, 0.001, 0.05, 0.07, 0.29, 0.5, 0.99, 1.0):
                with self.subTest(size=size, target=target):
                    threshold = training.select_clean_threshold(scores, target)
                    self.assertLessEqual(float(np.mean(scores >= threshold)), target)

    def test_invalid_calibration_inputs(self):
        for scores, target in (
            ([], 0.05),
            ([np.nan], 0.05),
            ([[0.5]], 0.05),
            ([np.inf], 0.05),
            ([-0.1], 0.05),
            ([1.1], 0.05),
            ([0.5], -0.1),
            ([0.5], 1.1),
            ([0.5], np.nan),
        ):
            with self.subTest(scores=scores, target=target):
                with self.assertRaises(ValueError):
                    training.select_clean_threshold(scores, target)


class TrainingWorkflowTests(unittest.TestCase):
    def test_fit_only_then_verified_frozen_evaluation_never_refits(self):
        table = make_feature_table()
        with tempfile.TemporaryDirectory(
            prefix=".training-test-", dir=Path(__file__).resolve().parents[1]
        ) as directory:
            root = Path(directory)
            path = root / "features.csv"
            write_features(table, path)
            args = argparse.Namespace(
                features=str(path),
                output_dir=str(root / "fitted"),
                classifier_seed=12,
                target_clean_fpr=0.05,
                feature_set="baseline",
                compare_stages=False,
                fit_only=True,
            )
            predict = training.predict_scores

            def validation_only(rows, *positional):
                self.assertEqual(set(rows["split"]), {"validation"})
                return predict(rows, *positional)

            with mock.patch.object(
                training, "predict_scores", side_effect=validation_only
            ):
                with contextlib.redirect_stdout(io.StringIO()):
                    training.main(args)
            fitted_dir = Path(args.output_dir)
            self.assertFalse((fitted_dir / "test_predictions.csv").exists())
            args.evaluate_bundle = str(fitted_dir / "detector_bundle.joblib")
            args.output_dir = str(root / "evaluation")
            args.fit_only = False

            def test_only(rows, *positional):
                self.assertEqual(set(rows["split"]), {"test"})
                return predict(rows, *positional)

            with mock.patch.object(
                training, "fit_detector", side_effect=AssertionError("refit")
            ):
                with mock.patch.object(
                    training, "predict_scores", side_effect=test_only
                ):
                    with contextlib.redirect_stdout(io.StringIO()):
                        training.main(args)
            self.assertTrue((Path(args.output_dir) / "test_predictions.csv").is_file())

    def test_stage_comparison_never_scores_test_or_selects_a_model(self):
        table = make_feature_table()
        with tempfile.TemporaryDirectory(
            prefix=".training-test-", dir=Path(__file__).resolve().parents[1]
        ) as directory:
            root = Path(directory)
            path = root / "features.csv"
            write_features(table, path)
            args = argparse.Namespace(
                features=str(path),
                output_dir=str(root / "comparison"),
                classifier_seed=12,
                target_clean_fpr=0.05,
                feature_set="all",
                compare_stages=True,
            )
            observed_splits = []
            predict = training.predict_scores

            def only_validation(rows, *positional):
                observed_splits.append(set(rows["split"]))
                self.assertEqual(set(rows["split"]), {"validation"})
                return predict(rows, *positional)

            with mock.patch.object(
                training, "predict_scores", side_effect=only_validation
            ):
                with contextlib.redirect_stdout(io.StringIO()):
                    training.main(args)
            self.assertTrue(observed_splits)
            output = Path(args.output_dir)
            comparison = pd.read_csv(output / "feature_comparison_validation.csv")
            self.assertEqual(len(comparison), 17)
            self.assertEqual(
                set(comparison["n_features"]),
                {1, 3, 4, 5, 6, 7, 8, 11, 12, 13, 15, len(FEATURE_COLUMNS)},
            )
            self.assertTrue((comparison["validation_clean_fpr"] <= 0.05).all())
            self.assertTrue(comparison["validation_roc_auc"].is_monotonic_decreasing)
            self.assertFalse((output / "test_predictions.csv").exists())
            self.assertFalse((output / "detector_bundle.joblib").exists())
            metadata = json.loads(
                (output / "feature_comparison_metadata.json").read_text()
            )
            self.assertFalse(metadata["test_evaluated"])

    def test_selected_baseline_fits_train_and_tests_at_frozen_threshold(self):
        table = make_feature_table()
        with tempfile.TemporaryDirectory(
            prefix=".training-test-", dir=Path(__file__).resolve().parents[1]
        ) as directory:
            root = Path(directory)
            path = root / "features.csv"
            write_features(table, path)
            args = argparse.Namespace(
                features=str(path),
                output_dir=str(root / "final"),
                classifier_seed=12,
                target_clean_fpr=0.05,
                feature_set="baseline",
                compare_stages=False,
            )
            fit = training.fit_detector
            scored = []
            predict = training.predict_scores

            def only_train(rows, *positional):
                self.assertEqual(set(rows["split"]), {"train"})
                self.assertEqual(set(rows["view"]), {"clean", "poison"})
                return fit(rows, *positional)

            def track_scores(rows, *positional):
                scored.append((set(rows["split"]), set(rows["view"])))
                return predict(rows, *positional)

            with mock.patch.object(
                training, "fit_detector", side_effect=only_train
            ) as fitting:
                with mock.patch.object(
                    training, "predict_scores", side_effect=track_scores
                ):
                    with contextlib.redirect_stdout(io.StringIO()):
                        training.main(args)
            self.assertEqual(fitting.call_count, 1)
            self.assertEqual(
                scored,
                [
                    ({"validation"}, {"clean", "poison"}),
                    ({"test"}, {"clean", "poison"}),
                    ({"test"}, {"random_control"}),
                ],
            )
            output = Path(args.output_dir)
            bundle = joblib.load(output / "detector_bundle.joblib")
            self.assertEqual(bundle["feature_columns"], BASE_FEATURE_COLUMNS)
            self.assertEqual(bundle["head_seed"], 12)
            self.assertEqual(bundle["attack_seed"], 0)
            self.assertEqual(len(bundle["features_sha256"]), 64)
            train = table.loc[
                (table["split"] == "train") & table["view"].isin(["clean", "poison"])
            ]
            np.testing.assert_allclose(
                bundle["scaler"].mean_, train[BASE_FEATURE_COLUMNS].mean()
            )
            metrics = json.loads((output / "metrics.json").read_text())
            self.assertLessEqual(metrics["validation"]["clean_fpr"], 0.05)
            self.assertEqual(
                metrics["validation"]["threshold"], metrics["test"]["threshold"]
            )
            predictions = pd.read_csv(output / "test_predictions.csv")
            np.testing.assert_array_equal(
                predictions["prediction"],
                (predictions["poison_probability"] >= bundle["threshold"]).astype(int),
            )


if __name__ == "__main__":
    unittest.main()
