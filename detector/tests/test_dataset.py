"""Tests of disjoint dataset pools, frozen inference, and bag-level accounting."""

import unittest
from unittest import mock

import numpy as np
import pandas as pd

from detector import BASE_FEATURE_COLUMNS
from detector import dataset_detector as dataset
from detector import train_detector as training
from detector.io_utils import feature_provenance
from detector.tests.test_training import make_feature_table, make_metadata


def dataset_fixture():
    base = make_feature_table()
    reserve = base.loc[base["original_index"] < 40].copy()
    reserve["original_index"] += 100
    reserve["source_index"] += 100
    reserve["split"] = "reserve"
    table = pd.concat([base, reserve], ignore_index=True)
    metadata = make_metadata()
    train = table.loc[
        (table["split"] == "train") & table["view"].isin(["clean", "poison"])
    ]
    validation = table.loc[
        (table["split"] == "validation") & table["view"].isin(["clean", "poison"])
    ]
    scaler, classifier = training.fit_detector(train, BASE_FEATURE_COLUMNS, 17)
    scores = training.predict_scores(
        validation, BASE_FEATURE_COLUMNS, scaler, classifier
    )
    threshold, _ = training.calibrate_validation(validation, scores, 0.05)
    sample_bundle = {
        "scaler": scaler,
        "classifier": classifier,
        "threshold": threshold,
        "feature_columns": list(BASE_FEATURE_COLUMNS),
        "provenance": feature_provenance(metadata),
        "fit_original_indices": sorted(train["original_index"].unique().tolist()),
        "calibration_original_indices": sorted(
            validation["original_index"].unique().tolist()
        ),
    }
    return table, metadata, sample_bundle


class AggregateTests(unittest.TestCase):
    def test_top_tail_ceil_and_suspicious_fraction(self):
        values = np.array([0.1, 0.2, 0.3, 0.8, 0.9])
        aggregates = dataset.aggregate_scores(values, 0.8, top_fraction=0.25)
        self.assertAlmostEqual(aggregates["top_tail_mean"], 0.85)
        self.assertAlmostEqual(aggregates["suspicious_fraction"], 0.4)
        self.assertAlmostEqual(aggregates["score_std"], np.std(values))

    def test_invalid_scores_and_fraction(self):
        for values, fraction in (
            ([], 0.1),
            ([np.nan], 0.1),
            ([1.1], 0.1),
            ([0.5], 0),
            ([0.5], 1.5),
        ):
            with self.subTest(values=values, fraction=fraction):
                with self.assertRaises(ValueError):
                    dataset.aggregate_scores(values, 0.5, fraction)

    def test_unique_originals_realized_count_and_reproducibility(self):
        pool = pd.DataFrame(
            [
                {"original_index": original, "view": view, "sample_poison_score": score}
                for original in range(20)
                for view, score in (
                    ("clean", 0.1),
                    ("poison", 0.9),
                    ("random_control", 0.4),
                )
            ]
        )
        first = dataset.simulate_bags(pool, [0, 0.25, 1], 10, 8, 17, 0.5)
        second = dataset.simulate_bags(pool, [0, 0.25, 1], 10, 8, 17, 0.5)
        pd.testing.assert_frame_equal(first, second)
        for ids in first["original_indices"]:
            self.assertEqual(len(ids.split("|")), len(set(ids.split("|"))))
        quarter = first.loc[first["requested_rate"] == 0.25]
        self.assertTrue((quarter["contamination_count"] == 3).all())
        self.assertTrue((quarter["realized_rate"] == 0.3).all())
        self.assertTrue((quarter["suspicious_fraction"] == 0.3).all())
        with self.assertRaisesRegex(ValueError, "distinct"):
            dataset.simulate_bags(pool, [0.1], 21, 1, 17, 0.5)


class DatasetWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.features, self.metadata, self.sample = dataset_fixture()

    def fit_bundle(self):
        return dataset.fit_dataset_detector(
            self.features,
            self.sample,
            self.metadata,
            task_size=10,
            repeats=20,
            top_fraction=0.1,
            target_clean_frr=0.05,
            rates=(0, 0.1, 0.25, 1),
        )

    def test_dataset_fit_scores_only_reserve_and_has_disjoint_calibration(self):
        observed = []
        score = dataset.score_feature_pool

        def reserve_only(table, *args):
            self.assertEqual(set(table["split"]), {"reserve"})
            observed.append(set(table["original_index"]))
            return score(table, *args)

        with mock.patch.object(dataset, "score_feature_pool", side_effect=reserve_only):
            bundle, report = self.fit_bundle()
        self.assertEqual(len(observed), 2)
        self.assertFalse(observed[0] & observed[1])
        self.assertFalse(
            set(bundle["fit_original_indices"])
            & dataset.sample_seen_indices(self.sample)
        )
        self.assertFalse(
            set(bundle["calibration_original_indices"])
            & dataset.sample_seen_indices(self.sample)
        )
        self.assertLessEqual(report["calibration_clean_frr"], 0.05)
        self.assertLessEqual(report["calibration_top_tail_clean_frr"], 0.05)
        self.assertFalse(report["test_evaluated"])

    def test_evaluation_uses_frozen_models_and_reports_random_specificity(self):
        bundle, _ = self.fit_bundle()
        old_threshold = bundle["threshold"]
        with mock.patch.object(
            bundle["classifier"], "fit", side_effect=AssertionError("refit")
        ):
            with mock.patch.object(
                self.sample["classifier"], "fit", side_effect=AssertionError("refit")
            ):
                report, predictions = dataset.evaluate_dataset_detector(
                    self.features, bundle, self.metadata
                )
        self.assertEqual(bundle["threshold"], old_threshold)
        self.assertEqual(
            set(predictions["alternative_view"]), {"poison", "random_control"}
        )
        metrics = {row["metric"] for row in report["dataset_results"]}
        self.assertEqual(
            metrics, {"clean_frr", "poison_tpr", "random_control_positive_rate"}
        )
        self.assertIn("sample_metrics", report)
        self.assertEqual(report["test_original_images"], 20)
        self.assertEqual(len(predictions), 2 * 4 * 20)
        poison_bags = predictions.loc[predictions["alternative_view"] == "poison"]
        control_bags = predictions.loc[
            predictions["alternative_view"] == "random_control"
        ]
        self.assertEqual(
            poison_bags["original_indices"].tolist(),
            control_bags["original_indices"].tolist(),
        )
        np.testing.assert_array_equal(
            poison_bags.loc[poison_bags["requested_rate"] == 0, "dataset_poison_score"],
            control_bags.loc[
                control_bags["requested_rate"] == 0, "dataset_poison_score"
            ],
        )

    def test_fitting_rejects_sample_leakage_into_reserve(self):
        self.sample["fit_original_indices"].append(100)
        with self.assertRaisesRegex(ValueError, "overlap"):
            self.fit_bundle()

    def test_evaluation_rejects_seen_test_images(self):
        bundle, _ = self.fit_bundle()
        bundle["fit_original_indices"].append(40)
        with self.assertRaisesRegex(ValueError, "overlap"):
            dataset.evaluate_dataset_detector(self.features, bundle, self.metadata)

    def test_prediction_reads_no_labels_or_views(self):
        bundle, _ = self.fit_bundle()
        rows = self.features.loc[
            (self.features["split"] == "test") & (self.features["view"] == "clean")
        ].head(10)
        raw = rows[["original_index"] + BASE_FEATURE_COLUMNS].copy()
        report = dataset.predict_dataset(raw, bundle, self.metadata)
        with_labels = raw.assign(
            detector_label="deliberately invalid",
            class_id=object(),
            view="not a benchmark",
        )
        self.assertEqual(
            report, dataset.predict_dataset(with_labels, bundle, self.metadata)
        )
        self.assertFalse(report["uses_poison_labels_at_inference"])
        self.assertIn(report["decision"], {"accept", "reject"})
        self.assertTrue(np.isfinite(report["dataset_poison_score"]))

    def test_prediction_rejects_incompatible_provenance_size_and_duplicate_ids(self):
        bundle, _ = self.fit_bundle()
        rows = self.features.loc[
            (self.features["split"] == "test") & (self.features["view"] == "clean")
        ].head(10)
        raw = rows[["original_index"] + BASE_FEATURE_COLUMNS].copy()
        different = dict(self.metadata, head_seed=999)
        with self.assertRaisesRegex(ValueError, "provenance"):
            dataset.predict_dataset(raw, bundle, different)
        with self.assertRaisesRegex(ValueError, "size"):
            dataset.predict_dataset(raw.head(9), bundle, self.metadata)
        duplicate = raw.copy()
        duplicate.iloc[1, duplicate.columns.get_loc("original_index")] = duplicate.iloc[
            0
        ]["original_index"]
        with self.assertRaisesRegex(ValueError, "once"):
            dataset.predict_dataset(duplicate, bundle, self.metadata)


if __name__ == "__main__":
    unittest.main()
