"""Synthetic tests for the label-free reference/permutation detection boundary."""

import copy
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from detector.unsupervised import (
    FEATURE_COLUMNS,
    LOG_COLUMNS,
    evaluate_benchmark,
    fit_reference,
    load_bundle,
    permutation_mmd,
    predict_dataset,
    save_bundle,
)

DETECTOR_ROOT = Path(__file__).resolve().parents[1]


def metadata_fixture(role="historical_inversion"):
    return {
        "feature_protocol": "pretraining_full_v2",
        "head_mode": "defender_fixed",
        "head_seed": 13,
        "label_mode": "predicted",
        "checkpoint_sha256": "a" * 64,
        "inversion_sha256": "b" * 64,
        "features_sha256": "c" * 64,
        "origin_role": role,
        "model_unchanged": True,
    }


def descriptors(count=80, seed=20, reference=False, shifted=False):
    rng = np.random.default_rng(seed)
    values = {}
    for column in FEATURE_COLUMNS:
        if column in ("confidence", "margin"):
            values[column] = rng.uniform(0.3, 0.6, size=count)
        else:
            values[column] = np.exp(rng.normal(0.5, 0.15, size=count))
        if shifted and column in LOG_COLUMNS:
            values[column] *= 30
    table = pd.DataFrame(values)
    if reference:
        table["reference_id"] = [
            "task0:sample{}".format(index) for index in range(count)
        ]
        table["original_index"] = np.arange(count)
    return table


def fit_small(reference=None, **kwargs):
    if reference is None:
        reference = descriptors(reference=True)
    options = {"permutations": 39, "n_components": 32, "seed": 11, "max_reference": 30}
    options.update(kwargs)
    return fit_reference(reference, metadata_fixture(), **options)


class PermutationMathTests(unittest.TestCase):
    def test_identical_descriptors_have_zero_distance_and_unit_p_value(self):
        result = permutation_mmd(
            np.ones((5, 3)), np.ones((7, 3)), permutations=39, seed=2
        )
        self.assertEqual(result["mmd_squared_rff"], 0)
        self.assertEqual(result["p_value"], 1)
        self.assertEqual(result["extreme_permutations"], 39)

    def test_p_value_matches_direct_label_permutations_and_plus_one_correction(self):
        reference = np.array([[0.0, 1.0], [0.2, 0.8], [0.4, 0.5]])
        incoming = np.array([[1.0, 0.2], [1.2, 0.1]])
        pooled = np.concatenate([reference, incoming])
        difference = reference.mean(axis=0) - incoming.mean(axis=0)
        expected_statistic = float(difference.dot(difference))
        rng = np.random.default_rng(19)
        extreme = 0
        for _ in range(29):
            permutation = rng.permutation(len(pooled))
            first, second = pooled[permutation[:3]], pooled[permutation[3:]]
            delta = first.mean(axis=0) - second.mean(axis=0)
            extreme += delta.dot(delta) >= expected_statistic - 1e-12
        result = permutation_mmd(reference, incoming, permutations=29, seed=19)
        self.assertAlmostEqual(result["mmd_squared_rff"], expected_statistic)
        self.assertEqual(result["extreme_permutations"], extreme)
        self.assertEqual(result["p_value"], (extreme + 1) / 30)
        self.assertGreaterEqual(result["p_value"], result["minimum_p_value"])

    def test_large_distribution_shift_is_detected(self):
        result = permutation_mmd(
            np.zeros((20, 3)), np.ones((20, 3)), permutations=99, seed=7
        )
        self.assertEqual(result["mmd_squared_rff"], 3)
        self.assertLessEqual(result["p_value"], 0.05)
        self.assertGreater(result["p_value"], 0)

    def test_rejects_nonfinite_or_insufficient_samples(self):
        for reference in (np.ones((1, 2)), np.full((3, 2), np.nan)):
            with self.subTest(shape=reference.shape):
                with self.assertRaises(ValueError):
                    permutation_mmd(reference, np.ones((3, 2)), permutations=9, seed=0)


class ReferenceFitTests(unittest.TestCase):
    def test_fit_bank_membership_is_disjoint_reproducible_and_complete(self):
        bundle = fit_small()
        repeated = fit_small()
        self.assertEqual(len(bundle["fit_reference_ids"]), 40)
        self.assertEqual(len(bundle["bank_reference_ids"]), 30)
        self.assertEqual(len(bundle["unused_reference_ids"]), 10)
        fit_ids, bank_ids = (
            set(bundle["fit_reference_ids"]),
            set(bundle["bank_reference_ids"]),
        )
        self.assertFalse(fit_ids & bank_ids)
        combined = fit_ids | bank_ids | set(bundle["unused_reference_ids"])
        self.assertEqual(combined, set(bundle["reference_ids"]))
        self.assertEqual(bundle["reference_original_indices"], list(range(80)))
        self.assertEqual(bundle["reference_origin_role"], "historical_inversion")
        for key in ("median", "scale", "projection", "phase", "reference_bank_rff"):
            np.testing.assert_array_equal(bundle[key], repeated[key])

    def test_preprocessing_and_bandwidth_never_fit_on_reference_bank(self):
        reference = descriptors(reference=True)
        original = fit_small(reference)
        contaminated_bank = reference.copy()
        bank_mask = contaminated_bank.reference_id.isin(original["bank_reference_ids"])
        contaminated_bank.loc[bank_mask, LOG_COLUMNS] *= 100
        changed = fit_small(contaminated_bank)
        for key in ("median", "scale", "active_mask", "projection", "phase"):
            np.testing.assert_array_equal(original[key], changed[key])
        self.assertEqual(original["bandwidth"], changed["bandwidth"])
        self.assertFalse(
            np.allclose(original["reference_bank_rff"], changed["reference_bank_rff"])
        )

    def test_fit_ignores_arbitrary_attack_and_class_labels(self):
        reference = descriptors(reference=True)
        original = fit_small(reference)
        reference["detector_label"] = "do not inspect this field"
        reference["class_id"] = np.nan
        reference["view"] = "poison"
        reference["split"] = "test"
        changed = fit_small(reference)
        for key in ("median", "scale", "projection", "phase", "reference_bank_rff"):
            np.testing.assert_array_equal(original[key], changed[key])

    def test_zero_scale_features_are_removed_using_reference_fit_only(self):
        reference = descriptors(reference=True)
        reference["entropy"] = 1.0
        bundle = fit_small(reference)
        self.assertIn("entropy", bundle["removed_feature_columns"])
        self.assertNotIn("entropy", bundle["active_feature_columns"])
        reference[FEATURE_COLUMNS] = 0.5
        with self.assertRaises(ValueError):
            fit_small(reference)

    def test_true_targets_untrusted_origin_and_duplicate_ids_are_rejected(self):
        reference = descriptors(reference=True)
        for key, value in (
            ("label_mode", "true"),
            ("origin_role", "paired_benchmark"),
            ("head_mode", "attacker"),
        ):
            with self.subTest(field=key):
                metadata = metadata_fixture()
                metadata[key] = value
                with self.assertRaises(ValueError):
                    fit_reference(reference, metadata)
        reference.loc[0, "reference_id"] = reference.loc[1, "reference_id"]
        with self.assertRaises(ValueError):
            fit_small(reference)


class DatasetPredictionTests(unittest.TestCase):
    def test_prediction_uses_frozen_settings_and_ignores_labels(self):
        bundle = fit_small(max_incoming=12)
        incoming = descriptors(count=30, seed=10, shifted=True)
        original = predict_dataset(bundle, incoming, metadata_fixture("incoming"))
        incoming["detector_label"] = np.arange(len(incoming)) % 2
        incoming["view"] = "arbitrary"
        incoming["class_id"] = None
        changed = predict_dataset(bundle, incoming, metadata_fixture("incoming"))
        self.assertEqual(original, changed)
        self.assertEqual(original["incoming_samples_available"], 30)
        self.assertEqual(original["incoming_samples_tested"], 12)
        self.assertEqual(original["permutations"], bundle["settings"]["permutations"])
        self.assertTrue(original["shift_detected"])
        self.assertTrue(original["not_poisoning_probability"])
        self.assertNotIn("poisoning_probability", original)

    def test_mismatched_feature_provenance_is_rejected(self):
        bundle = fit_small()
        incoming = descriptors(count=10, seed=21)
        for key, value in (
            ("head_seed", 100),
            ("checkpoint_sha256", "d" * 64),
            ("inversion_sha256", "e" * 64),
        ):
            with self.subTest(field=key):
                metadata = metadata_fixture("incoming")
                metadata[key] = value
                with self.assertRaises(ValueError):
                    predict_dataset(bundle, incoming, metadata)

    def test_reference_identity_overlap_is_rejected(self):
        reference = descriptors(reference=True)
        bundle = fit_small(reference)
        with self.assertRaises(ValueError):
            predict_dataset(bundle, reference.iloc[:10], metadata_fixture())

    def test_save_load_is_immutable_and_preserves_predictions(self):
        bundle = fit_small()
        incoming = descriptors(count=15, seed=3)
        expected = predict_dataset(bundle, incoming, metadata_fixture("incoming"))
        with tempfile.TemporaryDirectory(dir=DETECTOR_ROOT) as directory:
            path = save_bundle(bundle, directory)
            restored = load_bundle(path)
            actual = predict_dataset(restored, incoming, metadata_fixture("incoming"))
            self.assertEqual(actual, expected)
            with self.assertRaises(FileExistsError):
                save_bundle(bundle, directory)
            with path.open("ab") as artifact:
                artifact.write(b"corruption")
            with self.assertRaises(ValueError):
                load_bundle(path)


class BenchmarkBoundaryTests(unittest.TestCase):
    def benchmark(self):
        tables = []
        for split in ("train", "validation", "test"):
            for view in ("clean", "poison", "random_control"):
                table = descriptors(count=12, seed=77, shifted=view != "clean")
                offset = {"train": 0, "validation": 100, "test": 200}[split]
                table["original_index"] = np.arange(12) + offset
                table["split"] = split
                table["view"] = view
                tables.append(table)
        return pd.concat(tables, ignore_index=True)

    def test_benchmark_uses_only_test_split_and_does_not_tune_bundle(self):
        bundle = fit_small(permutations=19)
        before = copy.deepcopy(bundle)
        benchmark = self.benchmark()
        expected = evaluate_benchmark(
            bundle,
            benchmark,
            metadata_fixture("paired_benchmark"),
            tasks_per_rate=2,
            task_size=10,
            seed=9,
        )
        benchmark.loc[benchmark["split"] != "test", FEATURE_COLUMNS] = np.nan
        actual = evaluate_benchmark(
            bundle,
            benchmark,
            metadata_fixture("paired_benchmark"),
            tasks_per_rate=2,
            task_size=10,
            seed=9,
        )
        self.assertEqual(expected, actual)
        self.assertEqual(actual["evaluation_split"], "test")
        self.assertEqual(len(actual["summary"]), 9)
        self.assertEqual(len(actual["tasks"]), 18)
        self.assertFalse(actual["threshold_tuned_on_evaluation"])
        self.assertIn("clean_task_false_rejection_rate", actual)
        self.assertEqual(bundle["settings"], before["settings"])
        np.testing.assert_array_equal(bundle["projection"], before["projection"])
        np.testing.assert_array_equal(
            bundle["reference_bank_rff"], before["reference_bank_rff"]
        )

    def test_cross_split_identity_overlap_is_rejected(self):
        benchmark = self.benchmark()
        benchmark.loc[0, "original_index"] = 200
        with self.assertRaises(ValueError):
            evaluate_benchmark(
                fit_small(),
                benchmark,
                metadata_fixture("paired_benchmark"),
                tasks_per_rate=1,
                task_size=10,
            )


if __name__ == "__main__":
    unittest.main()
