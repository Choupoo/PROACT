"""Offline domain diagnostics never train on or inspect held-out descriptors."""

import copy
import unittest
from unittest import mock

import numpy as np

from detector.reference_audit import audit_reference
from detector.tests import test_unsupervised as fixtures
from detector.tests.test_unsupervised import fit_small, metadata_fixture
from detector.unsupervised import FEATURE_COLUMNS, predict_dataset


class ReferenceAuditTests(unittest.TestCase):
    def test_audit_uses_disjoint_validation_clean_batches_and_no_test_descriptors(self):
        features = fixtures.BenchmarkBoundaryTests().benchmark()
        bundle = fit_small()
        before = copy.deepcopy(bundle)
        metadata = metadata_fixture("paired_benchmark")
        expected = audit_reference(bundle, features, metadata, task_size=5)
        irrelevant = ~(
            (features["split"] == "validation") & (features["view"] == "clean")
        )
        features.loc[irrelevant, FEATURE_COLUMNS] = np.nan
        actual = audit_reference(bundle, features, metadata, task_size=5)
        self.assertEqual(expected, actual)
        ids = [i for row in actual["batches"] for i in row["original_indices"]]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertTrue(all(100 <= i < 112 for i in ids))
        self.assertFalse(actual["test_used"])
        self.assertFalse(actual["threshold_tuned"])
        np.testing.assert_array_equal(before["projection"], bundle["projection"])

    def test_all_clean_alerts_fail_the_audit_without_hiding_raw_shift_results(self):
        features = fixtures.BenchmarkBoundaryTests().benchmark()
        with mock.patch(
            "detector.reference_audit.predict_dataset",
            return_value={
                "shift_detected": True,
                "p_value": 0.005,
                "mmd_squared_rff": 1.0,
            },
        ):
            result = audit_reference(
                fit_small(), features, metadata_fixture("paired_benchmark"), task_size=5
            )
        self.assertEqual(result["reference_domain_status"], "validation_failed")
        self.assertEqual(result["validation_clean_shift_alert_rate"], 1)
        self.assertEqual(result["deployment_action"], "abstain")

    def test_historical_shift_is_not_a_poisoning_decision(self):
        rows = fixtures.BenchmarkBoundaryTests().benchmark().iloc[:10]
        result = predict_dataset(
            fit_small(), rows[FEATURE_COLUMNS], metadata_fixture("incoming")
        )
        self.assertEqual(result["poisoning_decision"], "undetermined")
        self.assertEqual(result["deployment_action"], "abstain")
        self.assertIn("p_value", result)

    def test_invalid_or_insufficient_audit_data_rejected(self):
        features = fixtures.BenchmarkBoundaryTests().benchmark()
        with self.assertRaisesRegex(ValueError, "distinct"):
            audit_reference(
                fit_small(),
                features,
                metadata_fixture("paired_benchmark"),
                task_size=20,
            )
        features.loc[0, "original_index"] = 200
        with self.assertRaisesRegex(ValueError, "overlap"):
            audit_reference(
                fit_small(), features, metadata_fixture("paired_benchmark"), task_size=5
            )


if __name__ == "__main__":
    unittest.main()
