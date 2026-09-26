"""Isolation, transform invariance, permutation composition and real CLI checks."""

import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

from detector import explain_detector, shape_reference
from detector import transfer_detector as supervised
from detector.common import save_json, sha256_file
from detector.io_utils import DETECTOR_ROOT
from detector.tests import test_rank_reference as fixtures
from detector.tests.test_transfer import feature_fixture, write_table


class SupervisedShapeTests(unittest.TestCase):
    def test_target_scaling_does_not_change_predictions_or_frozen_threshold(self):
        source, source_meta = feature_fixture(1)
        target, target_meta = feature_fixture(9)
        bundle = supervised.fit_source(
            source, source_meta, feature_set="shape", task_size=16
        )
        first, predictions, _ = supervised.evaluate(
            target, target_meta, bundle, repeats=1
        )
        changed = target.copy()
        changed[shape_reference.COLUMNS] *= np.logspace(-80, 80, len(changed))[:, None]
        second, other, _ = supervised.evaluate(changed, target_meta, bundle, repeats=1)
        np.testing.assert_allclose(predictions.score, other.score, atol=1e-12)
        self.assertEqual(first["sample_threshold"], second["sample_threshold"])
        self.assertEqual(list(supervised.FEATURE_GROUPS), ["portable", "extended"])

    def test_source_test_and_random_control_never_change_fit(self):
        table, meta = feature_fixture()
        expected = supervised.fit_source(table, meta, feature_set="shape", task_size=16)
        table.loc[
            (table.split == "test") | (table.view == "random_control"),
            shape_reference.COLUMNS[0],
        ] *= 100
        actual = supervised.fit_source(table, meta, feature_set="shape", task_size=16)
        np.testing.assert_array_equal(
            expected["classifier"].coef_, actual["classifier"].coef_
        )
        self.assertEqual(expected["count_calibration"], actual["count_calibration"])

    def test_shap_applies_same_transform_and_reconstructs_actual_classifier(self):
        table, meta = feature_fixture()
        bundle = supervised.fit_source(table, meta, feature_set="shape", task_size=16)
        background = table.loc[
            (table.split == "train") & table.view.isin(["clean", "poison"])
        ]
        samples = table.loc[table.split == "validation"]
        phi, base, logits, error = explain_detector.linear_shap(
            bundle, background, samples
        )
        transformed = supervised.prepare_features(samples, "shape")
        expected = bundle["classifier"].decision_function(
            bundle["scaler"].transform(transformed[supervised.SHAPE_COLUMNS].to_numpy())
        )
        np.testing.assert_allclose(logits, expected)
        np.testing.assert_allclose(phi.sum(axis=1) + base, expected, atol=1e-10)
        self.assertLess(error, 1e-10)

    def test_zero_shape_and_missing_transform_fail_explicitly(self):
        table, meta = feature_fixture()
        bundle = supervised.fit_source(table, meta, feature_set="shape", task_size=16)
        del bundle["feature_transform"]
        with self.assertRaisesRegex(ValueError, "transform"):
            supervised.evaluate(table, meta, bundle, source_control=True, repeats=1)
        table.loc[0, shape_reference.COLUMNS] = 0
        with self.assertRaisesRegex(ValueError, "Zero-gradient"):
            supervised.fit_source(table, meta, feature_set="shape", task_size=16)


class HistoricalShapeTests(unittest.TestCase):
    def fit(self, table=None):
        return shape_reference.fit_reference(
            fixtures.reference() if table is None else table,
            fixtures.metadata_fixture(),
            permutations=39,
            n_components=16,
        )

    def test_kernel_fit_does_not_see_bank(self):
        reference = fixtures.reference()
        original = self.fit(reference)
        bank_ids = [
            identity for p in original["profiles"] for identity in p["bank_ids"]
        ]
        reference.loc[
            reference.reference_id.isin(bank_ids), shape_reference.COLUMNS[0]
        ] *= 99
        changed = self.fit(reference)
        for left, right in zip(original["profiles"], changed["profiles"]):
            self.assertEqual(left["bandwidth"], right["bandwidth"])
            np.testing.assert_array_equal(left["projection"], right["projection"])
            self.assertFalse(np.array_equal(left["bank_rff"], right["bank_rff"]))
            self.assertFalse(set(left["fit_ids"]) & set(left["bank_ids"]))

    def test_fit_and_predict_ignore_labels_and_incoming_magnitude(self):
        reference = fixtures.reference()
        original = self.fit(reference)
        reference["class_id"], reference["view"], reference["detector_label"] = (
            "forbidden",
            "poison",
            np.nan,
        )
        changed = self.fit(reference)
        for left, right in zip(original["profiles"], changed["profiles"]):
            np.testing.assert_array_equal(left["bank_rff"], right["bank_rff"])
        table = fixtures.descriptors(32)
        first = shape_reference.predict_dataset(
            original, table, fixtures.metadata_fixture("incoming")
        )
        table[shape_reference.COLUMNS] *= np.logspace(-100, 100, len(table))[:, None]
        table["detector_label"], table["view"] = 1, "poison"
        second = shape_reference.predict_dataset(
            original, table, fixtures.metadata_fixture("incoming")
        )
        self.assertEqual(first["p_value"], second["p_value"])
        self.assertEqual(first["shift_detected"], second["shift_detected"])

    def test_union_rule_uses_maximum_not_minimum(self):
        bundle = self.fit()
        with mock.patch.object(
            shape_reference,
            "permutation_mmd",
            side_effect=[{"p_value": 0.01}, {"p_value": 0.8}],
        ):
            result = shape_reference.predict_dataset(
                bundle, fixtures.descriptors(24), fixtures.metadata_fixture("incoming")
            )
        self.assertEqual(result["p_value"], 0.8)
        self.assertFalse(result["shift_detected"])

    def test_large_shape_change_detectable_without_majority_clean(self):
        table = fixtures.descriptors(80)
        table[shape_reference.COLUMNS[0]] *= 10000
        result = shape_reference.predict_dataset(
            self.fit(), table, fixtures.metadata_fixture("incoming")
        )
        self.assertTrue(result["shift_detected"])
        self.assertGreater(result["p_value"], 0)
        self.assertEqual(result["poisoning_decision"], "undetermined")

    def test_zero_gradient_is_abstention_not_clean(self):
        table = fixtures.descriptors(24)
        table.loc[0, shape_reference.COLUMNS] = 0
        result = shape_reference.predict_dataset(
            self.fit(), table, fixtures.metadata_fixture("incoming")
        )
        self.assertIsNone(result["shift_detected"])
        self.assertEqual(result["invalid_samples"], 1)

    def test_origin_provenance_identity_and_size_checks(self):
        metadata = fixtures.metadata_fixture()
        del metadata["model_unchanged"]
        with self.assertRaisesRegex(ValueError, "frozen model"):
            shape_reference.fit_reference(fixtures.reference(), metadata)
        with self.assertRaises(ValueError):
            shape_reference.fit_reference(
                fixtures.reference(), fixtures.metadata_fixture("incoming")
            )
        meta = dict(fixtures.metadata_fixture(), label_mode="ground_truth")
        with self.assertRaises(ValueError):
            shape_reference.fit_reference(fixtures.reference(), meta)
        bundle = self.fit()
        with self.assertRaises(ValueError):
            shape_reference.predict_dataset(
                bundle, fixtures.reference(), fixtures.metadata_fixture()
            )
        with self.assertRaises(ValueError):
            shape_reference.predict_dataset(
                bundle, fixtures.descriptors(300), fixtures.metadata_fixture("incoming")
            )
        corrupt = copy.deepcopy(bundle)
        corrupt["profiles"][0]["bank_ids"][0] = corrupt["profiles"][0]["fit_ids"][0]
        with self.assertRaises(ValueError):
            shape_reference.validate_bundle(corrupt)

    def test_historical_audit_excludes_whole_task(self):
        audit = shape_reference.historical_audit(
            fixtures.reference(),
            fixtures.metadata_fixture(),
            permutations=39,
            n_components=16,
        )
        for row in audit["tasks"]:
            self.assertNotIn(
                row["held_out_task"], [p["task_id"] for p in row["per_task"]]
            )


class RevisionCliTests(unittest.TestCase):
    def execute(self, *args, success=True):
        result = subprocess.run(
            [sys.executable, "-B", "-m", "detector.revision_study", *map(str, args)],
            cwd=DETECTOR_ROOT.parent,
            capture_output=True,
            text=True,
        )
        if success:
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        else:
            self.assertNotEqual(result.returncode, 0)
        return result

    def test_both_complete_cli_flows_preserve_inputs_and_reject_overwrite(self):
        with tempfile.TemporaryDirectory(dir=DETECTOR_ROOT) as directory:
            root = Path(directory)
            source = root / "source"
            for task in (1, 9):
                folder = source / "seed3" / ("features_task" + str(task))
                folder.mkdir(parents=True)
                table, meta = feature_fixture(task)
                meta["checkpoint_seed"] = 3
                write_table(table, meta, folder / "features.csv")
            args = [
                "supervised",
                "--source-run",
                source,
                "--output-dir",
                root / "supervised",
                "--seeds",
                3,
                "--task-size",
                16,
                "--bags-per-rate",
                1,
            ]
            self.execute(*args, "--dry-run")
            self.assertFalse((root / "supervised").exists())
            before = {str(p): sha256_file(p) for p in source.rglob("*") if p.is_file()}
            self.execute(*args)
            self.execute(*args, success=False)
            state = json.loads((root / "supervised/run_state.json").read_text())
            self.assertEqual(state["status"], "complete")

            self.assertTrue(
                (root / "supervised/seed3/evaluate_task9_shape/rates.csv").is_file()
            )
            for path, value in before.items():
                self.assertEqual(sha256_file(path), value)
            unlabeled = root / "unlabeled"
            unlabeled.mkdir()
            for filename, table, role in (
                (
                    "reference_features.csv",
                    fixtures.reference(),
                    "historical_inversion",
                ),
                ("predicted_features.csv", fixtures.benchmark(), "paired_benchmark"),
            ):
                meta = dict(
                    fixtures.metadata_fixture(role),
                    feature_columns=list(fixtures.legacy.FEATURE_COLUMNS),
                    synthetic=True,
                )
                write_table(table, meta, unlabeled / filename)
            args = [
                "unsupervised",
                "--source-run",
                unlabeled,
                "--output-dir",
                root / "unsupervised",
                "--task-size",
                16,
                "--bags-per-rate",
                1,
            ]
            self.execute(*args)
            result = json.loads(
                (root / "unsupervised/evaluation_metrics.json").read_text()
            )
            self.assertTrue(result["synthetic"])
            self.assertIn("shape_alert_rate", result["summary"][0])
            state = json.loads((root / "unsupervised/run_state.json").read_text())
            self.assertEqual(state["status"], "complete")

            # A corrupt target fails only after historical models were persisted;
            # no target descriptors can influence their construction.
            sidecar = unlabeled / "predicted_features.metadata.json"
            corrupt = json.loads(sidecar.read_text())
            corrupt["features_sha256"] = "0" * 64
            save_json(corrupt, sidecar)
            failed = root / "failed_unsupervised"
            self.execute(
                "unsupervised",
                "--source-run",
                unlabeled,
                "--output-dir",
                failed,
                "--task-size",
                16,
                "--bags-per-rate",
                1,
                success=False,
            )
            state = json.loads((failed / "run_state.json").read_text())
            self.assertEqual(state["status"], "failed")
            self.assertEqual(
                state["steps"]["fit_and_freeze_historical_only_models"]["status"],
                "complete",
            )
            self.assertEqual(
                state["steps"]["load_benchmark_after_freeze"]["status"], "failed"
            )
            self.assertTrue((failed / "models/shape/bundle.joblib").is_file())

    def test_missing_features_fail_before_creating_output(self):
        with tempfile.TemporaryDirectory(dir=DETECTOR_ROOT) as directory:
            root = Path(directory)
            self.execute(
                "unsupervised",
                "--source-run",
                root / "missing",
                "--output-dir",
                root / "output",
                success=False,
            )
            self.assertFalse((root / "output").exists())


if __name__ == "__main__":
    unittest.main()
