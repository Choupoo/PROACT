"""New-schema explanation and genuine incoming-image prediction regressions."""

import argparse
import copy
import json
from pathlib import Path
import pickle
import tempfile
import unittest
from unittest import mock

import numpy as np
import pandas as pd
import torch

from detector import explain_detector as explain, predict, transfer_detector as base
from detector import transfer_inference as incoming
from detector.common import sha256_file
from detector.extract_features import load_input_npz
from detector.image_data import dataset_spec, task_order
from detector.io_utils import DETECTOR_ROOT, save_frozen_bundle
from detector.tests.test_advisor import predicted
from detector.tests.test_features import checkpoint_fixture, make_model
from detector.tests.test_transfer import write_table


class EntryPointTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(dir=DETECTOR_ROOT / "tests")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def source(self, dataset="split_cifar100"):
        table, meta = predicted(1)
        table = table.drop(columns="true_class_probability")
        meta["feature_columns"].remove("true_class_probability")
        meta["inference_schema"] = "predicted_without_true_class_v1"
        if dataset != "split_cifar100":
            meta.update(dataset=dataset, dataset_sha256="d" * 64,
                        task_order=task_order(dataset), class_ids=task_order(dataset)[1])
            table["original_uid"] = table.original_index.map(lambda i: "{}:task1:row{}".format(dataset, i))
        return table, meta

    def test_explanation_cli_accepts_new_schema_without_changing_bundle(self):
        table, meta = self.source()
        path = self.root / "source.csv"
        meta = write_table(table, meta, path)
        bundle = base.fit_source(table, meta, feature_set="inference_full", task_size=16)
        frozen = save_frozen_bundle(bundle, self.root / "frozen")
        before = sha256_file(frozen)
        output = self.root / "explanation"
        explain.main(argparse.Namespace(features=str(path), bundle=str(frozen), output_dir=str(output),
                                         split="validation", ablations=False))
        result = json.loads((output / "explanation.json").read_text())
        self.assertNotIn("true_class_probability", result["feature_columns"])
        self.assertLess(result["max_additivity_error"], 1e-10)
        self.assertEqual(before, sha256_file(frozen))
        self.assertTrue((output / "feature_importance.png").is_file())

    def test_tiny_ten_class_explanation_uses_new_validator(self):
        table, meta = self.source("split_tiny_imagenet")
        table.loc[table.original_index == table.original_index.min(), "class_id"] = 9
        bundle = base.fit_source(table, meta, feature_set="inference_full", task_size=16)
        report = explain.explain(table, meta, bundle, self.root)
        self.assertLess(report["max_additivity_error"], 1e-10)

    def test_explanation_rejects_another_source_and_unsafe_legacy_ablations(self):
        table, meta = self.source()
        bundle = base.fit_source(table, meta, feature_set="inference_full", task_size=16)
        with self.assertRaisesRegex(ValueError, "original frozen source"):
            explain.explain(table, dict(meta, features_sha256="f" * 64), bundle, self.root)
        with self.assertRaisesRegex(ValueError, "historical calibration"):
            explain.explain(table, meta, bundle, self.root, ablations=True)
        self.assertEqual(list(self.root.iterdir()), [])

    def frozen_fixture(self, dataset="split_cifar100"):
        table, meta = self.source(dataset)
        bundle = base.fit_source(table, meta, feature_set="inference_context", task_size=16)
        bundle["calibration_history"] = [dict(task=1), dict(task=4)]
        spec = dataset_spec(dataset)
        checkpoint = dict(checkpoint_fixture(), dataset=dataset, task_num=spec["tasks"] - 1,
                          task_order=task_order(dataset), class_num=spec["classes_per_task"],
                          seed=meta["checkpoint_seed"], cont_method_args={"method": "ewc"})
        return bundle, checkpoint

    def test_deployment_contract_rejects_wrong_task_method_seed_or_label_policy(self):
        bundle, checkpoint = self.frozen_fixture()
        self.assertEqual(incoming.validate_deployment(bundle, checkpoint), (9, "split_cifar100"))
        for changed in (dict(checkpoint, task_num=4), dict(checkpoint, seed=99),
                        dict(checkpoint, cont_method_args={"method": "mas"})):
            with self.assertRaises(ValueError):
                incoming.validate_deployment(bundle, changed)
        for changed in (dict(bundle, class_label_policy="ground_truth"),
                        dict(bundle, feature_columns=bundle["feature_columns"] + ["true_class_probability"]),
                        dict(bundle, task_size=17)):
            with self.assertRaises(ValueError):
                incoming.validate_deployment(changed, checkpoint)

    def test_npz_predicted_loader_ignores_targets_at_all_native_resolutions(self):
        for size in (32, 64, 84):
            path = self.root / (str(size) + ".npz")
            np.savez(path, images=np.zeros((2, 3, size, size), dtype=np.float32),
                     targets=np.array([object()], dtype=object))
            images, targets = load_input_npz(path, "predicted", image_size=size)
            self.assertEqual(tuple(images.shape), (2, 3, size, size))
            self.assertIsNone(targets)

    def test_prediction_cli_end_to_end_without_incoming_labels(self):
        for dataset in ("split_cifar100", "split_mini_imagenet", "split_tiny_imagenet"):
            with self.subTest(dataset=dataset):
                case = self.root / dataset
                case.mkdir()
                bundle, checkpoint = self.frozen_fixture(dataset)
                frozen = save_frozen_bundle(bundle, case / "frozen")
                original_bundle = frozen.read_bytes()
                checkpoint_path = case / "checkpoint.pkl"
                with checkpoint_path.open("wb") as file:
                    pickle.dump(checkpoint, file)
                size = dataset_spec(dataset)["image_size"]
                refs = case / "inversions"
                refs.mkdir()
                rng = np.random.RandomState(33)
                for task in range(checkpoint["task_num"]):
                    np.savez(refs / ("inversions_tid_{:02d}.npz".format(task)),
                             x_dst=rng.rand(2, 3, size, size).astype(np.float32), y_dst=np.array([0, 1]))
                images = rng.rand(16, 3, size, size).astype(np.float32)
                source = case / "images.npz"
                np.savez(source, images=images)  # No labels, splits or clean/poison views.
                args = argparse.Namespace(bundle=str(frozen), checkpoint=str(checkpoint_path),
                    inversion_dir=str(refs), input_npz=str(source), output_dir=str(case / "prediction"), device="cpu")
                def synthetic_defender(*unused):
                    model = make_model()
                    model.heads = torch.nn.ModuleList([copy.deepcopy(model.heads[0])
                                                       for _ in range(checkpoint["task_num"] + 1)])
                    return model.eval()
                with mock.patch.object(incoming, "load_defender", side_effect=synthetic_defender):
                    predict.main(args)
                    np.savez(source, images=images, targets=np.array([object()], dtype=object))
                    again = copy.copy(args)
                    again.output_dir = str(case / "prediction_with_ignored_targets")
                    predict.main(again)
                first = pd.read_csv(case / "prediction/sample_predictions.csv")
                second = pd.read_csv(case / "prediction_with_ignored_targets/sample_predictions.csv")
                pd.testing.assert_frame_equal(first, second)
                result = json.loads((case / "prediction/prediction.json").read_text())
                self.assertEqual(result["dataset"], dataset)
                self.assertFalse(result["incoming_class_labels_used"])
                self.assertFalse(result["target_used_for_fitting_or_calibration"])
                self.assertEqual(result["suspicious_count"], int(first.flagged.sum()))
                self.assertEqual(result["alert"], result["suspicious_count"] >= bundle["count_calibration"]["critical_suspicious_count"])
                self.assertEqual(frozen.read_bytes(), original_bundle)
                columns = pd.read_csv(case / "prediction/incoming_features.csv").columns
                self.assertNotIn("true_class_probability", columns)
                self.assertNotIn("detector_label", columns)
                self.assertNotIn("class_id", columns)
                self.assertTrue(torch.isfinite(torch.tensor(first.score.to_numpy())).all())

    def test_wrong_incoming_size_fails_before_model_loading_or_writes(self):
        bundle, checkpoint = self.frozen_fixture()
        frozen = save_frozen_bundle(bundle, self.root / "frozen")
        source = self.root / "images.npz"
        np.savez(source, images=np.zeros((15, 3, 32, 32), dtype=np.float32))
        args = argparse.Namespace(bundle=str(frozen), checkpoint="unused", inversion_dir="unused",
            input_npz=str(source), output_dir=str(self.root / "prediction"), device="cpu")
        with mock.patch.object(incoming, "load_pickle", return_value=checkpoint), \
             mock.patch.object(incoming, "load_defender") as loader:
            with self.assertRaisesRegex(ValueError, "dataset size"):
                incoming.predict_incoming(args, bundle)
        loader.assert_not_called()
        self.assertFalse((self.root / "prediction").exists())


if __name__ == "__main__":
    unittest.main()
