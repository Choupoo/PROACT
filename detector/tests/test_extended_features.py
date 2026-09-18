"""Independent numerical checks for the full feature and unlabeled-input paths."""

import pickle
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch
import torch.nn.functional as F

from detector import FEATURE_COLUMNS, GRAD_LAYER_PREFIX, GRAD_PARAM_PREFIX
from detector.common import assert_model_unchanged, snapshot_state_dict
from detector.extract_features import (
    backbone_parameters,
    build_parser,
    build_past_references,
    extract_one,
    gradient_group_features,
    load_input_npz,
    main,
    reference_samples,
)
from test_features import checkpoint_fixture, direct_gradient, make_model, sample_images


class ExtendedFeatureTests(unittest.TestCase):
    def setUp(self):
        self.model = make_model()
        self.named, self.parameters = backbone_parameters(self.model)
        self.image = sample_images(1, 311)[0]
        self.gradient = direct_gradient(
            self.model, self.image.unsqueeze(0), torch.tensor([2]), 9
        )
        direction = torch.linspace(-1, 1, self.gradient.numel())
        self.direction = direction / direction.norm()

    def test_uncertainty_activation_and_task_cosines_match_independent_calculations(
        self,
    ):
        captured = []
        hook = self.model.heads[-1].register_forward_pre_hook(
            lambda _module, inputs: captured.append(inputs[0].detach())
        )
        try:
            logits = self.model(self.image.unsqueeze(0))[-1][0].detach()
        finally:
            hook.remove()
        probabilities = logits.softmax(dim=0)
        ordered = probabilities.sort(descending=True).values
        unit_gradient = self.gradient / self.gradient.norm()
        tasks = torch.stack([unit_gradient, -unit_gradient, self.direction])
        expected_cosines = F.cosine_similarity(self.gradient.unsqueeze(0), tasks, dim=1)
        before = snapshot_state_dict(self.model)
        features = extract_one(
            self.model,
            self.named,
            self.parameters,
            self.direction,
            self.image,
            2,
            "cpu",
            task_directions=tasks,
        )
        self.assertTrue(set(FEATURE_COLUMNS).issubset(features))
        self.assertEqual(len(FEATURE_COLUMNS), 16)
        self.assertEqual(len(set(FEATURE_COLUMNS)), 16)
        self.assertAlmostEqual(
            features["entropy"],
            -(probabilities * probabilities.log()).sum().item(),
            places=6,
        )
        self.assertAlmostEqual(features["confidence"], ordered[0].item(), places=6)
        self.assertAlmostEqual(
            features["true_class_probability"], probabilities[2].item(), places=6
        )
        self.assertAlmostEqual(
            features["margin"], (ordered[0] - ordered[1]).item(), places=6
        )
        self.assertAlmostEqual(
            features["activation_norm_l2"], captured[0].norm().item(), places=6
        )
        for suffix, value in (
            ("min", expected_cosines.min()),
            ("max", expected_cosines.max()),
            ("mean", expected_cosines.mean()),
        ):
            self.assertAlmostEqual(
                features["grad_cosine_task_" + suffix], value.item(), places=6
            )
        for task_id, value in enumerate(expected_cosines):
            self.assertAlmostEqual(
                features["grad_cosine_task_{}".format(task_id)], value.item(), places=6
            )
        assert_model_unchanged(self.model, before)
        self.assertEqual(len(self.model.heads[-1]._forward_pre_hooks), 0)

    def test_predicted_mode_never_converts_or_uses_supplied_target(self):
        class UnreadableTarget:
            def __int__(self):
                raise AssertionError("True label must not be read in predicted mode.")

        features = extract_one(
            self.model,
            self.named,
            self.parameters,
            self.direction,
            self.image,
            UnreadableTarget(),
            "cpu",
            label_mode="predicted",
        )
        logits = self.model(self.image.unsqueeze(0))[-1]
        predicted = logits.argmax(dim=1)
        expected = direct_gradient(self.model, self.image.unsqueeze(0), predicted, 9)
        self.assertAlmostEqual(
            features["loss"], F.cross_entropy(logits, predicted).item(), places=6
        )
        self.assertAlmostEqual(
            features["grad_norm_l2"], expected.norm().item(), places=6
        )
        self.assertEqual(features["true_class_probability"], features["confidence"])

    def test_per_tensor_and_layer_norms_reconstruct_global_norm(self):
        gradients = [
            torch.full_like(parameter, (index + 1) / 100)
            for index, parameter in enumerate(self.parameters)
        ]
        features = gradient_group_features(self.named, gradients)
        parameter_sum = sum(
            value**2
            for name, value in features.items()
            if name.startswith(GRAD_PARAM_PREFIX)
        )
        layer_sum = sum(
            value**2
            for name, value in features.items()
            if name.startswith(GRAD_LAYER_PREFIX)
        )
        expected = sum(
            gradient.double().square().sum().item() for gradient in gradients
        )
        self.assertAlmostEqual(parameter_sum, expected, places=5)
        self.assertAlmostEqual(layer_sum, expected, places=5)
        for (name, _), gradient in zip(self.named, gradients):
            self.assertAlmostEqual(
                features[GRAD_PARAM_PREFIX + name], gradient.norm().item(), places=6
            )
        self.assertNotIn(GRAD_PARAM_PREFIX + "heads.9.weight", features)

    def test_activation_hook_is_removed_after_forward_failure(self):
        original_forward = self.model.forward

        def fail_after_head(images):
            original_forward(images)
            raise RuntimeError("Intentional forward failure.")

        with mock.patch.object(self.model, "forward", fail_after_head):
            with self.assertRaisesRegex(RuntimeError, "Intentional"):
                extract_one(
                    self.model,
                    self.named,
                    self.parameters,
                    self.direction,
                    self.image,
                    2,
                    "cpu",
                )
        self.assertEqual(len(self.model.heads[-1]._forward_pre_hooks), 0)

    def test_reference_matrix_uses_each_distinct_historical_head_in_task_order(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            paths, expected = [], {}
            for task_id in (2, 0, 1):
                images = sample_images(task_id + 2, task_id + 70)
                labels = torch.arange(len(images)) % 3
                path = Path(directory) / "task_{}.npz".format(task_id)
                np.savez(path, x_dst=images.numpy(), y_dst=labels.numpy(), tid=task_id)
                paths.append(path)
                gradient = direct_gradient(self.model, images, labels, task_id)
                expected[task_id] = gradient / gradient.norm()
            past, matrix, records = build_past_references(
                self.model, paths, "cpu", batch_size=2
            )
            self.assertEqual([record["task_id"] for record in records], [0, 1, 2])
            torch.testing.assert_close(
                matrix,
                torch.stack([expected[index] for index in range(3)]),
                rtol=1e-5,
                atol=1e-6,
            )
            torch.testing.assert_close(past, F.normalize(matrix.mean(dim=0), dim=0))


class InputModeTests(unittest.TestCase):
    def test_predicted_npz_loading_ignores_even_unreadable_object_targets(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            path = Path(directory) / "incoming.npz"
            np.savez(
                path,
                images=sample_images(2, 88).numpy(),
                targets=np.array([object()], dtype=object),
            )
            images, targets = load_input_npz(path, "predicted")
            self.assertEqual(tuple(images.shape), (2, 3, 32, 32))
            self.assertIsNone(targets)
            with self.assertRaises(ValueError):
                load_input_npz(path, "ground_truth")

    def test_raw_input_rejects_fractional_labels_and_out_of_range_pixels(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            path = Path(directory) / "incoming.npz"
            images = sample_images(2, 92).numpy()
            np.savez(path, images=images, targets=np.array([0.5, 1]))
            with self.assertRaises(ValueError):
                load_input_npz(path, "ground_truth")
            images[0, 0, 0, 0] = 1.1
            np.savez(path, images=images)
            with self.assertRaises(ValueError):
                load_input_npz(path, "predicted")

    def test_reference_rows_have_unique_ids_without_true_or_poison_labels(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            paths = []
            for task_id in range(2):
                path = Path(directory) / "reference_{}.npz".format(task_id)
                np.savez(
                    path, x_dst=sample_images(2, 30 + task_id).numpy(), tid=task_id
                )
                paths.append(path)
            rows = list(reference_samples(paths))
            self.assertEqual([row[2]["original_index"] for row in rows], [0, 1, 2, 3])
            self.assertEqual(len({row[2]["reference_id"] for row in rows}), 4)
            for _, target, metadata in rows:
                self.assertIsNone(target)
                self.assertNotIn("detector_label", metadata)
                self.assertNotIn("view", metadata)

    def test_incoming_and_reference_cli_emit_matching_provenance(self):
        from detector.io_utils import read_feature_table

        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            folder = Path(directory)
            inversion = folder / "inversion"
            inversion.mkdir()
            for task_id in range(9):
                np.savez(
                    inversion / "data_tid_{:02d}.npz".format(task_id),
                    x_dst=sample_images(2, 120 + task_id).numpy(),
                    y_dst=np.array([0, 1]),
                    tid=task_id,
                )
            checkpoint = checkpoint_fixture()
            checkpoint["model"] = make_model().state_dict()
            checkpoint_path = folder / "checkpoint.pkl"
            with checkpoint_path.open("wb") as stream:
                pickle.dump(checkpoint, stream)
            incoming_path = folder / "incoming.npz"
            np.savez(incoming_path, images=sample_images(3, 61).numpy())
            common = [
                "--checkpoint",
                str(checkpoint_path),
                "--inversion-dir",
                str(inversion),
                "--label-mode",
                "predicted",
                "--device",
                "cpu",
                "--log-every",
                "0",
            ]
            stub_utils = types.SimpleNamespace(
                create_load_add_head=lambda **_kwargs: make_model()
            )
            with mock.patch.dict(sys.modules, {"utils": stub_utils}):
                incoming_output = folder / "incoming.csv"
                main(
                    build_parser().parse_args(
                        common
                        + [
                            "--input-npz",
                            str(incoming_path),
                            "--output",
                            str(incoming_output),
                        ]
                    )
                )
                reference_output = folder / "reference.csv"
                main(
                    build_parser().parse_args(
                        common + ["--reference-only", "--output", str(reference_output)]
                    )
                )
            incoming, incoming_meta = read_feature_table(incoming_output)
            reference, reference_meta = read_feature_table(reference_output)
            self.assertEqual(len(incoming), 3)
            self.assertEqual(len(reference), 18)
            self.assertEqual(incoming_meta["origin_role"], "incoming")
            self.assertEqual(reference_meta["origin_role"], "historical_inversion")
            self.assertEqual(
                incoming_meta["checkpoint_sha256"], reference_meta["checkpoint_sha256"]
            )
            self.assertEqual(
                incoming_meta["inversion_sha256"], reference_meta["inversion_sha256"]
            )
            self.assertEqual(
                incoming_meta["feature_columns"], reference_meta["feature_columns"]
            )
            self.assertTrue(incoming_meta["model_unchanged"])
            self.assertNotIn("detector_label", incoming)
            self.assertNotIn("detector_label", reference)


if __name__ == "__main__":
    unittest.main()
