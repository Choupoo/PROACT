"""CPU regression tests for feature math, frozen state, and sample identity.

Run from PROACT with ``python -m unittest discover -s detector/tests -v``.
All fixtures are synthetic; no checkpoints or datasets need downloading.
"""

import copy
import random
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from detector import FEATURE_COLUMNS, STAGE_NAMES
from detector.common import (
    assert_model_unchanged,
    compare_checkpoint_identity,
    initialize_defender_head,
    matching_inversion_files,
    snapshot_state_dict,
    validate_attack_artifact,
)
from detector.create_manifest import (
    create_manifest,
    validate_manifest,
    validate_manifest_table,
)
from detector.extract_features import (
    backbone_parameters,
    compute_past_gradient_direction,
    compute_stage_gradient_norms,
    extract_one,
    matched_random_noise,
)
from torch import nn


class TinyMultiheadModel(nn.Module):
    """Small differentiable model with the supported ResNet group names."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Linear(3, 4)
        self.bn1 = nn.BatchNorm1d(4)
        self.layer1 = nn.Linear(4, 4)
        self.layer2 = nn.Linear(4, 4)
        self.layer3 = nn.Linear(4, 4)
        self.layer4 = nn.Linear(4, 4)
        self.heads = nn.ModuleList(nn.Linear(4, 3) for _ in range(10))

    def forward(self, images):
        features = images.mean(dim=(-2, -1))
        features = torch.tanh(self.bn1(self.conv1(features)))
        for layer in (self.layer1, self.layer2, self.layer3, self.layer4):
            features = torch.tanh(layer(features))
        return [head(features) for head in self.heads]


def make_model():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(101)
        return TinyMultiheadModel().eval()


def sample_images(count, seed):
    generator = torch.Generator().manual_seed(seed)
    return torch.rand(count, 3, 32, 32, generator=generator)


def direct_gradient(model, images, targets, task_id):
    """Independent oracle: differentiate one full-batch mean loss."""
    parameters = [
        parameter
        for name, parameter in model.named_parameters()
        if not name.startswith("heads.")
    ]
    loss = F.cross_entropy(model(images)[task_id], targets)
    gradients = torch.autograd.grad(loss, parameters)
    return torch.cat([gradient.reshape(-1) for gradient in gradients]).detach()


def checkpoint_fixture():
    return {
        "dataset": "split_cifar100",
        "task_num": 9,
        "seed": 0,
        "task_order": [list(range(task * 10, (task + 1) * 10)) for task in range(10)],
        "model": {"conv1.weight": torch.ones(2, 3)},
        "class_num": 10,
        "model_type": "resnet",
    }


class ManifestTests(unittest.TestCase):
    def setUp(self):
        self.targets = np.repeat(np.arange(3), 9)
        self.counts = {
            "train_per_class": 3,
            "validation_per_class": 2,
            "test_per_class": 1,
        }
        self.manifest = create_manifest(self.targets, split_seed=17, **self.counts)

    def test_split_is_reproducible_complete_and_stratified(self):
        repeated = create_manifest(self.targets, split_seed=17, **self.counts)
        pd.testing.assert_frame_equal(self.manifest, repeated)
        validate_manifest(self.manifest, self.targets, **self.counts)
        validate_manifest_table(self.manifest, self.targets)
        self.assertEqual(self.manifest["original_index"].tolist(), list(range(27)))
        for class_id in range(3):
            counts = self.manifest.loc[self.manifest.class_id == class_id, "split"]
            self.assertEqual(
                counts.value_counts().to_dict(),
                {"train": 3, "validation": 2, "test": 1, "reserve": 3},
            )

    def test_rejects_duplicate_identity_and_incorrect_labels(self):
        for column, value in (("original_index", 1), ("class_id", 2)):
            with self.subTest(column=column):
                invalid = self.manifest.copy()
                invalid.loc[0, column] = value
                with self.assertRaises((ValueError, RuntimeError)):
                    validate_manifest_table(invalid, self.targets)

    def test_rejects_unknown_split(self):
        invalid = self.manifest.copy()
        invalid.loc[0, "split"] = "holdout_typo"
        with self.assertRaises((ValueError, RuntimeError)):
            validate_manifest_table(invalid, self.targets)

    def test_rejects_fractional_identity_instead_of_truncating_it(self):
        invalid = self.manifest.astype({"original_index": float})
        invalid.loc[0, "original_index"] = 0.5
        with self.assertRaises((ValueError, RuntimeError)):
            validate_manifest_table(invalid, self.targets)

    def test_requires_nonempty_reserve(self):
        with self.assertRaises(ValueError):
            create_manifest(np.zeros(6), split_seed=17, **self.counts)


class FeatureMathTests(unittest.TestCase):
    def test_actual_proact_resnet_supports_every_backbone_parameter_group(self):
        from resnet import ResNet18

        # Use the project's real block/shortcut/BN layout at a smaller width.
        model = ResNet18(task_num=10, nclasses=10, nf=4).eval()
        named, parameters = backbone_parameters(model)
        direction = torch.ones(sum(parameter.numel() for parameter in parameters))
        direction /= direction.norm()
        before = snapshot_state_dict(model)
        result = extract_one(
            model, named, parameters, direction, torch.rand(3, 32, 32), 3, "cpu"
        )
        self.assertTrue(set(FEATURE_COLUMNS).issubset(result))
        self.assertTrue(all(np.isfinite(value) for value in result.values()))
        self.assertGreater(result["grad_norm_l2"], 0)
        assert_model_unchanged(model, before)

    def setUp(self):
        self.model = make_model()
        self.named, self.parameters = backbone_parameters(self.model)

    def test_backbone_excludes_all_task_heads(self):
        expected = {
            id(parameter)
            for name, parameter in self.model.named_parameters()
            if not name.startswith("heads.")
        }
        self.assertEqual({id(parameter) for parameter in self.parameters}, expected)
        self.assertTrue(all(not name.startswith("heads.") for name, _ in self.named))

    def test_stage_norms_reconstruct_global_gradient(self):
        gradients = [
            torch.full_like(parameter, (index + 1) / 10)
            for index, parameter in enumerate(self.parameters)
        ]
        norms = compute_stage_gradient_norms(self.named, gradients)
        expected_squared = sum(
            gradient.double().square().sum().item() for gradient in gradients
        )
        self.assertEqual(
            set(norms), {f"grad_norm_stage_{stage}" for stage in STAGE_NAMES}
        )
        self.assertAlmostEqual(
            sum(value**2 for value in norms.values()), expected_squared, places=4
        )

    def test_stage_norms_reject_missing_gradient_entries(self):
        gradients = [torch.ones_like(parameter) for parameter in self.parameters[:-1]]
        with self.assertRaises((ValueError, RuntimeError)):
            compute_stage_gradient_norms(self.named, gradients)

    def test_single_sample_features_match_direct_derivatives_and_preserve_state(self):
        image = sample_images(1, 10)[0]
        target = 2
        expected_gradient = direct_gradient(
            self.model, image.unsqueeze(0), torch.tensor([target]), 9
        )
        reference = torch.linspace(-1, 1, expected_gradient.numel())
        reference = reference / reference.norm()
        expected_loss = F.cross_entropy(
            self.model(image.unsqueeze(0))[-1], torch.tensor([target])
        )
        before = snapshot_state_dict(self.model)

        features = extract_one(
            self.model,
            self.named,
            self.parameters,
            reference,
            image,
            target,
            torch.device("cpu"),
        )

        self.assertTrue(set(FEATURE_COLUMNS).issubset(features))
        self.assertAlmostEqual(features["loss"], expected_loss.item(), places=6)
        self.assertAlmostEqual(
            features["grad_norm_l2"], expected_gradient.norm().item(), places=6
        )
        expected_cosine = F.cosine_similarity(
            expected_gradient, reference, dim=0
        ).item()
        self.assertAlmostEqual(features["grad_cosine_past"], expected_cosine, places=6)
        self.assertTrue(
            all(parameter.grad is None for parameter in self.model.parameters())
        )
        assert_model_unchanged(self.model, before)

    def test_reference_uses_task_ids_and_weights_partial_batches_by_sample_count(self):
        images_by_task = {0: sample_images(5, 22), 1: sample_images(3, 23)}
        targets_by_task = {0: torch.tensor([0, 2, 1, 0, 2]), 1: torch.tensor([1, 1, 2])}
        gradients = [
            direct_gradient(
                self.model, images_by_task[task], targets_by_task[task], task
            )
            for task in range(2)
        ]
        expected = sum(gradient / gradient.norm() for gradient in gradients)
        expected = expected / expected.norm()
        before = snapshot_state_dict(self.model)

        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            paths = []
            for task_id in (1, 0):
                path = Path(directory) / f"reference_{task_id}.npz"
                np.savez(
                    path,
                    x_dst=images_by_task[task_id].numpy(),
                    y_dst=targets_by_task[task_id].numpy(),
                    tid=task_id,
                )
                paths.append(path)
            actual, records = compute_past_gradient_direction(
                self.model, paths, device=torch.device("cpu"), batch_size=2
            )

        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
        by_task = {record["task_id"]: record for record in records}
        self.assertEqual(set(by_task), {0, 1})
        for task_id in range(2):
            self.assertEqual(by_task[task_id]["samples"], len(images_by_task[task_id]))
            self.assertAlmostEqual(
                by_task[task_id]["mean_gradient_norm"],
                gradients[task_id].norm().item(),
                places=6,
            )
        assert_model_unchanged(self.model, before)

    def test_reference_rejects_cancelling_task_directions(self):
        with torch.no_grad():
            self.model.layer4.weight.zero_()
            self.model.layer4.bias.zero_()
            head_weight = torch.tensor(
                [[1.0, 0, 0, 0], [-1.0, 0, 0, 0], [0.0, 0, 0, 0]]
            )
            for task_id, sign in ((0, 1), (1, -1)):
                self.model.heads[task_id].weight.copy_(sign * head_weight)
                self.model.heads[task_id].bias.zero_()
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            paths = []
            for task_id in range(2):
                path = Path(directory) / f"reference_tid_{task_id:02d}.npz"
                np.savez(path, x_dst=np.zeros((1, 3, 32, 32)), y_dst=np.array([0]))
                paths.append(path)
            with self.assertRaises((ValueError, RuntimeError)):
                compute_past_gradient_direction(
                    self.model, paths, torch.device("cpu"), batch_size=1
                )

    def test_training_mode_is_rejected_before_batchnorm_mutation(self):
        self.model.train()
        before = snapshot_state_dict(self.model)
        reference = torch.ones(sum(parameter.numel() for parameter in self.parameters))
        with self.assertRaises((ValueError, RuntimeError)):
            extract_one(
                self.model,
                self.named,
                self.parameters,
                reference,
                sample_images(1, 2)[0],
                1,
                torch.device("cpu"),
            )
        assert_model_unchanged(self.model, before)


class FrozenStateAndNoiseTests(unittest.TestCase):
    def test_head_reset_is_deterministic_and_preserves_global_rng_states(self):
        model = make_model()
        before = snapshot_state_dict(model)
        python_rng = random.getstate()
        numpy_rng = np.random.get_state()
        torch_rng = torch.random.get_rng_state().clone()

        self.assertEqual(initialize_defender_head(model, head_seed=45), 9)

        self.assertEqual(random.getstate(), python_rng)
        numpy_after = np.random.get_state()
        self.assertEqual(numpy_after[0], numpy_rng[0])
        np.testing.assert_array_equal(numpy_after[1], numpy_rng[1])
        self.assertEqual(numpy_after[2:], numpy_rng[2:])
        torch.testing.assert_close(
            torch.random.get_rng_state(), torch_rng, rtol=0, atol=0
        )
        after = snapshot_state_dict(model)
        for name, original in before.items():
            if not name.startswith("heads.9."):
                torch.testing.assert_close(after[name], original, rtol=0, atol=0)
        self.assertFalse(torch.equal(before["heads.9.weight"], after["heads.9.weight"]))
        initialize_defender_head(model, head_seed=45)
        assert_model_unchanged(model, after)

    def test_snapshot_detects_parameter_and_buffer_mutations(self):
        for tensor_name in ("weight", "running_mean"):
            with self.subTest(tensor=tensor_name):
                model = make_model()
                before = snapshot_state_dict(model)
                with torch.no_grad():
                    getattr(model.bn1, tensor_name).add_(1)
                with self.assertRaises(RuntimeError):
                    assert_model_unchanged(model, before)

    def test_random_control_preserves_perturbation_norms_before_clipping(self):
        perturbation = torch.linspace(-0.3, 0.25, 60).reshape(3, 4, 5)
        before = perturbation.clone()
        random_state = torch.random.get_rng_state().clone()
        control = matched_random_noise(perturbation, seed=38)
        torch.testing.assert_close(control, matched_random_noise(perturbation, seed=38))
        torch.testing.assert_close(
            control.abs().flatten().sort().values,
            perturbation.abs().flatten().sort().values,
            rtol=0,
            atol=0,
        )
        self.assertAlmostEqual(
            control.double().norm().item(), perturbation.double().norm().item()
        )
        self.assertEqual(control.abs().max().item(), perturbation.abs().max().item())
        torch.testing.assert_close(perturbation, before, rtol=0, atol=0)
        torch.testing.assert_close(
            torch.random.get_rng_state(), random_state, rtol=0, atol=0
        )
        self.assertFalse(torch.equal(control, perturbation))


class ArtifactIdentityTests(unittest.TestCase):
    def setUp(self):
        self.checkpoint = checkpoint_fixture()
        self.artifact = {
            "pretrained_ckpt": self.checkpoint,
            "rnd_idx_train": torch.tensor([2, 0, 1]),
            "latest_noise": torch.zeros(3, 3, 32, 32),
            "delta": 0.3,
            "seed": 0,
            "mode": "reckless",
            "attacked_task": 9,
        }

    def test_valid_artifact_and_exact_checkpoint_identity(self):
        validate_attack_artifact(self.artifact, expected_size=3)
        compare_checkpoint_identity(self.checkpoint, copy.deepcopy(self.checkpoint))

    def test_nonfinite_budget_and_noise_are_rejected(self):
        for field in ("delta", "latest_noise"):
            with self.subTest(field=field):
                invalid = copy.deepcopy(self.artifact)
                if field == "delta":
                    invalid[field] = float("nan")
                else:
                    invalid[field][0, 0, 0, 0] = float("nan")
                with self.assertRaises(ValueError):
                    validate_attack_artifact(invalid, expected_size=3)

    def test_fractional_permutation_is_rejected(self):
        self.artifact["rnd_idx_train"] = torch.tensor([2.5, 0, 1])
        with self.assertRaises(ValueError):
            validate_attack_artifact(self.artifact, expected_size=3)

    def test_checkpoint_metadata_and_tensor_mismatches_are_rejected(self):
        for field in ("seed", "model"):
            with self.subTest(field=field):
                invalid = copy.deepcopy(self.checkpoint)
                if field == "seed":
                    invalid[field] = 1
                else:
                    invalid[field]["conv1.weight"][0, 0] += 1
                with self.assertRaises(RuntimeError):
                    compare_checkpoint_identity(self.checkpoint, invalid)

    def test_inversion_order_uses_ids_and_rejects_duplicate_or_missing_ids(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            first = Path(directory) / "a_reference.npz"
            second = Path(directory) / "z_reference.npz"
            arrays = {"x_dst": np.zeros((1, 3, 32, 32)), "y_dst": np.array([0])}
            np.savez(first, **arrays, tid=1)
            np.savez(second, **arrays, tid=0)
            self.assertEqual(
                matching_inversion_files(directory, expected_count=2), [second, first]
            )

            np.savez(second, **arrays, tid=1)
            with self.assertRaises(ValueError):
                matching_inversion_files(directory, expected_count=2)

            np.savez(second, **arrays)
            with self.assertRaises(ValueError):
                matching_inversion_files(directory, expected_count=2)

    def test_inversion_filename_fallback_and_conflicting_metadata(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            path = Path(directory) / "generated_tid_00.npz"
            arrays = {"x_dst": np.zeros((1, 3, 32, 32)), "y_dst": np.array([0])}
            np.savez(path, **arrays)
            self.assertEqual(
                matching_inversion_files(directory, expected_count=1), [path]
            )
            np.savez(path, **arrays, tid=1)
            with self.assertRaises(ValueError):
                matching_inversion_files(directory, expected_count=1)


if __name__ == "__main__":
    unittest.main()
