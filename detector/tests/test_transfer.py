"""CPU contract/math/integration checks; never real GPU evidence."""

import argparse
import copy
import itertools
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest import mock

import numpy as np
import pandas as pd
import torch

from detector import FEATURE_COLUMNS
from detector import explain_detector as explain
from detector import transfer_core as core
from detector import transfer_detector as detection
from detector import transfer_pipeline as workflow
from detector.common import save_json, sha256_file
from detector.io_utils import DETECTOR_ROOT
from detector.tests.test_dataset import dataset_fixture
from detector.tests.test_features import checkpoint_fixture


def feature_fixture(task=1):
    table, metadata, _ = dataset_fixture()
    table["task_index"] = task
    table["original_uid"] = table.original_index.map(
        lambda i: "cifar100:task{}:row{}".format(task, i)
    )
    metadata.update(
        transfer_protocol=core.TRANSFER_PROTOCOL,
        task_index=task,
        task_order=core.TASK_ORDER,
        class_ids=core.TASK_ORDER[task],
        checkpoint_sha256=str(task) * 64,
        model_unchanged=True,
        features_sha256="c" * 64,
        input_sha256={"attack": str(task) * 64},
        attack_seed=0,
        synthetic=True,
    )
    return table, metadata


def write_table(table, metadata, path):
    table.to_csv(path, index=False)
    metadata = dict(metadata, features_sha256=sha256_file(path), row_count=len(table))
    save_json(metadata, path.with_suffix(".metadata.json"))
    return metadata


class TransferContracts(unittest.TestCase):
    def test_fixed_partition_and_prefix_no_class_repartition(self):
        checkpoint = checkpoint_fixture()
        checkpoint["task_num"] = 1
        original = copy.deepcopy(checkpoint)
        fake = types.ModuleType("data_utils")
        fake.generate_split_cifar100_tasks = mock.Mock(
            return_value=(
                {"train": list(range(10)), "test": list(range(10))},
                core.TASK_ORDER,
            )
        )
        with mock.patch.dict(sys.modules, {"data_utils": fake}):
            data, order, size, classes, _ = core.fixed_dataset_specs(**checkpoint)
        self.assertEqual(
            fake.generate_split_cifar100_tasks.call_args.kwargs["task_num"], 10
        )
        self.assertEqual(data["train"], [0, 1])
        self.assertEqual(order, core.TASK_ORDER)
        self.assertEqual((size, classes), (32, 10))
        self.assertEqual(checkpoint["task_num"], original["task_num"])

    def test_effectiveness_trains_only_incoming_with_ten_head_dimensions(self):
        data = {"ncla": 100}
        result = core.truncate_training_tasks(
            (data, [(i, 10) for i in range(10)], [3, 32, 32], core.TASK_ORDER), 1
        )
        self.assertEqual(result[1], [(0, 10), (1, 10)])
        self.assertEqual(result[0]["ncla"], 100)

    def test_source_defender_handles_unused_future_heads_and_fishers(self):
        from resnet import ResNet18

        model = ResNet18(10, 10)
        state = dict(model.state_dict())
        state.update(
            {
                n.replace(".", "_") + "_fisher": torch.zeros_like(p)
                for n, p in model.named_parameters()
            }
        )
        checkpoint = dict(checkpoint_fixture(), task_num=1, model=state)
        loaded = core.load_defender(checkpoint, torch.device("cpu"), 12)
        self.assertEqual(len(loaded.heads), 2)
        self.assertTrue(torch.equal(loaded.conv1.weight, model.conv1.weight))
        self.assertTrue(torch.equal(loaded.heads[0].weight, model.heads[0].weight))
        repeated = core.load_defender(checkpoint, torch.device("cpu"), 12)
        self.assertTrue(torch.equal(loaded.heads[1].weight, repeated.heads[1].weight))
        del checkpoint["model"]["conv1.weight"]
        with self.assertRaisesRegex(ValueError, "missing"):
            core.load_defender(checkpoint, torch.device("cpu"), 12)

    def test_checkpoint_rejects_task_repartition_and_bad_stage(self):
        checkpoint = checkpoint_fixture()
        for task in (0, 10, 1.5, True):
            with self.assertRaises(ValueError):
                core.validate_checkpoint(dict(checkpoint, task_num=task))
        with self.assertRaisesRegex(ValueError, "class order"):
            core.validate_checkpoint(
                dict(
                    checkpoint,
                    task_num=1,
                    task_order=[list(range(50)), list(range(50, 100))],
                )
            )

    def test_artifact_must_belong_to_exact_stage_and_checkpoint(self):
        checkpoint = dict(checkpoint_fixture(), task_num=1)
        artifact = {
            "pretrained_ckpt": copy.deepcopy(checkpoint),
            "rnd_idx_train": torch.arange(2),
            "latest_noise": torch.zeros(2, 3, 32, 32),
            "delta": 0.3,
            "seed": 0,
            "mode": "reckless",
            "attacked_task": 1,
        }
        core.validate_artifact(artifact, checkpoint, expected_size=2)
        artifact["pretrained_ckpt"]["model"]["conv1.weight"] += 1
        with self.assertRaisesRegex(RuntimeError, "differ"):
            core.validate_artifact(artifact, checkpoint, expected_size=2)
        artifact["attacked_task"] = 9
        with self.assertRaisesRegex(ValueError, "different"):
            core.validate_artifact(artifact, checkpoint, expected_size=2)


class TransferStatistics(unittest.TestCase):
    def setUp(self):
        self.table, self.meta = feature_fixture()
        self.bundle = detection.fit_source(self.table, self.meta, task_size=16)

    def test_source_test_and_random_controls_cannot_change_fit(self):
        changed = self.table.copy()
        mask = (changed.split == "test") | (changed.view == "random_control")
        changed.loc[mask, FEATURE_COLUMNS] *= 100
        other = detection.fit_source(changed, self.meta, task_size=16)
        np.testing.assert_array_equal(
            self.bundle["classifier"].coef_, other["classifier"].coef_
        )
        np.testing.assert_array_equal(
            self.bundle["scaler"].mean_, other["scaler"].mean_
        )
        self.assertEqual(self.bundle["threshold"], other["threshold"])
        self.assertEqual(self.bundle["count_calibration"], other["count_calibration"])

    def test_target_preprocessing_frozen_and_non_test_rows_not_used(self):
        target, meta = feature_fixture(9)
        before = copy.deepcopy(self.bundle)
        report, predictions, bags = detection.evaluate(
            target, meta, self.bundle, repeats=3
        )
        altered = target.copy()
        altered.loc[altered.split != "test", FEATURE_COLUMNS] *= 10000
        report2, predictions2, bags2 = detection.evaluate(
            altered, meta, self.bundle, repeats=3
        )
        pd.testing.assert_frame_equal(predictions, predictions2)
        pd.testing.assert_frame_equal(bags, bags2)
        self.assertEqual(report, report2)
        np.testing.assert_array_equal(
            before["scaler"].mean_, self.bundle["scaler"].mean_
        )
        self.assertEqual(before["count_calibration"], self.bundle["count_calibration"])

    def test_rejects_same_task_label_modes_and_mutated_schema(self):
        with self.assertRaisesRegex(ValueError, "later"):
            detection.evaluate(self.table, self.meta, self.bundle, repeats=1)
        target, meta = feature_fixture(9)
        for key, value in (("head_seed", 99), ("label_mode", "ground_truth")):
            altered = dict(meta, **{key: value})
            changed = target.copy()
            changed[key] = value
            with self.assertRaisesRegex(ValueError, "mismatch"):
                detection.evaluate(changed, altered, self.bundle, repeats=1)
        with self.assertRaisesRegex(ValueError, "UID"):
            detection.evaluate(
                target.assign(original_uid="wrong"), meta, self.bundle, repeats=1
            )

    def test_source_control_requires_original_identity(self):
        report, _, _ = detection.evaluate(
            self.table, self.meta, self.bundle, source_control=True, repeats=2
        )
        self.assertTrue(report["source_control"])
        with self.assertRaisesRegex(ValueError, "original frozen"):
            detection.evaluate(
                self.table,
                dict(self.meta, features_sha256="e" * 64),
                self.bundle,
                source_control=True,
                repeats=2,
            )

    def test_linear_shap_matches_enumerated_coalitions_and_additivity(self):
        train = self.table.loc[
            (self.table.split == "train") & self.table.view.isin(["clean", "poison"])
        ]
        val = self.table.loc[self.table.split == "validation"].head(2)
        phi, base, logits, error = explain.linear_shap(self.bundle, train, val)
        self.assertLess(error, 1e-10)
        np.testing.assert_allclose(phi.sum(axis=1) + base, logits)
        # Independent permutation oracle for three dimensions, keeping the rest
        # at the background mean. The linear game's contributions do not depend
        # on the order or on the values of the other coordinates.
        cols = self.bundle["feature_columns"]
        scaler, clf = self.bundle["scaler"], self.bundle["classifier"]
        background = scaler.transform(train[cols].to_numpy()).mean(axis=0)
        x = scaler.transform(val[cols].to_numpy())[0]
        contributions = np.zeros(3)
        for perm in itertools.permutations(range(3)):
            current = background.copy()
            for j in perm:
                left = clf.decision_function(current[None, :])[0]
                current[j] = x[j]
                right = clf.decision_function(current[None, :])[0]
                contributions[j] += (right - left) / 6
        np.testing.assert_allclose(contributions, phi[0, :3], atol=1e-12)

    def test_feature_families_are_not_mistaken_for_parameters(self):
        self.assertEqual(
            explain.feature_family("grad_norm_param__layer2.0.conv1.weight"), "layer2"
        )
        self.assertEqual(explain.feature_family("grad_norm_stage_stem"), "stem")


class TransferWorkflow(unittest.TestCase):
    def test_completed_steps_resume_but_changed_outputs_are_rejected(self):
        with tempfile.TemporaryDirectory(dir=DETECTOR_ROOT / "tests") as tmp:
            root = Path(tmp)
            plan_path = root / "protocol.json"
            save_json(core.fingerprint({}), plan_path)
            artifact = root / "artifact.json"
            step = {
                "name": "fake",
                "command": ["fake"],
                "cwd": str(root),
                "outputs": [str(artifact)],
            }

            def execute(*args, **kwargs):
                save_json({"synthetic": True}, artifact)

            with mock.patch.object(workflow, "build_steps", return_value=[step]):
                with mock.patch.object(workflow, "summarize"):
                    with mock.patch.object(
                        workflow, "execute", side_effect=execute
                    ) as run:
                        args = argparse.Namespace(plan=str(plan_path), dry_run=False)
                        workflow.run(args)
                        workflow.run(args)
                        self.assertEqual(run.call_count, 1)
                        save_json({"synthetic": False}, artifact)
                        with self.assertRaisesRegex(ValueError, "outputs changed"):
                            workflow.run(args)

    def test_failed_steps_are_not_silently_restarted(self):
        with tempfile.TemporaryDirectory(dir=DETECTOR_ROOT / "tests") as tmp:
            root = Path(tmp)
            save_json(core.fingerprint({}), root / "protocol.json")
            save_json({"fake": {"status": "failed"}}, root / "run_state.json")
            step = {
                "name": "fake",
                "command": ["fake"],
                "cwd": str(root),
                "outputs": [str(root / "output.json")],
            }
            with mock.patch.object(workflow, "build_steps", return_value=[step]):
                with mock.patch.object(workflow, "execute") as execute:
                    with self.assertRaisesRegex(RuntimeError, "Incomplete step"):
                        workflow.run(
                            argparse.Namespace(
                                plan=str(root / "protocol.json"), dry_run=False
                            )
                        )
                    execute.assert_not_called()

    def test_plan_cli_parser_order_and_no_gpu_execution(self):
        with tempfile.TemporaryDirectory(dir=DETECTOR_ROOT / "tests") as tmp:
            root = Path(tmp) / "plan"
            args = workflow.build_parser().parse_args(
                ["plan", "--output-dir", str(root)]
            )
            with mock.patch.object(workflow, "execute") as execute:
                workflow.plan(args)
                execute.assert_not_called()
            payload = json.loads((root / "protocol.json").read_text())
            self.assertEqual(payload, core.fingerprint(payload["settings"]))
            steps = workflow.build_steps(root, payload["settings"])
            source_fits = [
                i
                for i, s in enumerate(steps)
                if s["name"].startswith("seed3_task1_fit")
            ]
            target_begin = next(
                i for i, s in enumerate(steps) if s["name"] == "seed3_task9_victim"
            )
            self.assertTrue(all(i < target_begin for i in source_fits))
            for step in steps:
                cwd = Path(step["cwd"])
                self.assertTrue(cwd == root or root in cwd.parents)
                if "command" not in step:
                    continue
                argv = step["command"]
                if argv[4] == "detector.transfer_upstream":
                    self.assertEqual(argv[5], "--incoming-task")
                    if "--tasknum" in argv:
                        self.assertEqual(argv[argv.index("--tasknum") + 1], "10")
                    if step["name"].endswith("_inversion") and "task1_" in step["name"]:
                        self.assertEqual(argv[argv.index("--task_lst") + 1], "0")
            subprocess.run(["bash", "-n", str(root / "commands.sh")], check=True)

    def test_wrong_registered_code_rejected_before_execution(self):
        with tempfile.TemporaryDirectory(dir=DETECTOR_ROOT / "tests") as tmp:
            plan = Path(tmp) / "protocol.json"
            save_json({"settings": {}, "fingerprint": "wrong"}, plan)
            with mock.patch.object(workflow, "execute") as execute:
                with self.assertRaisesRegex(ValueError, "Code/settings"):
                    workflow.run(argparse.Namespace(plan=str(plan), dry_run=False))
            execute.assert_not_called()

    def test_effectiveness_uses_current_row_not_untrained_future_tasks(self):
        with tempfile.TemporaryDirectory(dir=DETECTOR_ROOT / "tests") as tmp:
            root = Path(tmp)
            for label, filename, values in (
                ("clean", "acc_mat_clean.npy", [[0.8, 0], [0.7, 0.9]]),
                ("poison", "acc_mat_ours.npy", [[0.8, 0], [0.4, 0.5]]),
            ):
                path = root / "effectiveness" / label
                path.mkdir(parents=True)
                np.save(path / filename, np.array(values))
            result = workflow.effectiveness(root, 1)
            self.assertAlmostEqual(result["past_accuracy_drop_clean_minus_poison"], 0.3)
            self.assertAlmostEqual(result["clean"]["backward_transfer"], -0.1)

    def test_real_cli_fit_evaluate_explain_synthetic_tables(self):
        with tempfile.TemporaryDirectory(dir=DETECTOR_ROOT / "tests") as tmp:
            root = Path(tmp)
            for task, name in ((1, "source"), (9, "target")):
                table, meta = feature_fixture(task)
                write_table(table, meta, root / (name + ".csv"))
            env = dict(
                os.environ,
                PYTHONDONTWRITEBYTECODE="1",
                OPENBLAS_NUM_THREADS="1",
                OMP_NUM_THREADS="1",
                MPLCONFIGDIR=str(root / "mpl"),
                XDG_CACHE_HOME=str(root / "cache"),
            )
            commands = [
                workflow.command(
                    "transfer_detector",
                    "fit",
                    "--features",
                    root / "source.csv",
                    "--task-size",
                    16,
                    "--output-dir",
                    root / "model",
                ),
                workflow.command(
                    "transfer_detector",
                    "evaluate",
                    "--features",
                    root / "target.csv",
                    "--bundle",
                    root / "model/bundle.joblib",
                    "--bags-per-rate",
                    2,
                    "--output-dir",
                    root / "eval",
                ),
                workflow.command(
                    "explain_detector",
                    "--features",
                    root / "source.csv",
                    "--bundle",
                    root / "model/bundle.joblib",
                    "--output-dir",
                    root / "explain",
                ),
            ]
            for argv in commands:
                completed = subprocess.run(
                    argv,
                    cwd=DETECTOR_ROOT.parent,
                    env=env,
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(
                    completed.returncode, 0, completed.stdout + completed.stderr
                )
            result = json.loads((root / "eval/evaluation_metrics.json").read_text())
            self.assertTrue(result["synthetic"])
            self.assertFalse(result["target_used_for_fitting_or_calibration"])
            self.assertTrue((root / "explain/feature_importance.png").is_file())
            self.assertTrue((root / "eval/detection_curves.png").is_file())
            self.assertFalse(any(p.is_symlink() for p in root.rglob("*")))
            for example in json.loads((root / "explain/examples.json").read_text()):
                self.assertIn(example["view"], ("clean", "poison", "random_control"))
            self.assertLess(
                json.loads((root / "explain/explanation.json").read_text())[
                    "max_additivity_error"
                ],
                1e-10,
            )


if __name__ == "__main__":
    unittest.main()
