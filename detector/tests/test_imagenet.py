"""Synthetic CPU regressions for the full-data extension, not experimental results."""

import argparse
import ast
import copy
import json
import os
import sys
import types
from pathlib import Path
import tempfile
import tarfile
import unittest
from unittest import mock

import numpy as np
import torch

from detector import advisor_detector, advisor_final, advisor_imagenet as workflow
from detector import advisor_study, image_data, transfer_core, transfer_detector
from detector.common import save_json
from detector.io_utils import DETECTOR_ROOT, load_frozen_bundle
from detector.tests.test_advisor import predicted, settings
from detector.tests.test_features import checkpoint_fixture
from detector.tests.test_transfer import write_table
from detector.transfer_extract import reusable_controls


def small_arrays(name):
    spec = image_data.dataset_spec(name)
    spec.update(train_per_class=2, test_per_class=1, image_size=4)
    rng = np.random.RandomState(321)
    arrays = {}
    for xkey, ykey, count in (("train_data", "train_lbls", 2), ("tst_data", "tst_lbls", 1)):
        labels = np.repeat(np.arange(spec["classes"]), count)
        shape = (len(labels), 4, 4, 3) if name == "split_mini_imagenet" else (len(labels), 3, 4, 4)
        arrays[xkey] = rng.randint(0, 256, size=shape).astype(np.uint8) if name == "split_mini_imagenet" else rng.rand(*shape).astype(np.float32)
        arrays[ykey] = labels
    return spec, arrays


class ImageDataTests(unittest.TestCase):
    def test_full_sizes_and_fixed_class_partitions(self):
        mini = image_data.dataset_spec("split_mini_imagenet")
        tiny = image_data.dataset_spec("split_tiny_imagenet")
        self.assertEqual((mini["classes"], mini["train_per_class"], mini["image_size"]), (100, 500, 84))
        self.assertEqual((tiny["classes"], tiny["classes_per_task"], tiny["image_size"]), (200, 20, 64))
        self.assertEqual(image_data.task_order("split_tiny_imagenet")[9], list(range(180, 200)))
        with self.assertRaises(ValueError):
            image_data.dataset_spec("unknown")

    def test_incomplete_labels_wrong_encoding_and_nan_are_rejected(self):
        for name in workflow.DATASETS.values():
            spec, arrays = small_arrays(name)
            with mock.patch.object(image_data, "dataset_spec", return_value=spec):
                self.assertEqual(len(image_data.validate_arrays(name, arrays)["train"]), spec["classes"])
                reduced = dict(arrays, train_data=arrays["train_data"][::2], train_lbls=arrays["train_lbls"][::2])
                with self.assertRaises(ValueError):
                    image_data.validate_arrays(name, reduced)
                bad = dict(arrays, train_lbls=np.zeros_like(arrays["train_lbls"]))
                with self.assertRaises(ValueError):
                    image_data.validate_arrays(name, bad)
                bad = dict(arrays, train_data=arrays["train_data"].astype(np.int16))
                with self.assertRaises(ValueError):
                    image_data.validate_arrays(name, bad)
                if name == "split_tiny_imagenet":
                    bad = copy.deepcopy(arrays)
                    bad["train_data"][0, 0, 0, 0] = np.nan
                    with self.assertRaises(ValueError):
                        image_data.validate_arrays(name, bad)

    def test_prepared_mini_matches_upstream_tensors_order_and_rng(self):
        # Execute the unchanged loader's AST in isolation: importing the whole
        # upstream module also imports torchvision's native image extensions,
        # which are unnecessary for this NumPy/tensor equivalence check.
        source = DETECTOR_ROOT.parent / "approaches/data_utils.py"
        node = next(n for n in ast.parse(source.read_text()).body
                    if isinstance(n, ast.FunctionDef) and n.name == "generate_split_mini_imagenet_tasks")
        namespace = dict(np=np, torch=torch, os=os)
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), "exec"), namespace)
        original_loader = namespace[node.name]
        spec, arrays = small_arrays("split_mini_imagenet")
        names = {"train_x.npy": "train_data", "test_x.npy": "tst_data", "train_y.npy": "train_lbls", "test_y.npy": "tst_lbls"}
        with mock.patch.object(np, "load", side_effect=lambda path: arrays[names[Path(path).name]].copy()):
            original = original_loader("unused", task_num=10, seed=6, rnd_order=False, order=np.arange(100))
        expected_rng = torch.rand(4)
        with mock.patch.object(image_data, "verify"), mock.patch.object(image_data, "dataset_spec", return_value=spec), \
             mock.patch.object(image_data, "read_arrays", return_value=arrays) as loader:
            prepared = image_data.PreparedData(dict(dataset="split_mini_imagenet", spec=spec, files={}))
            adapted = prepared.training(4, True, task_num=10, seed=6, rnd_order=False, order=np.arange(100))
            torch.testing.assert_close(torch.rand(4), expected_rng, rtol=0, atol=0)
            for task in range(10):
                for split in ("train", "test"):
                    torch.testing.assert_close(original[0][task][split]["x"], adapted[0][task][split]["x"], rtol=0, atol=0)
                    self.assertTrue(torch.equal(original[0][task][split]["y"], adapted[0][task][split]["y"]))
            self.assertEqual(adapted[1], [(t, 10) for t in range(5)])
            self.assertEqual(adapted[2], [3, 4, 4])  # Actual image size, not upstream's fixed 32.
            prepared.training(9, False, task_num=10, rnd_order=False, order=np.arange(100))
            self.assertEqual(loader.call_count, 1)

    def test_cached_tiny_specs_have_fresh_wrappers_and_repeat_rng(self):
        spec, arrays = small_arrays("split_tiny_imagenet")
        checkpoint = dict(checkpoint_fixture(), dataset="split_tiny_imagenet", task_num=4,
                          class_num=20, task_order=image_data.task_order("split_tiny_imagenet"), seed=7)
        with mock.patch.object(image_data, "verify"), mock.patch.object(image_data, "dataset_spec", return_value=spec), \
             mock.patch.object(image_data, "read_arrays", return_value=arrays) as loader:
            prepared = image_data.PreparedData(dict(dataset="split_tiny_imagenet", spec=spec, files={}))
            first = prepared.specs(**checkpoint)
            expected = torch.rand(3)
            original = first[0]["train"][4].data.clone()
            first[0]["train"][4].data = first[0]["train"][4].data.flip(0)
            first[0]["train"][4].targets = first[0]["train"][4].targets.flip(0)
            second = prepared.specs(**checkpoint)
            self.assertTrue(torch.equal(second[0]["train"][4].data, original))
            torch.testing.assert_close(torch.rand(3), expected, rtol=0, atol=0)
            self.assertEqual(len(second[0]["train"]), 5)
            self.assertEqual(second[3], 20)
            self.assertEqual(loader.call_count, 1)

    def test_tiny_defender_has_twenty_class_heads(self):
        from resnet import ResNet18
        model = ResNet18(10, 20)
        checkpoint = dict(checkpoint_fixture(), dataset="split_tiny_imagenet", class_num=20,
                          task_order=image_data.task_order("split_tiny_imagenet"), task_num=1,
                          model=model.state_dict())
        loaded = transfer_core.load_defender(checkpoint, torch.device("cpu"), 11)
        self.assertEqual(loaded.heads[-1].out_features, 20)
        self.assertTrue(torch.equal(loaded.conv1.weight, model.conv1.weight))
        with self.assertRaises(ValueError):
            transfer_core.validate_checkpoint(dict(checkpoint, class_num=10))

    def test_backbone_features_work_at_both_native_resolutions(self):
        from resnet import ResNet18
        from detector.extract_features import backbone_parameters, extract_one
        from detector.common import assert_model_unchanged, snapshot_state_dict
        for width, size in ((10, 84), (20, 64)):
            model = ResNet18(2, width).eval()
            named, parameters = backbone_parameters(model)
            before = snapshot_state_dict(model)
            direction = torch.zeros(sum(p.numel() for p in parameters))
            row = extract_one(model, named, parameters, direction, torch.rand(3, size, size),
                              object(), torch.device("cpu"), label_mode="predicted")
            self.assertEqual(len([c for c in row if c.startswith("grad_norm_param__")]), 80)
            self.assertTrue(np.isfinite(list(row.values())).all())
            assert_model_unchanged(model, before)


class ImageWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(dir=DETECTOR_ROOT / "tests")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def extension_settings(self, dataset="split_mini_imagenet"):
        value = settings()
        value.update(history_tasks=[4], methods=["ewc"], seeds=[6], unsupervised=False,
                     dataset=dataset, data_manifest=str(self.root / "data_manifest.json"),
                     reuse_controls=True, shared_effectiveness=True, threshold_rule="history_mad")
        return value

    def test_dedup_schedule_and_all_four_attacks(self):
        _, steps = advisor_study.build_cell(self.root, self.extension_settings(), "ewc", 6)
        self.assertEqual(len(steps), 30)
        self.assertEqual(len([s for s in steps if s["name"].endswith("_attack")]), 6)
        self.assertEqual(len([s for s in steps if "_train_" in s["name"]]), 7)
        owned = [p for s in steps for p in s["owned"]]
        self.assertEqual(len(owned), len(set(owned)))
        for step in steps:
            argv = step["command"]
            if "detector.transfer_upstream" in argv or "detector.transfer_extract" in argv:
                self.assertIn("--data-manifest", argv)
            if "--experiment" in argv:
                self.assertEqual(argv[argv.index("--experiment") + 1], "split_mini_imagenet")
        extracts = [s for s in steps if s["name"].startswith("task9_") and s["name"].endswith("_extract")]
        self.assertNotIn("--reuse-controls", extracts[0]["command"])
        self.assertIn("reckless_0p1/features/features.csv", extracts[1]["command"][extracts[1]["command"].index("--reuse-controls") + 1])
        self.assertIn("reckless_0p3/features/features.csv", extracts[3]["command"][extracts[3]["command"].index("--reuse-controls") + 1])
        freeze = next(i for i, s in enumerate(steps) if s["name"] == "freeze_supervised")
        self.assertLess(freeze, next(i for i, s in enumerate(steps) if s["name"].startswith("task9_")))

    def test_shared_effects_are_included_in_analysis_package(self):
        settings = self.extension_settings()
        cell = self.root / "ewc/seed6"
        for delta in (.1, .3):
            key = "uniform_" + advisor_study.attack_name("budget", delta)
            folder = cell / "task9/effectiveness_shared" / key
            folder.mkdir(parents=True)
            np.save(folder / "acc_mat_uniform.npy", np.eye(10))
        folder = cell / "task9/effectiveness_shared/clean"
        folder.mkdir()
        np.save(folder / "acc_mat_clean.npy", np.eye(10))
        files = advisor_final.analysis_files(self.root, settings)
        self.assertEqual(len([p for p in files if "effectiveness_shared" in str(p)]), 3)
        case = cell / "task9/cautious_0p3"
        (case / "effectiveness/poison").mkdir(parents=True)
        np.save(case / "effectiveness/poison/acc_mat_ours.npy", np.eye(10))
        result = advisor_study.effect_metrics(case, 9, shared=True)
        self.assertEqual(result["poison"]["past_drop_vs_clean"], 0)

    def test_reuse_controls_checks_identity_and_budget(self):
        table, meta = predicted(9)
        meta.update(feature_seed=1, split_seed=2, random_control_seed=3,
                    random_control="uniform", gradient_target="predicted_argmax")
        path = self.root / "features.csv"
        meta = write_table(table, meta, path)
        rows, record = reusable_controls(path, meta, .3)
        self.assertEqual(set(v for _, v in rows), {"clean", "random_control"})
        clean = table[(table.view == "clean")].iloc[0]
        self.assertEqual(rows[(int(clean.original_index), "clean")]["entropy"], clean.entropy)
        rows, record = reusable_controls(path, meta, .1)
        self.assertEqual(set(v for _, v in rows), {"clean"})
        with self.assertRaisesRegex(ValueError, "provenance"):
            reusable_controls(path, dict(meta, checkpoint_sha256="f" * 64), .3)

    def test_tiny_fit_calibrate_and_target_evaluate(self):
        paths = []
        for task in (1, 4, 9):
            table, meta = predicted(task)
            table = table.drop(columns="true_class_probability")
            meta["feature_columns"].remove("true_class_probability")
            meta.update(dataset="split_tiny_imagenet", dataset_sha256="a" * 64,
                        inference_schema="predicted_without_true_class_v1",
                        task_order=image_data.task_order("split_tiny_imagenet"),
                        class_ids=image_data.task_order("split_tiny_imagenet")[task], checkpoint_seed=6)
            table["original_uid"] = table.original_index.map(lambda i: "split_tiny_imagenet:task{}:row{}".format(task, i))
            table.loc[table.original_index == table.original_index.min(), "class_id"] = 19
            path = self.root / ("task{}.csv".format(task))
            write_table(table, meta, path)
            paths.append(path)
        advisor_detector.develop(paths[0], [paths[1]], self.root / "frozen", task_size=16,
                                 repeats=2, ablations=True, threshold_rule="history_mad")
        advisor_study.evaluate(argparse.Namespace(features=str(paths[2]), frozen=str(self.root / "frozen"),
                              output_dir=str(self.root / "evaluation"), bags_per_rate=2, reference=None))
        bundle = load_frozen_bundle(self.root / "frozen/ablations/inference_full/bundle.joblib")
        self.assertNotIn("true_class_probability", bundle["feature_columns"])
        from detector.io_utils import read_feature_table
        source, _ = read_feature_table(paths[0])
        validation = source[(source.split == "validation") & source.view.isin(["clean", "poison"])]
        recomputed = transfer_detector.binary_metrics(validation.detector_label,
            advisor_detector.score_images(validation, bundle), bundle["threshold"])
        self.assertEqual(recomputed, bundle["validation"])
        target, meta = read_feature_table(paths[2])
        with self.assertRaisesRegex(ValueError, "dataset identity"):
            transfer_core.assert_transfer(bundle["source_metadata"], dict(meta, dataset_sha256="b" * 64), bundle["feature_columns"])

    def test_atomic_json_failure_keeps_previous_file(self):
        path = self.root / "state.json"
        save_json({"status": "complete"}, path)
        before = path.read_bytes()
        with mock.patch("detector.common.os.replace", side_effect=OSError("simulated failure")):
            with self.assertRaises(OSError):
                save_json({"status": "running"}, path)
        self.assertEqual(path.read_bytes(), before)
        self.assertEqual(sorted(p.name for p in self.root.iterdir()), ["state.json"])

    def test_two_plans_dry_run_status_and_incomplete_packages(self):
        output = self.root / "extension"
        def record(dataset, source):
            return dict(dataset=dataset, dataset_sha256="a" * 64, files={}, hashes={},
                        spec=image_data.dataset_spec(dataset), counts={}, full_training_data=True)
        args = argparse.Namespace(output_dir=str(output), seeds=[6], mini_data="unused-mini", tiny_data="unused-tiny")
        with mock.patch.object(workflow, "register", side_effect=record):
            workflow.prepare(args)
            workflow.prepare(args)
        for short in workflow.DATASETS:
            root, payload = advisor_study.read_plan(output / short / "plan.json")
            self.assertEqual(payload["settings"]["seeds"], [6])
            self.assertFalse(payload["settings"]["unsupervised"])
            profile = workflow.timing(root, payload)
            self.assertEqual(profile["cells"][0]["total"], 30)
            self.assertEqual(profile["cells"][0]["complete"], 0)
        args.allow_incomplete = True
        workflow.package(args)
        archives = list(output.glob("*_analysis_*.tar.gz"))
        self.assertEqual(len(archives), 2)
        for archive in archives:
            with tarfile.open(archive) as bundle:
                manifests = [n for n in bundle.getnames() if n.endswith("/package_manifest.json")]
                manifest = json.load(bundle.extractfile(manifests[0]))
                self.assertEqual(manifest["status"], "incomplete")
                self.assertIn("analysis/timing.json", manifest["files"])
                for name, expected in manifest["files"].items():
                    import hashlib
                    member = manifests[0].split("/")[0] + "/" + name
                    self.assertEqual(hashlib.sha256(bundle.extractfile(member).read()).hexdigest(), expected)
        manifest = output / "mini/data_manifest.json"
        save_json({"changed": True}, manifest)
        with self.assertRaisesRegex(ValueError, "manifest changed"):
            advisor_study.read_plan(output / "mini/plan.json")

    def test_upstream_adapter_passes_declared_seed_instead_of_zero(self):
        from detector import transfer_upstream
        manifest = self.root / "data_manifest.json"
        save_json({}, manifest)
        prepared = mock.Mock()
        prepared.record = {"dataset": "split_mini_imagenet"}
        fake = types.ModuleType("approaches.data_utils")
        arguments = ["transfer_upstream", "--incoming-task", "1", "--data-manifest", str(manifest),
                     "main_baselines.py", "--", "--tasknum", "10", "--lasttask", "1",
                     "--experiment", "split_mini_imagenet", "--seed", "6"]
        previous = Path.cwd()
        try:
            os.chdir(self.root)
            with mock.patch.object(sys, "argv", arguments), \
                 mock.patch.dict(sys.modules, {"approaches.data_utils": fake}), \
                 mock.patch.object(image_data, "PreparedData", return_value=prepared), \
                 mock.patch.object(transfer_upstream.runpy, "run_path", side_effect=lambda *a, **kw:
                     fake.generate_split_mini_imagenet_tasks("unused", task_num=10, rnd_order=False, order=np.arange(100))):
                transfer_upstream.main()
        finally:
            os.chdir(previous)
        self.assertEqual(prepared.training.call_args.kwargs["seed"], 6)

    def test_uniform_and_clean_inputs_do_not_depend_on_attack_mode(self):
        # Run the actual upstream input-perturbation branch, not a rewritten
        # oracle, with two artifacts differing in poison/permutation/mode.
        source = DETECTOR_ROOT.parent / "main_baselines.py"
        tree = ast.parse(source.read_text())
        node = next(n for n in ast.walk(tree) if isinstance(n, ast.If) and
                    isinstance(n.test, ast.BoolOp) and "addnoise" in ast.dump(n.test)
                    and "checkpoint" in ast.dump(n.test))
        code = compile(ast.Module(body=[node], type_ignores=[]), str(source), "exec")
        x = torch.rand(6, 3, 4, 4)
        y = torch.arange(6)
        for uniform, addnoise in ((True, True), (False, False)):
            results = []
            for mode in ("reckless", "cautious"):
                artifact = dict(mode=mode, delta=.3, latest_noise=torch.randn_like(x),
                                rnd_idx_train=torch.randperm(6))
                namespace = dict(torch=torch, checkpoint_dict=artifact, xtrain=x.clone(), ytrain=y.clone(),
                                 args=argparse.Namespace(checkpoint="noise.pkl", addnoise=addnoise,
                                                         uniform=uniform, inj_rate=1.0))
                torch.manual_seed(7)
                exec(code, namespace)
                results.append((namespace["xtrain"], namespace["ytrain"], torch.random.get_rng_state()))
            self.assertTrue(all(torch.equal(a, b) for a, b in zip(*results)))


if __name__ == "__main__":
    unittest.main()
