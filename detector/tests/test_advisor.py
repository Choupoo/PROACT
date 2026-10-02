"""Inference-label isolation, registered grid, threshold freeze and ablations."""

import argparse
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch

from detector import advisor_detector as advisor
from detector import advisor_study as study
from detector import transfer_detector as base
from detector.io_utils import DETECTOR_ROOT, load_frozen_bundle
from detector.tests.test_granularity import fixture
from detector.tests.test_transfer import write_table
from detector.tests.test_features import make_model, sample_images, checkpoint_fixture
from detector.extract_features import backbone_parameters, extract_one
from detector.transfer_core import load_defender, fingerprint


def predicted(task=1):
    table, meta = fixture(task)
    table["label_mode"] = "predicted"
    meta.update(label_mode="predicted", cl_method="ewc",
                attack_config={"mode": "reckless", "delta": .3})
    return table, meta


def settings():
    return dict(source_task=1, history_tasks=[4, 6], target_task=9, seeds=[5],
                methods=list(study.METHODS), method_parameters=study.METHODS,
                victim_epochs=1, inversion_iters=1, attack_epochs=100, delta=.3,
                task_size=16, bags_per_rate=2, alpha=.05, cautious_weight=1.,
                attacks=study.ATTACKS, ablations=True, unsupervised=True, effectiveness=True)


class AdvisorTests(unittest.TestCase):
    def test_gradient_extraction_never_reads_class_target(self):
        model = make_model()
        named, parameters = backbone_parameters(model)
        direction = torch.zeros(sum(p.numel() for p in parameters))
        image = sample_images(1, 12)[0]
        first = extract_one(model, named, parameters, direction, image, None,
                            torch.device("cpu"), label_mode="predicted")
        second = extract_one(model, named, parameters, direction, image, object(),
                             torch.device("cpu"), label_mode="predicted")
        self.assertEqual(first, second)

    def test_new_schema_has_no_true_class_column(self):
        table, meta = predicted()
        table = table.drop(columns=["true_class_probability"])
        meta["inference_schema"] = "predicted_without_true_class_v1"
        meta["feature_columns"].remove("true_class_probability")
        base.validate_table(table, meta)
        model = base.fit_source(table, meta, feature_set="inference_full", task_size=16)
        self.assertNotIn("true_class_probability", model["feature_columns"])

    def test_all_cl_buffer_schemas_load_without_changing_backbone(self):
        from resnet import ResNet18
        original = ResNet18(10, 10, nf=32)
        checkpoint = checkpoint_fixture()
        checkpoint["task_num"] = 1
        for prefix in ("omega_", "fisher_", "running_fisher_", "s_", "running_s_"):
            checkpoint["model"] = dict(original.state_dict())
            checkpoint["model"][prefix + "conv1_weight"] = torch.ones_like(original.conv1.weight)
            restored = load_defender(checkpoint, torch.device("cpu"), 13)
            self.assertTrue(torch.equal(restored.conv1.weight, original.conv1.weight))
        checkpoint["model"]["omega_conv1_weight"] = torch.zeros(1)
        with self.assertRaises(ValueError):
            load_defender(checkpoint, torch.device("cpu"), 13)

    def test_all_ablations_exclude_label_dependent_features(self):
        table, meta = predicted()
        for group in base.INFERENCE_GROUPS:
            cols = base.resolve_feature_columns(table, meta, group)
            self.assertFalse(set(cols) & {"true_class_probability", "loss", "grad_cosine_past"})
            self.assertTrue(all(c.startswith("grad_norm_") or c in base.INFERENCE_CONTEXT for c in cols))
        meta["label_mode"] = "ground_truth"
        with self.assertRaises(ValueError):
            base.fit_source(table, meta, feature_set="inference_full", task_size=16)

    def test_true_class_column_has_no_effect_and_predict_needs_only_descriptors(self):
        table, meta = predicted()
        fitted = base.fit_source(table, meta, feature_set="inference_full", task_size=16)
        changed = table.copy()
        changed["true_class_probability"] = 0.123
        second = base.fit_source(changed, meta, feature_set="inference_full", task_size=16)
        np.testing.assert_array_equal(fitted["classifier"].coef_, second["classifier"].coef_)
        expected = advisor.score_images(table, fitted)
        np.testing.assert_array_equal(expected, advisor.score_images(table[fitted["feature_columns"]], fitted))
        result = advisor.predict_dataset(table[fitted["feature_columns"]].iloc[:16], fitted)
        self.assertIn("alert", result)
        self.assertNotIn("probability", result)

    def test_calibration_does_not_read_test_or_poison_features(self):
        source, history = predicted(1), predicted(4)
        fitted = base.fit_source(*source, feature_set="inference_full", task_size=16)
        first = advisor.calibrate(fitted, [source, history], "history_quantile")
        changed = history[0].copy()
        mask = (changed.split == "test") | (changed.view != "clean")
        changed.loc[mask, fitted["feature_columns"]] *= 100
        second = advisor.calibrate(fitted, [source, (changed, history[1])], "history_quantile")
        self.assertEqual(first["threshold"], second["threshold"])
        self.assertEqual(first["count_calibration"], second["count_calibration"])

    def test_selection_reports_infeasibility(self):
        report = {"sample_metrics": {"clean_fpr": .8}, "dataset_results": [
            {"realized_rate": 0., "alert_rate": .9, "alternative": "poison", "requested_rate": 0.},
            {"realized_rate": .1, "alert_rate": .95, "alternative": "poison", "requested_rate": .1}]}
        selected = advisor.select_rule({r: report for r in advisor.RULES}, .05)
        self.assertFalse(selected["development_criterion_met"])
        self.assertIsNotNone(selected["warning"])

    def test_grid_freezes_before_target_and_covers_all_modes(self):
        root = DETECTOR_ROOT / "work/test_advisor_dry"
        for method in study.METHODS:
            _, steps = study.build_cell(root, settings(), method, 5)
            names = [s["name"] for s in steps]
            self.assertLess(names.index("freeze_supervised"), names.index("task9_victim"))
            self.assertEqual(len(names), len(set(names)))
            self.assertEqual(sum(n.endswith("_attack") for n in names), 7)
            self.assertEqual(sum(n.endswith("_train_uniform") for n in names), 4)
            for s in steps:
                if s["name"].endswith("_victim") or "_train_" in s["name"]:
                    argv = s["command"]
                    self.assertEqual(argv[argv.index("--approach") + 1], method)
                if s["name"].endswith("_extract"):
                    self.assertIn("predicted", s["command"])
            owners = [p for s in steps for p in s["owned"]]
            self.assertEqual(len(owners), len(set(owners)))

    def test_pilot_skips_deferred_work(self):
        options = settings()
        options.update(history_tasks=[4], attacks=[("reckless", .3)], ablations=False,
                       unsupervised=False, effectiveness=False)
        _, steps = study.build_cell(DETECTOR_ROOT / "work/test_advisor_pilot", options, "ewc", 5)
        names = [s["name"] for s in steps]
        self.assertNotIn("freeze_label_free", names)
        self.assertFalse(any("_train_" in n for n in names))
        self.assertEqual(sum(n.endswith("_attack") for n in names), 3)
        self.assertNotIn("--ablations", next(s["command"] for s in steps if s["name"] == "freeze_supervised"))

    def test_direct_extraction_scopes_dataset_download_and_restores_cwd(self):
        import os
        from detector import transfer_extract
        previous = Path.cwd()
        with tempfile.TemporaryDirectory(dir=DETECTOR_ROOT / "tests") as directory:
            root = Path(directory)
            try:
                os.chdir(DETECTOR_ROOT.parent)
                with mock.patch.object(transfer_extract, "fixed_dataset_specs", side_effect=lambda **kwargs: Path.cwd()):
                    actual = transfer_extract.scoped_dataset({}, root / "features")
                    self.assertEqual(actual, root / "dataset_cache")
                    self.assertEqual(Path.cwd(), DETECTOR_ROOT.parent)
            finally:
                os.chdir(previous)

    def test_development_end_to_end_and_no_target_calibration(self):
        with tempfile.TemporaryDirectory(dir=DETECTOR_ROOT / "tests") as directory:
            root = Path(directory)
            paths = []
            for task in (1, 4, 9):
                path = root / ("task{}.csv".format(task))
                write_table(*predicted(task), path)
                paths.append(path)
            advisor.develop(paths[0], paths[1:2], root / "frozen", task_size=16, repeats=2, ablations=True)
            freeze = json.loads((root / "frozen/freeze.json").read_text())
            self.assertEqual(freeze["calibration_tasks"], [1, 4])
            for group in base.INFERENCE_GROUPS:
                bundle = load_frozen_bundle(root / "frozen/ablations" / group / "bundle.joblib")
                report, _, _ = base.evaluate(*predicted(9), bundle, repeats=2)
                self.assertFalse(report["target_used_for_fitting_or_calibration"])
                self.assertEqual(report["sample_threshold"], bundle["threshold"])
            args = argparse.Namespace(output_dir=str(root / "rejected"), frozen=str(root / "frozen"), features=str(paths[1]))
            with self.assertRaisesRegex(ValueError, "later than every"):
                study.evaluate(args)

    def test_interrupted_stage_is_archived_and_completed_steps_are_verified(self):
        with tempfile.TemporaryDirectory(dir=DETECTOR_ROOT / "tests") as directory:
            cell = Path(directory)
            owned = cell / "features"
            file = owned / "features.csv"
            step = dict(name="extract", command=["fixture"], cwd=str(cell), owned=[str(owned)], outputs=[str(file)])
            payload = fingerprint(settings())
            def fail(*args, **kwargs):
                owned.mkdir()
                file.write_text("partial")
                raise RuntimeError("fixture failure")
            with mock.patch.object(study, "execute", side_effect=fail):
                with self.assertRaisesRegex(RuntimeError, "fixture failure"):
                    study.execute_cell(cell, [step], payload, False)
            with self.assertRaisesRegex(RuntimeError, "Partial stage"):
                study.execute_cell(cell, [step], payload, False)
            def succeed(*args, **kwargs):
                owned.mkdir()
                file.write_text("complete")
            with mock.patch.object(study, "execute", side_effect=succeed) as executor:
                study.execute_cell(cell, [step], payload, True)
                study.execute_cell(cell, [step], payload, False)
                self.assertEqual(executor.call_count, 1)
            self.assertEqual(next((cell / "interrupted").glob("*/features/features.csv")).read_text(), "partial")
            file.write_text("tampered")
            with self.assertRaisesRegex(ValueError, "artifacts changed"):
                study.execute_cell(cell, [step], payload, False)

    def test_compact_schema_pilot_evaluation_and_summary(self):
        with tempfile.TemporaryDirectory(dir=DETECTOR_ROOT / "tests") as directory:
            root = Path(directory)
            cell = root / "ewc/seed3"
            cell.mkdir(parents=True)
            paths = []
            for task in (1, 4, 9):
                table, meta = predicted(task)
                table = table.drop(columns=["true_class_probability"])
                meta["feature_columns"].remove("true_class_probability")
                meta["inference_schema"] = "predicted_without_true_class_v1"
                path = root / ("task{}.csv".format(task))
                write_table(table, meta, path)
                paths.append(path)
            frozen = cell / "frozen_supervised"
            advisor.develop(paths[0], paths[1:2], frozen, task_size=16, repeats=2)
            output = cell / "task9/reckless_0p3/evaluation"
            args = argparse.Namespace(output_dir=str(output), frozen=str(frozen), features=str(paths[2]),
                                      reference=None, bags_per_rate=2)
            study.evaluate(args)
            self.assertFalse((output / "unsupervised.json").exists())
            options = settings()
            options.update(methods=["ewc"], seeds=[3], history_tasks=[4], attacks=[("reckless", .3)],
                           ablations=False, unsupervised=False, effectiveness=False)
            with mock.patch.object(study, "read_plan", return_value=(root, fingerprint(options))), \
                 mock.patch.object(study, "build_cell", return_value=(cell, [])):
                study.summarize(argparse.Namespace(plan="fixture"))
            summary = json.loads((root / "analysis/summary.json").read_text())
            self.assertEqual(len(summary["supervised"]), 4)
            self.assertEqual(summary["status"], "complete")
            self.assertEqual(summary["unsupervised"], [])
            self.assertIn("Ablations are deferred", (root / "analysis/report.md").read_text())


if __name__ == "__main__":
    unittest.main()
