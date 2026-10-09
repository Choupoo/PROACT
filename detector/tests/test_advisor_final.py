"""CPU-only checks for the approved grid; fixtures are not thesis results."""

import argparse
import ast
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import unittest
from unittest import mock

import numpy as np

from detector import advisor_detector as advisor
from detector import advisor_final as final
from detector import advisor_study as study
from detector import transfer_detector as base
from detector.common import save_json, sha256_file
from detector.io_utils import DETECTOR_ROOT, load_frozen_bundle
from detector.tests.test_advisor import predicted, settings
from detector.tests.test_transfer import write_table
from detector.transfer_core import fingerprint


class FinalStudyTests(unittest.TestCase):
    def test_fixed_rule_does_not_switch_even_when_another_rule_is_feasible(self):
        reports = {}
        for rule in advisor.RULES:
            reports[rule] = dict(sample_metrics={"clean_fpr": .5 if rule == final.FIXED_RULE else .01},
                dataset_results=[dict(realized_rate=0, alert_rate=.5 if rule == final.FIXED_RULE else 0,
                    alternative="poison", requested_rate=0), dict(realized_rate=.1, alert_rate=.8,
                    alternative="poison", requested_rate=.1)])
        result = advisor.select_rule(reports, .05, fixed_rule=final.FIXED_RULE)
        self.assertEqual(result["selected_rule"], final.FIXED_RULE)
        self.assertFalse(result["development_criterion_met"])
        self.assertEqual(result["selection_mode"], "prespecified")
        self.assertIsNotNone(result["warning"])

    def test_group_removals_use_exact_full_subsets(self):
        table, meta = predicted()
        full = set(base.resolve_feature_columns(table, meta, "inference_full"))
        uncertainty = {"entropy", "confidence", "margin"}
        without = set(base.resolve_feature_columns(table, meta, "inference_without_uncertainty"))
        self.assertEqual(without, full - uncertainty)
        for feature in base.INFERENCE_CONTEXT:
            columns = set(base.resolve_feature_columns(table, meta, "inference_without_" + feature))
            self.assertEqual(columns, full - {feature})
        self.assertEqual(set(base.resolve_feature_columns(table, meta, "inference_context")),
                         set(base.INFERENCE_CONTEXT))

    def test_registered_final_plan_is_idempotent_and_rejects_scope_changes(self):
        with tempfile.TemporaryDirectory(dir=DETECTOR_ROOT / "tests") as directory:
            root = Path(directory) / "run"
            args = argparse.Namespace(output_dir=str(root), methods=list(study.METHODS), seeds=[6, 7, 8])
            final.prepare(args)
            before = sha256_file(root / "plan.json")
            final.prepare(args)
            self.assertEqual(sha256_file(root / "plan.json"), before)
            _, payload = study.read_plan(root / "plan.json")
            options = payload["settings"]
            self.assertTrue(options["ablations"])
            self.assertTrue(options["unsupervised"])
            self.assertTrue(options["effectiveness"])
            self.assertEqual(options["threshold_rule"], final.FIXED_RULE)
            for method in study.METHODS:
                cell, steps = study.build_cell(root, options, method, 6)
                freeze = next(s for s in steps if s["name"] == "freeze_supervised")
                argv = freeze["command"]
                self.assertEqual(argv[argv.index("--threshold-rule") + 1], final.FIXED_RULE)
                target_attacks = [s for s in steps if s["name"].startswith("task9_") and s["name"].endswith("_attack")]
                self.assertEqual(len(target_attacks), 4)
                actual = {(s["command"][s["command"].index("--mode") + 1],
                           float(s["command"][s["command"].index("--delta") + 1])) for s in target_attacks}
                self.assertEqual(actual, set(study.ATTACKS))
                for s in steps:
                    self.assertTrue(cell == Path(s["cwd"]) or cell in Path(s["cwd"]).parents)
                self.assertLess(steps.index(freeze), next(i for i, s in enumerate(steps) if s["name"] == "task9_victim"))
            changed = argparse.Namespace(**vars(args))
            changed.seeds = [6, 7]
            with self.assertRaisesRegex(ValueError, "Existing plan differs"):
                final.prepare(changed)

    def test_cuda_preflight_blocks_real_run_but_not_dry_run(self):
        args = argparse.Namespace(output_dir="unused", method=None, seed=None,
                                  dry_run=False, retry_failed=False)
        with mock.patch.object(final, "diagnose", return_value={"ready": False}), \
             mock.patch.object(study, "run") as runner:
            with self.assertRaisesRegex(RuntimeError, "No training was started"):
                final.run(args)
            runner.assert_not_called()
            args.dry_run = True
            final.run(args)
            runner.assert_called_once()

    def test_fixed_rule_ablations_and_summary_end_to_end(self):
        with tempfile.TemporaryDirectory(dir=DETECTOR_ROOT / "tests") as directory:
            root = Path(directory)
            cell = root / "ewc/seed3"
            cell.mkdir(parents=True)
            paths = []
            for task in (1, 4, 9):
                path = root / ("task{}.csv".format(task))
                write_table(*predicted(task), path)
                paths.append(path)
            frozen = cell / "frozen_supervised"
            advisor.develop(paths[0], paths[1:2], frozen, task_size=16, repeats=2,
                            ablations=True, threshold_rule=final.FIXED_RULE)
            freeze = json.loads((frozen / "freeze.json").read_text())
            self.assertEqual(freeze["prespecified_threshold_rule"], final.FIXED_RULE)
            self.assertEqual(len(freeze["variants"]), 9)
            for name in freeze["variants"]:
                bundle = load_frozen_bundle(frozen / "ablations" / name / "bundle.joblib")
                self.assertEqual(bundle["threshold_rule"], final.FIXED_RULE)
            output = cell / "task9/reckless_0p3/evaluation"
            study.evaluate(argparse.Namespace(output_dir=str(output), frozen=str(frozen),
                features=str(paths[2]), reference=None, bags_per_rate=2))
            options = settings()
            options.update(methods=["ewc"], seeds=[3], history_tasks=[4], attacks=[("reckless", .3)],
                           ablations=True, unsupervised=False, effectiveness=False,
                           threshold_rule=final.FIXED_RULE)
            with mock.patch.object(study, "read_plan", return_value=(root, fingerprint(options))), \
                 mock.patch.object(study, "build_cell", return_value=(cell, [])):
                study.summarize(argparse.Namespace(plan="fixture"))
            summary = json.loads((root / "analysis/summary.json").read_text())
            self.assertEqual(summary["status"], "complete")
            self.assertEqual(len(summary["paired_ablation_deltas"]), 8)
            baseline = next(r for r in summary["supervised"] if r["variant"] == "inference_full")
            for item in summary["paired_ablation_deltas"]:
                variant = next(r for r in summary["supervised"] if r["variant"] == item["variant"])
                self.assertAlmostEqual(item["roc_auc_minus_full"], variant["roc_auc"] - baseline["roc_auc"])
                self.assertEqual(len(item["dataset_detection_deltas"]), len(baseline["dataset_results"]))
            self.assertTrue(all(r["n_seeds"] == 1 and r["roc_auc"]["std"] is None for r in summary["aggregates"]))
            self.assertTrue(all(r["dataset_results"] for r in summary["aggregates"]))
            text = (root / "analysis/report.md").read_text()
            self.assertIn("prespecified as history_mad", text)
            self.assertIn("Rank/MMD were not enabled", text)
            self.assertNotIn("contains every ablation", text)

    def test_packaging_marks_incomplete_and_excludes_heavy_artifacts(self):
        with tempfile.TemporaryDirectory(dir=DETECTOR_ROOT / "tests") as directory:
            root = Path(directory) / "run"
            args = argparse.Namespace(output_dir=str(root), methods=["ewc"], seeds=[6])
            final.prepare(args)
            feature = root / "ewc/seed6/task1/reckless_0p3/features"
            feature.mkdir(parents=True)
            (feature / "features.csv").write_text("omitted heavy input")
            (feature / "features.metadata.json").write_text("{}")
            (feature / "manifest.csv").write_text("original_index,split\n0,train\n")
            model = root / "ewc/seed6/frozen_supervised"
            model.mkdir(parents=True)
            (model / "bundle.joblib").write_text("omitted model")
            (model / "freeze.json").write_text("{}")
            package_args = argparse.Namespace(output_dir=str(root), archive=str(Path(directory) / "analysis.tar.gz"),
                                              allow_incomplete=False)
            with self.assertRaisesRegex(RuntimeError, "Run is incomplete"):
                final.package(package_args)
            self.assertFalse(Path(package_args.archive).exists())
            package_args.allow_incomplete = True
            archive = final.package(package_args)
            with tarfile.open(archive) as bundle:
                names = bundle.getnames()
                self.assertTrue(any(n.endswith("manifest.csv") for n in names))
                self.assertFalse(any(n.endswith("features.csv") or n.endswith(".joblib") for n in names))
                manifest = json.load(bundle.extractfile(root.name + "/package_manifest.json"))
                self.assertEqual(manifest["status"], "incomplete")
                for name, checksum in manifest["files"].items():
                    self.assertEqual(checksum, sha256_file(root / name))
            with self.assertRaises(FileExistsError):
                final.package(package_args)

    def test_changed_python_files_parse_as_python38(self):
        for name in ("advisor_final.py", "advisor_study.py", "advisor_detector.py", "transfer_detector.py"):
            ast.parse((DETECTOR_ROOT / name).read_text(), filename=name, feature_version=8)

    def test_shell_entrypoint_and_server_local_python_paths(self):
        with tempfile.TemporaryDirectory(dir=DETECTOR_ROOT / "tests") as directory:
            env = dict(os.environ, DETECTOR_PYTHON=sys.executable,
                       DETECTOR_FINAL_OUTPUT=str(Path(directory) / "run"))
            script = str(DETECTOR_ROOT / "run_advisor_final.sh")
            registered = subprocess.run(["bash", script, "plan", "--methods", "ewc", "mas", "--seeds", "6"],
                env=env, cwd=directory, capture_output=True, text=True)
            self.assertEqual(registered.returncode, 0, registered.stderr)
            dry = subprocess.run(["bash", script, "run", "--method", "mas", "--seed", "6", "--dry-run"],
                env=env, cwd=directory, capture_output=True, text=True)
            self.assertEqual(dry.returncode, 0, dry.stderr)
            self.assertIn("--approach mas", dry.stdout)
            self.assertIn("--threshold-rule history_mad", dry.stdout)
            self.assertIn(sys.executable, dry.stdout)
            self.assertFalse((Path(directory) / "run/mas/seed6/run_state.json").exists())

    def test_seed_aggregation_and_unsupported_rank_are_not_clean_decisions(self):
        with tempfile.TemporaryDirectory(dir=DETECTOR_ROOT / "tests") as directory:
            root = Path(directory)
            options = settings()
            options.update(methods=["ewc"], seeds=[6, 7], history_tasks=[4], attacks=[("reckless", .3)],
                           ablations=False, unsupervised=True, effectiveness=False,
                           threshold_rule=final.FIXED_RULE)
            for seed, auc in ((6, .8), (7, .9)):
                cell = root / "ewc" / ("seed" + str(seed))
                save_json(dict(development_criterion_met=True, selected_rule=final.FIXED_RULE),
                          cell / "frozen_supervised/threshold_selection.json")
                case = cell / "task9/reckless_0p3/evaluation"
                report = dict(sample_metrics=dict(roc_auc=auc, average_precision=auc,
                    clean_fpr=.01, poison_tpr=.9), random_control_sample_alert_rate=.1,
                    threshold_rule=final.FIXED_RULE, dataset_results=[
                        dict(alternative="poison", requested_rate=0., realized_rate=0., alert_rate=.02),
                        dict(alternative="poison", requested_rate=.05, realized_rate=8 / 150, alert_rate=auc)])
                for family, names in (("ablations", ["inference_full"]), ("thresholds", advisor.RULES)):
                    for name in names:
                        save_json(report, case / family / name / "evaluation_metrics.json")
                save_json(dict(summary=[dict(scenario="clean", requested_rate=0., realized_rate=0.,
                    rank_alert_rate=.2 if seed == 6 else None, rank_coverage=1. if seed == 6 else .5,
                    legacy_alert_rate=1.)], limitations=[]), case / "unsupervised.json")
            def empty_steps(unused_root, unused_settings, method, seed):
                return root / method / ("seed" + str(seed)), []
            with mock.patch.object(study, "read_plan", return_value=(root, fingerprint(options))), \
                 mock.patch.object(study, "build_cell", side_effect=empty_steps):
                study.summarize(argparse.Namespace(plan="fixture"))
            summary = json.loads((root / "analysis/summary.json").read_text())
            aggregate = next(r for r in summary["aggregates"] if r["variant"] == "inference_full")
            self.assertEqual(aggregate["n_seeds"], 2)
            self.assertAlmostEqual(aggregate["roc_auc"]["mean"], .85)
            self.assertAlmostEqual(aggregate["roc_auc"]["std"], np.sqrt(.005))
            self.assertEqual(aggregate["dataset_results"][1]["n_seeds"], 2)
            rank = summary["unsupervised_aggregates"][0]
            self.assertIsNone(rank["rank_alert_rate"]["mean"])
            self.assertEqual(rank["rank_alert_rate"]["n_supported_seeds"], 1)
            self.assertEqual(rank["rank_coverage"]["mean"], .75)


if __name__ == "__main__":
    unittest.main()
