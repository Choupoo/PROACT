"""Negative-control training isolation and fixed-method thesis workflow."""

import copy
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

from detector import FEATURE_COLUMNS
from detector import explain_detector, thesis_closeout
from detector import transfer_detector as detector
from detector.common import save_json, sha256_file
from detector.io_utils import DETECTOR_ROOT
from detector.tests import test_rank_reference as fixtures
from detector.tests.test_transfer import feature_fixture, write_table


class NegativeControlTests(unittest.TestCase):
    def test_only_source_training_random_rows_are_relabelled_with_balanced_weight(self):
        table, meta = feature_fixture()
        before = table.copy(deep=True)
        with mock.patch.object(
            detector, "fit_detector", wraps=detector.fit_detector
        ) as fitting:
            bundle = detector.fit_source(
                table, meta, negative_policy="clean_and_random", task_size=16
            )
        training = fitting.call_args.args[0]
        weights = fitting.call_args.kwargs["sample_weight"]
        self.assertEqual(set(training.split), {"train"})
        self.assertEqual(set(training.view), {"clean", "poison", "random_control"})
        self.assertTrue(
            (
                training.loc[training.view == "random_control", "detector_label"] == 0
            ).all()
        )
        self.assertEqual(
            weights[training.detector_label == 0].sum(),
            weights[training.detector_label == 1].sum(),
        )
        pd.testing.assert_frame_equal(table, before)
        self.assertEqual(
            bundle["count_calibration"]["calibration_original_images"],
            int(((table.split == "reserve") & (table.view == "clean")).sum()),
        )

    def test_heldout_views_cannot_change_fit_or_thresholds(self):
        table, meta = feature_fixture()
        first = detector.fit_source(
            table, meta, negative_policy="clean_and_random", task_size=16
        )
        changed = table.copy()
        mask = (table.split == "test") | (
            (table.split != "train") & (table.view == "random_control")
        )
        changed.loc[mask, FEATURE_COLUMNS] *= 50
        second = detector.fit_source(
            changed, meta, negative_policy="clean_and_random", task_size=16
        )
        np.testing.assert_array_equal(
            first["classifier"].coef_, second["classifier"].coef_
        )
        np.testing.assert_array_equal(first["scaler"].mean_, second["scaler"].mean_)
        self.assertEqual(first["threshold"], second["threshold"])
        self.assertEqual(first["count_calibration"], second["count_calibration"])

    def test_only_opt_in_model_responds_to_training_random_controls(self):
        table, meta = feature_fixture()
        changed = table.copy()
        changed.loc[
            (table.split == "train") & (table.view == "random_control"), FEATURE_COLUMNS
        ] *= 5
        for policy in ("clean_only", "clean_and_random"):
            first = detector.fit_source(
                table, meta, negative_policy=policy, task_size=16
            )
            second = detector.fit_source(
                changed, meta, negative_policy=policy, task_size=16
            )
            equal = np.array_equal(
                first["classifier"].coef_, second["classifier"].coef_
            )
            self.assertEqual(equal, policy == "clean_only")

    def test_explanation_includes_actual_random_training_background(self):
        table, meta = feature_fixture()
        bundle = detector.fit_source(
            table, meta, negative_policy="clean_and_random", task_size=16
        )
        with tempfile.TemporaryDirectory(dir=DETECTOR_ROOT / "tests") as directory:
            report = explain_detector.explain(table, meta, bundle, Path(directory))
        self.assertEqual(report["background_rows"], int((table.split == "train").sum()))
        self.assertEqual(report["background_views"], bundle["fit_views"])
        self.assertLess(report["max_additivity_error"], 1e-10)
        with self.assertRaisesRegex(ValueError, "ablations"):
            explain_detector.explain(table, meta, bundle, DETECTOR_ROOT, ablations=True)

    def test_aggregate_is_run_weighted_and_missing_coverage_not_zero(self):
        self.assertEqual(thesis_closeout.stats([0.0, 1.0])["mean"], 0.5)
        self.assertAlmostEqual(thesis_closeout.stats([0.0, 1.0])["std"], np.sqrt(0.5))
        self.assertIsNone(thesis_closeout.stats([0.0])["std"])
        self.assertIsNone(thesis_closeout.stats([0.0, None])["mean"])


class CloseoutCliTests(unittest.TestCase):
    def execute(self, module, *args, success=True):
        result = subprocess.run(
            [sys.executable, "-B", "-m", "detector." + module, *map(str, args)],
            cwd=DETECTOR_ROOT.parent,
            capture_output=True,
            text=True,
        )
        if success:
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        else:
            self.assertNotEqual(result.returncode, 0)
        return result

    def test_complete_paired_study_rank_recheck_summary_and_rejections(self):
        with tempfile.TemporaryDirectory(dir=DETECTOR_ROOT / "tests") as directory:
            root = Path(directory)
            source = root / "source"
            for task in (1, 9):
                folder = source / "seed3" / ("features_task" + str(task))
                folder.mkdir(parents=True)
                table, meta = feature_fixture(task)
                meta["checkpoint_seed"] = 3
                write_table(table, meta, folder / "features.csv")
            sources = []
            for index in (1, 2):
                folder = root / ("unlabeled" + str(index))
                folder.mkdir()
                sources.append(folder)
                for name, table, role in (
                    (
                        "reference_features.csv",
                        fixtures.reference(),
                        "historical_inversion",
                    ),
                    (
                        "predicted_features.csv",
                        fixtures.benchmark(),
                        "paired_benchmark",
                    ),
                ):
                    meta = dict(
                        fixtures.metadata_fixture(role),
                        feature_columns=list(fixtures.legacy.FEATURE_COLUMNS),
                        synthetic=True,
                        checkpoint_sha256=str(index) * 64,
                        input_sha256={"attack": str(index + 2) * 64},
                    )
                    write_table(table, meta, folder / name)
            inputs = {
                str(p): sha256_file(p)
                for folder in [source] + sources
                for p in folder.rglob("*")
                if p.is_file()
            }
            sup = root / "supervised"
            args = [
                "supervised",
                "--source-run",
                source,
                "--output-dir",
                sup,
                "--seeds",
                3,
                "--task-size",
                16,
                "--bags-per-rate",
                1,
                "--negative-policies",
                "clean_only",
                "clean_and_random",
            ]
            self.execute("revision_study", *args, "--dry-run")
            self.assertFalse(sup.exists())
            self.execute("revision_study", *args)
            summary = json.loads((sup / "summary.json").read_text())
            self.assertEqual(len(summary["runs"]), 12)
            self.assertEqual(
                {r["negative_policy"] for r in summary["runs"]},
                {"clean_only", "clean_and_random"},
            )
            for row in summary["runs"]:
                self.assertIn("poison_vs_random_roc_auc", row)
            ranked = root / "rank"
            args = [
                "rank",
                "--source-runs",
                *sources,
                "--output-dir",
                ranked,
                "--task-size",
                16,
                "--bags-per-rate",
                1,
            ]
            self.execute("thesis_closeout", *args, "--dry-run")
            self.assertFalse(ranked.exists())
            self.execute("thesis_closeout", *args)
            self.execute("thesis_closeout", *args, success=False)
            self.assertEqual(
                json.loads((ranked / "run_state.json").read_text())["status"],
                "complete",
            )
            rank_summary = json.loads((ranked / "summary.json").read_text())
            self.assertEqual(rank_summary["distinct_checkpoints"], 2)
            metrics = [
                json.loads(
                    (
                        ranked / ("replicate" + str(i)) / "evaluation_metrics.json"
                    ).read_text()
                )
                for i in (1, 2)
            ]
            duplicate = copy.deepcopy(metrics)
            duplicate[1]["replication_identity"] = duplicate[0]["replication_identity"]
            with self.assertRaisesRegex(ValueError, "Repeated"):
                thesis_closeout.summarize_rank(duplicate)
            final = root / "final"
            args = [
                "summarize",
                "--supervised-run",
                sup,
                "--rank-run",
                ranked,
                "--output-dir",
                final,
            ]
            self.execute("thesis_closeout", *args)
            combined = json.loads((final / "thesis_summary.json").read_text())
            self.assertFalse(combined["automatic_thesis_pass"])
            self.assertTrue(combined["synthetic"])
            self.assertTrue((final / "thesis_summary.md").is_file())
            self.execute("thesis_closeout", *args, success=False)
            # A forged complete run missing a variant must not silently summarize.
            summary["runs"].pop()
            save_json(summary, sup / "summary.json")
            self.execute(
                "thesis_closeout",
                "summarize",
                "--supervised-run",
                sup,
                "--rank-run",
                ranked,
                "--output-dir",
                root / "invalid_summary",
                success=False,
            )
            self.assertFalse((root / "invalid_summary").exists())
            for path, digest in inputs.items():
                self.assertEqual(sha256_file(path), digest)


if __name__ == "__main__":
    unittest.main()
