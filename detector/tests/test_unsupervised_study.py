"""Local novelty math, frozen label-free boundaries and study workflow tests."""

import argparse
import copy
import json
import shlex
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

from detector import local_reference as local
from detector import unsupervised_study as study
from detector.common import save_json, sha256_file
from detector.io_utils import read_feature_table
from detector.tests import test_rank_reference as fixtures

ROOT = Path(__file__).resolve().parents[1]


class LocalReferenceTests(unittest.TestCase):
    def fit(self, table=None):
        return local.fit_reference(
            fixtures.reference() if table is None else table,
            fixtures.metadata_fixture(),
            task_size=20,
        )

    def test_shape_is_invariant_to_independent_sample_magnitude(self):
        table = fixtures.descriptors(20)
        shapes, valid = local.shapes(table)
        table[local.COLUMNS] *= np.logspace(-120, 120, 20)[:, None]
        changed, valid_changed = local.shapes(table)
        np.testing.assert_allclose(shapes, changed, atol=1e-14)
        np.testing.assert_array_equal(valid, valid_changed)
        np.testing.assert_allclose(np.linalg.norm(shapes, axis=1), 1)

    def test_nearest_neighbor_score_matches_direct_distance(self):
        table = fixtures.descriptors(20)
        values, _ = local.shapes(table)
        actual = local.score_shapes(values[:5], values[5:], 3)
        expected = [
            np.sort(np.linalg.norm(values[5:] - row, axis=1))[:3].mean() / np.sqrt(2)
            for row in values[:5]
        ]
        np.testing.assert_allclose(actual, expected)

    def test_three_reference_pools_are_disjoint_and_complete(self):
        bundle = self.fit()
        parts = bundle["partitions"]
        self.assertEqual(
            {k: len(v) for k, v in parts.items()},
            {"fit": 64, "threshold": 32, "calibration": 32},
        )
        ids = [x for part in parts.values() for x in part]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(set(ids), set(bundle["reference_ids"]))

    def test_calibration_cannot_change_bank_or_sample_threshold(self):
        table = fixtures.reference()
        first = self.fit(table)
        table.loc[
            table.reference_id.isin(first["partitions"]["calibration"]),
            local.COLUMNS[0],
        ] *= 100
        second = self.fit(table)
        np.testing.assert_array_equal(first["fit_bank"], second["fit_bank"])
        self.assertEqual(first["sample_threshold"], second["sample_threshold"])
        self.assertGreater(
            second["count_calibration"]["calibration_sample_alarms"],
            first["count_calibration"]["calibration_sample_alarms"],
        )

    def test_labels_are_ignored_by_fit_and_prediction(self):
        reference = fixtures.reference()
        expected = self.fit(reference)
        (
            reference["class_id"],
            reference["view"],
            reference["detector_label"],
            reference["split"],
        ) = np.nan, "poison", None, "test"
        actual = self.fit(reference)
        self.assertEqual(expected["sample_threshold"], actual["sample_threshold"])
        np.testing.assert_array_equal(expected["fit_bank"], actual["fit_bank"])
        incoming = fixtures.descriptors(20)
        result = local.predict_dataset(
            actual, incoming, fixtures.metadata_fixture("incoming")
        )
        incoming["class_id"], incoming["detector_label"], incoming["view"] = (
            "ignored",
            np.nan,
            "clean",
        )
        self.assertEqual(
            result,
            local.predict_dataset(
                actual, incoming, fixtures.metadata_fixture("incoming")
            ),
        )

    def test_sparse_shape_perturbations_have_sample_level_signal(self):
        bundle = self.fit()
        table = fixtures.descriptors(20, 22)
        table.loc[:7, local.COLUMNS[0]] *= 1000
        result = local.predict_dataset(
            bundle, table, fixtures.metadata_fixture("incoming")
        )
        self.assertTrue(
            all(x >= bundle["sample_threshold"] for x in result["sample_scores"][:8])
        )
        self.assertEqual(
            result["shift_detected"],
            result["suspicious_count"] >= result["critical_count"],
        )
        self.assertEqual(result["poisoning_decision"], "undetermined")

    def test_zero_gradient_missing_not_accepted_clean(self):
        bundle = self.fit()
        table = fixtures.descriptors(20)
        table.loc[0, local.COLUMNS] = 0
        result = local.predict_dataset(
            bundle, table, fixtures.metadata_fixture("incoming")
        )
        self.assertIsNone(result["shift_detected"])
        self.assertEqual(result["invalid_samples"], 1)

    def test_origin_mode_overlap_and_size_rejected(self):
        for key, value in (("origin_role", "incoming"), ("label_mode", "ground_truth")):
            meta = fixtures.metadata_fixture()
            meta[key] = value
            with self.assertRaises(ValueError):
                local.fit_reference(fixtures.reference(), meta)
        with self.assertRaises(ValueError):
            local.predict_dataset(
                self.fit(),
                fixtures.descriptors(19),
                fixtures.metadata_fixture("incoming"),
            )
        with self.assertRaises(ValueError):
            local.predict_dataset(
                self.fit(), fixtures.reference().iloc[:20], fixtures.metadata_fixture()
            )

    def test_historical_audit_has_no_incoming_data(self):
        result = local.historical_audit(
            fixtures.reference(), fixtures.metadata_fixture(), task_size=20
        )
        self.assertEqual(len(result["tasks"]), 2)
        self.assertFalse(result["threshold_tuned"])

    def test_frozen_calibration_corruption_is_rejected(self):
        bundle = self.fit()
        bundle["count_calibration"]["critical_suspicious_count"] += 1
        with self.assertRaises(ValueError):
            local.validate_bundle(bundle)


class StudyTests(unittest.TestCase):
    def write_source(self, root):
        root.mkdir()
        for name, table, role in (
            ("reference_features.csv", fixtures.reference(), "historical_inversion"),
            ("predicted_features.csv", fixtures.benchmark(), "paired_benchmark"),
        ):
            path = root / name
            table.to_csv(path, index=False)
            meta = dict(
                fixtures.metadata_fixture(role),
                feature_columns=fixtures.legacy.FEATURE_COLUMNS,
                row_count=len(table),
                features_sha256=sha256_file(path),
                synthetic=True,
            )
            if role == "paired_benchmark":
                meta["input_sha256"] = {"attack": "d" * 64}
            save_json(meta, path.with_suffix(".metadata.json"))

    def args(self, source, output):
        return argparse.Namespace(
            source_run=str(source),
            output_dir=str(output),
            task_size=20,
            tasks_per_rate=1,
            bootstrap_draws=39,
            protocol=None,
            evaluation_role="exploratory_reused",
            dry_run=False,
        )

    def test_full_study_freezes_before_benchmark_and_preserves_sources(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "tests") as temp:
            root = Path(temp)
            source, output = root / "source", root / "study"
            self.write_source(source)
            initial = {p.name: sha256_file(p) for p in source.iterdir()}
            args = self.args(source, output)
            args.dry_run = True
            study.run(args)
            self.assertFalse(output.exists())
            args.dry_run = False

            def guarded_read(path):
                if Path(path).name == "predicted_features.csv":
                    for name in ("legacy", "rank", "local"):
                        self.assertTrue(
                            (output / "models" / name / "bundle.joblib").is_file()
                        )
                return read_feature_table(path)

            with mock.patch.object(
                study, "read_feature_table", side_effect=guarded_read
            ):
                study.run(args)
            self.assertEqual(
                initial, {p.name: sha256_file(p) for p in source.iterdir()}
            )
            result = json.loads((output / "evaluation_metrics.json").read_text())
            self.assertTrue(all("local_alert_rate" in row for row in result["summary"]))
            self.assertTrue(
                all(len(set(row["original_indices"])) == 20 for row in result["tasks"])
            )
            state = json.loads((output / "run_state.json").read_text())
            self.assertTrue(all(x["status"] == "complete" for x in state.values()))
            with self.assertRaises(FileExistsError):
                study.run(args)
            summary = study.summarize([output], root / "summary")
            self.assertEqual(summary["runs"], 1)
            self.assertTrue(
                all(row["std_across_runs"] is None for row in summary["rows"])
            )
            with self.assertRaises(ValueError):
                study.summarize([output, output], root / "bad_summary")
            self.assertFalse((root / "bad_summary").exists())
            process = subprocess.run(
                [
                    sys.executable,
                    "-B",
                    "-m",
                    "detector.unsupervised_study",
                    "run",
                    "--source-run",
                    str(source),
                    "--output-dir",
                    str(root / "cli"),
                    "--task-size",
                    "20",
                    "--tasks-per-rate",
                    "1",
                    "--bootstrap-draws",
                    "39",
                ],
                cwd=ROOT.parent,
                capture_output=True,
                text=True,
                timeout=60,
            )
            self.assertEqual(process.returncode, 0, process.stderr)

    def test_protocol_changes_and_unregistered_prospective_rejected(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "tests") as temp:
            root = Path(temp)
            args = self.args(root / "missing", root / "output")
            args.evaluation_role = "prospective_declared"
            with self.assertRaises(ValueError):
                study.run(args)
            path = root / "protocol.json"
            save_json(study.fingerprint(study.DEFAULTS), path)
            args.protocol = str(path)
            with self.assertRaises(ValueError):
                study.run(
                    args
                )  # Smaller test settings do not match the frozen defaults.
            self.assertFalse((root / "output").exists())

    def test_plan_has_only_unsupervised_preparation_and_does_not_execute(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "tests") as temp:
            root = Path(temp) / "plan"
            study.plan(root, [1, 2])
            script = (root / "commands.sh").read_text()
            self.assertNotIn("train_detector", script)
            self.assertNotIn("pipeline", script)
            self.assertNotIn("ground_truth", script)
            self.assertNotIn("--data-cwd", script)
            self.assertIn("export PYTHONPATH=", script)
            self.assertIn("cd " + str(root / "artifacts_seed1"), script)
            from detector.extract_features import build_parser

            for line in script.splitlines():
                if "detector.extract_features" in line:
                    build_parser().parse_args(shlex.split(line)[5:])
            syntax = subprocess.run(
                ["bash", "-n", str(root / "commands.sh")],
                capture_output=True,
                text=True,
            )
            self.assertEqual(syntax.returncode, 0, syntax.stderr)
            self.assertEqual(script.count("detector.bootstrap"), 2)
            self.assertEqual(script.count("detector.unsupervised_study run"), 2)
            self.assertEqual(
                set(p.name for p in root.iterdir()), {"commands.sh", "protocol.json"}
            )
            self.assertEqual(
                json.loads((root / "protocol.json").read_text()),
                study.fingerprint(study.DEFAULTS),
            )

    def test_local_replay_ignores_non_test_descriptors(self):
        meta = fixtures.metadata_fixture()
        table = fixtures.benchmark()
        reference = fixtures.reference()
        ranked = fixtures.rank.fit_reference(reference, meta, bootstrap_draws=39)
        original = fixtures.legacy.fit_reference(reference, meta, permutations=19)
        bundle = local.fit_reference(reference, meta, task_size=20)
        before = copy.deepcopy(bundle)
        base = study.adaptation.evaluate_benchmark(
            ranked,
            original,
            table,
            fixtures.metadata_fixture("paired_benchmark"),
            task_size=20,
            tasks_per_rate=1,
        )
        expected = study.evaluate_local(
            base, bundle, table, fixtures.metadata_fixture("paired_benchmark")
        )
        table.loc[table.split != "test", local.COLUMNS] = np.nan
        actual = study.evaluate_local(
            base, bundle, table, fixtures.metadata_fixture("paired_benchmark")
        )
        self.assertEqual(expected, actual)
        np.testing.assert_array_equal(before["fit_bank"], bundle["fit_bank"])
        self.assertEqual(before["count_calibration"], bundle["count_calibration"])


if __name__ == "__main__":
    unittest.main()
