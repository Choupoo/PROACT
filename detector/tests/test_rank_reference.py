"""Math, supervision boundaries, blind spots and isolated CLI integration."""

import argparse
import copy
import itertools
import tempfile
import unittest
from pathlib import Path
import subprocess
import sys
from unittest import mock

import numpy as np
import pandas as pd

from detector import rank_reference as rank
from detector import unsupervised as legacy
from detector import unsupervised_adapt as experiment
from detector.common import save_json, sha256_file
from detector.io_utils import load_frozen_bundle, save_frozen_bundle
from detector.tests.test_unsupervised import metadata_fixture

ROOT = Path(__file__).resolve().parents[1]


def descriptors(n=64, seed=12, reverse=False):
    rng = np.random.default_rng(seed)
    common = rng.normal(size=(n, 1))
    values = common + rng.normal(scale=0.45, size=(n, len(legacy.FEATURE_COLUMNS)))
    if reverse:
        values[:, 0] *= -1
    for index, column in enumerate(legacy.FEATURE_COLUMNS):
        values[:, index] = (
            1 / (1 + np.exp(-values[:, index]))
            if column in ("confidence", "margin")
            else np.exp(values[:, index] / 3 + 1)
        )
    return pd.DataFrame(values, columns=legacy.FEATURE_COLUMNS)


def reference(tasks=2, n=64):
    tables = []
    for task in range(tasks):
        table = descriptors(n, 20 + task)
        table["reference_id"] = ["task{}:sample{}".format(task, i) for i in range(n)]
        table["reference_task_id"] = task
        tables.append(table)
    return pd.concat(tables, ignore_index=True)


def benchmark():
    tables = []
    for offset, split in enumerate(("train", "validation", "test")):
        for view in ("clean", "poison", "random_control"):
            table = descriptors(32, 33 + offset, reverse=view == "poison")
            if view == "random_control":
                table["grad_norm_l2"] *= 4
            table["original_index"] = np.arange(32) + 100 * offset
            table["split"] = split
            table["view"] = view
            tables.append(table)
    return pd.concat(tables, ignore_index=True)


def small():
    return rank.fit_reference(reference(), metadata_fixture(), bootstrap_draws=39)


class RankMathTests(unittest.TestCase):
    def test_tau_matches_explicit_pairs_including_ties(self):
        x = np.array([[1, 3, 2], [1, 2, 4], [3, 1, 4], [4, 4, 1]], dtype=float)
        profile = rank.rank_profile(x)
        expected = [
            np.mean(
                [
                    np.sign(x[a, i] - x[b, i]) * np.sign(x[a, j] - x[b, j])
                    for a, b in itertools.combinations(range(len(x)), 2)
                ]
            )
            for i, j in itertools.combinations(range(3), 2)
        ]
        np.testing.assert_allclose(profile["tau"], expected)

    def test_pseudovalues_match_direct_leave_one_out_jackknife(self):
        x = np.random.default_rng(2).normal(size=(9, 4))
        profile = rank.rank_profile(x)
        pseudo = np.array(
            [
                9 * profile["tau"]
                - 8 * rank.rank_profile(np.delete(x, i, axis=0))["tau"]
                for i in range(9)
            ]
        )
        np.testing.assert_allclose(
            profile["pseudo"], pseudo - pseudo.mean(axis=0), atol=1e-14
        )
        draws = rank.multiplier_draws(profile, 39, 7)
        expected = np.random.default_rng(7).normal(size=(39, 9)).dot(
            profile["pseudo"]
        ) / np.sqrt(72)
        np.testing.assert_allclose(draws, expected)

    def test_exact_invariance_to_monotone_marginal_changes_and_blindness(self):
        x = descriptors().to_numpy()
        first, changed = rank.rank_profile(x), rank.rank_profile(x**3 + 9)
        np.testing.assert_array_equal(first["tau"], changed["tau"])
        np.testing.assert_array_equal(first["pseudo"], changed["pseudo"])

    def test_invalid_profiles_rejected(self):
        for x in (np.ones((3, 2)), np.ones((4, 1)), np.full((4, 2), np.nan)):
            with self.assertRaises(ValueError):
                rank.rank_profile(x)


class RankBoundaryTests(unittest.TestCase):
    def test_fit_ignores_all_class_attack_and_split_labels(self):
        table = reference()
        expected = small()
        table["class_id"], table["view"], table["detector_label"], table["split"] = (
            np.nan,
            "poison",
            "invalid",
            "test",
        )
        actual = rank.fit_reference(table, metadata_fixture(), bootstrap_draws=39)
        for a, b in zip(expected["profiles"], actual["profiles"]):
            np.testing.assert_array_equal(a["tau"], b["tau"])
            np.testing.assert_array_equal(a["bootstrap_errors"], b["bootstrap_errors"])

    def test_rejects_clean_incoming_reference_true_labels_and_bad_identities(self):
        for key, value in (
            ("origin_role", "paired_benchmark"),
            ("label_mode", "ground_truth"),
        ):
            meta = metadata_fixture()
            meta[key] = value
            with self.assertRaises(ValueError):
                rank.fit_reference(reference(), meta)
        for column, value in (
            ("reference_id", "task0:sample0"),
            ("reference_task_id", 7),
        ):
            table = reference()
            table[column] = value
            with self.assertRaises(ValueError):
                rank.fit_reference(table, metadata_fixture())

    def test_fit_rejects_degenerate_or_too_small_references(self):
        table = reference()
        table[legacy.FEATURE_COLUMNS] = 0.5
        with self.assertRaises(ValueError):
            rank.fit_reference(table, metadata_fixture())
        with self.assertRaises(ValueError):
            rank.fit_reference(reference(n=12), metadata_fixture())

    def test_prediction_invariant_to_labels_row_order_and_marginal_scale(self):
        bundle = small()
        table = descriptors(48, 44)
        meta = metadata_fixture("incoming")
        expected = rank.predict_dataset(bundle, table, meta)
        table["class_id"], table["view"], table["detector_label"] = (
            None,
            "arbitrary",
            "not read",
        )
        self.assertEqual(rank.predict_dataset(bundle, table.iloc[::-1], meta), expected)
        for column in legacy.FEATURE_COLUMNS:
            table[column] = (
                table[column] ** 3
                if column in ("confidence", "margin")
                else table[column] * 30 + 100
            )
        self.assertEqual(rank.predict_dataset(bundle, table, meta), expected)

    def test_changed_dependence_is_detected_in_synthetic_example(self):
        result = rank.predict_dataset(
            small(), descriptors(128, 55, reverse=True), metadata_fixture("incoming")
        )
        self.assertTrue(result["shift_detected"])
        self.assertEqual(
            result["p_value_approx"],
            max(row["p_value_approx"] for row in result["comparisons"]),
        )
        self.assertEqual(result["poisoning_decision"], "undetermined")

    def test_one_compatible_history_prevents_union_rejection(self):
        table = reference()
        table.loc[table.reference_task_id == 1, legacy.FEATURE_COLUMNS] = descriptors(
            64, 21, reverse=True
        ).to_numpy()
        bundle = rank.fit_reference(table, metadata_fixture(), bootstrap_draws=39)
        incoming = descriptors(64, 20)
        result = rank.predict_dataset(bundle, incoming, metadata_fixture("incoming"))
        self.assertEqual(result["p_value_approx"], 1.0)
        self.assertFalse(result["shift_detected"])
        self.assertTrue(
            any(row["p_value_approx"] <= 0.05 for row in result["comparisons"])
        )

    def test_unsupported_constant_input_is_not_counted_as_clean(self):
        table = descriptors()
        table[legacy.FEATURE_COLUMNS] = 0.5
        result = rank.predict_dataset(small(), table, metadata_fixture("incoming"))
        self.assertIsNone(result["shift_detected"])
        self.assertIsNone(result["p_value_approx"])
        self.assertEqual(result["status"], "unsupported_constant_features")

    def test_provenance_overlap_and_corrupt_bundle_rejected(self):
        bundle = small()
        with self.assertRaises(ValueError):
            rank.predict_dataset(bundle, reference().iloc[:32], metadata_fixture())
        meta = metadata_fixture("incoming")
        meta["head_seed"] = 999
        with self.assertRaises(ValueError):
            rank.predict_dataset(bundle, descriptors(), meta)
        bundle["profiles"][0]["bootstrap_errors"][0, 0] = np.nan
        with self.assertRaises(ValueError):
            rank.validate_bundle(bundle)

    def test_historical_audit_excludes_held_task(self):
        result = rank.historical_audit(small())
        self.assertEqual(len(result["tasks"]), 2)
        for row in result["tasks"]:
            self.assertNotIn(
                row["held_out_task"],
                [r["reference_task_id"] for r in row["comparisons"]],
            )
        self.assertFalse(result["threshold_tuned"])

    def test_bundle_round_trip_is_immutable(self):
        bundle = small()
        with tempfile.TemporaryDirectory(dir=ROOT / "tests") as temp:
            path = save_frozen_bundle(bundle, temp, "rank_bundle.joblib")
            restored = load_frozen_bundle(path)
            self.assertEqual(
                rank.predict_dataset(
                    bundle, descriptors(), metadata_fixture("incoming")
                ),
                rank.predict_dataset(
                    restored, descriptors(), metadata_fixture("incoming")
                ),
            )
            with self.assertRaises(FileExistsError):
                save_frozen_bundle(bundle, temp, "rank_bundle.joblib")


class ExperimentTests(unittest.TestCase):
    def baseline(self):
        return legacy.fit_reference(
            reference(), metadata_fixture(), permutations=19, n_components=16
        )

    def test_evaluation_only_uses_test_and_never_refits_model(self):
        bundle = small()
        before = copy.deepcopy(bundle)
        table = benchmark()
        kwargs = dict(task_size=20, tasks_per_rate=1)
        result = experiment.evaluate_benchmark(
            bundle,
            self.baseline(),
            table,
            metadata_fixture("paired_benchmark"),
            **kwargs,
        )
        table.loc[table.split != "test", legacy.FEATURE_COLUMNS] = np.nan
        self.assertEqual(
            result,
            experiment.evaluate_benchmark(
                bundle,
                self.baseline(),
                table,
                metadata_fixture("paired_benchmark"),
                **kwargs,
            ),
        )
        self.assertEqual(len(result["summary"]), 11)
        self.assertEqual(len(result["skipped_scenarios"]), 2)
        self.assertTrue(
            all(
                row["modified_count"] > 0
                for row in result["summary"]
                if row["scenario"] != "clean"
            )
        )
        self.assertFalse(result["threshold_tuned_on_evaluation"])
        for a, b in zip(bundle["profiles"], before["profiles"]):
            np.testing.assert_array_equal(a["tau"], b["tau"])
            np.testing.assert_array_equal(a["bootstrap_errors"], b["bootstrap_errors"])

    def test_no_silent_subsampling_or_original_overlap(self):
        table = benchmark()
        for changed, size in ((table, 300), (table.assign(original_index=0), 20)):
            with self.assertRaises(ValueError):
                experiment.evaluate_benchmark(
                    small(),
                    self.baseline(),
                    changed,
                    metadata_fixture("paired_benchmark"),
                    task_size=size,
                    tasks_per_rate=1,
                )

    def test_missing_coverage_stays_missing_not_zero_false_alerts(self):
        table = benchmark()
        table.loc[
            (table.split == "test") & (table.view == "clean"), legacy.FEATURE_COLUMNS
        ] = 0.5
        result = experiment.evaluate_benchmark(
            small(),
            self.baseline(),
            table,
            metadata_fixture("paired_benchmark"),
            task_size=20,
            tasks_per_rate=1,
        )
        row = result["summary"][0]
        self.assertEqual(row["rank_coverage"], 0)
        self.assertIsNone(row["rank_alert_rate"])

    def test_complete_experiment_preserves_source_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "tests") as temp:
            root = Path(temp)
            source = root / "source"
            source.mkdir()

            def write(table, name, role):
                path = source / name
                table.to_csv(path, index=False)
                meta = dict(
                    metadata_fixture(role),
                    feature_columns=legacy.FEATURE_COLUMNS,
                    row_count=len(table),
                    features_sha256=sha256_file(path),
                    synthetic=True,
                )
                save_json(meta, path.with_suffix(".metadata.json"))
                return meta

            ref = reference()
            meta = write(ref, "reference_features.csv", "historical_inversion")
            write(benchmark(), "predicted_features.csv", "paired_benchmark")
            baseline = legacy.fit_reference(ref, meta, permutations=19, n_components=16)
            legacy.save_bundle(baseline, source / "unsupervised")
            initial = {str(p): sha256_file(p) for p in source.rglob("*") if p.is_file()}
            args = argparse.Namespace(
                source_run=str(source),
                output_dir=str(root / "result"),
                alpha=0.05,
                bootstrap_draws=39,
                seed=12,
                max_reference_per_task=64,
                max_incoming=64,
                task_size=20,
                tasks_per_rate=1,
                dry_run=True,
            )
            experiment.run(args)
            self.assertFalse((root / "result").exists())
            args.dry_run = False
            reader = experiment.read_feature_table

            def read_after_freeze(path):
                if Path(path).name == "predicted_features.csv":
                    self.assertTrue(
                        (root / "result" / "rank" / "rank_bundle.joblib").is_file()
                    )
                return reader(path)

            with mock.patch.object(
                experiment, "read_feature_table", side_effect=read_after_freeze
            ):
                experiment.run(args)
            self.assertIn(
                "SYNTHETIC FIXTURE ONLY", (root / "result" / "report.md").read_text()
            )
            self.assertEqual(
                initial,
                {str(p): sha256_file(p) for p in source.rglob("*") if p.is_file()},
            )
            with self.assertRaises(ValueError):
                experiment.run(args)
            # Actual command-line run, separate output, same frozen input artifacts.
            process = subprocess.run(
                [
                    sys.executable,
                    "-B",
                    "-m",
                    "detector.unsupervised_adapt",
                    "run",
                    "--source-run",
                    str(source),
                    "--output-dir",
                    str(root / "cli_result"),
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
            self.assertTrue((root / "cli_result" / "report.md").is_file())
            self.assertEqual(
                initial,
                {str(p): sha256_file(p) for p in source.rglob("*") if p.is_file()},
            )


if __name__ == "__main__":
    unittest.main()
