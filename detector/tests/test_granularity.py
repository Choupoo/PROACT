"""Granularity isolation, schema freezing and eight-arm CLI integration."""

import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from detector import FEATURE_COLUMNS, STAGE_GRAD_FEATURE_COLUMNS
from detector import transfer_detector as detection
from detector.common import sha256_file
from detector.io_utils import DETECTOR_ROOT
from detector.tests.test_transfer import feature_fixture, write_table


def fixture(task=1, seed=3):
    table, meta = feature_fixture(task)
    # Two modules per stage, each with weight and bias tensors. Fine norms
    # recombine exactly to the existing stage norms, then to the global norm.
    rng = np.random.default_rng(seed)
    extra = []
    for stage, stage_col in zip(
        ("stem", "layer1", "layer2", "layer3", "layer4"), STAGE_GRAD_FEATURE_COLUMNS
    ):
        parts = rng.uniform(0.1, 1, size=(len(table), 4))
        parts *= (
            table[stage_col].to_numpy()[:, None]
            / np.linalg.norm(parts, axis=1)[:, None]
        )
        for module in range(2):
            name = (
                ("conv1" if module == 0 else "bn1")
                if stage == "stem"
                else stage + ".0." + ("conv1" if module == 0 else "bn1")
            )
            for index, kind in enumerate(("weight", "bias")):
                col = "grad_norm_param__" + name + "." + kind
                table[col] = parts[:, 2 * module + index]
                extra.append(col)
            col = "grad_norm_layer__" + name
            table[col] = np.linalg.norm(parts[:, 2 * module : 2 * module + 2], axis=1)
            extra.append(col)
    table["grad_norm_l2"] = np.linalg.norm(table[STAGE_GRAD_FEATURE_COLUMNS], axis=1)
    meta["feature_columns"] = list(FEATURE_COLUMNS) + extra
    meta["checkpoint_seed"] = seed
    identity = (format(seed, "x") + str(task)) * 32
    meta["checkpoint_sha256"] = identity
    meta["input_sha256"]["attack"] = identity
    return table, meta


class GranularityTests(unittest.TestCase):
    def test_source_schema_selects_exact_norms_and_constant_context(self):
        table, meta = fixture()
        counts = dict(global_=1, stage=5, layer=10, parameter=20)
        for level in ("global", "stage", "layer", "parameter"):
            name = "norm_" + level
            base = detection.resolve_feature_columns(table, meta, name)
            context = detection.resolve_feature_columns(table, meta, name + "_context")
            self.assertEqual(
                len(base), counts["global_" if level == "global" else level]
            )
            self.assertEqual(context, base + detection.CONTEXT_COLUMNS)
            reordered = detection.resolve_feature_columns(
                table[table.columns[::-1]], meta, name
            )
            self.assertEqual(base, reordered)
        self.assertEqual(list(detection.FEATURE_GROUPS), ["portable", "extended"])

    def test_missing_schema_or_head_cannot_silently_change_comparison(self):
        table, meta = fixture()
        for prefix, name in (
            ("grad_norm_param__", "norm_parameter"),
            ("grad_norm_layer__", "norm_layer"),
        ):
            with self.assertRaisesRegex(ValueError, "schema"):
                detection.resolve_feature_columns(
                    table.drop(columns=[c for c in table if c.startswith(prefix)]),
                    meta,
                    name,
                )
            changed = copy.deepcopy(meta)
            changed["feature_columns"] = [
                c for c in changed["feature_columns"] if not c.startswith(prefix)
            ]
            with self.assertRaisesRegex(ValueError, "schema"):
                detection.resolve_feature_columns(table, changed, name)
        table["grad_norm_param__heads.1.weight"] = 1.0
        meta["feature_columns"].append("grad_norm_param__heads.1.weight")
        with self.assertRaisesRegex(ValueError, "backbone"):
            detection.resolve_feature_columns(table, meta, "norm_parameter")

    def test_heldout_rows_and_random_controls_cannot_change_frozen_fit(self):
        table, meta = fixture()
        columns = [
            c for c in table if c.startswith("grad_norm_")
        ] + detection.CONTEXT_COLUMNS
        changed = table.copy()
        changed.loc[
            (table.split == "test") | (table.view == "random_control"), columns
        ] *= 100
        for name in detection.GRANULARITY_GROUPS:
            first = detection.fit_source(table, meta, feature_set=name, task_size=16)
            second = detection.fit_source(changed, meta, feature_set=name, task_size=16)
            np.testing.assert_array_equal(
                first["classifier"].coef_, second["classifier"].coef_
            )
            self.assertEqual(first["threshold"], second["threshold"])
            self.assertEqual(first["count_calibration"], second["count_calibration"])

    def test_missing_target_parameter_is_rejected_without_changing_bundle(self):
        source, meta = fixture()
        target, target_meta = fixture(9)
        bundle = detection.fit_source(
            source, meta, feature_set="norm_parameter", task_size=16
        )
        old = bundle["classifier"].coef_.copy()
        target_meta["feature_columns"].remove(bundle["feature_columns"][0])
        with self.assertRaisesRegex(ValueError, "missing frozen"):
            detection.evaluate(target, target_meta, bundle, repeats=1)
        np.testing.assert_array_equal(old, bundle["classifier"].coef_)

    def test_cli_all_arms_reports_shap_deltas_and_preserves_inputs(self):
        with tempfile.TemporaryDirectory(dir=DETECTOR_ROOT / "tests") as directory:
            root = Path(directory)
            for seed in (3, 4):
                for task in (1, 9):
                    folder = root / "source" / f"seed{seed}" / f"features_task{task}"
                    folder.mkdir(parents=True)
                    table, meta = fixture(task, seed)
                    write_table(table, meta, folder / "features.csv")
            hashes = {
                p: sha256_file(p) for p in (root / "source").rglob("*") if p.is_file()
            }
            output = root / "result"
            command = [
                sys.executable,
                "-B",
                "-m",
                "detector.granularity_study",
                "--source-run",
                str(root / "source"),
                "--output-dir",
                str(output),
                "--task-size",
                "16",
                "--bags-per-rate",
                "1",
            ]
            for extra in (["--dry-run"], []):
                result = subprocess.run(
                    command + extra,
                    cwd=DETECTOR_ROOT.parent,
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                if extra:
                    self.assertFalse(output.exists())
            state = json.loads((output / "run_state.json").read_text())
            self.assertEqual(state["status"], "complete")
            self.assertEqual(
                state["steps"]["write_granularity_comparison"]["status"], "complete"
            )
            runs = json.loads((output / "summary.json").read_text())["runs"]
            self.assertEqual(len(runs), 32)
            report = json.loads((output / "granularity_summary.json").read_text())
            self.assertTrue(report["synthetic"])
            paired = next(
                r
                for r in report["paired_differences_vs_stage"]
                if r["task"] == 9
                and r["feature_set"] == "norm_parameter"
                and r["metric"] == "roc_auc"
            )
            expected = []
            for seed in (3, 4):
                values = {
                    r["raw_feature_set"]: r["sample_metrics"]["roc_auc"]
                    for r in runs
                    if r["seed"] == seed and r["task"] == 9
                }
                expected.append(values["norm_parameter"] - values["norm_stage"])
            np.testing.assert_allclose(paired["values"], expected)
            self.assertAlmostEqual(paired["std"], np.std(expected, ddof=1))
            self.assertEqual(
                len(list(output.glob("seed*/explain_*/explanation.json"))), 16
            )
            self.assertIn(
                "SYNTHETIC TEST OUTPUT", (output / "professor_update.md").read_text()
            )
            self.assertTrue((output / "shap_summary.md").is_file())
            for p, checksum in hashes.items():
                self.assertEqual(sha256_file(p), checksum)
            result = subprocess.run(
                command, cwd=DETECTOR_ROOT.parent, capture_output=True, text=True
            )
            self.assertNotEqual(result.returncode, 0)


if __name__ == "__main__":
    unittest.main()
