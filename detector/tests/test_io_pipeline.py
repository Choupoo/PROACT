"""Output confinement, artifact compatibility, and orchestration regressions."""

import argparse
import io
import json
from pathlib import Path
import pickle
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch

from detector import bootstrap
from detector.common import _CPUUnpickler, load_pickle
from detector.demo import make_fixtures
from detector.extract_features import validate_loaded_model_keys
from detector.io_utils import (
    DETECTOR_ROOT,
    ensure_output_path,
    load_frozen_bundle,
    read_feature_table,
    save_frozen_bundle,
)
from detector.pipeline import commands, load_config
from detector.report import build_report
from test_features import make_model


class IOTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(dir=Path(__file__).parent)
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def test_all_outputs_are_confined_including_symlinked_children(self):
        self.assertEqual(
            ensure_output_path(self.root / "nested" / "result.json"),
            self.root / "nested" / "result.json",
        )
        with self.assertRaises(ValueError):
            ensure_output_path(DETECTOR_ROOT.parent / "not_allowed.json")
        link = self.root / "escaped"
        link.symlink_to(DETECTOR_ROOT.parent, target_is_directory=True)
        with self.assertRaises(ValueError):
            ensure_output_path(link / "not_allowed.json")
        with self.assertRaises(ValueError):
            ensure_output_path(self.root)

    def test_csv_hash_and_sidecar_required(self):
        make_fixtures(self.root)
        path = self.root / "features.csv"
        table, metadata = read_feature_table(path)
        self.assertEqual(len(table), 720)
        self.assertTrue(metadata["synthetic"])
        with path.open("a", encoding="utf-8") as output:
            output.write("\n")
        with self.assertRaisesRegex(ValueError, "hash"):
            read_feature_table(path)

    def test_frozen_bundle_refuses_overwrite_and_corruption(self):
        path = save_frozen_bundle({"sentinel": 42}, self.root)
        self.assertEqual(load_frozen_bundle(path), {"sentinel": 42})
        with self.assertRaises(FileExistsError):
            save_frozen_bundle({"sentinel": 43}, self.root)
        with path.open("ab") as output:
            output.write(b"corruption")
        with self.assertRaisesRegex(ValueError, "checksum"):
            load_frozen_bundle(path)

    def test_trusted_tensor_pickle_remaps_storage_to_cpu(self):
        path = self.root / "tensor.pkl"
        with path.open("wb") as target:
            pickle.dump({"tensor": torch.arange(4)}, target)
        loaded = load_pickle(path)
        self.assertEqual(loaded["tensor"].device.type, "cpu")
        torch.testing.assert_close(loaded["tensor"], torch.arange(4))
        loader = _CPUUnpickler(io.BytesIO()).find_class(
            "torch.storage", "_load_from_bytes"
        )
        with mock.patch(
            "detector.common.torch.load", return_value="cpu_storage"
        ) as remap:
            self.assertEqual(loader(b"storage"), "cpu_storage")
        self.assertEqual(remap.call_args.kwargs["map_location"], "cpu")

    def test_actual_ewc_fisher_buffer_naming_is_supported(self):
        model = make_model()
        state = dict(model.state_dict())
        for name, parameter in model.named_parameters():
            state[name.replace(".", "_") + "_fisher"] = torch.ones_like(parameter)
        validate_loaded_model_keys(model, state)
        with self.assertRaisesRegex(RuntimeError, "Unexpected"):
            validate_loaded_model_keys(model, dict(state, unrelated=torch.zeros(1)))
        state["conv1_weight_fisher"] = torch.tensor([-1.0])
        with self.assertRaisesRegex(RuntimeError, "Fisher"):
            validate_loaded_model_keys(model, state)
        state = dict(model.state_dict())
        del state["conv1.weight"]
        with self.assertRaisesRegex(RuntimeError, "backbone"):
            validate_loaded_model_keys(model, state)

    def test_pipeline_resolves_config_paths_and_freezes_before_test(self):
        config = load_config(DETECTOR_ROOT / "config.example.json")
        self.assertEqual(
            Path(config["checkpoint"]),
            DETECTOR_ROOT / "work/artifacts/victim/checkpoint.pkl",
        )
        plan = commands(config)
        names = [name for _, name, _ in plan]
        self.assertLess(names.index("dataset_fit"), names.index("dataset_evaluate"))
        self.assertLess(
            names.index("unsupervised_fit"), names.index("unsupervised_evaluate")
        )
        sample_fit = next(command for _, name, command in plan if name == "sample_fit")
        self.assertIn("--fit-only", sample_fit)
        sample_test = next(
            command for _, name, command in plan if name == "sample_evaluate"
        )
        self.assertIn("--evaluate-bundle", sample_test)
        for invalid in (1, 4, 5.5, True):
            path = self.root / "config.json"
            path.write_text(
                json.dumps(dict(config, task_size=invalid)), encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "task_size"):
                load_config(path)

    def test_bootstrap_uses_basename_for_upstream_inversion_and_scoped_cwd(self):
        args = argparse.Namespace(
            output_dir=str(self.root),
            victim_epochs=20,
            inversion_iters=10000,
            attack_epochs=5000,
            delta=0.3,
            seed=0,
        )
        plan = bootstrap.build_commands(args)
        inversion = next(command for name, command, _ in plan if name == "inversion")
        self.assertEqual(inversion[inversion.index("--save_dir") + 1], "inversions")
        for _, _, output in plan:
            self.assertIn(self.root, output.parents)

    def test_effectiveness_formula_and_missing_report_results(self):
        clean = np.zeros((10, 10))
        np.fill_diagonal(clean, 0.8)
        clean[9, :9] = 0.7
        poison = clean.copy()
        poison[9, :9] = 0.4
        np.save(self.root / "clean.npy", clean)
        np.save(self.root / "poison.npy", poison)
        report = bootstrap.attack_effectiveness(
            self.root / "clean.npy", self.root / "poison.npy"
        )
        self.assertAlmostEqual(report["clean"]["backward_transfer"], -0.1)
        self.assertAlmostEqual(report["past_accuracy_drop_clean_minus_poison"], 0.3)
        (self.root / "run_config.json").write_text(
            '{"synthetic":true}', encoding="utf-8"
        )
        text = build_report(self.root)
        self.assertIn("人工合成特征", text)
        self.assertIn("尚未", text)
        self.assertIn("p-value 不是投毒概率", text)


if __name__ == "__main__":
    unittest.main()
