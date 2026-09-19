"""CPU integration demo using SYNTHETIC descriptors, never scientific results."""

import argparse
import hashlib

import numpy as np
import pandas as pd

from detector import (
    FEATURE_COLUMNS,
    FEATURE_PROTOCOL,
    HEAD_MODE,
    STAGE_GRAD_FEATURE_COLUMNS,
)
from detector.common import save_json, sha256_file
from detector.io_utils import DETECTOR_ROOT, ensure_output_path
from detector.pipeline import child_environment, commands, execute


def synthetic_descriptor(rng, shift=0.0):
    """Deliberately easy analytic descriptors; these are not extracted images."""
    stages = np.exp(rng.normal(-0.5 + shift, 0.2, 5))
    row = {
        name: float(value) for name, value in zip(STAGE_GRAD_FEATURE_COLUMNS, stages)
    }
    row.update(
        {
            "loss": float(np.exp(rng.normal(1 + shift, 0.2))),
            "grad_norm_l2": float(np.linalg.norm(stages)),
            "grad_cosine_past": float(np.tanh(rng.normal(0.2 - shift, 0.1))),
            "entropy": float(1.5 + 0.5 * np.tanh(rng.normal(shift, 0.1))),
            "confidence": float(0.5 + 0.2 * np.tanh(rng.normal(-shift, 0.1))),
            "true_class_probability": float(
                0.4 + 0.1 * np.tanh(rng.normal(-shift, 0.1))
            ),
            "margin": float(0.2 + 0.1 * np.tanh(rng.normal(-shift, 0.1))),
            "activation_norm_l2": float(np.exp(rng.normal(1 + shift, 0.2))),
        }
    )
    cosines = np.tanh(rng.normal(0.2 - shift, 0.1, 9))
    row.update(
        {
            "grad_cosine_task_{}".format(i): float(value)
            for i, value in enumerate(cosines)
        }
    )
    for name, value in (
        ("min", cosines.min()),
        ("max", cosines.max()),
        ("mean", cosines.mean()),
    ):
        row["grad_cosine_task_" + name] = float(value)
    for index, value in enumerate(stages):
        row["grad_norm_param__synthetic_layer{}.weight".format(index)] = float(value)
        row["grad_norm_layer__synthetic_layer{}".format(index)] = float(value)
    return row


def write_fixture(table, path, label_mode, origin_role):
    provenance = {
        "feature_protocol": FEATURE_PROTOCOL,
        "head_mode": HEAD_MODE,
        "head_seed": 20260720,
        "label_mode": label_mode,
        "checkpoint_sha256": hashlib.sha256(b"SYNTHETIC_NO_CHECKPOINT").hexdigest(),
        "inversion_sha256": hashlib.sha256(b"SYNTHETIC_NO_INVERSION").hexdigest(),
    }
    for key, value in provenance.items():
        table[key] = value
    table.to_csv(path, index=False)
    columns = list(synthetic_descriptor(np.random.default_rng(0)))
    assert set(FEATURE_COLUMNS).issubset(columns)
    save_json(
        dict(
            provenance,
            synthetic=True,
            origin_role=origin_role,
            feature_columns=columns,
            features_sha256=sha256_file(path),
            data_generation="Analytic CPU integration fixture; no real model or image dataset.",
        ),
        path.with_suffix(".metadata.json"),
    )


def make_fixtures(root):
    rng = np.random.default_rng(72)
    rows = []
    original = 0
    for split, count in (
        ("train", 80),
        ("validation", 40),
        ("test", 40),
        ("reserve", 80),
    ):
        for _ in range(count):
            for view, label, shift in (
                ("clean", 0, 0.0),
                ("poison", 1, 0.8),
                ("random_control", -1, 0.15),
            ):
                row = synthetic_descriptor(rng, shift)
                row.update(
                    original_index=original,
                    source_index=original,
                    class_id=original % 10,
                    split=split,
                    view=view,
                    detector_label=label,
                    attack_seed=0,
                )
                rows.append(row)
            original += 1
    paired = pd.DataFrame(rows)
    write_fixture(
        paired.copy(), root / "features.csv", "ground_truth", "paired_benchmark"
    )
    write_fixture(
        paired.copy(), root / "predicted_features.csv", "predicted", "paired_benchmark"
    )
    reference_rows = []
    for index in range(120):
        row = synthetic_descriptor(rng)
        row.update(reference_id="synthetic:{}".format(index), original_index=index)
        reference_rows.append(row)
    write_fixture(
        pd.DataFrame(reference_rows),
        root / "reference_features.csv",
        "predicted",
        "historical_inversion",
    )


def run_demo(output_dir):
    root = ensure_output_path(output_dir)
    if root.exists() and any(root.iterdir()):
        raise FileExistsError("Choose an empty demo output directory: {}".format(root))
    root.mkdir(parents=True, exist_ok=True)
    make_fixtures(root)
    config = {
        "synthetic": True,
        "run_dir": str(root),
        "checkpoint": "NOT_USED",
        "artifact": "NOT_USED",
        "inversion_dir": "NOT_USED",
        "data_cwd": str(DETECTOR_ROOT.parent),
        "device": "cpu",
        "feature_set": "extended",
        "split_seed": 20260720,
        "head_seed": 20260720,
        "feature_seed": 20260721,
        "random_control_seed": 20260722,
        "task_size": 20,
        "bags_per_rate": 20,
        "unsupervised_tasks_per_rate": 5,
        "permutations": 39,
        "alpha": 0.05,
        "target_clean_fpr": 0.05,
        "dataset_decision_rule": "count_bound",
    }
    save_json(config, root / "run_config.json")
    env = child_environment(root)
    for group, name, command in commands(config):
        if group == "prepare" and name != "analysis":
            continue
        execute(
            command,
            cwd=DETECTOR_ROOT.parent,
            log_path=root / "logs" / (name + ".log"),
            env=env,
        )
    # Exercise both standalone prediction CLIs with descriptor-only incoming
    # tables. No class, view or clean/poison labels cross this boundary.
    import sys

    paired = pd.read_csv(root / "features.csv")
    descriptors = list(synthetic_descriptor(np.random.default_rng(0)))
    incoming = paired.loc[
        (paired["split"] == "test") & (paired["view"] == "clean"),
        descriptors + ["original_index"],
    ].head(config["task_size"])
    for module, label_mode, filename in (
        ("dataset_detector", "ground_truth", "dataset/dataset_bundle.joblib"),
        ("unsupervised", "predicted", "unsupervised/unsupervised_bundle.joblib"),
    ):
        path = root / (module + "_incoming.csv")
        write_fixture(incoming.copy(), path, label_mode, "incoming")
        command = [
            sys.executable,
            "-B",
            "-u",
            "-m",
            "detector." + module,
            "predict",
            "--features",
            str(path),
            "--bundle",
            str(root / filename),
        ]
        if module == "unsupervised":
            command += ["--output", str(root / "unsupervised_prediction.json")]
        else:
            command += ["--output-dir", str(root / "dataset_prediction")]
        execute(
            command,
            cwd=DETECTOR_ROOT.parent,
            log_path=root / "logs" / (module + "_predict.log"),
            env=env,
        )
    print("\nCPU integration demo completed. SYNTHETIC ONLY:", root / "report.md")
    return root


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", default=str(DETECTOR_ROOT / "work" / "cpu_demo")
    )
    run_demo(parser.parse_args().output_dir)
