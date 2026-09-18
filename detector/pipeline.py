"""Reproducible orchestration; every output stays beneath detector/."""

import argparse
from datetime import datetime, timezone
import json
import importlib.metadata
import os
from pathlib import Path
import shlex
import subprocess
import sys

from detector.common import save_json, sha256_file
from detector.io_utils import DETECTOR_ROOT, ensure_output_path

DEFAULTS = {
    "device": "cuda",
    "feature_set": "extended",
    "split_seed": 20260720,
    "head_seed": 20260720,
    "feature_seed": 20260721,
    "random_control_seed": 20260722,
    "task_size": 150,
    "bags_per_rate": 1000,
    "unsupervised_tasks_per_rate": 100,
    "permutations": 199,
    "alpha": 0.05,
    "target_clean_fpr": 0.05,
}
PATH_FIELDS = ("checkpoint", "artifact", "inversion_dir", "data_cwd", "run_dir")


def load_config(path):
    path = Path(path).resolve()
    with path.open(encoding="utf-8") as source:
        values = json.load(source)
    unknown = set(values) - set(DEFAULTS) - set(PATH_FIELDS)
    if unknown:
        raise ValueError("Unknown configuration fields: {}".format(sorted(unknown)))
    missing = set(PATH_FIELDS) - set(values)
    if missing:
        raise ValueError("Missing configuration fields: {}".format(sorted(missing)))
    config = dict(DEFAULTS, **values)
    for key in PATH_FIELDS:
        value = Path(config[key]).expanduser()
        config[key] = str((path.parent / value).resolve())
    ensure_output_path(config["run_dir"])
    if config["feature_set"] not in {
        "baseline",
        "stages",
        "extended",
        "parameters",
        "layers",
        "all",
    }:
        raise ValueError("Unsupported feature_set.")
    if (
        isinstance(config["task_size"], bool)
        or not isinstance(config["task_size"], int)
        or not 5 <= config["task_size"] <= 500
    ):
        raise ValueError(
            "task_size must be an integer in 5..500 for the default split and 10% contamination rate."
        )
    for key in ("bags_per_rate", "unsupervised_tasks_per_rate", "permutations"):
        if (
            isinstance(config[key], bool)
            or not isinstance(config[key], int)
            or config[key] <= 0
        ):
            raise ValueError("{} must be a positive integer.".format(key))
    for key in ("alpha", "target_clean_fpr"):
        if not 0 < config[key] < 1:
            raise ValueError("{} must lie in (0, 1).".format(key))
    return config


def commands(config):
    """Return a readable, inspectable plan without executing or creating files."""
    root = Path(config["run_dir"])
    features = root / "features.csv"
    predicted = root / "predicted_features.csv"
    reference = root / "reference_features.csv"
    manifest = root / "manifest.csv"
    sample = root / "sample" / "detector_bundle.joblib"
    dataset = root / "dataset" / "dataset_bundle.joblib"
    unsupervised = root / "unsupervised" / "unsupervised_bundle.joblib"
    extract = [
        "--checkpoint",
        config["checkpoint"],
        "--inversion-dir",
        config["inversion_dir"],
        "--device",
        config["device"],
        "--head-seed",
        config["head_seed"],
        "--feature-seed",
        config["feature_seed"],
        "--random-control-seed",
        config["random_control_seed"],
    ]
    paired = ["--artifact", config["artifact"], "--manifest", manifest]
    definitions = [
        (
            "prepare",
            "manifest",
            "create_manifest",
            [
                "--checkpoint",
                config["checkpoint"],
                "--output",
                manifest,
                "--split-seed",
                config["split_seed"],
            ],
        ),
        (
            "prepare",
            "features",
            "extract_features",
            extract + paired + ["--output", features],
        ),
        (
            "prepare",
            "reference_features",
            "extract_features",
            extract
            + ["--reference-only", "--label-mode", "predicted", "--output", reference],
        ),
        (
            "prepare",
            "predicted_features",
            "extract_features",
            extract + paired + ["--label-mode", "predicted", "--output", predicted],
        ),
        (
            "prepare",
            "analysis",
            "train_detector",
            [
                "--features",
                features,
                "--output-dir",
                root / "analysis",
                "--compare-features",
                "--target-clean-fpr",
                config["target_clean_fpr"],
            ],
        ),
        (
            "fit",
            "sample_fit",
            "train_detector",
            [
                "--features",
                features,
                "--output-dir",
                root / "sample",
                "--feature-set",
                config["feature_set"],
                "--target-clean-fpr",
                config["target_clean_fpr"],
                "--fit-only",
            ],
        ),
        (
            "fit",
            "dataset_fit",
            "dataset_detector",
            [
                "fit",
                "--features",
                features,
                "--sample-bundle",
                sample,
                "--output-dir",
                root / "dataset",
                "--task-size",
                config["task_size"],
                "--bags-per-rate",
                config["bags_per_rate"],
                "--target-clean-frr",
                config["target_clean_fpr"],
            ],
        ),
        (
            "fit",
            "unsupervised_fit",
            "unsupervised",
            [
                "fit",
                "--reference-features",
                reference,
                "--output-dir",
                root / "unsupervised",
                "--alpha",
                config["alpha"],
                "--permutations",
                config["permutations"],
            ],
        ),
        (
            "evaluate",
            "sample_evaluate",
            "train_detector",
            [
                "--features",
                features,
                "--evaluate-bundle",
                sample,
                "--output-dir",
                root / "sample_test",
            ],
        ),
        (
            "evaluate",
            "dataset_evaluate",
            "dataset_detector",
            [
                "evaluate",
                "--features",
                features,
                "--bundle",
                dataset,
                "--output-dir",
                root / "dataset_test",
            ],
        ),
        (
            "evaluate",
            "unsupervised_evaluate",
            "unsupervised",
            [
                "evaluate",
                "--features",
                predicted,
                "--bundle",
                unsupervised,
                "--output",
                root / "unsupervised_test.json",
                "--task-size",
                config["task_size"],
                "--tasks-per-rate",
                config["unsupervised_tasks_per_rate"],
            ],
        ),
        ("report", "report", "report", ["--run-dir", root]),
    ]
    return [
        (
            group,
            name,
            [sys.executable, "-B", "-u", "-m", "detector." + module]
            + [str(arg) for arg in args],
        )
        for group, name, module, args in definitions
    ]


def child_environment(run_dir):
    run_dir = ensure_output_path(run_dir)
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONPATH"] = (
        str(DETECTOR_ROOT.parent) + os.pathsep + env.get("PYTHONPATH", "")
    )
    env["MPLCONFIGDIR"] = str(run_dir / ".matplotlib")
    env["XDG_CACHE_HOME"] = str(run_dir / ".cache")
    # Small CPU matrix products are slower with dozens of BLAS threads.
    env.setdefault("OPENBLAS_NUM_THREADS", "1")
    env.setdefault("OMP_NUM_THREADS", "1")
    return env


def environment_record():
    """Record code and installed versions so a resumed run cannot silently drift."""
    versions = {}
    for name in (
        "numpy",
        "pandas",
        "scipy",
        "scikit-learn",
        "torch",
        "torchvision",
        "joblib",
    ):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    sources = {
        path.name: sha256_file(path) for path in sorted(DETECTOR_ROOT.glob("*.py"))
    }
    return {
        "python": sys.version,
        "versions": versions,
        "detector_source_sha256": sources,
    }


def execute(command, *, cwd, log_path, env):
    log_path = ensure_output_path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print("\n$ " + shlex.join(command), flush=True)
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        try:
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
                log.flush()
            return_code = process.wait()
        finally:
            process.stdout.close()
            if process.poll() is None:
                process.terminate()
                process.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, command)


def main(args):
    config = load_config(args.config)
    plan = [
        item
        for item in commands(config)
        if args.stages == "all" or item[0] == args.stages
    ]
    if args.dry_run:
        print("Working directory:", config["data_cwd"])
        for _, name, command in plan:
            print("\n[{}]\n{}".format(name, shlex.join(command)))
        return
    root = ensure_output_path(config["run_dir"])
    for key in ("checkpoint", "artifact", "inversion_dir", "data_cwd"):
        if not Path(config[key]).exists():
            raise FileNotFoundError("{}: {}".format(key, config[key]))
    root.mkdir(parents=True, exist_ok=True)
    environment_path = root / "run_environment.json"
    environment = environment_record()
    if environment_path.exists():
        with environment_path.open(encoding="utf-8") as source:
            if json.load(source) != environment:
                raise ValueError(
                    "Detector code or dependency versions changed. Use a new run_dir."
                )
    else:
        save_json(environment, environment_path)
    frozen_config = root / "run_config.json"
    if frozen_config.exists():
        with frozen_config.open(encoding="utf-8") as source:
            if json.load(source) != config:
                raise ValueError(
                    "This run has a different frozen configuration. Choose a new run_dir."
                )
    else:
        save_json(config, frozen_config)
    state_path = root / "run_state.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {}
    env = child_environment(root)
    for _, name, command in plan:
        if state.get(name, {}).get("status") == "complete":
            print("Already complete (not rerunning):", name, flush=True)
            continue
        state[name] = {
            "status": "running",
            "command": command,
            "started_utc": datetime.now(timezone.utc).isoformat(),
        }
        save_json(state, state_path)
        try:
            execute(
                command,
                cwd=config["data_cwd"],
                log_path=root / "logs" / (name + ".log"),
                env=env,
            )
        except BaseException:
            state[name]["status"] = "failed"
            save_json(state, state_path)
            raise
        state[name]["status"] = "complete"
        save_json(state, state_path)
    print("\nRun directory:", root)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        required=True,
        help="JSON; relative paths resolve against its directory.",
    )
    parser.add_argument(
        "--stages",
        choices=("all", "prepare", "fit", "evaluate", "report"),
        default="all",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands only; do not touch files.",
    )
    main(parser.parse_args())
