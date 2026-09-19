"""Reassess existing features without retraining PROACT or altering old outputs.

This revision follows observed test results. Reusing the same split is explicitly
diagnostic, not a fresh independent confirmation experiment.
"""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

from detector.common import save_json, sha256_file
from detector.io_utils import (
    ensure_output_path,
    load_frozen_bundle,
    read_feature_table,
    assert_compatible_provenance,
)
from detector.pipeline import child_environment, environment_record, execute


def preflight(source):
    """Validate all source features BEFORE creating a new output directory."""
    source = Path(source).resolve()
    config = json.loads((source / "run_config.json").read_text())
    records = {}
    metadata = {}
    for name in ("features.csv", "predicted_features.csv", "reference_features.csv"):
        _, meta = read_feature_table(source / name)
        metadata[name] = meta
        records[name] = meta["features_sha256"]
    sample_path = source / "sample" / "detector_bundle.joblib"
    historical_path = source / "unsupervised" / "unsupervised_bundle.joblib"
    sample = load_frozen_bundle(sample_path)
    historical = load_frozen_bundle(historical_path)
    assert_compatible_provenance(sample["provenance"], metadata["features.csv"])
    assert_compatible_provenance(
        historical["provenance"], metadata["predicted_features.csv"]
    )
    assert_compatible_provenance(
        historical["provenance"], metadata["reference_features.csv"]
    )
    if sample.get("features_sha256") != metadata["features.csv"]["features_sha256"]:
        raise ValueError("Source sample bundle was fitted on different features.")
    for path in (sample_path, historical_path, source / "run_config.json"):
        records[str(path.relative_to(source))] = sha256_file(path)
    for key in (
        "task_size",
        "bags_per_rate",
        "unsupervised_tasks_per_rate",
        "target_clean_fpr",
    ):
        if key not in config:
            raise ValueError("Source config is missing {}.".format(key))
    return config, records


def build_plan(source, root, config):
    source, root = Path(source), Path(root)
    features = source / "features.csv"
    predicted = source / "predicted_features.csv"
    sample = source / "sample" / "detector_bundle.joblib"
    historical = source / "unsupervised" / "unsupervised_bundle.joblib"
    dataset = root / "dataset" / "dataset_bundle.joblib"
    definitions = [
        (
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
                "--decision-rule",
                "count_bound",
            ],
        ),
        (
            "reference_audit",
            "reference_audit",
            [
                "--bundle",
                historical,
                "--features",
                predicted,
                "--task-size",
                config["task_size"],
                "--output",
                root / "reference_audit.json",
            ],
        ),
        (
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
            "unsupervised_evaluate",
            "unsupervised",
            [
                "evaluate",
                "--features",
                predicted,
                "--bundle",
                historical,
                "--output",
                root / "unsupervised_test.json",
                "--task-size",
                config["task_size"],
                "--tasks-per-rate",
                config["unsupervised_tasks_per_rate"],
            ],
        ),
        ("report", "report", ["--run-dir", root]),
    ]
    return [
        (
            name,
            [sys.executable, "-B", "-u", "-m", "detector." + module]
            + [str(x) for x in argv],
        )
        for name, module, argv in definitions
    ]


def main(args):
    source = Path(args.source_run).resolve()
    root = ensure_output_path(args.output_dir)
    if root == source or source in root.parents or root in source.parents:
        raise ValueError(
            "Use an independent sibling output directory, not the source or its parent/child."
        )
    if root.exists():
        raise FileExistsError(
            "Choose a new output directory; old results are preserved: {}".format(root)
        )
    config, source_hashes = preflight(source)
    if args.dry_run:
        import shlex

        for name, command in build_plan(source, root, config):
            print("[{}] {}".format(name, shlex.join(command)))
        return
    root.mkdir(parents=True)
    revised = dict(
        config,
        run_dir=str(root),
        dataset_decision_rule="count_bound",
        revision_after_observed_test_results=True,
        evaluation_scope="post_result_diagnostic_not_independent_confirmation",
        source_run=str(source),
        source_sha256=source_hashes,
    )
    save_json(revised, root / "run_config.json")
    save_json(environment_record(), root / "run_environment.json")
    state = {}
    env = child_environment(root)
    for name, command in build_plan(source, root, config):
        state[name] = {
            "status": "running",
            "command": command,
            "started_utc": datetime.now(timezone.utc).isoformat(),
        }
        save_json(state, root / "run_state.json")
        try:
            execute(
                command, cwd=root, log_path=root / "logs" / (name + ".log"), env=env
            )
        except BaseException:
            state[name]["status"] = "failed"
            save_json(state, root / "run_state.json")
            raise
        state[name]["status"] = "complete"
        save_json(state, root / "run_state.json")
    print("Diagnostic reassessment complete:", root)
    print("These reused test results are exploratory, not independent confirmation.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dry-run", action="store_true")
    main(parser.parse_args())
