"""Separate, label-free adaptation experiment; never refit supervised models.

run freezes a rank-reference bundle BEFORE reading the benchmark CSV, then
compares it with the unchanged historical MMD bundle on identical simulated
bags. Labels exist only inside the offline evaluation function. Old artifacts
are read-only; outputs require a new sibling directory within detector/.
"""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np

from detector.common import save_json, sha256_file
from detector.io_utils import (
    assert_compatible_provenance,
    ensure_output_path,
    read_feature_table,
    save_frozen_bundle,
    load_frozen_bundle,
)
from detector.pipeline import environment_record
from detector import rank_reference as rank
from detector import unsupervised as legacy

RATES = (0.01, 0.05, 0.1, 0.25, 0.5, 1.0)


def evaluate_benchmark(
    bundle,
    baseline,
    features,
    metadata,
    *,
    task_size=150,
    tasks_per_rate=100,
    seed=20260921,
    progress=False,
):
    """Offline ground truth assembles bags; predictors get descriptor columns only."""
    rank.validate_bundle(bundle)
    legacy._validate_bundle(baseline)
    for fitted in (bundle, baseline):
        assert_compatible_provenance(
            fitted["provenance"], legacy._validate_metadata(metadata)
        )
    if metadata.get("origin_role") != "paired_benchmark":
        raise ValueError("Evaluation requires paired_benchmark metadata.")
    task_size = rank.integer(task_size, "task_size", rank.MIN_SAMPLES)
    if task_size > min(
        bundle["settings"]["max_incoming"], baseline["settings"]["max_incoming"]
    ):
        raise ValueError(
            "Increase both detector caps explicitly; comparison must test whole bags."
        )
    tasks_per_rate = rank.integer(tasks_per_rate, "tasks_per_rate", 1)
    seed = rank.integer(seed, "seed", 0)
    required = ["original_index", "split", "view"]
    if not set(required).issubset(features) or features[required].isna().any().any():
        raise ValueError("Missing benchmark identities, splits or views.")
    if (features.groupby("original_index")["split"].nunique() != 1).any():
        raise ValueError("Original images overlap benchmark splits.")
    test = features.loc[features["split"] == "test"].copy()
    legacy._descriptor_values(test)
    expected = {"clean", "poison", "random_control"}
    if (
        set(test["view"]) != expected
        or test.duplicated(["original_index", "view"]).any()
    ):
        raise ValueError(
            "Test requires one clean/poison/random_control view per original."
        )
    groups = test.groupby("original_index")["view"].nunique()
    if (groups != 3).any() or len(groups) < task_size:
        raise ValueError("Test triples incomplete or too few distinct originals.")
    ids = groups.index.to_numpy()
    lookup = test.set_index(["original_index", "view"])
    rng = np.random.default_rng(seed)
    summary, results, skipped = [], [], []
    scenarios = [("clean", 0.0)] + [
        (view, rate) for view in ("poison", "random_control") for rate in RATES
    ]
    for view, rate in scenarios:
        modified = int(np.floor(rate * task_size + 0.5))
        if rate > 0 and modified == 0:
            skipped.append(
                {
                    "scenario": view,
                    "requested_rate": rate,
                    "reason": "Rounds to zero modified images at this task size; not an attack experiment.",
                }
            )
            continue
        start = len(results)
        for repeat in range(tasks_per_rate):
            selected = rng.choice(ids, task_size, replace=False)
            keys = [
                (identity, view if i < modified else "clean")
                for i, identity in enumerate(selected)
            ]
            # No view/class/poison labels cross this boundary.
            bag = lookup.loc[keys, legacy.FEATURE_COLUMNS].reset_index(drop=True)
            adapted = rank.predict_dataset(bundle, bag, metadata)
            original = legacy.predict_dataset(baseline, bag, metadata)
            results.append(
                {
                    "scenario": view,
                    "requested_rate": rate,
                    "modified_count": modified,
                    "bag_index": repeat,
                    "original_indices": selected.tolist(),
                    "rank_status": adapted["status"],
                    "rank_alert": adapted["shift_detected"],
                    "rank_p_value_approx": adapted["p_value_approx"],
                    "rank_comparisons": adapted["comparisons"],
                    "max_feature_tie_fraction": adapted["max_feature_tie_fraction"],
                    "legacy_alert": original["shift_detected"],
                    "legacy_p_value": original["p_value"],
                }
            )
        rows = results[start:]
        supported = [row for row in rows if row["rank_status"] == "ok"]
        alerts = sum(bool(row["rank_alert"]) for row in supported)
        summary.append(
            {
                "scenario": view,
                "requested_rate": rate,
                "realized_rate": modified / task_size,
                "modified_count": modified,
                "bags": tasks_per_rate,
                "rank_supported_bags": len(supported),
                "rank_coverage": len(supported) / tasks_per_rate,
                "rank_alerts": alerts,
                "rank_alert_rate": alerts / tasks_per_rate
                if len(supported) == tasks_per_rate
                else None,
                "rank_alert_rate_among_supported": alerts / len(supported)
                if supported
                else None,
                "legacy_alert_rate": float(
                    np.mean([row["legacy_alert"] for row in rows])
                ),
            }
        )
        if progress:
            print(
                "{} rate={:.3f}: rank={}, coverage={:.1%}, old MMD={:.1%}".format(
                    view,
                    modified / task_size,
                    summary[-1]["rank_alert_rate"],
                    summary[-1]["rank_coverage"],
                    summary[-1]["legacy_alert_rate"],
                ),
                flush=True,
            )
    return {
        "method": rank.METHOD,
        "baseline_method": legacy.METHOD,
        "evaluation_split": "test",
        "task_size": task_size,
        "test_originals": len(ids),
        "evaluation_seed": seed,
        "summary": summary,
        "tasks": results,
        "skipped_scenarios": skipped,
        "rank_alpha": bundle["settings"]["alpha"],
        "legacy_alpha": baseline["settings"]["alpha"],
        "labels_used_for": "Offline bag assembly and reporting only, never fitting, feature selection, adaptation or thresholds.",
        "threshold_tuned_on_evaluation": False,
        "same_bags_for_both_methods": True,
        "poisoning_probability_available": False,
        "reused_test_after_method_revision": True,
        "sampling_note": "Each bag has distinct originals; bags reuse a finite pool and are not independent experiments.",
        "limitations": list(rank.LIMITATIONS),
    }


def write_report(result, audit, bundle, output, synthetic=False):
    def percent(value):
        return "unavailable" if value is None else "{:.1%}".format(value)

    lines = ["# Strict label-free historical adaptation experiment", ""]
    if synthetic:
        lines += ["**SYNTHETIC FIXTURE ONLY: not real attack performance.**", ""]
    lines += [
        "This is a new experimental rank-dependence test, not a validated poisoning detector.",
        "No supervised model/result is changed. No trusted clean incoming sample or incoming attack/class label is used for fitting or prediction.",
        "This revision follows observed test results. Reused test performance is exploratory, not independent confirmation.",
        "",
        "## Method",
        "",
        "Compare Kendall tau-a feature pairs against each historical task separately. Gaussian jackknife multipliers approximate the centered difference distribution. The union-null p-value is the maximum across tasks: alert only if ALL references disagree.",
        "The p-values are approximate and require nondegenerate, independent observations. They are NOT exact permutation p-values or deployment poisoning probabilities.",
        "Reference tasks: {}; retained pairs: {}/{}. Alpha: {}.".format(
            len(bundle["profiles"]),
            int(bundle["active_pairs"].sum()),
            len(rank.PAIR_NAMES),
            bundle["settings"]["alpha"],
        ),
        "",
        "## Same-bag comparison",
        "",
        "Clean alert rate is a false alert; poison alert rate describes sensitivity; random-control alert rate describes noise sensitivity. Unsupported bags are not silently counted as accepted clean.",
        "",
        "| Scenario | Modified fraction | Old raw MMD alert | New rank alert | New coverage |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in result["summary"]:
        lines.append(
            "| {} | {} | {} | {} | {} |".format(
                row["scenario"],
                percent(row["realized_rate"]),
                percent(row["legacy_alert_rate"]),
                percent(row["rank_alert_rate"]),
                percent(row["rank_coverage"]),
            )
        )
    lines += [
        "",
        "## Historical task transfer audit",
        "",
        "Leave-task-out alert rate: {}. This is historical-only and never tunes settings; it does not establish inversion-to-real transfer.".format(
            percent(audit.get("leave_task_out_alert_rate"))
        ),
        "",
        "## Interpret the result",
        "",
        "Lower clean alerts count as progress only alongside retained poison sensitivity. Low alerts for BOTH clean and poisoned data indicate blindness, not success. High clean alerts mean rank-dependence mismatch remains. Inspect random controls separately.",
        "Pure monotone marginal attacks are invisible by design. Higher-order changes with unchanged pairwise tau can also be missed. Even clean rank compatibility does not certify safety.",
        "Freeze the method before new model/attack/task seeds. Do not choose descriptors, alpha, tasks or another rule using this test report.",
        "",
        "## Sources and scope",
        "",
        "The bootstrap construction is motivated by [Chen (2018), Gaussian and bootstrap approximations for high-dimensional U-statistics](https://arxiv.org/abs/1610.00032). This taskwise two-sample application is an experimental implementation, not a claimed reproduction or a proof of poisoning-detection validity.",
        "",
    ]
    output.write_text("\n".join(lines), encoding="utf-8")


def run(args):
    source = Path(args.source_run).resolve()
    root = ensure_output_path(args.output_dir)
    if (
        root == source
        or root in source.parents
        or source in root.parents
        or root.exists()
    ):
        raise ValueError(
            "Use a NEW independent sibling output directory; do not overwrite old results."
        )
    reference_path = source / "reference_features.csv"
    incoming_path = source / "predicted_features.csv"
    baseline_path = source / "unsupervised" / "unsupervised_bundle.joblib"
    reference, metadata = read_feature_table(reference_path)
    baseline = legacy.load_bundle(baseline_path)
    assert_compatible_provenance(baseline["provenance"], metadata)
    if baseline.get("reference_features_sha256") != metadata["features_sha256"]:
        raise ValueError("Historical baseline belongs to different reference features.")
    if (
        not incoming_path.is_file()
        or not incoming_path.with_suffix(".metadata.json").is_file()
    ):
        raise FileNotFoundError(
            "Need complete predicted_features.csv and metadata on the server."
        )
    settings = dict(
        alpha=args.alpha,
        bootstrap_draws=args.bootstrap_draws,
        seed=args.seed,
        max_reference_per_task=args.max_reference_per_task,
        max_incoming=args.max_incoming,
    )
    # fit_reference has no incoming argument and has not read benchmark values.
    bundle = rank.fit_reference(reference, metadata, **settings)
    rank.integer(args.task_size, "task_size", rank.MIN_SAMPLES)
    rank.integer(args.tasks_per_rate, "tasks_per_rate", 1)
    if args.task_size > min(args.max_incoming, baseline["settings"]["max_incoming"]):
        raise ValueError(
            "task_size exceeds a detector cap; whole-bag comparison is required."
        )
    if args.dry_run:
        print(
            "Input reference verified; historical-only fit is feasible. No output written."
        )
        print(
            "Incoming benchmark was not read. Run without --dry-run to freeze and evaluate."
        )
        return
    hashes = {
        str(path.relative_to(source)): sha256_file(path)
        for path in (
            reference_path,
            reference_path.with_suffix(".metadata.json"),
            incoming_path,
            incoming_path.with_suffix(".metadata.json"),
            baseline_path,
            baseline_path.with_suffix(".joblib.sha256.json"),
        )
    }
    root.mkdir(parents=True)
    save_json(
        dict(
            settings,
            source_run=str(source),
            source_sha256=hashes,
            task_size=args.task_size,
            tasks_per_rate=args.tasks_per_rate,
            evaluation_seed=args.seed + 1,
            synthetic=bool(metadata.get("synthetic")),
            supervision="historical_only_no_incoming_clean_reference",
            revised_after_observed_test=True,
            supervised_artifacts_changed=False,
        ),
        root / "run_config.json",
    )
    save_json(environment_record(), root / "run_environment.json")
    state = {}

    def stage(name, action, json_output=None):
        state[name] = {
            "status": "running",
            "started_utc": datetime.now(timezone.utc).isoformat(),
        }
        save_json(state, root / "run_state.json")
        try:
            value = action()
            if json_output is not None:
                save_json(value, root / json_output)
        except BaseException:
            state[name]["status"] = "failed"
            save_json(state, root / "run_state.json")
            raise
        state[name]["status"] = "complete"
        save_json(state, root / "run_state.json")
        return value

    stage(
        "freeze_historical_rank_model",
        lambda: save_frozen_bundle(bundle, root / "rank", "rank_bundle.joblib"),
    )
    audit = stage(
        "historical_leave_task_out",
        lambda: rank.historical_audit(bundle),
        "historical_audit.json",
    )
    # Benchmark is loaded only AFTER the model is saved and historical audit is fixed.
    features, benchmark_meta = stage(
        "load_benchmark_after_freeze", lambda: read_feature_table(incoming_path)
    )
    result = stage(
        "same_bag_evaluation",
        lambda: evaluate_benchmark(
            bundle,
            baseline,
            features,
            benchmark_meta,
            task_size=args.task_size,
            tasks_per_rate=args.tasks_per_rate,
            seed=args.seed + 1,
            progress=True,
        ),
        "evaluation_metrics.json",
    )
    stage(
        "report",
        lambda: write_report(
            result, audit, bundle, root / "report.md", bool(metadata.get("synthetic"))
        ),
    )

    def verify_sources():
        if any(
            sha256_file(source / name) != expected for name, expected in hashes.items()
        ):
            raise RuntimeError(
                "Source artifacts changed during the run; discard this evaluation."
            )

    stage("verify_sources_unchanged", verify_sources)
    print("Experimental unsupervised comparison complete:", root, flush=True)
    print(
        "This does not establish poisoning detection or independent generalization.",
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    experiment = commands.add_parser(
        "run", help="Historical-only fit, frozen model, same-bag old/new evaluation."
    )
    experiment.add_argument("--source-run", required=True)
    experiment.add_argument("--output-dir", required=True)
    experiment.add_argument("--alpha", type=float, default=0.05)
    experiment.add_argument("--bootstrap-draws", type=int, default=499)
    experiment.add_argument("--seed", type=int, default=20260920)
    experiment.add_argument("--max-reference-per-task", type=int, default=256)
    experiment.add_argument("--max-incoming", type=int, default=256)
    experiment.add_argument("--task-size", type=int, default=150)
    experiment.add_argument("--tasks-per-rate", type=int, default=100)
    experiment.add_argument("--dry-run", action="store_true")
    predict = commands.add_parser(
        "predict", help="Apply frozen rank model to one unlabeled feature dataset."
    )
    predict.add_argument("--bundle", required=True)
    predict.add_argument("--features", required=True)
    predict.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.command == "run":
        run(args)
    else:
        output = ensure_output_path(args.output)
        if output.exists():
            raise FileExistsError("Choose a new prediction output.")
        table, metadata = read_feature_table(args.features)
        if metadata.get("origin_role") == "paired_benchmark":
            raise ValueError(
                "Use run for a paired benchmark, not whole-table deployment prediction."
            )
        result = rank.predict_dataset(load_frozen_bundle(args.bundle), table, metadata)
        save_json(result, output)
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
