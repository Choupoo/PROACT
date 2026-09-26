"""Re-evaluate existing features with opt-in supervised/strict-unlabeled revisions.

No upstream training, inversion, feature re-extraction or target calibration.
All re-used test results are explicitly exploratory, not fresh confirmation.
"""

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from detector import local_reference, rank_reference, shape_reference, unsupervised
from detector import transfer_detector as supervised
from detector import unsupervised_adapt, unsupervised_study
from detector.common import save_json, sha256_file
from detector.io_utils import DETECTOR_ROOT, read_feature_table, save_frozen_bundle
from detector.pipeline import environment_record


def _inputs(paths):
    """Check complete feature artifacts without parsing target benchmark values."""
    hashes = {}
    for path in paths:
        for part in (path, path.with_suffix(".metadata.json")):
            if not part.is_file():
                raise FileNotFoundError(
                    "Need the full server feature artifact: " + str(part)
                )
            hashes[str(part)] = sha256_file(part)
    return hashes


class Run:
    def __init__(
        self,
        root,
        args,
        hashes,
        protocol="detector_revision_shape_v1",
        evaluation_role="exploratory_reused_test_after_revision",
    ):
        self.root, self.hashes = root, hashes
        self.state = {"status": "running", "steps": {}}
        root.mkdir(parents=True)
        save_json(
            {
                "protocol": protocol,
                "command": vars(args),
                "evaluation_role": evaluation_role,
                "target_used_for_fitting_or_calibration": False,
                "input_sha256": hashes,
                "source_code_sha256": {
                    p.name: sha256_file(p) for p in sorted(DETECTOR_ROOT.glob("*.py"))
                },
            },
            root / "run_config.json",
        )
        save_json(environment_record(), root / "run_environment.json")
        self.save()

    def save(self):
        save_json(self.state, self.root / "run_state.json")

    @contextmanager
    def stage(self, name):
        entry = {
            "status": "running",
            "started_utc": datetime.now(timezone.utc).isoformat(),
        }
        self.state["steps"][name] = entry
        self.save()
        print(name, flush=True)
        try:
            yield
        except BaseException as error:
            entry.update(status="failed", error=str(error))
            self.state["status"] = "failed"
            self.save()
            raise
        entry.update(
            status="complete", finished_utc=datetime.now(timezone.utc).isoformat()
        )
        self.save()

    def finish(self):
        with self.stage("verify_original_inputs_unchanged"):
            if any(sha256_file(p) != value for p, value in self.hashes.items()):
                raise RuntimeError("An input artifact changed during evaluation.")
        self.state["status"] = "complete"
        self.save()
        print("Saved:", self.root, flush=True)


def run_supervised(args, *, finalize=None, protocol="detector_revision_shape_v1"):
    source = Path(args.source_run).resolve()
    root = unsupervised_study.fresh_output(args.output_dir, source)
    if not args.seeds or len(set(args.seeds)) != len(args.seeds):
        raise ValueError("Specify distinct seeds.")
    for seed in args.seeds:
        rank_reference.integer(seed, "seed", 0)
    if not 1 <= args.source_task < args.target_task <= 9:
        raise ValueError("Need 1 <= source task < target task <= 9.")
    rank_reference.integer(args.task_size, "task_size", 2)
    rank_reference.integer(args.bags_per_rate, "bags_per_rate", 1)
    files = {
        (seed, task): source
        / ("seed" + str(seed))
        / ("features_task" + str(task))
        / "features.csv"
        for seed in args.seeds
        for task in (args.source_task, args.target_task)
    }
    hashes = _inputs(files.values())
    policies = getattr(args, "negative_policies", ["clean_only"])
    features = getattr(args, "feature_sets", ["portable", "extended", "shape"])
    if (
        not features
        or len(set(features)) != len(features)
        or not set(features).issubset(supervised.registered_feature_sets())
    ):
        raise ValueError("Specify distinct registered feature sets.")
    if (
        not policies
        or len(set(policies)) != len(policies)
        or not set(policies).issubset(supervised.NEGATIVE_POLICIES)
    ):
        raise ValueError("Specify distinct registered negative policies.")
    if args.dry_run:
        print(
            "Complete feature files found. Will fit {} on source only; negative policies: {}. No output written.".format(
                features, policies
            )
        )
        return
    run = Run(root, args, hashes, protocol=protocol)
    summaries = []
    for seed in args.seeds:
        with run.stage("seed{}_fit_and_freeze_source_models".format(seed)):
            source_table, source_meta = read_feature_table(
                files[seed, args.source_task]
            )
            if (
                source_meta["task_index"] != args.source_task
                or source_meta.get("checkpoint_seed") != seed
            ):
                raise ValueError("Source metadata differs from declared task/seed.")
            bundles = {}
            variants = [
                (feature, policy) for feature in features for policy in policies
            ]
            for feature, policy in variants:
                name = (
                    feature if policies == ["clean_only"] else feature + "__" + policy
                )
                bundle = supervised.fit_source(
                    source_table,
                    source_meta,
                    feature_set=feature,
                    task_size=args.task_size,
                    negative_policy=policy,
                )
                bundle["input_file_sha256"] = source_meta["features_sha256"]
                bundles[name] = bundle
                destination = root / ("seed" + str(seed)) / ("source_" + name)
                save_frozen_bundle(bundle, destination)
                save_json(
                    {
                        k: v
                        for k, v in bundle.items()
                        if k not in ("scaler", "classifier")
                    },
                    destination / "fit_metrics.json",
                )
                if getattr(args, "explanations", False):
                    from detector.explain_detector import explain

                    explanation = root / ("seed" + str(seed)) / ("explain_" + name)
                    explanation.mkdir(parents=True)
                    explain(
                        source_table,
                        source_meta,
                        bundle,
                        explanation,
                        split="validation",
                    )
        # Source models are persisted before loading any target benchmark values.
        for task in (args.source_task, args.target_task):
            with run.stage("seed{}_evaluate_task{}".format(seed, task)):
                table, metadata = read_feature_table(files[seed, task])
                if (
                    metadata["task_index"] != task
                    or metadata.get("checkpoint_seed") != seed
                ):
                    raise ValueError(
                        "Evaluation metadata differs from declared task/seed."
                    )
                if bool(metadata.get("synthetic")) != bool(
                    source_meta.get("synthetic")
                ):
                    raise ValueError("Cannot mix synthetic and real inputs.")
                for name, bundle in bundles.items():
                    report, predictions, bags = supervised.evaluate(
                        table,
                        metadata,
                        bundle,
                        source_control=task == args.source_task,
                        repeats=args.bags_per_rate,
                    )
                    report["evaluation_role"] = "exploratory_reused_test_after_revision"
                    destination = (
                        root
                        / ("seed" + str(seed))
                        / ("evaluate_task{}_".format(task) + name)
                    )
                    destination.mkdir(parents=True)
                    save_json(report, destination / "evaluation_metrics.json")
                    predictions.to_csv(
                        destination / "sample_predictions.csv", index=False
                    )
                    bags.to_csv(destination / "bag_predictions.csv", index=False)
                    supervised.write_report(report, destination)
                    supervised.write_curves(report, predictions, destination)
                    summaries.append(
                        dict(
                            seed=seed,
                            task=task,
                            feature_set=name,
                            negative_policy=bundle.get("negative_policy", "clean_only"),
                            raw_feature_set=bundle["feature_set"],
                            n_features=len(bundle["feature_columns"]),
                            replication_identity={
                                "checkpoint": metadata["checkpoint_sha256"],
                                "inversions": metadata["inversion_sha256"],
                                "attack": metadata.get("input_sha256", {}).get(
                                    "attack"
                                ),
                            },
                            synthetic=report["synthetic"],
                            sample_metrics=report["sample_metrics"],
                            random_control_sample_alert_rate=report[
                                "random_control_sample_alert_rate"
                            ],
                            poison_vs_random_roc_auc=report["poison_vs_random_roc_auc"],
                            dataset_results=report["dataset_results"],
                        )
                    )
    with run.stage("write_summary"):
        save_json(
            {
                "runs": summaries,
                "evaluation_role": "exploratory",
                "automatically_selected_winner": False,
            },
            root / "summary.json",
        )
        lines = [
            "# Supervised revision: exploratory reused test",
            "",
            "Feature sets: {}. Shape, if included, uses per-image stage L2 normalization. No target calibration or automatic model selection.".format(
                ", ".join(features)
            ),
            "",
            "| Seed | Task | Variant | Synthetic | Clean/poison AUC | Clean FPR | Poison TPR | Random sample alerts | Poison/random AUC | Clean bag alerts |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
        for row in summaries:
            sample = row["sample_metrics"]
            clean = next(
                r["alert_rate"]
                for r in row["dataset_results"]
                if r["alternative"] == "poison" and r["requested_rate"] == 0
            )
            lines.append(
                "| {} | {} | {} | {} | {:.4f} | {:.2%} | {:.2%} | {:.2%} | {:.4f} | {:.2%} |".format(
                    row["seed"],
                    row["task"],
                    row["feature_set"],
                    row["synthetic"],
                    sample["roc_auc"],
                    sample["clean_fpr"],
                    sample["poison_tpr"],
                    row["random_control_sample_alert_rate"],
                    row["poison_vs_random_roc_auc"],
                    clean,
                )
            )
        lines += [
            "",
            "Per-run rates include low-contamination and random controls. Lower false alerts alone do not establish success if detection power disappears.",
        ]
        (root / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    if finalize is not None:
        with run.stage("write_granularity_comparison"):
            finalize(root, summaries)
    run.finish()


def add_shape_results(base, bundle, table, metadata):
    lookup = table.loc[table.split == "test"].set_index(["original_index", "view"])
    result = dict(base, tasks=[], summary=[])
    for index, row in enumerate(base["tasks"]):
        keys = [
            (identity, row["scenario"] if i < row["modified_count"] else "clean")
            for i, identity in enumerate(row["original_indices"])
        ]
        bag = lookup.loc[keys, shape_reference.COLUMNS].reset_index(drop=True)
        prediction = shape_reference.predict_dataset(bundle, bag, metadata)
        result["tasks"].append(
            dict(
                row,
                shape_status=prediction["status"],
                shape_alert=prediction["shift_detected"],
                shape_p_value=prediction["p_value"],
                shape_per_task=prediction["per_task"],
            )
        )
        if (index + 1) % 100 == 0:
            print(
                "Shape MMD: {}/{} bags".format(index + 1, len(base["tasks"])),
                flush=True,
            )
    for row in base["summary"]:
        group = [
            r
            for r in result["tasks"]
            if r["scenario"] == row["scenario"]
            and r["requested_rate"] == row["requested_rate"]
        ]
        supported = [r for r in group if r["shape_alert"] is not None]
        alerts = sum(r["shape_alert"] for r in supported)
        result["summary"].append(
            dict(
                row,
                shape_coverage=len(supported) / len(group),
                shape_alert_rate=alerts / len(group)
                if len(supported) == len(group)
                else None,
                shape_alert_rate_among_supported=alerts / len(supported)
                if supported
                else None,
            )
        )
    result.update(
        shape_method=shape_reference.METHOD,
        shape_limitations=shape_reference.LIMITATIONS,
        evaluation_role="exploratory_reused_test_after_revision",
        automatically_selected_winner=False,
    )
    return result


def run_unsupervised(args):
    source = Path(args.source_run).resolve()
    root = unsupervised_study.fresh_output(args.output_dir, source)
    rank_reference.integer(args.task_size, "task_size", 16)
    rank_reference.integer(args.bags_per_rate, "bags_per_rate", 1)
    if args.task_size > 256:
        raise ValueError("Task size must not exceed 256; no silent subsampling.")
    refpath, targetpath = (
        source / "reference_features.csv",
        source / "predicted_features.csv",
    )
    hashes = _inputs([refpath, targetpath])
    if args.dry_run:
        print(
            "Complete historical/predicted feature files found. Will compare four methods on identical bags; no output written."
        )
        return
    run = Run(root, args, hashes)
    with run.stage("fit_and_freeze_historical_only_models"):
        reference, refmeta = read_feature_table(refpath)
        models = {
            "legacy": unsupervised.fit_reference(reference, refmeta),
            "rank": rank_reference.fit_reference(reference, refmeta),
            "local": local_reference.fit_reference(
                reference, refmeta, task_size=args.task_size
            ),
            "shape": shape_reference.fit_reference(reference, refmeta),
        }
        for name, model in models.items():
            save_frozen_bundle(model, root / "models" / name)
    with run.stage("historical_audit"):
        audit = shape_reference.historical_audit(reference, refmeta)
        save_json(audit, root / "historical_audit.json")
    with run.stage("load_benchmark_after_freeze"):
        table, metadata = read_feature_table(targetpath)
        if bool(metadata.get("synthetic")) != bool(refmeta.get("synthetic")):
            raise ValueError("Cannot mix synthetic and real inputs.")
    with run.stage("compare_four_methods_same_bags"):
        base = unsupervised_adapt.evaluate_benchmark(
            models["rank"],
            models["legacy"],
            table,
            metadata,
            task_size=args.task_size,
            tasks_per_rate=args.bags_per_rate,
            progress=True,
        )
        base = unsupervised_study.evaluate_local(base, models["local"], table, metadata)
        result = add_shape_results(base, models["shape"], table, metadata)
        result["synthetic"] = bool(metadata.get("synthetic"))
        result["replication_identity"] = {
            "checkpoint": metadata["checkpoint_sha256"],
            "inversions": metadata["inversion_sha256"],
            "attack": metadata.get("input_sha256", {}).get("attack"),
        }
        save_json(result, root / "evaluation_metrics.json")
        pd.DataFrame(result["summary"]).to_csv(root / "rates.csv", index=False)
    with run.stage("write_report"):
        lines = [
            "# Strict label-free revision: exploratory reused test",
            "",
            "SYNTHETIC FIXTURE ONLY"
            if result["synthetic"]
            else "Real features; distributions and poisoning attribution remain distinct.",
            "",
            "All four models frozen from historical inversions before loading benchmark values. No trusted incoming clean subset.",
            "",
            "| Scenario | Fraction | Raw MMD | Rank | Local | Shape MMD | Shape coverage |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]

        def percent(value):
            return "unavailable" if value is None else "{:.2%}".format(value)

        for row in result["summary"]:
            lines.append(
                "| {} | {} | {} | {} | {} | {} | {} |".format(
                    row["scenario"],
                    *[
                        percent(row[key])
                        for key in (
                            "realized_rate",
                            "legacy_alert_rate",
                            "rank_alert_rate",
                            "local_alert_rate",
                            "shape_alert_rate",
                            "shape_coverage",
                        )
                    ],
                )
            )
        lines += [
            "",
            "Clean alerts are false alerts. Report power at every contamination rate and random controls. Bags reuse images; they are not independent experimental replications.",
            "",
            "Model limitations:",
            "",
        ] + ["- " + note for note in shape_reference.LIMITATIONS]
        (root / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    run.finish()


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("supervised", "unsupervised"):
        p = sub.add_parser(name)
        p.add_argument("--source-run", required=True)
        p.add_argument("--output-dir", required=True)
        p.add_argument("--task-size", type=int, default=150)
        p.add_argument(
            "--bags-per-rate", type=int, default=100 if name == "unsupervised" else 1000
        )
        p.add_argument("--dry-run", action="store_true")
        if name == "supervised":
            p.add_argument(
                "--feature-sets",
                nargs="+",
                choices=supervised.registered_feature_sets(),
                default=["portable", "extended", "shape"],
            )
            p.add_argument("--seeds", type=int, nargs="+", default=[3, 4])
            p.add_argument("--source-task", type=int, default=1)
            p.add_argument("--target-task", type=int, default=9)
            p.add_argument(
                "--negative-policies",
                nargs="+",
                choices=supervised.NEGATIVE_POLICIES,
                default=["clean_only"],
            )
            p.add_argument(
                "--explanations",
                action="store_true",
                help="Explain each frozen model on source validation using its actual training views.",
            )
    return parser


if __name__ == "__main__":
    arguments = build_parser().parse_args()
    (run_supervised if arguments.command == "supervised" else run_unsupervised)(
        arguments
    )
