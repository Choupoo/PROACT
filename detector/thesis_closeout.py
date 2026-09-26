"""Fixed-method rechecks and descriptive thesis tables from existing features.

No threshold tuning on target tasks. Previously viewed seeds are rechecks, not
unseen confirmation. Completion reports evidence, never automatic graduation.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from detector import rank_reference, unsupervised, unsupervised_adapt
from detector.common import save_json, sha256_file
from detector.io_utils import read_feature_table, save_frozen_bundle
from detector.revision_study import Run, _inputs
from detector.unsupervised_study import fresh_output

PROTOCOL = "thesis_closeout_fixed_recheck_v1"
RANK_SETTINGS = {
    "alpha": 0.05,
    "bootstrap_draws": 499,
    "seed": 20260920,
    "max_reference_per_task": 256,
    "max_incoming": 256,
}


def identity(metadata):
    result = {
        "checkpoint": metadata["checkpoint_sha256"],
        "inversions": metadata["inversion_sha256"],
        "attack": metadata.get("input_sha256", {}).get("attack"),
    }
    for key, value in result.items():
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(c not in "0123456789abcdef" for c in value.lower())
        ):
            raise ValueError("Need a known artifact SHA-256 for " + key)
    return result


def stats(values):
    if not values or any(value is None for value in values):
        return {
            "n_runs": len(values),
            "mean": None,
            "std": None,
            "min": None,
            "max": None,
        }
    values = np.asarray(values, dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Cannot summarize nonfinite measurements.")
    return {
        "n_runs": len(values),
        "mean": float(values.mean()),
        "std": float(values.std(ddof=1)) if len(values) > 1 else None,
        "min": float(values.min()),
        "max": float(values.max()),
    }


def summarize_rank(results):
    if not results:
        raise ValueError("No rank results.")
    if len({r["synthetic"] for r in results}) != 1:
        raise ValueError("Cannot pool synthetic and real experiments.")
    checkpoints = [r["replication_identity"]["checkpoint"] for r in results]
    attacks = [r["replication_identity"]["attack"] for r in results]
    if len(set(checkpoints)) != len(results) or len(set(attacks)) != len(results):
        raise ValueError(
            "Repeated checkpoint/attack artifacts do not count as different model/attack rechecks."
        )
    grid = [
        (x["scenario"], x["requested_rate"], x["realized_rate"])
        for x in results[0]["summary"]
    ]
    for result in results:
        if [
            (x["scenario"], x["requested_rate"], x["realized_rate"])
            for x in result["summary"]
        ] != grid:
            raise ValueError("Recheck scenario grids differ.")
    rows = []
    for i, (scenario, requested, actual) in enumerate(grid):
        for method in ("rank", "legacy"):
            values = [r["summary"][i][method + "_alert_rate"] for r in results]
            rows.append(
                dict(
                    scenario=scenario,
                    requested_rate=requested,
                    realized_rate=actual,
                    method=method,
                    run_rates=values,
                    **stats(values),
                )
            )
    return {
        "protocol": PROTOCOL,
        "rank_settings": dict(RANK_SETTINGS),
        "runs": len(results),
        "distinct_checkpoints": len(set(checkpoints)),
        "distinct_attacks": len(set(attacks)),
        "synthetic": results[0]["synthetic"],
        "rows": rows,
        "evaluation_role": "fixed_method_recheck_existing_data",
        "independent_confirmation_verified": False,
        "note": "Equal weights across runs; std is across runs, not reused bags. Shared images and previously observed seeds limit independence.",
    }


def run_rank(args):
    root = fresh_output(args.output_dir)
    sources = [Path(p).resolve() for p in args.source_runs]
    if len(set(sources)) != len(sources):
        raise ValueError("Duplicate source directories.")
    for source in sources:
        fresh_output(root, source)
    if not 16 <= args.task_size <= 256 or args.bags_per_rate < 1:
        raise ValueError("Require task_size in 16..256 and positive bags_per_rate.")
    hashes = _inputs(
        [
            source / name
            for source in sources
            for name in ("reference_features.csv", "predicted_features.csv")
        ]
    )
    if args.dry_run:
        print("Full input files found. Rank settings frozen:", RANK_SETTINGS)
        print("No output written. Existing seeds will be labelled as rechecks.")
        return
    run = Run(
        root,
        args,
        hashes,
        protocol=PROTOCOL,
        evaluation_role="fixed_method_recheck_existing_data",
    )
    results = []
    for i, source in enumerate(sources, 1):
        label = "replicate" + str(i)
        output = root / label
        with run.stage(label + "_fit_and_freeze"):
            reference, refmeta = read_feature_table(source / "reference_features.csv")
            ranked = rank_reference.fit_reference(reference, refmeta, **RANK_SETTINGS)
            baseline = unsupervised.fit_reference(reference, refmeta)
            for name, model in (("rank", ranked), ("legacy", baseline)):
                save_frozen_bundle(model, output / "models" / name)
            save_json(
                rank_reference.historical_audit(ranked),
                output / "historical_audit.json",
            )
        with run.stage(label + "_load_target_after_freeze"):
            features, metadata = read_feature_table(source / "predicted_features.csv")
            if bool(metadata.get("synthetic")) != bool(refmeta.get("synthetic")):
                raise ValueError("Cannot mix synthetic and real inputs.")
            identifiers = identity(metadata)
            if any(
                identifiers["checkpoint"] == r["replication_identity"]["checkpoint"]
                or identifiers["attack"] == r["replication_identity"]["attack"]
                for r in results
            ):
                raise ValueError("Repeated checkpoint/attack in cross-run recheck.")
        with run.stage(label + "_evaluate"):
            result = unsupervised_adapt.evaluate_benchmark(
                ranked,
                baseline,
                features,
                metadata,
                task_size=args.task_size,
                tasks_per_rate=args.bags_per_rate,
                seed=20260921,
                progress=True,
            )
            result.update(
                replication_identity=identifiers,
                source_run=str(source),
                synthetic=bool(metadata.get("synthetic")),
                rank_settings=dict(RANK_SETTINGS),
                evaluation_role="fixed_method_recheck_existing_data",
                independent_confirmation_verified=False,
            )
            save_json(result, output / "evaluation_metrics.json")
            pd.DataFrame(result["summary"]).to_csv(output / "rates.csv", index=False)
            results.append(result)
    with run.stage("summarize"):
        summary = summarize_rank(results)
        save_json(summary, root / "summary.json")
        pd.DataFrame(summary["rows"]).to_csv(root / "summary.csv", index=False)
        lines = [
            "# Frozen Rank cross-run recheck",
            "",
            summary["note"],
            "",
            "SYNTHETIC ONLY"
            if summary["synthetic"]
            else "Existing real feature experiments; not unseen confirmation.",
            "",
            "| Scenario | Actual fraction | Method | Per-run rates | Mean | Std across runs |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
        for row in summary["rows"]:
            lines.append(
                "| {} | {:.2%} | {} | {} | {} | {} |".format(
                    row["scenario"],
                    row["realized_rate"],
                    row["method"],
                    ", ".join(fmt(v) for v in row["run_rates"]),
                    fmt(row["mean"]),
                    fmt(row["std"]),
                )
            )
        lines += [
            "",
            "Zero observed clean alerts is not a population zero-error guarantee. Inspect full coverage, every poisoning rate and random controls.",
        ]
        (root / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    run.finish()


def fmt(value):
    return "unavailable" if value is None else "{:.2%}".format(value)


def completed(path):
    path = Path(path).resolve()
    state = json.loads((path / "run_state.json").read_text())
    if (
        state.get("status") != "complete"
        or not state.get("steps")
        or any(s.get("status") != "complete" for s in state["steps"].values())
    ):
        raise ValueError("Incomplete experiment: " + str(path))
    return path


def summarize(args):
    root = fresh_output(args.output_dir)
    supervised, ranked = completed(args.supervised_run), completed(args.rank_run)
    for source in (supervised, ranked):
        fresh_output(root, source)
    supervised_config = json.loads((supervised / "run_config.json").read_text())
    rank_config = json.loads((ranked / "run_config.json").read_text())
    rows = json.loads((supervised / "summary.json").read_text())["runs"]
    rank_summary = json.loads((ranked / "summary.json").read_text())
    if (
        rank_summary.get("protocol") != PROTOCOL
        or rank_config.get("protocol") != PROTOCOL
    ):
        raise ValueError("Expected the fixed Rank recheck protocol.")
    if rank_summary.get("rank_settings") != RANK_SETTINGS:
        raise ValueError("Rank settings differ from the frozen closeout protocol.")
    if {row["synthetic"] for row in rows} != {rank_summary["synthetic"]}:
        raise ValueError("Cannot mix synthetic and real results.")
    actual_seeds = set(supervised_config["command"]["seeds"])
    actual_tasks = {
        supervised_config["command"]["source_task"],
        supervised_config["command"]["target_task"],
    }
    expected = {
        (seed, task, feature, policy)
        for seed in actual_seeds
        for task in actual_tasks
        for feature in ("portable", "extended", "shape")
        for policy in ("clean_only", "clean_and_random")
    }
    seen = [
        (r["seed"], r["task"], r.get("raw_feature_set"), r.get("negative_policy"))
        for r in rows
    ]
    if len(seen) != len(set(seen)) or set(seen) != expected:
        raise ValueError(
            "Need the complete paired negative-policy experiment, without duplicate or missing rows."
        )
    grouped = []
    for task in sorted(actual_tasks):
        for feature in ("portable", "extended", "shape"):
            for policy in ("clean_only", "clean_and_random"):
                group = sorted(
                    [
                        r
                        for r in rows
                        if r["task"] == task
                        and r["raw_feature_set"] == feature
                        and r["negative_policy"] == policy
                    ],
                    key=lambda r: r["seed"],
                )
                if len({r["replication_identity"]["checkpoint"] for r in group}) != len(
                    group
                ):
                    raise ValueError(
                        "Repeated checkpoint labelled as distinct supervised seeds."
                    )
                measurements = {
                    key: [r["sample_metrics"][key] for r in group]
                    for key in ("roc_auc", "clean_fpr", "poison_tpr")
                }
                measurements.update(
                    random_alerts=[
                        r["random_control_sample_alert_rate"] for r in group
                    ],
                    poison_vs_random_auc=[r["poison_vs_random_roc_auc"] for r in group],
                )
                grid = [
                    (x["alternative"], x["requested_rate"], x["realized_rate"])
                    for x in group[0]["dataset_results"]
                ]
                if any(
                    [
                        (x["alternative"], x["requested_rate"], x["realized_rate"])
                        for x in r["dataset_results"]
                    ]
                    != grid
                    for r in group
                ):
                    raise ValueError("Supervised dataset scenario grids differ.")
                for index, (alternative, rate, _) in enumerate(grid):
                    measurements["bag_{}_{}".format(alternative, rate)] = [
                        r["dataset_results"][index]["alert_rate"] for r in group
                    ]
                for metric, values in measurements.items():
                    grouped.append(
                        dict(
                            task=task,
                            feature_set=feature,
                            negative_policy=policy,
                            metric=metric,
                            seeds=[r["seed"] for r in group],
                            values=values,
                            **stats(values),
                        )
                    )
    root.mkdir(parents=True)
    summary = {
        "protocol": PROTOCOL,
        "supervised_rows": grouped,
        "rank": rank_summary,
        "synthetic": rank_summary["synthetic"],
        "automatic_thesis_pass": False,
        "sources": {
            str(p): sha256_file(p / "summary.json") for p in (supervised, ranked)
        },
        "limitations": [
            "Observed existing seeds; no claim of unseen independent confirmation.",
            "Cross-seed std is descriptive; repeated bags do not increase the number of independent models.",
            "Random noise is a non-BrainWash control, not established harmless training data.",
            "AUC does not establish frozen-threshold calibration or target dataset false-positive control.",
            "No post-detection continual-training defense efficacy experiment was performed by these commands.",
        ],
    }
    save_json(summary, root / "thesis_summary.json")
    pd.DataFrame(grouped).to_csv(root / "supervised_summary.csv", index=False)
    pd.DataFrame(rank_summary["rows"]).to_csv(root / "rank_summary.csv", index=False)
    lines = [
        "# Thesis evidence summary",
        "",
        "SYNTHETIC ONLY"
        if summary["synthetic"]
        else "Existing real experiments; all results remain exploratory/rechecks.",
        "",
        "| Task | Features | Negative policy | Metric | Per-seed values | Mean | Std |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in grouped:
        if row["metric"] in (
            "roc_auc",
            "clean_fpr",
            "poison_tpr",
            "random_alerts",
            "poison_vs_random_auc",
            "bag_poison_0.0",
        ):
            lines.append(
                "| {} | {} | {} | {} | {} | {} | {} |".format(
                    row["task"],
                    row["feature_set"],
                    row["negative_policy"],
                    row["metric"],
                    ", ".join(fmt(v) for v in row["values"]),
                    fmt(row["mean"]),
                    fmt(row["std"]),
                )
            )
    lines += [
        "",
        "## Rank and original MMD",
        "",
        "| Scenario | Actual fraction | Method | Per-run rates | Mean | Std |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in rank_summary["rows"]:
        lines.append(
            "| {} | {:.2%} | {} | {} | {} | {} |".format(
                row["scenario"],
                row["realized_rate"],
                row["method"],
                ", ".join(fmt(v) for v in row["run_rates"]),
                fmt(row["mean"]),
                fmt(row["std"]),
            )
        )
    lines += ["", "## Interpretation boundaries", ""] + [
        "- " + x for x in summary["limitations"]
    ]
    lines += [
        "",
        "Check jointly: clean false alerts, poison sensitivity, random controls, coverage and seed variability. No automatic winner or graduation verdict.",
    ]
    (root / "thesis_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("Thesis tables saved:", root)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("rank")
    p.add_argument("--source-runs", nargs="+", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--task-size", type=int, default=150)
    p.add_argument("--bags-per-rate", type=int, default=100)
    p.add_argument("--dry-run", action="store_true")
    p = sub.add_parser("summarize")
    p.add_argument("--supervised-run", required=True)
    p.add_argument("--rank-run", required=True)
    p.add_argument("--output-dir", required=True)
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    (run_rank if args.command == "rank" else summarize)(args)
