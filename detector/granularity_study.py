"""Fixed eight-arm source-only gradient granularity ablation.

Reuses existing features. Parameter means a named parameter tensor, not an
individual scalar weight. Real results are generated only after evaluation.
"""

import argparse
import json
import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning

from detector.common import save_json
from detector import revision_study
from detector.transfer_detector import GRANULARITY_GROUPS

PROTOCOL = "gradient_granularity_ablation_v1"


def describe(values):
    values = [float(v) for v in values]
    return {
        "values": values,
        "mean": float(np.mean(values)),
        "std": float(np.std(values, ddof=1)) if len(values) > 1 else None,
    }


def metrics(row):
    result = {
        key: row["sample_metrics"][key]
        for key in ("roc_auc", "clean_fpr", "poison_tpr")
    }
    result["random_alerts"] = row["random_control_sample_alert_rate"]
    result["poison_vs_random_auc"] = row["poison_vs_random_roc_auc"]
    for bag in row["dataset_results"]:
        result["bag_{}_{}".format(bag["alternative"], bag["requested_rate"])] = bag[
            "alert_rate"
        ]
    return result


def write_comparison(root, runs):
    """Paired differences against stage in the same feature/context arm."""
    config = json.loads((root / "run_config.json").read_text())["command"]
    seeds = config["seeds"]
    tasks = (config["source_task"], config["target_task"])
    lookup = {(r["seed"], r["task"], r["raw_feature_set"]): r for r in runs}
    expected = {(s, t, g) for s in seeds for t in tasks for g in GRANULARITY_GROUPS}
    if set(lookup) != expected or len(runs) != len(expected):
        raise ValueError("Incomplete or duplicate granularity result grid.")
    if len({r["synthetic"] for r in runs}) != 1:
        raise ValueError("Mixed synthetic and real granularity results.")
    for task in tasks:
        checkpoints, attacks = set(), set()
        for seed in seeds:
            group = [lookup[seed, task, name] for name in GRANULARITY_GROUPS]
            identities = [r["replication_identity"] for r in group]
            if any(identity != identities[0] for identity in identities):
                raise ValueError(
                    "Paired models must use identical benchmark identities."
                )
            if (
                identities[0]["checkpoint"] in checkpoints
                or identities[0]["attack"] in attacks
            ):
                raise ValueError(
                    "Repeated checkpoint/attack cannot count as another seed."
                )
            checkpoints.add(identities[0]["checkpoint"])
            attacks.add(identities[0]["attack"])
    # Ensure the eight arms actually share fit/validation/reserve identities.
    schemas = {}
    for seed in seeds:
        baseline = None
        for name in GRANULARITY_GROUPS:
            fit = json.loads(
                (
                    root / f"seed{seed}" / ("source_" + name) / "fit_metrics.json"
                ).read_text()
            )
            signature = [
                fit[k]
                for k in (
                    "fit_original_indices",
                    "calibration_original_indices",
                    "count_calibration_original_indices",
                    "fit_views",
                    "settings",
                )
            ]
            if baseline is not None and signature != baseline:
                raise ValueError(
                    "Granularity arms must share training/calibration protocol."
                )
            baseline = signature
            if name in schemas and schemas[name] != fit["feature_columns"]:
                raise ValueError("Feature schema differs across seeds.")
            schemas[name] = fit["feature_columns"]
    aggregate, deltas = [], []
    for task in tasks:
        for name in GRANULARITY_GROUPS:
            context = name.endswith("_context")
            baseline = "norm_stage" + ("_context" if context else "")
            rows = [lookup[s, task, name] for s in seeds]
            dimensions = {r["n_features"] for r in rows}
            if len(dimensions) != 1:
                raise ValueError("Feature count differs across seeds.")
            for metric in metrics(rows[0]):
                values = [metrics(r)[metric] for r in rows]
                common = dict(
                    task=task,
                    feature_set=name,
                    context=context,
                    n_features=rows[0]["n_features"],
                    metric=metric,
                    seeds=seeds,
                )
                aggregate.append(dict(common, **describe(values)))
                paired = [
                    v - metrics(lookup[s, task, baseline])[metric]
                    for s, v in zip(seeds, values)
                ]
                deltas.append(dict(common, baseline=baseline, **describe(paired)))
    synthetic = bool(runs[0]["synthetic"])
    save_json(
        {
            "protocol": PROTOCOL,
            "synthetic": synthetic,
            "evaluation_role": "exploratory_reused_test",
            "parameter_definition": "One L2 gradient norm per named backbone parameter tensor, including weight and bias tensors; not per scalar weight.",
            "aggregate": aggregate,
            "paired_differences_vs_stage": deltas,
            "automatically_selected_winner": False,
        },
        root / "granularity_summary.json",
    )
    pd.DataFrame(aggregate).to_csv(root / "granularity_summary.csv", index=False)
    pd.DataFrame(deltas).to_csv(root / "paired_differences_vs_stage.csv", index=False)
    values = {(r["task"], r["feature_set"], r["metric"]): r["mean"] for r in aggregate}
    lines = [
        "# Gradient granularity comparison",
        "",
        "SYNTHETIC TEST DATA — not empirical research results."
        if synthetic
        else "Existing real feature data; exploratory reused-test comparison.",
        "",
        "Only gradient granularity varies within each block. Context adds the same entropy, confidence, true-class probability, margin and activation norm. No target calibration.",
        "",
        "Global pools the backbone; stage pools stem/layer1–4; layer pools each named module; parameter separates each named weight/bias tensor. Scalar-weight features are not tested.",
        "",
    ]
    header = "| Features | Dimensions | AUC | Clean FPR | Poison TPR | Random alerts | Poison/random AUC | Clean bag alerts |"
    for task in tasks:
        lines += [
            f"## Task {task} (zero-based)",
            "",
            "Equal-weight means across seeds; per-seed values and sample standard deviations are in granularity_summary.csv.",
            "",
            header,
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        for name in GRANULARITY_GROUPS:
            dims = lookup[seeds[0], task, name]["n_features"]
            vals = [
                values[task, name, key]
                for key in (
                    "roc_auc",
                    "clean_fpr",
                    "poison_tpr",
                    "random_alerts",
                    "poison_vs_random_auc",
                    "bag_poison_0.0",
                )
            ]
            lines.append(
                "| {} | {} | {} |".format(
                    name, dims, " | ".join(f"{v:.2%}" for v in vals)
                )
            )
    lines += [
        "",
        "## Paired changes versus stage",
        "",
        "Percentage-point changes (candidate minus stage in the same block). Higher AUC/TPR is favorable; lower clean/random alerts is favorable. No significance claim from two seeds.",
        "",
        "| Task | Features | AUC change | Clean FPR change | Poison TPR change | Clean bag change |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for task in tasks:
        for name in GRANULARITY_GROUPS:
            if name.split("_")[1] == "stage":
                continue
            baseline = "norm_stage" + ("_context" if name.endswith("_context") else "")
            differences = [
                100 * (values[task, name, k] - values[task, baseline, k])
                for k in ("roc_auc", "clean_fpr", "poison_tpr", "bag_poison_0.0")
            ]
            lines.append(
                "| {} | {} | {} |".format(
                    task, name, " | ".join(f"{v:+.2f}" for v in differences)
                )
            )
    lines += [
        "",
        "## Interpretation",
        "",
        "- Read AUC, frozen-threshold FPR, TPR, random-control alerts and bag curves together. Better ranking alone does not solve deployment decisions.",
        "- All models use the same StandardScaler and fixed LogisticRegression settings. Dimensions change with granularity; this is not a capacity-matched or separately tuned model comparison.",
        "- SHAP is computed on source validation with source training background. It describes a fitted logit, not causal layer importance. Correlated features can share credit.",
        "- Reused original images across simulated bags do not create independent replications. Previously inspected tasks/seeds are not fresh confirmation.",
        "- This comparison tests global versus stage as well as layer/parameter versus stage. No automatic selection or threshold adjustment follows target results.",
    ]
    (root / "granularity_summary.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    # Aggregate descriptive source-validation explanations while preserving runs.
    shap_rows = []
    for seed in seeds:
        for name in GRANULARITY_GROUPS:
            path = root / f"seed{seed}" / ("explain_" + name) / "feature_importance.csv"
            frame = pd.read_csv(path)
            total = float(frame.mean_abs_shap_logit.sum())
            frame["share_of_absolute_logit_attribution"] = (
                frame.mean_abs_shap_logit / total if total > 0 else np.nan
            )
            frame["seed"], frame["feature_set"] = seed, name
            shap_rows.append(frame)
    combined_shap = pd.concat(shap_rows, ignore_index=True)
    combined_shap.to_csv(root / "shap_by_seed.csv", index=False)
    shap_lines = [
        "# Source-validation SHAP by granularity",
        "",
        "Mean absolute logit attribution, not accuracy improvement or causal importance. Each model is explained with its own source-training background.",
        "",
        "| Seed | Features | Top five gradient features (absolute logit SHAP) |",
        "| --- | --- | --- |",
    ]
    for (seed, name), frame in combined_shap.groupby(
        ["seed", "feature_set"], sort=False
    ):
        top = (
            frame.loc[frame.feature.str.startswith("grad_norm_")]
            .sort_values("mean_abs_shap_logit", ascending=False)
            .head(5)
        )
        shap_lines.append(
            "| {} | {} | {} |".format(
                seed,
                name,
                "; ".join(
                    "{} ({:.3g})".format(r.feature, r.mean_abs_shap_logit)
                    for r in top.itertuples()
                ),
            )
        )
    if synthetic:
        shap_lines.insert(2, "SYNTHETIC TEST DATA — not empirical research results.")
    (root / "shap_summary.md").write_text(
        "\n".join(shap_lines) + "\n", encoding="utf-8"
    )
    email = [
        "# Draft update to Prof. Carta",
        "",
        "Dear Prof. Carta,",
        "",
        "I have completed a controlled comparison of gradient-norm granularity using the existing feature artifacts. I compared a single backbone norm, five stage norms, norms for individual modules, and norms for individual named parameter tensors (weights and biases). The last representation pools each tensor; it does not retain every scalar weight gradient.",
        "",
        "I ran two matched blocks: gradient norms alone, and gradient norms plus an identical set of uncertainty and activation features. Data splits, logistic-regression settings and calibration rules were fixed. All models were fitted on the early source task, with frozen evaluation on the later target task. SHAP explanations use the source validation split.",
        "",
        "The attached granularity_summary.md reports ranking, frozen-threshold decisions, random controls and dataset-level false alerts. The paired_differences_vs_stage.csv gives per-seed improvements or regressions relative to stage norms.",
        "",
        "The following are mean results on the target task; the full report includes source controls and both individual seeds:",
        "",
        header,
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name in GRANULARITY_GROUPS:
        vals = [
            values[tasks[1], name, k]
            for k in (
                "roc_auc",
                "clean_fpr",
                "poison_tpr",
                "random_alerts",
                "poison_vs_random_auc",
                "bag_poison_0.0",
            )
        ]
        email.append(
            "| {} | {} | {} |".format(
                name,
                lookup[seeds[0], tasks[1], name]["n_features"],
                " | ".join(f"{v:.2%}" for v in vals),
            )
        )
    email += [
        "",
        "These are exploratory results on previously inspected tasks and seeds. AUC improvements do not by themselves establish successful threshold transfer or dataset-level calibration. I would appreciate your feedback on whether this tensor-level interpretation addresses your suggested fine-grained norm analysis.",
        "",
        "Best regards,",
        "Pan Zhang",
    ]
    if synthetic:
        email.insert(2, "SYNTHETIC TEST OUTPUT — do not send as experimental evidence.")
    (root / "professor_update.md").write_text("\n".join(email) + "\n", encoding="utf-8")


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", default="detector/work/meeting3_transfer_v1")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[3, 4])
    parser.add_argument("--source-task", type=int, default=1)
    parser.add_argument("--target-task", type=int, default=9)
    parser.add_argument("--task-size", type=int, default=150)
    parser.add_argument("--bags-per-rate", type=int, default=1000)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(args):
    args.feature_sets = list(GRANULARITY_GROUPS)
    args.negative_policies = ["clean_only"]
    args.explanations = True
    args.command = "gradient_granularity"
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        revision_study.run_supervised(
            args, finalize=write_comparison, protocol=PROTOCOL
        )


if __name__ == "__main__":
    main(build_parser().parse_args())
