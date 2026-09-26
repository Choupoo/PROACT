"""Source-only supervised fitting and frozen cross-task evaluation.

Target labels are confined to offline metrics/bag assembly. Target statistics
never alter the scaler, feature set, classifier or sample/count thresholds.
"""

import argparse
import os

import numpy as np
import pandas as pd

from detector import (
    FEATURE_COLUMNS,
    STAGE_GRAD_FEATURE_COLUMNS,
    UNCERTAINTY_FEATURE_COLUMNS,
)
from detector.calibration import count_decision, fit_count_calibration
from detector.common import save_json, sha256_file
from detector.dataset_detector import simulate_bags
from detector.io_utils import (
    DETECTOR_ROOT,
    ensure_output_path,
    load_frozen_bundle,
    read_feature_table,
    save_frozen_bundle,
)
from detector.train_detector import (
    binary_metrics,
    fit_detector,
    predict_scores,
    select_clean_threshold,
    validate_feature_table,
)
from detector.transfer_core import TRANSFER_PROTOCOL, assert_transfer, contract

FEATURE_GROUPS = {
    "portable": list(STAGE_GRAD_FEATURE_COLUMNS)
    + list(UNCERTAINTY_FEATURE_COLUMNS)
    + ["activation_norm_l2"],
    "extended": list(FEATURE_COLUMNS),
}
# Opt-in research revision; the original registered pipeline still runs only
# FEATURE_GROUPS. Do not silently add a method to an existing experiment.
SHAPE_COLUMNS = [
    name.replace("grad_norm_", "grad_shape_") for name in STAGE_GRAD_FEATURE_COLUMNS
]
EXPERIMENTAL_FEATURE_GROUPS = {"shape": SHAPE_COLUMNS}
RATES = (0.0, 0.01, 0.05, 0.1, 0.25, 0.5, 1.0)


def prepare_features(table, feature_set):
    """Deterministic per-image transform: no dataset statistics or labels."""
    if feature_set != "shape":
        return table
    from detector.local_reference import shapes

    values, valid = shapes(table)
    if not valid.all():
        raise ValueError(
            "Zero-gradient rows have undefined shape; cannot classify them as clean."
        )
    result = table.copy()
    result[SHAPE_COLUMNS] = values
    return result


def fresh_output(path):
    path = ensure_output_path(path)
    if path.exists():
        raise FileExistsError("Use a fresh output directory: {}".format(path))
    return path


def validate_table(table, metadata):
    validate_feature_table(table)
    contract(metadata)
    for key in (
        "feature_protocol",
        "head_mode",
        "head_seed",
        "label_mode",
        "task_index",
    ):
        if (
            key not in table
            or table[key].nunique() != 1
            or table[key].iloc[0] != metadata[key]
        ):
            raise ValueError("Row/sidecar mismatch: " + key)
    if metadata.get("attack_seed") != int(table.attack_seed.iloc[0]):
        raise ValueError("Attack seed differs between rows and metadata.")
    expected = table.original_index.map(
        lambda i: "cifar100:task{}:row{}".format(metadata["task_index"], int(i))
    )
    if "original_uid" not in table or not (table.original_uid == expected).all():
        raise ValueError("Global original UID differs from task/original identity.")


def fit_source(
    table,
    metadata,
    *,
    feature_set="portable",
    task_size=150,
    alpha=0.05,
    classifier_seed=20260924,
):
    validate_table(table, metadata)
    groups = dict(FEATURE_GROUPS, **EXPERIMENTAL_FEATURE_GROUPS)
    if feature_set not in groups:
        raise ValueError("Unknown registered feature set.")
    columns = groups[feature_set]
    table = prepare_features(table, feature_set)
    supervised = table.loc[table.view.isin(["clean", "poison"])]
    train = supervised.loc[supervised.split == "train"]
    validation = supervised.loc[supervised.split == "validation"]
    scaler, classifier = fit_detector(train, columns, classifier_seed)
    validation_scores = predict_scores(validation, columns, scaler, classifier)
    threshold = select_clean_threshold(
        validation_scores[validation.detector_label.to_numpy() == 0], alpha
    )
    calibration = table.loc[(table.split == "reserve") & (table.view == "clean")]
    if len(calibration) < task_size:
        raise ValueError("Insufficient source reserve originals for count calibration.")
    alarms = predict_scores(calibration, columns, scaler, classifier) >= threshold
    bundle = {
        "kind": TRANSFER_PROTOCOL,
        "feature_set": feature_set,
        "feature_columns": columns,
        "scaler": scaler,
        "classifier": classifier,
        "threshold": threshold,
        "count_calibration": fit_count_calibration(alarms, task_size, alpha),
        "task_size": int(task_size),
        "source_metadata": metadata,
        "provenance": {
            key: metadata[key]
            for key in (
                "feature_protocol",
                "head_mode",
                "head_seed",
                "label_mode",
                "checkpoint_sha256",
                "inversion_sha256",
            )
        },
        "fit_original_indices": sorted(train.original_index.unique().tolist()),
        "calibration_original_indices": sorted(
            validation.original_index.unique().tolist()
        ),
        "count_calibration_original_indices": sorted(
            calibration.original_index.unique().tolist()
        ),
        "validation": binary_metrics(
            validation.detector_label, validation_scores, threshold
        ),
        "settings": {"classifier_seed": classifier_seed, "alpha": alpha},
        "target_used_for_fitting_or_calibration": False,
        "count_bound_scope": "Source clean calibration only; no target-task FPR guarantee.",
        "class_label_policy": metadata["label_mode"],
        "supervision": "Source clean/attack labels; source validation-clean threshold; source reserve-clean count calibration.",
        "score_interpretation": "Logistic sample score, not calibrated deployment poisoning probability.",
    }
    if feature_set == "shape":
        bundle.update(
            raw_feature_columns=list(STAGE_GRAD_FEATURE_COLUMNS),
            feature_transform="per_sample_stage_l2_v1",
            revision_role="exploratory_after_transfer_v1",
            transformation_limit="Ignores overall gradient magnitude; does not guarantee target FPR or preserve attack signal.",
        )
    return bundle


def evaluate(
    table, metadata, bundle, *, source_control=False, repeats=1000, seed=20260925
):
    validate_table(table, metadata)
    if bundle.get("kind") != TRANSFER_PROTOCOL:
        raise ValueError("Expected a source-transfer bundle.")
    source = bundle["source_metadata"]
    if source_control:
        for key in (
            "features_sha256",
            "checkpoint_sha256",
            "task_index",
            "label_mode",
            "head_seed",
        ):
            if metadata[key] != source[key]:
                raise ValueError(
                    "Source control requires the original frozen source features."
                )
        check = {
            "source_control": True,
            "target_used_for_fitting_or_calibration": False,
        }
    else:
        check = assert_transfer(
            source,
            metadata,
            bundle.get("raw_feature_columns", bundle["feature_columns"]),
        )
    if (
        bundle["feature_set"] == "shape"
        and bundle.get("feature_transform") != "per_sample_stage_l2_v1"
    ):
        raise ValueError("Missing or unsupported frozen shape transform.")
    table = prepare_features(table, bundle["feature_set"])
    test = table.loc[table.split == "test"].copy()
    if source_control:
        seen = set(
            bundle["fit_original_indices"]
            + bundle["calibration_original_indices"]
            + bundle["count_calibration_original_indices"]
        )
        if seen & set(test.original_index):
            raise ValueError("Source test overlaps source fit/calibration.")
    # Predict before consulting view/poison labels for metrics.
    scores = predict_scores(
        test, bundle["feature_columns"], bundle["scaler"], bundle["classifier"]
    )
    threshold = bundle["threshold"]
    predictions = test[
        ["original_uid", "original_index", "class_id", "view", "detector_label"]
    ].copy()
    predictions["score"] = scores
    predictions["predicted_poison"] = scores >= threshold
    paired = predictions.view.isin(["clean", "poison"])
    sample = binary_metrics(
        predictions.loc[paired, "detector_label"], scores[paired], threshold
    )
    random = predictions.view == "random_control"
    pool = predictions[["original_index", "view"]].copy()
    pool["sample_poison_score"] = scores
    summaries, frames = [], []
    rates = [
        rate
        for rate in RATES
        if rate == 0 or int(np.floor(bundle["task_size"] * rate + 0.5)) > 0
    ]
    for alternative in ("poison", "random_control"):
        bags = simulate_bags(
            pool,
            rates,
            bundle["task_size"],
            repeats,
            seed,
            threshold,
            alternative=alternative,
        )
        counts = np.rint(
            bags.suspicious_fraction.to_numpy() * bundle["task_size"]
        ).astype(int)
        bags["suspicious_count"] = counts
        bags["rejected"], bags["source_binomial_tail"] = count_decision(
            counts, bundle["count_calibration"]
        )
        for rate, group in bags.groupby("requested_rate"):
            summaries.append(
                {
                    "alternative": alternative,
                    "requested_rate": float(rate),
                    "realized_rate": float(group.realized_rate.iloc[0]),
                    "bags": len(group),
                    "alert_rate": float(group.rejected.mean()),
                }
            )
        frames.append(bags)
    report = {
        "protocol": TRANSFER_PROTOCOL,
        "feature_set": bundle["feature_set"],
        "feature_transform": bundle.get("feature_transform", "identity"),
        "source_task_index": source["task_index"],
        "evaluation_task_index": metadata["task_index"],
        "source_control": source_control,
        "synthetic": bool(metadata.get("synthetic", False)),
        "contract_check": check,
        "sample_metrics": sample,
        "random_control_sample_alert_rate": float(np.mean(scores[random] >= threshold)),
        "dataset_results": summaries,
        "omitted_rates_rounding_to_zero": [rate for rate in RATES if rate not in rates],
        "sample_threshold": threshold,
        "count_calibration": bundle["count_calibration"],
        "evaluation_seed": seed,
        "test_originals": int(test.original_index.nunique()),
        "evaluation_features_sha256": metadata["features_sha256"],
        "target_used_for_fitting_or_calibration": False,
        "limitations": [
            "Source threshold has no target-domain false-positive guarantee.",
            "Bags reuse a finite pool; they are not independent tasks or seeds.",
            "Synthetic attacks are generated for benchmark originals, including held-out detection originals; this is not unseen-attack generalization.",
            "Clean historical checkpoints and early source data are assumed available.",
            "Detection does not establish the benefit of filtering during continual training.",
        ],
    }
    return report, predictions, pd.concat(frames, ignore_index=True)


def write_report(report, output):
    sample = report["sample_metrics"]
    lines = [
        "# Frozen source-to-target evaluation",
        "",
        "SYNTHETIC FIXTURE ONLY"
        if report["synthetic"]
        else "Real feature experiment; inspect attack-effectiveness metrics alongside detection.",
        "",
        "Source task: {}; evaluated task: {}; source control: {}.".format(
            report["source_task_index"],
            report["evaluation_task_index"],
            report["source_control"],
        ),
        "Feature set: {}. AUC: {:.4f}; clean FPR: {:.2%}; poison TPR: {:.2%}.".format(
            report["feature_set"],
            sample["roc_auc"],
            sample["clean_fpr"],
            sample["poison_tpr"],
        ),
        "Random-control sample alerts: {:.2%}.".format(
            report["random_control_sample_alert_rate"]
        ),
        "",
        "| Alternative | Actual fraction | Bag alert rate |",
        "| --- | ---: | ---: |",
    ]
    for row in report["dataset_results"]:
        lines.append(
            "| {} | {:.2%} | {:.2%} |".format(
                row["alternative"], row["realized_rate"], row["alert_rate"]
            )
        )
    lines += [
        "",
        "All thresholds and preprocessing were fitted on the source task only.",
        "",
    ]
    lines.extend("- " + item for item in report["limitations"])
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_curves(report, predictions, output):
    """Descriptive test curves only; never select a new threshold from them."""
    from sklearn.metrics import roc_curve

    paired = predictions.loc[predictions.view.isin(["clean", "poison"])]
    fpr, tpr, thresholds = roc_curve(paired.detector_label, paired.score)
    pd.DataFrame({"fpr": fpr, "tpr": tpr, "threshold": thresholds}).to_csv(
        output / "sample_roc.csv", index=False
    )
    pd.DataFrame(report["dataset_results"]).to_csv(output / "rates.csv", index=False)
    os.environ["MPLCONFIGDIR"] = str(output / ".matplotlib")
    os.environ["FONTCONFIG_FILE"] = str(DETECTOR_ROOT / "plot-fontconfig.xml")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(fpr, tpr, label="Test ROC (descriptive)")
    sample = report["sample_metrics"]
    axes[0].scatter(
        [sample["clean_fpr"]], [sample["poison_tpr"]], label="Frozen source threshold"
    )
    axes[0].set(xlabel="Clean FPR", ylabel="Poison TPR", xlim=(0, 1), ylim=(0, 1))
    axes[0].legend(fontsize=8)
    for alternative in ("poison", "random_control"):
        rows = [r for r in report["dataset_results"] if r["alternative"] == alternative]
        axes[1].plot(
            [r["realized_rate"] for r in rows],
            [r["alert_rate"] for r in rows],
            marker="o",
            label=alternative,
        )
    axes[1].set(
        xlabel="Actual contaminated fraction", ylabel="Bag alert rate", ylim=(0, 1.02)
    )
    axes[1].legend(fontsize=8)
    if report["synthetic"]:
        fig.suptitle("SYNTHETIC FIXTURE ONLY — not experimental evidence")
    fig.tight_layout()
    fig.savefig(output / "detection_curves.png", dpi=160)
    plt.close(fig)


def main(args):
    output = fresh_output(args.output_dir)
    table, metadata = read_feature_table(args.features)
    if args.command == "fit":
        bundle = fit_source(
            table,
            metadata,
            feature_set=args.feature_set,
            task_size=args.task_size,
            alpha=args.alpha,
        )
        bundle["input_file_sha256"] = sha256_file(args.features)
        output.mkdir(parents=True)
        save_frozen_bundle(bundle, output)
        save_json(
            {k: v for k, v in bundle.items() if k not in ("scaler", "classifier")},
            output / "fit_metrics.json",
        )
    else:
        bundle = load_frozen_bundle(args.bundle)
        report, predictions, bags = evaluate(
            table,
            metadata,
            bundle,
            source_control=args.source_control,
            repeats=args.bags_per_rate,
        )
        report["bundle_sha256"] = sha256_file(args.bundle)
        output.mkdir(parents=True)
        save_json(report, output / "evaluation_metrics.json")
        predictions.to_csv(output / "sample_predictions.csv", index=False)
        bags.to_csv(output / "bag_predictions.csv", index=False)
        write_report(report, output)
        write_curves(report, predictions, output)
    print("Saved:", output)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="command", required=True)
    for name in ("fit", "evaluate"):
        p = subs.add_parser(name)
        p.add_argument("--features", required=True)
        p.add_argument("--output-dir", required=True)
        if name == "fit":
            p.add_argument(
                "--feature-set",
                choices=list(FEATURE_GROUPS) + list(EXPERIMENTAL_FEATURE_GROUPS),
                default="portable",
            )
            p.add_argument("--task-size", type=int, default=150)
            p.add_argument("--alpha", type=float, default=0.05)
        else:
            p.add_argument("--bundle", required=True)
            p.add_argument("--source-control", action="store_true")
            p.add_argument("--bags-per-rate", type=int, default=1000)
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
