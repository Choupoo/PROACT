"""Predicted-gradient detector, development-only threshold selection and ablation.

Training uses clean/attack labels. Feature extraction and inference do not use
image-class labels. No target fitting; heuristic calibration is not a guarantee
under task shift. Legacy experiments retain their original definitions.
"""

import argparse
import copy

import numpy as np
from scipy.special import expit, logit

from detector import transfer_detector as base
from detector.calibration import count_decision, fit_count_calibration
from detector.common import save_json
from detector.io_utils import read_feature_table, save_frozen_bundle
from detector.train_detector import predict_scores, select_clean_threshold
from detector.transfer_core import assert_transfer

RULES = ("source_quantile", "history_quantile", "history_mad")


def check_predicted(table, metadata):
    base.validate_table(table, metadata)
    if metadata["label_mode"] != "predicted" or set(table.label_mode) != {"predicted"}:
        raise ValueError("Re-extract predicted gradients; ground_truth CSVs cannot be relabelled.")


def score_images(descriptors, bundle):
    """Deployment interface: descriptors only, with no labels/views/splits required."""
    if bundle.get("class_label_policy") != "predicted":
        raise ValueError("Deployment requires a predicted-gradient bundle.")
    columns = bundle["feature_columns"]
    if "true_class_probability" in columns:
        raise ValueError("True-class probability is unavailable at inference.")
    values = descriptors[columns].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite descriptors.")
    return predict_scores(descriptors, columns, bundle["scaler"], bundle["classifier"])


def predict_dataset(descriptors, bundle):
    if len(descriptors) != bundle["task_size"]:
        raise ValueError("Dataset size differs from frozen calibration.")
    scores = score_images(descriptors, bundle)
    count = int(np.sum(scores >= bundle["threshold"]))
    reject, _ = count_decision(np.array([count]), bundle["count_calibration"])
    return {"alert": bool(reject[0]), "suspicious_count": count,
            "sample_threshold": bundle["threshold"],
            "interpretation": "Risk alert; not a calibrated poisoning probability."}


def calibrate(bundle, histories, rule, alpha=0.05, mad_multiplier=3.0):
    """Use only clean validation/reserve splits of declared development tasks.

    Source_quantile reproduces the old source-only heuristic. History rules
    use the maximum task-wise cutoff and the most conservative count cutoff.
    Historical calibration knows which historical data are clean; this is a
    supervised detector, not the strictly label-free Rank method.
    """
    if rule not in RULES or not histories:
        raise ValueError("Unknown rule or empty historical calibration.")
    if not np.isfinite(mad_multiplier) or mad_multiplier <= 0:
        raise ValueError("MAD multiplier must be finite and positive.")
    result = copy.deepcopy(bundle)
    chosen = histories[:1] if rule == "source_quantile" else histories
    cutoffs, reserves, identities = [], [], []
    source = bundle["source_metadata"]
    for i, (table, meta) in enumerate(chosen):
        check_predicted(table, meta)
        if i == 0:
            if meta["features_sha256"] != source["features_sha256"]:
                raise ValueError("First calibration history must be the fitted source.")
        else:
            assert_transfer(source, meta, bundle["feature_columns"])
        if any(meta.get(k) != source.get(k) for k in ("cl_method", "checkpoint_seed")):
            raise ValueError("Calibration must use the fitted CL method and seed.")
        if meta["task_index"] in [r["task"] for r in identities]:
            raise ValueError("Repeated calibration task.")
        validation = table.loc[(table.split == "validation") & (table.view == "clean")]
        reserve = table.loc[(table.split == "reserve") & (table.view == "clean")]
        if validation.empty or len(reserve) < bundle["task_size"]:
            raise ValueError("Insufficient clean validation/reserve originals.")
        scores = score_images(validation, result)
        quantile = select_clean_threshold(scores, alpha)
        if rule == "history_mad":
            logits = logit(np.clip(scores, 1e-12, 1 - 1e-12))
            center = np.median(logits)
            scale = 1.4826 * np.median(np.abs(logits - center))
            # Never lower the observed clean quantile, including tied scores.
            quantile = max(quantile, float(expit(center + mad_multiplier * scale)))
        cutoffs.append(float(quantile))
        reserves.append(score_images(reserve, result))
        identities.append({"task": meta["task_index"], "features_sha256": meta["features_sha256"],
                           "validation_originals": len(validation), "reserve_originals": len(reserve)})
    threshold = max(cutoffs)
    counts = [fit_count_calibration(r >= threshold, bundle["task_size"], alpha) for r in reserves]
    count = copy.deepcopy(max(counts, key=lambda c: c["sample_fpr_upper_bound"]))
    count["assumptions"] = "Maximum historical-task count bound; empirical heuristic under task shift, no target FPR guarantee."
    result.update(threshold=threshold, count_calibration=count,
                  threshold_rule=rule, calibration_history=identities,
                  taskwise_sample_thresholds=cutoffs, taskwise_count_calibration=counts,
                  mad_multiplier=mad_multiplier,
                  supervision="Source clean/attack fitting; historical clean validation/reserve calibration.",
                  count_bound_scope="Development histories only; no target-task guarantee.")
    return result


def select_rule(reports, alpha, fixed_rule=None):
    """Prespecified development criterion; random controls are not negatives.

    Feasible: sample and dataset clean alerts <= alpha. Among feasible rules,
    maximize mean bag detection at 10/25/50/100% poison. If none are feasible,
    choose smallest worst clean rate and explicitly report failed feasibility.
    With fixed_rule, retain the prespecified rule even if another is feasible.
    """
    if fixed_rule is not None and fixed_rule not in RULES:
        raise ValueError("Unknown prespecified threshold rule.")
    rows = []
    for rule in RULES:
        report = reports[rule]
        clean = next(r["alert_rate"] for r in report["dataset_results"] if r["realized_rate"] == 0)
        power = np.mean([r["alert_rate"] for r in report["dataset_results"]
                         if r["alternative"] == "poison" and r["requested_rate"] >= 0.1])
        fpr = report["sample_metrics"]["clean_fpr"]
        rows.append({"rule": rule, "sample_fpr": fpr, "dataset_fpr": clean,
                     "poison_power": float(power), "feasible": max(fpr, clean) <= alpha})
    feasible = [r for r in rows if r["feasible"]]
    if fixed_rule is not None:
        selected = next(r for r in rows if r["rule"] == fixed_rule)
        criterion_met = selected["feasible"]
    elif feasible:
        selected = min(feasible, key=lambda r: (-r["poison_power"], RULES.index(r["rule"])))
        criterion_met = True
    else:
        selected = min(rows, key=lambda r: (max(r["sample_fpr"], r["dataset_fpr"]), -r["poison_power"], RULES.index(r["rule"])))
        criterion_met = False
    return {"selected_rule": selected["rule"], "development_criterion_met": bool(criterion_met),
            "candidates": rows, "selection_data": (
                "Rule fixed before this experiment; development test used for diagnostics only, never final target"
                if fixed_rule else "Development task test split; never final target"),
            "selection_mode": "prespecified" if fixed_rule else "development_comparison",
            "warning": None if criterion_met else (
                "Prespecified rule failed the development clean-error criterion; retained without switching rules."
                if fixed_rule else "No candidate met the development clean-error criterion. Final evaluation remains diagnostic.")}


def develop(source_path, history_paths, output, task_size=150, repeats=100, alpha=0.05, ablations=False,
            threshold_rule=None):
    output = base.fresh_output(output)
    source = read_feature_table(source_path)
    histories = [source] + [read_feature_table(p) for p in history_paths]
    if len(histories) < 2:
        raise ValueError("Need a separate development task before target evaluation.")
    tasks = [m["task_index"] for _, m in histories]
    if tasks != sorted(set(tasks)):
        raise ValueError("Development tasks must be distinct and chronologically ordered.")
    for t, m in histories:
        check_predicted(t, m)
        if m.get("cl_method") != source[1].get("cl_method") or m.get("checkpoint_seed") != source[1].get("checkpoint_seed"):
            raise ValueError("Development must use the same CL method and seed.")
    output.mkdir(parents=True)
    fitted = base.fit_source(*source, feature_set="inference_full", task_size=task_size, alpha=alpha)
    reports, candidates = {}, {}
    for rule in RULES:
        candidates[rule] = calibrate(fitted, histories, rule, alpha)
        reports[rule], _, _ = base.evaluate(*histories[-1], candidates[rule], repeats=repeats)
    selection = select_rule(reports, alpha, fixed_rule=threshold_rule)
    save_json(selection, output / "threshold_selection.json")
    save_json(reports, output / "development_metrics.json")
    # Save every threshold candidate for honest frozen target comparison.
    for rule, bundle in candidates.items():
        save_frozen_bundle(bundle, output / "thresholds" / rule)
    variants = list(base.INFERENCE_GROUPS) if ablations else ["inference_full"]
    for group in variants:
        bundle = fitted if group == "inference_full" else base.fit_source(
            *source, feature_set=group, task_size=task_size, alpha=alpha)
        bundle = calibrate(bundle, histories, selection["selected_rule"], alpha)
        bundle["threshold_selection"] = selection
        save_frozen_bundle(bundle, output / "ablations" / group)
        report = {k: v for k, v in bundle.items() if k not in ("scaler", "classifier")}
        save_json(report, output / "ablations" / group / "fit_metrics.json")
    save_json({"label_mode": "predicted", "forbidden_feature": "true_class_probability",
               "source": str(source_path), "histories": list(map(str, history_paths)),
               "calibration_tasks": tasks, "selection": selection,
               "prespecified_threshold_rule": threshold_rule,
               "variants": variants,
               "target_used_for_fitting_or_calibration": False}, output / "freeze.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--histories", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--task-size", type=int, default=150)
    parser.add_argument("--bags-per-rate", type=int, default=100)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--ablations", action="store_true", help="Deferred until methodology is agreed; disabled by default.")
    parser.add_argument("--threshold-rule", choices=RULES,
                        help="Keep this rule fixed, including when the development criterion fails.")
    args = parser.parse_args()
    develop(args.source, args.histories, args.output_dir, args.task_size, args.bags_per_rate, args.alpha,
            args.ablations, args.threshold_rule)


if __name__ == "__main__":
    main()
