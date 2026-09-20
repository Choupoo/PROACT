import argparse
import json

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from detector.calibration import count_decision, fit_count_calibration
from detector.common import save_json, sha256_file
from detector.io_utils import (
    assert_compatible_provenance,
    ensure_output_path,
    feature_provenance,
    load_frozen_bundle,
    read_feature_table,
    save_frozen_bundle,
)
from detector.train_detector import (
    binary_metrics,
    predict_scores,
    select_clean_threshold,
    validate_feature_table,
)

AGGREGATE_COLUMNS = [
    "top_tail_mean",
    "suspicious_fraction",
    "score_mean",
    "score_std",
    "score_median",
    "score_q75",
    "score_q90",
    "score_q95",
    "score_max",
]
DEFAULT_RATES = (0.0, 0.1, 0.25, 0.5, 1.0)
DECISION_RULES = ("count_bound", "legacy_lr")
DEPENDENCE_NOTE = (
    "Bags contain distinct original images internally, but repeated bags reuse "
    "a finite pool. Rates describe this simulation, not independent datasets "
    "or a population-level guarantee."
)
SCORE_NOTE = (
    "dataset_poison_score is a supervised logistic-regression score conditional "
    "on the simulated clean/poison mixtures, not a calibrated deployment posterior."
)


def aggregate_scores(scores, sample_threshold, top_fraction=0.1):
    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 1 or not values.size or not np.isfinite(values).all():
        raise ValueError("Sample scores must be a nonempty finite vector.")
    if np.any((values < 0) | (values > 1)):
        raise ValueError("Sample scores must lie in [0, 1].")
    if not np.isfinite(top_fraction) or not 0 < top_fraction <= 1:
        raise ValueError("top_fraction must lie in (0, 1].")
    if not np.isfinite(sample_threshold):
        raise ValueError("The frozen sample threshold must be finite.")
    count = max(1, int(np.ceil(values.size * top_fraction)))
    return dict(
        zip(
            AGGREGATE_COLUMNS,
            [
                float(np.sort(values)[-count:].mean()),
                float(np.mean(values >= sample_threshold)),
                float(values.mean()),
                float(values.std()),
                float(np.median(values)),
                float(np.quantile(values, 0.75)),
                float(np.quantile(values, 0.9)),
                float(np.quantile(values, 0.95)),
                float(values.max()),
            ],
        )
    )


def validate_numeric_features(table, columns):
    if table.empty or table.columns.duplicated().any():
        raise ValueError("Feature table must be nonempty and have unique columns.")
    missing = set(columns) - set(table)
    if missing:
        raise ValueError("Missing frozen features: {}".format(sorted(missing)))
    for column in columns:
        if not pd.api.types.is_numeric_dtype(table[column]):
            raise ValueError("Feature {} must be numeric.".format(column))
    if not np.isfinite(table[columns].to_numpy(dtype=np.float64)).all():
        raise ValueError("Inference features must all be finite.")


def score_feature_pool(table, sample_bundle):
    columns = sample_bundle["feature_columns"]
    validate_numeric_features(table, columns)
    result = table[["original_index", "view"]].copy()
    result["sample_poison_score"] = predict_scores(
        table, columns, sample_bundle["scaler"], sample_bundle["classifier"]
    )
    return result


def simulate_bags(
    scored_pool,
    rates,
    task_size,
    repeats,
    seed,
    sample_threshold,
    top_fraction=0.1,
    alternative="poison",
):
    if not isinstance(task_size, (int, np.integer)) or task_size <= 0:
        raise ValueError("task_size must be a positive integer.")
    if not isinstance(repeats, (int, np.integer)) or repeats <= 0:
        raise ValueError("repeats must be a positive integer.")
    rates = [float(rate) for rate in rates]
    if not rates or any(not np.isfinite(rate) or not 0 <= rate <= 1 for rate in rates):
        raise ValueError("Mixture rates must be finite values in [0, 1].")
    if scored_pool[["original_index", "view"]].duplicated().any():
        raise ValueError("Each original/view pair must occur exactly once.")
    pivot = scored_pool.pivot(
        index="original_index", columns="view", values="sample_poison_score"
    )
    if "clean" not in pivot or (
        any(rate > 0 for rate in rates) and alternative not in pivot
    ):
        raise ValueError("The requested clean/alternative views are unavailable.")
    required = ["clean"] + ([alternative] if any(rate > 0 for rate in rates) else [])
    if not np.isfinite(pivot[required].to_numpy(dtype=np.float64)).all():
        raise ValueError("Every original image must have finite requested view scores.")
    if task_size > len(pivot):
        raise ValueError(
            "task_size exceeds the number of distinct available originals."
        )
    ids = pivot.index.to_numpy()
    clean = pivot["clean"].to_numpy(dtype=np.float64)
    other = (
        pivot[alternative].to_numpy(dtype=np.float64) if alternative in pivot else clean
    )
    rng = np.random.default_rng(int(seed))
    rows = []
    for rate in rates:
        contamination_count = int(np.floor(task_size * rate + 0.5))
        for repeat in range(repeats):
            positions = rng.choice(len(ids), size=task_size, replace=False)
            scores = clean[positions].copy()
            scores[:contamination_count] = other[positions[:contamination_count]]
            row = aggregate_scores(scores, sample_threshold, top_fraction)
            row.update(
                {
                    "bag_index": repeat,
                    "requested_rate": rate,
                    "realized_rate": contamination_count / task_size,
                    "contamination_count": contamination_count,
                    "task_size": task_size,
                    "alternative_view": alternative,
                    "original_indices": "|".join(str(int(i)) for i in ids[positions]),
                }
            )
            rows.append(row)
    return pd.DataFrame(rows)


def sample_seen_indices(bundle):
    required = {"fit_original_indices", "calibration_original_indices", "provenance"}
    if not required.issubset(bundle):
        raise ValueError(
            "Re-fit the sample bundle with --fit-only to record provenance and ID pools."
        )
    fit = set(int(i) for i in bundle["fit_original_indices"])
    calibration = set(int(i) for i in bundle["calibration_original_indices"])
    if fit & calibration:
        raise ValueError("Sample fitting and calibration IDs overlap.")
    return fit | calibration


def dataset_scores(bags, bundle):
    values = bundle["scaler"].transform(
        bags[AGGREGATE_COLUMNS].to_numpy(dtype=np.float64)
    )
    return bundle["classifier"].predict_proba(values)[:, 1]


def task_decisions(bags, bundle, scores):
    if bundle.get("decision_rule", "legacy_lr") == "legacy_lr":
        return scores >= bundle["threshold"], None
    if bundle["decision_rule"] != "count_bound":
        raise ValueError("Unsupported dataset decision rule.")
    counts = np.rint(bags["suspicious_fraction"].to_numpy() * bundle["task_size"])
    return count_decision(counts, bundle["count_calibration"])


def fit_dataset_detector(
    features,
    sample_bundle,
    metadata,
    task_size=150,
    repeats=1000,
    top_fraction=0.1,
    target_clean_frr=0.05,
    split_seed=20260724,
    bag_seed=20260725,
    classifier_seed=20260726,
    evaluation_seed=20260727,
    rates=DEFAULT_RATES,
    decision_rule="count_bound",
):
    validate_feature_table(features)
    if decision_rule not in DECISION_RULES:
        raise ValueError("Unsupported dataset decision rule.")
    provenance = feature_provenance(metadata)
    assert_compatible_provenance(sample_bundle["provenance"], provenance)
    seen = sample_seen_indices(sample_bundle)
    rates = tuple(float(rate) for rate in rates)
    if 0.0 not in rates or not any(rate > 0 for rate in rates):
        raise ValueError("Fitting requires clean and contaminated mixture rates.")
    if any(rate > 0 and int(np.floor(task_size * rate + 0.5)) == 0 for rate in rates):
        raise ValueError(
            "A positive requested rate rounds to zero contaminated samples."
        )
    reserve = features.loc[features["split"] == "reserve"]
    reserve_ids = np.sort(reserve["original_index"].unique().astype(np.int64))
    if set(reserve_ids) & seen:
        raise ValueError(
            "Dataset reserve images overlap sample fitting/calibration images."
        )
    if len(reserve_ids) < 2 * task_size:
        raise ValueError("Reserve must contain at least 2 * task_size distinct images.")
    rng = np.random.default_rng(int(split_seed))
    rng.shuffle(reserve_ids)
    midpoint = len(reserve_ids) // 2
    fit_ids, calibration_ids = reserve_ids[:midpoint], reserve_ids[midpoint:]
    fit_pool = score_feature_pool(
        reserve.loc[reserve["original_index"].isin(fit_ids)], sample_bundle
    )
    calibration_pool = score_feature_pool(
        reserve.loc[
            (reserve["original_index"].isin(calibration_ids))
            & (reserve["view"] == "clean")
        ],
        sample_bundle,
    )
    fit_bags = simulate_bags(
        fit_pool,
        rates,
        task_size,
        repeats,
        bag_seed,
        sample_bundle["threshold"],
        top_fraction,
    )
    calibration_bags = simulate_bags(
        calibration_pool,
        [0.0],
        task_size,
        repeats,
        bag_seed + 1,
        sample_bundle["threshold"],
        top_fraction,
    )
    scaler = StandardScaler()
    x_fit = scaler.fit_transform(fit_bags[AGGREGATE_COLUMNS].to_numpy(dtype=np.float64))
    labels = (fit_bags["contamination_count"].to_numpy() > 0).astype(np.int64)
    classifier = LogisticRegression(
        C=1.0,
        class_weight="balanced",
        solver="lbfgs",
        max_iter=2000,
        random_state=int(classifier_seed),
    )
    classifier.fit(x_fit, labels)
    bundle = {
        "kind": "supervised_dataset_detector_v2",
        "decision_rule": decision_rule,
        "sample_bundle": sample_bundle,
        "provenance": provenance,
        "scaler": scaler,
        "classifier": classifier,
        "aggregate_columns": list(AGGREGATE_COLUMNS),
        "task_size": int(task_size),
        "top_fraction": float(top_fraction),
        "fit_mixture_rates": list(rates),
        "evaluation_mixture_rates": list(rates),
        "bags_per_rate": int(repeats),
        "target_clean_frr": float(target_clean_frr),
        "reserve_split_seed": int(split_seed),
        "bag_seed": int(bag_seed),
        "classifier_seed": int(classifier_seed),
        "evaluation_seed": int(evaluation_seed),
        "fit_original_indices": sorted(int(i) for i in fit_ids),
        "calibration_original_indices": sorted(int(i) for i in calibration_ids),
        "sample_seen_original_indices": sorted(seen),
        "supervision": "Sample clean/poison labels and simulated reserve-fit bag labels; reserve-calibration CLEAN labels.",
        "score_interpretation": SCORE_NOTE,
        "dependence_note": DEPENDENCE_NOTE,
        "test_evaluated_during_fit": False,
    }
    calibration_scores = dataset_scores(calibration_bags, bundle)
    bundle["threshold"] = select_clean_threshold(calibration_scores, target_clean_frr)
    bundle["top_tail_threshold"] = select_clean_threshold(
        calibration_bags["top_tail_mean"].to_numpy(), target_clean_frr
    )
    clean_alarms = (
        calibration_pool["sample_poison_score"].to_numpy() >= sample_bundle["threshold"]
    )
    bundle["count_calibration"] = fit_count_calibration(
        clean_alarms, task_size, target_clean_frr
    )
    primary_decisions, _ = task_decisions(calibration_bags, bundle, calibration_scores)
    report = {
        "decision_rule": decision_rule,
        "count_calibration": bundle["count_calibration"],
        "calibration_clean_frr": float(np.mean(primary_decisions)),
        "legacy_lr_calibration_clean_frr": float(
            np.mean(calibration_scores >= bundle["threshold"])
        ),
        "calibration_top_tail_clean_frr": float(
            np.mean(
                calibration_bags["top_tail_mean"].to_numpy()
                >= bundle["top_tail_threshold"]
            )
        ),
        "calibration_bags": int(len(calibration_bags)),
        "fit_bags": int(len(fit_bags)),
        "threshold": bundle["threshold"],
        "top_tail_threshold": bundle["top_tail_threshold"],
        "task_size": bundle["task_size"],
        "fit_original_images": len(fit_ids),
        "calibration_original_images": len(calibration_ids),
        "test_evaluated": False,
        "supervision": bundle["supervision"],
        "score_interpretation": SCORE_NOTE,
        "dependence_note": DEPENDENCE_NOTE,
    }
    return bundle, report


def evaluate_dataset_detector(features, bundle, metadata):
    validate_feature_table(features)
    assert_compatible_provenance(bundle["provenance"], feature_provenance(metadata))
    test = features.loc[features["split"] == "test"]
    forbidden = (
        set(bundle["fit_original_indices"])
        | set(bundle["calibration_original_indices"])
        | sample_seen_indices(bundle["sample_bundle"])
    )
    if set(test["original_index"]) & forbidden:
        raise ValueError(
            "Test images overlap sample/dataset fitting or calibration IDs."
        )
    pool = score_feature_pool(test, bundle["sample_bundle"])
    frames = []
    summaries = []
    for alternative in ("poison", "random_control"):
        bags = simulate_bags(
            pool,
            bundle["evaluation_mixture_rates"],
            bundle["task_size"],
            bundle["bags_per_rate"],
            bundle["evaluation_seed"],
            bundle["sample_bundle"]["threshold"],
            bundle["top_fraction"],
            alternative,
        )
        scores = dataset_scores(bags, bundle)
        bags["dataset_poison_score"] = scores
        bags["legacy_lr_rejected"] = scores >= bundle["threshold"]
        bags["rejected"], tail_bounds = task_decisions(bags, bundle, scores)
        if tail_bounds is not None:
            bags["count_tail_bound"] = tail_bounds
        bags["top_tail_rejected"] = (
            bags["top_tail_mean"] >= bundle["top_tail_threshold"]
        )
        for rate, group in bags.groupby("requested_rate", sort=True):
            positive_rate = float(group["rejected"].mean())
            summaries.append(
                {
                    "alternative_view": alternative,
                    "requested_rate": float(rate),
                    "realized_rate": float(group["realized_rate"].iloc[0]),
                    "contamination_count": int(group["contamination_count"].iloc[0]),
                    "bags": int(len(group)),
                    "positive_rate": positive_rate,
                    "legacy_lr_positive_rate": float(
                        group["legacy_lr_rejected"].mean()
                    ),
                    "metric": "clean_frr"
                    if rate == 0
                    else (
                        "poison_tpr"
                        if alternative == "poison"
                        else "random_control_positive_rate"
                    ),
                    "top_tail_positive_rate": float(group["top_tail_rejected"].mean()),
                }
            )
        frames.append(bags)
    supervised = test.loc[test["view"].isin(["clean", "poison"])]
    sample = bundle["sample_bundle"]
    sample_scores = predict_scores(
        supervised, sample["feature_columns"], sample["scaler"], sample["classifier"]
    )
    random_scores = pool.loc[
        pool["view"] == "random_control", "sample_poison_score"
    ].to_numpy()
    report = {
        "decision_rule": bundle.get("decision_rule", "legacy_lr"),
        "count_calibration": bundle.get("count_calibration"),
        "test_original_images": int(test["original_index"].nunique()),
        "sample_metrics": binary_metrics(
            supervised["detector_label"].to_numpy(), sample_scores, sample["threshold"]
        ),
        "random_control_sample_positive_rate": float(
            np.mean(random_scores >= sample["threshold"])
        ),
        "dataset_results": summaries,
        "frozen_threshold": bundle["threshold"],
        "frozen_top_tail_threshold": bundle["top_tail_threshold"],
        "test_used_for_fitting_or_calibration": False,
        "score_interpretation": SCORE_NOTE,
        "dependence_note": DEPENDENCE_NOTE,
    }
    return report, pd.concat(frames, ignore_index=True)


def predict_dataset(features, bundle, metadata):
    """Make one dataset decision from features alone, without label/view access."""
    assert_compatible_provenance(bundle["provenance"], feature_provenance(metadata))
    if len(features) != bundle["task_size"]:
        raise ValueError(
            "Incoming size must equal the frozen task_size {}.".format(
                bundle["task_size"]
            )
        )
    if "original_index" not in features or features["original_index"].isna().any():
        raise ValueError("Incoming rows need an original_index per unique image.")
    if features["original_index"].duplicated().any():
        raise ValueError("Incoming data must contain each original image only once.")
    sample = bundle["sample_bundle"]
    validate_numeric_features(features, sample["feature_columns"])
    scores = predict_scores(
        features, sample["feature_columns"], sample["scaler"], sample["classifier"]
    )
    aggregates = aggregate_scores(scores, sample["threshold"], bundle["top_fraction"])
    score = float(dataset_scores(pd.DataFrame([aggregates]), bundle)[0])
    decisions, tails = task_decisions(
        pd.DataFrame([aggregates]), bundle, np.array([score])
    )
    rejected = bool(decisions[0])
    return {
        "dataset_poison_score": score,
        "score_used_for_primary_decision": bundle.get("decision_rule", "legacy_lr")
        == "legacy_lr",
        "decision_rule": bundle.get("decision_rule", "legacy_lr"),
        "rejected": rejected,
        "decision": "reject" if rejected else "accept",
        "legacy_lr_rejected": bool(score >= bundle["threshold"]),
        "count_calibration": bundle.get("count_calibration"),
        "count_tail_bound": None if tails is None else float(tails[0]),
        "suspicious_count": int(np.count_nonzero(scores >= sample["threshold"])),
        "threshold": bundle["threshold"],
        "top_tail_rejected": bool(
            aggregates["top_tail_mean"] >= bundle["top_tail_threshold"]
        ),
        "task_size": len(features),
        "aggregates": aggregates,
        "uses_poison_labels_at_inference": False,
        "supervised_training": True,
        "score_interpretation": SCORE_NOTE,
    }


def main(args):
    output_dir = ensure_output_path(args.output_dir)
    features, metadata = read_feature_table(args.features)
    if args.command == "fit":
        if metadata.get("origin_role") != "paired_benchmark":
            raise ValueError(
                "Dataset supervised fitting requires paired_benchmark features."
            )
        sample_bundle = load_frozen_bundle(args.sample_bundle)
        bundle, report = fit_dataset_detector(
            features,
            sample_bundle,
            metadata,
            task_size=args.task_size,
            repeats=args.bags_per_rate,
            top_fraction=args.top_fraction,
            target_clean_frr=args.target_clean_frr,
            split_seed=args.split_seed,
            bag_seed=args.bag_seed,
            classifier_seed=args.classifier_seed,
            evaluation_seed=args.evaluation_seed,
            decision_rule=args.decision_rule,
        )
        bundle["sample_bundle_sha256"] = sha256_file(args.sample_bundle)
        bundle["features_sha256"] = sha256_file(args.features)
        output_dir.mkdir(parents=True, exist_ok=True)
        save_frozen_bundle(bundle, output_dir, filename="dataset_bundle.joblib")
        save_json(report, output_dir / "fit_metrics.json")
    else:
        bundle = load_frozen_bundle(args.bundle)
        if bundle.get("kind") not in {
            "supervised_dataset_detector_v1",
            "supervised_dataset_detector_v2",
        }:
            raise ValueError("Expected a frozen supervised dataset detector bundle.")
        if args.command == "evaluate":
            if metadata.get("origin_role") != "paired_benchmark":
                raise ValueError("Evaluation requires paired_benchmark features.")
            report, predictions = evaluate_dataset_detector(features, bundle, metadata)
            output_dir.mkdir(parents=True, exist_ok=True)
            predictions.to_csv(output_dir / "task_predictions.csv", index=False)
            pd.DataFrame(report["dataset_results"]).to_csv(
                output_dir / "rates.csv", index=False
            )
            save_json(report, output_dir / "evaluation_metrics.json")
        else:
            if metadata.get("origin_role") != "incoming":
                raise ValueError("Prediction requires incoming feature provenance.")
            report = predict_dataset(features, bundle, metadata)
            output_dir.mkdir(parents=True, exist_ok=True)
            save_json(report, output_dir / "prediction.json")
    print(json.dumps(report, indent=2))


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("fit", "evaluate", "predict"):
        subparser = subparsers.add_parser(command)
        subparser.add_argument("--features", required=True)
        subparser.add_argument("--output-dir", required=True)
        if command == "fit":
            subparser.add_argument("--sample-bundle", required=True)
            subparser.add_argument(
                "--decision-rule", choices=DECISION_RULES, default="count_bound"
            )
            subparser.add_argument("--task-size", type=int, default=150)
            subparser.add_argument("--bags-per-rate", type=int, default=1000)
            subparser.add_argument("--top-fraction", type=float, default=0.1)
            subparser.add_argument("--target-clean-frr", type=float, default=0.05)
            subparser.add_argument("--split-seed", type=int, default=20260724)
            subparser.add_argument("--bag-seed", type=int, default=20260725)
            subparser.add_argument("--classifier-seed", type=int, default=20260726)
            subparser.add_argument("--evaluation-seed", type=int, default=20260727)
        else:
            subparser.add_argument("--bundle", required=True)
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
