"""Train a paired supervised baseline with validation-only threshold calibration."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

from detector import (
    ACTIVATION_FEATURE_COLUMNS,
    BASE_FEATURE_COLUMNS,
    FEATURE_COLUMNS,
    FEATURE_PROTOCOL,
    HEAD_MODE,
    STAGE_GRAD_FEATURE_COLUMNS,
    STAGE_NAMES,
    TASK_COSINE_FEATURE_COLUMNS,
    UNCERTAINTY_FEATURE_COLUMNS,
)
from detector.common import save_json, sha256_file
from detector.io_utils import (
    assert_compatible_provenance,
    ensure_output_path,
    feature_provenance,
    load_frozen_bundle,
    read_feature_table,
    save_frozen_bundle,
)

FEATURE_SETS = {
    "baseline": BASE_FEATURE_COLUMNS,
    "stages": STAGE_GRAD_FEATURE_COLUMNS,
    "extended": FEATURE_COLUMNS,
    "parameters": None,
    "layers": None,
    "all": FEATURE_COLUMNS,
}
IDENTITY_COLUMNS = [
    "original_index",
    "source_index",
    "class_id",
    "split",
    "view",
    "detector_label",
]
INTEGER_COLUMNS = [
    "original_index",
    "source_index",
    "class_id",
    "detector_label",
    "head_seed",
    "attack_seed",
]
METADATA_COLUMNS = IDENTITY_COLUMNS + [
    "feature_protocol",
    "head_mode",
    "head_seed",
    "attack_seed",
    "label_mode",
]
THRESHOLD_RULE = (
    "With prediction score >= threshold, allow at most floor(target_clean_fpr * "
    "n_validation_clean) clean positives. Set threshold immediately above the "
    "next excluded score using numpy.nextafter, conservatively excluding ties. "
    "This controls observed validation FPR only, not population or test FPR."
)


def feature_sets(features):
    """Discover frozen scalar, stage, layer, and parameter-tensor feature groups."""
    parameter_columns = sorted(c for c in features if c.startswith("grad_norm_param__"))
    layer_columns = sorted(c for c in features if c.startswith("grad_norm_layer__"))
    task_columns = sorted(
        c
        for c in features
        if c.startswith("grad_cosine_task_") and c.rsplit("_", 1)[-1].isdigit()
    )
    return {
        "baseline": list(BASE_FEATURE_COLUMNS),
        "stages": list(STAGE_GRAD_FEATURE_COLUMNS),
        "extended": list(FEATURE_COLUMNS),
        "parameters": parameter_columns,
        "layers": layer_columns,
        "all": list(FEATURE_COLUMNS) + parameter_columns + layer_columns + task_columns,
    }


def validate_feature_table(features):
    """Reject incomplete pairs, inconsistent provenance, and invalid values."""
    required = set(METADATA_COLUMNS + FEATURE_COLUMNS)
    missing = required - set(features.columns)
    if missing:
        raise KeyError("Feature table is missing columns: {}".format(sorted(missing)))
    if features.columns.duplicated().any():
        raise RuntimeError("Feature table contains duplicate column names.")
    if features.empty or features[list(required)].isna().any().any():
        raise RuntimeError(
            "Feature table is empty or contains missing features/metadata."
        )

    for column in INTEGER_COLUMNS:
        if not pd.api.types.is_numeric_dtype(features[column]):
            raise RuntimeError("{} must contain numeric integers.".format(column))
        values = features[column].to_numpy(dtype=np.float64)
        if (
            not np.isfinite(values).all()
            or not np.equal(values, np.floor(values)).all()
            or np.any(values < -(2**63))
            or np.any(values >= 2**63)
        ):
            raise RuntimeError(
                "{} must contain finite int64-compatible integers.".format(column)
            )
        if column != "detector_label" and np.any(values < 0):
            raise RuntimeError("{} must be nonnegative.".format(column))

    if set(features["feature_protocol"]) != {FEATURE_PROTOCOL}:
        raise RuntimeError(
            "Unexpected feature protocol; re-extract the current feature schema."
        )
    if set(features["head_mode"]) != {HEAD_MODE}:
        raise RuntimeError("Features were not produced with defender_fixed head mode.")
    for column in ("head_seed", "attack_seed", "label_mode"):
        if features[column].nunique() != 1:
            raise RuntimeError(
                "The single-attack baseline requires one {}.".format(column)
            )

    if not set(features["label_mode"]).issubset({"ground_truth", "predicted"}):
        raise RuntimeError("label_mode must be ground_truth or predicted.")
    required_splits = {"train", "validation", "test"}
    allowed_splits = required_splits | {"reserve"}
    allowed_views = {"clean", "poison", "random_control"}
    if not required_splits.issubset(set(features["split"])) or not set(
        features["split"]
    ).issubset(allowed_splits):
        raise RuntimeError(
            "Features must contain train, validation and test; reserve is optional."
        )
    if set(features["view"]) != allowed_views:
        raise RuntimeError(
            "Feature table must contain clean, poison and random_control."
        )
    if not features["class_id"].between(0, 9).all():
        raise RuntimeError("Expected task-local class_id values in 0..9.")

    for original_index, group in features.groupby("original_index"):
        if len(group) != 3 or set(group["view"]) != allowed_views:
            raise RuntimeError(
                "original_index {} must have exactly three distinct views.".format(
                    original_index
                )
            )
        for column in ("split", "class_id", "source_index"):
            if group[column].nunique() != 1:
                raise RuntimeError(
                    "Views of original_index {} disagree on {}.".format(
                        original_index, column
                    )
                )
        labels = dict(zip(group["view"], group["detector_label"]))
        if labels != {"clean": 0, "poison": 1, "random_control": -1}:
            raise RuntimeError(
                "Incorrect detector labels for original_index {}.".format(
                    original_index
                )
            )

    sources = features[["original_index", "source_index"]].drop_duplicates()
    if sources["source_index"].duplicated().any():
        raise RuntimeError(
            "Different original images share the same attack source_index."
        )

    all_columns = feature_sets(features)["all"]
    for column in all_columns:
        if not pd.api.types.is_numeric_dtype(features[column]):
            raise RuntimeError("Feature {} must be numeric.".format(column))
    if not np.isfinite(features[all_columns].to_numpy(dtype=np.float64)).all():
        raise RuntimeError("Feature table contains non-finite feature values.")


def select_clean_threshold(clean_scores, target_clean_fpr):
    """Choose a >= threshold with empirical clean FPR at most the target."""
    scores = np.asarray(clean_scores, dtype=np.float64)
    target = float(target_clean_fpr)
    if scores.ndim != 1 or scores.size == 0 or not np.isfinite(scores).all():
        raise ValueError(
            "clean_scores must be a non-empty finite one-dimensional array."
        )
    if not np.isfinite(target) or not 0.0 <= target <= 1.0:
        raise ValueError("target_clean_fpr must lie in [0, 1].")
    if np.any((scores < 0.0) | (scores > 1.0)):
        raise ValueError("Clean classifier probabilities must lie in [0, 1].")

    # Decrement if floating-point multiplication rounded the budget upward.
    allowed = int(np.floor(target * scores.size))
    while allowed > 0 and allowed / scores.size > target:
        allowed -= 1
    if allowed == scores.size:
        return float(scores.min())
    descending = np.sort(scores)[::-1]
    return float(np.nextafter(descending[allowed], np.inf))


def binary_metrics(y_true, probabilities, threshold):
    """Summarize ranking and decisions for both detector classes."""
    predictions = (probabilities >= threshold).astype(np.int64)
    matrix = confusion_matrix(y_true, predictions, labels=[0, 1])
    tn, fp, fn, tp = matrix.ravel()
    return {
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "average_precision": float(average_precision_score(y_true, probabilities)),
        "threshold": float(threshold),
        "clean_fpr": float(fp / max(tn + fp, 1)),
        "poison_tpr": float(tp / max(tp + fn, 1)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predictions)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "confusion_matrix": matrix.tolist(),
    }


def fit_detector(train, columns, classifier_seed):
    """Fit scaling and logistic regression using training rows only."""
    scaler = StandardScaler()
    x_train = scaler.fit_transform(train[columns].to_numpy(dtype=np.float64))
    classifier = LogisticRegression(
        C=1.0,
        penalty="l2",
        solver="lbfgs",
        max_iter=2000,
        random_state=int(classifier_seed),
    )
    classifier.fit(x_train, train["detector_label"].to_numpy(dtype=np.int64))
    return scaler, classifier


def predict_scores(table, columns, scaler, classifier):
    """Return sample scores; these are not calibrated dataset probabilities."""
    values = scaler.transform(table[columns].to_numpy(dtype=np.float64))
    return classifier.predict_proba(values)[:, 1]


def calibrate_validation(validation, scores, target_clean_fpr):
    """Select the clean threshold and report held-out validation performance."""
    labels = validation["detector_label"].to_numpy(dtype=np.int64)
    threshold = select_clean_threshold(scores[labels == 0], target_clean_fpr)
    return threshold, binary_metrics(labels, scores, threshold)


def compare_feature_sets(train, validation, classifier_seed, target_clean_fpr):
    """Compare fixed feature sets on identical train/validation rows only."""
    groups = feature_sets(train)
    candidates = [(name, columns) for name, columns in groups.items() if columns] + [
        ("stage_" + stage, [column])
        for stage, column in zip(STAGE_NAMES, STAGE_GRAD_FEATURE_COLUMNS)
    ]
    candidates += [
        (column, [column]) for column in groups["layers"] + groups["parameters"]
    ]
    families = {
        "task_cosines": TASK_COSINE_FEATURE_COLUMNS,
        "uncertainty": UNCERTAINTY_FEATURE_COLUMNS,
        "activation": ACTIVATION_FEATURE_COLUMNS,
        "stages": STAGE_GRAD_FEATURE_COLUMNS,
    }
    for family, columns in families.items():
        candidates.append(("baseline_plus_" + family, BASE_FEATURE_COLUMNS + columns))
        candidates.append(
            (
                "extended_without_" + family,
                [column for column in FEATURE_COLUMNS if column not in columns],
            )
        )
    rows = []
    for name, columns in candidates:
        scaler, classifier = fit_detector(train, columns, classifier_seed)
        scores = predict_scores(validation, columns, scaler, classifier)
        threshold, metrics = calibrate_validation(validation, scores, target_clean_fpr)
        rows.append(
            {
                "feature_set": name,
                "feature_columns": "|".join(columns),
                "n_features": len(columns),
                "validation_roc_auc": metrics["roc_auc"],
                "validation_average_precision": metrics["average_precision"],
                "validation_poison_tpr": metrics["poison_tpr"],
                "validation_clean_fpr": metrics["clean_fpr"],
                "target_clean_fpr": float(target_clean_fpr),
                "threshold": threshold,
                "evaluation_split": "validation",
            }
        )
    comparison = (
        pd.DataFrame(rows)
        .sort_values(
            ["validation_roc_auc", "validation_poison_tpr", "feature_set"],
            ascending=[False, False, True],
        )
        .reset_index(drop=True)
    )
    comparison.insert(0, "validation_rank", np.arange(1, len(comparison) + 1))
    return comparison


def save_predictions(table, probabilities, threshold, path):
    """Write scores alongside the source identity of each evaluated view."""
    predictions = table[IDENTITY_COLUMNS].copy()
    predictions["poison_probability"] = probabilities
    predictions["prediction"] = (probabilities >= threshold).astype(np.int64)
    predictions.to_csv(path, index=False)


def evaluate_frozen_sample(features, bundle, metadata, output_dir):
    """Score held-out test views with a verified frozen model and threshold."""
    assert_compatible_provenance(bundle["provenance"], feature_provenance(metadata))
    test_all = features.loc[features["split"] == "test"]
    forbidden = set(bundle["fit_original_indices"]) | set(
        bundle["calibration_original_indices"]
    )
    if set(test_all["original_index"]) & forbidden:
        raise ValueError("Test originals overlap sample fitting/calibration IDs.")
    test = test_all.loc[test_all["view"].isin(["clean", "poison"])]
    random_test = test_all.loc[test_all["view"] == "random_control"]
    columns, scaler, classifier = (
        bundle["feature_columns"],
        bundle["scaler"],
        bundle["classifier"],
    )
    threshold = bundle["threshold"]
    test_scores = predict_scores(test, columns, scaler, classifier)
    random_scores = predict_scores(random_test, columns, scaler, classifier)
    metrics = {
        key: value
        for key, value in bundle.items()
        if key not in {"scaler", "classifier"}
    }
    metrics.update(
        {
            "test": binary_metrics(
                test["detector_label"].to_numpy(dtype=np.int64), test_scores, threshold
            ),
            "random_control_test": {
                "samples": int(len(random_test)),
                "mean_poison_probability": float(random_scores.mean()),
                "positive_rate_at_frozen_threshold": float(
                    np.mean(random_scores >= threshold)
                ),
            },
            "test_original_images": int(test["original_index"].nunique()),
            "test_used_for_training": False,
            "test_used_for_threshold_selection": False,
            "test_used_for_feature_comparison": False,
            "evaluation_features_sha256": metadata["features_sha256"],
        }
    )
    output_dir = ensure_output_path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_predictions(test, test_scores, threshold, output_dir / "test_predictions.csv")
    save_predictions(
        random_test,
        random_scores,
        threshold,
        output_dir / "random_control_test_predictions.csv",
    )
    save_json(metrics, output_dir / "metrics.json")
    return metrics


def main(args):
    features_path = Path(args.features)
    output_dir = ensure_output_path(args.output_dir)
    feature_set = getattr(args, "feature_set", "all")
    compare_stages = bool(
        getattr(args, "compare_stages", False)
        or getattr(args, "compare_features", False)
    )
    fit_only = bool(getattr(args, "fit_only", False))
    target_clean_fpr = float(args.target_clean_fpr)
    # Check calibration arguments before any model fit or output.
    select_clean_threshold([0.5], target_clean_fpr)

    features, feature_metadata = read_feature_table(features_path)
    validate_feature_table(features)
    if feature_metadata.get("origin_role") != "paired_benchmark":
        raise ValueError(
            "Supervised fitting requires paired_benchmark feature provenance."
        )
    row_provenance = feature_provenance(feature_metadata)
    for key in ("feature_protocol", "head_mode", "head_seed", "label_mode"):
        row_provenance[key] = features[key].iloc[0]
    assert_compatible_provenance(feature_provenance(feature_metadata), row_provenance)
    if getattr(args, "evaluate_bundle", None):
        if compare_stages or fit_only:
            raise ValueError(
                "--evaluate-bundle cannot be combined with fitting/comparison modes."
            )
        bundle = load_frozen_bundle(args.evaluate_bundle)
        metrics = evaluate_frozen_sample(features, bundle, feature_metadata, output_dir)
        print(json.dumps(metrics["test"], indent=2))
        return
    columns = feature_sets(features)[feature_set]
    if not columns:
        raise RuntimeError("No columns found for feature set {}.".format(feature_set))
    if not set(feature_sets(features)["all"]).issubset(
        feature_metadata["feature_columns"]
    ):
        raise ValueError(
            "An input feature is absent from the extraction metadata schema."
        )
    supervised = features.loc[features["view"].isin(["clean", "poison"])]
    splits = {
        name: supervised.loc[supervised["split"] == name].copy()
        for name in ("train", "validation", "test")
    }
    train, validation, test = (splits[name] for name in ("train", "validation", "test"))
    provenance = {
        "feature_protocol": FEATURE_PROTOCOL,
        "head_mode": HEAD_MODE,
        "head_seed": int(features["head_seed"].iloc[0]),
        "attack_seed": int(features["attack_seed"].iloc[0]),
        "label_mode": str(features["label_mode"].iloc[0]),
        "feature_metadata": feature_metadata,
        "provenance": feature_provenance(feature_metadata),
        "classifier_seed": int(args.classifier_seed),
        "features_path": str(features_path.resolve()),
        "features_sha256": sha256_file(features_path),
        "target_clean_fpr": target_clean_fpr,
        "threshold_rule": THRESHOLD_RULE,
        "fit_original_indices": sorted(
            int(i) for i in train["original_index"].unique()
        ),
        "calibration_original_indices": sorted(
            int(i) for i in validation["original_index"].unique()
        ),
        "supervision": "Clean/poison sample labels used for fitting; validation-clean labels used for threshold calibration.",
    }
    if compare_stages:
        comparison = compare_feature_sets(
            train,
            validation,
            args.classifier_seed,
            target_clean_fpr,
        )
        comparison_metadata = dict(provenance)
        comparison_metadata.update(
            {
                "evaluation_split": "validation",
                "selection": "Diagnostic ranking only; choose --feature-set explicitly for the final run.",
                "fpr_interpretation": "Every candidate uses the same empirical clean FPR cap; ties may be conservative.",
                "train_original_images": int(train["original_index"].nunique()),
                "validation_original_images": int(
                    validation["original_index"].nunique()
                ),
                "test_evaluated": False,
                "feature_sets": comparison[["feature_set", "feature_columns"]].to_dict(
                    orient="records"
                ),
            }
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        comparison.to_csv(output_dir / "feature_comparison_validation.csv", index=False)
        save_json(comparison_metadata, output_dir / "feature_comparison_metadata.json")
        print(comparison.drop(columns=["feature_columns"]).to_string(index=False))
        print("\nSaved validation-only comparison in:", output_dir)
        return

    scaler, classifier = fit_detector(train, columns, args.classifier_seed)
    validation_scores = predict_scores(validation, columns, scaler, classifier)
    threshold, validation_metrics = calibrate_validation(
        validation,
        validation_scores,
        target_clean_fpr,
    )
    provenance.update(
        {
            "feature_set": feature_set,
            "feature_columns": columns,
            "threshold": threshold,
        }
    )

    bundle = dict(provenance, scaler=scaler, classifier=classifier)
    bundle["validation"] = validation_metrics
    bundle["score_interpretation"] = (
        "Sample-level logistic score trained on clean/poison views, not a dataset posterior."
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    bundle_path = output_dir / "detector_bundle.joblib"
    save_frozen_bundle(bundle, output_dir, filename="detector_bundle.joblib")
    pd.DataFrame(
        {"feature": columns, "standardized_coefficient": classifier.coef_[0]}
    ).to_csv(output_dir / "coefficients.csv", index=False)
    if fit_only:
        metrics = dict(provenance, validation=validation_metrics, test_evaluated=False)
        save_json(metrics, output_dir / "metrics.json")
        print("Saved frozen sample detector without test evaluation:", bundle_path)
        return

    metrics = evaluate_frozen_sample(features, bundle, feature_metadata, output_dir)
    print(json.dumps(metrics["test"], indent=2))
    print("\nSaved detector outputs in:", output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a paired supervised detector with validation-only calibration and stage comparisons."
    )
    parser.add_argument("--features", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--classifier-seed", type=int, default=20260723)
    parser.add_argument("--target-clean-fpr", type=float, default=0.05)
    parser.add_argument(
        "--feature-set",
        choices=sorted(FEATURE_SETS),
        default="all",
        help="Choose a fixed feature group before inspecting test results.",
    )
    parser.add_argument(
        "--compare-stages",
        "--compare-features",
        action="store_true",
        help="Compare feature groups and individual stages/layers/tensors on validation, then exit.",
    )
    parser.add_argument(
        "--fit-only",
        action="store_true",
        help="Freeze the chosen sample detector without evaluating test; use before dataset fitting.",
    )
    parser.add_argument(
        "--evaluate-bundle",
        help="Evaluate a frozen sample bundle on test views without fitting.",
    )
    main(parser.parse_args())
