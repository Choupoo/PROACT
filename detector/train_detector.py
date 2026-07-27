import argparse
import json
from pathlib import Path
import joblib
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
    FEATURE_COLUMNS,
    FEATURE_PROTOCOL,
    HEAD_MODE,
)
from detector.common import *


def validate_feature_table(features):
    required = {"original_index", "source_index", "class_id", "split", "view", "detector_label", "feature_protocol", "head_mode", "head_seed"}.union(FEATURE_COLUMNS)

    missing = required - set(features.columns)
    if missing:
        raise KeyError("Feature table is missing columns: {}".format(sorted(missing)))

    if set(features["feature_protocol"].unique()) != {FEATURE_PROTOCOL}:
        raise RuntimeError("Unexpected feature protocol.")

    if set(features["head_mode"].unique()) != {HEAD_MODE}:
        raise RuntimeError("Features were not produced with defender_fixed head mode.")

    allowed_splits = {"train", "validation", "test"}
    if set(features["split"].unique()) != allowed_splits:
        raise RuntimeError("Features must contain only train, validation and test.")

    allowed_views = {"clean", "poison", "random_control"}
    if set(features["view"].unique()) != allowed_views:
        raise RuntimeError("Feature table must contain clean, poison and random_control.")

    for original_index, group in features.groupby("original_index"):
        if len(group) != 3:
            raise RuntimeError("original_index {} does not have three views.".format(original_index))

        if set(group["view"]) != allowed_views:
            raise RuntimeError("original_index {} has an incomplete view set.".format(original_index))

        if group["split"].nunique() != 1:
            raise RuntimeError("Views of original_index {} are in different splits.".format(original_index))

        if group["class_id"].nunique() != 1:
            raise RuntimeError("Views of original_index {} have different labels.".format(original_index))

        labels = dict(zip(group["view"], group["detector_label"]))
        if labels != {"clean": 0, "poison": 1, "random_control": -1}:
            raise RuntimeError("Incorrect detector labels for original_index {}.".format(original_index))

    split_sets = {
        split_name: set(features.loc[features["split"] == split_name, "original_index"].astype(int))
        for split_name in allowed_splits
    }

    split_names = sorted(allowed_splits)
    for i, first in enumerate(split_names):
        for second in split_names[i + 1:]:
            overlap = split_sets[first] & split_sets[second]
            if overlap:
                raise RuntimeError("{} and {} share original images.".format(first, second))

    if features[FEATURE_COLUMNS].isna().any().any():
        raise RuntimeError("Feature table contains NaN values.")

    if not np.isfinite(features[FEATURE_COLUMNS].to_numpy(dtype=float)).all():
        raise RuntimeError("Feature table contains non-finite values.")


def binary_metrics(y_true, probabilities, threshold):
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


def main(args):
    features_path = Path(args.features)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    features = pd.read_csv(features_path)
    validate_feature_table(features)

    supervised = features.loc[features["view"].isin(["clean", "poison"])].copy()

    train = supervised.loc[supervised["split"] == "train"].copy()
    validation = supervised.loc[supervised["split"] == "validation"].copy()
    test = supervised.loc[supervised["split"] == "test"].copy()

    for name, table in [("train", train), ("validation", validation), ("test", test)]:
        if table.empty:
            raise RuntimeError("{} split is empty.".format(name))

        if set(table["detector_label"].unique()) != {0, 1}:
            raise RuntimeError("{} split must contain clean and poison rows.".format(name))

    scaler = StandardScaler()
    x_train = scaler.fit_transform(train[FEATURE_COLUMNS].to_numpy(dtype=np.float64))
    y_train = train["detector_label"].to_numpy(dtype=np.int64)

    classifier = LogisticRegression(C=1.0, penalty="l2", solver="lbfgs", max_iter=2000, random_state=int(args.classifier_seed))
    classifier.fit(x_train, y_train)

    x_validation = scaler.transform(validation[FEATURE_COLUMNS].to_numpy(dtype=np.float64))
    validation_probabilities = (classifier.predict_proba(x_validation)[:, 1])

    validation_clean_scores = (validation_probabilities[validation["detector_label"].to_numpy()== 0])

    threshold = quantile_higher(validation_clean_scores, 1.0 - float(args.target_clean_fpr))

    validation_metrics = binary_metrics(validation["detector_label"].to_numpy(dtype=np.int64), validation_probabilities, threshold)

    x_test = scaler.transform(test[FEATURE_COLUMNS].to_numpy(dtype=np.float64))
    test_probabilities = (classifier.predict_proba(x_test)[:, 1])

    test_metrics = binary_metrics(test["detector_label"].to_numpy(dtype=np.int64), test_probabilities, threshold)

    random_test = features.loc[(features["split"] == "test") & (features["view"] == "random_control")].copy()

    random_probabilities = classifier.predict_proba(scaler.transform(random_test[FEATURE_COLUMNS].to_numpy(dtype=np.float64)))[:, 1]

    random_positive_rate = float(np.mean(random_probabilities >= threshold))

    metrics = {
        "feature_protocol": FEATURE_PROTOCOL,
        "feature_columns": FEATURE_COLUMNS,
        "classifier": (
            "StandardScaler fit on train only, followed by "
            "L2 logistic regression."
        ),
        "classifier_seed": int(args.classifier_seed),
        "threshold_rule": (
            "Higher empirical quantile of validation-clean scores "
            "at 1 - target_clean_fpr."
        ),
        "target_clean_fpr": float(
            args.target_clean_fpr
        ),
        "threshold": float(threshold),
        "train_original_images": int(
            train["original_index"].nunique()
        ),
        "validation_original_images": int(
            validation["original_index"].nunique()
        ),
        "test_original_images": int(
            test["original_index"].nunique()
        ),
        "validation": validation_metrics,
        "test": test_metrics,
        "random_control_test": {
            "samples": int(len(random_test)),
            "mean_poison_probability": float(
                random_probabilities.mean()
            ),
            "positive_rate_at_frozen_threshold": (
                random_positive_rate
            ),
        },
        "test_used_for_training": False,
        "test_used_for_threshold_selection": False,
    }

    bundle = {
        "scaler": scaler,
        "classifier": classifier,
        "threshold": float(threshold),
        "feature_columns": list(FEATURE_COLUMNS),
        "feature_protocol": FEATURE_PROTOCOL,
        "classifier_seed": int(
            args.classifier_seed
        ),
        "target_clean_fpr": float(
            args.target_clean_fpr
        ),
    }

    bundle_path = output_dir / "detector_bundle.joblib"
    metrics_path = output_dir / "metrics.json"
    predictions_path = output_dir / "test_predictions.csv"
    coefficients_path = output_dir / "coefficients.csv"

    joblib.dump(bundle, bundle_path)

    predictions = test[
        [
            "original_index",
            "source_index",
            "class_id",
            "split",
            "view",
            "detector_label",
        ]
    ].copy()
    predictions["poison_probability"] = test_probabilities
    predictions["prediction"] = (
        test_probabilities >= threshold
    ).astype(np.int64)
    predictions.to_csv(
        predictions_path,
        index=False,
    )

    random_predictions = random_test[
        [
            "original_index",
            "source_index",
            "class_id",
            "split",
            "view",
        ]
    ].copy()
    random_predictions["poison_probability"] = (
        random_probabilities
    )
    random_predictions["prediction"] = (
        random_probabilities >= threshold
    ).astype(np.int64)
    random_predictions.to_csv(
        output_dir / "random_control_test_predictions.csv",
        index=False,
    )

    coefficients = pd.DataFrame(
        {
            "feature": FEATURE_COLUMNS,
            "standardized_coefficient": (
                classifier.coef_[0]
            ),
        }
    )
    coefficients.to_csv(
        coefficients_path,
        index=False,
    )

    metrics["features_path"] = str(
        features_path.resolve()
    )
    metrics["features_sha256"] = sha256_file(
        features_path
    )
    save_json(metrics, metrics_path)

    print(json.dumps(metrics, indent=2))
    print("\nSaved:", bundle_path)
    print("Saved:", metrics_path)
    print("Saved:", predictions_path)
    print("Saved:", coefficients_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=("Train the clean supervised detector baseline without pair overlap, post-hoc deletion or test-set tuning."))
    parser.add_argument("--features", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--classifier-seed", type=int, default=20260723)
    parser.add_argument("--target-clean-fpr", type=float, default=0.05,)
    main(parser.parse_args())