"""Label-free, reference-based dataset distribution-shift detection.

The detector fits only historical inversion descriptors computed using the same
frozen incoming-task head and predicted targets as incoming descriptors. An
independent reference-fit split determines preprocessing and a random Fourier
approximation of the RBF kernel. The remaining reference bank participates in a
two-sample permutation test. Neither fitting nor prediction reads attack labels,
class labels, or clean/poison views.

A significant result means distribution shift, not attack attribution or a
poisoning probability. Historical synthetic and new real inputs need not be
exchangeable even without an attack; benchmark clean false alerts explicitly.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from detector import STAGE_GRAD_FEATURE_COLUMNS
from detector.common import save_json
from detector.io_utils import (
    assert_compatible_provenance,
    ensure_output_path,
    feature_provenance,
    load_frozen_bundle,
    read_feature_table,
    save_frozen_bundle,
)

METHOD = "reference_rff_mmd_permutation_v1"
EXPECTED_FEATURE_PROTOCOL = "pretraining_full_v2"
FEATURE_COLUMNS = (
    ["grad_norm_l2"]
    + STAGE_GRAD_FEATURE_COLUMNS
    + [
        "entropy",
        "confidence",
        "margin",
        "activation_norm_l2",
    ]
)
LOG_COLUMNS = ["grad_norm_l2"] + STAGE_GRAD_FEATURE_COLUMNS + ["activation_norm_l2"]
LIMITATIONS = [
    "A permutation p-value is not a poisoning probability.",
    "Exchangeability between historical synthetic reference data and clean incoming data is unverified.",
    "Novel tasks, classes, benign noise, and synthetic-to-real domain shift can trigger alerts.",
    "Fixed seeded subsampling can miss a small contaminated subset of a larger incoming dataset.",
]


def _positive_integer(value, name, minimum=1):
    if isinstance(value, (bool, np.bool_)) or int(value) != value or value < minimum:
        raise ValueError("{} must be an integer >= {}.".format(name, minimum))
    return int(value)


def _validate_settings(settings):
    for name in ("permutations", "n_components"):
        _positive_integer(settings[name], name)
    for name in ("max_reference", "max_incoming"):
        _positive_integer(settings[name], name, minimum=2)
    _positive_integer(settings["seed"], "seed", minimum=0)
    if not 0 < settings["alpha"] < 1:
        raise ValueError("alpha must lie strictly between zero and one.")
    if not 0 < settings["reference_fit_fraction"] < 1:
        raise ValueError(
            "reference_fit_fraction must lie strictly between zero and one."
        )
    if not np.isfinite(settings["clip"]) or settings["clip"] <= 0:
        raise ValueError("clip must be finite and positive.")
    if not np.isfinite(settings["scale_floor"]) or settings["scale_floor"] <= 0:
        raise ValueError("scale_floor must be finite and positive.")


def _validate_metadata(metadata):
    if metadata.get("feature_protocol") != EXPECTED_FEATURE_PROTOCOL:
        raise ValueError("Re-extract features with protocol pretraining_full_v2.")
    if metadata.get("label_mode") != "predicted":
        raise ValueError(
            "Label-free detection requires predicted-target feature extraction."
        )
    if metadata.get("head_mode") != "defender_fixed":
        raise ValueError(
            "Reference and incoming descriptors must use the fixed defender head."
        )
    _positive_integer(metadata.get("head_seed"), "head_seed", minimum=0)
    if metadata.get("model_unchanged") is False:
        raise ValueError(
            "Features cannot come from a model modified during extraction."
        )
    return feature_provenance(metadata)


def _descriptor_values(table):
    """Select only the fixed descriptor columns; ignore labels and views."""
    if not isinstance(table, pd.DataFrame) or table.columns.duplicated().any():
        raise ValueError("Expected a feature DataFrame with unique column names.")
    missing = set(FEATURE_COLUMNS) - set(table.columns)
    if missing:
        raise KeyError("Missing label-free descriptors: {}.".format(sorted(missing)))
    values = table[FEATURE_COLUMNS].to_numpy(dtype=np.float64, copy=True)
    if len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("Each dataset needs at least two finite descriptor rows.")
    if np.any(values < 0):
        raise ValueError(
            "Gradient norms, uncertainty, and activation descriptors must be nonnegative."
        )
    for column in ("confidence", "margin"):
        if np.any(values[:, FEATURE_COLUMNS.index(column)] > 1):
            raise ValueError("{} must lie in [0, 1].".format(column))
    return values


def _log_transform(values):
    values = values.copy()
    indices = [FEATURE_COLUMNS.index(column) for column in LOG_COLUMNS]
    values[:, indices] = np.log1p(values[:, indices])
    return values


def _standardize(values, bundle):
    transformed = _log_transform(values)
    active = bundle["active_mask"]
    return np.clip(
        (transformed[:, active] - bundle["median"][active]) / bundle["scale"][active],
        -bundle["settings"]["clip"],
        bundle["settings"]["clip"],
    )


def _median_bandwidth(values):
    norms = np.einsum("ij,ij->i", values, values)
    squared = np.maximum(norms[:, None] + norms[None, :] - 2 * values.dot(values.T), 0)
    distances = np.sqrt(squared[np.triu_indices(len(values), k=1)])
    distances = distances[distances > 1e-12]
    if not len(distances):
        raise ValueError(
            "Reference-fit descriptors have no nonzero pairwise distances."
        )
    return float(np.median(distances))


def _random_features(values, bundle):
    return np.sqrt(2.0 / bundle["settings"]["n_components"]) * np.cos(
        values.dot(bundle["projection"]) + bundle["phase"]
    )


def permutation_mmd(reference, incoming, *, permutations, seed):
    """Biased squared RFF-mean distance with a conservative Monte Carlo p-value.

    Preprocessing and the feature map must have been fitted independently of
    both samples. All permutations use the same map and group sizes. The +1
    numerator/denominator correction prevents reporting a zero p-value.
    """
    reference = np.asarray(reference, dtype=np.float64)
    incoming = np.asarray(incoming, dtype=np.float64)
    permutations = _positive_integer(permutations, "permutations")
    seed = _positive_integer(seed, "seed", minimum=0)
    if (
        reference.ndim != 2
        or incoming.ndim != 2
        or min(len(reference), len(incoming)) < 2
        or reference.shape[1] != incoming.shape[1]
        or not reference.shape[1]
        or not np.isfinite(reference).all()
        or not np.isfinite(incoming).all()
    ):
        raise ValueError(
            "MMD requires two finite 2D samples with matching dimensions and n >= 2."
        )
    difference = reference.mean(axis=0) - incoming.mean(axis=0)
    observed = float(difference.dot(difference))
    pooled = np.concatenate([reference, incoming], axis=0)
    n_reference, n_incoming = len(reference), len(incoming)
    rng = np.random.default_rng(seed)
    # Bound temporary allocation when a caller intentionally requests many permutations.
    extreme = 0
    for start in range(0, permutations, 128):
        count = min(128, permutations - start)
        contrasts = np.full((count, len(pooled)), -1.0 / n_incoming)
        for row in contrasts:
            row[rng.permutation(len(pooled))[:n_reference]] = 1.0 / n_reference
        differences = contrasts.dot(pooled)
        null_statistics = np.einsum("ij,ij->i", differences, differences)
        # Count numerical ties conservatively, including reversed equal-size groups.
        tolerance = 1e-12 * max(1.0, observed)
        extreme += int(np.count_nonzero(null_statistics >= observed - tolerance))
    return {
        "mmd_squared_rff": observed,
        "p_value": float((1 + extreme) / (permutations + 1)),
        "permutations": permutations,
        "extreme_permutations": extreme,
        "minimum_p_value": float(1.0 / (permutations + 1)),
    }


def fit_reference(
    reference_features,
    metadata,
    *,
    alpha=0.05,
    permutations=199,
    reference_fit_fraction=0.5,
    max_reference=512,
    max_incoming=256,
    seed=20260720,
    n_components=128,
):
    """Freeze a label-free reference detector; never consult incoming datasets."""
    provenance = _validate_metadata(metadata)
    if metadata.get("origin_role") != "historical_inversion":
        raise ValueError(
            "Fit only reference descriptors whose origin_role is historical_inversion."
        )
    values = _descriptor_values(reference_features)
    if "reference_id" not in reference_features:
        raise KeyError(
            "Reference features require globally unique reference_id values."
        )
    ids = reference_features["reference_id"]
    if ids.isna().any() or ids.astype(str).duplicated().any():
        raise ValueError("reference_id values must be nonempty and unique.")
    reference_ids = ids.astype(str).tolist()
    if any(not value.strip() for value in reference_ids):
        raise ValueError("reference_id values must be nonempty and unique.")
    settings = {
        "alpha": float(alpha),
        "permutations": permutations,
        "reference_fit_fraction": float(reference_fit_fraction),
        "max_reference": max_reference,
        "max_incoming": max_incoming,
        "seed": seed,
        "n_components": n_components,
        "clip": 12.0,
        "scale_floor": 1e-8,
    }
    _validate_settings(settings)
    for name in (
        "permutations",
        "max_reference",
        "max_incoming",
        "seed",
        "n_components",
    ):
        settings[name] = int(settings[name])
    n_fit = int(np.floor(len(values) * reference_fit_fraction))
    if n_fit < 2 or len(values) - n_fit < 2:
        raise ValueError(
            "Need at least two reference-fit and two independent reference-bank rows."
        )
    rng = np.random.default_rng(settings["seed"])
    order = rng.permutation(len(values))
    fit_indices = order[:n_fit]
    bank_indices = order[n_fit : n_fit + settings["max_reference"]]
    unused_indices = order[n_fit + len(bank_indices) :]
    fitting_values = _log_transform(values[fit_indices])
    median = np.median(fitting_values, axis=0)
    scale = 1.4826 * np.median(np.abs(fitting_values - median), axis=0)
    active = scale > settings["scale_floor"]
    if not active.any():
        raise ValueError(
            "All reference-fit descriptors have zero or near-zero robust scale."
        )
    bundle = {
        "schema_version": 1,
        "method": METHOD,
        "provenance": provenance,
        "settings": settings,
        "feature_columns": list(FEATURE_COLUMNS),
        "log1p_columns": list(LOG_COLUMNS),
        "active_mask": active,
        "active_feature_columns": [
            name for name, keep in zip(FEATURE_COLUMNS, active) if keep
        ],
        "removed_feature_columns": [
            name for name, keep in zip(FEATURE_COLUMNS, active) if not keep
        ],
        "median": median,
        "scale": np.maximum(scale, settings["scale_floor"]),
        "reference_origin_role": "historical_inversion",
        "reference_features_sha256": metadata.get("features_sha256"),
        "reference_ids": reference_ids,
        "fit_reference_ids": [reference_ids[index] for index in fit_indices],
        "bank_reference_ids": [reference_ids[index] for index in bank_indices],
        "unused_reference_ids": [reference_ids[index] for index in unused_indices],
        "reference_split_rule": "Seeded random row split; disjoint fit and test-bank identities.",
        "limitations": list(LIMITATIONS),
    }
    if "original_index" in reference_features:
        bundle["reference_original_indices"] = reference_features[
            "original_index"
        ].tolist()
    standardized_fit = _standardize(values[fit_indices], bundle)
    bandwidth = _median_bandwidth(standardized_fit)
    bundle["bandwidth"] = bandwidth
    bundle["projection"] = rng.normal(
        scale=1.0 / bandwidth, size=(int(active.sum()), settings["n_components"])
    )
    bundle["phase"] = rng.uniform(0, 2 * np.pi, size=settings["n_components"])
    bundle["reference_bank_rff"] = _random_features(
        _standardize(values[bank_indices], bundle), bundle
    )
    _validate_bundle(bundle)
    return bundle


def _validate_bundle(bundle):
    if bundle.get("schema_version") != 1 or bundle.get("method") != METHOD:
        raise ValueError("Unsupported unsupervised detector bundle.")
    if bundle.get("feature_columns") != FEATURE_COLUMNS:
        raise ValueError(
            "Unsupervised feature schema does not match this implementation."
        )
    _validate_settings(bundle["settings"])
    if bundle.get("reference_origin_role") != "historical_inversion":
        raise ValueError(
            "The frozen reference must originate from historical inversions."
        )
    all_ids = bundle["reference_ids"]
    parts = [
        bundle[name]
        for name in ("fit_reference_ids", "bank_reference_ids", "unused_reference_ids")
    ]
    partition_ids = [identity for part in parts for identity in part]
    if (
        len(set(all_ids)) != len(all_ids)
        or len(set(partition_ids)) != len(partition_ids)
        or set(all_ids) != set(partition_ids)
        or min(len(parts[0]), len(parts[1])) < 2
    ):
        raise ValueError(
            "Reference-fit, bank, and unused identities must partition the reference pool."
        )
    dimension = len(FEATURE_COLUMNS)
    active = np.asarray(bundle["active_mask"])
    if active.dtype != np.bool_ or active.shape != (dimension,) or not active.any():
        raise ValueError("Invalid active-feature mask.")
    shapes = {
        "median": (dimension,),
        "scale": (dimension,),
        "projection": (int(active.sum()), bundle["settings"]["n_components"]),
        "phase": (bundle["settings"]["n_components"],),
        "reference_bank_rff": (len(parts[1]), bundle["settings"]["n_components"]),
    }
    for name, expected_shape in shapes.items():
        values = np.asarray(bundle[name])
        if values.shape != expected_shape or not np.isfinite(values).all():
            raise ValueError("Invalid frozen {} array.".format(name))
    if (
        np.any(bundle["scale"] <= 0)
        or not np.isfinite(bundle["bandwidth"])
        or bundle["bandwidth"] <= 0
    ):
        raise ValueError("Frozen scales and bandwidth must be finite and positive.")


def predict_dataset(bundle, features, metadata):
    """Test one incoming dataset with frozen preprocessing and permutation count."""
    _validate_bundle(bundle)
    provenance = _validate_metadata(metadata)
    assert_compatible_provenance(bundle["provenance"], provenance)
    values = _descriptor_values(features)
    if "reference_id" in features:
        overlap = set(features["reference_id"].dropna().astype(str)) & set(
            bundle["reference_ids"]
        )
        if overlap:
            raise ValueError(
                "Incoming rows overlap the frozen historical reference identities."
            )
    settings = bundle["settings"]
    incoming_count = len(values)
    if incoming_count > settings["max_incoming"]:
        rng = np.random.default_rng(settings["seed"] + 1)
        indices = rng.choice(
            incoming_count, size=settings["max_incoming"], replace=False
        )
        values = values[indices]
    mapped = _random_features(_standardize(values, bundle), bundle)
    result = permutation_mmd(
        bundle["reference_bank_rff"],
        mapped,
        permutations=settings["permutations"],
        seed=settings["seed"] + 2,
    )
    result.update(
        {
            "method": METHOD,
            "decision_type": "dataset_distribution_shift",
            "alpha": settings["alpha"],
            "shift_detected": bool(result["p_value"] <= settings["alpha"]),
            "reference_bank_samples": len(bundle["reference_bank_rff"]),
            "incoming_samples_available": incoming_count,
            "incoming_samples_tested": len(values),
            "subsampling_rule": "Without replacement with frozen seed; ignores labels and views.",
            "not_poisoning_probability": True,
            "reference_domain_assumption": "Unverified exchangeability of clean incoming and historical synthetic reference descriptors.",
            "limitations": list(LIMITATIONS),
        }
    )
    return result


def save_bundle(bundle, output_dir):
    """Persist a frozen artifact and its integrity sidecar without overwriting."""
    _validate_bundle(bundle)
    return save_frozen_bundle(bundle, output_dir, filename="unsupervised_bundle.joblib")


def load_bundle(path):
    bundle = load_frozen_bundle(path)
    _validate_bundle(bundle)
    return bundle


def evaluate_benchmark(
    bundle, features, metadata, *, tasks_per_rate=100, task_size=150, seed=20260723
):
    """Use held-out benchmark labels only to assemble bags and report error rates.

    No evaluation result changes alpha, preprocessing, projections, reference
    membership, or permutations. Reused images make simulated bags dependent.
    """
    _validate_bundle(bundle)
    assert_compatible_provenance(bundle["provenance"], _validate_metadata(metadata))
    if metadata.get("origin_role") != "paired_benchmark":
        raise ValueError(
            "Evaluation requires a held-out paired_benchmark feature table."
        )
    tasks_per_rate = _positive_integer(tasks_per_rate, "tasks_per_rate")
    task_size = _positive_integer(task_size, "task_size", minimum=2)
    seed = _positive_integer(seed, "seed", minimum=0)
    required = {"original_index", "split", "view"}
    if not required.issubset(features.columns):
        raise KeyError("Benchmark evaluation requires original_index, split, and view.")
    if features[list(required)].isna().any().any():
        raise ValueError("Benchmark identities, splits, and views cannot be missing.")
    if (features.groupby("original_index")["split"].nunique() != 1).any():
        raise ValueError("Benchmark original identities overlap across data splits.")
    test = features.loc[features["split"] == "test"].copy()
    _descriptor_values(test)
    expected_views = {"clean", "poison", "random_control"}
    if set(test["view"]) != expected_views:
        raise ValueError(
            "Held-out test data must contain clean, poison, and random_control views."
        )
    if test.duplicated(["original_index", "view"]).any():
        raise ValueError("Each test original needs exactly one row per view.")
    groups = test.groupby("original_index")["view"].nunique()
    if (groups != 3).any() or len(groups) < task_size:
        raise ValueError(
            "Need complete held-out triples and at least task_size distinct originals."
        )
    originals = groups.index.to_numpy()
    lookup = test.set_index(["original_index", "view"])
    rng = np.random.default_rng(seed)
    scenarios = [("clean", 0.0)] + [
        (view, rate)
        for view in ("poison", "random_control")
        for rate in (0.1, 0.25, 0.5, 1.0)
    ]
    summaries, task_results = [], []
    for view, rate in scenarios:
        n_modified = int(np.floor(task_size * rate + 0.5))
        alerts = 0
        for task_id in range(tasks_per_rate):
            selected = rng.choice(originals, size=task_size, replace=False)
            keys = [
                (identity, view if index < n_modified else "clean")
                for index, identity in enumerate(selected)
            ]
            # Descriptor-only handoff makes the label-free prediction boundary explicit.
            bag = lookup.loc[keys, FEATURE_COLUMNS].reset_index(drop=True)
            prediction = predict_dataset(bundle, bag, metadata)
            alerts += int(prediction["shift_detected"])
            task_results.append(
                {
                    "scenario": view,
                    "requested_modified_fraction": rate,
                    "task_id": task_id,
                    "mmd_squared_rff": prediction["mmd_squared_rff"],
                    "p_value": prediction["p_value"],
                    "shift_detected": prediction["shift_detected"],
                }
            )
        metric_name = {
            "clean": "clean_task_false_rejection_rate",
            "poison": "poison_task_detection_rate",
            "random_control": "random_control_task_alert_rate",
        }[view]
        summaries.append(
            {
                "scenario": view,
                "requested_modified_fraction": rate,
                "actual_modified_fraction": n_modified / task_size,
                "tasks": tasks_per_rate,
                "alerts": alerts,
                "alert_rate": alerts / tasks_per_rate,
                "metric_name": metric_name,
            }
        )
    return {
        "method": METHOD,
        "evaluation_split": "test",
        "alpha": bundle["settings"]["alpha"],
        "task_size": task_size,
        "evaluation_seed": seed,
        "incoming_samples_tested_per_bag": min(
            task_size, bundle["settings"]["max_incoming"]
        ),
        "test_originals": len(originals),
        "summary": summaries,
        "tasks": task_results,
        "clean_task_false_rejection_rate": summaries[0]["alert_rate"],
        "threshold_tuned_on_evaluation": False,
        "conclusion": "Descriptive benchmark only; no automatic poisoning-detector pass. Interpret attack detection alongside clean false alerts and random-control alerts.",
        "sampling_note": "Originals are sampled without replacement within each bag and reused across bags; simulated tasks are not independent experiments.",
        "limitations": list(LIMITATIONS),
    }


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    fit = commands.add_parser(
        "fit", help="Freeze a historical-reference detector without attack labels."
    )
    fit.add_argument("--reference-features", required=True)
    fit.add_argument("--output-dir", required=True)
    fit.add_argument("--alpha", type=float, default=0.05)
    fit.add_argument("--permutations", type=int, default=199)
    fit.add_argument("--reference-fit-fraction", type=float, default=0.5)
    fit.add_argument("--max-reference", type=int, default=512)
    fit.add_argument("--max-incoming", type=int, default=256)
    fit.add_argument("--seed", type=int, default=20260720)
    for command in ("predict", "evaluate"):
        subparser = commands.add_parser(command)
        subparser.add_argument("--bundle", required=True)
        subparser.add_argument("--features", required=True)
        subparser.add_argument("--output", required=True)
        if command == "evaluate":
            subparser.add_argument("--tasks-per-rate", type=int, default=100)
            subparser.add_argument("--task-size", type=int, default=150)
            subparser.add_argument("--seed", type=int, default=20260723)
    return parser


def main(args):
    if args.command == "fit":
        reference, metadata = read_feature_table(args.reference_features)
        bundle = fit_reference(
            reference,
            metadata,
            alpha=args.alpha,
            permutations=args.permutations,
            reference_fit_fraction=args.reference_fit_fraction,
            max_reference=args.max_reference,
            max_incoming=args.max_incoming,
            seed=args.seed,
        )
        path = save_bundle(bundle, args.output_dir)
        print("Saved frozen label-free reference detector:", path)
        print(
            "Reference fit rows: {}; test-bank rows: {}".format(
                len(bundle["fit_reference_ids"]), len(bundle["bank_reference_ids"])
            )
        )
        return
    output_path = Path(args.output)
    ensure_output_path(output_path)
    if output_path.exists():
        raise FileExistsError(
            "Refusing to overwrite evaluation output: {}".format(output_path)
        )
    bundle = load_bundle(args.bundle)
    features, metadata = read_feature_table(args.features)
    if args.command == "predict":
        result = predict_dataset(bundle, features, metadata)
    else:
        result = evaluate_benchmark(
            bundle,
            features,
            metadata,
            tasks_per_rate=args.tasks_per_rate,
            task_size=args.task_size,
            seed=args.seed,
        )
    save_json(result, output_path)
    print("Saved:", output_path)
    print("This is a distribution-shift test, not a poisoning probability.")


if __name__ == "__main__":
    main(build_parser().parse_args())
