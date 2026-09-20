"""Historical-only local gradient-shape novelty, then batch count aggregation.

No target fitting, majority-clean assumption, or incoming labels. This is a
research baseline, not a guaranteed detector. Normalized stage norms discard
sample-wide gradient magnitude but preserve relative layer geometry. Historical
fit, threshold and count-calibration identities are disjoint.
"""

import numpy as np

from detector import STAGE_GRAD_FEATURE_COLUMNS
from detector.calibration import fit_count_calibration, count_decision
from detector.io_utils import assert_compatible_provenance
from detector.rank_reference import _task_ids, integer
from detector.unsupervised import _validate_metadata
from detector.train_detector import select_clean_threshold

METHOD = "historical_local_gradient_shape_knn_v1"
COLUMNS = list(STAGE_GRAD_FEATURE_COLUMNS)
LIMITATIONS = [
    "No target clean references or incoming fitting; historical inversion geometry may still differ from real clean tasks.",
    "Per-sample uniform gradient rescaling is deliberately invisible; changes outside the five stage norm ratios can be missed.",
    "Nearest-neighbor distances measure novelty, not malicious intent or a calibrated poisoning probability.",
    "Count calibration uses historical inversions only. The binomial working model is not target-domain FPR control; task heterogeneity, fixed task allocation and correlated inversions can violate it.",
    "Finite reference calibration may make the critical count too large for sparse poisoning; this is reported, not corrected using test labels.",
]


def shapes(table):
    if (
        table.empty
        or table.columns.duplicated().any()
        or not set(COLUMNS).issubset(table)
    ):
        raise ValueError("Need a nonempty table with unique stage-gradient columns.")
    raw = table[COLUMNS].to_numpy(dtype=np.float64)
    if not np.isfinite(raw).all() or np.any(raw < 0):
        raise ValueError("Stage norms must be finite and nonnegative.")
    maximum = raw.max(axis=1)
    valid = maximum > 0
    # Scale before taking the norm to avoid overflow/underflow.
    normalized = raw / np.where(valid, maximum, 1)[:, None]
    length = np.linalg.norm(normalized, axis=1)
    normalized /= np.where(valid, length, 1)[:, None]
    return normalized, valid


def score_shapes(values, bank, neighbors):
    """Mean k-neighbor Euclidean distance / sqrt(2), bounded by 1 on this domain."""
    rows = []
    for start in range(0, len(values), 256):
        difference = values[start : start + 256, None, :] - bank[None, :, :]
        distances = np.sqrt(np.sum(difference * difference, axis=2))
        nearest = np.partition(distances, neighbors - 1, axis=1)[:, :neighbors]
        rows.append(nearest.mean(axis=1) / np.sqrt(2))
    return np.clip(np.concatenate(rows), 0, 1)


def fit_reference(
    table,
    metadata,
    *,
    task_size=150,
    neighbors=5,
    sample_tail=0.01,
    alpha=0.05,
    seed=20260922,
):
    provenance = _validate_metadata(metadata)
    if metadata.get("origin_role") != "historical_inversion":
        raise ValueError("Only historical inversions may fit this detector.")
    task_size = integer(task_size, "task_size", 1)
    neighbors = integer(neighbors, "neighbors", 1)
    seed = integer(seed, "seed", 0)
    if not np.isfinite(sample_tail) or not 0 < sample_tail < 1:
        raise ValueError("sample_tail must lie in (0,1).")
    tasks = _task_ids(table)
    values, valid = shapes(table)
    if not valid.all():
        raise ValueError(
            "Zero-gradient historical rows cannot calibrate gradient shape."
        )
    partitions = {key: [] for key in ("fit", "threshold", "calibration")}
    for task in sorted(set(tasks)):
        indices = np.flatnonzero(tasks == task)
        indices = indices[np.argsort(table.iloc[indices].reference_id.to_numpy())]
        if len(indices) < 16:
            raise ValueError("Need at least 16 reference samples per historical task.")
        indices = np.random.default_rng(seed + int(task)).permutation(indices)
        a, b = len(indices) // 2, 3 * len(indices) // 4
        for key, part in zip(partitions, (indices[:a], indices[a:b], indices[b:])):
            partitions[key].extend(part.tolist())
    bank = values[partitions["fit"]].copy()
    if neighbors > len(bank):
        raise ValueError("neighbors exceeds historical fit bank size.")
    threshold_scores = score_shapes(values[partitions["threshold"]], bank, neighbors)
    threshold = select_clean_threshold(threshold_scores, sample_tail)
    calibration_scores = score_shapes(
        values[partitions["calibration"]], bank, neighbors
    )
    calibration = fit_count_calibration(
        calibration_scores >= threshold, task_size, alpha
    )
    bundle = {
        "method": METHOD,
        "schema_version": 1,
        "feature_columns": COLUMNS,
        "provenance": provenance,
        "reference_origin_role": "historical_inversion",
        "reference_features_sha256": metadata.get("features_sha256"),
        "reference_ids": table.reference_id.astype(str).tolist(),
        "partitions": {
            key: table.iloc[indices].reference_id.astype(str).tolist()
            for key, indices in partitions.items()
        },
        "fit_bank": bank,
        "sample_threshold": threshold,
        "count_calibration": calibration,
        "settings": {
            "task_size": task_size,
            "neighbors": neighbors,
            "sample_tail": float(sample_tail),
            "alpha": float(alpha),
            "seed": seed,
        },
        "fit_uses_incoming_data": False,
        "requires_majority_clean_incoming": False,
        "calibration_scope": "historical_only_not_target_domain_fpr_control",
        "limitations": list(LIMITATIONS),
    }
    validate_bundle(bundle)
    return bundle


def validate_bundle(bundle):
    if (
        bundle.get("method") != METHOD
        or bundle.get("schema_version") != 1
        or bundle.get("feature_columns") != COLUMNS
    ):
        raise ValueError("Unsupported local-reference schema.")
    if bundle.get("reference_origin_role") != "historical_inversion":
        raise ValueError("Historical inversion references required.")
    settings = bundle["settings"]
    integer(settings["task_size"], "task_size", 1)
    k = integer(settings["neighbors"], "neighbors", 1)
    integer(settings["seed"], "seed", 0)
    for name in ("alpha", "sample_tail"):
        if not np.isfinite(settings[name]) or not 0 < settings[name] < 1:
            raise ValueError("Invalid frozen {}.".format(name))
    parts = bundle["partitions"]
    if set(parts) != {"fit", "threshold", "calibration"} or any(
        not x for x in parts.values()
    ):
        raise ValueError("Need three nonempty historical partitions.")
    ids = [x for values in parts.values() for x in values]
    if (
        len(ids) != len(set(ids))
        or set(ids) != set(bundle["reference_ids"])
        or len(ids) != len(bundle["reference_ids"])
    ):
        raise ValueError(
            "Historical fit, threshold and calibration identities must partition references."
        )
    bank = np.asarray(bundle["fit_bank"])
    if (
        bank.shape != (len(parts["fit"]), 5)
        or len(bank) < k
        or not np.isfinite(bank).all()
        or np.any(bank < 0)
    ):
        raise ValueError("Invalid local reference bank.")
    if not np.allclose(np.linalg.norm(bank, axis=1), 1):
        raise ValueError("Reference shapes must have unit norm.")
    if not np.isfinite(bundle["sample_threshold"]) or not 0 <= bundle[
        "sample_threshold"
    ] <= np.nextafter(1.0, np.inf):
        raise ValueError("Invalid sample threshold.")
    cal = bundle["count_calibration"]
    if cal["task_size"] != settings["task_size"] or cal[
        "calibration_original_images"
    ] != len(parts["calibration"]):
        raise ValueError(
            "Count calibration does not match reference identities or task size."
        )
    alarms_count = integer(cal["calibration_sample_alarms"], "calibration alarms", 0)
    if alarms_count > len(parts["calibration"]):
        raise ValueError("More calibration alarms than historical samples.")
    alarms = [1] * alarms_count + [0] * (len(parts["calibration"]) - alarms_count)
    expected = fit_count_calibration(alarms, settings["task_size"], settings["alpha"])
    for key in (
        "critical_suspicious_count",
        "target_clean_frr",
        "sample_fpr_upper_bound",
    ):
        if not np.isclose(cal[key], expected[key], rtol=1e-12, atol=1e-15):
            raise ValueError(
                "Frozen count calibration disagrees with its counts/settings."
            )


def predict_dataset(bundle, table, metadata):
    validate_bundle(bundle)
    assert_compatible_provenance(bundle["provenance"], _validate_metadata(metadata))
    if len(table) != bundle["settings"]["task_size"]:
        raise ValueError(
            "Local detector requires the exact frozen task size; no subsampling."
        )
    if "reference_id" in table and set(table.reference_id.dropna().astype(str)) & set(
        bundle["reference_ids"]
    ):
        raise ValueError("Incoming identities overlap historical references.")
    values, valid = shapes(table)
    result = {
        "method": METHOD,
        "poisoning_decision": "undetermined",
        "not_poisoning_probability": True,
        "calibration_scope": bundle["calibration_scope"],
        "limitations": list(LIMITATIONS),
    }
    if not valid.all():
        result.update(
            status="unsupported_zero_gradient",
            shift_detected=None,
            suspicious_count=None,
            invalid_samples=int((~valid).sum()),
        )
        return result
    scores = score_shapes(values, bundle["fit_bank"], bundle["settings"]["neighbors"])
    alarms = scores >= bundle["sample_threshold"]
    count = int(alarms.sum())
    decision, tail = count_decision([count], bundle["count_calibration"])
    result.update(
        status="ok",
        shift_detected=bool(decision[0]),
        suspicious_count=count,
        critical_count=bundle["count_calibration"]["critical_suspicious_count"],
        sample_threshold=bundle["sample_threshold"],
        historical_binomial_tail=float(tail[0]),
        sample_scores=scores.tolist(),
    )
    return result


def historical_audit(table, metadata, **settings):
    """Leave-one-task-out sample novelty audit. No incoming data or tuning."""
    tasks = _task_ids(table)
    if len(set(tasks)) < 2:
        return {"status": "unavailable_single_task", "tasks": []}
    rows = []
    for held in sorted(set(tasks)):
        bundle = fit_reference(table.loc[tasks != held], metadata, **settings)
        values, valid = shapes(table.loc[tasks == held])
        if not valid.all():
            raise ValueError("Zero-gradient historical audit input.")
        scores = score_shapes(
            values, bundle["fit_bank"], bundle["settings"]["neighbors"]
        )
        rows.append(
            {
                "held_out_task": int(held),
                "samples": len(scores),
                "sample_novelty_fraction": float(
                    np.mean(scores >= bundle["sample_threshold"])
                ),
            }
        )
    return {
        "status": "historical_only_diagnostic",
        "tasks": rows,
        "threshold_tuned": False,
        "note": "Sample fractions, NOT task FPR or independent experiments. High values expose historical transfer mismatch.",
    }
