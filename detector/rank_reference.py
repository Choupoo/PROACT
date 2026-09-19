"""Experimental, strictly label-free historical rank-dependence detector.

Each historical task is a separate reference. Compare vectors of Kendall tau-a
using a centered jackknife Gaussian multiplier bootstrap; reject the union
null only if ALL historical tasks differ (p_union = max(p_task)). Marginal
monotone transforms are removed, not estimated using trusted incoming data.

This tests pairwise rank dependence, not full copula equality or poisoning.
Bootstrap inference is approximate, not an exact permutation test. Independent
observations, nondegenerate projections, and a matching historical dependence
pattern are assumptions, not established facts about inversion samples.
"""

import re

import numpy as np

from detector.io_utils import assert_compatible_provenance
from detector.unsupervised import (
    FEATURE_COLUMNS,
    _descriptor_values,
    _validate_metadata,
)

METHOD = "historical_task_kendall_jackknife_v1"
MIN_SAMPLES = 16
PAIR_INDICES = list(zip(*np.triu_indices(len(FEATURE_COLUMNS), 1)))
PAIR_NAMES = [FEATURE_COLUMNS[i] + " :: " + FEATURE_COLUMNS[j] for i, j in PAIR_INDICES]
LIMITATIONS = [
    "Approximate multiplier-bootstrap p-values, not exact finite-sample guarantees or poisoning probabilities.",
    "Clean incoming data must share pairwise rank dependence with at least one historical task; new benign dependence can still alert.",
    "Strictly increasing per-feature transformations are deliberately invisible, including attacks that only change marginal magnitudes.",
    "Higher-order dependence changes with unchanged pairwise Kendall taus can be missed.",
    "Heavy ties, correlated inversion samples, small tasks and contaminated historical references can invalidate inference.",
    "More reference tasks make the union-null rule conservative and can reduce attack sensitivity.",
]


def integer(value, name, minimum):
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        or value < minimum
    ):
        raise ValueError("{} must be an integer >= {}.".format(name, minimum))
    return int(value)


def rank_profile(values):
    """Kendall tau-a and exact centered leave-one-out jackknife pseudovalues.

    h_ij = sign(x_i - x_j) sign(y_i - y_j), tau = mean_{i != j} h_ij.
    P_i - mean(P) = 2(n-1)/(n-2) * (mean_{j != i} h_ij - tau).
    Ties contribute zero. No attack, class, split, or clean labels are accepted.
    """
    values = np.asarray(values, dtype=np.float64)
    if (
        values.ndim != 2
        or len(values) < 4
        or values.shape[1] < 2
        or not np.isfinite(values).all()
    ):
        raise ValueError(
            "Rank profiles require at least four finite rows and two columns."
        )
    n, d = values.shape
    signs = np.sign(values[:, None, :] - values[None, :, :])
    pairs = list(zip(*np.triu_indices(d, 1)))
    conditional = np.column_stack(
        [(signs[:, :, i] * signs[:, :, j]).sum(axis=1) / (n - 1) for i, j in pairs]
    )
    tau = conditional.mean(axis=0)
    pseudo = 2 * (n - 1) / (n - 2) * (conditional - tau)
    tied = 1 - np.abs(signs).sum(axis=(0, 1)) / (n * (n - 1))
    return {"n": n, "tau": tau, "pseudo": pseudo, "tie_fractions": tied}


def multiplier_draws(profile, draws, seed):
    """Centered first-order errors; do not resample already-frozen row ranks."""
    draws = integer(draws, "bootstrap_draws", 19)
    seed = integer(seed, "seed", 0)
    n = profile["n"]
    weights = np.random.default_rng(seed).normal(size=(draws, n))
    return weights.dot(profile["pseudo"]) / np.sqrt(n * (n - 1))


def compare_profiles(reference, incoming, incoming_draws, active):
    difference = incoming["tau"][active] - reference["tau"][active]
    observed = float(np.max(np.abs(difference)))
    null = incoming_draws[:, active] - reference["bootstrap_errors"][:, active]
    null_statistics = np.max(np.abs(null), axis=1)
    extreme = int(np.count_nonzero(null_statistics >= observed - 1e-12))
    most_changed = np.flatnonzero(active)[np.argmax(np.abs(difference))]
    return {
        "reference_task_id": reference["task_id"],
        "max_absolute_tau_difference": observed,
        "p_value_approx": (extreme + 1) / (len(null_statistics) + 1),
        "extreme_bootstrap_draws": extreme,
        "most_changed_pair": PAIR_NAMES[most_changed],
        "reference_tau_at_most_changed_pair": float(reference["tau"][most_changed]),
        "incoming_tau_at_most_changed_pair": float(incoming["tau"][most_changed]),
    }


def _task_ids(table):
    if "reference_id" not in table or table.reference_id.isna().any():
        raise ValueError("Historical reference_id is required.")
    ids = table.reference_id.astype(str)
    if ids.duplicated().any():
        raise ValueError("Historical reference_id must be unique.")
    matches = [re.fullmatch(r"task(\d+):sample(\d+)", value) for value in ids]
    if any(match is None for match in matches):
        raise ValueError("Expected extraction identities taskN:sampleM.")
    tasks = np.array([int(match.group(1)) for match in matches])
    if "reference_task_id" in table:
        declared = table.reference_task_id.to_numpy(dtype=np.float64)
        if not np.array_equal(declared, tasks):
            raise ValueError("reference_task_id disagrees with reference_id.")
    return tasks


def fit_reference(
    table,
    metadata,
    *,
    alpha=0.05,
    bootstrap_draws=499,
    seed=20260920,
    max_reference_per_task=256,
    max_incoming=256,
):
    provenance = _validate_metadata(metadata)
    if metadata.get("origin_role") != "historical_inversion":
        raise ValueError(
            "Fit only historical inversions; incoming clean references are forbidden."
        )
    if not np.isfinite(alpha) or not 0 < alpha < 1:
        raise ValueError("alpha must lie in (0, 1).")
    bootstrap_draws = integer(bootstrap_draws, "bootstrap_draws", 19)
    seed = integer(seed, "seed", 0)
    if 1 / (bootstrap_draws + 1) > alpha:
        raise ValueError("Bootstrap resolution is too coarse for alpha.")
    max_reference_per_task = integer(
        max_reference_per_task, "max_reference_per_task", MIN_SAMPLES
    )
    max_incoming = integer(max_incoming, "max_incoming", MIN_SAMPLES)
    tasks = _task_ids(table)
    values = _descriptor_values(table)
    profiles = []
    for task_id in sorted(set(tasks)):
        indices = np.flatnonzero(tasks == task_id)
        # Canonical identity order before seeded subsampling.
        indices = indices[np.argsort(table.iloc[indices].reference_id.to_numpy())]
        if len(indices) < MIN_SAMPLES:
            raise ValueError(
                "Each historical task needs at least {} samples.".format(MIN_SAMPLES)
            )
        if len(indices) > max_reference_per_task:
            indices = np.random.default_rng(seed + int(task_id)).choice(
                indices, max_reference_per_task, replace=False
            )
        profile = rank_profile(values[indices])
        profile.update(
            task_id=int(task_id),
            reference_ids=table.iloc[indices].reference_id.astype(str).tolist(),
        )
        profile["bootstrap_errors"] = multiplier_draws(
            profile, bootstrap_draws, seed + 1000 + int(task_id)
        )
        profiles.append(profile)
    # Select nondegenerate pairs using historical references only, never a new task.
    active = np.logical_and.reduce(
        [np.var(p["pseudo"], axis=0) > 1e-12 for p in profiles]
    )
    if not active.any():
        raise ValueError(
            "No nondegenerate pair shared by historical tasks; cannot fit rank test."
        )
    result = {
        "method": METHOD,
        "schema_version": 1,
        "feature_columns": list(FEATURE_COLUMNS),
        "pair_names": PAIR_NAMES,
        "provenance": provenance,
        "reference_origin_role": "historical_inversion",
        "reference_features_sha256": metadata.get("features_sha256"),
        "reference_ids": table.reference_id.astype(str).tolist(),
        "profiles": profiles,
        "active_pairs": active,
        "settings": {
            "alpha": float(alpha),
            "bootstrap_draws": bootstrap_draws,
            "seed": seed,
            "max_reference_per_task": max_reference_per_task,
            "max_incoming": max_incoming,
        },
        "null_hypothesis": "Incoming pairwise rank dependence matches at least one historical task on retained pairs.",
        "aggregation": "max of per-task p-values; reject only when every reference task rejects",
        "fit_uses_incoming_data": False,
        "fit_uses_attack_or_class_labels": False,
        "limitations": list(LIMITATIONS),
    }
    validate_bundle(result)
    return result


def validate_bundle(bundle):
    if bundle.get("method") != METHOD or bundle.get("schema_version") != 1:
        raise ValueError("Unsupported rank-reference bundle.")
    if (
        bundle.get("feature_columns") != FEATURE_COLUMNS
        or bundle.get("pair_names") != PAIR_NAMES
    ):
        raise ValueError("Rank-reference feature schema mismatch.")
    if bundle.get("reference_origin_role") != "historical_inversion":
        raise ValueError("Only historical inversion references are permitted.")
    settings = bundle["settings"]
    b = integer(settings["bootstrap_draws"], "bootstrap_draws", 19)
    integer(settings["seed"], "seed", 0)
    integer(settings["max_incoming"], "max_incoming", MIN_SAMPLES)
    integer(settings["max_reference_per_task"], "max_reference_per_task", MIN_SAMPLES)
    if (
        not np.isfinite(settings["alpha"])
        or not 0 < settings["alpha"] < 1
        or 1 / (b + 1) > settings["alpha"]
    ):
        raise ValueError("Invalid alpha or bootstrap resolution.")
    active = np.asarray(bundle["active_pairs"])
    d = len(PAIR_INDICES)
    if active.dtype != np.bool_ or active.shape != (d,) or not active.any():
        raise ValueError("Invalid active-pair mask.")
    profiles = bundle["profiles"]
    if not profiles or len({p["task_id"] for p in profiles}) != len(profiles):
        raise ValueError("Historical tasks must be nonempty and distinct.")
    ids = bundle["reference_ids"]
    used = [x for p in profiles for x in p["reference_ids"]]
    if (
        len(set(ids)) != len(ids)
        or len(set(used)) != len(used)
        or not set(used).issubset(ids)
    ):
        raise ValueError("Invalid reference identities.")
    for profile in profiles:
        n = integer(profile["n"], "reference samples", MIN_SAMPLES)
        if len(profile["reference_ids"]) != n:
            raise ValueError("Reference identity count mismatch.")
        for key, shape in (
            ("tau", (d,)),
            ("pseudo", (n, d)),
            ("bootstrap_errors", (b, d)),
            ("tie_fractions", (len(FEATURE_COLUMNS),)),
        ):
            array = np.asarray(profile[key])
            if array.shape != shape or not np.isfinite(array).all():
                raise ValueError("Invalid frozen {}.".format(key))
        if np.any(np.var(profile["pseudo"], axis=0)[active] <= 1e-12):
            raise ValueError("Retained reference pairs must be nondegenerate.")


def predict_dataset(bundle, table, metadata):
    validate_bundle(bundle)
    assert_compatible_provenance(bundle["provenance"], _validate_metadata(metadata))
    values = _descriptor_values(table)
    if len(values) < MIN_SAMPLES:
        raise ValueError("Need at least {} incoming rows.".format(MIN_SAMPLES))
    if "reference_id" in table and set(table.reference_id.dropna().astype(str)) & set(
        bundle["reference_ids"]
    ):
        raise ValueError("Incoming and historical reference identities overlap.")
    settings = bundle["settings"]
    total = len(values)
    # Row-order invariant selection, not selection by labels or detector scores.
    values = values[
        np.lexsort(tuple(values[:, i] for i in reversed(range(values.shape[1]))))
    ]
    if len(values) > settings["max_incoming"]:
        indices = np.random.default_rng(settings["seed"] + 2000).choice(
            len(values), settings["max_incoming"], replace=False
        )
        values = values[indices]
    profile = rank_profile(values)
    active = bundle["active_pairs"]
    used_features = sorted(
        {i for pair, keep in zip(PAIR_INDICES, active) if keep for i in pair}
    )
    result = {
        "method": METHOD,
        "decision_type": "historical_rank_dependence_shift",
        "alpha": settings["alpha"],
        "bootstrap_draws": settings["bootstrap_draws"],
        "incoming_samples_available": total,
        "incoming_samples_tested": len(values),
        "active_pair_count": int(active.sum()),
        "max_feature_tie_fraction": float(profile["tie_fractions"].max()),
        "poisoning_decision": "undetermined",
        "deployment_action": "research_only",
        "not_poisoning_probability": True,
        "limitations": list(LIMITATIONS),
    }
    if np.any(profile["tie_fractions"][used_features] >= 1 - 1e-12):
        result.update(
            status="unsupported_constant_features",
            shift_detected=None,
            p_value_approx=None,
            comparisons=[],
        )
        return result
    draws = multiplier_draws(
        profile, settings["bootstrap_draws"], settings["seed"] + 3000
    )
    comparisons = [
        compare_profiles(p, profile, draws, active) for p in bundle["profiles"]
    ]
    p_value = max(row["p_value_approx"] for row in comparisons)
    result.update(
        status="ok",
        shift_detected=bool(p_value <= settings["alpha"]),
        p_value_approx=p_value,
        comparisons=comparisons,
    )
    return result


def historical_audit(bundle):
    """Leave one historical TASK out; diagnostic only, never changes the bundle."""
    validate_bundle(bundle)
    results = []
    if len(bundle["profiles"]) < 2:
        return {"status": "unavailable_single_historical_task", "tasks": []}
    for held in bundle["profiles"]:
        comparisons = [
            compare_profiles(p, held, held["bootstrap_errors"], bundle["active_pairs"])
            for p in bundle["profiles"]
            if p["task_id"] != held["task_id"]
        ]
        p_value = max(row["p_value_approx"] for row in comparisons)
        results.append(
            {
                "held_out_task": held["task_id"],
                "p_value_approx": p_value,
                "shift_detected": bool(p_value <= bundle["settings"]["alpha"]),
                "comparisons": comparisons,
            }
        )
    return {
        "status": "diagnostic_only",
        "tasks": results,
        "leave_task_out_alert_rate": float(
            np.mean([row["shift_detected"] for row in results])
        ),
        "threshold_tuned": False,
        "note": "Shared reference tasks make these audits dependent; inversion-to-real transfer remains unverified.",
    }
