"""Historical-task union-null MMD on per-image gradient shapes.

No incoming clean subset, target labels, or incoming-fitted preprocessing.
Kernel fitting and permutation reference banks use disjoint historical rows.
This tests distribution shift under an explicit matching-task assumption.
"""

import numpy as np

from detector.io_utils import assert_compatible_provenance
from detector.local_reference import COLUMNS, shapes
from detector.rank_reference import _task_ids, integer
from detector.unsupervised import _median_bandwidth, _validate_metadata, permutation_mmd

METHOD = "historical_task_shape_mmd_v1"
LIMITATIONS = [
    "Clean incoming shapes must be exchangeable with at least one historical task's held-out bank; synthetic-to-real equivalence is unverified.",
    "Shared inversion optimization can violate independence/exchangeability.",
    "Per-image normalization deliberately loses gradient magnitude; magnitude-only attacks can be invisible.",
    "Max-p across tasks is conservative; sparse poisoning and a matching historical task can reduce power.",
    "A finite random-feature map can miss distribution differences.",
    "A p-value measures distributional incompatibility, not a poisoning probability or safe deployment decision.",
]


def _map(values, profile):
    return np.sqrt(2 / len(profile["phase"])) * np.cos(
        values.dot(profile["projection"]) + profile["phase"]
    )


def fit_reference(
    table,
    metadata,
    *,
    permutations=199,
    n_components=64,
    alpha=0.05,
    seed=20260926,
    max_incoming=256,
):
    provenance = _validate_metadata(metadata)
    if metadata.get("model_unchanged") is not True:
        raise ValueError(
            "Historical extraction must explicitly confirm a frozen model."
        )
    if metadata.get("origin_role") != "historical_inversion":
        raise ValueError(
            "Fit only historical inversions; incoming clean references are forbidden."
        )
    settings = {
        "permutations": integer(permutations, "permutations", 19),
        "n_components": integer(n_components, "n_components", 1),
        "seed": integer(seed, "seed", 0),
        "max_incoming": integer(max_incoming, "max_incoming", 2),
        "alpha": float(alpha),
    }
    if not np.isfinite(alpha) or not 0 < alpha < 1 or 1 / (permutations + 1) > alpha:
        raise ValueError("Invalid alpha or insufficient permutation resolution.")
    tasks = _task_ids(table)
    values, valid = shapes(table)
    if not len(values) or not valid.all():
        raise ValueError("Historical gradient shapes must be nonempty and nonzero.")
    profiles = []
    for task in sorted(set(tasks)):
        indices = np.flatnonzero(tasks == task)
        indices = indices[np.argsort(table.iloc[indices].reference_id.to_numpy())]
        if len(indices) < 16:
            raise ValueError("Need at least 16 historical images per task.")
        rng = np.random.default_rng(seed + int(task))
        indices = rng.permutation(indices)
        fit, bank = np.array_split(indices, 2)
        bandwidth = _median_bandwidth(values[fit])
        profile = {
            "task_id": int(task),
            "fit_ids": table.iloc[fit].reference_id.astype(str).tolist(),
            "bank_ids": table.iloc[bank].reference_id.astype(str).tolist(),
            "bandwidth": bandwidth,
            "projection": rng.normal(scale=1 / bandwidth, size=(5, n_components)),
            "phase": rng.uniform(0, 2 * np.pi, n_components),
        }
        profile["bank_rff"] = _map(values[bank], profile)
        profiles.append(profile)
    bundle = {
        "method": METHOD,
        "schema_version": 1,
        "settings": settings,
        "feature_columns": list(COLUMNS),
        "provenance": provenance,
        "reference_origin_role": "historical_inversion",
        "reference_features_sha256": metadata.get("features_sha256"),
        "reference_ids": table.reference_id.astype(str).tolist(),
        "profiles": profiles,
        "fit_uses_incoming_data": False,
        "fit_uses_attack_or_class_labels": False,
        "aggregation": "max per-task p-value; reject only when all tasks reject",
        "limitations": list(LIMITATIONS),
    }
    validate_bundle(bundle)
    return bundle


def validate_bundle(bundle):
    if bundle.get("method") != METHOD or bundle.get("schema_version") != 1:
        raise ValueError("Unsupported shape-MMD bundle.")
    if (
        bundle.get("feature_columns") != list(COLUMNS)
        or bundle.get("reference_origin_role") != "historical_inversion"
    ):
        raise ValueError("Historical shape feature contract mismatch.")
    s = bundle["settings"]
    for key, minimum in (
        ("permutations", 19),
        ("n_components", 1),
        ("seed", 0),
        ("max_incoming", 2),
    ):
        integer(s[key], key, minimum)
    if (
        not np.isfinite(s["alpha"])
        or not 0 < s["alpha"] < 1
        or 1 / (s["permutations"] + 1) > s["alpha"]
    ):
        raise ValueError("Invalid frozen alpha/permutation resolution.")
    if not bundle["profiles"]:
        raise ValueError("No historical tasks.")
    ids, task_ids = [], []
    for p in bundle["profiles"]:
        task_ids.append(integer(p["task_id"], "task_id", 0))
        if min(len(p["fit_ids"]), len(p["bank_ids"])) < 8:
            raise ValueError("Historical task split is too small.")
        ids += p["fit_ids"] + p["bank_ids"]
        if any(
            not identity.startswith("task{}:sample".format(p["task_id"]))
            for identity in p["fit_ids"] + p["bank_ids"]
        ):
            raise ValueError("Historical task identity mismatch.")
        for name, expected in (
            ("projection", (5, s["n_components"])),
            ("phase", (s["n_components"],)),
            ("bank_rff", (len(p["bank_ids"]), s["n_components"])),
        ):
            value = np.asarray(p[name])
            if value.shape != expected or not np.isfinite(value).all():
                raise ValueError("Invalid frozen " + name)
        if not np.isfinite(p["bandwidth"]) or p["bandwidth"] <= 0:
            raise ValueError("Invalid kernel bandwidth.")
    if (
        len(task_ids) != len(set(task_ids))
        or len(ids) != len(set(ids))
        or sorted(ids) != sorted(bundle["reference_ids"])
    ):
        raise ValueError(
            "Historical fit and bank identities must partition all references."
        )


def predict_dataset(bundle, table, metadata):
    validate_bundle(bundle)
    assert_compatible_provenance(bundle["provenance"], _validate_metadata(metadata))
    if metadata.get("model_unchanged") is not True:
        raise ValueError("Incoming extraction must explicitly confirm a frozen model.")
    if "reference_id" in table and set(table.reference_id.dropna().astype(str)) & set(
        bundle["reference_ids"]
    ):
        raise ValueError("Incoming samples overlap fitted historical references.")
    if not 2 <= len(table) <= bundle["settings"]["max_incoming"]:
        raise ValueError("Incoming size outside frozen bounds; no hidden subsampling.")
    values, valid = shapes(table)
    result = {
        "method": METHOD,
        "status": "ok",
        "incoming_samples_tested": len(table),
        "invalid_samples": int((~valid).sum()),
        "p_value": None,
        "shift_detected": None,
        "poisoning_decision": "undetermined",
        "deployment_action": "abstain",
        "not_poisoning_probability": True,
        "alpha": bundle["settings"]["alpha"],
        "per_task": [],
    }
    if not valid.all():
        result["status"] = "undefined_gradient_shape"
        return result
    for profile in bundle["profiles"]:
        test = permutation_mmd(
            profile["bank_rff"],
            _map(values, profile),
            permutations=bundle["settings"]["permutations"],
            seed=bundle["settings"]["seed"] + 10000 + profile["task_id"],
        )
        result["per_task"].append(dict(test, task_id=profile["task_id"]))
    result["p_value"] = max(p["p_value"] for p in result["per_task"])
    result["shift_detected"] = bool(result["p_value"] <= result["alpha"])
    return result


def historical_audit(table, metadata, **settings):
    """Leave an entire historical task out, including its kernel-fitting rows."""
    tasks = _task_ids(table)
    if len(set(tasks)) < 2:
        return {"status": "insufficient_tasks", "tasks": []}
    results = []
    for task in sorted(set(tasks)):
        bundle = fit_reference(table.loc[tasks != task], metadata, **settings)
        held = table.loc[tasks == task].sort_values("reference_id")
        cap = bundle["settings"]["max_incoming"]
        if len(held) > cap:
            indices = np.random.default_rng(
                bundle["settings"]["seed"] + int(task)
            ).choice(len(held), cap, replace=False)
            held = held.iloc[indices]
        prediction = predict_dataset(bundle, held, metadata)
        results.append(
            dict(
                prediction,
                held_out_task=int(task),
                available_samples=int((tasks == task).sum()),
            )
        )
    return {
        "status": "ok",
        "tasks": results,
        "note": "Historical diagnostic only; no hyperparameter selection and no incoming real-domain guarantee.",
    }
