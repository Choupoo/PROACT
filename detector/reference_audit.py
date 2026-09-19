"""Diagnose historical-reference mismatch on validation-clean originals only.

This is an offline labeled audit, not a new label-free training procedure. Never
use test rows, retune alpha, or relabel abstentions as correct clean decisions.
"""

import argparse
import numpy as np

from detector.common import save_json
from detector.io_utils import (
    ensure_output_path,
    read_feature_table,
    assert_compatible_provenance,
)
from detector.unsupervised import (
    FEATURE_COLUMNS,
    _descriptor_values,
    _log_transform,
    _validate_bundle,
    _validate_metadata,
    load_bundle,
    predict_dataset,
)


def audit_reference(bundle, features, metadata, task_size=150, seed=20260728):
    _validate_bundle(bundle)
    assert_compatible_provenance(bundle["provenance"], _validate_metadata(metadata))
    if metadata.get("origin_role") != "paired_benchmark":
        raise ValueError("Offline reference audit requires paired_benchmark metadata.")
    if (
        isinstance(task_size, bool)
        or not isinstance(task_size, (int, np.integer))
        or task_size < 2
    ):
        raise ValueError("task_size must be an integer >= 2.")
    required = ["original_index", "split", "view"]
    if not set(required).issubset(features) or features[required].isna().any().any():
        raise ValueError("Audit requires nonmissing original IDs, splits and views.")
    if (features.groupby("original_index")["split"].nunique() != 1).any():
        raise ValueError("Original identities overlap data splits.")
    clean = features.loc[
        (features["split"] == "validation") & (features["view"] == "clean")
    ].sort_values("original_index")
    if clean["original_index"].duplicated().any() or len(clean) < task_size:
        raise ValueError("Need at least task_size distinct validation-clean originals.")
    values = _descriptor_values(clean)
    transformed = _log_transform(values)
    active = bundle["active_mask"]
    standardized = (transformed[:, active] - bundle["median"][active]) / bundle[
        "scale"
    ][active]
    feature_diagnostics = [
        {
            "feature": column,
            "validation_clean_median": float(
                np.median(values[:, FEATURE_COLUMNS.index(column)])
            ),
            "absolute_median_shift_reference_scale_units": float(
                abs(np.median(standardized[:, index]))
            ),
            "fraction_outside_reference_clip": float(
                np.mean(np.abs(standardized[:, index]) > bundle["settings"]["clip"])
            ),
        }
        for index, column in enumerate(bundle["active_feature_columns"])
    ]
    order = np.random.default_rng(seed).permutation(len(clean))
    results = []
    for start in range(0, len(clean) - task_size + 1, task_size):
        bag = clean.iloc[order[start : start + task_size]]
        prediction = predict_dataset(bundle, bag[FEATURE_COLUMNS], metadata)
        results.append(
            {
                "original_indices": [int(x) for x in bag["original_index"]],
                "p_value": prediction["p_value"],
                "mmd_squared_rff": prediction["mmd_squared_rff"],
                "shift_detected": prediction["shift_detected"],
            }
        )
    rate = float(np.mean([x["shift_detected"] for x in results]))
    return {
        "audit_split": "validation_clean",
        "test_used": False,
        "threshold_tuned": False,
        "trusted_clean_labels_used_for_offline_audit_only": True,
        "task_size": int(task_size),
        "disjoint_batches": len(results),
        "unused_clean_originals": len(clean) - task_size * len(results),
        "validation_clean_shift_alert_rate": rate,
        "reference_domain_status": "validation_failed"
        if rate > bundle["settings"]["alpha"]
        else "not_failed_in_this_small_audit",
        "deployment_action": "abstain",
        "note": "Batches have disjoint originals but share the reference bank. This small audit can expose mismatch, not certify deployment FPR. Raw shift detection remains available for diagnosis; abstentions are not accepted-clean decisions.",
        "feature_diagnostics": sorted(
            feature_diagnostics,
            key=lambda x: x["absolute_median_shift_reference_scale_units"],
            reverse=True,
        ),
        "batches": results,
    }


def main(args):
    output = ensure_output_path(args.output)
    if output.exists():
        raise FileExistsError("Choose a new audit output: {}".format(output))
    features, metadata = read_feature_table(args.features)
    result = audit_reference(
        load_bundle(args.bundle), features, metadata, args.task_size
    )
    save_json(result, output)
    print("Saved offline reference audit:", output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--features", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--task-size", type=int, default=150)
    main(parser.parse_args())
