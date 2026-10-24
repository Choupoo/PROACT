"""Label-free inference with a frozen, historically calibrated transfer detector.

This consumes a genuine incoming batch, not clean/poison benchmark triplets.
It neither fits a model nor recalibrates a threshold on the incoming images.
"""

import numpy as np
import pandas as pd
import torch

from detector import FEATURE_PROTOCOL, HEAD_MODE
from detector.advisor_detector import predict_dataset, score_images
from detector.common import (assert_model_unchanged, load_pickle, matching_inversion_files,
                             save_json, sha256_file, snapshot_state_dict)
from detector.extract_features import backbone_parameters, build_past_references, extract_one, load_input_npz
from detector.image_data import dataset_spec, task_order
from detector.io_utils import ensure_output_path
from detector.transfer_core import TRANSFER_PROTOCOL, contract, load_defender, validate_checkpoint
from detector.transfer_detector import INFERENCE_GROUPS


def validate_deployment(bundle, checkpoint):
    if bundle.get("kind") != TRANSFER_PROTOCOL:
        raise ValueError("Expected a frozen transfer detector.")
    if bundle.get("class_label_policy") != "predicted" or bundle.get("feature_set") not in INFERENCE_GROUPS:
        raise ValueError("Deployment requires a predicted-class inference feature group.")
    if "true_class_probability" in bundle["feature_columns"]:
        raise ValueError("True-class probability is unavailable at inference.")
    source = contract(bundle["source_metadata"])
    if (source["label_mode"] != "predicted" or source["feature_protocol"] != FEATURE_PROTOCOL
            or source["head_mode"] != HEAD_MODE):
        raise ValueError("Unsupported frozen extraction protocol.")
    for key in ("head_seed", "label_mode", "head_mode", "feature_protocol"):
        if bundle["provenance"][key] != source[key]:
            raise ValueError("Frozen source/provenance mismatch: " + key)
    task = validate_checkpoint(checkpoint)
    dataset = source.get("dataset", "split_cifar100")
    if checkpoint["dataset"] != dataset or task_order(dataset) != source["task_order"]:
        raise ValueError("Incoming checkpoint and fitted dataset differ.")
    calibrated = [r["task"] for r in bundle.get("calibration_history", [])] + [source["task_index"]]
    if task <= max(calibrated):
        raise ValueError("Incoming task must be later than every fitting/calibration task.")
    method = checkpoint.get("cont_method_args", {}).get("method")
    if source.get("cl_method") is not None and method != source["cl_method"]:
        raise ValueError("Incoming CL method differs from the frozen detector.")
    if source.get("checkpoint_seed") is not None and checkpoint["seed"] != source["checkpoint_seed"]:
        raise ValueError("Incoming checkpoint seed differs from the registered experiment.")
    if bundle["task_size"] != bundle["count_calibration"]["task_size"]:
        raise ValueError("Frozen dataset size and count calibration disagree.")
    return task, dataset


def predict_incoming(args, bundle):
    output = ensure_output_path(args.output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Choose an empty prediction directory: " + str(output))
    before_bundle = sha256_file(args.bundle)
    checkpoint = load_pickle(args.checkpoint)
    task, dataset = validate_deployment(bundle, checkpoint)
    spec = dataset_spec(dataset)
    images, _ = load_input_npz(args.input_npz, "predicted", spec["image_size"], spec["classes_per_task"])
    if len(images) != bundle["task_size"]:
        raise ValueError("Incoming dataset size differs from the frozen count calibration.")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable.")
    files = matching_inversion_files(args.inversion_dir, expected_count=task,
                                     image_size=spec["image_size"], classes_per_task=spec["classes_per_task"])
    model = load_defender(checkpoint, device, bundle["source_metadata"]["head_seed"])
    before_model = snapshot_state_dict(model)
    past, directions, records = build_past_references(model, files, device, batch_size=32)
    named, parameters = backbone_parameters(model)
    past, directions = past.to(device), directions.to(device)
    rows = []
    for index, image in enumerate(images):
        row = extract_one(model, named, parameters, past, image, None, device,
                          directions, label_mode="predicted")
        row.pop("true_class_probability", None)
        row["original_index"] = index
        rows.append(row)
    descriptors = pd.DataFrame(rows)
    decision = predict_dataset(descriptors, bundle)
    scores = score_images(descriptors, bundle)
    assert_model_unchanged(model, before_model)
    if sha256_file(args.bundle) != before_bundle:
        raise RuntimeError("Frozen bundle changed during prediction.")
    output.mkdir(parents=True, exist_ok=True)
    feature_path = output / "incoming_features.csv"
    descriptors.to_csv(feature_path, index=False)
    pd.DataFrame(dict(original_index=np.arange(len(images)), score=scores,
                      flagged=scores >= bundle["threshold"])).to_csv(output / "sample_predictions.csv", index=False)
    references = [{"task_id": r["task_id"], "sha256": sha256_file(r["inversion_file"])} for r in records]
    metadata = dict(feature_protocol=FEATURE_PROTOCOL, head_mode=HEAD_MODE,
                    head_seed=bundle["source_metadata"]["head_seed"], label_mode="predicted",
                    checkpoint_sha256=sha256_file(args.checkpoint), inversion_sha256=references_digest(references),
                    dataset=dataset, task_index=task, origin_role="incoming_unlabelled",
                    model_unchanged=True, row_count=len(descriptors),
                    feature_columns=[c for c in descriptors if c != "original_index"],
                    features_sha256=sha256_file(feature_path), input_sha256=sha256_file(args.input_npz),
                    reference_tasks=references, incoming_class_labels_used=False,
                    historical_inversion_labels_used=True)
    save_json(metadata, feature_path.with_suffix(".metadata.json"))
    decision.update(method=TRANSFER_PROTOCOL, dataset=dataset, task_index=task,
                    count_calibration=bundle["count_calibration"], bundle_sha256=before_bundle,
                    checkpoint_sha256=metadata["checkpoint_sha256"], input_sha256=metadata["input_sha256"],
                    features_sha256=metadata["features_sha256"], incoming_class_labels_used=False,
                    target_used_for_fitting_or_calibration=False,
                    limitations=["Risk alert, not a poisoning probability or a target false-positive guarantee.",
                                 "The caller must supply the matching clean historical checkpoint and inversion references.",
                                 "Incoming batch membership and original-image uniqueness are caller assumptions."])
    save_json(decision, output / "prediction.json")
    return decision


def references_digest(references):
    import hashlib
    import json
    return hashlib.sha256(json.dumps(references, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
