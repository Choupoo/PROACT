"""Extract stage-aware paired features with unchanged legacy feature semantics."""

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from detector import FEATURE_COLUMNS, FEATURE_PROTOCOL, HEAD_MODE
from detector.common import (
    assert_model_unchanged,
    load_pickle,
    matching_inversion_files,
    save_json,
    set_seed,
    sha256_file,
    snapshot_state_dict,
)
from detector.create_manifest import create_manifest, validate_manifest_table
from detector.extract_features import (
    backbone_parameters,
    build_past_references,
    extract_one,
    matched_random_noise,
)
from detector.io_utils import DETECTOR_ROOT, ensure_output_path, read_feature_table
from detector.train_detector import feature_sets, validate_feature_table
from detector.image_data import dataset_spec, task_order
from detector.transfer_core import (
    TRANSFER_PROTOCOL,
    fixed_dataset_specs,
    load_defender,
    validate_artifact,
    validate_checkpoint,
)


def reusable_controls(path, metadata, delta):
    """Reuse only identical frozen-model measurements; never reuse poison views."""
    if not path:
        return {}, None
    from detector.transfer_detector import validate_table
    table, saved = read_feature_table(path)
    # Round-trip decimal values, rather than accepting pandas' faster parser's
    # occasional one-ULP change in a cached measurement.
    table = pd.read_csv(path, float_precision="round_trip")
    validate_table(table, saved)
    keys = ("dataset", "dataset_sha256", "checkpoint_sha256", "inversion_sha256",
            "task_index", "head_seed", "feature_seed", "split_seed", "label_mode",
            "random_control_seed", "gradient_target", "inference_schema")
    if any(saved.get(k) != metadata.get(k) for k in keys):
        raise ValueError("Control cache has incompatible data/model/extraction provenance.")
    if saved.get("random_control") != "uniform":
        raise ValueError("Control reuse requires registered uniform perturbations.")
    views = ["clean"]
    if float(saved["attack_config"]["delta"]) == float(delta):
        views.append("random_control")
    rows = {(int(row["original_index"]), row["view"]): row
            for row in table.loc[table.view.isin(views)].to_dict("records")}
    return rows, dict(features_sha256=saved["features_sha256"], path=str(Path(path).resolve()), views=views)


def scoped_dataset(checkpoint, output):
    """Upstream uses relative data/; direct CLI calls must not write at repo root."""
    previous = Path.cwd()
    if previous == DETECTOR_ROOT or DETECTOR_ROOT in previous.parents:
        return fixed_dataset_specs(**checkpoint)
    cache = ensure_output_path(output.parent / "dataset_cache")
    cache.mkdir(parents=True, exist_ok=True)
    try:
        os.chdir(cache)
        return fixed_dataset_specs(**checkpoint)
    finally:
        os.chdir(previous)


def main(args):
    inference_schema = getattr(args, "inference_schema", False)
    if inference_schema and args.label_mode != "predicted":
        raise ValueError("Inference schema requires predicted gradients.")
    if getattr(args, "reuse_controls", None) and getattr(args, "random_control", "matched") != "uniform":
        raise ValueError("Control reuse requires --random-control uniform.")
    output = ensure_output_path(args.output_dir)
    if output.exists():
        raise FileExistsError(
            "Extraction output exists; inspect incomplete work or choose a new directory."
        )
    checkpoint = load_pickle(args.checkpoint)
    task = validate_checkpoint(checkpoint, args.incoming_task)
    dataset = checkpoint["dataset"]
    spec = dataset_spec(dataset)
    registration = None
    if getattr(args, "data_manifest", None):
        registration = json.loads(Path(args.data_manifest).read_text())
        if registration["dataset"] != dataset:
            raise ValueError("Checkpoint/data registration mismatch.")
    artifact = load_pickle(args.artifact)
    validate_artifact(artifact, checkpoint)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable.")
    data, _, _, _, _ = scoped_dataset(dict(checkpoint, data_registration=registration)
                                     if registration else checkpoint, output)
    incoming = data["train"][task]
    targets = torch.as_tensor(incoming.targets).long().cpu().numpy()
    if not np.array_equal(np.bincount(targets), [spec["train_per_class"]] * spec["classes_per_task"]):
        raise ValueError("Expected all 500 training originals per incoming class.")
    manifest = create_manifest(targets, args.split_seed, 300, 50, 50)
    validate_manifest_table(manifest, targets)
    set_seed(args.feature_seed)
    model = load_defender(checkpoint, device, args.head_seed)
    before = snapshot_state_dict(model)
    files = matching_inversion_files(args.inversion_dir, expected_count=task,
                                     image_size=spec["image_size"], classes_per_task=spec["classes_per_task"])
    past, directions, records = build_past_references(
        model, files, device, args.reference_batch_size
    )
    for record in records:
        record["sha256"] = sha256_file(record["inversion_file"])
    inv_hash = hashlib.sha256(
        json.dumps(
            [{"task_id": r["task_id"], "sha256": r["sha256"]} for r in records],
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    metadata = {
        "feature_protocol": FEATURE_PROTOCOL,
        "transfer_protocol": TRANSFER_PROTOCOL,
        "head_mode": HEAD_MODE,
        "head_seed": args.head_seed,
        "label_mode": args.label_mode,
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "inversion_sha256": inv_hash,
        "task_index": task,
        "task_order": task_order(dataset),
        "class_ids": task_order(dataset)[task],
        "origin_role": "paired_benchmark",
        "reference_tasks": records,
        "feature_seed": args.feature_seed,
        "split_seed": args.split_seed,
        "random_control_seed": args.random_control_seed,
        "synthetic": False,
        "input_sha256": {
            "checkpoint": sha256_file(args.checkpoint),
            "attack": sha256_file(args.artifact),
        },
        "checkpoint_seed": int(checkpoint["seed"]),
        "attack_seed": int(artifact["seed"]),
        "reference_trust_assumption": "All previously trained tasks and checkpoints are clean.",
        "random_norm_matching": "Before image clamping only",
        "cl_method": checkpoint.get("cont_method_args", {}).get("method"),
        "attack_config": {
            key: artifact.get(key) for key in
            ("mode", "delta", "seed", "n_epochs", "n_iters", "w_cur", "real", "reverse")
        },
        "gradient_target": "predicted_argmax" if args.label_mode == "predicted" else "ground_truth",
    }
    if dataset != "split_cifar100":
        if not registration:
            raise ValueError("Prepared ImageNet extraction needs --data-manifest.")
        metadata.update(dataset=dataset, dataset_sha256=registration["dataset_sha256"],
                        full_training_data=True, data_registration=registration)
    if inference_schema:
        metadata["inference_schema"] = "predicted_without_true_class_v1"
    cached, reuse = reusable_controls(getattr(args, "reuse_controls", None), metadata, artifact["delta"])
    if reuse:
        metadata["control_reuse"] = reuse
    permutation = torch.as_tensor(artifact["rnd_idx_train"]).long().cpu()
    inverse = torch.argsort(permutation)
    noise = torch.as_tensor(artifact["latest_noise"]).float().cpu()
    named, parameters = backbone_parameters(model)
    past_device, directions_device = past.to(device), directions.to(device)
    rows = []
    for item in manifest.itertuples(index=False):
        index = int(item.original_index)
        image = torch.as_tensor(incoming.data[index]).float().cpu()
        source = int(inverse[index])
        delta = noise[source]
        if getattr(args, "random_control", "matched") == "uniform":
            generator = torch.Generator().manual_seed(args.random_control_seed + index)
            random_delta = (torch.rand(delta.shape, generator=generator) * 2 - 1) * float(artifact["delta"])
        else:
            random_delta = matched_random_noise(delta, args.random_control_seed + index)
        for view, label, current in (
            ("clean", 0, image),
            ("poison", 1, (image + delta).clamp(0, 1)),
            ("random_control", -1, (image + random_delta).clamp(0, 1)),
        ):
            row = dict(cached[(index, view)]) if (index, view) in cached else extract_one(
                model,
                named,
                parameters,
                past_device,
                current,
                None if args.label_mode == "predicted" else int(item.class_id),
                device,
                directions_device,
                args.label_mode,
            )
            row.update(
                original_index=index,
                source_index=source,
                class_id=int(item.class_id),
                split=item.split,
                view=view,
                detector_label=label,
                feature_protocol=FEATURE_PROTOCOL,
                head_mode=HEAD_MODE,
                head_seed=args.head_seed,
                attack_seed=int(artifact["seed"]),
                label_mode=args.label_mode,
                task_index=task,
                original_uid="{}:task{}:row{}".format("cifar100" if dataset == "split_cifar100" else dataset, task, index),
            )
            rows.append(row)
        if args.log_every and (index + 1) % args.log_every == 0:
            print(
                "Task {}: {} / {} originals extracted".format(task, index + 1, len(targets)),
                flush=True,
            )
    table = pd.DataFrame(rows)
    fixed_columns = list(FEATURE_COLUMNS)
    if inference_schema:
        table = table.drop(columns=["true_class_probability"])
        fixed_columns.remove("true_class_probability")
        metadata["inference_schema"] = "predicted_without_true_class_v1"
    validate_feature_table(table, fixed_columns if inference_schema else None, spec["classes_per_task"])
    assert_model_unchanged(model, before)
    output.mkdir(parents=True)
    path = output / "features.csv"
    table.to_csv(path, index=False)
    manifest.to_csv(output / "manifest.csv", index=False)
    metadata["random_control"] = getattr(args, "random_control", "matched")
    if metadata["random_control"] == "uniform":
        metadata["random_norm_matching"] = "Uniform [-delta, delta], same L-infinity budget; not matched L2 norms"
    metadata.update(
        feature_columns=[c for c in feature_sets(table)["all"] if c in table],
        fixed_feature_columns=fixed_columns,
        model_unchanged=True,
        row_count=len(table),
        features_sha256=sha256_file(path),
    )
    save_json(metadata, path.with_suffix(".metadata.json"))
    torch.save(
        {"past_direction": past, "task_directions": directions},
        output / "references.pt",
    )
    print("Saved transfer features:", path)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("checkpoint", "artifact", "inversion-dir", "output-dir"):
        parser.add_argument("--" + flag, required=True)
    parser.add_argument("--incoming-task", type=int, required=True)
    parser.add_argument("--data-manifest")
    parser.add_argument("--reuse-controls", help="Previously extracted CSV from the same frozen checkpoint and data.")
    parser.add_argument(
        "--label-mode", choices=("ground_truth", "predicted"), default="predicted"
    )
    parser.add_argument("--random-control", choices=("matched", "uniform"), default="matched")
    parser.add_argument("--inference-schema", action="store_true", help="Omit the legacy true_class_probability column; requires predicted mode.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--head-seed", type=int, default=20260720)
    parser.add_argument("--feature-seed", type=int, default=20260721)
    parser.add_argument("--split-seed", type=int, default=20260720)
    parser.add_argument("--random-control-seed", type=int, default=20260722)
    parser.add_argument("--reference-batch-size", type=int, default=32)
    parser.add_argument("--log-every", type=int, default=100)
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
