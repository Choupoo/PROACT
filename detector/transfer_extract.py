"""Extract stage-aware paired features with unchanged legacy feature semantics."""

import argparse
import hashlib
import json

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
from detector.io_utils import ensure_output_path
from detector.train_detector import feature_sets, validate_feature_table
from detector.transfer_core import (
    TASK_ORDER,
    TRANSFER_PROTOCOL,
    fixed_dataset_specs,
    load_defender,
    validate_artifact,
    validate_checkpoint,
)


def main(args):
    output = ensure_output_path(args.output_dir)
    if output.exists():
        raise FileExistsError(
            "Extraction output exists; inspect incomplete work or choose a new directory."
        )
    checkpoint = load_pickle(args.checkpoint)
    task = validate_checkpoint(checkpoint, args.incoming_task)
    artifact = load_pickle(args.artifact)
    validate_artifact(artifact, checkpoint)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable.")
    data, _, _, _, _ = fixed_dataset_specs(**checkpoint)
    incoming = data["train"][task]
    targets = torch.as_tensor(incoming.targets).long().cpu().numpy()
    if len(targets) != 5000 or not np.array_equal(np.bincount(targets), [500] * 10):
        raise ValueError("Expected 500 CIFAR training originals per incoming class.")
    manifest = create_manifest(targets, args.split_seed, 300, 50, 50)
    validate_manifest_table(manifest, targets)
    set_seed(args.feature_seed)
    model = load_defender(checkpoint, device, args.head_seed)
    before = snapshot_state_dict(model)
    files = matching_inversion_files(args.inversion_dir, expected_count=task)
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
        "task_order": TASK_ORDER,
        "class_ids": TASK_ORDER[task],
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
    }
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
        random_delta = matched_random_noise(delta, args.random_control_seed + index)
        for view, label, current in (
            ("clean", 0, image),
            ("poison", 1, (image + delta).clamp(0, 1)),
            ("random_control", -1, (image + random_delta).clamp(0, 1)),
        ):
            row = extract_one(
                model,
                named,
                parameters,
                past_device,
                current,
                int(item.class_id),
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
                original_uid="cifar100:task{}:row{}".format(task, index),
            )
            rows.append(row)
        if args.log_every and (index + 1) % args.log_every == 0:
            print(
                "Task {}: {} / 5000 originals extracted".format(task, index + 1),
                flush=True,
            )
    table = pd.DataFrame(rows)
    validate_feature_table(table)
    assert_model_unchanged(model, before)
    output.mkdir(parents=True)
    path = output / "features.csv"
    table.to_csv(path, index=False)
    manifest.to_csv(output / "manifest.csv", index=False)
    metadata.update(
        feature_columns=feature_sets(table)["all"],
        fixed_feature_columns=FEATURE_COLUMNS,
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
    parser.add_argument(
        "--label-mode", choices=("ground_truth", "predicted"), default="ground_truth"
    )
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
