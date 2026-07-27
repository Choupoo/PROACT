import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from detector.common import *

def create_manifest(targets, split_seed, train_per_class, validation_per_class, test_per_class):
    targets = np.asarray(targets, dtype=np.int64)

    if targets.ndim != 1:
        raise ValueError("targets must be one-dimensional.")

    counts = {"train": int(train_per_class), "validation": int(validation_per_class),"test": int(test_per_class)}

    for name, count in counts.items():
        if count <= 0:
            raise ValueError("{}_per_class must be positive.".format(name))

    selected_count = sum(counts.values())
    all_indices = np.arange(len(targets), dtype=np.int64)
    classes = sorted(np.unique(targets).tolist())
    rng = np.random.default_rng(int(split_seed))
    rows = []

    for class_id in classes:
        class_indices = all_indices[targets == class_id].copy()

        rng.shuffle(class_indices)

        if len(class_indices) <= selected_count:
            raise ValueError("Class {} has {} samples, but {} selected samples and a non-empty reserve are required.".format(class_id, len(class_indices), selected_count))

        train_end = int(train_per_class)
        validation_end = train_end + int(validation_per_class)
        test_end = validation_end + int(test_per_class)

        assignments = {"train": class_indices[:train_end], "validation": class_indices[train_end:validation_end], "test": class_indices[validation_end:test_end], "reserve": class_indices[test_end:]}

        for split_name, indices in assignments.items():
            for original_index in indices.tolist():
                rows.append({"original_index": int(original_index), "class_id": int(class_id), "split": split_name})

    return pd.DataFrame(rows).sort_values("original_index").reset_index(drop=True)

def validate_manifest(manifest, targets, train_per_class, validation_per_class, test_per_class):
    targets = np.asarray(targets, dtype=np.int64)

    expected_columns = {"original_index", "class_id", "split"}

    missing = expected_columns - set(manifest.columns)
    if missing:
        raise KeyError("Manifest is missing columns: {}".format(sorted(missing)))

    if len(manifest) != len(targets):
        raise RuntimeError("Manifest has {} rows; task 9 has {} samples.".format(len(manifest),len(targets)))

    if manifest["original_index"].duplicated().any():
        raise RuntimeError("An original image appears in more than one split.")

    expected_indices = set(range(len(targets)))
    observed_indices = set(manifest["original_index"].astype(int))

    if observed_indices != expected_indices:
        raise RuntimeError("Manifest does not cover every task-9 image exactly once.")

    allowed_splits = {"train", "validation", "test", "reserve"}

    if set(manifest["split"].unique()) != allowed_splits:
        raise RuntimeError("Manifest must contain train, validation, test and reserve.")

    indices = manifest["original_index"].to_numpy(dtype=np.int64)
    labels = manifest["class_id"].to_numpy(dtype=np.int64)

    if not np.array_equal(labels, targets[indices]):
        raise RuntimeError("At least one manifest class_id is incorrect.")

    grouped = manifest.groupby(["class_id", "split"]).size()

    expected_selected = {"train": int(train_per_class), "validation": int(validation_per_class), "test": int(test_per_class)}

    for class_id in sorted(np.unique(targets).tolist()):
        class_total = int(np.sum(targets == class_id))

        for split_name, expected_count in expected_selected.items():
            observed = int(grouped.get((class_id, split_name), 0,))

            if observed != expected_count:
                raise RuntimeError("class={}, split={}: expected {}, found {}.".format(class_id, split_name, expected_count, observed))

        expected_reserve = (
            class_total
            - int(train_per_class)
            - int(validation_per_class)
            - int(test_per_class)
        )

        observed_reserve = int(
            grouped.get((class_id, "reserve"), 0,))

        if observed_reserve != expected_reserve:
            raise RuntimeError("class={}, reserve: expected {}, found {}.".format(class_id, expected_reserve, observed_reserve))


def main(args):
    checkpoint_path = Path(args.checkpoint)
    output_path = Path(args.output)

    checkpoint = load_pickle(checkpoint_path)
    validate_victim_checkpoint(checkpoint)

    incoming_dataset, _ = reconstruct_incoming_task(checkpoint)

    targets = (torch.as_tensor(incoming_dataset.targets).long().cpu().numpy())

    if len(incoming_dataset) != 5000:
        raise RuntimeError("Expected 5,000 task-9 training images, found {}.".format(len(incoming_dataset)))

    unique, counts = np.unique(targets, return_counts=True)

    if unique.tolist() != list(range(10)):
        raise RuntimeError("Expected task-local labels 0..9, found {}.".format(unique.tolist()))

    if not np.all(counts == 500):
        raise RuntimeError("Expected 500 images per class, found {}.".format(counts.tolist()))

    manifest = create_manifest(targets=targets, split_seed=args.split_seed, train_per_class=args.train_per_class, validation_per_class=args.validation_per_class, test_per_class=args.test_per_class)

    validate_manifest(manifest=manifest, targets=targets, train_per_class=args.train_per_class, validation_per_class=args.validation_per_class, test_per_class=args.test_per_class)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(output_path, index=False)

    print(manifest["split"].value_counts().sort_index())
    print()
    print(pd.crosstab(manifest["class_id"], manifest["split"]))
    print("\nSaved:", output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=("Create the pre-attack task-9 manifest for the clean supervised detector baseline."))
    parser.add_argument("--checkpoint", required=True, help=("Victim checkpoint trained on tasks 0-8, with checkpoint['task_num'] == 9."))
    parser.add_argument("--output", required=True)
    parser.add_argument("--split-seed", type=int, default=20260720)
    parser.add_argument("--train-per-class", type=int, default=300)
    parser.add_argument("--validation-per-class", type=int, default=50)
    parser.add_argument("--test-per-class", type=int, default=50,)
    main(parser.parse_args())