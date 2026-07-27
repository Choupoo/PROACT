import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from detector.common import *
from utils import create_load_add_head
from detector import (FEATURE_COLUMNS,
    FEATURE_PROTOCOL,
    HEAD_MODE,
)


def flatten_gradients(gradients):
    return torch.cat(
        [
            gradient.detach().reshape(-1)
            for gradient in gradients
        ]
    )


def backbone_parameters(model):
    named = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if "heads" not in name and parameter.requires_grad
    ]

    if not named:
        raise RuntimeError("No trainable backbone parameters were found.")

    return named, [parameter for _, parameter in named]


def compute_past_gradient_direction(model, inversion_files, device, batch_size):
    _, parameters = backbone_parameters(model)
    direction_sum = None
    records = []

    for task_id, inversion_path in enumerate(inversion_files):
        with np.load(inversion_path) as data:
            x_inv = torch.from_numpy(data["x_dst"]).float()
            y_inv = torch.from_numpy(data["y_dst"]).long()

        loader = DataLoader(TensorDataset(x_inv, y_inv), batch_size=int(batch_size), shuffle=False)

        gradient_sum = None
        samples_seen = 0

        for images, targets in loader:
            images = images.to(device)
            targets = targets.to(device)

            model.zero_grad(set_to_none=True)

            logits = model(images)[task_id]
            loss = F.cross_entropy(logits, targets)

            gradients = torch.autograd.grad(loss, parameters, retain_graph=False, create_graph=False, allow_unused=False)

            flat = flatten_gradients(gradients)
            weighted = flat * images.size(0)

            if gradient_sum is None:
                gradient_sum = weighted
            else:
                gradient_sum = gradient_sum + weighted

            samples_seen += int(images.size(0))

        if samples_seen == 0:
            raise RuntimeError("No inversion samples in {}.".format(inversion_path))

        task_gradient = gradient_sum / float(samples_seen)
        task_norm = task_gradient.norm()

        if task_norm.item() <= 0:
            raise RuntimeError("Historical task {} has a zero gradient.".format(task_id))

        task_direction = (task_gradient / task_norm.clamp_min(1e-12)).detach().cpu()

        if direction_sum is None:
            direction_sum = task_direction
        else:
            direction_sum = direction_sum + task_direction

        records.append({"task_id": int(task_id), "inversion_file": str(inversion_path), "samples": int(samples_seen), "mean_gradient_norm": float(task_norm.item())})

    past_direction = direction_sum / float(len(inversion_files))
    past_direction = (past_direction / past_direction.norm().clamp_min(1e-12))

    return past_direction, records


def matched_random_noise(delta, seed):
    delta = delta.detach().cpu().float()
    flat_abs = delta.reshape(-1).abs()

    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))

    permutation = torch.randperm(flat_abs.numel(), generator=generator)

    signs = torch.randint(low=0, high=2, size=(flat_abs.numel(),), generator=generator, dtype=torch.int64).float()

    signs = signs.mul(2.0).sub(1.0)

    random_flat = flat_abs[permutation] * signs
    random_delta = random_flat.reshape_as(delta)

    original_norm = torch.linalg.vector_norm(delta.double())
    random_norm = torch.linalg.vector_norm(random_delta.double())

    if not torch.isclose(original_norm, random_norm, rtol=1e-10, atol=1e-12):
        raise RuntimeError("Random control does not preserve the L2 norm.")

    if random_delta.abs().max() != delta.abs().max():
        raise RuntimeError("Random control does not preserve the L-infinity norm.")

    return random_delta


def extract_one(model, parameters, past_direction, image, target, device):
    image = image.unsqueeze(0).to(device)
    target = torch.tensor([int(target)], dtype=torch.long, device=device,)

    model.zero_grad(set_to_none=True)

    logits = model(image)[-1]
    loss = F.cross_entropy(logits, target)

    gradients = torch.autograd.grad(loss, parameters, retain_graph=False, create_graph=False, allow_unused=False)

    flat = flatten_gradients(gradients)
    gradient_norm = flat.norm()
    cosine = F.cosine_similarity(flat, past_direction, dim=0, eps=1e-12,)

    return {"loss": float(loss.item()), "grad_norm_l2": float(gradient_norm.item()), "grad_cosine_past": float(cosine.item())}


def validate_manifest_table(manifest, task_targets):
    required = {"original_index", "class_id", "split"}

    missing = required - set(manifest.columns)
    if missing:
        raise KeyError("Manifest is missing columns: {}".format(sorted(missing)))

    if manifest["original_index"].duplicated().any():
        raise RuntimeError("Manifest original_index values are not unique.")

    expected_indices = set(range(len(task_targets)))
    observed_indices = set(manifest["original_index"].astype(int))

    if expected_indices != observed_indices:
        raise RuntimeError("Manifest does not cover task 9 exactly once.")

    row_indices = manifest["original_index"].to_numpy(dtype=np.int64)
    row_labels = manifest["class_id"].to_numpy(dtype=np.int64)

    if not np.array_equal(row_labels, task_targets[row_indices]):
        raise RuntimeError("Manifest class labels do not match task 9.")


def main(args):
    set_seed(args.feature_seed)

    checkpoint_path = Path(args.checkpoint)
    artifact_path = Path(args.artifact)
    manifest_path = Path(args.manifest)
    output_path = Path(args.output)

    checkpoint = load_pickle(checkpoint_path)
    artifact = load_pickle(artifact_path)

    validate_victim_checkpoint(checkpoint)
    validate_attack_artifact(artifact)
    compare_checkpoint_identity(checkpoint, artifact["pretrained_ckpt"])

    manifest = pd.read_csv(manifest_path)

    incoming_dataset, _ = reconstruct_incoming_task(checkpoint)

    task_data = (torch.as_tensor(incoming_dataset.data).float().cpu())
    task_targets = (torch.as_tensor(incoming_dataset.targets).long().cpu().numpy())

    validate_manifest_table(manifest, task_targets)

    permutation = torch.as_tensor(artifact["rnd_idx_train"]).long().cpu()
    attack_noise = torch.as_tensor(artifact["latest_noise"]).float().cpu()

    inverse_permutation = torch.empty_like(permutation)
    inverse_permutation[permutation] = torch.arange(len(permutation), dtype=torch.long)

    if not torch.equal(permutation[inverse_permutation], torch.arange(len(permutation))):
        raise RuntimeError("Failed to invert rnd_idx_train.")

    device = torch.device(args.device)

    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")

    model = create_load_add_head(**checkpoint, load=True)
    model = model.to(device)
    model.eval()
    head_index = initialize_defender_head(model, args.head_seed)
    model.eval()

    if head_index != 9:
        raise RuntimeError("Expected the defender head index to be 9.")

    inversion_files = matching_inversion_files(args.inversion_dir, expected_count=9)

    past_direction_cpu, _ = (compute_past_gradient_direction(model=model, inversion_files=inversion_files, device=device, batch_size=args.reference_batch_size))
    past_direction = past_direction_cpu.to(device)

    _, parameters = backbone_parameters(model)
    model_before = snapshot_state_dict(model)

    selected = (manifest.loc[manifest["split"] != "reserve"].sort_values("original_index").reset_index(drop=True))

    rows = []

    for position, row in enumerate(selected.itertuples(index=False)):
        original_index = int(row.original_index)
        class_id = int(row.class_id)
        split_name = str(row.split)

        source_index = int(inverse_permutation[original_index].item())

        if int(task_targets[original_index]) != class_id:
            raise RuntimeError("Manifest label mismatch at original_index {}.".format(original_index))

        if int(task_targets[int(permutation[source_index].item())]) != class_id:
            raise RuntimeError("Permutation/label mismatch at source_index {}.".format(source_index))

        clean_image = task_data[original_index]
        delta = attack_noise[source_index]
        poison_image = torch.clamp(clean_image + delta, 0.0, 1.0)

        random_delta = matched_random_noise(delta, seed=(int(args.random_control_seed) + original_index))
        random_image = torch.clamp(clean_image + random_delta, 0.0, 1.0)

        views = [("clean", 0, clean_image), ("poison", 1, poison_image), ("random_control", -1, random_image)]

        for view_name, detector_label, image in views:
            features = extract_one(model=model, parameters=parameters, past_direction=past_direction, image=image, target=class_id, device=device)
            features.update({"original_index": original_index, "source_index": source_index, "class_id": class_id, "split": split_name, "view": view_name, "detector_label": int(detector_label), "attack_seed": int(artifact["seed"]), "feature_protocol": (FEATURE_PROTOCOL), "head_mode": HEAD_MODE, "head_seed": int(args.head_seed)})
            rows.append(features)

        if (args.log_every > 0 and (position + 1) % args.log_every == 0):
            print("Processed {}/{} original images.".format(position + 1, len(selected)))

    features = pd.DataFrame(rows)

    expected_feature_columns = set(FEATURE_COLUMNS)
    if not expected_feature_columns.issubset(set(features.columns)):
        raise RuntimeError("Feature table is missing pre-registered features.")

    pair_counts = features.groupby("original_index").size()

    if not (pair_counts == 3).all():
        raise RuntimeError("Each selected original image must have exactly clean, poison and random-control rows.")

    for original_index, group in features.groupby("original_index"):
        if set(group["view"]) != {"clean", "poison", "random_control"}:
            raise RuntimeError("Incomplete views for original_index {}.".format(original_index))

        if group["split"].nunique() != 1:
            raise RuntimeError("Views from one original image entered different splits.")

        if group["class_id"].nunique() != 1:
            raise RuntimeError("Views from one original image have different labels.")

    assert_model_unchanged(model,model_before)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(output_path, index=False)

    reference_path = output_path.with_name("past_gradient_direction.pt")
    torch.save(past_direction_cpu, reference_path)

    print("\nSaved:", output_path)
    print("Saved:", reference_path)
    print("\nView counts:")
    print(features["view"].value_counts())
    print("\nMean features:")
    print(features.groupby("view")[FEATURE_COLUMNS].mean())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=("Extract three pre-registered features from clean, BrainWash and matched-random task-9 views."))
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--inversion-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device",default="cuda")
    parser.add_argument("--head-seed", type=int, default=20260720)
    parser.add_argument("--feature-seed", type=int, default=20260721)
    parser.add_argument("--random-control-seed", type=int, default=20260722)
    parser.add_argument("--reference-batch-size", type=int, default=32)
    parser.add_argument("--log-every", type=int, default=100)
    main(parser.parse_args())