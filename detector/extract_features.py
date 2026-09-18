"""Measure a frozen model's response to clean, poisoned and random inputs."""

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from detector import (
    FEATURE_COLUMNS,
    FEATURE_PROTOCOL,
    GRAD_LAYER_PREFIX,
    GRAD_PARAM_PREFIX,
    HEAD_MODE,
    STAGE_NAMES,
)
from detector.common import (
    assert_model_unchanged,
    compare_checkpoint_identity,
    get_model_heads,
    initialize_defender_head,
    inversion_task_id,
    load_pickle,
    matching_inversion_files,
    reconstruct_incoming_task,
    save_json,
    set_seed,
    sha256_file,
    snapshot_state_dict,
    validate_attack_artifact,
    validate_victim_checkpoint,
)
from detector.create_manifest import validate_manifest_table


def flatten_gradients(gradients):
    return torch.cat([gradient.detach().reshape(-1) for gradient in gradients])


def backbone_parameters(model):
    """Return the same ordered shared parameters for every gradient vector."""
    named = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if not normalize_parameter_name(name).startswith("heads.")
    ]

    if not named:
        raise RuntimeError("No trainable backbone parameters were found.")

    if any(not parameter.requires_grad for _, parameter in named):
        raise RuntimeError(
            "Backbone requires_grad must remain enabled to measure gradients; omit optimizer updates to freeze weights."
        )

    return named, [parameter for _, parameter in named]


def normalize_parameter_name(name):
    if name.startswith("module."):
        name = name[len("module.") :]
    return name


def parameter_to_stage(name):
    name = normalize_parameter_name(name)
    if name.startswith("conv1.") or name.startswith("bn1."):
        return "stem"

    if name.startswith("layer1."):
        return "layer1"

    if name.startswith("layer2."):
        return "layer2"

    if name.startswith("layer3."):
        return "layer3"

    if name.startswith("layer4."):
        return "layer4"

    raise RuntimeError("Unknown backbone parameter: {}".format(name))


def compute_stage_gradient_norms(
    named_parameters,
    gradients,
):
    """One L2 norm per stage, including convolution/BN weights and biases."""
    if len(named_parameters) != len(gradients):
        raise ValueError("Parameters and gradients must have the same length.")
    if not gradients:
        raise ValueError("No backbone gradients were supplied.")
    for (name, _), gradient in zip(named_parameters, gradients):
        if gradient is None:
            raise RuntimeError("Missing backbone gradient: {}.".format(name))
    squared_sums = {
        stage: gradients[0].new_zeros((), dtype=torch.float32) for stage in STAGE_NAMES
    }

    for (name, _), gradient in zip(
        named_parameters,
        gradients,
    ):
        stage = parameter_to_stage(name)

        squared_sums[stage] += gradient.detach().float().square().sum()

    # One device-to-host transfer instead of synchronizing once per parameter.
    norms = (
        torch.stack([squared_sums[stage] for stage in STAGE_NAMES])
        .sqrt()
        .cpu()
        .tolist()
    )

    return {
        "grad_norm_stage_{}".format(stage): norm
        for stage, norm in zip(STAGE_NAMES, norms)
    }


def build_past_references(model, inversion_files, device, batch_size):
    """Average unit per-task mean gradients, then normalize that average.

    Within a task, weight each mean-loss batch gradient by its sample count.
    Across tasks, use unit directions so large task-gradient norms do not dominate.
    This is not the direction of the raw pooled historical gradient.
    """
    if any(module.training for module in model.modules()):
        raise RuntimeError("Reference gradients require model.eval().")
    if not inversion_files or batch_size <= 0:
        raise ValueError("Reference files and a positive batch size are required.")
    _, parameters = backbone_parameters(model)
    task_directions = {}
    records = []
    seen_tasks = set()
    past_task_count = len(get_model_heads(model)) - 1

    for inversion_path in inversion_files:
        with np.load(inversion_path) as data:
            task_id = inversion_task_id(inversion_path, data)
            if task_id in seen_tasks or not 0 <= task_id < past_task_count:
                raise ValueError(
                    "Duplicate or invalid historical task id: {}.".format(task_id)
                )
            seen_tasks.add(task_id)
            x_inv = torch.from_numpy(data["x_dst"]).float()
            y_inv = torch.from_numpy(data["y_dst"]).long()

        loader = DataLoader(
            TensorDataset(x_inv, y_inv), batch_size=int(batch_size), shuffle=False
        )

        gradient_sum = None
        samples_seen = 0

        for images, targets in loader:
            images = images.to(device)
            targets = targets.to(device)

            model.zero_grad(set_to_none=True)

            logits = model(images)[task_id]
            loss = F.cross_entropy(logits, targets)

            gradients = torch.autograd.grad(
                loss,
                parameters,
                retain_graph=False,
                create_graph=False,
                allow_unused=False,
            )

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

        if not torch.isfinite(task_norm) or task_norm.item() <= 1e-12:
            raise RuntimeError(
                "Historical task {} has a non-finite or near-zero gradient.".format(
                    task_id
                )
            )

        task_direction = (task_gradient / task_norm.clamp_min(1e-12)).detach().cpu()

        task_directions[task_id] = task_direction

        records.append(
            {
                "task_id": int(task_id),
                "inversion_file": str(inversion_path),
                "samples": int(samples_seen),
                "mean_gradient_norm": float(task_norm.item()),
            }
        )

    records.sort(key=lambda record: record["task_id"])
    directions = torch.stack([task_directions[record["task_id"]] for record in records])
    past_direction = directions.mean(dim=0)
    past_norm = past_direction.norm()
    if not torch.isfinite(past_norm) or past_norm.item() <= 1e-12:
        raise RuntimeError(
            "Historical task directions cancel; past cosine is undefined."
        )
    past_direction = past_direction / past_norm

    return past_direction, directions, records


def compute_past_gradient_direction(model, inversion_files, device, batch_size):
    """Backward-compatible two-result wrapper for the original baseline API."""
    direction, _, records = build_past_references(
        model, inversion_files, device, batch_size
    )
    return direction, records


def gradient_group_features(named_parameters, gradients):
    """Frobenius norm per parameter tensor and L2 norm per owning module.

    A layer/module norm includes its weight and bias tensors. These are norms
    of gradients, not the model weights, and the head is excluded upstream.
    """
    if len(named_parameters) != len(gradients) or not gradients:
        raise ValueError("Nonempty aligned parameter and gradient lists are required.")
    names, squared, groups = [], [], {}
    for (name, _), gradient in zip(named_parameters, gradients):
        if gradient is None:
            raise RuntimeError("Missing backbone gradient: {}.".format(name))
        name = normalize_parameter_name(name)
        names.append(name)
        value = gradient.detach().float().square().sum()
        squared.append(value)
        module_name = name.rsplit(".", 1)[0]
        groups[module_name] = groups.get(module_name, 0) + value
    module_names = list(groups)
    values = torch.stack(squared + [groups[name] for name in module_names])
    norms = values.sqrt().cpu().tolist()
    features = {
        GRAD_PARAM_PREFIX + name: norm for name, norm in zip(names, norms[: len(names)])
    }
    features.update(
        {
            GRAD_LAYER_PREFIX + name: norm
            for name, norm in zip(module_names, norms[len(names) :])
        }
    )
    return features


def matched_random_noise(delta, seed):
    """Match perturbation magnitudes before image clipping, not after clipping."""
    delta = delta.detach().cpu().float()
    flat_abs = delta.reshape(-1).abs()

    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))

    permutation = torch.randperm(flat_abs.numel(), generator=generator)

    signs = torch.randint(
        low=0, high=2, size=(flat_abs.numel(),), generator=generator, dtype=torch.int64
    ).float()

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


def extract_one(
    model,
    named_parameters,
    parameters,
    past_direction,
    image,
    target,
    device,
    task_directions=None,
    label_mode="ground_truth",
):
    """Measure one frozen forward/backward pass without an optimizer update.

    ``predicted`` chooses the incoming head's argmax as the CE target and never
    reads ``target``. In that mode, ``true_class_probability`` is the probability
    of the pseudo-target (equal to confidence), not a ground-truth quantity.
    Omitted task directions repeat the global reference for baseline API callers;
    production extraction always supplies the actual per-task directions.
    """
    if any(module.training for module in model.modules()):
        raise RuntimeError("Pre-training features require model.eval().")
    if label_mode not in {"ground_truth", "predicted"}:
        raise ValueError("label_mode must be ground_truth or predicted.")
    image = image.unsqueeze(0).to(device)

    model.zero_grad(set_to_none=True)

    activations = []

    def capture_head_input(_module, inputs):
        activations.append(inputs[0].detach())

    hook = get_model_heads(model)[-1].register_forward_pre_hook(capture_head_input)
    try:
        logits = model(image)[-1]
    finally:
        hook.remove()
    if len(activations) != 1:
        raise RuntimeError("Expected one incoming-head activation per sample.")
    predicted_class_id = int(logits.detach().argmax(dim=1).item())
    target_id = predicted_class_id if label_mode == "predicted" else int(target)
    target = torch.tensor([target_id], dtype=torch.long, device=device)
    log_probabilities = F.log_softmax(logits.detach(), dim=1)[0]
    probabilities = log_probabilities.exp()
    largest = probabilities.topk(2).values

    loss = F.cross_entropy(
        logits,
        target,
    )

    gradients = torch.autograd.grad(
        loss,
        parameters,
        retain_graph=False,
        create_graph=False,
        allow_unused=False,
    )

    flat = flatten_gradients(gradients)

    gradient_norm = flat.norm()

    cosine = F.cosine_similarity(
        flat,
        past_direction.to(device=flat.device, dtype=flat.dtype),
        dim=0,
        eps=1e-12,
    )

    if task_directions is None:
        task_directions = past_direction.unsqueeze(0)
    task_directions = task_directions.to(device=flat.device, dtype=flat.dtype)
    if task_directions.ndim != 2 or task_directions.shape[1] != flat.numel():
        raise ValueError("Task directions must be a nonempty task-by-parameter matrix.")
    if task_directions.shape[0] == 0 or not torch.isfinite(task_directions).all():
        raise ValueError("Task directions must be nonempty and finite.")
    task_cosines = F.cosine_similarity(
        flat.unsqueeze(0), task_directions, dim=1, eps=1e-12
    )

    stage_features = compute_stage_gradient_norms(
        named_parameters=named_parameters,
        gradients=gradients,
    )

    reconstructed_global_norm = math.sqrt(
        sum(value**2 for value in stage_features.values())
    )

    if not math.isclose(
        reconstructed_global_norm,
        float(gradient_norm.item()),
        rel_tol=1e-5,
        abs_tol=1e-7,
    ):
        raise RuntimeError(
            "Stage-wise norms do not reconstruct "
            "the global gradient norm: "
            "global={}, reconstructed={}".format(
                float(gradient_norm.item()),
                reconstructed_global_norm,
            )
        )

    features = {
        "loss": float(loss.item()),
        "grad_norm_l2": float(gradient_norm.item()),
        "grad_cosine_past": float(cosine.item()),
        "grad_cosine_task_min": float(task_cosines.min().item()),
        "grad_cosine_task_max": float(task_cosines.max().item()),
        "grad_cosine_task_mean": float(task_cosines.mean().item()),
        "entropy": float(-(probabilities * log_probabilities).sum().item()),
        "confidence": float(largest[0].item()),
        "true_class_probability": float(probabilities[target_id].item()),
        "margin": float((largest[0] - largest[1]).item()),
        "activation_norm_l2": float(activations[0].reshape(-1).norm().item()),
    }

    features.update(stage_features)
    features.update(gradient_group_features(named_parameters, gradients))
    features.update(
        {
            "grad_cosine_task_{}".format(task_id): float(value)
            for task_id, value in enumerate(task_cosines.detach().cpu().tolist())
        }
    )
    features["predicted_class_id"] = predicted_class_id

    if not all(math.isfinite(value) for value in features.values()):
        raise RuntimeError("Extracted features contain non-finite values.")

    return features


def load_input_npz(path, label_mode):
    """Load Task-9 images; predicted mode does not access a targets array."""
    if label_mode not in {"ground_truth", "predicted"}:
        raise ValueError("Unknown label mode: {}.".format(label_mode))
    with np.load(path, allow_pickle=False) as data:
        if "images" not in data:
            raise KeyError("Incoming NPZ must contain an images array.")
        images = np.asarray(data["images"])
        if (
            images.ndim != 4
            or tuple(images.shape[1:]) != (3, 32, 32)
            or not len(images)
        ):
            raise ValueError(
                "Incoming images must have nonempty N x 3 x 32 x 32 shape."
            )
        if not np.isfinite(images).all() or images.min() < 0 or images.max() > 1:
            raise ValueError("Incoming images must be finite and scaled to [0, 1].")
        targets = None
        if label_mode == "ground_truth":
            if "targets" not in data:
                raise KeyError(
                    "ground_truth mode requires task-local targets in the NPZ."
                )
            targets = np.asarray(data["targets"])
            if targets.shape != (len(images),) or not np.isfinite(targets).all():
                raise ValueError(
                    "Incoming targets must be finite and aligned with images."
                )
            if (
                not np.equal(targets, targets.astype(np.int64)).all()
                or not ((targets >= 0) & (targets < 10)).all()
            ):
                raise ValueError("Incoming targets must be local integers 0..9.")
            targets = targets.astype(np.int64)
    return torch.from_numpy(images).float(), targets


def benchmark_samples(args, checkpoint):
    """Yield all manifest splits, reserving reserve for dataset calibration."""
    artifact = load_pickle(args.artifact)
    validate_attack_artifact(artifact)
    compare_checkpoint_identity(checkpoint, artifact["pretrained_ckpt"])
    if not args.manifest:
        raise ValueError("--artifact requires --manifest.")
    manifest = pd.read_csv(args.manifest)
    incoming_dataset, _ = reconstruct_incoming_task(checkpoint)
    task_data = torch.as_tensor(incoming_dataset.data).float().cpu()
    task_targets = torch.as_tensor(incoming_dataset.targets).long().cpu().numpy()
    validate_manifest_table(manifest, task_targets)
    permutation = torch.as_tensor(artifact["rnd_idx_train"]).long().cpu()
    attack_noise = torch.as_tensor(artifact["latest_noise"]).float().cpu()
    inverse_permutation = torch.empty_like(permutation)
    inverse_permutation[permutation] = torch.arange(len(permutation), dtype=torch.long)
    if not torch.equal(
        permutation[inverse_permutation], torch.arange(len(permutation))
    ):
        raise RuntimeError("Failed to invert rnd_idx_train.")
    for row in manifest.sort_values("original_index").itertuples(index=False):
        index = int(row.original_index)
        source_index = int(inverse_permutation[index].item())
        image = task_data[index]
        delta = attack_noise[source_index]
        random_delta = matched_random_noise(
            delta, int(args.random_control_seed) + index
        )
        views = (
            ("clean", 0, image),
            ("poison", 1, (image + delta).clamp(0, 1)),
            ("random_control", -1, (image + random_delta).clamp(0, 1)),
        )
        for view, detector_label, current_image in views:
            yield (
                current_image,
                int(row.class_id),
                {
                    "original_index": index,
                    "source_index": source_index,
                    "class_id": int(row.class_id),
                    "split": str(row.split),
                    "view": view,
                    "detector_label": detector_label,
                    "attack_seed": int(artifact["seed"]),
                },
            )


def input_samples(path, label_mode):
    images, targets = load_input_npz(path, label_mode)
    for index, image in enumerate(images):
        target = None if targets is None else int(targets[index])
        yield image, target, {"original_index": index}


def reference_samples(inversion_files):
    """Reference inputs pass through the incoming head, never historical heads."""
    index = 0
    for path in inversion_files:
        with np.load(path, allow_pickle=False) as data:
            task_id = inversion_task_id(path, data)
            images = torch.from_numpy(data["x_dst"]).float()
        for sample_index, image in enumerate(images):
            yield (
                image,
                None,
                {
                    "original_index": index,
                    "reference_task_id": task_id,
                    "reference_id": "task{}:sample{}".format(task_id, sample_index),
                },
            )
            index += 1


def validate_loaded_model_keys(model, checkpoint_state):
    """Allow known EWC Fisher buffers, never arbitrary ignored backbone keys."""
    expected, saved = set(model.state_dict()), set(checkpoint_state)
    if (expected - saved) - {"heads.9.weight", "heads.9.bias"}:
        raise RuntimeError(
            "Checkpoint does not cover the backbone and historical heads."
        )
    fisher_shapes = {
        name.replace(".", "_") + "_fisher": parameter.shape
        for name, parameter in model.named_parameters()
    }
    for key in saved - expected:
        if key not in fisher_shapes:
            raise RuntimeError("Unexpected checkpoint model key: {}".format(key))
        value = torch.as_tensor(checkpoint_state[key])
        if (
            value.shape != fisher_shapes[key]
            or not torch.isfinite(value).all()
            or (value < 0).any()
        ):
            raise RuntimeError("Invalid EWC Fisher buffer: {}".format(key))


def main(args):
    # Keep model/data dependencies out of the numerical helpers and --help path.
    from detector.io_utils import ensure_output_path
    from utils import create_load_add_head

    mode_count = sum(
        bool(value) for value in (args.artifact, args.input_npz, args.reference_only)
    )
    if mode_count != 1:
        raise ValueError(
            "Choose exactly one of --artifact, --input-npz, --reference-only."
        )
    if args.manifest and not args.artifact:
        raise ValueError("--manifest is only valid with --artifact.")
    if args.artifact and not args.manifest:
        raise ValueError("--artifact requires --manifest.")
    if args.reference_only and args.label_mode != "predicted":
        raise ValueError("--reference-only requires --label-mode predicted.")
    checkpoint_path = Path(args.checkpoint)
    output_path = ensure_output_path(args.output)
    metadata_path = ensure_output_path(output_path.with_suffix(".metadata.json"))
    reference_path = ensure_output_path(output_path.with_suffix(".references.pt"))
    checkpoint = load_pickle(checkpoint_path)
    validate_victim_checkpoint(checkpoint)
    checkpoint_sha256 = sha256_file(checkpoint_path)
    input_hashes = {"checkpoint": checkpoint_sha256}
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    set_seed(args.feature_seed)
    model = create_load_add_head(**checkpoint, load=True).to(device)
    validate_loaded_model_keys(model, checkpoint["model"])
    model.eval()
    if initialize_defender_head(model, args.head_seed) != 9:
        raise RuntimeError("Expected the defender head index to be 9.")
    model_before = snapshot_state_dict(model)
    inversion_files = matching_inversion_files(args.inversion_dir, expected_count=9)
    past_cpu, tasks_cpu, records = build_past_references(
        model, inversion_files, device, args.reference_batch_size
    )
    inversion_identities = []
    for record in records:
        record["sha256"] = sha256_file(record["inversion_file"])
        inversion_identities.append(
            {"task_id": record["task_id"], "sha256": record["sha256"]}
        )
    inversion_sha256 = hashlib.sha256(
        json.dumps(inversion_identities, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()
    provenance = {
        "feature_protocol": FEATURE_PROTOCOL,
        "head_mode": HEAD_MODE,
        "head_seed": int(args.head_seed),
        "label_mode": args.label_mode,
        "checkpoint_sha256": checkpoint_sha256,
        "inversion_sha256": inversion_sha256,
    }
    if args.artifact:
        samples = benchmark_samples(args, checkpoint)
        origin_role = "paired_benchmark"
        input_hashes.update(
            {
                "attack": sha256_file(args.artifact),
                "manifest": sha256_file(args.manifest),
            }
        )
    elif args.input_npz:
        samples = input_samples(args.input_npz, args.label_mode)
        origin_role = "incoming"
        input_hashes["incoming_npz"] = sha256_file(args.input_npz)
    else:
        samples = reference_samples(inversion_files)
        origin_role = "historical_inversion"

    named, parameters = backbone_parameters(model)
    parameter_groups = [
        {
            "name": normalize_parameter_name(name),
            "shape": list(parameter.shape),
            "stage": parameter_to_stage(name),
            "layer": normalize_parameter_name(name).rsplit(".", 1)[0],
            "parameter_feature": GRAD_PARAM_PREFIX + normalize_parameter_name(name),
            "layer_feature": GRAD_LAYER_PREFIX
            + normalize_parameter_name(name).rsplit(".", 1)[0],
        }
        for name, parameter in named
    ]
    past_direction, task_directions = past_cpu.to(device), tasks_cpu.to(device)
    rows = []
    for count, (image, target, identity) in enumerate(samples, start=1):
        row = extract_one(
            model,
            named,
            parameters,
            past_direction,
            image,
            target,
            device,
            task_directions=task_directions,
            label_mode=args.label_mode,
        )
        row.update(identity)
        if "class_id" not in row:
            row["class_id"] = (
                int(target)
                if args.label_mode == "ground_truth"
                else row["predicted_class_id"]
            )
        row.update(provenance)
        rows.append(row)
        if args.log_every > 0 and count % args.log_every == 0:
            print("Processed {} sample views.".format(count))
    if not rows:
        raise ValueError("No inputs were available for feature extraction.")
    features = pd.DataFrame(rows)
    dynamic_columns = [
        column
        for column in features
        if (
            column.startswith(GRAD_PARAM_PREFIX)
            or column.startswith(GRAD_LAYER_PREFIX)
            or (
                column.startswith("grad_cosine_task_")
                and column.rsplit("_", 1)[-1].isdigit()
            )
        )
    ]
    feature_columns = FEATURE_COLUMNS + dynamic_columns
    if not np.isfinite(features[feature_columns].to_numpy(dtype=float)).all():
        raise RuntimeError("Extracted feature table contains nonfinite values.")
    assert_model_unchanged(model, model_before)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(output_path, index=False)
    torch.save(
        {
            "past_direction": past_cpu,
            "task_directions": tasks_cpu,
            "task_ids": [record["task_id"] for record in records],
        },
        reference_path,
    )
    metadata = dict(provenance)
    metadata.update(
        {
            "origin_role": origin_role,
            "feature_columns": feature_columns,
            "fixed_feature_columns": FEATURE_COLUMNS,
            "feature_seed": int(args.feature_seed),
            "random_control_seed": int(args.random_control_seed),
            "reference_batch_size": int(args.reference_batch_size),
            "reference_rule": "normalize(mean_t(normalize(mean_i(backbone_gradient(t, i)))))",
            "reference_tasks": records,
            "reference_direction_path": str(reference_path),
            "parameter_groups": parameter_groups,
            "model_unchanged": True,
            "random_norm_matching": "Before clamp(x + delta, 0, 1) only",
            "target_probability_semantics": "true-class probability"
            if args.label_mode == "ground_truth"
            else "frozen incoming-head argmax probability; no true labels used",
            "input_sha256": input_hashes,
            "features_sha256": sha256_file(output_path),
            "row_count": len(features),
        }
    )
    save_json(metadata, metadata_path)
    print(
        "Saved {} rows and {} features: {}".format(
            len(features), len(feature_columns), output_path
        )
    )
    print("Saved metadata:", metadata_path)


def build_parser():
    parser = argparse.ArgumentParser(
        description="Extract frozen-model gradients, uncertainty and activation features from benchmark, incoming or historical-reference inputs."
    )
    parser.add_argument("--checkpoint", required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--artifact",
        help="BrainWash artifact for paired clean/poison/random benchmark views.",
    )
    mode.add_argument(
        "--input-npz",
        help="Incoming NPZ: images N x 3 x 32 x 32 in [0,1], optional targets.",
    )
    mode.add_argument(
        "--reference-only",
        action="store_true",
        help="Extract historical proxies through the incoming head; requires predicted label mode.",
    )
    parser.add_argument(
        "--manifest",
        help="Required with --artifact; all splits, including reserve, are extracted.",
    )
    parser.add_argument(
        "--label-mode", choices=("ground_truth", "predicted"), default="ground_truth"
    )
    parser.add_argument("--inversion-dir", required=True)
    parser.add_argument(
        "--output", required=True, help="Feature CSV inside the detector directory."
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--head-seed", type=int, default=20260720)
    parser.add_argument("--feature-seed", type=int, default=20260721)
    parser.add_argument("--random-control-seed", type=int, default=20260722)
    parser.add_argument("--reference-batch-size", type=int, default=32)
    parser.add_argument("--log-every", type=int, default=100)
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
