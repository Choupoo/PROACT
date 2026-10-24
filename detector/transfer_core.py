"""Explicit stage contracts for the transfer experiment and ImageNet extension.

The legacy Task-9 protocol stays unchanged. Task indices here are zero based:
source 1 is the second task; target 9 is the tenth task. tinyImageNet has 20
tasks and uses target 19. The legacy default bounds remain unchanged.
"""

import copy
import hashlib
import json
import re

import numpy as np
import torch

from detector.common import normalize_task_order, validate_victim_checkpoint
from detector.image_data import dataset_spec, task_order

TRANSFER_PROTOCOL = "source_supervised_task_transfer_v1"
TASK_ORDER = [list(range(10 * t, 10 * (t + 1))) for t in range(10)]


def task_index(value, task_count=10):
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, np.integer))
        or not 1 <= value < task_count
    ):
        raise ValueError("Incoming task must be a zero-based integer in 1..{}.".format(task_count - 1))
    return int(value)


def validate_checkpoint(checkpoint, expected_task=None):
    """Validate real metadata; never relabel an early checkpoint as Task 9."""
    spec = dataset_spec(checkpoint.get("dataset", "split_cifar100"))
    task = task_index(checkpoint.get("task_num"), spec["tasks"])
    if expected_task is not None and task != expected_task:
        raise ValueError("Checkpoint stage differs from requested incoming task.")
    # Reuse legacy structural checks on a copy, then apply the transfer contract.
    structure = dict(checkpoint, task_num=9)
    validate_victim_checkpoint(structure, transfer_datasets=True)
    if normalize_task_order(checkpoint["task_order"]) != task_order(checkpoint["dataset"]):
        raise ValueError("Transfer requires the registered fixed class order.")
    return task


def validate_artifact(artifact, checkpoint, expected_size=None):
    from detector.common import compare_checkpoint_identity, validate_attack_artifact

    task = validate_checkpoint(checkpoint)
    spec = dataset_spec(checkpoint["dataset"])
    if expected_size is None:
        expected_size = spec["classes_per_task"] * spec["train_per_class"]
    validate_checkpoint(artifact["pretrained_ckpt"], task)
    if artifact.get("attacked_task") != task:
        raise ValueError("Attack targets a different incoming task.")
    # Existing noise/permutation and tensor identity checks are stage-independent.
    left = dict(checkpoint, task_num=9)
    right = dict(artifact["pretrained_ckpt"], task_num=9)
    compare_checkpoint_identity(left, right, transfer_datasets=True)
    validate_attack_artifact(
        dict(artifact, pretrained_ckpt=right, attacked_task=9), expected_size,
        allowed_modes=("reckless", "cautious"),
        transfer_datasets=True,
    )
    if artifact.get("real", False):
        raise ValueError("This registered protocol uses inversion-based BrainWash.")


def fixed_dataset_specs(**checkpoint):
    """Adapter for upstream get_dataset_specs: preserve 10 classes per task."""
    from data_utils import generate_split_cifar100_tasks

    task = validate_checkpoint(checkpoint)
    if checkpoint["dataset"] != "split_cifar100":
        from detector.image_data import PreparedData
        if "data_registration" not in checkpoint:
            raise ValueError("ImageNet extraction requires registered prepared data.")
        return PreparedData(checkpoint["data_registration"]).specs(**checkpoint)
    data, order = generate_split_cifar100_tasks(
        task_num=10, seed=checkpoint["seed"], rnd_order=False, order=np.arange(100)
    )
    if normalize_task_order(order) != TASK_ORDER:
        raise ValueError("Reconstructed class order differs from the fixed protocol.")
    # Upstream attacks [-1]; expose only the past and current tasks.
    return {k: v[: task + 1] for k, v in data.items()}, order, 32, 10, 1


def truncate_training_tasks(result, incoming):
    """Limit effectiveness training to one incoming task, not every future task."""
    data, taskcla, size, order = result
    if normalize_task_order(order) != TASK_ORDER or data["ncla"] != 100:
        raise ValueError("Upstream training changed the fixed ten-task partition.")
    return data, taskcla[: incoming + 1], size, order


def load_defender(checkpoint, device, head_seed):
    from resnet import ResNet18

    task = validate_checkpoint(checkpoint)
    spec = dataset_spec(checkpoint["dataset"])
    width = spec["classes_per_task"]
    model = ResNet18(task + 1, width, nf=32).to(device)
    expected = model.state_dict()
    saved = checkpoint["model"]
    filtered = {}
    fishers = {
        n.replace(".", "_") + "_fisher": p.shape for n, p in model.named_parameters()
    }
    # CL bookkeeping buffers are not forward-pass parameters. Validate against
    # the registered full-head schema before ignoring them (RWALK scores can be signed).
    shapes = {name: param.shape for name, param in model.named_parameters()}
    for index in range(task + 1, spec["tasks"]):
        shapes["heads.{}.weight".format(index)] = (width, model.emb_dim)
        shapes["heads.{}.bias".format(index)] = (width,)
    bookkeeping = {}
    for name, shape in shapes.items():
        key = name.replace(".", "_")
        for prefix in ("fisher_", "running_fisher_", "omega_", "s_", "running_s_"):
            bookkeeping[prefix + key] = (shape, "fisher" in prefix or prefix == "omega_")
    for key, value in saved.items():
        value = torch.as_tensor(value)
        if not torch.isfinite(value).all():
            raise ValueError("Nonfinite checkpoint tensor: " + key)
        if key in expected:
            if value.shape != expected[key].shape:
                raise ValueError("Checkpoint tensor shape mismatch: " + key)
            filtered[key] = value
            continue
        if key in bookkeeping:
            shape, nonnegative = bookkeeping[key]
            if value.shape != shape or (nonnegative and (value < 0).any()):
                raise ValueError("Invalid CL bookkeeping buffer: " + key)
            continue
        future = re.fullmatch(r"heads\.(\d+)\.(weight|bias)", key)
        if future and task < int(future.group(1)) < spec["tasks"]:
            shape = (width, model.emb_dim) if future.group(2) == "weight" else (width,)
            if tuple(value.shape) != shape:
                raise ValueError("Unexpected unused future head: " + key)
            continue
        future_fisher = re.fullmatch(r"heads_(\d+)_(weight|bias)_fisher", key)
        if future_fisher and task < int(future_fisher.group(1)) < spec["tasks"]:
            shape = (width, model.emb_dim) if future_fisher.group(2) == "weight" else (width,)
            if tuple(value.shape) != shape or (value < 0).any():
                raise ValueError("Invalid unused future-head Fisher buffer: " + key)
            continue
        if key not in fishers or value.shape != fishers[key] or (value < 0).any():
            raise ValueError("Unexpected checkpoint tensor: " + key)
    missing = set(expected) - set(filtered)
    if missing - {"heads.{}.weight".format(task), "heads.{}.bias".format(task)}:
        raise ValueError("Checkpoint is missing backbone or historical-head tensors.")
    model.load_state_dict(filtered, strict=False)
    devices = [device.index or 0] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(int(head_seed))
        model.heads[-1].reset_parameters()
    return model.eval()


def contract(metadata):
    required = (
        "transfer_protocol",
        "task_index",
        "task_order",
        "class_ids",
        "checkpoint_sha256",
        "inversion_sha256",
        "head_seed",
        "label_mode",
        "feature_protocol",
        "head_mode",
        "model_unchanged",
        "input_sha256",
    )
    if any(k not in metadata for k in required):
        raise ValueError("Missing transfer extraction metadata.")
    if metadata["transfer_protocol"] != TRANSFER_PROTOCOL:
        raise ValueError("Unsupported transfer protocol.")
    task = task_index(metadata["task_index"], dataset_spec(metadata.get("dataset", "split_cifar100"))["tasks"])
    order = task_order(metadata.get("dataset", "split_cifar100"))
    if metadata.get("dataset", "split_cifar100") != "split_cifar100":
        if not re.fullmatch(r"[0-9a-f]{64}", str(metadata.get("dataset_sha256", ""))):
            raise ValueError("Missing prepared-dataset identity.")
    if (
        metadata["task_order"] != order
        or metadata["class_ids"] != order[task]
    ):
        raise ValueError("Incorrect task/class identity.")
    if (
        metadata["model_unchanged"] is not True
        or metadata.get("origin_role") != "paired_benchmark"
    ):
        raise ValueError("Need frozen paired benchmark extraction.")
    if metadata["label_mode"] not in ("ground_truth", "predicted"):
        raise ValueError("Unknown class-label protocol.")
    return copy.deepcopy(metadata)


def assert_transfer(source, target, columns):
    source, target = contract(source), contract(target)
    for key in ("dataset", "dataset_sha256"):
        if source.get(key) != target.get(key):
            raise ValueError("Source/target dataset identity differs: " + key)
    if target["task_index"] <= source["task_index"]:
        raise ValueError("Target must be a later task than the source.")
    if set(source["class_ids"]) & set(target["class_ids"]):
        raise ValueError("Source and target classes overlap.")
    for key in (
        "transfer_protocol",
        "task_order",
        "head_seed",
        "label_mode",
        "feature_protocol",
        "head_mode",
    ):
        if source[key] != target[key]:
            raise ValueError("Source/target extraction mismatch: " + key)
    if source["checkpoint_sha256"] == target["checkpoint_sha256"]:
        raise ValueError("Transfer requires distinct stage checkpoints.")
    if not set(columns).issubset(target["feature_columns"]):
        raise ValueError("Target is missing frozen source feature columns.")
    return {
        "source": source,
        "target": target,
        "allowed_changes": [
            "task_index",
            "class_ids",
            "checkpoint_sha256",
            "inversion_sha256",
        ],
        "target_used_for_fitting_or_calibration": False,
    }


def fingerprint(settings):
    from detector.io_utils import DETECTOR_ROOT
    from detector.common import sha256_file

    # Register upstream code too; adapters execute these files without editing them.
    paths = list(DETECTOR_ROOT.glob("*.py")) + list(DETECTOR_ROOT.parent.glob("*.py"))
    paths += list((DETECTOR_ROOT.parent / "approaches").glob("*.py"))
    paths.append(DETECTOR_ROOT / "plot-fontconfig.xml")
    payload = {
        "protocol": TRANSFER_PROTOCOL,
        "settings": settings,
        "sources": {
            str(p.relative_to(DETECTOR_ROOT.parent)): sha256_file(p)
            for p in sorted(paths)
        },
    }
    payload["fingerprint"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode()
    ).hexdigest()
    return payload
