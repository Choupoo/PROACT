"""Explicit fixed-ten-task contracts for the meeting-3 transfer experiment.

The legacy Task-9 protocol stays unchanged. Task indices here are zero based:
source 1 is the second task; target 9 is the tenth task.
"""

import copy
import hashlib
import json
import re

import numpy as np
import torch

from detector.common import normalize_task_order, validate_victim_checkpoint

TRANSFER_PROTOCOL = "source_supervised_task_transfer_v1"
TASK_ORDER = [list(range(10 * t, 10 * (t + 1))) for t in range(10)]


def task_index(value):
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, np.integer))
        or not 1 <= value <= 9
    ):
        raise ValueError("Incoming task must be a zero-based integer in 1..9.")
    return int(value)


def validate_checkpoint(checkpoint, expected_task=None):
    """Validate real metadata; never relabel an early checkpoint as Task 9."""
    task = task_index(checkpoint.get("task_num"))
    if expected_task is not None and task != expected_task:
        raise ValueError("Checkpoint stage differs from requested incoming task.")
    # Reuse legacy structural checks on a copy, then apply the transfer contract.
    structure = dict(checkpoint, task_num=9)
    validate_victim_checkpoint(structure)
    if normalize_task_order(checkpoint["task_order"]) != TASK_ORDER:
        raise ValueError("Transfer requires the fixed ten-task CIFAR-100 class order.")
    return task


def validate_artifact(artifact, checkpoint, expected_size=5000):
    from detector.common import compare_checkpoint_identity, validate_attack_artifact

    task = validate_checkpoint(checkpoint)
    validate_checkpoint(artifact["pretrained_ckpt"], task)
    if artifact.get("attacked_task") != task:
        raise ValueError("Attack targets a different incoming task.")
    # Existing noise/permutation and tensor identity checks are stage-independent.
    left = dict(checkpoint, task_num=9)
    right = dict(artifact["pretrained_ckpt"], task_num=9)
    compare_checkpoint_identity(left, right)
    validate_attack_artifact(
        dict(artifact, pretrained_ckpt=right, attacked_task=9), expected_size
    )
    if artifact.get("real", False):
        raise ValueError("This registered protocol uses inversion-based BrainWash.")


def fixed_dataset_specs(**checkpoint):
    """Adapter for upstream get_dataset_specs: preserve 10 classes per task."""
    from data_utils import generate_split_cifar100_tasks

    task = validate_checkpoint(checkpoint)
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
    model = ResNet18(task + 1, 10, nf=32).to(device)
    expected = model.state_dict()
    saved = checkpoint["model"]
    filtered = {}
    fishers = {
        n.replace(".", "_") + "_fisher": p.shape for n, p in model.named_parameters()
    }
    for key, value in saved.items():
        value = torch.as_tensor(value)
        if not torch.isfinite(value).all():
            raise ValueError("Nonfinite checkpoint tensor: " + key)
        if key in expected:
            if value.shape != expected[key].shape:
                raise ValueError("Checkpoint tensor shape mismatch: " + key)
            filtered[key] = value
            continue
        future = re.fullmatch(r"heads\.(\d+)\.(weight|bias)", key)
        if future and task < int(future.group(1)) < 10:
            shape = (10, model.emb_dim) if future.group(2) == "weight" else (10,)
            if tuple(value.shape) != shape:
                raise ValueError("Unexpected unused future head: " + key)
            continue
        future_fisher = re.fullmatch(r"heads_(\d+)_(weight|bias)_fisher", key)
        if future_fisher and task < int(future_fisher.group(1)) < 10:
            shape = (10, model.emb_dim) if future_fisher.group(2) == "weight" else (10,)
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
    task = task_index(metadata["task_index"])
    if (
        metadata["task_order"] != TASK_ORDER
        or metadata["class_ids"] != TASK_ORDER[task]
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
