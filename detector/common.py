"""Artifact validation and shared utilities for the Task-9 detector."""

import hashlib
import inspect
import io
import json
import math
import pickle
import random
import re
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import torch


def save_json(data, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        json.dump(
            data,
            file,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        file.write("\n")


def sha256_file(path):
    path = Path(path)
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(
            lambda: file.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def set_seed(seed):
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_pickle(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError("File not found: {}".format(path))
    with path.open("rb") as file:
        return _CPUUnpickler(file).load()


class _CPUUnpickler(pickle.Unpickler):
    """Read TRUSTED upstream CUDA pickles on CPU before choosing a device.

    This is device remapping, not a safe-unpickling sandbox. A pickle can still
    execute arbitrary code; only use files from a trusted experiment.
    """

    def find_class(self, module, name):
        if module == "torch.storage" and name == "_load_from_bytes":
            load_options = {"map_location": "cpu"}
            # Older torch forwards unknown kwargs to pickle.load, which rejects
            # weights_only. Inspect support rather than retrying deserialization
            # after a TypeError that could indicate an unrelated corrupt file.
            if "weights_only" in inspect.signature(torch.load).parameters:
                load_options["weights_only"] = False
            return lambda value: torch.load(io.BytesIO(value), **load_options)
        return super().find_class(module, name)


def normalize_task_order(task_order):
    normalized = []
    for task_classes in task_order:
        if torch.is_tensor(task_classes):
            values = task_classes.detach().cpu().numpy().tolist()
        else:
            values = np.asarray(task_classes).tolist()
        normalized.append([int(value) for value in values])
    return normalized


def validate_victim_checkpoint(checkpoint):
    if not isinstance(checkpoint, Mapping):
        raise TypeError("Victim checkpoint must be a dictionary-like mapping.")

    required = {
        "dataset",
        "task_num",
        "seed",
        "task_order",
        "model",
        "class_num",
        "model_type",
    }

    missing = required - set(checkpoint.keys())
    if missing:
        raise KeyError("Victim checkpoint is missing keys: {}".format(sorted(missing)))

    if checkpoint["dataset"] != "split_cifar100":
        raise ValueError("This first experiment supports only split_cifar100.")

    if checkpoint["model_type"] != "resnet":
        raise ValueError("Stage-gradient features require the PROACT ResNet backbone.")

    if int(checkpoint["task_num"]) != 9:
        raise ValueError(
            "Expected checkpoint['task_num'] == 9. In PROACT this reconstructs ten tasks and leaves task 9 as the incoming task."
        )

    if int(checkpoint["class_num"]) != 10:
        raise ValueError("Expected ten task-local classes for Split CIFAR-100.")

    if not isinstance(checkpoint["model"], Mapping):
        raise TypeError("checkpoint['model'] must be a PyTorch state_dict mapping.")

    if not checkpoint["model"]:
        raise ValueError("checkpoint['model'] is empty.")


def validate_attack_artifact(artifact, expected_size=5000):
    if not isinstance(artifact, Mapping):
        raise TypeError("BrainWash artifact must be a dictionary-like mapping.")

    required = {
        "pretrained_ckpt",
        "rnd_idx_train",
        "latest_noise",
        "delta",
        "seed",
        "mode",
        "attacked_task",
    }

    missing = required - set(artifact.keys())
    if missing:
        raise KeyError("BrainWash artifact is missing keys: {}".format(sorted(missing)))

    validate_victim_checkpoint(artifact["pretrained_ckpt"])

    if artifact["mode"] != "reckless":
        raise ValueError("This detector experiment uses reckless BrainWash.")

    if artifact.get("reverse", False):
        raise ValueError("A reverse (defense) artifact is not a BrainWash attack.")

    if int(artifact["attacked_task"]) != 9:
        raise ValueError("Expected attacked_task == 9.")

    raw_permutation = torch.as_tensor(artifact["rnd_idx_train"]).cpu()
    permutation = raw_permutation.long()
    if not torch.isfinite(raw_permutation).all() or not torch.equal(
        # Old torch.equal requires matching dtypes. Round-trip the integer
        # conversion so fractional values still fail validation.
        raw_permutation,
        permutation.to(dtype=raw_permutation.dtype),
    ):
        raise ValueError("rnd_idx_train must contain finite integer indices.")

    noise = torch.as_tensor(artifact["latest_noise"]).float().cpu()

    if permutation.ndim != 1:
        raise ValueError("rnd_idx_train must be one-dimensional.")

    if len(permutation) != expected_size:
        raise ValueError(
            "Expected {} permutation entries, found {}.".format(
                expected_size, len(permutation)
            )
        )

    expected = torch.arange(expected_size)
    if not torch.equal(torch.sort(permutation).values, expected):
        raise ValueError(
            "rnd_idx_train is not a permutation of 0..{}.".format(expected_size - 1)
        )

    expected_shape = (expected_size, 3, 32, 32)
    if tuple(noise.shape) != expected_shape:
        raise ValueError(
            "Expected latest_noise shape {}, found {}.".format(
                expected_shape, tuple(noise.shape)
            )
        )

    delta = float(artifact["delta"])
    if not math.isfinite(delta) or delta <= 0:
        raise ValueError("The L-infinity budget must be finite and positive.")
    if not torch.isfinite(noise).all():
        raise ValueError("latest_noise contains non-finite values.")
    if noise.abs().max().item() > delta + 1e-5:
        raise ValueError("latest_noise exceeds the declared L-infinity budget.")


def compare_checkpoint_identity(victim_checkpoint, artifact_checkpoint):
    validate_victim_checkpoint(victim_checkpoint)
    validate_victim_checkpoint(artifact_checkpoint)

    metadata_keys = [
        "dataset",
        "task_num",
        "class_num",
        "seed",
        "model_type",
        "model_name",
    ]

    for key in metadata_keys:
        if key in victim_checkpoint or key in artifact_checkpoint:
            if victim_checkpoint.get(key) != artifact_checkpoint.get(key):
                raise RuntimeError(
                    "Victim/artifact checkpoint mismatch for {!r}: {} != {}".format(
                        key, victim_checkpoint.get(key), artifact_checkpoint.get(key)
                    )
                )

    victim_order = normalize_task_order(victim_checkpoint["task_order"])
    artifact_order = normalize_task_order(artifact_checkpoint["task_order"])

    if victim_order != artifact_order:
        raise RuntimeError("Victim and attack artifact use different task orders.")

    victim_state = victim_checkpoint["model"]
    artifact_state = artifact_checkpoint["model"]

    if set(victim_state.keys()) != set(artifact_state.keys()):
        raise RuntimeError("Victim and attack artifact contain different model keys.")

    for name in victim_state:
        left = torch.as_tensor(victim_state[name]).detach().cpu()
        right = torch.as_tensor(artifact_state[name]).detach().cpu()
        if not torch.equal(left, right):
            raise RuntimeError(
                "Victim and attack artifact differ at model tensor: {}".format(name)
            )


def reconstruct_incoming_task(checkpoint):
    # Run the CLI as `python -m detector.<module>` from the PROACT root.
    from data_utils import get_dataset_specs

    validate_victim_checkpoint(checkpoint)

    config = {
        "dataset": checkpoint["dataset"],
        "task_num": int(checkpoint["task_num"]),
        "seed": int(checkpoint["seed"]),
    }

    (
        dataset_dict,
        reconstructed_order,
        image_size,
        class_num,
        embedding_factor,
    ) = get_dataset_specs(**config)

    if len(dataset_dict["train"]) != 10:
        raise RuntimeError("Expected ten reconstructed training tasks.")

    saved_order = normalize_task_order(checkpoint["task_order"])
    reconstructed_order = normalize_task_order(reconstructed_order)

    if saved_order != reconstructed_order:
        raise RuntimeError(
            "Reconstructed task order differs from checkpoint task_order."
        )

    task_index = int(checkpoint["task_num"])
    dataset = dataset_dict["train"][task_index]

    return dataset, {
        "task_index": task_index,
        "task_order": reconstructed_order,
        "image_size": int(image_size),
        "class_num": int(class_num),
        "embedding_factor": int(embedding_factor),
    }


def get_model_heads(model):
    if isinstance(model, torch.nn.DataParallel):
        return model.module.heads
    return model.heads


def initialize_defender_head(model, head_seed):
    """Reset only the incoming head while preserving all global RNG states."""
    heads = get_model_heads(model)
    if len(heads) != 10:
        raise RuntimeError("Expected ten task heads, found {}.".format(len(heads)))
    head = heads[-1]
    torch_state = torch.random.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    numpy_state = np.random.get_state()
    python_state = random.getstate()
    try:
        set_seed(head_seed)
        if not hasattr(head, "reset_parameters"):
            raise RuntimeError("The incoming-task head has no reset_parameters method.")
        head.reset_parameters()
    finally:
        torch.random.set_rng_state(torch_state)
        if torch.cuda.is_available() and cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)
        np.random.set_state(numpy_state)
        random.setstate(python_state)
    return len(heads) - 1


def snapshot_state_dict(model):
    return {
        name: tensor.detach().cpu().clone()
        for name, tensor in model.state_dict().items()
    }


def assert_model_unchanged(model, before):
    after = model.state_dict()

    if set(before.keys()) != set(after.keys()):
        raise RuntimeError("Model state-dict keys changed.")

    for name, old_value in before.items():
        current = after[name].detach().cpu()

        if not torch.equal(old_value, current):
            raise RuntimeError(
                "Model changed during feature extraction: {}".format(name)
            )


def inversion_task_id(path, data):
    """Resolve the saved task id, falling back to PROACT's `_tid_XX` filename."""
    match = re.search(r"_tid_(\d+)$", Path(path).stem)
    filename_id = int(match.group(1)) if match else None
    if "tid" in data:
        value = np.asarray(data["tid"])
        if value.size != 1 or not np.isfinite(value).all():
            raise ValueError("Invalid inversion task id in {}.".format(path))
        scalar = value.item()
        task_id = int(scalar)
        if scalar != task_id:
            raise ValueError("Inversion task id must be an integer: {}.".format(path))
        if filename_id is not None and filename_id != task_id:
            raise ValueError(
                "Inversion task id disagrees with filename: {}.".format(path)
            )
        return task_id
    if filename_id is None:
        raise ValueError(
            "Missing inversion task id (tid or _tid_XX filename): {}.".format(path)
        )
    return filename_id


def matching_inversion_files(folder, expected_count=9):
    """Return one validated inversion file per task, ordered by actual task id."""
    folder = Path(folder)

    if not folder.is_dir():
        raise FileNotFoundError("Inversion folder not found: {}".format(folder))

    paths = sorted(path for path in folder.iterdir() if path.suffix == ".npz")

    if len(paths) != expected_count:
        raise RuntimeError(
            "Expected exactly {} inversion NPZ files, found {}: {}".format(
                expected_count, len(paths), [path.name for path in paths]
            )
        )

    by_task = {}
    for path in paths:
        with np.load(path) as data:
            missing = {"x_dst", "y_dst"} - set(data.files)
            if missing:
                raise KeyError("{} is missing arrays: {}".format(path, sorted(missing)))
            task_id = inversion_task_id(path, data)
            images, labels = data["x_dst"], data["y_dst"]
            if images.ndim != 4 or tuple(images.shape[1:]) != (3, 32, 32):
                raise ValueError(
                    "Expected N x 3 x 32 x 32 inversion images: {}.".format(path)
                )
            if labels.ndim != 1 or not len(labels) or len(images) != len(labels):
                raise ValueError(
                    "Inversion images and labels must be nonempty and aligned: {}.".format(
                        path
                    )
                )
            if not np.isfinite(images).all() or not np.isfinite(labels).all():
                raise ValueError("Non-finite inversion data: {}.".format(path))
            if (
                not np.equal(labels, labels.astype(np.int64)).all()
                or not ((labels >= 0) & (labels < 10)).all()
            ):
                raise ValueError(
                    "Inversion labels must be task-local integers 0..9: {}.".format(
                        path
                    )
                )
            if task_id in by_task:
                raise ValueError("Duplicate inversion task id: {}.".format(task_id))
            by_task[task_id] = path
    if set(by_task) != set(range(expected_count)):
        raise ValueError(
            "Expected inversion tasks 0..{}, found {}.".format(
                expected_count - 1, sorted(by_task)
            )
        )
    return [by_task[task_id] for task_id in range(expected_count)]
