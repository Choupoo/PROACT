import pickle
import random
import sys
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import torch

import hashlib
import json

def save_json(data, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        json.dump(
            data,
            file,
            indent=2,
            sort_keys=True,
        )


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

def find_proact_root(start=None):
    if start is None:
        start = Path.cwd()

    start = Path(start).resolve()
    candidates = [start] + list(start.parents)

    for candidate in candidates:
        if (
            (candidate / "main_brainwash.py").is_file()
            and (candidate / "main_baselines.py").is_file()
            and (candidate / "data_utils.py").is_file()
            and (candidate / "utils.py").is_file()
        ):
            return candidate

    raise FileNotFoundError(
        "Could not find the PROACT repository root. "
        "Run the command from inside the PROACT repository."
    )


def add_proact_to_path(proact_root=None):
    root = find_proact_root(proact_root)
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    return root


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
        return pickle.load(file)

def normalize_task_order(task_order):
    normalized = []
    for task_classes in task_order:
        if torch.is_tensor(task_classes):
            values = (task_classes.detach().cpu().numpy().tolist())
        else:
            values = np.asarray(task_classes).tolist()
        normalized.append(
            [int(value) for value in values]
        )
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
    }

    missing = required - set(checkpoint.keys())
    if missing:
        raise KeyError("Victim checkpoint is missing keys: {}".format(sorted(missing)))

    if checkpoint["dataset"] != "split_cifar100":
        raise ValueError("This first experiment supports only split_cifar100.")

    if int(checkpoint["task_num"]) != 9:
        raise ValueError("Expected checkpoint['task_num'] == 9. In PROACT this reconstructs ten tasks and leaves task 9 as the incoming task.")

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
        raise ValueError("The pre-registered first experiment uses reckless BrainWash.")

    if int(artifact["attacked_task"]) != 9:
        raise ValueError("Expected attacked_task == 9.")

    permutation = torch.as_tensor(artifact["rnd_idx_train"]).long().cpu()

    noise = torch.as_tensor(artifact["latest_noise"]).float().cpu()

    if permutation.ndim != 1:
        raise ValueError("rnd_idx_train must be one-dimensional.")

    if len(permutation) != expected_size:
        raise ValueError(
            "Expected {} permutation entries, found {}.".format(expected_size, len(permutation)))

    expected = torch.arange(expected_size)
    if not torch.equal(torch.sort(permutation).values,expected):
        raise ValueError("rnd_idx_train is not a permutation of 0..{}.".format(expected_size - 1))

    expected_shape = (expected_size, 3, 32, 32)
    if tuple(noise.shape) != expected_shape:
        raise ValueError(
            "Expected latest_noise shape {}, found {}.".format(expected_shape, tuple(noise.shape)))

    delta = float(artifact["delta"])
    if noise.abs().max().item() > delta + 1e-5:
        raise ValueError("latest_noise exceeds the declared L-infinity budget.")

def compare_checkpoint_identity(victim_checkpoint, artifact_checkpoint):
    validate_victim_checkpoint(victim_checkpoint)
    validate_victim_checkpoint(artifact_checkpoint)

    metadata_keys = ["dataset", "task_num", "class_num", "seed", "model_type", "model_name"]

    for key in metadata_keys:
        if key in victim_checkpoint or key in artifact_checkpoint:
            if victim_checkpoint.get(key) != artifact_checkpoint.get(key):
                raise RuntimeError("Victim/artifact checkpoint mismatch for {!r}: {} != {}".format(key, victim_checkpoint.get(key), artifact_checkpoint.get(key)))

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
            raise RuntimeError("Victim and attack artifact differ at model tensor: {}".format(name))


def reconstruct_incoming_task(checkpoint):
    add_proact_to_path()
    from data_utils import get_dataset_specs

    validate_victim_checkpoint(checkpoint)

    config = {"dataset": checkpoint["dataset"], "task_num": int(checkpoint["task_num"]), "seed": int(checkpoint["seed"]),}

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
        raise RuntimeError("Reconstructed task order differs from checkpoint task_order.")

    task_index = int(checkpoint["task_num"])
    dataset = dataset_dict["train"][task_index]

    return dataset, {"task_index": task_index, "task_order": reconstructed_order, "image_size": int(image_size), "class_num": int(class_num), "embedding_factor": int(embedding_factor)}


def get_model_heads(model):
    if isinstance(model, torch.nn.DataParallel):
        return model.module.heads
    return model.heads


def initialize_defender_head(model, head_seed):
    heads = get_model_heads(model)
    if len(heads) != 10:
        raise RuntimeError("Expected ten task heads, found {}.".format(len(heads)))
    head = heads[-1]
    torch_state = torch.random.get_rng_state()
    cuda_state = (
        torch.cuda.get_rng_state_all()
        if torch.cuda.is_available()
        else None
    )
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
            raise RuntimeError("Model changed during feature extraction: {}".format(name))


def matching_inversion_files(folder, expected_count=9):
    folder = Path(folder)

    if not folder.is_dir():
        raise FileNotFoundError("Inversion folder not found: {}".format(folder))

    paths = sorted(
        path
        for path in folder.iterdir()
        if path.suffix == ".npz"
    )

    if len(paths) != expected_count:
        raise RuntimeError("Expected exactly {} inversion NPZ files, found {}: {}".format(expected_count, len(paths), [path.name for path in paths]))

    for path in paths:
        with np.load(path) as data:
            missing = {"x_dst", "y_dst"} - set(data.files)
            if missing:
                raise KeyError("{} is missing arrays: {}".format(path, sorted(missing)))
    return paths


def quantile_higher(values, q):
    values = np.asarray(values, dtype=np.float64)
    try:
        return float(np.quantile(values, q, method="higher"))
    except TypeError:
        return float(np.quantile(values, q, interpolation="higher"))