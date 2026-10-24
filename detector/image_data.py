"""Full, prepared ImageNet benchmarks for detector-only upstream adapters.

No downloads, resizing, class subsampling or modifications of source datasets.
The accepted files are the arrays consumed by PROACT's existing loaders.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

SPECS = {
    "split_cifar100": dict(classes=100, classes_per_task=10, image_size=32,
                           train_per_class=500, test_per_class=100),
    "split_mini_imagenet": dict(classes=100, classes_per_task=10, image_size=84,
                               train_per_class=500, test_per_class=100),
    "split_tiny_imagenet": dict(classes=200, classes_per_task=20, image_size=64,
                               train_per_class=500, test_per_class=50),
}


def dataset_spec(name):
    if name not in SPECS:
        raise ValueError("Unsupported dataset: " + str(name))
    return dict(SPECS[name])


def task_order(name):
    width = dataset_spec(name)["classes_per_task"]
    return [list(range(t * width, (t + 1) * width)) for t in range(10)]


def array_sources(name, source):
    source = Path(source).expanduser().resolve()
    if name == "split_mini_imagenet":
        return {key: str(source / filename) for key, filename in (
            ("train_data", "train_x.npy"), ("train_lbls", "train_y.npy"),
            ("tst_data", "test_x.npy"), ("tst_lbls", "test_y.npy"))}
    if name == "split_tiny_imagenet":
        return {"archive": str(source)}
    raise ValueError("This entry point accepts miniImageNet or tinyImageNet only.")


def read_arrays(name, files):
    if name == "split_tiny_imagenet":
        with np.load(files["archive"], allow_pickle=False) as archive:
            return {k: archive[k] for k in ("train_data", "train_lbls", "tst_data", "tst_lbls")}
    return {k: np.load(v, mmap_mode="r", allow_pickle=False) for k, v in files.items()}


def validate_arrays(name, arrays):
    """Fail before GPU work if arrays are partial, reduced or wrongly encoded."""
    spec = dataset_spec(name)
    counts = {}
    for split, image_key, label_key, expected in (
        ("train", "train_data", "train_lbls", spec["train_per_class"]),
        ("test", "tst_data", "tst_lbls", spec["test_per_class"]),
    ):
        images, labels = arrays[image_key], arrays[label_key]
        size = spec["image_size"]
        shape = (size, size, 3) if name == "split_mini_imagenet" else (3, size, size)
        if images.ndim != 4 or tuple(images.shape[1:]) != shape:
            raise ValueError("{} {}: expected N x {}.".format(name, split, shape))
        if labels.ndim != 1 or len(labels) != len(images) or not np.issubdtype(labels.dtype, np.integer):
            raise ValueError("Labels must be aligned one-dimensional integer arrays.")
        if len(labels) != spec["classes"] * expected or np.any(labels < 0) or np.any(labels >= spec["classes"]):
            raise ValueError("Incomplete full-data split or invalid global class IDs.")
        observed = np.bincount(labels, minlength=spec["classes"])
        if not np.all(observed == expected):
            raise ValueError("Every {} class must contain {} images; subsets are not accepted.".format(split, expected))
        if name == "split_mini_imagenet":
            if images.dtype != np.uint8:
                raise ValueError("miniImageNet images must be uint8 NHWC, as in PROACT.")
        elif not np.issubdtype(images.dtype, np.floating):
            raise ValueError("tinyImageNet NPZ images must be floating-point NCHW in [0,1].")
        # Bounded-size validation avoids a second full-dataset temporary array.
        for start in range(0, len(images), 256):
            block = images[start:start + 256]
            if not np.isfinite(block).all() or (name != "split_mini_imagenet" and
                                               (np.any(block < 0) or np.any(block > 1))):
                raise ValueError("Invalid image values in " + split)
        counts[split] = observed.tolist()
    return counts


def register(name, source):
    from detector.common import sha256_file
    files = array_sources(name, source)
    for path in files.values():
        if not Path(path).is_file():
            raise FileNotFoundError("Prepared benchmark file not found: " + path)
    hashes = {k: sha256_file(v) for k, v in files.items()}
    counts = validate_arrays(name, read_arrays(name, files))
    # Paths may be relocated, but the bytes and the declared split may not change.
    identity = dict(dataset=name, hashes=hashes, spec=dataset_spec(name), counts=counts)
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    return dict(identity, files=files, dataset_sha256=digest, full_training_data=True)


def verify(record):
    from detector.common import sha256_file
    if record.get("full_training_data") is not True or record["spec"] != dataset_spec(record["dataset"]):
        raise ValueError("Invalid full-data registration.")
    if {k: sha256_file(v) for k, v in record["files"].items()} != record["hashes"]:
        raise ValueError("Registered dataset changed; do not mix versions.")
    identity = {k: record[k] for k in ("dataset", "hashes", "spec", "counts")}
    if hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest() != record["dataset_sha256"]:
        raise ValueError("Invalid dataset identity hash.")


class TensorImages(Dataset):
    def __init__(self, data, targets):
        self.data, self.targets = data, targets

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        return self.data[index], self.targets[index]


class PreparedData:
    """One load per subprocess; fresh wrappers prevent repeated shuffling.

    The cached tensors are immutable inputs. Upstream assigns reordered/clamped
    tensors to wrapper attributes; it must never update the cached tensors in place.
    """
    def __init__(self, record):
        verify(record)
        self.record = record
        self.tensors = None

    def _load(self):
        if self.tensors is not None:
            return
        name = self.record["dataset"]
        arrays = read_arrays(name, self.record["files"])
        validate_arrays(name, arrays)
        width = dataset_spec(name)["classes_per_task"]
        tensors = {"train": [], "test": []}
        for split, xkey, ykey in (("train", "train_data", "train_lbls"), ("test", "tst_data", "tst_lbls")):
            labels = arrays[ykey]
            for task in range(10):
                mask = (labels >= task * width) & (labels < (task + 1) * width)
                images = torch.from_numpy(np.array(arrays[xkey][mask], copy=True))
                if name == "split_mini_imagenet":
                    images = images.permute(0, 3, 1, 2).float().div_(255)
                else:
                    images = images.float()
                local = torch.from_numpy(np.array(labels[mask] - task * width, dtype=np.int64))
                tensors[split].append((images, local))
        self.tensors = tensors

    def specs(self, **checkpoint):
        from detector.transfer_core import validate_checkpoint
        task = validate_checkpoint(checkpoint)
        if checkpoint["dataset"] != self.record["dataset"]:
            raise ValueError("Checkpoint and prepared dataset differ.")
        # Match upstream loaders' RNG reset on EVERY call, including cache hits.
        np.random.seed(checkpoint["seed"])
        torch.manual_seed(checkpoint["seed"])
        self._load()
        data = {k: [TensorImages(x, y) for x, y in pairs[:task + 1]]
                for k, pairs in self.tensors.items()}
        spec = self.record["spec"]
        return data, task_order(checkpoint["dataset"]), spec["image_size"], spec["classes_per_task"], 1

    def training(self, task, continuation, *args, **kwargs):
        if kwargs.get("task_num") != 10 or kwargs.get("rnd_order") is not False:
            raise ValueError("Require the fixed ten-task class partition.")
        name = self.record["dataset"]
        if not np.array_equal(kwargs.get("order"), np.arange(self.record["spec"]["classes"])):
            raise ValueError("Unexpected class order.")
        # transfer_upstream supplies the registered experiment seed explicitly.
        np.random.seed(kwargs.get("seed", 0))
        torch.manual_seed(kwargs.get("seed", 0))
        self._load()
        data = {}
        width = self.record["spec"]["classes_per_task"]
        for t in range(10):
            data[t] = dict(name=name, ncla=width)
            for split in ("train", "test"):
                x, y = self.tensors[split][t]
                data[t][split] = dict(x=x, y=y)
        data["ncla"] = self.record["spec"]["classes"]
        stop = task + 1 if continuation else 10
        size = self.record["spec"]["image_size"]
        return data, [(t, width) for t in range(stop)], [3, size, size], task_order(name)
