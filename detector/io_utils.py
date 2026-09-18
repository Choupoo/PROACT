"""Scoped outputs, feature provenance, and immutable model artifacts.

Hashes detect accidental mismatch, not malicious artifacts. Only load pickle/joblib
files from trusted sources: deserialization can execute Python code.
"""

import hashlib
import json
import os
from pathlib import Path

import joblib
import pandas as pd

DETECTOR_ROOT = Path(__file__).resolve().parent
PROVENANCE_FIELDS = (
    "feature_protocol",
    "head_mode",
    "head_seed",
    "label_mode",
    "checkpoint_sha256",
    "inversion_sha256",
)


def ensure_output_path(path):
    """Resolve symlinks and reject every output outside this detector directory."""
    result = Path(path).expanduser().resolve()
    if result != DETECTOR_ROOT and DETECTOR_ROOT not in result.parents:
        raise ValueError(
            "All outputs must be inside {}: {}".format(DETECTOR_ROOT, result)
        )
    # Downstream libraries may create nested files themselves. Reject symlinked
    # children in an existing output tree before handing a directory to them.
    if result.is_dir():
        for current, directories, files in os.walk(result, followlinks=False):
            for name in directories + files:
                child = Path(current) / name
                if child.is_symlink():
                    raise ValueError(
                        "Output directories must not contain symlinks: {}".format(child)
                    )
    return result


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def feature_provenance(metadata):
    """The representation identity, excluding intended changes of data origin."""
    missing = [key for key in PROVENANCE_FIELDS if metadata.get(key) is None]
    if missing:
        raise ValueError("Feature metadata is missing provenance: {}".format(missing))
    result = {key: metadata[key] for key in PROVENANCE_FIELDS}
    for key in ("checkpoint_sha256", "inversion_sha256"):
        value = result[key]
        if not isinstance(value, str) or len(value) != 64:
            raise ValueError("{} must be a SHA-256 hex digest.".format(key))
        try:
            int(value, 16)
        except ValueError as error:
            raise ValueError("{} must be a SHA-256 hex digest.".format(key)) from error
    return result


def assert_compatible_provenance(expected, actual):
    expected = feature_provenance(expected)
    actual = feature_provenance(actual)
    mismatches = [key for key in PROVENANCE_FIELDS if expected[key] != actual[key]]
    if mismatches:
        raise ValueError("Incompatible feature provenance: {}".format(mismatches))


def read_feature_table(path):
    """Read a CSV only after checking its extraction sidecar and content hash."""
    path = Path(path)
    metadata_path = path.with_suffix(".metadata.json")
    with metadata_path.open(encoding="utf-8") as source:
        metadata = json.load(source)
    feature_provenance(metadata)
    if metadata.get("features_sha256") != _sha256(path):
        raise ValueError(
            "Feature CSV hash does not match its metadata: {}".format(path)
        )
    table = pd.read_csv(path)
    columns = metadata.get("feature_columns")
    if (
        not isinstance(columns, list)
        or not columns
        or len(columns) != len(set(columns))
    ):
        raise ValueError("Metadata must declare unique feature_columns.")
    missing = set(columns) - set(table.columns)
    if missing:
        raise ValueError("Declared features are missing: {}".format(sorted(missing)))
    return table, metadata


def save_frozen_bundle(bundle, output_dir, filename="bundle.joblib"):
    """Save once; require a fresh directory/name to replace a fitted detector."""
    output_dir = ensure_output_path(output_dir)
    path = ensure_output_path(output_dir / filename)
    sidecar = path.with_suffix(path.suffix + ".sha256.json")
    if path.exists() or sidecar.exists():
        raise FileExistsError(
            "Frozen artifact already exists; choose a new run: {}".format(path)
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path)
    with sidecar.open("x", encoding="utf-8") as target:
        json.dump({"sha256": _sha256(path), "filename": path.name}, target, indent=2)
        target.write("\n")
    return path


def load_frozen_bundle(path):
    """Verify accidental corruption before deserializing a TRUSTED model file."""
    path = Path(path)
    with path.with_suffix(path.suffix + ".sha256.json").open(
        encoding="utf-8"
    ) as source:
        metadata = json.load(source)
    if metadata.get("sha256") != _sha256(path):
        raise ValueError("Frozen bundle checksum mismatch: {}".format(path))
    return joblib.load(path)
