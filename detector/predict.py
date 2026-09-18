"""One-command incoming-dataset feature extraction and frozen decision."""

import argparse
from pathlib import Path
import sys

from detector.common import sha256_file
from detector.io_utils import ensure_output_path, load_frozen_bundle
from detector.pipeline import child_environment, execute
from detector.unsupervised import METHOD


def main(args):
    bundle_path = Path(args.bundle).resolve()
    checkpoint = Path(args.checkpoint).resolve()
    inversions = Path(args.inversion_dir).resolve()
    incoming = Path(args.input_npz).resolve()
    bundle = load_frozen_bundle(bundle_path)
    is_unsupervised = bundle.get("method") == METHOD
    if not is_unsupervised and bundle.get("kind") != "supervised_dataset_detector_v1":
        raise ValueError(
            "Use a dataset_bundle.joblib or unsupervised_bundle.joblib, not a sample detector."
        )
    provenance = bundle["provenance"]
    if sha256_file(checkpoint) != provenance["checkpoint_sha256"]:
        raise ValueError(
            "The checkpoint differs from the frozen detector's checkpoint."
        )
    output = ensure_output_path(args.output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Choose an empty prediction directory: {}".format(output))
    output.mkdir(parents=True, exist_ok=True)
    features = output / "incoming_features.csv"
    command = [
        sys.executable,
        "-B",
        "-u",
        "-m",
        "detector.extract_features",
        "--checkpoint",
        str(checkpoint),
        "--inversion-dir",
        str(inversions),
        "--input-npz",
        str(incoming),
        "--output",
        str(features),
        "--device",
        args.device,
        "--head-seed",
        str(provenance["head_seed"]),
        "--label-mode",
        provenance["label_mode"],
    ]
    env = child_environment(output)
    execute(command, cwd=output, log_path=output / "extraction.log", env=env)
    module = "detector.unsupervised" if is_unsupervised else "detector.dataset_detector"
    command = [
        sys.executable,
        "-B",
        "-u",
        "-m",
        module,
        "predict",
        "--bundle",
        str(bundle_path),
        "--features",
        str(features),
    ]
    command += (
        ["--output", str(output / "prediction.json")]
        if is_unsupervised
        else ["--output-dir", str(output)]
    )
    execute(command, cwd=output, log_path=output / "prediction.log", env=env)
    print("Saved incoming-dataset decision:", output / "prediction.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--inversion-dir", required=True)
    parser.add_argument("--input-npz", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    main(parser.parse_args())
