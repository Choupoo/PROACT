"""Extract historical descriptors and freeze Rank/MMD without incoming data."""

import argparse
import hashlib
import json

import pandas as pd
import torch

from detector import FEATURE_PROTOCOL, HEAD_MODE
from detector import rank_reference, unsupervised
from detector.common import (load_pickle, matching_inversion_files, save_json,
                             sha256_file, snapshot_state_dict, assert_model_unchanged,
                             set_seed)
from detector.extract_features import backbone_parameters, build_past_references, extract_one, reference_samples
from detector.io_utils import save_frozen_bundle
from detector.transfer_core import load_defender, validate_checkpoint
from detector.transfer_detector import fresh_output


def run(args):
    output = fresh_output(args.output_dir)
    checkpoint = load_pickle(args.checkpoint)
    task = validate_checkpoint(checkpoint)
    device = torch.device(args.device)
    set_seed(20260721)
    model = load_defender(checkpoint, device, args.head_seed)
    before = snapshot_state_dict(model)
    files = matching_inversion_files(args.inversion_dir, expected_count=task)
    past, directions, records = build_past_references(model, files, device, 32)
    hashes = [{"task_id": r["task_id"], "sha256": sha256_file(r["inversion_file"])} for r in records]
    inv_hash = hashlib.sha256(json.dumps(hashes, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    named, parameters = backbone_parameters(model)
    past, directions = past.to(device), directions.to(device)
    rows = []
    for index, (image, _, identity) in enumerate(reference_samples(files), 1):
        raw = extract_one(model, named, parameters, past, image, None, device,
                          task_directions=directions, label_mode="predicted")
        rows.append(dict(identity, **{c: raw[c] for c in unsupervised.FEATURE_COLUMNS}))
        if index % 100 == 0:
            print("Historical descriptors:", index, flush=True)
    assert_model_unchanged(model, before)
    output.mkdir(parents=True)
    table = pd.DataFrame(rows)
    path = output / "reference_features.csv"
    table.to_csv(path, index=False)
    metadata = dict(feature_protocol=FEATURE_PROTOCOL, head_mode=HEAD_MODE,
                    head_seed=args.head_seed, label_mode="predicted",
                    checkpoint_sha256=sha256_file(args.checkpoint), inversion_sha256=inv_hash,
                    origin_role="historical_inversion", model_unchanged=True,
                    feature_columns=list(unsupervised.FEATURE_COLUMNS),
                    features_sha256=sha256_file(path), task_index=task,
                    reference_tasks=hashes, row_count=len(table))
    save_json(metadata, path.with_suffix(".metadata.json"))
    rank = rank_reference.fit_reference(table, metadata, alpha=args.alpha,
                                        bootstrap_draws=args.bootstrap_draws,
                                        max_incoming=args.task_size)
    mmd = unsupervised.fit_reference(table, metadata, alpha=args.alpha,
                                     max_incoming=args.task_size)
    save_frozen_bundle(rank, output / "rank")
    save_frozen_bundle(mmd, output / "mmd")
    save_json({"incoming_data_used": False, "true_class_labels_used": False,
               "interpretation": "Distribution-shift tests, not poisoning probabilities.",
               "reference_assumption": "Historical checkpoints are clean; inversion fidelity is not guaranteed."},
              output / "reference_audit.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("checkpoint", "inversion-dir", "output-dir"):
        parser.add_argument("--" + flag, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--head-seed", type=int, default=20260720)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--task-size", type=int, default=150)
    parser.add_argument("--bootstrap-draws", type=int, default=499)
    run(parser.parse_args())
