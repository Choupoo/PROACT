"""Registered full-data EWC extension: one independent worker per dataset.

Tasks 1/4/9, four target attacks, historical MAD rule and nine retrained feature
ablations. The default is one seed per dataset, not a reduced image subset.
"""

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path

from detector import advisor_final, advisor_study as study
from detector.common import save_json, sha256_file
from detector.doctor import diagnose
from detector.image_data import register, verify
from detector.io_utils import DETECTOR_ROOT, ensure_output_path
from detector.transfer_core import fingerprint

DATASETS = {"mini": "split_mini_imagenet", "tiny": "split_tiny_imagenet"}
DEFAULT_OUTPUT = DETECTOR_ROOT / "work/advisor_imagenet_v1"


def plan_settings(root, dataset, seeds, record):
    return dict(methods=["ewc"], seeds=list(seeds), source_task=1, history_tasks=[4], target_task=9,
                victim_epochs=100, inversion_iters=2000, attack_epochs=5000, delta=.3,
                cautious_weight=1., method_parameters={"ewc": list(study.METHODS["ewc"])},
                task_size=150, bags_per_rate=100, alpha=.05, label_mode="predicted",
                protocol="advisor_full_imagenet_v1", attacks=[list(a) for a in study.ATTACKS],
                threshold_rule="history_mad", ablations=True, unsupervised=False, effectiveness=True,
                scope="full_data_ewc_extension", dataset=dataset,
                dataset_sha256=record["dataset_sha256"], full_training_data=True,
                data_manifest=str(root / "data_manifest.json"),
                data_manifest_sha256=sha256_file(root / "data_manifest.json"),
                reuse_controls=True, shared_effectiveness=True)


def prepare(args):
    if not args.seeds or len(set(args.seeds)) != len(args.seeds) or min(args.seeds) < 0:
        raise ValueError("Seeds must be unique nonnegative integers.")
    for short, dataset in DATASETS.items():
        root = ensure_output_path(Path(args.output_dir) / short)
        source = args.mini_data if short == "mini" else args.tiny_data
        print("Validating complete prepared data:", short, flush=True)
        record = register(dataset, source)
        if (root / "plan.json").exists():
            _, payload = study.read_plan(root / "plan.json")
            saved = json.loads((root / "data_manifest.json").read_text())
            if record != saved or payload["settings"] != plan_settings(root, dataset, args.seeds, record):
                raise ValueError("Existing plan differs; use a new output directory.")
            print("Reusing verified plan:", root / "plan.json")
            continue
        if root.exists():
            raise FileExistsError("Unregistered output directory exists: " + str(root))
        root.mkdir(parents=True)
        save_json(record, root / "data_manifest.json")
        save_json(fingerprint(plan_settings(root, dataset, args.seeds, record)), root / "plan.json")
        print("Registered:", root / "plan.json", flush=True)


def run(args):
    root, payload = study.read_plan(Path(args.output_dir) / args.dataset / "plan.json")
    if payload["settings"].get("dataset") != DATASETS[args.dataset]:
        raise ValueError("Wrong dataset plan.")
    if not args.dry_run:
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        if not visible.strip() or len(visible.split(",")) != 1:
            raise ValueError("Assign one GPU per worker using CUDA_VISIBLE_DEVICES=0 (or 1).")
        environment = diagnose(require_cuda=True)
        if not environment["ready"]:
            print(json.dumps(environment, indent=2))
            raise RuntimeError("CUDA environment is not ready; no training started.")
        verify(json.loads((root / "data_manifest.json").read_text()))
    study.run(argparse.Namespace(plan=str(root / "plan.json"), method="ewc", seed=args.seed,
                                dry_run=args.dry_run, retry_failed=args.retry_failed))
    if not args.dry_run:
        summarize_one(root)


def stage_category(name):
    for category in ("attack", "extract", "victim", "inversion", "evaluate"):
        if name.endswith("_" + category):
            return category
    return "continuation_training" if "_train_" in name else "detector_fitting"


def timing(root, payload):
    groups = defaultdict(list)
    cells = []
    for seed in payload["settings"]["seeds"]:
        cell, steps = study.build_cell(root, payload["settings"], "ewc", seed)
        state_path = cell / "run_state.json"
        state = json.loads(state_path.read_text()) if state_path.exists() else {}
        done = sum(state.get(s["name"], {}).get("status") == "complete" for s in steps)
        active, failed = [], []
        for step in steps:
            name = step["name"]
            record = state.get(name, {})
            if record.get("status") == "running":
                active.append(name)
            if record.get("status") == "failed":
                failed.append(name)
            if record.get("status") == "complete" and "elapsed_seconds" in record:
                groups[stage_category(name)].append(float(record["elapsed_seconds"]))
        cells.append(dict(seed=seed, complete=done, total=len(steps), running=active, failed=failed))
    categories = {k: dict(stages=len(v), wall_hours=sum(v) / 3600, mean_seconds=sum(v) / len(v))
                  for k, v in groups.items()}
    return dict(dataset=payload["settings"]["dataset"], cells=cells, categories=categories,
                measured_completed_stage_wall_hours=sum(sum(v) for v in groups.values()) / 3600,
                note="Subprocess wall time, not measured GPU kernel time. Stage counts are not time percentages.")


def summarize_one(root):
    _, payload = study.read_plan(root / "plan.json")
    study.summarize(argparse.Namespace(plan=str(root / "plan.json")))
    profile = timing(root, payload)
    save_json(profile, root / "analysis/timing.json")
    path = root / "analysis/report.md"
    report = path.read_text()
    header = ("# Full-data EWC extension: {}\n\n"
              "All training images are retained. Each dataset has its own source-trained detector and historical calibration; "
              "this is not training on CIFAR-100 and transferring the detector across datasets.\n\n"
              "The numerical cutoffs are fitted on Tasks 1/4 of this dataset and frozen before Task 9. "
              "The historical MAD heuristic and feature definitions are unchanged. Rank/MMD are not rerun here. "
              "One seed does not quantify seed-to-seed variability.\n\n").format(payload["settings"]["dataset"])
    path.write_text(header + report, encoding="utf-8")
    return profile


def summarize(args):
    root = ensure_output_path(args.output_dir)
    profiles = {}
    for short in DATASETS:
        profiles[short] = summarize_one(root / short)
    save_json(profiles, root / "timing.json")


def status(args):
    for short in DATASETS:
        root, payload = study.read_plan(Path(args.output_dir) / short / "plan.json")
        print(json.dumps(timing(root, payload), indent=2))


def package(args):
    # Use the existing checked analysis exporter. Each dataset has its own
    # archive/manifest and clearly retains incomplete status when permitted.
    for short in DATASETS:
        root = ensure_output_path(Path(args.output_dir) / short)
        archive = root.parent / (short + "_analysis_" + datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f") + ".tar.gz")
        advisor_final.package(argparse.Namespace(output_dir=str(root), archive=str(archive),
                                                 allow_incomplete=args.allow_incomplete),
                              summarizer=lambda unused: summarize_one(root))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="action", required=True)
    for action in ("plan", "run", "status", "summarize", "package"):
        p = subs.add_parser(action)
        p.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
        if action == "plan":
            p.add_argument("--mini-data", required=True, help="Directory containing train_x/y.npy and test_x/y.npy.")
            p.add_argument("--tiny-data", required=True, help="PROACT-format tiny_imagenet.npz.")
            p.add_argument("--seeds", type=int, nargs="+", default=[6])
        elif action == "run":
            p.add_argument("--dataset", choices=list(DATASETS), required=True)
            p.add_argument("--seed", type=int)
            p.add_argument("--dry-run", action="store_true")
            p.add_argument("--retry-failed", action="store_true")
        elif action == "package":
            p.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()
    {"plan": prepare, "run": run, "status": status, "summarize": summarize, "package": package}[args.action](args)


if __name__ == "__main__":
    main()
