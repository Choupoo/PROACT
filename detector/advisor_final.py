"""Approved CIFAR-100 expansion: a fixed rule, multiple CL methods and ablations.

This prepares/runs real experiments; it never manufactures missing GPU results.
Existing pilot runs are left untouched. All writes are scoped to detector/.
"""

import argparse
from datetime import datetime, timezone
import io
import json
from pathlib import Path
import tarfile

from detector import advisor_study as study
from detector.common import sha256_file
from detector.doctor import diagnose
from detector.io_utils import DETECTOR_ROOT, ensure_output_path

DEFAULT_OUTPUT = DETECTOR_ROOT / "work/advisor_final_v1"
DEFAULT_SEEDS = [6, 7, 8]
FIXED_RULE = "history_mad"


def plan_options(args):
    return argparse.Namespace(
        output_dir=args.output_dir, methods=args.methods, seeds=args.seeds,
        expanded_settings=True, ablations=True, threshold_rule=FIXED_RULE,
        source_task=1, history_tasks=[4], target_task=9, victim_epochs=100,
        inversion_iters=2000, attack_epochs=5000, task_size=150,
        bags_per_rate=100, alpha=0.05,
    )


def prepare(args):
    root = ensure_output_path(args.output_dir)
    options = plan_options(args)
    if (root / "plan.json").exists():
        _, payload = study.read_plan(root / "plan.json")
        settings = payload["settings"]
        expected = {
            "methods": options.methods, "seeds": options.seeds,
            "scope": "expanded", "threshold_rule": FIXED_RULE,
            "ablations": True, "unsupervised": True, "effectiveness": True,
            "source_task": 1, "history_tasks": [4], "target_task": 9,
            "victim_epochs": 100, "inversion_iters": 2000, "attack_epochs": 5000,
            "task_size": 150, "bags_per_rate": 100, "alpha": 0.05,
            "label_mode": "predicted", "attacks": [list(a) for a in study.ATTACKS],
            "method_parameters": {m: list(study.METHODS[m]) for m in options.methods},
            "cautious_weight": 1.0,
        }
        mismatches = [key for key, value in expected.items() if settings.get(key) != value]
        if mismatches:
            raise ValueError("Existing plan differs ({}); use a new output directory.".format(
                ", ".join(mismatches)))
        print("Reusing verified plan:", root / "plan.json")
        return
    study.plan(options)


def run(args):
    if not args.dry_run:
        environment = diagnose(require_cuda=True)
        if not environment["ready"]:
            print(json.dumps(environment, indent=2))
            raise RuntimeError("GPU environment is not ready. No training was started.")
    study.run(argparse.Namespace(plan=str(Path(args.output_dir) / "plan.json"),
                                method=args.method, seed=args.seed,
                                dry_run=args.dry_run, retry_failed=args.retry_failed))


def summarize(args):
    study.summarize(argparse.Namespace(plan=str(Path(args.output_dir) / "plan.json")))


def analysis_files(root, settings):
    """Export analysis evidence, not datasets/checkpoints/deserializable models."""
    files = {root / "plan.json"}
    files.update(path for path in (root / "analysis").rglob("*") if path.is_file()
                 and path.suffix in (".json", ".csv", ".md"))
    for method in settings["methods"]:
        for seed in settings["seeds"]:
            cell = root / method / ("seed" + str(seed))
            for name in ("run_state.json", "run_environment.json"):
                path = cell / name
                if path.is_file():
                    files.add(path)
            for folder in (cell / "logs", cell / "frozen_supervised"):
                files.update(path for path in folder.rglob("*") if path.is_file()
                             and path.suffix in (".json", ".log", ".csv", ".md"))
            for task in [settings["source_task"]] + settings["history_tasks"] + [settings["target_task"]]:
                task_root = cell / ("task" + str(task))
                cases = settings["attacks"] if task == settings["target_task"] else (("reckless", .3),)
                for mode, delta in cases:
                    case = task_root / study.attack_name(mode, delta)
                    for name in ("features.metadata.json", "manifest.csv"):
                        path = case / "features" / name
                        if path.is_file():
                            files.add(path)
                    files.update(path for path in (case / "evaluation").rglob("*")
                                 if path.is_file() and path.suffix in (".json", ".csv", ".md"))
                    # Small accuracy matrices support recomputation of forgetting metrics.
                    files.update((case / "effectiveness").glob("*/acc_mat_*.npy"))
                reference = task_root / "historical_reference"
                files.update(path for path in reference.rglob("*") if path.is_file()
                             and path.suffix in (".json", ".csv"))
    return sorted(path for path in files if path.is_file())


def package(args):
    root = ensure_output_path(args.output_dir)
    # Rebuild and validate the summary first; packaging cannot imply completion.
    summarize(args)
    _, payload = study.read_plan(root / "plan.json")
    summary = json.loads((root / "analysis/summary.json").read_text())
    if summary["status"] != "complete" and not args.allow_incomplete:
        raise RuntimeError("Run is incomplete. See analysis/summary.json; use --allow-incomplete only for diagnostics.")
    files = analysis_files(root, payload["settings"])
    archive = ensure_output_path(args.archive or root.parent / (
        root.name + "_analysis_" + datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f") + ".tar.gz"))
    if archive in files or archive.exists():
        raise FileExistsError("Choose a new archive filename: " + str(archive))
    archive.parent.mkdir(parents=True, exist_ok=True)
    records = {str(path.relative_to(root)): sha256_file(path) for path in files}
    manifest = json.dumps(dict(status=summary["status"], files=records,
        omitted="Datasets, checkpoints, noise, feature matrices and joblib models; this is an analysis package, not a full reproduction archive."),
        indent=2).encode("utf-8")
    with archive.open("xb") as target:
        with tarfile.open(fileobj=target, mode="w:gz") as bundle:
            for path in files:
                bundle.add(str(path), arcname=root.name + "/" + str(path.relative_to(root)), recursive=False)
            item = tarfile.TarInfo(root.name + "/package_manifest.json")
            item.size = len(manifest)
            bundle.addfile(item, io.BytesIO(manifest))
    print("Analysis archive:", archive)
    return archive


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="action", required=True)
    for action in ("plan", "run", "summarize", "package"):
        p = subs.add_parser(action)
        p.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
        if action == "plan":
            p.add_argument("--methods", nargs="+", choices=list(study.METHODS), default=list(study.METHODS))
            p.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
        elif action == "run":
            p.add_argument("--method", choices=list(study.METHODS))
            p.add_argument("--seed", type=int)
            p.add_argument("--dry-run", action="store_true")
            p.add_argument("--retry-failed", action="store_true")
        elif action == "package":
            p.add_argument("--archive")
            p.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()
    {"plan": prepare, "run": run, "summarize": summarize, "package": package}[args.action](args)


if __name__ == "__main__":
    main()
