"""Register and execute a source-to-later-task study, entirely inside detector/."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shlex
import sys

import numpy as np

from detector.bootstrap import build_commands as legacy_commands
from detector.common import save_json, sha256_file
from detector.io_utils import DETECTOR_ROOT, ensure_output_path
from detector.pipeline import child_environment, environment_record, execute
from detector.transfer_core import fingerprint, task_index
from detector.transfer_detector import FEATURE_GROUPS, fresh_output


def command(module, *args):
    return [sys.executable, "-B", "-u", "-m", "detector." + module] + [
        str(x) for x in args
    ]


def artifact_steps(root, task, seed, settings):
    options = argparse.Namespace(
        output_dir=str(root),
        seed=seed,
        victim_epochs=settings["victim_epochs"],
        inversion_iters=settings["inversion_iters"],
        attack_epochs=settings["attack_epochs"],
        delta=settings["delta"],
    )
    steps = []
    for name, argv, expected in legacy_commands(options):
        script, arguments = Path(argv[3]).name, argv[4:]
        if "--lasttask" in arguments:
            arguments[arguments.index("--lasttask") + 1] = str(task)
        if "--task_lst" in arguments:
            arguments[arguments.index("--task_lst") + 1] = ",".join(
                map(str, range(task))
            )
            outputs = [
                root / "inversions" / "inversions_tid_{:02d}.npz".format(i)
                for i in range(task)
            ]
        else:
            outputs = [expected]
        argv = command(
            "transfer_upstream", "--incoming-task", task, script, "--", *arguments
        )
        steps.append(
            {
                "name": name,
                "command": argv,
                "cwd": str(root),
                "outputs": list(map(str, outputs)),
            }
        )
    return steps


def effectiveness(root, task):
    result = {}
    for label, filename in (
        ("clean", "acc_mat_clean.npy"),
        ("poison", "acc_mat_ours.npy"),
    ):
        path = root / "effectiveness" / label / filename
        matrix = np.load(path, allow_pickle=False)
        if (
            matrix.shape != (task + 1, task + 1)
            or not np.isfinite(matrix).all()
            or np.any((matrix < 0) | (matrix > 1))
        ):
            raise ValueError("Invalid stage-specific accuracy matrix.")
        result[label] = {
            "past_task_mean_accuracy": float(matrix[task, :task].mean()),
            "incoming_task_accuracy": float(matrix[task, task]),
            "backward_transfer": float(
                (matrix[task, :task] - np.diag(matrix)[:task]).mean()
            ),
            "sha256": sha256_file(path),
        }
    result["past_accuracy_drop_clean_minus_poison"] = (
        result["clean"]["past_task_mean_accuracy"]
        - result["poison"]["past_task_mean_accuracy"]
    )
    result["task_index"] = task
    result["note"] = (
        "Paired full-task attack-effectiveness control, not detection/filtering defense efficacy. No population significance claim."
    )
    save_json(result, root / "effectiveness" / "metrics.json")
    return result


def build_steps(root, settings):
    """Freeze source models before generating or reading target benchmark features."""
    steps = []
    for seed in settings["seeds"]:
        experiment = root / ("seed" + str(seed))
        source = settings["source_task"]
        for task in [source] + settings["targets"]:
            prefix = "seed{}_task{}".format(seed, task)
            artifact = experiment / ("artifacts_task" + str(task))
            features = experiment / ("features_task" + str(task))
            for step in artifact_steps(artifact, task, seed, settings):
                steps.append(dict(step, name=prefix + "_" + step["name"]))
            steps.append(
                {
                    "name": prefix + "_effectiveness",
                    "action": "effectiveness",
                    "task": task,
                    "cwd": str(artifact),
                    "outputs": [str(artifact / "effectiveness/metrics.json")],
                }
            )
            steps.append(
                {
                    "name": prefix + "_extract",
                    "cwd": str(artifact),
                    "command": command(
                        "transfer_extract",
                        "--checkpoint",
                        artifact / "victim/checkpoint.pkl",
                        "--artifact",
                        artifact / "attack/noise.pkl",
                        "--inversion-dir",
                        artifact / "inversions",
                        "--incoming-task",
                        task,
                        "--label-mode",
                        settings["label_mode"],
                        "--output-dir",
                        features,
                    ),
                    "outputs": [str(features)],
                }
            )
            for feature_set in settings["feature_sets"]:
                model = experiment / ("source_" + feature_set)
                if task == source:
                    steps.append(
                        {
                            "name": prefix + "_fit_" + feature_set,
                            "cwd": str(root),
                            "command": command(
                                "transfer_detector",
                                "fit",
                                "--features",
                                features / "features.csv",
                                "--feature-set",
                                feature_set,
                                "--task-size",
                                settings["task_size"],
                                "--output-dir",
                                model,
                            ),
                            "outputs": [str(model)],
                        }
                    )
                    explanation = experiment / ("explain_" + feature_set)
                    extra = ["--ablations"] if feature_set == "extended" else []
                    steps.append(
                        {
                            "name": prefix + "_explain_" + feature_set,
                            "cwd": str(root),
                            "command": command(
                                "explain_detector",
                                "--features",
                                features / "features.csv",
                                "--bundle",
                                model / "bundle.joblib",
                                "--output-dir",
                                explanation,
                                *extra,
                            ),
                            "outputs": [str(explanation)],
                        }
                    )
                evaluation = experiment / (
                    "evaluate_task{}_{}".format(task, feature_set)
                )
                extra = ["--source-control"] if task == source else []
                steps.append(
                    {
                        "name": prefix + "_evaluate_" + feature_set,
                        "cwd": str(root),
                        "command": command(
                            "transfer_detector",
                            "evaluate",
                            "--features",
                            features / "features.csv",
                            "--bundle",
                            model / "bundle.joblib",
                            "--bags-per-rate",
                            settings["bags_per_rate"],
                            "--output-dir",
                            evaluation,
                            *extra,
                        ),
                        "outputs": [str(evaluation)],
                    }
                )
    return steps


def output_hashes(paths):
    hashes = {}
    for value in paths:
        path = Path(value)
        if not path.exists():
            raise FileNotFoundError("Step did not create: " + str(path))
        files = (
            sorted(p for p in path.rglob("*") if p.is_file())
            if path.is_dir()
            else [path]
        )
        if not files:
            raise ValueError("Empty step output: " + str(path))
        for file in files:
            if file.is_symlink():
                raise ValueError("Symlink in step output.")
            hashes[str(file)] = sha256_file(file)
    return hashes


def summarize(root, settings, protocol):
    rows = []
    for seed in settings["seeds"]:
        for task in [settings["source_task"]] + settings["targets"]:
            effect = json.loads(
                (
                    root
                    / "seed{}".format(seed)
                    / "artifacts_task{}".format(task)
                    / "effectiveness/metrics.json"
                ).read_text()
            )
            for group in settings["feature_sets"]:
                path = (
                    root
                    / "seed{}".format(seed)
                    / "evaluate_task{}_{}".format(task, group)
                    / "evaluation_metrics.json"
                )
                item = json.loads(path.read_text())
                rows.append(
                    {
                        "seed": seed,
                        "task_index": task,
                        "feature_set": group,
                        "source_control": item["source_control"],
                        "sample_metrics": item["sample_metrics"],
                        "random_sample_alert_rate": item[
                            "random_control_sample_alert_rate"
                        ],
                        "dataset_results": item["dataset_results"],
                        "attack_past_accuracy_drop": effect[
                            "past_accuracy_drop_clean_minus_poison"
                        ],
                        "path": str(path),
                    }
                )
    aggregates = []
    for task in [settings["source_task"]] + settings["targets"]:
        for group in settings["feature_sets"]:
            subset = [
                r for r in rows if r["task_index"] == task and r["feature_set"] == group
            ]
            stats = {}
            for metric in ("roc_auc", "clean_fpr", "poison_tpr"):
                values = [r["sample_metrics"][metric] for r in subset]
                stats[metric] = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values, ddof=1)) if len(values) > 1 else None,
                }
            dataset_stats = []
            conditions = sorted(
                {
                    (entry["alternative"], entry["requested_rate"])
                    for run in subset
                    for entry in run["dataset_results"]
                }
            )
            for alternative, rate in conditions:
                values = [
                    entry["alert_rate"]
                    for run in subset
                    for entry in run["dataset_results"]
                    if entry["alternative"] == alternative
                    and entry["requested_rate"] == rate
                ]
                dataset_stats.append(
                    {
                        "alternative": alternative,
                        "requested_rate": rate,
                        "mean_alert_rate": float(np.mean(values)),
                        "std_alert_rate": float(np.std(values, ddof=1))
                        if len(values) > 1
                        else None,
                    }
                )
            aggregates.append(
                {
                    "task_index": task,
                    "feature_set": group,
                    "runs": len(subset),
                    "sample": stats,
                    "dataset": dataset_stats,
                }
            )
    save_json(
        {
            "protocol": protocol,
            "runs": rows,
            "aggregates": aggregates,
            "limitations": "Shared CIFAR class order and image pool; few seeds. Do not treat bags as independent repetitions. No target calibration or automatic winning-model selection.",
        },
        root / "summary.json",
    )
    lines = [
        "# Source-to-target study",
        "",
        "All task indices are zero based. Source: {}.".format(settings["source_task"]),
        "",
        "| Seed | Task | Features | AUC | Clean FPR | Poison TPR | Attack historical accuracy drop |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for r in rows:
        s = r["sample_metrics"]
        lines.append(
            "| {} | {} | {} | {:.4f} | {:.2%} | {:.2%} | {:.2%} |".format(
                r["seed"],
                r["task_index"],
                r["feature_set"],
                s["roc_auc"],
                s["clean_fpr"],
                s["poison_tpr"],
                r["attack_past_accuracy_drop"],
            )
        )
    lines += [
        "",
        "Inspect per-task reports for dataset-level curves, random controls and source count thresholds.",
        "Source performance does not guarantee target performance; all results, including failures, must be retained.",
    ]
    (root / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args):
    plan = Path(args.plan).resolve()
    payload = json.loads(plan.read_text())
    settings = payload["settings"]
    if fingerprint(settings) != payload:
        raise ValueError(
            "Code/settings differ from the registered protocol. Use a new experiment, not a rewritten protocol."
        )
    root = ensure_output_path(plan.parent)
    steps = build_steps(root, settings)
    if args.dry_run:
        for step in steps:
            print(step["name"], "cwd=" + step["cwd"])
            print(
                shlex.join(step["command"])
                if "command" in step
                else "Compute paired attack-effectiveness metrics"
            )
        return
    state_path = root / "run_state.json"
    environment_path = root / "run_environment.json"
    current_environment = environment_record()
    if environment_path.exists():
        if json.loads(environment_path.read_text()) != current_environment:
            raise ValueError(
                "Execution environment changed; use a new registered experiment."
            )
    else:
        save_json(current_environment, environment_path)
    state = json.loads(state_path.read_text()) if state_path.exists() else {}
    for step in steps:
        name = step["name"]
        if name in state:
            if state[name]["status"] != "complete":
                raise RuntimeError(
                    "Incomplete step {}: inspect its log and partial outputs. Automatic in-epoch resume is unsupported.".format(
                        name
                    )
                )
            if output_hashes(step["outputs"]) != state[name]["outputs"]:
                raise ValueError("Completed outputs changed: " + name)
            continue
        if any(Path(p).exists() for p in step["outputs"]):
            raise FileExistsError("Unregistered output exists for " + name)
        cwd = ensure_output_path(step["cwd"])
        cwd.mkdir(parents=True, exist_ok=True)
        state[name] = {
            "status": "running",
            "started_utc": datetime.now(timezone.utc).isoformat(),
            "command": step.get("command"),
            "cwd": str(cwd),
        }
        save_json(state, state_path)
        try:
            if "command" in step:
                execute(
                    step["command"],
                    cwd=cwd,
                    log_path=root / "logs" / (name + ".log"),
                    env=child_environment(cwd),
                )
            else:
                effectiveness(cwd, step["task"])
            if fingerprint(settings) != payload:
                raise ValueError("Method code changed during execution.")
            state[name].update(
                status="complete", outputs=output_hashes(step["outputs"])
            )
        except BaseException:
            state[name]["status"] = "failed"
            save_json(state, state_path)
            raise
        save_json(state, state_path)
    summarize(root, settings, payload)
    print("Transfer study completed:", root)


def plan(args):
    root = fresh_output(args.output_dir)
    source = task_index(args.source_task)
    targets = [task_index(t) for t in args.targets]
    if (
        not targets
        or len(set(targets)) != len(targets)
        or any(t <= source for t in targets)
    ):
        raise ValueError("Specify distinct later target tasks.")
    if not args.seeds or len(set(args.seeds)) != len(args.seeds) or min(args.seeds) < 0:
        raise ValueError("Specify distinct nonnegative seeds.")
    if args.attack_epochs < 100 or args.attack_epochs % 100:
        raise ValueError("Attack epochs must be a positive multiple of 100.")
    if min(args.victim_epochs, args.inversion_iters, args.bags_per_rate) < 1:
        raise ValueError("Epochs/iterations/bag counts must be positive.")
    settings = {
        "source_task": source,
        "targets": targets,
        "seeds": args.seeds,
        "label_mode": args.label_mode,
        "feature_sets": list(FEATURE_GROUPS),
        "task_size": 150,
        "bags_per_rate": args.bags_per_rate,
        "victim_epochs": args.victim_epochs,
        "inversion_iters": args.inversion_iters,
        "attack_epochs": args.attack_epochs,
        "delta": 0.3,
        "target_label_access": "Class labels allowed only when ground_truth; poison labels offline evaluation only.",
        "prior_tasks_clean": True,
        "source_class_reuse_in_later_backbone": True,
        "registration_role": "planned_replication_after_exploratory_task9_results",
    }
    payload = fingerprint(settings)
    root.mkdir(parents=True)
    save_json(payload, root / "protocol.json")
    script = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "cd " + shlex.quote(str(DETECTOR_ROOT.parent)),
        "export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1",
        shlex.join(
            command("transfer_pipeline", "run", "--plan", root / "protocol.json")
        ),
    ]
    (root / "commands.sh").write_text("\n".join(script) + "\n", encoding="utf-8")
    print("Plan saved, no training launched:", root / "commands.sh")


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("plan")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--source-task", type=int, default=1)
    p.add_argument("--targets", type=int, nargs="+", default=[9])
    p.add_argument("--seeds", type=int, nargs="+", default=[3, 4])
    p.add_argument(
        "--label-mode", choices=("ground_truth", "predicted"), default="ground_truth"
    )
    p.add_argument("--victim-epochs", type=int, default=20)
    p.add_argument("--inversion-iters", type=int, default=10000)
    p.add_argument("--attack-epochs", type=int, default=5000)
    p.add_argument("--bags-per-rate", type=int, default=1000)
    p = sub.add_parser("run")
    p.add_argument("--plan", required=True)
    p.add_argument("--dry-run", action="store_true")
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    plan(args) if args.command == "plan" else run(args)
