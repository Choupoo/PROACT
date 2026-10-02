"""Registered predicted-gradient study: development, freeze, then target tests.

All outputs stay under detector/. Old runs are never modified. A completed
stage is reused only if its hashes match; partial stages require explicit retry.
"""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shlex

import numpy as np
import pandas as pd

from detector import advisor_detector as advisor
from detector import transfer_detector as supervised
from detector.common import save_json, sha256_file
from detector.io_utils import ensure_output_path, load_frozen_bundle, read_feature_table
from detector.pipeline import child_environment, environment_record, execute
from detector.transfer_core import fingerprint
from detector.transfer_pipeline import artifact_steps, command, output_hashes
from detector.unsupervised_adapt import evaluate_benchmark

# Explicit benchmark settings, not a claim of exact paper-table replication.
METHODS = {"ewc": (500000., None), "mas": (10., None), "rwalk": (1., None),
           "afec_ewc": (500000., 100.), "ancl_ewc": (500000., 100.)}
ATTACKS = (("reckless", .1), ("reckless", .3), ("cautious", .1), ("cautious", .3))


def replace_flag(argv, flag, value):
    argv = list(argv)
    if flag in argv:
        argv[argv.index(flag) + 1] = str(value)
    else:
        argv.extend([flag, str(value)])
    return argv


def attack_name(mode, delta):
    return "{}_{}".format(mode, format(delta, ".1f").replace(".", "p"))


def build_cell(root, settings, method, seed):
    """GPU work is shared within a task, never across distinct CL checkpoints."""
    cell = root / method / ("seed" + str(seed))
    steps = []
    histories = []
    model_dir = cell / "frozen_supervised"
    for task in [settings["source_task"]] + settings["history_tasks"] + [settings["target_task"]]:
        task_root = cell / ("task" + str(task))
        base = {r["name"]: r for r in artifact_steps(task_root, task, seed, settings)}
        for name in ("victim", "inversion"):
            step = dict(base[name], name="task{}_{}".format(task, name))
            if name == "victim":
                step["command"] = replace_flag(step["command"], "--approach", method)
                step["command"] = replace_flag(step["command"], "--lamb", settings["method_parameters"][method][0])
                if settings["method_parameters"][method][1] is not None:
                    step["command"] = replace_flag(step["command"], "--lamb_emp", settings["method_parameters"][method][1])
            step["owned"] = [str(task_root / ("victim" if name == "victim" else "inversions"))]
            steps.append(step)
        is_target = task == settings["target_task"]
        reference = task_root / "historical_reference"
        if is_target and settings["unsupervised"]:
            steps.append(dict(name="freeze_label_free", cwd=str(task_root),
                              command=command("advisor_reference", "--checkpoint", task_root / "victim/checkpoint.pkl",
                                              "--inversion-dir", task_root / "inversions", "--output-dir", reference,
                                              "--task-size", settings["task_size"], "--alpha", settings["alpha"]),
                              outputs=[str(reference)], owned=[str(reference)]))
        for mode, delta in (settings["attacks"] if is_target else (("reckless", .3),)):
            case = task_root / attack_name(mode, delta)
            attack = case / "attack"
            features = case / "features"
            prefix = "task{}_{}".format(task, attack_name(mode, delta))
            argv = replace_flag(base["attack"]["command"], "--mode", mode)
            argv = replace_flag(argv, "--delta", delta)
            argv = replace_flag(argv, "--w_cur", settings["cautious_weight"])
            argv = replace_flag(argv, "--output_dir", attack)
            steps.append(dict(name=prefix + "_attack", command=argv, cwd=str(task_root),
                              outputs=[str(attack / "noise.pkl")], owned=[str(attack)]))
            steps.append(dict(name=prefix + "_extract", cwd=str(task_root),
                              command=command("transfer_extract", "--checkpoint", task_root / "victim/checkpoint.pkl",
                                              "--artifact", attack / "noise.pkl", "--inversion-dir", task_root / "inversions",
                                              "--incoming-task", task, "--label-mode", "predicted", "--inference-schema", "--random-control", "uniform",
                                              "--output-dir", features), outputs=[str(features)], owned=[str(features)]))
            if not is_target:
                histories.append(features / "features.csv")
                continue
            # Full-task clean/poison/uniform training measures forgetting, not
            # the benefit of filtering. Uniform uses an independent noise draw.
            for label, base_name, filename in (("clean", "clean_training", "clean"),
                                                ("poison", "poison_training", "ours"),
                                                ("uniform", "poison_training", "uniform")):
                if not settings["effectiveness"]:
                    continue
                effect = case / "effectiveness" / label
                argv = replace_flag(base[base_name]["command"], "--checkpoint", attack / "noise.pkl")
                argv = replace_flag(argv, "--output_dir", effect)
                argv = replace_flag(argv, "--approach", method)
                argv = replace_flag(argv, "--lamb", settings["method_parameters"][method][0])
                if settings["method_parameters"][method][1] is not None:
                    argv = replace_flag(argv, "--lamb_emp", settings["method_parameters"][method][1])
                if label == "uniform":
                    argv.append("--uniform")
                steps.append(dict(name=prefix + "_train_" + label, command=argv, cwd=str(task_root),
                                  outputs=[str(effect / ("acc_mat_" + filename + ".npy"))], owned=[str(effect)]))
            result = case / "evaluation"
            steps.append(dict(name=prefix + "_evaluate", cwd=str(task_root),
                              command=command("advisor_study", "evaluate", "--features", features / "features.csv",
                                              "--frozen", model_dir, "--output-dir", result,
                                              *(["--reference", reference] if settings["unsupervised"] else []),
                                              "--bags-per-rate", settings["bags_per_rate"]),
                              outputs=[str(result)], owned=[str(result)]))
        if task == settings["history_tasks"][-1]:
            steps.append(dict(name="freeze_supervised", cwd=str(cell),
                              command=command("advisor_detector", "--source", histories[0], "--histories", *histories[1:],
                                              "--output-dir", model_dir, "--task-size", settings["task_size"],
                                              "--bags-per-rate", settings["bags_per_rate"], "--alpha", settings["alpha"],
                                              *(["--ablations"] if settings["ablations"] else [])),
                              outputs=[str(model_dir)], owned=[str(model_dir)]))
    return cell, steps


def evaluate(args):
    output = supervised.fresh_output(args.output_dir)
    frozen = Path(args.frozen)
    freeze = json.loads((frozen / "freeze.json").read_text())
    table, meta = read_feature_table(args.features)
    advisor.check_predicted(table, meta)
    if meta["task_index"] <= max(freeze["calibration_tasks"]):
        raise ValueError("Final target must be later than every development task.")
    output.mkdir(parents=True)
    for family, names in (("ablations", freeze["variants"]), ("thresholds", advisor.RULES)):
        for name in names:
            bundle = load_frozen_bundle(frozen / family / name / "bundle.joblib")
            source = bundle["source_metadata"]
            if any(meta.get(k) != source.get(k) for k in ("cl_method", "checkpoint_seed")):
                raise ValueError("Target CL method/seed differs from the fitted experiment.")
            report, predictions, bags = supervised.evaluate(table, meta, bundle, repeats=args.bags_per_rate)
            report.update(threshold_rule=bundle["threshold_rule"], calibration_history=bundle["calibration_history"],
                          threshold_selection=freeze["selection"], attack_config=meta["attack_config"],
                          cl_method=meta["cl_method"], checkpoint_seed=meta["checkpoint_seed"])
            destination = output / family / name
            save_json(report, destination / "evaluation_metrics.json")
            predictions.to_csv(destination / "predictions.csv", index=False)
            bags.to_csv(destination / "bags.csv", index=False)
    if not args.reference:
        return
    rank = load_frozen_bundle(Path(args.reference) / "rank/bundle.joblib")
    mmd = load_frozen_bundle(Path(args.reference) / "mmd/bundle.joblib")
    result = evaluate_benchmark(rank, mmd, table, meta, task_size=rank["settings"]["max_incoming"],
                                tasks_per_rate=args.bags_per_rate, progress=True)
    # Class/image pools have been explored previously; new seeds alone do not
    # make this an independent new dataset or attack-family confirmation.
    result.update(attack_config=meta["attack_config"], cl_method=meta["cl_method"],
                  reference_tasks=list(range(meta["task_index"])),
                  previous_benchmark_exposure="Same dataset/task order; new registered runs, not a new population.")
    save_json(result, output / "unsupervised.json")


def effect_metrics(case, task):
    result = {}
    for label, suffix in (("clean", "clean"), ("poison", "ours"), ("uniform", "uniform")):
        path = case / "effectiveness" / label / ("acc_mat_" + suffix + ".npy")
        matrix = np.load(path, allow_pickle=False)
        if matrix.shape != (task + 1, task + 1) or not np.isfinite(matrix).all() or np.any((matrix < 0) | (matrix > 1)):
            raise ValueError("Invalid accuracy matrix: " + str(path))
        result[label] = dict(past_accuracy=float(matrix[task, :task].mean()),
                             incoming_accuracy=float(matrix[task, task]),
                             bwt=float((matrix[task, :task] - np.diag(matrix)[:task]).mean()), sha256=sha256_file(path))
    for label in ("poison", "uniform"):
        result[label]["past_drop_vs_clean"] = result["clean"]["past_accuracy"] - result[label]["past_accuracy"]
    result["uniform_note"] = "Independent uniform draw at the same L-infinity budget, not the detector-control realization."
    return result


def read_plan(path):
    path = Path(path).resolve()
    payload = json.loads(path.read_text())
    if fingerprint(payload["settings"]) != payload:
        raise ValueError("Registered code/settings changed. Use a new run directory; do not mix versions.")
    return ensure_output_path(path.parent), payload


def run(args):
    root, payload = read_plan(args.plan)
    settings = payload["settings"]
    methods = [args.method] if args.method else settings["methods"]
    seeds = [args.seed] if args.seed is not None else settings["seeds"]
    if not set(methods) <= set(settings["methods"]) or not set(seeds) <= set(settings["seeds"]):
        raise ValueError("Requested cell is not in this plan.")
    for method in methods:
        for seed in seeds:
            cell, steps = build_cell(root, settings, method, seed)
            if args.dry_run:
                for step in steps:
                    print(step["name"], "cwd=" + step["cwd"])
                    print(shlex.join(step["command"]))
                continue
            cell.mkdir(parents=True, exist_ok=True)
            # OS advisory lock is released on exit/crash; separate cells may run concurrently.
            import fcntl
            with (cell / ".run.lock").open("a") as lock:
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError as error:
                    raise RuntimeError("This method/seed is already running.") from error
                execute_cell(cell, steps, payload, args.retry_failed)


def execute_cell(cell, steps, payload, retry_failed):
    env_file = cell / "run_environment.json"
    environment = environment_record()
    if env_file.exists() and json.loads(env_file.read_text()) != environment:
        raise ValueError("Execution environment changed; use a new experiment.")
    save_json(environment, env_file)
    state_file = cell / "run_state.json"
    state = json.loads(state_file.read_text()) if state_file.exists() else {}
    for step in steps:
        name = step["name"]
        record = state.get(name)
        if record and record["status"] == "complete":
            if output_hashes(step["outputs"]) != record["outputs"]:
                raise ValueError("Completed artifacts changed: " + name)
            continue
        if record:
            if not retry_failed:
                raise RuntimeError("Partial stage {}. Inspect its log, then add --retry-failed to archive and restart this stage.".format(name))
            archive = cell / "interrupted" / (name + "_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f"))
            archive.mkdir(parents=True)
            # Only move this stage's exclusive output directories, never data,
            # shared checkpoint/inversions, or another stage's artifacts.
            for value in step["owned"]:
                owned = ensure_output_path(value)
                if cell not in owned.parents:
                    raise ValueError("Refusing out-of-cell recovery.")
                if owned.exists():
                    owned.rename(archive / owned.name)
            old_log = cell / "logs" / (name + ".log")
            if old_log.exists():
                old_log.rename(archive / old_log.name)
            save_json(record, archive / "previous_state.json")
            print("Archived incomplete stage:", archive, flush=True)
        elif any(Path(p).exists() for p in step["owned"]):
            raise FileExistsError("Untracked outputs exist for " + name)
        state[name] = dict(status="running", command=step["command"], started=datetime.now(timezone.utc).isoformat())
        save_json(state, state_file)
        cwd = ensure_output_path(step["cwd"])
        cwd.mkdir(parents=True, exist_ok=True)
        try:
            execute(step["command"], cwd=cwd, log_path=cell / "logs" / (name + ".log"), env=child_environment(cwd))
            if fingerprint(payload["settings"]) != payload:
                raise ValueError("Code changed while running; do not combine outputs.")
            state[name].update(status="complete", outputs=output_hashes(step["outputs"]), finished=datetime.now(timezone.utc).isoformat())
        except BaseException as error:
            state[name].update(status="failed", error=str(error))
            save_json(state, state_file)
            raise
        save_json(state, state_file)


def summarize(args):
    root, payload = read_plan(args.plan)
    settings = payload["settings"]
    rows, label_free, missing, selections, effects = [], [], [], [], []
    for method in settings["methods"]:
        for seed in settings["seeds"]:
            cell, steps = build_cell(root, settings, method, seed)
            state_path = cell / "run_state.json"
            state = json.loads(state_path.read_text()) if state_path.exists() else {}
            incomplete = [s["name"] for s in steps if state.get(s["name"], {}).get("status") != "complete"]
            if incomplete:
                missing.append(dict(method=method, seed=seed, stages=incomplete))
                continue
            for step in steps:
                if output_hashes(step["outputs"]) != state[step["name"]]["outputs"]:
                    raise ValueError("Artifact hash changed: " + step["name"])
            selection = json.loads((cell / "frozen_supervised/threshold_selection.json").read_text())
            selections.append(dict(method=method, seed=seed, **selection))
            for mode, delta in settings["attacks"]:
                case = cell / ("task" + str(settings["target_task"])) / attack_name(mode, delta)
                identity = dict(method=method, seed=seed, attack=mode, delta=delta)
                if settings["effectiveness"]:
                    effects.append(dict(identity, **effect_metrics(case, settings["target_task"])))
                variants = supervised.INFERENCE_GROUPS if settings["ablations"] else ["inference_full"]
                for family, variants in (("ablations", variants), ("thresholds", advisor.RULES)):
                    for variant in variants:
                        report = json.loads((case / "evaluation" / family / variant / "evaluation_metrics.json").read_text())
                        clean = next(r["alert_rate"] for r in report["dataset_results"] if r["realized_rate"] == 0)
                        row = dict(identity, family=family, variant=variant, **report["sample_metrics"],
                                   clean_dataset_fpr=clean, random_sample_alert=report["random_control_sample_alert_rate"],
                                   threshold_rule=report["threshold_rule"], dataset_results=report["dataset_results"])
                        rows.append(row)
                if settings["unsupervised"]:
                    result = json.loads((case / "evaluation/unsupervised.json").read_text())
                    label_free.append(dict(identity, summary=result["summary"], limitations=result["limitations"]))
    output = ensure_output_path(root / "analysis")
    output.mkdir(exist_ok=True)
    aggregates, paired_ablation_deltas = [], []
    frame = pd.DataFrame(rows)
    if len(frame):
        frame.drop(columns=["dataset_results"]).to_csv(output / "supervised.csv", index=False)
        for keys, group in frame.groupby(["method", "attack", "delta", "family", "variant"]):
            item = dict(zip(["method", "attack", "delta", "family", "variant"], keys))
            item["n_seeds"] = len(group)
            for metric in ("roc_auc", "clean_fpr", "poison_tpr", "clean_dataset_fpr"):
                item[metric] = dict(mean=float(group[metric].mean()), std=float(group[metric].std(ddof=1)) if len(group) > 1 else None)
            aggregates.append(item)
        for keys, group in frame[frame.family == "ablations"].groupby(["method", "seed", "attack", "delta"]):
            baseline = group[group.variant == "inference_full"].iloc[0]
            for _, row in group.iterrows():
                if row.variant == "inference_full":
                    continue
                item = dict(zip(["method", "seed", "attack", "delta"], keys))
                item["variant"] = row.variant
                for metric in ("roc_auc", "clean_fpr", "poison_tpr", "clean_dataset_fpr"):
                    item[metric + "_minus_full"] = float(row[metric] - baseline[metric])
                paired_ablation_deltas.append(item)
        if paired_ablation_deltas:
            pd.DataFrame(paired_ablation_deltas).to_csv(output / "paired_ablation_deltas.csv", index=False)
    summary = dict(status="complete" if not missing else "incomplete", missing=missing,
                   selections=selections, supervised=rows, aggregates=aggregates, unsupervised=label_free,
                   paired_ablation_deltas=paired_ablation_deltas,
                   attack_effectiveness=effects, plan_fingerprint=payload["fingerprint"],
                   limitations=["Bags reuse finite images; summarize variability over seeds, not bags.",
                                "Same class order and dataset; seed variation is not a new population.",
                                "No target calibration. Failed thresholds must be reported.",
                                "CIFAR-100 methods/attack matrix, not a replication of all BrainWash datasets.",
                                "No detection/filtering defense-efficacy claim."])
    save_json(summary, output / "summary.json")
    lines = ["# Predicted-gradient detector update", "", "Status: **{}**.".format(summary["status"]), "",
             "The detector uses predicted classes, not image-class labels. Tensor norms are retained.",
             "Threshold rules are selected on development tasks and frozen before Task 9. Random alerts are noise sensitivity, not clean FPR.", "",
             "| CL method | Seed | Attack | Budget | AUC | Sample clean FPR | Poison TPR | Dataset clean FPR |",
             "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |"]
    for row in rows:
        if row["family"] == "ablations" and row["variant"] == "inference_full":
            lines.append("| {method} | {seed} | {attack} | {delta} | {roc_auc:.4f} | {clean_fpr:.2%} | {poison_tpr:.2%} | {clean_dataset_fpr:.2%} |".format(**row))
    lines += ["", "## Dataset-level detection", "",
              "Each bag contains distinct original images. Requested fractions are rounded to whole images; use the realized fraction when interpreting low-rate detection.", "",
              "| CL method | Seed | Attack | Budget | View | Realized fraction | Alert rate |",
              "| --- | ---: | --- | ---: | --- | ---: | ---: |"]
    for row in rows:
        if row["family"] == "ablations" and row["variant"] == "inference_full":
            for rate in row["dataset_results"]:
                if rate["alternative"] == "random_control" and rate["realized_rate"] == 0:
                    continue  # The same clean bags must not be counted twice.
                lines.append("| {} | {} | {} | {} | {} | {:.3%} | {:.2%} |".format(
                    row["method"], row["seed"], row["attack"], row["delta"],
                    "clean" if rate["realized_rate"] == 0 else rate["alternative"], rate["realized_rate"], rate["alert_rate"]))
    lines += ["", "## Strictly label-free comparison", "",
              "Rank tests Kendall tau-a feature-pair relationships separately against each historical inversion task. It alerts only when all references disagree. MMD tests the pooled descriptor distribution. Both use predicted-gradient descriptors, no trusted target-clean calibration, and neither produces a poisoning probability.", "",
              "| CL method | Seed | Attack | Budget | View | Realized fraction | Rank alert | Rank coverage | MMD alert |",
              "| --- | ---: | --- | ---: | --- | ---: | ---: | ---: | ---: |"]
    if not settings["unsupervised"]:
        lines += ["", "Not rerun in this methodology pilot; see the methodology note for the definition of Rank and simulated bags."]
    for item in label_free:
        for row in item["summary"]:
            alert = "unsupported" if row["rank_alert_rate"] is None else "{:.2%}".format(row["rank_alert_rate"])
            lines.append("| {} | {} | {} | {} | {} | {:.3%} | {} | {:.2%} | {:.2%} |".format(
                item["method"], item["seed"], item["attack"], item["delta"], row["scenario"], row["realized_rate"],
                alert, row["rank_coverage"], row["legacy_alert_rate"]))
    lines += ["", "## Does perturbation cause forgetting?", "",
              "These are full-task training controls, not experiments showing that filtering helps. The uniform-control training and detector features use independent draws at the same budget.", "",
              "| CL method | Seed | Attack | Budget | Poison past-accuracy drop vs clean | Uniform drop vs clean | Poison BWT | Uniform BWT |",
              "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |"]
    for row in effects:
        lines.append("| {} | {} | {} | {} | {:.2%} | {:.2%} | {:.4f} | {:.4f} |".format(
            row["method"], row["seed"], row["attack"], row["delta"], row["poison"]["past_drop_vs_clean"],
            row["uniform"]["past_drop_vs_clean"], row["poison"]["bwt"], row["uniform"]["bwt"]))
    if not settings["effectiveness"]:
        lines += ["", "Additional forgetting controls are deferred; this pilot only fixes features and evaluates threshold rules."]
    lines += ["", "## Ablations and seed variability", "",
              ("Ablations are retrained on the same source splits; paired_ablation_deltas.csv gives variant minus full-model differences. These are diagnostic comparisons, not target-based selection."
               if settings["ablations"] else "Ablations are deferred until the methodology is agreed. Only the corrected full model is fitted."), "",
              "| CL method | Attack | Budget | Variant | Seeds | Mean AUC | AUC sample SD | Mean sample clean FPR | Mean dataset clean FPR |",
              "| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |"]
    for row in aggregates:
        if row["family"] == "ablations":
            sd = "n/a" if row["roc_auc"]["std"] is None else "{:.4f}".format(row["roc_auc"]["std"])
            lines.append("| {} | {} | {} | {} | {} | {:.4f} | {} | {:.2%} | {:.2%} |".format(
                row["method"], row["attack"], row["delta"], row["variant"], row["n_seeds"],
                row["roc_auc"]["mean"], sd, row["clean_fpr"]["mean"], row["clean_dataset_fpr"]["mean"]))
    lines += ["", "## What still needs attention", ""]
    if missing:
        lines.append("{} method/seed cells remain incomplete. This is not a final experimental result.".format(len(missing)))
    failed = sum(not r["development_criterion_met"] for r in selections)
    lines.append("{} completed cells had no threshold rule satisfying the development clean-error criterion.".format(failed))
    for row in rows:
        if row["family"] == "ablations" and row["variant"] == "inference_full" and max(row["clean_fpr"], row["clean_dataset_fpr"]) > settings["alpha"]:
            lines.append("- {} seed {} {} {}: the frozen rule exceeded the nominal clean-error target on Task 9.".format(row["method"], row["seed"], row["attack"], row["delta"]))
    lines += ["", "The JSON contains every ablation, threshold comparison, poisoning fraction, Rank/MMD result and clean/poison/uniform forgetting control. Do not select a new winner from target results.", ""] + summary["limitations"]
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("Saved:", output)


def plan(args):
    root = supervised.fresh_output(args.output_dir)
    tasks = [args.source_task] + args.history_tasks + [args.target_task]
    if tasks != sorted(set(tasks)) or min(tasks) < 1 or max(tasks) > 9:
        raise ValueError("Require distinct ordered source < development tasks < target in 1..9.")
    if len(set(args.seeds)) != len(args.seeds) or min(args.seeds) < 0 or len(set(args.methods)) != len(args.methods):
        raise ValueError("Seeds/methods must be unique; seeds nonnegative.")
    if args.attack_epochs < 100 or args.attack_epochs % 100 or min(args.victim_epochs, args.inversion_iters, args.bags_per_rate) < 1:
        raise ValueError("Positive epochs/repeats; attack epochs a multiple of 100.")
    if not 16 <= args.task_size <= 256 or not 0 < args.alpha < 1:
        raise ValueError("Require task size 16..256 and alpha in (0,1).")
    if not args.expanded_settings and (args.methods != ["ewc"] or args.ablations):
        raise ValueError("Professor requested methodology first. Broader methods/ablations require explicit --expanded-settings after agreement.")
    settings = dict(methods=args.methods, seeds=args.seeds, source_task=args.source_task,
                    history_tasks=args.history_tasks, target_task=args.target_task,
                    victim_epochs=args.victim_epochs, inversion_iters=args.inversion_iters,
                    attack_epochs=args.attack_epochs, delta=.3, cautious_weight=1.,
                    method_parameters={m: list(METHODS[m]) for m in args.methods},
                    task_size=args.task_size, bags_per_rate=args.bags_per_rate, alpha=args.alpha,
                    label_mode="predicted", protocol="advisor_predicted_v1",
                    attacks=[list(a) for a in (ATTACKS if args.expanded_settings else (("reckless", .3),))],
                    ablations=args.ablations, unsupervised=args.expanded_settings,
                    effectiveness=args.expanded_settings,
                    scope="expanded" if args.expanded_settings else "methodology_pilot")
    root.mkdir(parents=True)
    save_json(fingerprint(settings), root / "plan.json")
    print("Registered:", root / "plan.json")
    print("Scope:", settings["scope"], "; cells:", len(args.methods) * len(args.seeds), "; target attacks per cell:", len(settings["attacks"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="action", required=True)
    p = subs.add_parser("plan")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--methods", nargs="+", choices=list(METHODS), default=["ewc"])
    p.add_argument("--seeds", type=int, nargs="+", default=[5])
    p.add_argument("--expanded-settings", action="store_true", help="Deferred: use only after methodology agreement.")
    p.add_argument("--ablations", action="store_true", help="Deferred; requires --expanded-settings.")
    p.add_argument("--source-task", type=int, default=1)
    p.add_argument("--history-tasks", type=int, nargs="+", default=[4])
    p.add_argument("--target-task", type=int, default=9)
    p.add_argument("--victim-epochs", type=int, default=100)
    p.add_argument("--inversion-iters", type=int, default=2000)
    p.add_argument("--attack-epochs", type=int, default=5000)
    p.add_argument("--task-size", type=int, default=150)
    p.add_argument("--bags-per-rate", type=int, default=100)
    p.add_argument("--alpha", type=float, default=.05)
    p = subs.add_parser("run")
    p.add_argument("--plan", required=True)
    p.add_argument("--method", choices=list(METHODS))
    p.add_argument("--seed", type=int)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--retry-failed", action="store_true")
    p = subs.add_parser("summarize")
    p.add_argument("--plan", required=True)
    p = subs.add_parser("evaluate")
    for flag in ("features", "frozen", "output-dir"):
        p.add_argument("--" + flag, required=True)
    p.add_argument("--reference", help="Optional deferred Rank/MMD comparison.")
    p.add_argument("--bags-per-rate", type=int, default=100)
    args = parser.parse_args()
    {"plan": plan, "run": run, "evaluate": evaluate, "summarize": summarize}[args.action](args)


if __name__ == "__main__":
    main()
