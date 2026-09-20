"""Bounded thesis study: three fixed label-free methods, no supervised changes.

run: fit using historical references, freeze, then evaluate identical test bags.
plan: write (do not execute) independent-seed preparation commands and protocol.
summarize: summarize run-level rates, never mistake repeated bags for seed trials.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shlex
import sys

import numpy as np

from detector import local_reference as local
from detector import rank_reference as rank
from detector import unsupervised as legacy
from detector import unsupervised_adapt as adaptation
from detector.common import save_json, sha256_file
from detector.io_utils import (
    DETECTOR_ROOT,
    ensure_output_path,
    read_feature_table,
    save_frozen_bundle,
)
from detector.pipeline import environment_record

PROTOCOL = "strict_unlabeled_three_method_study_v1"
DEFAULTS = {
    "alpha": 0.05,
    "task_size": 150,
    "tasks_per_rate": 100,
    "bootstrap_draws": 499,
    "rank_seed": 20260920,
    "local_seed": 20260922,
    "evaluation_seed": 20260921,
    "neighbors": 5,
    "sample_tail": 0.01,
}
CODE_FILES = (
    "unsupervised_study.py",
    "local_reference.py",
    "rank_reference.py",
    "unsupervised.py",
    "unsupervised_adapt.py",
    "calibration.py",
    "io_utils.py",
    "train_detector.py",
    "common.py",
    "__init__.py",
)


def fingerprint(settings):
    record = {
        "protocol": PROTOCOL,
        "settings": settings,
        "source_code_sha256": {
            name: sha256_file(DETECTOR_ROOT / name) for name in CODE_FILES
        },
    }
    record["fingerprint"] = hashlib.sha256(
        json.dumps(record, sort_keys=True).encode()
    ).hexdigest()
    return record


def fresh_output(output, source=None):
    root = ensure_output_path(output)
    if root.exists():
        raise FileExistsError(
            "Use a new output directory; existing results are preserved."
        )
    if source is not None and (
        root == source or root in source.parents or source in root.parents
    ):
        raise ValueError(
            "Output must be independent of the source, not its parent or child."
        )
    return root


def evaluate_local(
    base_result,
    bundle,
    features,
    metadata,
    progress=False,
    evaluation_role="exploratory_reused",
):
    """Replay exactly the original/modified-view keys already tested by baselines."""
    local.validate_bundle(bundle)
    if base_result["task_size"] != bundle["settings"]["task_size"]:
        raise ValueError("Local and baseline task sizes differ.")
    lookup = features.loc[features["split"] == "test"].set_index(
        ["original_index", "view"]
    )
    result = dict(base_result)
    result["tasks"] = []
    for index, row in enumerate(base_result["tasks"]):
        keys = [
            (identity, row["scenario"] if i < row["modified_count"] else "clean")
            for i, identity in enumerate(row["original_indices"])
        ]
        # Only descriptors cross into prediction, not labels or view identities.
        bag = lookup.loc[keys, local.COLUMNS].reset_index(drop=True)
        prediction = local.predict_dataset(bundle, bag, metadata)
        result["tasks"].append(
            dict(
                row,
                local_status=prediction["status"],
                local_alert=prediction["shift_detected"],
                local_suspicious_count=prediction["suspicious_count"],
            )
        )
        if progress and (index + 1) % 200 == 0:
            print(
                "Local shape detector: {}/{} bags".format(
                    index + 1, len(base_result["tasks"])
                ),
                flush=True,
            )
    result["summary"] = []
    for row in base_result["summary"]:
        bags = [
            x
            for x in result["tasks"]
            if x["scenario"] == row["scenario"]
            and x["requested_rate"] == row["requested_rate"]
        ]
        supported = [x for x in bags if x["local_status"] == "ok"]
        count = sum(bool(x["local_alert"]) for x in supported)
        result["summary"].append(
            dict(
                row,
                local_supported_bags=len(supported),
                local_coverage=len(supported) / len(bags),
                local_alerts=count,
                local_alert_rate=count / len(bags)
                if len(supported) == len(bags)
                else None,
                local_alert_rate_among_supported=count / len(supported)
                if supported
                else None,
            )
        )
    result.update(
        study_protocol=PROTOCOL,
        local_method=local.METHOD,
        local_calibration=bundle["count_calibration"],
        local_limitations=list(local.LIMITATIONS),
        evaluation_role=evaluation_role,
        reused_test_after_method_revision=True
        if evaluation_role == "exploratory_reused"
        else None,
        independent_confirmation_verified=False,
    )
    return result


def report(result, root, config):
    def percent(x):
        return "未判定" if x is None else "{:.1%}".format(x)

    lines = [
        "# 严格无标签方法比较",
        "",
        "研究结果，不是自动判定毕设合格或检测器有效。未修改有监督模型与结果。",
        "",
        "评估角色（用户声明）：{}。模型、阈值及特征选择不读取 incoming 标签。".format(
            config["evaluation_role"]
        ),
        "看到测试后的方法比较仅作探索；改变抽样种子或增加 bags 不是独立模型验证。",
        "",
    ]
    if config["synthetic"]:
        lines += ["**人工合成数据：不能用于真实攻击性能结论。**", ""]
    lines += [
        "| 场景 | 实际修改比例 | 原 MMD | 排序关系 | 局部梯度形状 | 排序覆盖率 | 局部覆盖率 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in result["summary"]:
        lines.append(
            "| {} | {} | {} | {} | {} | {} | {} |".format(
                row["scenario"],
                percent(row["realized_rate"]),
                percent(row["legacy_alert_rate"]),
                percent(row["rank_alert_rate"]),
                percent(row["local_alert_rate"]),
                percent(row["rank_coverage"]),
                percent(row["local_coverage"]),
            )
        )
    cal = result["local_calibration"]
    lines += [
        "",
        "clean 行是误报，poison 行是检出，random_control 行是非目标扰动告警。三种方法使用相同批次。",
        "",
        "局部方法：逐样本将五个 stage 梯度范数归一化，计算历史拟合库中的 5 近邻平均距离，先标记样本，再计数。",
        "历史阈值和计数校准使用不同原图；未从 incoming 中提取假定干净子集。",
        "",
        "计数校准历史样本数：{}；观察告警：{}；每批 {} 张，至少 {} 个样本告警才触发。".format(
            cal["calibration_original_images"],
            cal["calibration_sample_alarms"],
            cal["task_size"],
            cal["critical_suspicious_count"],
        ),
        "这是历史分布下的工作模型，不能视为新任务的误报保证；历史任务异质性及反演样本相关性也会影响校准。",
        "",
        "## 如何下结论",
        "",
        "- 原 MMD：检查域差异是否仍导致高误报。",
        "- 排序方法：检查降低误报是否以低比例漏检为代价。",
        "- 局部方法：检查样本级证据能否提高低比例检出，同时报告 clean 与 random_control 误报。",
        "- 三种方法单独报告，不根据这份 test 自动选择最优方法或组合规则。",
        "- 若局部方法失败，也保留结果：不能再以同一 test 不断挑特征、调整阈值。",
        "- 少于多个不同模型/攻击实验时，跨种子稳定性仍未验证。",
        "",
        "## 明确的失败边界",
        "",
    ] + ["- " + x for x in local.LIMITATIONS]
    if not cal["detection_possible_at_this_size"]:
        lines += ["", "**当前历史校准无法在此批次大小产生告警，不能把零误报当成功。**"]
    lines += [
        "",
        "方法及统计假设见 detector/UNSUPERVISED_THESIS.md；逐历史任务留出结果见 historical_audit.json。",
        "",
    ]
    (root / "report.md").write_text("\n".join(lines), encoding="utf-8")


def run(args):
    source = Path(args.source_run).resolve()
    root = fresh_output(args.output_dir, source)
    settings = dict(DEFAULTS)
    settings.update(
        task_size=args.task_size,
        tasks_per_rate=args.tasks_per_rate,
        bootstrap_draws=args.bootstrap_draws,
    )
    rank.integer(settings["task_size"], "task_size", 16)
    rank.integer(settings["tasks_per_rate"], "tasks_per_rate", 1)
    if settings["task_size"] > 256:
        raise ValueError(
            "Study batch size is capped at 256 for all methods; no hidden subsampling."
        )
    protocol = fingerprint(settings)
    if args.protocol:
        expected = json.loads(Path(args.protocol).read_text())
        if expected != protocol:
            raise ValueError(
                "Code/settings changed after protocol registration. Do not silently update an independent study."
            )
    if args.evaluation_role == "prospective_declared" and not args.protocol:
        raise ValueError(
            "Prospective evaluation requires a protocol saved before new experiments."
        )
    refpath, targetpath = (
        source / "reference_features.csv",
        source / "predicted_features.csv",
    )
    reference, meta = read_feature_table(refpath)
    if (
        not targetpath.is_file()
        or not targetpath.with_suffix(".metadata.json").is_file()
    ):
        raise FileNotFoundError(
            "Need complete predicted features and metadata from the server."
        )
    # Fit all models only on historical references, before reading benchmark values.
    # The old MMD fit uses its original default settings, not a supervised artifact.
    baseline = legacy.fit_reference(reference, meta)
    ranked = rank.fit_reference(
        reference,
        meta,
        alpha=settings["alpha"],
        bootstrap_draws=settings["bootstrap_draws"],
        seed=settings["rank_seed"],
    )
    local_settings = dict(
        task_size=settings["task_size"],
        neighbors=settings["neighbors"],
        sample_tail=settings["sample_tail"],
        alpha=settings["alpha"],
        seed=settings["local_seed"],
    )
    localized = local.fit_reference(reference, meta, **local_settings)
    if args.dry_run:
        print(
            "All three historical-only fits feasible; no benchmark values read and no files written."
        )
        print(
            "Local critical count:",
            localized["count_calibration"]["critical_suspicious_count"],
        )
        return
    paths = [
        refpath,
        refpath.with_suffix(".metadata.json"),
        targetpath,
        targetpath.with_suffix(".metadata.json"),
    ]
    hashes = {p.name: sha256_file(p) for p in paths}
    root.mkdir(parents=True)
    config = dict(
        protocol=protocol,
        source_run=str(source),
        source_sha256=hashes,
        evaluation_role=args.evaluation_role,
        synthetic=bool(meta.get("synthetic")),
        input_provenance=meta,
        supervised_artifacts_changed=False,
    )
    save_json(config, root / "run_config.json")
    save_json(environment_record(), root / "run_environment.json")
    state = {}

    def stage(name, action, filename=None):
        state[name] = {
            "status": "running",
            "started_utc": datetime.now(timezone.utc).isoformat(),
        }
        save_json(state, root / "run_state.json")
        try:
            result = action()
            if filename:
                save_json(result, root / filename)
        except BaseException:
            state[name]["status"] = "failed"
            save_json(state, root / "run_state.json")
            raise
        state[name]["status"] = "complete"
        save_json(state, root / "run_state.json")
        return result

    def freeze():
        for name, model in (
            ("legacy", baseline),
            ("rank", ranked),
            ("local", localized),
        ):
            save_frozen_bundle(model, root / "models" / name)

    stage("freeze_all_models", freeze)
    stage(
        "historical_audits",
        lambda: {
            "rank": rank.historical_audit(ranked),
            "local": local.historical_audit(reference, meta, **local_settings),
        },
        "historical_audit.json",
    )
    features, targetmeta = stage(
        "load_benchmark_after_freeze", lambda: read_feature_table(targetpath)
    )
    config["replication_identity"] = {
        "checkpoint": targetmeta["checkpoint_sha256"],
        "inversions": targetmeta["inversion_sha256"],
        "attack": targetmeta.get("input_sha256", {}).get("attack"),
        "feature_table": targetmeta["features_sha256"],
    }
    save_json(config, root / "run_config.json")
    base = stage(
        "baseline_evaluation",
        lambda: adaptation.evaluate_benchmark(
            ranked,
            baseline,
            features,
            targetmeta,
            task_size=settings["task_size"],
            tasks_per_rate=settings["tasks_per_rate"],
            seed=settings["evaluation_seed"],
            progress=True,
        ),
    )
    result = stage(
        "three_method_evaluation",
        lambda: evaluate_local(
            base, localized, features, targetmeta, True, args.evaluation_role
        ),
        "evaluation_metrics.json",
    )
    stage("report", lambda: report(result, root, config))

    def verify_sources():
        if any(sha256_file(source / name) != digest for name, digest in hashes.items()):
            raise RuntimeError(
                "Source files changed during study; do not interpret the result."
            )

    stage("verify_sources_unchanged", verify_sources)
    print("Study finished:", root)
    print(
        "Method comparison completed, NOT automatic proof of detector effectiveness or thesis acceptance."
    )


def summarize(run_dirs, output):
    root = fresh_output(output)
    runs, identities = [], set()
    for directory in run_dirs:
        directory = Path(directory).resolve()
        if root == directory or root in directory.parents or directory in root.parents:
            raise ValueError(
                "Summary output must not contain or be contained by an input run."
            )
        state = json.loads((directory / "run_state.json").read_text())
        required = {
            "freeze_all_models",
            "historical_audits",
            "load_benchmark_after_freeze",
            "baseline_evaluation",
            "three_method_evaluation",
            "report",
            "verify_sources_unchanged",
        }
        if not required.issubset(state) or any(
            v["status"] != "complete" for v in state.values()
        ):
            raise ValueError("Incomplete study: {}".format(directory))
        config = json.loads((directory / "run_config.json").read_text())
        metrics = json.loads((directory / "evaluation_metrics.json").read_text())
        identity = config["replication_identity"]
        key = (identity["checkpoint"], identity["inversions"], identity["attack"])
        if key in identities:
            raise ValueError(
                "Repeated model/inversion/attack artifacts are not another independent run."
            )
        identities.add(key)
        runs.append((config, metrics))
    if not runs:
        raise ValueError("Need at least one completed study.")
    if len({c["protocol"]["fingerprint"] for c, _ in runs}) != 1:
        raise ValueError("Do not pool studies with different code or settings.")
    if len({c["evaluation_role"] for c, _ in runs}) != 1:
        raise ValueError(
            "Summarize exploratory and prospectively declared studies separately."
        )
    if len({c["synthetic"] for c, _ in runs}) != 1:
        raise ValueError("Never pool synthetic fixtures with real experiments.")
    keys = [(r["scenario"], r["requested_rate"]) for r in runs[0][1]["summary"]]
    if any(
        [(r["scenario"], r["requested_rate"]) for r in m["summary"]] != keys
        for _, m in runs
    ):
        raise ValueError("Scenario grids differ.")
    rows = []
    for index, (scenario, rate) in enumerate(keys):
        for method in ("legacy", "rank", "local"):
            values = [m["summary"][index][method + "_alert_rate"] for _, m in runs]
            supported = [x for x in values if x is not None]
            rows.append(
                {
                    "scenario": scenario,
                    "requested_rate": rate,
                    "method": method,
                    "run_rates": values,
                    "complete_run_coverage": len(supported) == len(runs),
                    "mean_rate": float(np.mean(supported))
                    if len(supported) == len(runs)
                    else None,
                    "std_across_runs": float(np.std(supported, ddof=1))
                    if len(supported) == len(runs) and len(runs) > 1
                    else None,
                }
            )
    result = {
        "runs": len(runs),
        "distinct_checkpoints": len(
            {c["replication_identity"]["checkpoint"] for c, _ in runs}
        ),
        "distinct_known_attacks": len(
            {
                c["replication_identity"]["attack"]
                for c, _ in runs
                if c["replication_identity"]["attack"]
            }
        ),
        "evaluation_roles": [c["evaluation_role"] for c, _ in runs],
        "synthetic_present": any(c["synthetic"] for c, _ in runs),
        "rows": rows,
        "note": "Means/std are across artifact-distinct runs, not bags. Distinct hashes alone do not prove statistical independence; common images and shared training choices remain. No automatic thesis pass.",
    }
    root.mkdir(parents=True)
    save_json(result, root / "summary.json")
    lines = [
        "# 跨实验汇总",
        "",
        "实验数：{}；不同 checkpoint：{}；不同已知攻击文件：{}。".format(
            result["runs"],
            result["distinct_checkpoints"],
            result["distinct_known_attacks"],
        ),
        "",
        "每个实验等权；std 只在至少两次完整实验时报告。不同文件哈希本身不能证明独立性。",
        "若只有一个 checkpoint，不能声称跨独立模型验证。不得把重复 bags 当作独立种子。",
        "",
        "| 场景 | 请求比例 | 方法 | 跨实验平均告警率 | 跨实验 std |",
        "| --- | ---: | --- | ---: | ---: |",
    ]
    if result["synthetic_present"]:
        lines.insert(2, "**人工合成输入：不是实际投毒实验结果。**")

    def fmt(x):
        return "未验证/缺失" if x is None else "{:.1%}".format(x)

    for row in rows:
        lines.append(
            "| {} | {:.0%} | {} | {} | {} |".format(
                row["scenario"],
                row["requested_rate"],
                row["method"],
                fmt(row["mean_rate"]),
                fmt(row["std_across_runs"]),
            )
        )
    (root / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result


def plan(output, seeds):
    """Generate explicit GPU commands only. No downloads/training are launched."""
    root = fresh_output(output)
    seeds = [rank.integer(seed, "seed", 0) for seed in seeds]
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Provide distinct new experiment seeds.")
    root.mkdir(parents=True)
    protocol_path = root / "protocol.json"
    save_json(fingerprint(DEFAULTS), protocol_path)
    commands = [
        "set -e",
        "cd " + shlex.quote(str(DETECTOR_ROOT.parent)),
        "export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1",
        "export PYTHONPATH="
        + shlex.quote(str(DETECTOR_ROOT.parent))
        + '"${PYTHONPATH:+:$PYTHONPATH}"',
    ]
    results = []
    for seed in seeds:
        artifacts, source, result = (
            root / (name + "_seed" + str(seed))
            for name in ("artifacts", "source", "study")
        )
        results.append(str(result))
        checkpoint, inversions = (
            artifacts / "victim/checkpoint.pkl",
            artifacts / "inversions",
        )
        common = [
            "--checkpoint",
            str(checkpoint),
            "--inversion-dir",
            str(inversions),
            "--label-mode",
            "predicted",
        ]
        definitions = [
            ("bootstrap", ["--seed", str(seed), "--output-dir", str(artifacts)]),
            (
                "create_manifest",
                [
                    "--checkpoint",
                    str(checkpoint),
                    "--output",
                    str(source / "manifest.csv"),
                    "--split-seed",
                    "20260720",
                ],
            ),
            (
                "extract_features",
                common
                + [
                    "--reference-only",
                    "--output",
                    str(source / "reference_features.csv"),
                ],
            ),
            (
                "extract_features",
                common
                + [
                    "--artifact",
                    str(artifacts / "attack/noise.pkl"),
                    "--manifest",
                    str(source / "manifest.csv"),
                    "--output",
                    str(source / "predicted_features.csv"),
                ],
            ),
            (
                "unsupervised_study",
                [
                    "run",
                    "--source-run",
                    str(source),
                    "--output-dir",
                    str(result),
                    "--protocol",
                    str(protocol_path),
                    "--evaluation-role",
                    "prospective_declared",
                ],
            ),
        ]
        for module, arguments in definitions:
            if module == "bootstrap":
                commands.append("cd " + shlex.quote(str(DETECTOR_ROOT.parent)))
            elif module == "create_manifest":
                # Upstream CIFAR loaders use ./data. Reuse the bootstrap data
                # beneath detector/, never create data in the repository root.
                commands.append("cd " + shlex.quote(str(artifacts)))
            commands.append(
                shlex.join(
                    [sys.executable, "-B", "-u", "-m", "detector." + module] + arguments
                )
            )
    commands.append(
        shlex.join(
            [
                sys.executable,
                "-B",
                "-m",
                "detector.unsupervised_study",
                "summarize",
                "--runs",
            ]
            + results
            + ["--output-dir", str(root / "summary")]
        )
    )
    (root / "commands.sh").write_text("\n".join(commands) + "\n", encoding="utf-8")
    print("Plan only; no training started:", root / "commands.sh")
    print(
        "Review cost and paths, then run this script on the GPU server. Do not change the registered method after seeing new results."
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    experiment = commands.add_parser("run")
    experiment.add_argument("--source-run", required=True)
    experiment.add_argument("--output-dir", required=True)
    experiment.add_argument("--task-size", type=int, default=150)
    experiment.add_argument("--tasks-per-rate", type=int, default=100)
    experiment.add_argument("--bootstrap-draws", type=int, default=499)
    experiment.add_argument(
        "--evaluation-role",
        choices=("exploratory_reused", "prospective_declared"),
        default="exploratory_reused",
    )
    experiment.add_argument("--protocol")
    experiment.add_argument("--dry-run", action="store_true")
    collection = commands.add_parser("summarize")
    collection.add_argument("--runs", nargs="+", required=True)
    collection.add_argument("--output-dir", required=True)
    preparation = commands.add_parser("plan")
    preparation.add_argument("--output-dir", required=True)
    preparation.add_argument("--seeds", type=int, nargs="+", default=[1, 2])
    args = parser.parse_args()
    if args.command == "run":
        run(args)
    elif args.command == "plan":
        plan(args.output_dir, args.seeds)
    else:
        summarize(args.runs, args.output_dir)


if __name__ == "__main__":
    main()
