"""Assemble an honest, compact experiment report from completed pipeline outputs."""

import argparse
import json
from pathlib import Path

import pandas as pd

from detector.io_utils import ensure_output_path


def _table(frame, columns):
    frame = frame[columns]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in frame.itertuples(index=False, name=None):
        values = [
            "{:.4f}".format(value)
            if isinstance(value, float)
            else str(value).replace("|", "/")
            for value in row
        ]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def build_report(run_dir):
    root = Path(run_dir)
    lines = ["# Detector 实验报告", ""]
    config_path = root / "run_config.json"
    config = json.loads(config_path.read_text()) if config_path.exists() else {}
    if config.get("synthetic"):
        lines += [
            "**本报告来自人工合成特征，只验证代码链路，不是 CIFAR-100 / BrainWash 实验结果，不得用于论文或简历性能指标。**",
            "",
        ]
    lines += [
        "协议：`pretraining_full_v2`。模型在 incoming task 训练前固定；测试集不参与拟合、特征排序或阈值选择。",
        "",
        "## 1. 教授要求：哪些层更有区分力？",
        "",
    ]
    comparison = root / "analysis" / "feature_comparison_validation.csv"
    if comparison.exists():
        frame = pd.read_csv(comparison)
        lines += [
            "以下按验证集 ROC-AUC 排序（前 15 项）。这是探索性诊断，不是独立测试结果；比较大量层后应在新攻击/任务上确认。",
            "",
            _table(
                frame.head(15),
                [
                    "feature_set",
                    "validation_roc_auc",
                    "validation_poison_tpr",
                    "validation_clean_fpr",
                ],
            ),
            "",
        ]
        broad = frame.loc[
            frame["feature_set"].isin(
                ["baseline", "stages", "extended", "parameters", "layers", "all"]
            )
        ]
        lines += [
            "固定特征组比较：",
            "",
            _table(
                broad, ["feature_set", "validation_roc_auc", "validation_poison_tpr"]
            ),
            "",
        ]
    else:
        lines += ["尚未生成验证集特征比较。", ""]
    sample_path = root / "sample_test" / "metrics.json"
    lines += ["## 2. 固定样本检测器：测试集", ""]
    if sample_path.exists():
        sample = json.loads(sample_path.read_text())
        lines += [
            "特征组：`{}`。以下为固定阈值下的测试指标。".format(
                sample.get("feature_set", "unknown")
            ),
            "",
            "```json",
            json.dumps(
                {
                    "test": sample["test"],
                    "random_control_test": sample["random_control_test"],
                },
                ensure_ascii=False,
                indent=2,
            ),
            "```",
            "",
        ]
    else:
        lines += ["尚未执行冻结模型的测试集评估。", ""]
    lines += ["## 3. 有监督数据集级检测", ""]
    rates_path = root / "dataset_test" / "rates.csv"
    if rates_path.exists():
        rates = pd.read_csv(rates_path)
        lines += [
            _table(
                rates,
                [
                    "alternative_view",
                    "realized_rate",
                    "positive_rate",
                    "top_tail_positive_rate",
                ],
            ),
            "",
        ]
        clean_rates = rates.loc[rates["realized_rate"] == 0, "positive_rate"]
        target = config.get("target_clean_fpr", 0.05)
        if not clean_rates.empty and clean_rates.max() > target:
            lines += [
                "**有监督数据集模型的测试 clean 误拒率超过预设校准目标；校准池上的约束不能保证测试表现，不能据高投毒检出率宣称可靠防御。**",
                "",
            ]
    else:
        lines += ["尚未执行数据集级评估。", ""]
    lines += [
        "`positive_rate` 在纯 clean 数据集上是误拒率，在 poison 数据集上是检出率。random_control 表示同预算随机扰动的告警率。风险分数是模拟混合分布下的分类分数，不是经过部署分布校准的投毒概率。",
        "",
        "## 4. 无投毒标签：参考分布检验",
        "",
    ]
    unsupervised_path = root / "unsupervised_test.json"
    if unsupervised_path.exists():
        result = json.loads(unsupervised_path.read_text())
        lines += [
            _table(
                pd.DataFrame(result["summary"]),
                ["scenario", "actual_modified_fraction", "alert_rate"],
            ),
            "",
            "clean 误拒率：{:.2%}；预设 alpha：{:.2%}。".format(
                result["clean_task_false_rejection_rate"], result["alpha"]
            ),
            "",
        ]
        if result["clean_task_false_rejection_rate"] > result["alpha"]:
            lines += [
                "**观察到的 clean 误拒率高于预设 alpha。不能宣称该方法已实现可靠的投毒识别，应优先检查历史合成数据与新任务真实数据的域差异。**",
                "",
            ]
    else:
        lines += ["尚未执行无标签参考分布检验的测试评估。", ""]
    lines += [
        "检验只回答描述符分布是否偏离参考：p-value 不是投毒概率；正常新类别、自然任务迁移、随机噪声、反演数据与真实数据的差异都可能触发告警。",
        "",
        "## 5. 可以与不可以得出的结论",
        "",
        "- 同一原始图像的三种视图始终属于同一个分区；样本拟合、样本阈值、数据集拟合、数据集阈值和测试使用互斥原图。",
        "- 每个模拟数据集内部不重复原图，但不同模拟数据集会复用有限图像池，因此不能把 bags 数量当成独立实验次数或据此宣称统计显著。",
        "- 需要额外验证攻击确实导致历史任务遗忘：查看 bootstrap 生成的 `effectiveness/metrics.json`。仅区分 clean/poison 特征不等于证明防御有效。",
        "- 单个 Task 9、单个 checkpoint、单个攻击预算的测试只支持该设定内结论；泛化需要新的任务顺序、攻击、预算和独立模型种子。",
        "- 所有超参数应在查看新测试结果之前冻结。没有自动以测试结果重新选择特征或降低阈值的步骤。",
        "",
        "## 方法依据",
        "",
        "分布检验采用 [MMD](https://jmlr.org/papers/v13/gretton12a.html) 的随机特征近似；[Random Fourier Features](https://papers.nips.cc/paper/2007/hash/013a006f03dbc5392effeb8f18fda755-Abstract.html) 用于降低重复置换的计算成本。这里只借用分布检验方法，不声称文献证明其能特异检测 BrainWash。",
        "",
    ]
    return "\n".join(lines)


def main(args):
    root = ensure_output_path(args.run_dir)
    if not root.is_dir():
        raise FileNotFoundError(root)
    output = ensure_output_path(root / "report.md")
    output.write_text(build_report(root), encoding="utf-8")
    print("Saved report:", output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    main(parser.parse_args())
