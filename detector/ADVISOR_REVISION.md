# 教授最新反馈：先修正特征与阈值

这份文档记录之前的单配置方法检查。教授现已认可方法，后续扩展实验请使用
[ADVISOR_FINAL_RUN.md](ADVISOR_FINAL_RUN.md)，不要把下面的 pilot 当成完整实验矩阵。

当前默认入口仅运行 **EWC / reckless BrainWash / L∞ 0.3 / 一个 seed** 的方法检查。
多 CL 方法、多攻击和消融已设为显式的后续选项，默认不会执行。
这不是已经验证误报下降的结果，也不能替代教授对方法的确认。

英文方法说明见 [methodology-03-10-2026.md](results/methodology-03-10-2026.md)。
它解释了标签边界、两个监督阈值、Rank 的直接数据集级检验，以及重复 bags 的构造和依赖性。

## 已改动的逻辑

- 使用预测类别计算损失和 backbone 梯度，不读取真实类别。
- 正式新 CSV 删除 `true_class_probability`；新检测器不选用 loss / historical cosine。
- 保留每个 weight/bias 参数张量一个范数，加 entropy、confidence、margin、activation norm。
- 只在源任务训练，在历史开发任务比较阈值规则，冻结之后才评估最终任务。
- 若所有阈值候选均不满足开发误报条件，明确记录失败，不自动宣称修复。
- 旧实验与报告原样保留。新入口默认不跑无监督重评估、遗忘控制或消融。

## 优先复用已有 artifacts

教授要求先明确方法，**不必为了这次特征修正重训所有 PROACT 模型**。
已有 checkpoint、BrainWash noise.pkl、同模型的 inversion NPZ 可以用于重新提取特征；
旧 ground_truth 特征 CSV 不可复用为 predicted 特征。

下面是每个阶段重新提取的模板。将尖括号项替换成服务器实际路径；task 必须与 checkpoint 匹配。
源任务和开发任务用于拟合／校准，最终任务不可混入它们。

```bash
python -B -u -m detector.transfer_extract \
  --checkpoint <checkpoint.pkl> \
  --artifact <noise.pkl> \
  --inversion-dir <inversions目录> \
  --incoming-task <1或4或9> \
  --label-mode predicted --inference-schema --random-control uniform \
  --output-dir detector/work/feature_threshold_pilot/feature_task<编号>
```

假设已经生成 Task 1、4、9 的新特征：

```bash
python -B -u -m detector.advisor_detector \
  --source detector/work/feature_threshold_pilot/feature_task1/features.csv \
  --histories detector/work/feature_threshold_pilot/feature_task4/features.csv \
  --output-dir detector/work/feature_threshold_pilot/frozen

python -B -u -m detector.advisor_study evaluate \
  --features detector/work/feature_threshold_pilot/feature_task9/features.csv \
  --frozen detector/work/feature_threshold_pilot/frozen \
  --output-dir detector/work/feature_threshold_pilot/evaluation
```

这条路径默认只拟合修正后的完整模型和三种阈值规则，不运行消融。
Task 4 是一个可调整的开发任务选择，不是教授指定编号；必须在看最终结果前确定。
既有 seed 3/4 的 Task 9 已被分析过，因此复用它们是探索性复核，不是全新独立确认。

## 缺少兼容 artifacts 时，才从头运行单配置 pilot

所有命令在服务器实际的 PROACT 根目录执行。以下路径来自之前提供的服务器日志；若位置已变，请替换。
先同步更新后的 detector 代码，不要覆盖旧 work/results。

```bash
cd /home/p.zhang/PROACT
conda activate proact38
python -B -m detector.doctor --require-cuda
python -B -m unittest detector.tests.test_advisor

python -B -m detector.advisor_study plan \
  --output-dir detector/work/feature_threshold_v1

python -B -m detector.advisor_study run \
  --plan detector/work/feature_threshold_v1/plan.json --dry-run
```

`plan` 不训练。默认 seed 5 / source Task 1 / development Task 4 / target Task 9，
100 训练 epochs、2,000 反演迭代、5,000 攻击 epochs。seed 5 是计划默认值，
不保证你从未运行过它。请在开始前确认编号、成本和方法，不要把缩短 epochs 的 smoke test 当作正式结果。
一个 pilot 仍需三个阶段的 artifacts；不提供没有实测依据的耗时承诺。

需要实际运行时，在 tmux 中执行：

```bash
tmux new -s detector-threshold
cd /home/p.zhang/PROACT
conda activate proact38
python -B -u -m detector.advisor_study run \
  --plan detector/work/feature_threshold_v1/plan.json

python -B -m detector.advisor_study summarize \
  --plan detector/work/feature_threshold_v1/plan.json
```

`Ctrl+B` 后按 `D` 只分离 tmux；不要在运行中输入 `exit`。
正常重启同一个 run 命令会检查并复用已完成阶段。中断阶段不会被当作完成：
先查看对应 log，确认需要重做后，给 run 加 `--retry-failed`。
它将该阶段的独占输出移动到 `interrupted/` 留档，再从该阶段开头重跑；
不是训练 epoch 内断点恢复，不删除共享数据集，也不重新执行已经校验通过的阶段。

注册后代码、参数或环境改变会拒绝混合运行。请建立新实验目录，不能修改 plan.json 掩盖变化。

## 分析文件

自动流程输出位于 `detector/work/feature_threshold_v1/`：

- `plan.json`：实验设置与源码校验。
- `ewc/seed5/run_state.json`、`run_environment.json`、`logs/`：完成状态、环境、逐阶段日志。
- `ewc/seed5/frozen_supervised/threshold_selection.json`、`development_metrics.json`、`freeze.json`：阈值选择依据。
- `ewc/seed5/task9/reckless_0p3/evaluation/`：完整模型和三个阈值规则的冻结测试结果。
- `analysis/summary.json`、`report.md`、`supervised.csv`：汇总；未完成的实验会明确标记。

手动复用 artifacts 的流程，返回 `frozen/` 中的上述 JSON、`evaluation/`、
各 Task 的 `features.metadata.json` 即可先分析；必要时再提供 CSV。

Python 3.8 保留兼容写法，但需要现有可用的 PyTorch / torchvision / NumPy / SciPy / scikit-learn 环境。
不要直接在 proact38 中安装本目录面向 Python 3.10+ 的通用 requirements.txt，更不要为了本次修改盲目升级 CUDA/PyTorch。
本地 CPU 合成测试不能证明服务器 GPU 的完整训练链路或真实检测效果。
