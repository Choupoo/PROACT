# 教授确认方法后的扩展实验

教授已经接受当前方法，剩下的是不同 CL 方法和攻击、特征消融，以及结果分析。
本入口先完成 Split CIFAR-100 的实验，不自动扩展 miniImageNet 或 tinyImageNet。
这是当前范围的实现，不代表教授已经确认只做一个数据集即可毕业。

**运行脚本准备好不等于实验完成。** 必须在 CUDA 服务器运行，检查最终状态，再根据真实输出写结论。
CPU 单元测试只检查数学、数据隔离和流程，不能代替真实实验。

## 1. 固定的实验协议

| 项目 | 设置 |
| --- | --- |
| 数据集 | Split CIFAR-100，固定类别顺序，10 个任务，每任务 10 类 |
| CL 方法 | EWC、MAS、RWalk、AFEC-EWC、ANCL-EWC |
| 新实验 seeds | 默认 6、7、8；请先确认服务器没有用这些编号生成本轮结果 |
| 源任务 | Task 1；仅用其 training split 拟合 StandardScaler 和逻辑回归 |
| 校准任务 | Task 1、Task 4 的 clean validation / reserve split |
| 最终任务 | Task 9；不拟合、不校准、不选择阈值或特征 |
| 源任务正类 | reckless BrainWash，L∞ 预算 0.3 |
| 源任务负类 | 对应的原始真实图像；随机扰动不作训练负类 |
| 目标攻击 | reckless / cautious × L∞ 预算 0.1 / 0.3，共四种配置 |
| 监督特征 | 80 个 backbone 参数张量梯度范数 + entropy、confidence、margin、activation norm |
| 梯度目标 | 模型预测类别；不读取真实图片类别，不使用 true-class probability |
| 阈值规则 | 固定 `history_mad`；另外两种规则仅作对照 |
| 数据集评估 | 每 bag 150 张不同原图，每污染比例重复 100 次 |
| 无监督 | 冻结历史 Task 0–8 inversion references，重评估 Rank 与原 MMD |
| 遗忘对照 | 在同一 victim checkpoint 上继续学习完整 clean / BrainWash / uniform 任务 |

默认有 15 个 method/seed 单元，每个单元有四种目标攻击，即 60 个目标配置。
每个单元还单独生成 Task 1、Task 4 的校准攻击，总共 90 次攻击生成。
Victim 训练、反演和特征提取成本较高；不要把整个矩阵当成单配置 pilot 的耗时。

每种 CL 方法/seed 都有自己的 source detector，不是把 EWC 的 LR 直接应用到其他方法。
同一个 method/seed 的四种目标攻击使用同一冻结检测器，测试从源任务 reckless 0.3 到其他攻击配置的迁移。
不同 seed 也重新训练 victim、生成攻击和拟合检测器，不只是改变 bags 的采样顺序。

固定的是阈值**规则**，不是所有模型都用 0.496810 / 4 这两个 pilot 数值。
每个模型使用自己的 Task 1/4 历史纯净数据计算数值阈值，然后冻结。
即使 `history_mad` 在某个开发配置未达到 5% 误报条件，也保留并记录失败，不切换到更好看的候选。
另外两种规则的 Task 9 结果可以比较，但不能据此替换报告方法。

CL 正则化参数是现有实现中的显式 benchmark 设置：

| CLI 方法名 | λ | λ_emp |
| --- | ---: | ---: |
| ewc | 500000 | 不使用 |
| mas | 10 | 不使用 |
| rwalk | 1 | 不使用 |
| afec_ewc | 500000 | 100 |
| ancl_ewc | 500000 | 100 |

Victim 每任务 100 epochs，反演 2,000 iterations，攻击 5,000 epochs。
cautious 的 `w_cur` 为 1。训练学习率 0.01、batch size 16，其余攻击设置沿用已确认的 pilot。
本实验覆盖 BrainWash 主表的方法/攻击轴，但不声称复现论文所有数据集、最优超参数或表中的原始数值。
若 clean CL 性能过低，需要在分析中说明，不能看过目标检测结果后偷偷调 λ 重跑并替换失败记录。

## 2. 消融怎么做

所有版本使用相同的 source 原图 split、训练正负类和 target bags。
每个版本重新拟合自己的 scaler 和 LR，并用相同历史 split 和固定 MAD 规则重新校准。
不把完整模型的数值阈值直接套到删除特征后的分数上。

| 版本 | 保留特征 | 在当前 backbone 下的维度 |
| --- | --- | ---: |
| inference_full | 参数张量范数 + 四个 context 特征 | 84 |
| inference_gradients | 仅参数张量范数；移除全部 context | 80 |
| inference_context | 仅四个 context；移除全部梯度范数 | 4 |
| inference_stage | stage 范数 + context；粒度对照，不是完整模型的纯删除消融 | 9 |
| inference_without_entropy | 完整模型移除 entropy | 83 |
| inference_without_confidence | 完整模型移除 confidence | 83 |
| inference_without_margin | 完整模型移除 margin | 83 |
| inference_without_activation_norm_l2 | 完整模型移除 activation norm | 83 |
| inference_without_uncertainty | 移除 entropy、confidence、margin | 81 |

以上消融针对监督检测器。Rank 保持已确认的 10 描述符和 499 次 bootstrap，不根据目标结果重选描述符。
消融比较包括 AUC、sample clean FPR、poison TPR、dataset clean FPR 和各污染数量的检出率。
单个配置的 AUC 几乎不变，不足以证明某个特征在所有配置里都不必要。

## 3. 服务器启动

先同步更新后的 detector 源码，保留旧 work/results；不要覆盖服务器上的实验产物。
以下路径是你之前日志里的位置。若项目已移动，只替换 `cd` 的路径。

```bash
cd /home/p.zhang/PROACT
conda activate proact38
python -B -m detector.doctor --require-cuda
python -B -m unittest detector.tests.test_advisor detector.tests.test_advisor_final -q

# 只注册计划，不训练；重复执行会校验同一计划，不覆盖。
bash detector/run_advisor_final.sh plan

# 只检查一单元的全部命令，不下载、不训练。
bash detector/run_advisor_final.sh run --method ewc --seed 6 --dry-run
```

计划在 `detector/work/advisor_final_v1/plan.json`。注册必须在服务器进行，
子命令使用注册/运行时 Python 的实际路径，不会包含本机 Mac 的 Python 或项目路径。
Python 3.8 语法兼容；仍需现有可用的 proact38 CUDA 环境。
不要盲目安装通用 requirements.txt 或升级服务器的 PyTorch/torchvision。

### 单块 V100：顺序跑完整计划

```bash
tmux new -s detector-final
cd /home/p.zhang/PROACT
conda activate proact38
export CUDA_VISIBLE_DEVICES=0
bash detector/run_advisor_final.sh run
```

已在 tmux 中就不需要再创建嵌套会话。
`Ctrl+B` 后按 `D` 只分离会话；不用输入 `exit`。
完整顺序运行结束后自动生成汇总。每个阶段单独留日志，失败时脚本立即停止，不假装整轮已经完成。

### 多 GPU：按 method/seed 拆分

先完成 `plan`，再启动不同单元。例如，两张 GPU 分别跑：

```bash
CUDA_VISIBLE_DEVICES=0 bash detector/run_advisor_final.sh run --method ewc --seed 6
CUDA_VISIBLE_DEVICES=1 bash detector/run_advisor_final.sh run --method mas --seed 6
```

这两条命令在不同 tmux 会话/窗口执行；其他 method/seed 同理。
同一单元不要重复启动，文件锁会拒绝；不同单元可并行。
也可以 `--method ewc` 顺序完成该方法的三个 seed。
所有 worker 都结束后，再统一执行汇总，避免一边运行一边把不完整结果当成最终报告。

### 更换 seeds 或输出目录

```bash
export DETECTOR_FINAL_OUTPUT=detector/work/advisor_final_v2
bash detector/run_advisor_final.sh plan --seeds 9 10 11
bash detector/run_advisor_final.sh run
```

这只是编号模板，不保证这些 seed 未被使用过。默认目录的已有计划不能直接改成其他 seeds。
若代码、参数或环境已变，建立新实验目录，不修改 plan/hash 来绕过校验。

### 中断恢复

正常重复运行会复用并校验已经完成的阶段。
未完成阶段先检查对应日志，再执行：

```bash
bash detector/run_advisor_final.sh run --method ewc --seed 6 --retry-failed
```

该阶段的独占产物会移入 `interrupted/`，从该阶段开头重跑。
这是阶段级恢复，不是训练 epoch 内断点续训；旧产物可恢复，数据集不会被删除。

## 4. 汇总和打包

全部单元完成后：

```bash
bash detector/run_advisor_final.sh summarize
python -B -c 'import json; d=json.load(open("detector/work/advisor_final_v1/analysis/summary.json")); print("status:", d["status"]); print("missing:", d["missing"])'
bash detector/run_advisor_final.sh package
```

只把 `status: complete`、`missing: []` 当作计划中全部阶段完成。
`complete` 不表示效果一定达到目标，也不表示已通过教授验收。
打包前会重建汇总、检查注册源码和已完成产物哈希；默认拒绝把不完整运行打包成最终结果。
包在 `detector/work/advisor_final_v1_analysis_<UTC时间>.tar.gz`，命令会打印完整路径。

遇到失败，需要把现有诊断资料发来时：

```bash
bash detector/run_advisor_final.sh package --allow-incomplete
```

包里会明确标记 incomplete。包含 plan、analysis、状态/环境/日志、冻结配置 JSON、
预测和 bags CSV、特征 metadata/manifest、无监督结果、历史描述符和小型 accuracy matrices。
不打包数据集、checkpoint、noise、完整特征矩阵或 joblib 模型。
附带每个导出文件的 SHA-256 清单；这是分析包，不是完整复现归档。

## 5. 结果位置和分析重点

- `analysis/summary.json`：完成状态、逐配置结果、多 seed 均值/标准差、dataset-rate 汇总、配对消融差值。
- `analysis/report.md`：完整模型、bag 检出率、Rank/MMD、遗忘对照和消融表。
- `analysis/supervised.csv`、`paired_ablation_deltas.csv`：可进一步绘图的监督指标。
- `<method>/seed<seed>/frozen_supervised/`：固定阈值规则、历史校准、各消融的特征清单和模型。
- `<method>/seed<seed>/task9/<attack>/evaluation/`：逐样本预测、逐 bag 判断和无监督结果。
- `<method>/seed<seed>/task9/<attack>/effectiveness/`：clean / poison / uniform 的 accuracy matrices。
- `<method>/seed<seed>/logs/`：每阶段日志。

分析要同时回答：排序是否有效、冻结阈值是否迁移、少量投毒是否能检出、哪些特征贡献稳定、
Rank 的历史参考是否会对纯净新任务产生误报，以及攻击是否确实造成遗忘。
不同 bags 复用原图和同一攻击，不用它们的数量制造独立样本量或显著性结论。
随机扰动可能有害，告警不是自动等于误报；只有 unmodified clean 上的告警才计作 clean FPR。
遗忘对照是完整任务的继续训练，不证明筛除告警图片能改善学习。
真实输出回来后还需要逐项分析并写讨论；生成一个汇总文件不能替代这一步。
