# 有监督与严格无监督：2026-09-26 的固定对照修订

本轮新增两个研究候选方法，复用服务器已有特征，不重训 PROACT、不重新生成攻击。
代码实现与合成测试不能证明真实误报问题已解决。所有旧模型、结果和文件保留。
新增代码只写新的 detector 子目录，拒绝覆盖，不能恢复到 epoch/评估步骤内部。

## 已观察的真实问题

证据为用户提供的 `meeting3_transfer_v1`，Task 1 → Task 9，seed3/4。
这里全部任务编号为零起始。

| Seed | 特征 | 目标样本 AUC | 目标 clean FPR | 目标纯干净 bag 告警 |
| --- | --- | --- | --- | --- |
| 3 | portable | 0.987932 | 18.6% | 100% |
| 3 | extended | 0.987856 | 24.6% | 100% |
| 4 | portable | 0.961952 | 12.8% | 99.5% |
| 4 | extended | 0.961676 | 12.8% | 99.5% |

样本排序与源域阈值跨域失效可以同时发生。源 clean 计数校准依赖相同误报概率，
不能在目标 clean 样本误报率明显变化后继续声称控制目标 bag 误报。
目前下载包没有完整原始特征，不能断言具体哪个特征的尺度变化导致了失败；
下面的归一化对照检验这一假设，不预先认定它就是根因。

会议中教授建议的早期任务人工攻击训练、后期任务检测属于源任务有监督迁移。
严格无监督仍独立保留历史参考路线，不能把前者改名为全无监督。
会议同时明确无监督没有预设准确率目标，应给出合理的方法、实验和真实结果。

## 两个固定候选方法

### 有监督 shape 对照

设单张图片的五个 stage 梯度范数为 g，输入逻辑回归的特征为 `g / ||g||_2`。
代码先除以该行最大值再归一化，避免极大/极小范数造成溢出。
特征名称是 `grad_shape_stage_stem` 到 `grad_shape_stage_layer4`。

- 仅用源任务 train clean/poison 拟合 StandardScaler 和 Logistic Regression。
- 仍在源 validation-clean 确定样本阈值，在源 reserve-clean 做计数校准。
- 不使用目标样本分布、目标干净子集、目标标签调参，不改变目标阈值。
- 三组 portable、extended、shape 都报告，没有自动挑选胜者。
- 原 `transfer_pipeline` 的注册默认组仍只有 portable、extended。
- 丢弃总梯度强度、激活、不确定性和历史 cosine，可能损失关键攻击信号。
  即使误报下降，也必须检查低比例投毒检出是否同时消失。
- 五个梯度全零时明确失败，不把无法定义的形状判成 clean。

shape 模型支持原 `explain_detector` 的 SHAP，解释前应用相同的确定性变换。
其贡献对应相对 stage 梯度，而不是原始范数。旧 raw-feature 消融不适用于 shape，
因此 shape 的解释命令不加 `--ablations`。

### 严格无监督历史任务 shape-MMD

采用相同的逐样本梯度形状，仅从 predicted 模式的历史反演特征拟合。
每个历史任务分别按固定 seed 将原图分为 kernel-fit 和 permutation-bank 两半。
前者固定 RBF 带宽、64 维随机 Fourier 映射，后者参与两样本置换检验。
incoming 不参与核参数、尺度、参考选择或阈值拟合。

每个历史任务产生一个 p 值，199 次置换，带 `(extreme + 1)/(199 + 1)` 修正。
最终取所有任务 p 值的最大值，最大值不超过 0.05 才告警。
这与“至少存在一个匹配历史任务”的原假设对应，不是事后挑选最好结果。
历史逐任务留出审计排除被留出任务的所有图片，包括核拟合图片。

与旧 MMD 的差异：逐图消除共同梯度幅度、按历史任务分别比较；不是简单地将
完整 backbone embedding 换成 detector 描述符，旧 MMD 已使用 detector 描述符。
与现有 local kNN 的差异：这次检验整批形状分布，不使用历史样本异常计数阈值。

条件和局限：clean incoming 必须与至少一个历史任务的形状参考可交换；反演与真实
图像不一定满足这一条件。反演样本之间也可能相关。整体幅度攻击、低比例投毒、
有限随机映射和保守的 max-p 规则都可能导致漏检。新任务的正常形状变化仍可能误报。
全零梯度返回未判定，coverage 单列，不计为正确 clean。检验输出始终是分布变化，
投毒结论为 `undetermined`，不返回部署投毒概率。

方法依据：[MMD 两样本检验原论文](https://www.jmlr.org/papers/v13/gretton12a.html)、
[max-p 组合检验](https://academic.oup.com/bioinformatics/article/38/1/141/6363784)。
这些是统计方法依据，不是该任务上有效性的证据。

## 服务器运行命令

在服务器同步本轮 detector 源码后执行。原 tmux 会话可用。
保留已有 work 目录。不要重新运行旧 `commands.sh`：那些计划绑定旧代码哈希。
本轮新入口直接只读旧特征，并记录当前代码、环境和输入哈希。

```bash
conda activate proact38
cd /home/p.zhang/PROACT
export PYTHONDONTWRITEBYTECODE=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python -B -m unittest discover -s detector/tests -q
```

### 1. 有监督 seed3/4：优先执行，完全复用现有 CSV

需要服务器原 `meeting3_transfer_v1/seed3` 和 `seed4` 的
`features_task1/features.csv`、`features_task9/features.csv` 及各自 `.metadata.json`。
只有下载的报告包不足以运行。不会读取旧模型或覆盖旧结果；portable/extended 以相同
固定算法重拟合作为对照，其参数默认与第一轮一致。

```bash
python -B -m detector.revision_study supervised \
  --source-run detector/work/meeting3_transfer_v1 \
  --output-dir detector/work/revision_shape_supervised_v1 \
  --seeds 3 4 --dry-run

python -B -u -m detector.revision_study supervised \
  --source-run detector/work/meeting3_transfer_v1 \
  --output-dir detector/work/revision_shape_supervised_v1 \
  --seeds 3 4
```

dry-run 只检查文件是否齐全并计算输入哈希；不证明内容有效或性能已改善。
正式运行会验证 CSV/metadata 哈希和来源，源模型保存后才读取目标特征值。
默认每 bag 150 张、每比例 1,000 bags。CPU 即可运行，不调用 CUDA 训练或数据下载。

可选解释：

```bash
python -B -m detector.explain_detector \
  --features detector/work/meeting3_transfer_v1/seed3/features_task1/features.csv \
  --bundle detector/work/revision_shape_supervised_v1/seed3/source_shape/bundle.joblib \
  --output-dir detector/work/revision_shape_supervised_v1/seed3/explain_shape
```

### 2. 严格无监督：先复用 seed0 完整 predicted/reference CSV

必须使用 predicted 特征和同一 checkpoint/head 的历史 reference 特征。
**不能用 Meeting 3 ground_truth 的 features.csv 替换 predicted_features.csv。**
输入是最初完整特征所在目录，不是 `unsupervised_thesis_*` 等只有研究报告的目录。

```bash
python -B -m detector.revision_study unsupervised \
  --source-run detector/work/full_v2_seed0_proact38 \
  --output-dir detector/work/revision_shape_unsupervised_seed0_v1 \
  --dry-run

python -B -u -m detector.revision_study unsupervised \
  --source-run detector/work/full_v2_seed0_proact38 \
  --output-dir detector/work/revision_shape_unsupervised_seed0_v1
```

默认每 bag 150 张、每比例 100 bags。重新拟合旧 MMD、rank、local 和新 shape 模型到
新目录，然后在完全相同的 bags 上比较四种方法；原模型文件不改变。
历史与目标类别/投毒标签不进入任何方法的拟合或预测，评估器仅用 view 组装实验场景。
保留 0%、1%、5%、10%、25%、50%、100% 投毒及随机扰动对照，记录舍入后的真实比例。
批次可全部投毒，不依赖 incoming 多数干净的假设。

此命令需要 `reference_features.csv`、`predicted_features.csv` 及各自 metadata。
若服务器旧目录名不同，将 `--source-run` 换成这四个文件实际所在目录。
seed1/2 同样可复用各自四个完整文件，必须使用各自独立的新输出目录。
本轮代码不自动启动新的 GPU 训练，也不将已观察过的种子称为新的独立确认。

## 应返回哪些结果

有监督：新根目录 `summary.json`、`summary.md`、`run_config.json`、`run_state.json`，
各 seed 的 `source_shape/fit_metrics.json`、所有 `evaluate_task*/evaluation_metrics.json`
和 `rates.csv`。需要归因时补发 `explain_shape/`。

严格无监督：`evaluation_metrics.json`、`rates.csv`、`historical_audit.json`、`report.md`、
`run_config.json`、`run_state.json`。模型和巨大原始特征不必作为首轮分析附件。

失败时先看终端错误和 run_state；新命令不支持覆盖失败目录，修复原因后换新输出目录。
旧结果始终保留，不能通过修改模型校验文件或旧实验协议让命令强行继续。

评价顺序：输入/协议正确 → source 对照 → target clean 样本与 bag 误报 → 各比例检出
→ 随机扰动特异性 → 多种子稳定性。若新形状方法失败，保留这一负结果，不反复用
同一 test 挑选参数。是否继续独立确认取决于这次结果与教授反馈。
