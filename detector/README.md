# 毕设 Detector：完整研究流程与运行指南

## Meeting 3 新入口（2026-09-24）

新增特征解释/消融及“早期任务训练 → 冻结后检测后期任务”的独立研究流程。
它是源任务有监督、目标任务无投毒标签的迁移方法，不是严格全无监督，也不覆盖旧结果。
见 [研究协议](MEETING3_PROTOCOL.md)、[完整服务器命令](MEETING3_RUN.md)、
[教授邮件草稿](MEETING3_EMAIL_DRAFT.md)。真实 GPU 结果仍需运行验证。

---

本目录实现教授邮件/会议提出的研究路线：三特征基线 → 细粒度梯度和层分析 → 不确定性/激活特征 → 数据集级检测 → 不依赖 incoming task 投毒标签的参考分布检验。

**实现完成不等于研究假设已验证。** 真实效果必须使用你的 checkpoint、反演样本、BrainWash artifact 重跑。无标签方法是待验证的分布异常检测方案，不能把异常或 p-value 叫作“投毒概率”。CPU demo 只验证代码链路。

所有新代码、日志、数据、模型、报告限定在 `PROACT/detector/`。上游代码只读调用；输出检查目录边界并拒绝含符号链接的既有输出子树。历史 `results/` 保留不变，不代表当前协议的性能。

严格无标签历史参考适配的新研究入口见
[UNSUPERVISED_ADAPTATION.md](UNSUPERVISED_ADAPTATION.md)。
该入口独立运行 `detector.unsupervised_adapt`，保留有监督结果和原 MMD 对照不变。
它是待验证的排序关系检测候选方法，不代表已解决真实实验中的无监督失败。

无监督部分收敛为三方法对照及独立模型/攻击复核的入口：
[UNSUPERVISED_THESIS.md](UNSUPERVISED_THESIS.md)。新 `detector.unsupervised_study`
只增加局部梯度形状候选方法与研究流程，不更改有监督结果，不自动宣布检测有效或毕设验收通过。

## 2026-09-20：针对真实结果的修订及重评估

旧实验的数据集级 clean FRR 为 19.1%，历史参考无标签检验对所有 clean bags
也产生告警。代码修订不等于这些真实指标已经改善：必须重跑并同时检查误报和检出率。

- 新拟合的数据集模型默认采用 `count_bound`：冻结样本检测器和样本阈值，
  对独立 calibration 池的每张 clean 原图只计数一次，用单侧 Clopper–Pearson
  上界估计样本误报概率，再以二项尾概率设置 incoming 数据集的可疑样本数阈值。
  将目标错误预算等分给估计误差与检验尾部；重复抽取的 bags 不增加校准样本量。
  该界要求校准与 incoming clean 原图独立、具有相同误报概率，不保证域偏移或
  相关样本下的误报率。保守阈值可能损失低投毒率检出能力；无法检出时会明确报告。
  计算依据见 [Clopper–Pearson 区间](https://docs.scipy.org/doc/scipy-1.15.3/reference/generated/scipy.stats._result_classes.BinomTestResult.proportion_ci.html)
  和 [二项分布尾概率](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.binom.html)。
- LR 风险分数和旧阈值作为对照保留，旧 bundle 仍按原 LR 规则预测。
  随机扰动对照的高告警率不能由此宣称已经解决。
- 新增 `reference_audit`，仅用 validation clean 数据诊断历史合成参考与新任务
  特征的差异；不以 test 调整 MMD 阈值。保留原始 MMD 告警及其误报统计。
  无标签预测明确返回 `poisoning_decision: undetermined`、`deployment_action: abstain`，
  不能自动接受或拒绝数据。这是诊断和使用边界修正，不是无标签检测能力已恢复。
- CSV 加载同时检查哈希、声明的行数和特征缺失值；不完整传输应重新获取原文件，
  不能修改元数据或哈希绕过检查。

同步整个更新后的 `detector` 代码到服务器，保留服务器已有 `work/` 产物。
在原 tmux 会话也可以运行以下命令；先确认没有另一个实验正在使用同一输出目录：

```bash
conda activate proact38
cd /home/p.zhang/PROACT
python -B -m unittest discover -s detector/tests -q

# 只验证输入并打印命令，不创建结果目录。
python -B -m detector.reassess \
  --source-run detector/work/full_v2_seed0_proact38 \
  --output-dir detector/work/reassessment_count_v1 --dry-run

# 复用已有特征和冻结模型，不重跑 PROACT、反演、攻击或 GPU 特征提取。
python -B -u -m detector.reassess \
  --source-run detector/work/full_v2_seed0_proact38 \
  --output-dir detector/work/reassessment_count_v1
```

源目录必须包含完整的三份特征 CSV 及 metadata、sample/unsupervised 模型及校验文件、
`run_config.json`；只有汇总报告不足以重评估。输出目录必须是全新目录，旧结果不覆盖。
重点查看新目录内 `report.md`、`dataset/fit_metrics.json`、`dataset_test/rates.csv`、
`reference_audit.json` 和 `unsupervised_test.json`。
本次方案修订发生在观察旧 test 后，复用该 test 的结果只能作为诊断，不能当作独立确认；
最终结论仍需要新的模型/攻击种子或独立数据验证。
使用可信干净新任务参考的单类方案会改变方法假设，本次没有自动加入。

## 1. 教授要求与实现

| 要求 | 实现 | 结果 |
| --- | --- | --- |
| 明确三个基线特征的计算 | `extract_features.py` | 下文公式及 `*.metadata.json` |
| 更细粒度范数、哪些层有区分力 | 参数张量/模块/五个 stage 的梯度范数 | `analysis/feature_comparison_validation.csv` |
| 各历史任务参考方向、min/max/mean | 当前样本一个梯度，对比 Task 0–8 九个参考方向 | 特征 CSV、`*.references.pt` |
| uncertainty / activation | entropy、confidence、target probability、margin、激活范数 | 固定 16 维特征 |
| dataset-level detection | 独立 clean 原图计数校准；保留数据集 LR 和 top-tail 对照 | `dataset_test/rates.csv` |
| 无新任务投毒标签 | 历史反演参考 + 伪标签特征 + RFF-MMD 置换检验 | `unsupervised_test.json` |
| 可复现且说得清楚 | 原图级分区、固定种子、哈希、冻结模型、独立测试、随机扰动对照 | `run_config.json`、`report.md` |

## 2. 先验证环境和代码

所有命令从仓库 **`PROACT` 目录**运行，`python` 应是同一个环境的解释器：

```bash
cd /你的路径/Unipi-Thesis-2026-PanZhang/PROACT
export PYTHONDONTWRITEBYTECODE=1

# 不下载数据、不需要 GPU 的单元测试。
python -B -m unittest discover -s detector/tests -v

# 人工合成特征的完整 CPU 演示，不是论文性能实验。
python -B -m detector.demo --output-dir detector/work/cpu_demo_new

# 真实实验环境诊断。
python -B -m detector.doctor --require-cuda
```

演示报告：`detector/work/cpu_demo_new/report.md`。再次运行请换一个空目录，不覆盖冻结模型。测试覆盖数学公式、实际 PROACT ResNet 分组、模型/RNG 不变、样本对齐、无标签边界、测试隔离及文件完整性。

环境优先沿用原来能运行 PROACT 的环境；新环境建议 Python 3.10/3.11。先按 [PyTorch 官方安装说明](https://pytorch.org/get-started/locally/) 安装匹配的 torch/torchvision 和适合驱动的 CUDA 构建，再安装其余依赖：

```bash
python -m pip install -r detector/requirements.txt
python -B -m detector.doctor --require-cuda
```

依赖安装是你显式执行的环境操作；项目程序只写 detector。requirements 不强行替换现有 torch/CUDA。CPU demo 不导入 torchvision；真实 CIFAR 提取和上游训练需要兼容的 torchvision。`doctor` 区分数值依赖、torch、torchvision 导入和 CUDA 可用性。

### 已有 Python 3.8 / PyTorch 1.10 实验环境

服务器已完成 PROACT 训练时，不必为已知兼容问题重装环境或重跑 bootstrap。
`common.py` 会检查 `torch.load` 是否支持 `weights_only`；旧版不传该参数，
新版显式传 `False`，两者都将可信 pickle 中的 tensor storage 映射到 CPU。
索引整数性检查采用同类型张量比较，兼容旧版 `torch.equal`，仍拒绝小数及非有限值。
这不是安全反序列化沙箱，只能加载可信实验产物。

同步修复后的代码及测试到服务器后，从 `PROACT` 目录执行：

```bash
python -B -m detector.doctor --require-cuda
python -m pip check
python -B -m unittest discover -s detector/tests -v
python -B -c "from detector.common import load_pickle; load_pickle('detector/work/artifacts/victim/checkpoint.pkl'); print('Checkpoint loaded successfully')"
python -B -u -m detector.pipeline --config detector/config.proact38.json
```

`config.proact38.json` 复用 `work/artifacts/` 中的模型、反演样本、攻击文件与数据，
仅将 detector 输出改到新的 `work/full_v2_seed0_proact38/`；实验参数与默认配置一致。
如果之前使用了自定义材料路径，请同步调整此配置；如果这个输出目录也已有其他版本
运行记录，请改用另一个未使用的 `run_dir`。不要删除旧运行的环境记录以绕过哈希校验。

`doctor` 的 `ready` 仅表示基础依赖检查通过，不保证所有代码路径或 GPU 实验成功。
本地回归测试模拟了旧版接口行为，不等同于在服务器 Python 3.8 / torch 1.10 上
完成真实全流程验证；仍需在目标环境运行上述检查和实验。

## 3. 没有实验材料：从零生成

先查看命令，然后在 NVIDIA GPU 环境启动：

```bash
python -B -m detector.bootstrap --dry-run
python -B -m detector.bootstrap
```

默认依次调用但不修改上游程序：

1. EWC 学习 Task 0–8，每任务 20 epochs，保存 Task 9 训练前 checkpoint。
2. 历史任务各反演 128 张图，10,000 iterations。
3. 生成 Task 9 reckless BrainWash，L∞ 预算 0.3、5,000 epochs。
4. 分别用 clean / poisoned Task 9 训练，验证攻击是否增加遗忘。

这是长时间 GPU 实验，耗时取决于硬件。输出均在 `detector/work/artifacts/`：

```text
data/cifar-100-python/                  # 首次由上游下载
victim/checkpoint.pkl
inversions/inversions_tid_00.npz ... inversions_tid_08.npz
attack/noise.pkl
effectiveness/clean/acc_mat_clean.npy
effectiveness/poison/acc_mat_ours.npy
effectiveness/metrics.json
logs/
```

`effectiveness/metrics.json` 报告历史平均准确率、Task 9 准确率、BWT，以及 clean 减 poison 的历史准确率差。BWT = mean_t<9(A[9,t] - A[t,t])。正差值说明该次攻击降低历史准确率，不是跨种子的显著性结论。

可单独运行 `--stage victim` / `inversion` / `attack` / `effectiveness`。已有目标文件会拒绝覆盖；用 `--stage` 继续未完成阶段，新实验用新的 `--output-dir`。`--attack-epochs` 必须为 100 的正整数倍，确保上游保存最终噪声。bootstrap 的工作目录也在 detector 内，避免上游相对路径写出目录。

## 4. 已有材料：一键完整实验

编辑 `detector/config.example.json`。相对路径**相对于配置文件目录**解析；也可填绝对路径：

- `checkpoint`：Task 0–8 victim。
- `artifact`：与该 victim 完全一致的 `noise.pkl`。
- `inversion_dir`：覆盖 Task 0–8 的九个 `.npz`。
- `data_cwd`：该目录下面必须已有 `data/cifar-100-python/`。已有数据在 `PROACT/data` 时填 `".."`。
- `run_dir`：新的 detector 子目录。

配置默认值与上一节 bootstrap 输出直接匹配。

```bash
# 只打印展开后的命令，不生成文件。
python -B -m detector.pipeline --config detector/config.example.json --dry-run

# 提取、分析、拟合、独立测试、生成中文报告。
python -B -m detector.pipeline --config detector/config.example.json
```

顺序：manifest → ground-truth 特征 → predicted 历史参考 → predicted benchmark → 验证集层分析 → 冻结样本模型 → 冻结数据集模型 → 冻结无标签检验 → 三项独立测试 → 报告。

也可分段运行：

```bash
python -B -m detector.pipeline --config detector/config.example.json --stages prepare
python -B -m detector.pipeline --config detector/config.example.json --stages fit
python -B -m detector.pipeline --config detector/config.example.json --stages evaluate
python -B -m detector.pipeline --config detector/config.example.json --stages report
```

完成步骤会跳过，失败保留日志。配置首次运行时冻结；`run_environment.json` 记录依赖版本和 detector 源码哈希；配置、代码或依赖改变时使用新 `run_dir`。这是实验步骤记录，不是 GPU 训练断点续训器；部分输出已生成而步骤失败时，核查日志、使用新输出目录，不应覆盖冻结模型。

默认 `feature_set="extended"` 是预先固定的 16 维方案，**不会挑选最高 test 分数**。可预先设置 baseline / stages / extended / parameters / layers / all。若观察 validation 后决定特征组，使用下节独立命令保存到新目录，不得根据 test 结果反复选择。

### 输出怎么看

```text
work/full_v2_seed0/
  run_config.json, run_state.json, logs/
  manifest.csv
  features.csv, predicted_features.csv, reference_features.csv
  *.metadata.json, *.references.pt
  analysis/feature_comparison_validation.csv
  sample/detector_bundle.joblib, coefficients.csv, metrics.json
  dataset/dataset_bundle.joblib, fit_metrics.json
  unsupervised/unsupervised_bundle.joblib
  sample_test/metrics.json, test_predictions.csv, random_control_test_predictions.csv
  dataset_test/evaluation_metrics.json, rates.csv, task_predictions.csv
  unsupervised_test.json
  report.md
```

每个 `.joblib` 配一个 `.joblib.sha256.json`，CSV 配同名 `.metadata.json`，不要拆开。校验包含内容哈希、协议、checkpoint、反演文件、head seed 和 label mode。哈希防意外混用，不是签名；只加载可信 pickle/joblib。CUDA tensor pickle 会先映射到 CPU，再按 device 移动。

## 5. 独立运行分析、拟合与评估

已有 pipeline 特征时不必重新提取。分析同时输出单层/单参数张量排名，以及 baseline 分别加一种特征族、extended 分别移除一种特征族的消融比较，全部只用 validation。以下命令也适用于观察 validation 后冻结某个特征组：

```bash
python -B -m detector.train_detector \
  --features detector/work/full_v2_seed0/features.csv \
  --output-dir detector/work/manual/analysis --compare-features

python -B -m detector.train_detector \
  --features detector/work/full_v2_seed0/features.csv \
  --output-dir detector/work/manual/sample --feature-set extended --fit-only

python -B -m detector.dataset_detector fit \
  --features detector/work/full_v2_seed0/features.csv \
  --sample-bundle detector/work/manual/sample/detector_bundle.joblib \
  --output-dir detector/work/manual/dataset --task-size 150

python -B -m detector.train_detector \
  --features detector/work/full_v2_seed0/features.csv \
  --evaluate-bundle detector/work/manual/sample/detector_bundle.joblib \
  --output-dir detector/work/manual/sample_test

python -B -m detector.dataset_detector evaluate \
  --features detector/work/full_v2_seed0/features.csv \
  --bundle detector/work/manual/dataset/dataset_bundle.joblib \
  --output-dir detector/work/manual/dataset_test

python -B -m detector.unsupervised fit \
  --reference-features detector/work/full_v2_seed0/reference_features.csv \
  --output-dir detector/work/manual/unsupervised --alpha 0.05 --permutations 199

python -B -m detector.unsupervised evaluate \
  --features detector/work/full_v2_seed0/predicted_features.csv \
  --bundle detector/work/manual/unsupervised/unsupervised_bundle.joblib \
  --output detector/work/manual/unsupervised_test.json
```

所有模块都有 `--help`；`pipeline --dry-run` 给出完整 manifest 和三种 extraction 命令，避免手动重复配置。paired extraction 需要 CIFAR 位于工作目录的 data/；`--reference-only` 和 `--input-npz` 不重建 CIFAR。

## 6. 最终使用：判断一个新的 incoming dataset

NPZ 输入包含 `images`：N×3×32×32 浮点数组，值域 [0,1]。ground_truth 还需要 `targets`：N 个 task-local 0–9 标签；predicted 完全不读取 targets。每个原图只放一条，不能混放成对对照视图。

**不需要新任务类别标签或投毒标签：**

```bash
python -B -m detector.predict \
  --bundle detector/work/full_v2_seed0/unsupervised/unsupervised_bundle.joblib \
  --checkpoint detector/work/artifacts/victim/checkpoint.pkl \
  --inversion-dir detector/work/artifacts/inversions \
  --input-npz /你的只读输入/incoming.npz \
  --output-dir detector/work/incoming_check --device cuda
```

读取 `prediction.json` 的 `shift_detected`、`p_value`、`mmd_squared_rff` 和限制说明。N≥2；默认 N>256 时固定种子抽 256 条，可能漏掉低比例投毒。结果是分布异常，不是投毒确认。

**有监督风险分数：**替换 bundle 为 `dataset/dataset_bundle.joblib`，换一个空输出目录。默认 ground_truth 模型要求 targets，但不需要 clean/poison 标签；N 必须等于拟合的 task_size（默认 150），不能把该阈值直接用于任意大小数据集。输出 `dataset_poison_score` 和 accept/reject。若需要不使用类别标签的有监督版本，须在 predicted 特征上重新拟合样本及数据集模型，不能直接更换输入语义。

## 7. 数学定义

### 固定模型和三项基线

模型学习了 Task 0–8；Task 9 分类头以固定 head_seed 重新初始化，是**未训练的头**。其 entropy/confidence 不是训练良好、校准过的任务置信度。

θ_b 包含共享 backbone 卷积/BN 的 weight 和 bias，排除所有分类头：

```text
L(x,y) = CE(model(x)[9], y)
g      = ∇_{θ_b} L(x,y)
loss = L; grad_norm_l2 = ||g||₂; grad_cosine_past = cos(g, d_past)

g_t    = (1/N_t) Σ_i ∇_{θ_b} CE(model(x_ti)[t], y_ti)
u_t    = g_t / ||g_t||₂
d_past = normalize(mean_t u_t)
c_t    = cos(g, u_t), t=0,...,8
```

历史反演样本通过各自的历史分类头求梯度。这里先按任务平均样本梯度，再归一化并等权平均任务方向，**不是直接混合所有未归一化梯度**。不足一个 batch 的梯度按实际样本数加权。保存九个 c_t 及其 min/max/mean。

### 扩展特征

固定 16 维 = baseline3 + task-cosine 统计3 + uncertainty4 + activation1 + stage norms5：

- entropy = -Σp log p，自然对数；confidence = max p；true_class_probability = p_y；margin = p_top1 - p_top2。
- activation = 同一次 forward 中 Task 9 分类头输入向量的 L2 范数，临时 hook 用后移除，不进行第二次 forward。
- stage = stem(conv1+bn1)、layer1–4；范数平方相加应等于全 backbone 范数平方。
- `grad_norm_param__<name>` = 每个**参数张量**的 Frobenius/L2 范数，例如某个卷积 weight；`grad_norm_layer__<module>` 合并该模块 weight/bias。不是把几百万个标量的绝对值全部单独作为特征。
- `all` 使用固定 16 维、参数/模块范数和九个 `grad_cosine_task_<id>`；`extended` 只用固定 16 维。元数据记录实际维数、张量形状和分组。

model.eval 固定 BN，autograd.grad 求导，无优化器更新；前后核对所有参数与缓冲区。predicted 模式以固定头 argmax 为 CE target，完全忽略传入标签；此时 true_class_probability 等于 confidence，是伪标签概率。无标签检验不选这个重复特征。

## 8. 数据分区与两条检测路线

默认每类 300 train / 50 validation / 50 test / 100 reserve，合计 3,000 / 500 / 500 / 1,000 原图。先分区再生成 clean、poison、random_control，共 15,000 行；同一原图不会跨区。

扰动用 rnd_idx_train 的逆置换对齐。random_control 保持原始扰动的 L2/L∞ 范数并随机化位置/符号；匹配成立于 clamp(x+δ,0,1)**之前**，之后有效图像扰动可能不同。随机对照不参与拟合/校准。

### 有监督路线

样本 StandardScaler + L2 LR 只拟合 train；阈值只用 validation-clean。`score >= threshold` 保守处理 ties，控制的是校准集观测误报率，不保证 test/总体误报率。

reserve 按原图等分为 500 dataset-fit / 500 dataset-calibration，与样本 train/validation/test 不重叠。每个模拟数据集默认 150 张不同原图，投毒比例 0/10/25/50/100%，记录实际整数投毒数量。聚合 top10%-mean、可疑比例、均值、标准差、中位数、75/90/95 分位数、最大值；在 fit 池拟合数据集 LR，用 calibration 池的纯 clean bags 校准阈值。另报告仅 top-tail mean 的简单基线。

上述 LR/top-tail 是保留的对照路线。新模型默认使用独立 calibration clean 原图的
`count_bound` 判定，详见本页修订说明；可显式用 `--decision-rule legacy_lr` 拟合旧规则。
`dataset_poison_score` 始终是模拟混合分布下的 LR 分类风险分数，不是经过实际部署
分布校准的投毒后验概率，也不是 `count_bound` 的主要判定依据。

### 无标签路线

使用 predicted 历史参考的 10 维描述符：global norm + stage norms5 + entropy/confidence/margin + activation norm。拟合和预测不读取投毒标签、view 或真实类别标签。参考一半拟合变换，一半作为独立检验库（最多 512 条）：

1. 范数取 log1p；仅 reference-fit 拟合 median/MAD，去掉近零尺度特征并固定裁剪范围。
2. 仅 reference-fit 选择 RBF median bandwidth，冻结 128 维 Random Fourier Features。
3. 计算参考库与 incoming 特征均值差平方，作为近似 MMD²。
4. 默认 199 次固定种子置换；p=(1+极端置换数)/(199+1)，以事先给定 alpha=0.05 告警。

依据：[MMD](https://jmlr.org/papers/v13/gretton12a.html)、[Random Fourier Features](https://papers.nips.cc/paper/2007/hash/013a006f03dbc5392effeb8f18fda755-Abstract.html)。有效显著性解释需要可交换性等假设；历史合成参考与新任务真实数据存在域差异，**本实现没有证明假设成立**。必须同时看 clean 误拒、poison 检出、random-control 告警，不能只挑高检出率。

## 9. 向教授汇报时必须保留的边界

- 层排名依据同一 train/validation 上的分类性能。范数大或相关特征的 LR 系数大，不证明该层导致遗忘；大量层比较有选择偏差，需要独立攻击/任务确认。
- 每个 bag 内部原图互异，不同 bags 复用有限原图池；1,000 bags 不是 1,000 次独立实验，不能直接按独立试验计算置信区间。
- 攻击优化看过 Task 9 原图池；隔离的是 detector 的拟合/测试，不声称测试图对攻击者不可见。
- split/head/control/classifier seed 不是独立模型/攻击种子。上游重建数据可能重设 RNG，只改 attack seed 不足以证明独立攻击，应核查新 artifact 的内容哈希。
- 新 checkpoint/历史任务集合需要重新提取和拟合。当前固定 Split CIFAR-100、9 个历史任务、Task 9、ResNet、reckless L∞ BrainWash，不声称支持全部持续学习场景。
- `results/` 是历史 baseline3（ROC-AUC 0.880064），`code/phase2` 是另一旧实验；本次协议 `pretraining_full_v2` 必须重跑，不能换标签后复用旧成绩。
- 真实 GPU 实验和多设定验证尚需运行，才能回答“特征是否提升”“无标签检验能否区分正常新任务和投毒”。实现与合成测试不能代替这些研究结论。
