# 无监督部分：有边界的毕设实验方案

目标是完成可信的方法比较、独立模型/攻击复核和失败边界分析，不是靠反复调旧测试集
得到漂亮数字。是否达到学院/教授的验收标准仍需教授确认。

所有变化仅位于 detector。原 MMD、排序方法、有监督训练代码、既有模型和实验文件
均保留。新的 `unsupervised_study` 不调用有监督拟合或 ground-truth 特征提取。

## 固定的三个方法

| 方法 | 检查的信号 | 主要已知局限 |
| --- | --- | --- |
| 原历史 RFF-MMD | 整批原始描述符分布 | 正常任务差异/反演到真实的域差异会告警 |
| 历史任务 Kendall 排序比较 | 整批两两特征排序关系 | 低比例变化可能被稀释，纯边际变化被忽略 |
| 新增局部梯度形状 kNN + 计数 | 单张图片的层间梯度形状，再统计异常样本数 | 历史形状也可能不能迁移，计数门槛可能漏掉稀疏投毒 |

三者使用同一批次、同一原图、同一 clean/poison/random_control 视图安排。
不根据 test 挑选最好的方法、历史任务或特征，也不自动 OR/AND 合并三个告警。
新增方法只是补充不同的研究假设，不能事先保证优于排序方法。

### 新增局部方法的计算

对每个样本，提取已有的五个 stage 梯度范数 `g`，计算 `z = g / ||g||_2`。
这是**每张图片内部归一化**，不从 incoming 中估计均值、选干净子集或拟合模型。
每张图片梯度整体乘任意正数不改变 z，因此舍弃整体强度，保留层间相对分布。

1. 每个历史任务按固定种子拆成 50% fit、25% threshold、25% calibration，三部分原图互斥。
2. 在历史 fit 形状中求 k=5 个最近邻。异常分数为平均欧氏距离除以 sqrt(2)。
   z 为非负单位向量，分数因此位于 [0,1]。这里不是有监督 kNN 分类。
3. 仅用历史 threshold 部分设定样本阈值，预设经验尾部比例 1%，保守排除边界并列值。
4. 仅用独立历史 calibration 部分统计样本告警；复用计数校准函数，确定固定大小
   incoming 批次的可疑样本数量门槛。没有复用或改动有监督检测器的拟合结果。
5. 推理只计算单张图片分数和整批计数，不需要 incoming 多数干净的假设。
   零梯度导致无法定义形状时明确未判定，不能算正确 clean。

计数函数中的 Clopper–Pearson 上界和二项尾概率是**历史参考下的工作模型**。
不能声称它在新任务上控制 5% 误报：反演样本相关、不同历史任务的告警概率不同、
固定任务配额以及反演/真实差异都可能破坏其假设。真实 clean FPR 必须独立测量。
这不是先前被拒绝的“使用同域可信干净新任务样本”方案；所有拟合材料仍仅来自历史反演。

计算背景：[NearestNeighbors 文档](https://scikit-learn.org/1.0/modules/generated/sklearn.neighbors.NearestNeighbors.html)、
[Clopper–Pearson 区间说明](https://docs.scipy.org/doc/scipy-1.15.3/reference/generated/scipy.stats._result_classes.BinomTestResult.proportion_ci.html)。
代码用 NumPy 直接计算距离；这些资料不是投毒检测有效性或目标域校准的证明。

## A. 先用已有特征完成三方法对照

在服务器同步更新的 detector 代码，**不要删除或覆盖任何旧 work 目录**。
已有 seed0 的特征可复用，不必再训练 PROACT、反演或生成攻击。

```bash
conda activate proact38
cd /home/p.zhang/PROACT
python -B -m unittest discover -s detector/tests -q

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
python -B -u -m detector.unsupervised_study run \
  --source-run detector/work/full_v2_seed0_proact38 \
  --output-dir detector/work/unsupervised_thesis_seed0
```

`--source-run` 指向完整 predicted/reference CSV 所在目录，不能指向只有报告的
`proact-result` 或 `unsupervised_rank_v1`。两份 CSV 的 metadata 必须完整、哈希匹配。
原 MMD 用其原默认设置在历史参考上重建并保存至新目录，不替换原 bundle。

默认每批 150 张，每场景 100 批，预设修改比例 1%、5%、10%、25%、50%、100%。
报告使用舍入后的实际比例（2/150、8/150、15/150、38/150、75/150、150/150）。
新 run 目录必须不存在。这个阶段是已看过测试后的探索性比较，不是独立确认。

输出：

```text
unsupervised_thesis_seed0/
  models/legacy/bundle.joblib + checksum
  models/rank/bundle.joblib + checksum
  models/local/bundle.joblib + checksum
  historical_audit.json       # 两个新方法逐历史任务留出，未使用 incoming clean 标签
  evaluation_metrics.json    # 三方法同批次完整结果，包括 coverage 和局部可疑计数
  report.md
  run_config.json             # 设置、方法代码指纹、输入哈希、模型/攻击身份
  run_environment.json
  run_state.json
```

发回 `report.md`、`evaluation_metrics.json`、`historical_audit.json`、
`run_config.json`、`run_state.json` 即可分析。不要覆盖前两轮结果，论文需要它们作为对照。

## B. 冻结方法，再做小规模独立模型/攻击复核

只改变批次抽样种子不算新实验。补充新的模型训练/攻击种子，例如 seed1、seed2；
两组是小规模复核，不是显著性证明，模型/数据共享和有限种子仍需在论文中说明。
新种子需要真实 GPU 训练、反演和攻击，成本明显高于 A；本工具不会自动启动。

**请在 GPU 服务器上生成计划**，生成的解释器和项目路径属于执行计划的那台机器：

```bash
python -B -m detector.unsupervised_study plan \
  --output-dir detector/work/unsupervised_confirm_v1 \
  --seeds 1 2
```

此命令只生成 `protocol.json` 和 `commands.sh`，不下载数据、不训练。
检查 GPU 资源、路径及所需时间后，才显式执行：

```bash
bash detector/work/unsupervised_confirm_v1/commands.sh
```

脚本依次生成新 checkpoint/反演/攻击与攻击效果验证，然后只提取 predicted reference
和 predicted benchmark 特征，执行三方法比较。上游需要的相对 `data/` 路径限定在
各自 detector/work/.../artifacts_seedN 内，并通过 PYTHONPATH 读取项目代码。
不调用 supervised train_detector 或原 pipeline 的有监督分析阶段。

方法源代码和参数在生成计划时固定；运行时若指纹变化会拒绝继续。
`prospective_declared` 只是用户对事先规划的声明，代码不会据此宣布结果独立有效。
同一 CIFAR 数据源仍可能重复使用图片，应把结果称为不同模型/攻击种子的复核，
不要称为完全独立的数据总体实验。

脚本使用 `set -e`，失败后停止。不要从头覆盖重跑已有材料；根据失败日志查看
`commands.sh`，bootstrap 可用其 `--stage` 继续尚未完成阶段，研究结果则使用新目录。
不要删除旧产物或篡改协议指纹来绕过检查。

## C. 汇总而不是把 bags 当独立实验

上述脚本最后自动汇总两个新实验，也可手动：

```bash
python -B -m detector.unsupervised_study summarize \
  --runs detector/work/unsupervised_confirm_v1/study_seed1 \
         detector/work/unsupervised_confirm_v1/study_seed2 \
  --output-dir detector/work/unsupervised_confirm_summary_v1
```

- 输出 `summary.md`、`summary.json`，每个实验等权报告告警率均值与跨实验 std。
- 一个实验时 std 为缺失，不用重复 bags 伪造标准差或置信区间。
- 同一 checkpoint/inversion/attack 组合重复出现会拒绝计为新实验。
- 不同设置/代码、合成与真实数据、探索性与预先声明实验不能静默混合汇总。
- 不同文件哈希不等于证明独立性；汇总明确报告不同 checkpoint/攻击数量和限制。

## 无监督部分怎样收尾

完成 A、得到 B 的复核结果后，论文应回答三个问题：

1. 直接比较原始分布为什么误报高？证据来自 clean 和历史参考诊断，而不是猜测。
2. 舍弃边际强度能否降低误报，代价是什么？报告排序方法的低、中、高比例检出。
3. 单样本局部形状能否补充低比例证据？同时报告 clean、poison、random_control 和 coverage。

**成功、部分有效或失败，都按实际结果写。** 若第三个方法也无效，不再依靠同一
test 无限制改模型。可以将无监督部分定位为“严格无标签检测方法比较与局限分析”，
而不能称为“已经解决所有投毒检测问题”。教授是否接受这一研究范围，需要单独确认。

不自动挑测试集最高分，不把异常概率叫投毒概率，不用零误报掩盖零检出，也不把
检测准确率当作减少遗忘的防御效果。若论文要声称防御有效，必须另做干预后的训练验证。
