# 毕设论文证据与写作提纲

这是基于已有结果的写作框架，不是最终论文或教授验收证明。
本轮收尾实验尚需在服务器运行；所有待填结果标注为待填，不预设改善。

## 建议研究问题

1. 学习新任务之前，梯度/不确定性/激活特征能否区分该任务的 clean 与 BrainWash 样本？
2. 哪些特征/层贡献较大，加入更细粒度特征后表现怎样？
3. 用第二个任务的人工攻击训练检测器，能否迁移到第十个任务？
4. 在不使用 incoming 投毒标签或可信 clean 子集时，历史反演参考能否支持数据集检测？

第3项是源任务有监督、目标任务无投毒标签的迁移；第4项是严格无监督历史参考方法。
两者的监督条件必须分别写明。教授说的“应用时不知道是否投毒”不能自动等同于
整个检测器训练过程中从未使用过任何人工投毒标签。

## 论文结构与可用证据

| 内容 | 实验/产物 | 写作边界 |
| --- | --- | --- |
| 问题与威胁模型 | PROACT / BrainWash、固定十任务 EWC | 先前模型和历史任务假设干净 |
| 特征定义 | extract_features、feature metadata | 类别标签与投毒标签分开；predicted 模式使用固定新 head 的预测 |
| 基础有监督 | 样本/数据集报告、验证集消融 | 指标必须注明特征组、分区、阈值 |
| 特征解释 | Meeting 3 explain_* 的 SHAP/层贡献/相关矩阵 | log-odds 贡献，非因果解释；不要猜哪些层最重要 |
| 跨任务迁移 | Task1→Task9 的 seed3/4 报告 | AUC、固定阈值 FPR/TPR、bag FPR 分开 |
| 梯度形状对照 | revision_shape_supervised_v1 | 归一化改善部分误报但跨 seed 不稳定 |
| 无监督比较 | 原 MMD、Rank、local、shape-MMD | 反演参考的分布检验，不是部署投毒概率 |
| 本轮特异性对照 | thesis_closeout*/supervised | 待服务器运行；同时报告随机告警与poison TPR |
| 本轮 Rank 复核 | thesis_closeout*/rank | 待服务器运行；旧seed重用，不称为未见确认 |
| 总结与局限 | thesis_closeout*/tables | 不承诺固定目标 FPR、泛化或部署安全 |

## 已有真实结果（2026-09-26，非本轮新实验）

来源：用户下载的 revision_results_20260926_101113。
监督训练任务为0-based Task1，目标为Task9，每次测试各500张原图的对应视图。

| Seed | 目标特征组 | AUC | Clean FPR | Poison TPR | 纯clean bag告警 |
| --- | --- | --- | --- | --- | --- |
| 3 | portable | 0.987932 | 18.6% | 99.2% | 100% |
| 3 | extended | 0.987856 | 24.6% | 99.2% | 100% |
| 3 | shape | 0.987428 | 8.4% | 96.6% | 14.4% |
| 4 | portable | 0.961952 | 12.8% | 95.0% | 99.5% |
| 4 | extended | 0.961676 | 12.8% | 95.4% | 99.5% |
| 4 | shape | 0.964844 | 11.6% | 97.0% | 93.5% |

可用表述：检测特征保留跨任务排序能力，但源校准阈值不能稳定控制目标误报。
shape 整体变化属于多项特征同时改变的对照：它除了归一化，也移除了不确定性、
activation 和其他特征。因此不能凭此证明“整体梯度尺度”是唯一或确定的因果因素。
目标 shape 的随机扰动样本告警分别为57%和51%，攻击特异性仍有限。

无监督 seed0：原 MMD/local/shape-MMD 对所有场景均100%告警，无法支持投毒区分。
Rank 对100个模拟 clean bags观察到0次告警；各随机扰动比例告警为0%–1%。

| 实际投毒比例 | 1.33% | 5.33% | 10% | 25.33% | 50% | 100% |
| --- | --- | --- | --- | --- | --- | --- |
| Rank检出 | 0% | 0% | 2% | 5% | 31% | 96% |

这里使用实际比例：150张 bag 中2张=1.33%，8张=5.33%，38张=25.33%。
0次告警不是总体零误报保证；bags复用有限原图，不能当100个独立真实任务。

历史shape-MMD审计对九个历史留出任务均未告警，但真实clean bags全部告警。
这支持“历史反演参考与真实incoming不匹配”的解释。它同时混合了合成/真实、
任务类别与模型适配等差异，尚未单独隔离各因素，也不排除其他未检测实现问题。
不能据此宣称已经证明唯一根因。

## 本轮结果应怎样填入论文

- 固定比较三个特征组 × 两种负类策略，不能只保留最好的种子或模型。
- 同时比较 clean FPR、poison TPR、随机告警、poison-vs-random AUC和各比例bag告警。
- 若随机告警下降但poison检出也下降，应报告特异性与敏感性的权衡。
- 若Rank在seed1/2低误报仍成立且高比例检出稳定，可称“在这些模型/攻击复核中观察到一致趋势”；
  不扩大为所有持续学习任务、所有攻击或无标签投毒检测已解决。
- 若固定方案失败，也作为负结果收尾，后续研究写入限制章节；不要反复根据旧test换参数。

## 可使用的英文结论草稿

The supervised detector retained strong sample-ranking performance when transferred
from an early task to a later task. However, source-calibrated decision thresholds
did not reliably control target-task false positives, particularly after aggregating
sample alarms into dataset-level decisions. Per-sample gradient-shape normalization
reduced false positives in some runs but did not provide consistent calibration.

Under the strict label-free setting, tests based on historical inversion references
often detected distribution differences even for clean incoming data. The observed
historical-to-incoming mismatch is consistent with synthetic-to-real and task-related
shift, although these factors were not causally isolated. Rank-dependence testing
showed low observed false-alert rates in the seed0 experiment and substantial power
at full contamination, while sensitivity to sparse poisoning remained limited.
Cross-run results and the source random-control training ablation must be added
after the fixed closeout experiments have completed.

## 交给教授确认的内容

提供主表、SHAP图、原MMD失败与Rank局限，以及本轮固定对照结果，请教授确认是否以
这一范围结束实验并完成写作。尚未进行的未见攻击、任务顺序泛化、真正独立确认、
检测过滤后的防御收益等，列为未来工作；是否需要其中某项由教授决定。
