# 教授要求的梯度粒度受控对照

目的：回答更细粒度梯度范数是否优于全局或 stage 范数，以及哪些层/参数张量最具判别性。
本轮只补有监督梯度粒度实验。原有无监督和收尾结果无需重跑。

## 固定八组比较

| 梯度粒度 | 仅范数 | 同一组附加特征 |
| --- | --- | --- |
| Global：整个 backbone 一个 L2 范数 | norm_global | norm_global_context |
| Stage：stem、layer1–4 五个范数 | norm_stage | norm_stage_context |
| Layer：各具名模块，合并该模块 weight/bias 梯度 | norm_layer | norm_layer_context |
| Parameter：每个具名参数张量的 Frobenius/L2 范数 | norm_parameter | norm_parameter_context |

附加特征固定为 entropy、confidence、true_class_probability、margin、activation_norm_l2。
例如 conv1.weight 和 conv1.bias 在 Parameter 组中是两个范数，在 Layer 组中合并为一个。
这些不是每个标量权重各自的梯度绝对值。教授的 “norm for each weight” 存在口语歧义，
向教授汇报时应明确这里采用 named parameter tensor 的解释。

每个比较块内只更换梯度粒度。沿用 seed3/4、Task 1 → Task 9（零基编号）、源训练
clean/poison、StandardScaler、LogisticRegression(C=1, L2, lbfgs, max_iter=2000)。
源 validation clean 冻结样本阈值，独立 reserve clean 冻结计数阈值，alpha=0.05。
源 test 和目标 test 仅评估；random_control 仅作评估对照。分类器不收敛时实验显式失败。
模型维度随粒度变化，此比较不是参数数量相等的模型比较，也不单独调节各组超参数。

旧 train_detector.compare_feature_sets 已有逐参数验证集分析；本轮补的是固定八组、
相同校准规则、源测试与跨任务测试、多 seed、数据集级指标的正式对照。
本轮特征组不根据目标表现筛选；动态特征列按源文件的完整元数据冻结。

## 在服务器原 tmux 中运行

先同步本次修改的 detector 代码，然后执行：

```bash
conda activate proact38
cd /home/p.zhang/PROACT
bash detector/run_granularity.sh
```

默认复用以下四组完整输入（每组 CSV 及其 metadata.json）：

```text
detector/work/meeting3_transfer_v1/seed3/features_task1/features.csv
detector/work/meeting3_transfer_v1/seed3/features_task9/features.csv
detector/work/meeting3_transfer_v1/seed4/features_task1/features.csv
detector/work/meeting3_transfer_v1/seed4/features_task9/features.csv
```

文件必须包含 grad_norm_layer__* 和 grad_norm_param__* 特征；只有最近下载的报告包
不足以重拟合。无需重新训练 PROACT、攻击、下载数据集或提取已有完整特征。主要使用 CPU。
dry-run 只核对文件和哈希，完整 schema 在实际读取时检查；缺列会报错，不静默替换粒度。

源目录不同时可指定：

```bash
THESIS_TRANSFER_RUN=/home/p.zhang/PROACT/detector/work/你的原始特征目录 \
  bash detector/run_granularity.sh
```

需要固定输出路径时（必须不存在）：

```bash
bash detector/run_granularity.sh detector/work/granularity_v1
```

若中断，保留失败目录，使用新输出目录重跑。本轮分类器拟合成本远低于上游 GPU 训练。

## 产物与判断标准

默认输出 detector/work/granularity_日期_时间/，包含：

- run_config.json、run_environment.json、run_state.json：实验设置、版本与完整状态。
- granularity_summary.md/json/csv：两任务、八特征组、两个 seed 的各项指标。
- paired_differences_vs_stage.csv：同一 seed 内相对 stage 的差值，再计算均值及样本标准差。
- shap_summary.md、shap_by_seed.csv：各粒度的层/参数解释，来自源验证集。
- professor_update.md：根据本轮真实结果自动填入数值的英文教授汇报草稿。
- seed3/、seed4/：每组冻结模型、SHAP 图、样本预测、数据集曲线与逐模型结果。

先检查 run_state.status=complete。判断更细粒度是否有帮助时，同时比较 AUC、clean FPR、
poison TPR、random_control、纯干净数据集告警和低投毒比例曲线。若 AUC 提高但误报恶化，
应报告“排序改善，阈值迁移未改善”，不能笼统声称检测已解决。
两个 seed 的标准差只作描述，重复 bags 不增加独立模型数量。已看过的 Task 9 属于探索性
复核。SHAP 反映模型使用哪些特征，相关特征可能共享贡献，不能代替性能对照或因果证据。

## 打包与教授汇报

运行结束打印确切输出目录。以下以固定目录 granularity_v1 为例：

```bash
cd /home/p.zhang/PROACT
tar -czf detector/work/granularity_v1_results.tar.gz \
  -C detector/work granularity_v1
```

下载该压缩包即可分析。优先阅读 granularity_summary.md 与 shap_summary.md，然后根据
实测差值补充教授邮件的结论。professor_update.md 只是本地草稿，脚本不会发送邮件。
若 finer norms 未提高结果，也已经回答了这项研究问题，应该如实汇报负面结果。
