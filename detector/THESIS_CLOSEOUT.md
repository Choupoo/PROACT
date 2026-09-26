# 毕设收尾：固定协议、完整对照与论文证据

目标：完成教授提出的特征解释、早期任务训练到后期任务检测，以及无监督 detector
特征分布比较，并呈现可复现的正负结果。毕业验收由教授决定，本流程不设置自动通过。
不承诺再做某个固定轮数就能解决跨任务阈值或低比例投毒问题。

## 本轮只做两件事

### 1. 有监督源任务随机扰动负类对照

复用 Task 1 → Task 9、seed3/4 的完整特征，不重新生成模型、攻击或特征。
task 编号为零起始，即第二个任务训练检测器、第十个任务评估。

固定比较 portable、extended、shape 三个特征组，每组各训练两个模型：

- `clean_only`：沿用原始 clean=0 / BrainWash=1 训练。
- `clean_and_random`：源训练分区 clean=0、random-control=0、BrainWash=1。
  clean 和 random-control 各权重 0.5，BrainWash 权重 1；每张原图的总损失权重
  与原模型相同，避免增加一种负类视图后仅靠类别比例变化获得改善。
  StandardScaler 对实际训练视图等权拟合；只有分类器损失使用上述权重。

原始 CSV 中 random-control 标签仍为 -1，仅在新模型内部的训练副本中映射为 0。
validation/test/reserve 的随机扰动样本、任何目标样本均不进入训练。
每个模型仍仅以源 validation-clean 设置样本阈值，源 reserve-clean 校准 bag 计数。
不假设匹配随机噪声一定无害，它只是已知非 BrainWash 扰动；当前检验的是攻击特异性。

所有六个模型在读取目标特征前保存冻结，分别报告：

1. clean-vs-poison AUC、冻结阈值 clean FPR / poison TPR；
2. random-control 样本告警率、poison-vs-random AUC；
3. 相同混合比例下的 clean、poison、random-control bag 告警；
4. 源 validation 的 SHAP、层重要性、示例与相关矩阵。

SHAP 背景包含该模型实际使用的训练视图；新模型不会错误遗漏 random-control 背景。
SHAP 使用这些背景行的均匀经验分布，解释仍位于 log-odds 空间，不是因果归因。
新模型禁用旧 raw clean/poison-only 消融入口；本轮六模型对照就是固定的消融设计。

成功与失败都保留。如果随机告警下降同时 poison TPR 大幅下降，就说明存在区分能力
权衡，不能只报告减少的误报。如果目标 clean bag 仍高误报，阈值迁移仍未解决。

### 2. 严格无监督 Rank 固定参数跨种子复核

使用现有 seed1/2 的 `reference_features.csv` 和 `predicted_features.csv`。
Rank 方法完全不变：alpha=0.05、bootstrap=499、seed=20260920、参考/输入上限256。
历史任务分别拟合，max-p 合并规则保持不变；不增加可信目标 clean 校准集。
同时运行原 MMD，对相同 bags 保留教授建议路线的失败/成功对照。

每个模型仅根据该次实验的历史反演参考重新拟合，然后冻结，再读取该次 incoming
特征。它是“固定算法和超参数的跨模型/攻击复核”，不是同一个已拟合模型跨 seed 应用。
投毒标签只在离线混合 bag 构造和指标计算中使用。

已有 seed1/2 在本项目历史中曾被观察，因此明确标为 `fixed_method_recheck_existing_data`。
不能称为未见种子盲测、独立确认或泛化保证。不同 checkpoint/attack 哈希只是身份检查，
不证明实验总体独立；相同 CIFAR 图片池和固定类别顺序仍被复用。

默认每批150张不同原图、每场景100个 bags；合并按实验等权，标准差按实验数计算。
未判定批次不当作正确 clean，保留 coverage。重复 checkpoint/attack 不计作新的模型实验。

## 服务器一键运行

同步整个 detector 源码，保留已有 work/results。原 tmux 环境可以继续使用。
脚本仅复用 CSV；不下载数据、不训练 PROACT、不运行模型反演或攻击。

```bash
conda activate proact38
cd /home/p.zhang/PROACT
bash detector/run_thesis_closeout.sh
```

默认输入目录：

```text
detector/work/meeting3_transfer_v1/seed3/features_task1/
detector/work/meeting3_transfer_v1/seed3/features_task9/
detector/work/meeting3_transfer_v1/seed4/features_task1/
detector/work/meeting3_transfer_v1/seed4/features_task9/
detector/work/unsupervised_confirm_v1/source_seed1/
detector/work/unsupervised_confirm_v1/source_seed2/
```

这是先前命令生成的预期路径，脚本会先检查全部输入；目录缺失就停止，不会自动启动
昂贵的上游实验。前四个目录各需要 features.csv 和 features.metadata.json；后两个各
需要 reference/predicted_features.csv 及对应 .metadata.json。只有结果报告包不够。

如果实际目录不同，可显式指定；环境变量仅改变输入路径：

```bash
THESIS_TRANSFER_RUN=detector/work/meeting3_transfer_v1 \
THESIS_RANK_SOURCE1=detector/work/unsupervised_confirm_v1/source_seed1 \
THESIS_RANK_SOURCE2=detector/work/unsupervised_confirm_v1/source_seed2 \
bash detector/run_thesis_closeout.sh detector/work/thesis_closeout_v1
```

输出路径必须在当前 detector 目录内且尚不存在。默认使用带时间戳的新目录。
失败时保留已完成步骤和错误状态；不支持覆盖或自动恢复失败目录。
代码、设置、输入文件哈希和环境都会记录。不要运行绑定旧代码哈希的旧 commands.sh。

### 分步执行（需要分别跑两条路线时）

```bash
export PYTHONDONTWRITEBYTECODE=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1

python -B -u -m detector.revision_study supervised \
  --source-run detector/work/meeting3_transfer_v1 \
  --output-dir detector/work/thesis_closeout_v1/supervised \
  --seeds 3 4 --negative-policies clean_only clean_and_random --explanations

python -B -u -m detector.thesis_closeout rank \
  --source-runs detector/work/unsupervised_confirm_v1/source_seed1 \
                detector/work/unsupervised_confirm_v1/source_seed2 \
  --output-dir detector/work/thesis_closeout_v1/rank

python -B -u -m detector.thesis_closeout summarize \
  --supervised-run detector/work/thesis_closeout_v1/supervised \
  --rank-run detector/work/thesis_closeout_v1/rank \
  --output-dir detector/work/thesis_closeout_v1/tables
```

前两条可附加 `--dry-run` 检查输入文件是否齐全；内容/协议兼容性在正式运行时验证。
如果只跑某一条路线，不必等待另一条；最终汇总要求两者均完成。

## 输出与分析材料

```text
thesis_closeout_日期_时间/
  supervised/
    summary.json / summary.md / run_config.json / run_state.json
    seed3/ 和 seed4/
      source_portable__clean_only/              # 另有其余五模型
      source_portable__clean_and_random/
      explain_portable__clean_only/             # 源 validation SHAP
      evaluate_task1_portable__clean_only/      # 源 test
      evaluate_task9_portable__clean_only/      # 目标 test
  rank/
    summary.json / summary.md / summary.csv / run_config.json / run_state.json
    replicate1/ 和 replicate2/                  # 对应 --source-runs 顺序
      models/ / historical_audit.json / evaluation_metrics.json / rates.csv
  tables/
    thesis_summary.md / thesis_summary.json
    supervised_summary.csv / rank_summary.csv
```

最后的主表保留每 seed 数值、均值、标准差和全部投毒/随机比例；不给出自动毕业通过结论。
将本次整个输出目录打包返回即可，不必包含输入数据集/模型/特征 CSV。

## 收尾规则和仍待完成的事

1. 完成本轮固定对照并审阅所有结果，包括没有改善的部分；不基于这批 test 自动选最优阈值。
2. 论文保留原监督、跨任务、原 MMD、rank/local/shape 的实验演进和观察顺序。
3. 有监督报告排序、操作阈值、数据集决策、随机控制四个不同层面的结果，不能由 AUC
   推断目标误报已受控。
4. 无监督报告 clean FPR、coverage、高/低污染检出，不把 seed0 的0/100告警写成总体零误报。
5. 独立沿用已完成攻击效果报告。检测性能不等于过滤数据后持续学习准确率会恢复，
   本轮没有新增防御训练有效性实验。
6. 将主表、SHAP、局限交给教授确认收尾范围。若需要真正未见实验，应在新种子生成前
   固定方案另作注册；本脚本不把旧种子自动包装成独立验证。

论文写作提纲和已经证实/尚未证实的表述见 [THESIS_WRITEUP.md](THESIS_WRITEUP.md)。
