# 严格无标签：历史任务排序关系适配实验

这是**候选研究方法，不是已经解决无监督投毒检测的最终版本**。
有监督模型、count_bound 校准、既有实验文件和原始 MMD 实现均不修改。
不使用可信干净新任务参考，不筛选“看起来干净”的新任务样本来拟合模型。

## 为什么增加这条路线

已有实验中，历史反演参考与干净新任务的原始特征分布不同，原 MMD
对所有干净批次也告警。原方法检验的是分布差异，本身不能将差异归因为投毒。
新方法明确放弃部分信息，研究正常域差异能否与攻击信号分离：

1. 使用原无标签路线相同的 10 个 predicted-target 描述符，不按测试集挑选特征。
2. 每个历史任务单独建参考，不将所有任务混成一个参考分布。
   `reference_task_id` 是来源任务标识，不是新任务类别或 clean/poison 标签。
3. 计算每对特征的 Kendall tau-a，共 45 对；它只取决于样本间的相对大小。
   对整批数据各列施加严格递增的变换，排序关系不变。
4. 根据历史参考中的非退化程度固定可检验的特征对，再冻结模型。
5. incoming 全批次计算同样的排序关系，与每个历史任务比较。
   不需要假定其中多数样本干净，也不需要从中挑出干净子集。
6. 使用中心化 jackknife Gaussian multiplier bootstrap 估计每个对比的尾概率。
   对全部历史任务取 **max p-value**；只有每个历史任务都不兼容才告警。

这改变了统计问题：从“完整特征分布是否相同”变为
“保留的两两排序关系是否与至少一个历史任务相容”。
没有通过调高 alpha、隐藏原始告警、或把 abstain 当正确 clean 来降低误报。

## 数学实现和边界

对特征对 a,b，令
`h(i,j) = sign(x[i,a]-x[j,a]) * sign(x[i,b]-x[j,b])`，并列值贡献 0。
tau 是所有不同样本对的 h 均值。令 `hbar_i` 是固定 i 后其余 j 的均值：

```text
centered_pseudovalue_i = 2(n-1)/(n-2) * (hbar_i - tau)
bootstrap_error = sum_i(normal_i * centered_pseudovalue_i) / sqrt(n(n-1))
observed = max_pair abs(tau_incoming - tau_reference)
bootstrap_null = max_pair abs(error_incoming - error_reference)
p_approx = (1 + count(bootstrap_null >= observed)) / (1 + draws)
p_union = max_task p_approx(task)
```

各特征对共享同一组样本 multiplier，保留特征对之间的相关性；两组样本的
multiplier 独立。使用同时最大差异而非未经多重比较处理的逐对阈值。
max-task 对应 union null：只要一个历史任务确实相容，就不应仅因其他任务不同而拒绝。

这只是**近似 bootstrap 推断**，不是精确置换检验，也没有无条件的有限样本 5% 保证。
它需要独立观测、非退化的一阶项等条件；反演样本可能相关，连续性/并列值情况
也需要检查。新 clean 任务若具有所有历史任务都没有的排序关系，仍然会误报。

方法参考：[Chen (2018), Gaussian and bootstrap approximations for high-dimensional
U-statistics and their applications](https://arxiv.org/abs/1610.00032)。
文献支持相关 bootstrap 思路，不是本项目两样本/多历史任务应用有效性的证明，
更不是投毒识别保证。

### 必须公开的盲区

- 整批投毒如果只改变各特征的边际数值、保持排序关系，会被该路线忽略。
- 只改变高阶关系、但不改变两两 tau 的攻击也可能漏检。
- 低投毒比例会稀释排序变化；新增 1%、5% 比例评估，不只检查原 10% 以上情况。
- 与任一个历史任务相似都不告警，可能牺牲攻击敏感性；不能挑选让测试效果更好的参考任务。
- 特征常数化等不支持的输入明确报告缺失判定，coverage 下降；不计为正确接受。
- 无告警不代表安全；告警也不代表证明是恶意投毒。输出没有投毒概率。

## 服务器运行：只做新无监督实验

先同步新增代码，**保留服务器所有旧 work 目录**。无需重跑 PROACT、反演、攻击、
特征提取，也不训练任何有监督检测器。必须使用旧完整实验目录，不能用只包含
汇总文件的 Downloads 目录或缺少 CSV 的 reassessment 目录作为 source。

```bash
conda activate proact38
cd /home/p.zhang/PROACT
python -B -m unittest discover -s detector/tests -q

# 可选：只核对历史参考并试拟合，不读取 benchmark 内容，不写结果。
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
python -B -m detector.unsupervised_adapt run \
  --source-run detector/work/full_v2_seed0_proact38 \
  --output-dir detector/work/unsupervised_rank_v1 --dry-run

# 实际运行；每个场景打印进度。可在原 tmux 会话执行。
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
python -B -u -m detector.unsupervised_adapt run \
  --source-run detector/work/full_v2_seed0_proact38 \
  --output-dir detector/work/unsupervised_rank_v1
```

默认 alpha=0.05，499 次 bootstrap，task_size=150，每场景 100 个模拟批次。
1% 实际对应 2/150，5% 对应 8/150，25% 对应 38/150；报告给出实际比例。
若自定义很小的 task_size 使某个比例舍入为 0，该场景会明确跳过，不冒充投毒实验。
输出目录必须不存在；失败后先查看原因，修复后选择另一个全新目录，不覆盖旧结果。

输入要求：

```text
full_v2_seed0_proact38/
  reference_features.csv + reference_features.metadata.json
  predicted_features.csv + predicted_features.metadata.json
  unsupervised/unsupervised_bundle.joblib + .joblib.sha256.json
```

reference CSV 使用原提取器生成的 `taskN:sampleM` 身份。
两个 CSV 都要求 `label_mode=predicted` 和相同冻结模型/head/inversion 来源。
程序先保存新 rank 模型，再读取 benchmark CSV；标签只用于离线组装测试批次和计分。
原 MMD 与新方法使用**同一批原图和修改视图**对照，不是挑选两批不同数据。

输出：

```text
unsupervised_rank_v1/
  rank/rank_bundle.joblib + .joblib.sha256.json
  historical_audit.json     # 逐历史任务留出，不含新任务 clean 标签
  evaluation_metrics.json  # 两路线同批次指标、逐批 p 值及特征对诊断
  report.md
  run_config.json           # 冻结设置、源文件哈希、研究限制
  run_environment.json
  run_state.json
```

完成后发 `report.md`、`evaluation_metrics.json`、`historical_audit.json` 和
`run_state.json`，无需再次上传有监督结果。服务器实际效果尚未验证。

## 单独检测一批无标签数据

使用原提取入口生成一批 incoming 的 predicted 特征（不是含多视图的 benchmark）：

```bash
python -B -m detector.unsupervised_adapt predict \
  --bundle detector/work/unsupervised_rank_v1/rank/rank_bundle.joblib \
  --features detector/work/incoming/predicted_features.csv \
  --output detector/work/incoming/rank_prediction.json
```

至少 16 张；超过冻结的 max_incoming 时按确定性规则抽样，输出实际样本数。
研究评估命令不允许悄悄截断批次。结果是排序关系异常证据，不用于自动宣布数据安全。

## 如何判断研究有没有进展

同时看 clean 误报、poison 检出、random_control 告警和 coverage：

- clean 下降且 poison 检出保留：有进展，仍需独立任务/攻击/模型种子验证。
- clean 和 poison 都不告警：失去攻击信号，不算解决。
- clean 仍全部告警：历史与新任务的关系结构也不同，当前适配假设不成立。
- coverage 不完整：不能把未判定样本藏进 clean 正确率。

报告明确标记这是看过旧测试结果后的研究修订。即使本次数据变好，也不是新的
独立确认实验。不要用 test 反复挑特征、改 alpha 或选历史任务。
