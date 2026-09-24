# Meeting 3：现有证据核对（2026-09-24）

只读检查了本仓库 `detector/results/` 和本机
`/Users/zhangpan/Downloads/proact-result/`。机器生成的完整清单在
`detector/work/meeting3_local_audit/audit.json` 和 `report.md`。
本文件不是新实验结果。

## 已核实

- 仓库现存 `results/metrics.json` 属于旧三特征协议 `clean_supervised_baseline_v1`。
  test ROC-AUC 为 0.880064、poison TPR 为 44.8%、clean FPR 为 4.4%。
  不能把它作为会议上新版完整特征模型的结果。
- 当前 `config.proact38.json` 的样本模型配置选择 `extended`。
  此组仍含 activation norm 与历史 gradient cosine；不能据此证明会议所说的删特征模型已运行。
- 下载目录中现有 seed 子目录主要是无监督评估及运行配置，不含用于重建会议模型的完整
  监督特征 CSV、冻结 bundle 及其校验文件。
- 当前 MMD 方法已经比较低维 detector 描述符，不是完整 backbone embedding。

## 尚无法核实

- **86.4% 的准确来源**：清单没有找到能确认此值、特征组、分区、阈值和模型版本的完整记录。
  这不表示数字错误，只表示现有证据不足。
- 哪些层/特征真正最重要：新的解释器和消融入口已实现，但不能从汇总指标推算真实 SHAP。
- 新源任务到目标任务方案是否改善：需要 GPU 正式运行，不能用 CPU 合成测试作性能证据。

下一步按 `MEETING3_RUN.md` 在服务器对会议使用的原模型生成解释，同时执行新注册实验。
保留旧有监督结果和严格无监督失败/对照结果；不把新方法的成功运行或改名当作旧问题解决。
