# 验证记录（2026-09-15）

## 已实际执行

- `python -B -m unittest discover -s detector/tests -q`：**78 项通过**。
- Ruff 静态检查、格式检查通过；Git diff 空白检查通过。
- 20 个 Python 源文件通过 Python 3.8 语法解析；Notebook 的 4 个代码单元通过语法解析。语法兼容不是对 Python 3.8 依赖环境的运行保证，新环境仍建议 Python 3.10/3.11。
- 11 个命令入口的 `--help` 均正常退出。
- `pipeline --dry-run` 与 `bootstrap --dry-run` 正常生成全部预期命令。
- `demo --output-dir detector/work/cpu_demo_final` 完整通过：合成特征 → 验证集分析/消融 → 冻结样本/数据集/无标签模型 → 独立测试 → 报告 → 两条路线的独立预测 CLI。
- 提取单元测试使用小型模型，另有实际 PROACT ResNet 参数分组测试；incoming/reference 提取入口采用模拟 checkpoint 与依赖替身。EWC Fisher 缓冲区、可信 tensor pickle 的 CPU 映射另有回归测试。
- 代码修改与新增文件范围检查：仅 `PROACT/detector/`；历史 `detector/results/` 未改动。Notebook 的旧分散命令已替换为集中入口，原版本可从 Git 恢复。

合成演示的报告位于 `work/cpu_demo_final/report.md`，开头明确标为人工合成。其任何 AUC、误报或检出率都不能作为真实 CIFAR-100/BrainWash 实验指标。演示中即使出现高误拒率也如实报告，不以“跑通”为由标记算法有效。

## 本机环境与未完成的实证验证

实际测试解释器为 `/opt/miniconda3/bin/python3`，Python 3.12.7，torch 2.9.0、NumPy 1.26.4、pandas 2.3.3、scikit-learn 1.7.2。

`python -B -m detector.doctor` 实际检测到：

- 数值依赖与 torch 导入正常。
- CUDA 不可用。
- 安装的 torchvision 为 0.21.0，导入子进程以信号 11 退出。

因此没有在本机执行真实上游训练/反演/攻击，也没有用真实 checkpoint 跑新协议完整评估。没有修改 detector 之外的环境来修复依赖。应在兼容的 GPU 环境按 README 运行；不能据本记录断言细粒度特征提升性能、无标签方案具有攻击特异性，或已满足教授的实证验收标准。
