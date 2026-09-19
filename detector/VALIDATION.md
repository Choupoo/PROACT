# 验证记录

## 2026-09-20：数据集误报校准与历史参考诊断

- 本机 `python -B -m unittest discover -s detector/tests -q`：**96 项全部通过**。
  新增覆盖计数校准的边界与误报预算、按原图而非重复 bags 计数、旧 bundle
  兼容、测试集不影响拟合、validation-only 参考审计、CSV 完整性与重评估命令。
- `demo --output-dir detector/work/cpu_demo_calibration_v2` 完整通过；
  `reassess --source-run detector/work/cpu_demo_calibration_v2 --output-dir
  detector/work/cpu_reassessment_v2` 的 dry-run 及实际执行均通过。
  两次使用人工合成特征，只验证链路；报告明确标记，不是实际攻击性能结果。
- Ruff 静态检查、Git diff 空白检查通过；25 个顶层及测试 Python 文件通过
  Python 3.8 语法解析。本机仍为 Python 3.12 / torch 2.9，未在服务器旧环境重跑。
- 新拟合模型使用 count_bound 主判定，保留 LR/top-tail 对照；旧模型保持原行为。
  误报预算有独立同分布等适用条件，保守性可能降低低投毒率检出能力。
- 历史 MMD 保留真实原始告警，新增参考域诊断与投毒判定 abstain；
  没有把“拒绝给出投毒判断”当作误报改善，没有声称修复无标签检测能力。
- 尚未对完整真实服务器产物重评估，不能宣称原 19.1% clean FRR 已下降，
  或原无标签 100% 告警已解决。修改后复用旧测试集仅作诊断，仍需独立确认实验。
- 全部代码及文档修改限定在 detector；旧实验结果未覆盖，未修改外部环境。

## 2026-09-19：旧版 PyTorch 兼容修复

- 用户提供的服务器日志：Python 3.8.20、torch 1.10.1、torchvision 0.11.2，
  CUDA 可用、依赖检查通过；原 78 项测试中 75 项通过、3 项报错。
  两项报错来自 `weights_only` 不被旧版 `torch.load` 支持，另一项来自
  `torch.equal` 跨 dtype 比较。这不是本机直接访问服务器得到的结果。
- 修复 `common.py`：按 `torch.load` 的显式签名选择是否传 `weights_only=False`，
  始终保留 CPU 映射；索引整数性检查使用同 dtype 的往返转换比较。
- 新增 4 项测试，覆盖旧版加载接口、新版显式参数、无关 TypeError 不被重试吞掉，
  以及旧版严格 dtype 比较下整数/小数/NaN/Inf 的校验。
  新增测试在修复前实际复现加载和 dtype 错误；修复后本机 **82 项全部通过**。
- 新增 `config.proact38.json`，仅更换 detector 运行目录，复用已有 PROACT 产物；
  已验证该配置的 pipeline dry-run 正常生成全部命令。
- 尚未在目标服务器的 Python 3.8 / torch 1.10 环境重新运行修复后的测试或真实实验。
  本地旧接口替身回归通过，不等于该服务器完整流程已成功。

## 2026-09-15：初始验证

### 已实际执行

- `python -B -m unittest discover -s detector/tests -q`：**78 项通过**。
- Ruff 静态检查、格式检查通过；Git diff 空白检查通过。
- 20 个 Python 源文件通过 Python 3.8 语法解析；Notebook 的 4 个代码单元通过语法解析。语法兼容不是对 Python 3.8 依赖环境的运行保证，新环境仍建议 Python 3.10/3.11。
- 11 个命令入口的 `--help` 均正常退出。
- `pipeline --dry-run` 与 `bootstrap --dry-run` 正常生成全部预期命令。
- `demo --output-dir detector/work/cpu_demo_final` 完整通过：合成特征 → 验证集分析/消融 → 冻结样本/数据集/无标签模型 → 独立测试 → 报告 → 两条路线的独立预测 CLI。
- 提取单元测试使用小型模型，另有实际 PROACT ResNet 参数分组测试；incoming/reference 提取入口采用模拟 checkpoint 与依赖替身。EWC Fisher 缓冲区、可信 tensor pickle 的 CPU 映射另有回归测试。
- 代码修改与新增文件范围检查：仅 `PROACT/detector/`；历史 `detector/results/` 未改动。Notebook 的旧分散命令已替换为集中入口，原版本可从 Git 恢复。

合成演示的报告位于 `work/cpu_demo_final/report.md`，开头明确标为人工合成。其任何 AUC、误报或检出率都不能作为真实 CIFAR-100/BrainWash 实验指标。演示中即使出现高误拒率也如实报告，不以“跑通”为由标记算法有效。

### 本机环境与未完成的实证验证

实际测试解释器为 `/opt/miniconda3/bin/python3`，Python 3.12.7，torch 2.9.0、NumPy 1.26.4、pandas 2.3.3、scikit-learn 1.7.2。

`python -B -m detector.doctor` 实际检测到：

- 数值依赖与 torch 导入正常。
- CUDA 不可用。
- 安装的 torchvision 为 0.21.0，导入子进程以信号 11 退出。

因此没有在本机执行真实上游训练/反演/攻击，也没有用真实 checkpoint 跑新协议完整评估。没有修改 detector 之外的环境来修复依赖。应在兼容的 GPU 环境按 README 运行；不能据本记录断言细粒度特征提升性能、无标签方案具有攻击特异性，或已满足教授的实证验收标准。
