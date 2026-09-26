# 验证记录

## 2026-09-26：梯度粒度受控补充实验

- 新增 global / stage / layer / named parameter tensor 四种粒度，分别比较仅梯度与
  加入同一组 uncertainty/activation 的模型。训练、验证校准、reserve 计数校准和
  Task 1 → Task 9 测试规则固定，默认 seed3/4，随机扰动仅用于评估。
- 本机全量 unittest **169 项通过**（104.657 秒）。新增 5 项覆盖源 schema 冻结、
  缺少细粒度特征/混入 head 的拒绝、测试及随机对照隔离、目标缺列拒绝，以及
  八组 × 两 seed × 两任务的实际 CLI 合成流程、SHAP、配对差值和输入哈希保留。
- Ruff 全目录、Git diff 空白与 bash 语法检查通过；46 个 Python 文件通过
  Python 3.8 语法解析。测试仍在本机 Python 3.12 执行，不代替服务器 3.8 实测。
- 本机没有原始完整特征 CSV，真实逐参数性能对照尚未运行。服务器命令和回传
  文件见 [GRANULARITY_STUDY.md](GRANULARITY_STUDY.md)。只有实际结果才能回答
  更细粒度是否提升性能；professor_update.md 将在实际实验完成后生成，未发送邮件。
- 所有改动限于 detector；原有无监督方法及既有实验产物未修改。

## 2026-09-26：毕设收尾实验与固定方法跨运行复核

- 全量 unittest **164 项通过**（84.264 秒），执行环境为本机 Python 3.12，
  不是服务器 Python 3.8 的实测结果。新增 6 项测试覆盖源训练随机负类及权重、
  验证/测试隔离、默认行为兼容、SHAP 背景一致性、跨运行统计，以及实际 CLI
  有监督六组对照 → 两次历史参考复核 → 汇总的合成流程。
- Ruff 全目录检查、Git diff 空白检查、入口脚本 bash 语法检查通过；44 个 Python
  文件通过 Python 3.8 语法解析。合成结果仅用于验证实现，不作为真实性能证据。
- 有监督新增源训练随机扰动负类对照；目标数据不参与训练或阈值校准。
  无监督保持 Rank 参数与严格无标签拟合约束，保留原 MMD 同批次对照。
  旧 seed 的复核不冒充未见数据上的独立确认实验。
- 所有修改位于 detector；旧结果保留。真实收尾实验未在本机执行，尚不能声称
  冻结阈值迁移、数据集级判断或低比例投毒检测已经解决，也不能代替教授验收。
- 服务器入口与输出说明见 [THESIS_CLOSEOUT.md](THESIS_CLOSEOUT.md)，
  已核实结果与论文写作框架见 [THESIS_WRITEUP.md](THESIS_WRITEUP.md)。

## 2026-09-26：有监督 shape 对照与严格无监督逐历史任务 shape-MMD

- 本次所有改动位于 detector；未删除既有文件、修改旧结果或启动 GPU 训练。
- 全量 unittest **158 项通过**，新增 13 项覆盖逐图尺度不变性、源 test/随机扰动
  隔离、SHAP 变换一致性、历史核拟合与检验库分离、标签完全忽略、max-p 规则、
  整批形状扰动、零梯度未判定、来源/任务/分区完整性，以及两条实际 CLI 合成流程。
- 最后增加显式 frozen-model 来源检查和合成/真实输入混用检查后，针对新增模块的
  13 项测试再次通过；CLI 测试还确认目标 CSV 校验失败时，历史模型已先保存，
  run_state 正确标记失败。测试临时产物自动清理，不作为真实性能证据。
- Ruff 全目录检查、Git diff 空白检查通过；42 个 Python 文件通过 Python 3.8
  语法解析。执行环境仍是本机 Python 3.12 / torch 2.9，未在服务器 Python 3.8
  上执行新实验，不将语法兼容等同于目标环境全流程成功。
- 只读核对下载的 Meeting 3 seed3/4 真实结果：目标样本 FPR 12.8%–24.6%，
  目标纯干净 bag 告警 99.5%–100%。报告包缺少原始 features.csv，尚未用真实
  特征拟合本次 shape 候选方法，不能声称阈值迁移或严格无监督检测已经解决。
- 运行命令、输入要求、统计假设和返回文件详见 [REVISION_RUN.md](REVISION_RUN.md)。

## 2026-09-24：Meeting 3 特征解释与跨任务检测

- 所有修改限于 detector；旧监督/无监督方法、PROACT 上游文件与旧结果未修改。
- 本机完整 unittest：**145 项通过**。新增 17 项覆盖固定十任务划分、源任务 EWC
  future-head/Fisher 兼容、模型/攻击身份、源 test 与随机对照不参与拟合、目标不参与校准、
  跨任务来源约束、SHAP 全排列数学核对及逐样本可加性、实际 CLI 合成集成、命令生成、
  已完成步骤哈希验证及失败步骤拒绝静默重启。
- Ruff 静态检查通过；39 个 Python 文件通过 Python 3.8 语法解析。
  测试运行环境仍为本机 Python 3.12 / torch 2.9，不等于服务器 Python 3.8 实测通过。
- 实际执行只读现有结果审计，输出 `work/meeting3_local_audit/`。
  未找到足以确认会议 86.4% 对应模型的完整证据，未编造真实 SHAP 排名。
- 另实际执行合成特征的 fit → SHAP/消融 → source-control → target evaluation，
  产物在 `work/meeting3_cpu_smoke_v2/`，仅为 CPU 流程验证，不是实验性能。
  检查了特征重要性图和 ROC/数据集曲线；新增绘图 fontconfig 配置，避免系统字体缓存
  自动生成软链接而与输出完整性约束冲突。最初 smoke 目录的失败记录保留，未覆盖。
- GPU 训练、反演、攻击、真实特征提取和新方案性能尚待服务器执行。
  测试通过不代表跨任务检测有效、目标误报受控或严格无监督问题已解决。

## 2026-09-20：严格无标签历史排序关系候选方法

- 仅新增 `rank_reference.py`、`unsupervised_adapt.py` 和对应测试/说明。
  有监督训练、数据集校准、原 `unsupervised.py`、原 pipeline 均未修改；旧产物不覆盖。
- 本机 **114 项测试全部通过**。新增 18 项覆盖 tau 的逐对计算、精确 leave-one-out
  jackknife 伪值、单调变换不变性及其攻击盲区、标签完全忽略、来源/身份校验、
  常数输入缺失判定、max-p union 规则、逐任务留出、模型不变及源文件哈希不变。
- 合成输入的实际 `python -B -m detector.unsupervised_adapt run` 命令通过。
  测试确认 benchmark 读取前新模型已保存；dry-run 不写输出，旧目录拒绝覆盖。
  正式实验将旧 MMD 与新方法放到相同批次上，不重新训练有监督模型。
- Ruff 检查及新 Python 文件格式检查通过；28 个顶层和测试 Python 文件通过
  Python 3.8 语法解析。这不是在服务器 Python 3.8 / torch 1.10 上的实际运行证明。
- 另外执行 50 次独立合成统计检查：每次 2 个历史任务，各 128 行，incoming 150 行，
  bootstrap=499，alpha=0.05。使用新测试文件的 `descriptors` 生成器，历史 seed
  `10000 + 2*trial + task`，incoming seed `20000 + trial`，trial=0..49。
  clean 告警 1/50；整批单调变换（norm *30+100、confidence/margin 立方）也为 1/50；
  第一特征的潜变量符号反转导致的关系变化检出 50/50；仅替换前 15/150 行时检出
  20/50。低比例敏感性不足如实保留，未据此选择新的参数。
- 上述全部为合成方法检查，不能作为 CIFAR-100/BrainWash 实验指标。
  严格无标签方案仍未在完整真实服务器特征上验证；不能声称解决原 clean 100% 告警，
  或满足低投毒率检出要求。即使重用旧 test 有改善，仍需新的独立模型/攻击/任务验证。

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
