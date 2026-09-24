# Meeting 3：服务器运行指南

代码已提供；本机 CPU 测试不能证明真实 GPU 全流程成功，更不能证明迁移有效。
请先同步新的 `detector/` 代码，保留服务器原 `work/` 和 `results/`。
以下服务器根目录采用你之前提供的 `/home/p.zhang/PROACT`；不是 Mac 路径。

## 1. 检查环境

```bash
conda activate proact38
cd /home/p.zhang/PROACT
export PYTHONDONTWRITEBYTECODE=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python -B -m detector.doctor --require-cuda
python -B -m unittest discover -s detector/tests -q
```

新代码保持 Python 3.8 语法，无须安装 SHAP 包；仍使用项目已有 torch、torchvision、
NumPy、pandas、SciPy、scikit-learn、joblib、matplotlib。只加载可信 checkpoint/joblib 文件。
先保证以上检查通过，不为了运行新代码盲目升级服务器的 torch/torchvision。

## 2. 正式实验：第二任务训练，第十任务检测

以下 plan 只生成协议及脚本，不启动训练。**必须在 GPU 服务器上生成 plan**：
脚本会登记服务器解释器、目录、源代码哈希；不要把 Mac 生成的脚本直接搬到服务器运行。

```bash
python -B -m detector.transfer_pipeline plan \
  --output-dir detector/work/meeting3_transfer_v1 \
  --source-task 1 --targets 9 --seeds 3 4 \
  --label-mode ground_truth

python -B -m detector.transfer_pipeline run \
  --plan detector/work/meeting3_transfer_v1/protocol.json --dry-run

bash detector/work/meeting3_transfer_v1/commands.sh
```

正式默认参数见协议：victim 20 epochs、inversion 10,000 iterations、attack 5,000 epochs。
两种固定特征组都会执行。默认按 seed 顺序串行执行，不会自动占用所有 GPU。
一个 seed 需要在源任务和目标任务各运行训练、反演、攻击与 clean/poison 效果评估，
然后提取特征；这不是只重跑 CPU 分类器的小任务。

如需两张 GPU 并行，可在**不同 tmux 会话、不同输出目录**各登记一个 seed，
随后分别运行对应脚本；此时各目录是单 seed 报告，不会自动合并成一份跨目录汇总：

```bash
# 第一会话
CUDA_VISIBLE_DEVICES=0 python -B -m detector.transfer_pipeline plan \
  --output-dir detector/work/meeting3_transfer_seed3 --seeds 3
CUDA_VISIBLE_DEVICES=0 bash detector/work/meeting3_transfer_seed3/commands.sh

# 第二会话
CUDA_VISIBLE_DEVICES=1 python -B -m detector.transfer_pipeline plan \
  --output-dir detector/work/meeting3_transfer_seed4 --seeds 4
CUDA_VISIBLE_DEVICES=1 bash detector/work/meeting3_transfer_seed4/commands.sh
```

已有空闲 tmux 会话可以直接运行；新建可用 `tmux new -s meeting3`。
`Ctrl+B` 再 `D` 只脱离会话，不中断任务；`Ctrl+C` 会中断。
下载/CUDA/训练失败后日志保留，`run_state.json` 标记失败；脚本不会把部分结果当完成。
再次运行会跳过校验通过的完整步骤，但遇到失败/中断步骤会停下。
不要手改状态为 complete 或删除校验：先检查日志，修复后选择新目录重跑；
若要复用昂贵的完整上游产物，应先逐项核验并另行制定恢复方案。
运行期间不要更换方法代码或环境版本。

可选扩展必须在观察结果前单独登记：`--targets 4 9` 增加中期任务；
`--label-mode predicted` 使用预测类别。两者都要新 output-dir，不覆盖主实验。

## 3. 已有模型的特征解释与报告核对

先在服务器核对会议上 86.4% 对应哪个模型/数据分区，不能从汇总数字猜模型：

```bash
python -B -m detector.meeting_audit \
  --roots detector/results detector/work/full_v2_seed0_proact38 \
  --output-dir detector/work/meeting3_existing_audit
```

已有正式 pipeline 的样本模型是 `sample/detector_bundle.joblib`，特征是 `features.csv`；
运行前确认源目录确实存在这两个文件和各自校验文件。若配置选择了其他 label mode，
必须使用对应模型当初实际训练的特征，不能混用 `predicted_features.csv`。

```bash
python -B -m detector.explain_detector \
  --features detector/work/full_v2_seed0_proact38/features.csv \
  --bundle detector/work/full_v2_seed0_proact38/sample/detector_bundle.joblib \
  --output-dir detector/work/meeting3_existing_explanation \
  --ablations
```

若目录不是这个名字，请替换为模型的真实来源路径。脚本校验失败应找回完整原产物，
不要改哈希绕过校验。只有 `metrics.json` / `report.md` 无法还原 SHAP。
需要 `features.csv`、同名 `.metadata.json`、模型 `.joblib` 及其
同名 `.joblib.sha256.json` 校验文件（旧模型名为 `detector_bundle`，新迁移模型名为 `bundle`）。

## 4. 输出目录与发回文件

```text
detector/work/meeting3_transfer_v1/
  protocol.json / run_environment.json / run_state.json
  commands.sh / logs/
  summary.json / summary.md                    # 所有种子完成后生成
  seed3/                                      # seed4 相同结构
    artifacts_task1/                          # data、victim、inversions、attack
      effectiveness/metrics.json              # 以及 clean/poison accuracy matrices
    artifacts_task9/
    features_task1/                           # CSV、metadata、manifest、references
    features_task9/
    source_portable/                          # 冻结 bundle、SHA、fit_metrics
    source_extended/
    explain_portable/                         # SHAP、图、相关矩阵、示例
    explain_extended/                         # 另含 validation 消融
    evaluate_task1_portable/                  # 源任务 test 对照
    evaluate_task1_extended/
    evaluate_task9_portable/                  # 目标 test；指标、rates、ROC、预测、图
    evaluate_task9_extended/
```

首轮分析请发：根目录四份 JSON/MD 汇总与协议、run_state/run_environment，以及每 seed 的
两个 `effectiveness/metrics.json`、两个 `source_*/fit_metrics.json`、所有
`evaluate_*/evaluation_metrics.json` 和 `rates.csv`、两个 `explain_*/` 中的
`explanation.json`、`feature_importance.csv`、`family_importance.csv`、`examples.json`，
以及 extended 的 `feature_comparison_validation.csv`。也可以直接打包这些报告目录。
不必首先传巨大数据集、checkpoint、noise 或 references；需要进一步复算时再提供。
如果失败，发 `run_state.json` 与对应 `logs/<step>.log` 最后部分。

看结果时同时检查目标 clean FPR、投毒 TPR、随机扰动告警以及攻击是否真的降低历史准确率。
不因命令正常退出就判定研究问题已解决。
