# Meeting 3 server commands

Historical commands for the true-class source-to-target protocol in
[MEETING3_PROTOCOL.md](MEETING3_PROTOCOL.md). For current predicted-class runs,
use [ADVISOR_FINAL_RUN.md](ADVISOR_FINAL_RUN.md).

## Environment

```bash
conda activate proact38
cd /home/p.zhang/PROACT
export PYTHONDONTWRITEBYTECODE=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python -B -m detector.doctor --require-cuda
python -B -m unittest discover -s detector/tests -q
```

Use the existing compatible torch/torchvision and numerical stack. Linear SHAP
is implemented directly; no SHAP package installation is required. Only load
trusted serialized artifacts.

## Register and run

Register on the GPU server: generated paths and source hashes belong to that
machine. Preserve old outputs and do not change code during an active run.

```bash
python -B -m detector.transfer_pipeline plan \
  --output-dir detector/work/meeting3_transfer_v1 \
  --source-task 1 --targets 9 --seeds 3 4 \
  --label-mode ground_truth

python -B -m detector.transfer_pipeline run \
  --plan detector/work/meeting3_transfer_v1/protocol.json --dry-run

bash detector/work/meeting3_transfer_v1/commands.sh
```

The plan uses 20 victim epochs, 10,000 inversion iterations and 5,000 attack
epochs. Each seed generates source and target artifacts, extracts features and
evaluates the two fixed feature sets. Seeds run sequentially by default.

For separate single-seed plans in different tmux windows:

```bash
# First window.
CUDA_VISIBLE_DEVICES=0 python -B -m detector.transfer_pipeline plan \
  --output-dir detector/work/meeting3_transfer_seed3 --seeds 3
CUDA_VISIBLE_DEVICES=0 bash detector/work/meeting3_transfer_seed3/commands.sh

# Second window.
CUDA_VISIBLE_DEVICES=1 python -B -m detector.transfer_pipeline plan \
  --output-dir detector/work/meeting3_transfer_seed4 --seeds 4
CUDA_VISIBLE_DEVICES=1 bash detector/work/meeting3_transfer_seed4/commands.sh
```

These directories contain separate reports, not an automatically merged summary.
Detach with `Ctrl+B`, then `D`. Failures leave logs and state; rerunning verifies
and skips completed steps but stops at failed/incomplete steps. Inspect the
failure and use a new directory rather than editing completion flags or hashes.

Optional `--targets 4 9` or `--label-mode predicted` changes require a separate
plan fixed before target inspection.

## Audit or explain an earlier detector

```bash
python -B -m detector.meeting_audit \
  --roots detector/results detector/work/full_v2_seed0_proact38 \
  --output-dir detector/work/meeting3_existing_audit

python -B -m detector.explain_detector \
  --features detector/work/full_v2_seed0_proact38/features.csv \
  --bundle detector/work/full_v2_seed0_proact38/sample/detector_bundle.joblib \
  --output-dir detector/work/meeting3_existing_explanation \
  --ablations
```

Use the actual matching feature/model paths and label mode. Explanation needs the
CSV, metadata, bundle and checksum; metrics alone cannot reconstruct SHAP. These
legacy explanation commands are not instructions for current advisor bundles.

## Outputs

The root contains `protocol.json`, `commands.sh`, environment/state, logs and,
after all seeds finish, `summary.json` and `summary.md`. Each seed contains:

- `artifacts_task1/`, `artifacts_task9/`: victim, inversion, attack and effectiveness.
- `features_task1/`, `features_task9/`: CSVs, metadata, manifest and references.
- `source_portable/`, `source_extended/`: frozen models and fit metrics.
- `explain_portable/`, `explain_extended/`: SHAP, correlations and examples;
  extended also has validation ablations.
- `evaluate_task1_*/`, `evaluate_task9_*/`: source/target test metrics,
  rates, predictions and plots.

For analysis, provide root configuration/state/summary, per-task effectiveness
metrics, source fit metrics, evaluation metrics/rates and explanation JSON/CSVs.
Datasets, checkpoints and full features are needed only for deeper recomputation.
For a failure, include its stage log and state. Evaluate attack damage alongside
clean FPR, poison TPR and random alerts.
