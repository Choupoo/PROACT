# Historical closeout comparisons

This records the September closeout workflow, not the current definition of
thesis completion. It combines a source random-negative ablation with fixed
Rank rechecks. Current experiments are in
[ADVISOR_FINAL_RUN.md](ADVISOR_FINAL_RUN.md).

## Supervised random-negative ablation

Reuse Task 1 → Task 9 features for seeds 3/4. For each portable, extended and
shape representation, fit two models:

- `clean_only`: clean=0 and BrainWash=1.
- `clean_and_random`: clean=0, random=0 and BrainWash=1 in source training only.
  Clean and random have loss weights 0.5 each; BrainWash has weight 1. This
  preserves total loss weight per original. StandardScaler weights actual
  training views equally; only classifier loss uses these weights.

Original random-control labels remain -1 in the CSV. The training copy remaps
them to zero. No random validation/test/reserve rows or target rows enter fitting.
Source validation-clean sets the sample cutoff; source reserve-clean calibrates
the count. Random perturbations are a non-BrainWash control, not assumed harmless.

All six models freeze before target values are read. Report clean-versus-poison
AUC, FPR/TPR, poison-versus-random AUC, sample and bag random alerts, and
source-validation SHAP. Each explanation uses that model's actual training
background, including random views where fitted; attributions are uniform-
background, log-odds contributions. The six-model comparison replaces the old
raw-feature ablation option.

## Fixed Rank rechecks

Use seed 1/2 complete predicted/reference CSVs. Fixed settings are alpha=0.05,
499 bootstrap draws, seed=20260920 and reference/incoming limits of 256.
Each run fits and freezes its own historical references, then evaluates Rank
and original MMD on the same bags. No incoming clean calibration is added.

These seeds were previously inspected and are labelled
`fixed_method_recheck_existing_data`, not untouched confirmation. Different
artifact hashes establish identity, not independence. The same CIFAR pool and
class order are reused. Default evaluation uses 150 distinct originals per bag
and 100 bags per scenario; unsupported cases reduce coverage.

## Run

Required source directories:

```text
detector/work/meeting3_transfer_v1/seed3/features_task1/
detector/work/meeting3_transfer_v1/seed3/features_task9/
detector/work/meeting3_transfer_v1/seed4/features_task1/
detector/work/meeting3_transfer_v1/seed4/features_task9/
detector/work/unsupervised_confirm_v1/source_seed1/
detector/work/unsupervised_confirm_v1/source_seed2/
```

The first four need features CSV/metadata; the last two need reference and
predicted CSV/metadata. Missing inputs stop the script rather than trigger
upstream training. Use the intended code version before starting a fresh run;
do not alter active registered studies.

```bash
conda activate proact38
cd /home/p.zhang/PROACT
bash detector/run_thesis_closeout.sh
```

To specify input roots and an unused output directory:

```bash
THESIS_TRANSFER_RUN=detector/work/meeting3_transfer_v1 \
THESIS_RANK_SOURCE1=detector/work/unsupervised_confirm_v1/source_seed1 \
THESIS_RANK_SOURCE2=detector/work/unsupervised_confirm_v1/source_seed2 \
bash detector/run_thesis_closeout.sh detector/work/thesis_closeout_v1
```

This reuses CSVs without downloading data, training PROACT or generating attacks.
Default output names include a timestamp. Failed runs remain intact; the wrapper
does not overwrite or automatically resume them.

For separate branches:

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

The first two accept `--dry-run` for file checks. Actual loading validates
content/protocol compatibility. Summary requires both branches to complete.

## Analysis

`supervised/` contains root summary/configuration/state and per-seed frozen,
explanation and task evaluation directories. `rank/` contains equivalent records,
per-replicate models, historical audits, evaluation metrics and rates.
`tables/` contains `thesis_summary.md/json`, `supervised_summary.csv` and
`rank_summary.csv`.

Retain all seeds, fractions and negative results. Lower random alerts coupled
with lower poison detection are a trade-off, not an unqualified improvement.
AUC, operating thresholds and dataset decisions answer different questions.
Repeated bags do not add independent experiments; no unsupported decision is
counted as correctly clean. Full-task attack damage is not evidence that
filtering improves accuracy.

Writing guidance is in [THESIS_WRITEUP.md](THESIS_WRITEUP.md). The current
experiment scope supersedes this historical closeout plan.
