# Gradient-shape comparisons — 26 September 2026

Historical revision following the Task 1 → Task 9 results for seeds 3/4. It
reuses existing features without retraining PROACT or regenerating attacks.
The supervised branch predates the current predicted-class protocol.

## Motivation

| Seed | Features | Target sample AUC | Target clean FPR | Clean bag alerts |
| --- | --- | --- | --- | --- |
| 3 | portable | 0.987932 | 18.6% | 100% |
| 3 | extended | 0.987856 | 24.6% | 100% |
| 4 | portable | 0.961952 | 12.8% | 99.5% |
| 4 | extended | 0.961676 | 12.8% | 99.5% |

Good ranking coexisted with poor threshold transfer. The downloaded report-only
package could not identify which feature changes caused the failures. Shape
normalisation tested one hypothesis; it did not isolate a causal explanation.

## Fixed comparisons

### Supervised shape model

For each image, divide its five stage norms `g` by `||g||_2`. Implementation
first divides by the row maximum for numerical stability. The resulting inputs
are `grad_shape_stage_stem` through `grad_shape_stage_layer4`.

Source train clean/poison fit StandardScaler and LR; source validation-clean
sets the sample threshold, and reserve-clean calibrates the count. Target data
do not fit or recalibrate the model. Portable, extended and shape are all
reported. Shape removes overall gradient magnitude, uncertainty, activation and
cosines together, so improved results cannot be attributed to scale alone.
All-zero gradients fail rather than being treated as normal.

SHAP applies the same deterministic shape transformation. Its contributions
refer to relative stage gradients. Do not add the old raw-feature
`--ablations` option to shape explanations.

### Label-free historical shape-MMD

Only predicted-class historical inversion features fit this model. Each task
has a fixed-seed split into kernel-fit and permutation-bank halves. The first
half sets the RBF bandwidth and 64-dimensional random Fourier map; the second
supplies the reference for 199 permutations.

Each historical task yields a p-value `(extreme + 1)/(199 + 1)`. Alert only if
the maximum task p-value is at most 0.05. Leave-one-task-out audits exclude every
image of the held-out task, including kernel-fit rows. This compares whole-bag
shape distributions, unlike the local kNN count method.

Inference needs a clean incoming shape distribution compatible with at least
one historical reference. Synthetic-to-real shift, dependent inversion images,
sparse poisoning, discarded magnitude information and finite random features
can violate that working model or reduce power. Undefined shapes are unsupported
and reduce coverage. Outputs describe shift, not a poisoning posterior.

References: [MMD](https://www.jmlr.org/papers/v13/gretton12a.html) and
[max-p combination](https://academic.oup.com/bioinformatics/article/38/1/141/6363784).

## Run supervised comparisons

Use a new output directory and the intended code version. Do not replace code
in active studies or rerun old scripts against changed fingerprints.

```bash
conda activate proact38
cd /home/p.zhang/PROACT
export PYTHONDONTWRITEBYTECODE=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1

python -B -m detector.revision_study supervised \
  --source-run detector/work/meeting3_transfer_v1 \
  --output-dir detector/work/revision_shape_supervised_v1 \
  --seeds 3 4 --dry-run

python -B -u -m detector.revision_study supervised \
  --source-run detector/work/meeting3_transfer_v1 \
  --output-dir detector/work/revision_shape_supervised_v1 \
  --seeds 3 4
```

Required inputs are each seed's Task 1 and Task 9 `features.csv` and metadata.
Models freeze before target values are read. Default evaluation uses 150 images
and 1,000 bags per fraction. This stage uses CPU classifiers, not GPU training.

Optional source explanation:

```bash
python -B -m detector.explain_detector \
  --features detector/work/meeting3_transfer_v1/seed3/features_task1/features.csv \
  --bundle detector/work/revision_shape_supervised_v1/seed3/source_shape/bundle.joblib \
  --output-dir detector/work/revision_shape_supervised_v1/seed3/explain_shape
```

## Run label-free comparisons

The source directory must contain complete `reference_features.csv` and
`predicted_features.csv`, each with matching metadata and model/head provenance.
Do not substitute Meeting 3 true-class features.

```bash
python -B -m detector.revision_study unsupervised \
  --source-run detector/work/full_v2_seed0_proact38 \
  --output-dir detector/work/revision_shape_unsupervised_seed0_v1 \
  --dry-run

python -B -u -m detector.revision_study unsupervised \
  --source-run detector/work/full_v2_seed0_proact38 \
  --output-dir detector/work/revision_shape_unsupervised_seed0_v1
```

This refits MMD, Rank, local shape and shape-MMD in a new directory and compares
them on identical bags. Default size is 150, with 100 bags per fraction and
requested contamination 0%, 1%, 5%, 10%, 25%, 50% and 100%; actual rounded counts
are saved. Labels construct offline mixtures and metrics, not fitted rules.
Other seeds require their own complete inputs and new output directories.
Previously inspected seeds remain rechecks, not untouched confirmation.

## Analysis files

For supervised runs, retain root summary/configuration/state, each source
`fit_metrics.json`, all task evaluation metrics and `rates.csv`; include
explanations when discussing attribution. For label-free runs, retain
`evaluation_metrics.json`, `rates.csv`, `historical_audit.json`, `report.md`,
configuration and state.

Failed directories remain intact; correct the cause and use a fresh output.
Interpret source controls, target clean sample/bag errors, fraction-specific
detection, random alerts and cross-seed variation together. Do not tune on the
same inspected test set until a favourable result appears.
