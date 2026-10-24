# Gradient-granularity comparison

Historical study: this reuses the true-class features from seeds 3/4. It is not
the corrected predicted-class experiment. See
[ADVISOR_FINAL_RUN.md](ADVISOR_FINAL_RUN.md) for current ablations.

## Design

| Gradient representation | Norms only | With context |
| --- | --- | --- |
| One backbone L2 norm | norm_global | norm_global_context |
| Five norms: stem and layer1–4 | norm_stage | norm_stage_context |
| Named modules, pooling weight/bias gradients | norm_layer | norm_layer_context |
| One norm per named parameter tensor | norm_parameter | norm_parameter_context |

Context comprises entropy, confidence, true-class probability, margin and
activation norm. Parameter features distinguish a module's weight and bias
tensors; module norms pool them. Neither representation uses one feature per
scalar weight. The supervisor subsequently confirmed the per-tensor meaning.

Each comparison uses seeds 3/4, zero-indexed Task 1 → Task 9, source clean/poison
training, StandardScaler and LogisticRegression(C=1, L2, lbfgs, max_iter=2000).
Source validation-clean selects the sample threshold; separate reserve-clean
calibrates the count at alpha=0.05. Source/target test and random controls are
evaluation-only. Nonconvergence fails the run.

Feature dimensions differ, so this is not a comparison with equal coefficient
counts. Hyperparameters are fixed across representations. Feature schemas are
frozen from source metadata rather than selected on target performance.

## Run with existing features

Use the intended code version before starting a new run. Preserve existing
outputs and do not alter active registered experiments.

```bash
conda activate proact38
cd /home/p.zhang/PROACT
bash detector/run_granularity.sh
```

Required CSVs, each with its metadata:

```text
detector/work/meeting3_transfer_v1/seed3/features_task1/features.csv
detector/work/meeting3_transfer_v1/seed3/features_task9/features.csv
detector/work/meeting3_transfer_v1/seed4/features_task1/features.csv
detector/work/meeting3_transfer_v1/seed4/features_task9/features.csv
```

Inputs must contain `grad_norm_layer__*` and `grad_norm_param__*`. Report-only
packages are insufficient. This CPU comparison does not retrain PROACT, regenerate
attacks or re-extract features. Dry-run checks files and hashes; actual loading
validates the schema and rejects missing columns.

For another input root or an explicit, unused output directory:

```bash
THESIS_TRANSFER_RUN=detector/work/alternative_transfer_run \
  bash detector/run_granularity.sh

bash detector/run_granularity.sh detector/work/granularity_v1
```

Default outputs use `detector/work/granularity_<date>_<time>/`. After interruption,
retain the failed directory and rerun into a new one.

## Outputs and interpretation

- `run_config.json`, `run_environment.json`, `run_state.json`: settings and state.
- `granularity_summary.md/json/csv`: both tasks, eight variants and both seeds.
- `paired_differences_vs_stage.csv`: within-seed differences, means and sample SD.
- `shap_summary.md`, `shap_by_seed.csv`: source-validation attributions.
- `professor_update.md`: a generated draft, not an automatically sent email.
- `seed3/`, `seed4/`: frozen models, predictions, figures and detailed results.

Check completion, then compare AUC, clean FPR, poison TPR, random alerts and
dataset detection at every fraction. Better ranking does not imply better
threshold transfer. The two-seed SD is descriptive; repeated bags do not create
new model runs. SHAP explains fitted scores, not the cause of forgetting.

For an explicit output directory:

```bash
tar -czf detector/work/granularity_v1_results.tar.gz \
  -C detector/work granularity_v1
```
