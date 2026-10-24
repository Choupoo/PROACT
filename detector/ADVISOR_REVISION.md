# Feature correction and threshold pilot

This is the historical single-configuration pilot. Use
[ADVISOR_FINAL_RUN.md](ADVISOR_FINAL_RUN.md) for the subsequent multi-method study.
The [methodology note](results/methodology-03-10-2026.md) defines label access,
the two thresholds, Rank and the repeated dataset simulations.

## Changes from the earlier detector

Gradients use the predicted class, and inference CSVs omit
`true_class_probability`. The full classifier uses one gradient norm per
backbone weight/bias tensor, entropy, confidence, margin and activation norm;
loss and historical cosines are not classifier inputs.

The detector is fitted on Task 1, calibrated and developed using historical
tasks, then frozen for Task 9. The default pilot is EWC, reckless BrainWash,
L-infinity budget 0.3 and seed 5. It does not automatically run the full attack
matrix, label-free reevaluation, forgetting controls or ablations.

## Reuse compatible artifacts

Checkpoints, attack artifacts and matching inversion NPZs can be reused for
feature extraction. Old true-class feature CSVs cannot be reused as predicted
features. Replace the placeholders with matching server paths and task indices:

```bash
python -B -u -m detector.transfer_extract \
  --checkpoint <checkpoint.pkl> \
  --artifact <noise.pkl> \
  --inversion-dir <inversion_directory> \
  --incoming-task <task_index> \
  --label-mode predicted --inference-schema --random-control uniform \
  --output-dir detector/work/feature_threshold_pilot/feature_task<task_index>
```

After extracting Tasks 1, 4 and 9:

```bash
python -B -u -m detector.advisor_detector \
  --source detector/work/feature_threshold_pilot/feature_task1/features.csv \
  --histories detector/work/feature_threshold_pilot/feature_task4/features.csv \
  --output-dir detector/work/feature_threshold_pilot/frozen

python -B -u -m detector.advisor_study evaluate \
  --features detector/work/feature_threshold_pilot/feature_task9/features.csv \
  --frozen detector/work/feature_threshold_pilot/frozen \
  --output-dir detector/work/feature_threshold_pilot/evaluation
```

This fits the corrected full model and three threshold rules. Task 4 is a
development choice fixed before target evaluation, not a task number mandated by
the supervisor. Reusing previously inspected seeds 3/4 is an exploratory check.

## Generate missing artifacts

Register on the GPU server, using the intended code version and existing
compatible environment. Do not update an active registered run.

```bash
cd /home/p.zhang/PROACT
conda activate proact38
python -B -m detector.doctor --require-cuda
python -B -m unittest detector.tests.test_advisor

python -B -m detector.advisor_study plan \
  --output-dir detector/work/feature_threshold_v1

python -B -m detector.advisor_study run \
  --plan detector/work/feature_threshold_v1/plan.json --dry-run
```

Defaults are seed 5, source Task 1, development Task 4 and target Task 9; victim
training uses 100 epochs, inversion 2,000 iterations and attack optimisation
5,000 epochs. Check that the seed and output directory are appropriate.

In an idle tmux shell:

```bash
python -B -u -m detector.advisor_study run \
  --plan detector/work/feature_threshold_v1/plan.json

python -B -m detector.advisor_study summarize \
  --plan detector/work/feature_threshold_v1/plan.json
```

Recovery is stage-level: after inspecting a failed stage, add `--retry-failed`
to the run command. Exclusive incomplete outputs are archived under
`interrupted/`; completed stages are hash-checked and reused. Changed code,
settings or dependencies require a new study directory.

## Analysis files

Under `detector/work/feature_threshold_v1/`, inspect:

- `plan.json` and `ewc/seed5/run_state.json`, `run_environment.json`, `logs/`.
- `ewc/seed5/frozen_supervised/`: `threshold_selection.json`,
  `development_metrics.json` and `freeze.json`.
- `ewc/seed5/task9/reckless_0p3/evaluation/`: frozen evaluations.
- `analysis/`: `summary.json`, `report.md` and `supervised.csv`.

For a manual artifact-reuse run, provide the equivalent frozen JSON, evaluation
directory and each task's feature metadata; full CSVs may be needed to recompute
results. CPU tests do not replace server GPU validation.
