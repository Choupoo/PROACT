# CIFAR-100 multi-method study

This guide covers the registered experiment following the feature and threshold
pilot. It runs Split CIFAR-100 only. EWC on miniImageNet and tinyImageNet, with
all four attack settings, requires separate dataset support.

## Protocol

| Component | Setting |
| --- | --- |
| Dataset | Split CIFAR-100; fixed class order, ten tasks with ten classes each |
| CL methods | EWC, MAS, RWalk, AFEC-EWC and ANCL-EWC |
| Seeds | 6, 7 and 8 |
| Source | Task 1 training split; fit StandardScaler and logistic regression |
| Calibration | Tasks 1 and 4 clean validation and reserve splits |
| Final evaluation | Task 9; no fitting, recalibration or feature selection |
| Positive training views | Source reckless BrainWash, L-infinity budget 0.3 |
| Negative training views | Corresponding unmodified real images; no random negatives |
| Target attacks | Reckless/cautious × L-infinity budgets 0.1/0.3 |
| Inputs | 80 backbone tensor-gradient norms, entropy, confidence, margin and activation norm |
| Gradient target | Predicted class; no incoming true-class labels or true-class probability |
| Threshold rule | Fixed `history_mad`; two other rules retained as comparisons |
| Dataset simulations | 150 distinct originals per bag, 100 bags per modification fraction |
| Label-free tests | Rank and original MMD; historical Task 0–8 inversion references |
| Forgetting controls | Full incoming-task training with clean, BrainWash and uniform-perturbed images |

Task indices start at zero. The 15 method/seed cells contain 60 target attack
configurations, plus Task 1 and Task 4 attacks for each cell: 90 attack-generation
stages. There are 36 stages per cell and 540 overall. Counts are not proportional
to elapsed time.

Each cell fits its own detector, shared by its four target attacks. The fixed
choice is the threshold rule, not the pilot values 0.496810 and 4. Numerical
cutoffs come from each cell's historical clean data. Development failures remain
in the results; target performance does not select a replacement rule.

| CLI method | Lambda | Auxiliary lambda |
| --- | ---: | ---: |
| ewc | 500000 | Not used |
| mas | 10 | Not used |
| rwalk | 1 | Not used |
| afec_ewc | 500000 | 100 |
| ancl_ewc | 500000 | 100 |

Victim training uses 100 epochs per task, learning rate 0.01 and batch size 16.
Inversion uses 2,000 iterations; BrainWash uses 5,000 epochs. The cautious weight
`w_cur` is 1. These settings cover the method/attack axes of the benchmark, not
every dataset, optimal hyperparameter or original BrainWash table value.

## Feature ablations

Each variant refits its scaler and classifier on the same source split, then
recalibrates with the same historical splits and rule. It does not inherit the
full model's numerical cutoff. Supervised variants share target bags.

| Variant | Inputs | Dimensions |
| --- | --- | ---: |
| inference_full | Tensor norms and four context features | 84 |
| inference_gradients | Tensor norms only | 80 |
| inference_context | Entropy, confidence, margin and activation norm | 4 |
| inference_stage | Five stage norms and context; a granularity control | 9 |
| inference_without_entropy | Full model without entropy | 83 |
| inference_without_confidence | Full model without confidence | 83 |
| inference_without_margin | Full model without margin | 83 |
| inference_without_activation_norm_l2 | Full model without activation norm | 83 |
| inference_without_uncertainty | Full model without entropy, confidence and margin | 81 |

Rank keeps its ten descriptors and 499 bootstrap draws. These supervised
ablations do not select new Rank inputs from target results.

## Register and inspect

For a new study, place the intended code version on the server before registering.
Preserve existing outputs. Do not update code or dependencies in an active study.

```bash
cd /home/p.zhang/PROACT
conda activate proact38
python -B -m detector.doctor --require-cuda
python -B -m unittest detector.tests.test_advisor detector.tests.test_advisor_final -q
bash detector/run_advisor_final.sh plan
bash detector/run_advisor_final.sh run --method ewc --seed 6 --dry-run
```

The plan is `detector/work/advisor_final_v1/plan.json`. Register on the server so
the saved interpreter and project paths refer to that machine. Confirm seed use
before starting a new study and retain the compatible `proact38` environment.

## Run on one GPU

```bash
tmux new -s detector-final
cd /home/p.zhang/PROACT
conda activate proact38
CUDA_VISIBLE_DEVICES=0 bash detector/run_advisor_final.sh run
```

Do not create nested tmux sessions. Detach with `Ctrl+B`, then `D`; `exit` closes
the shell. An unfiltered full run summarises on completion. Failures stop the
script and leave stage logs.

## Run cells in parallel

Register once, then run each command in a separate idle tmux window:

```bash
CUDA_VISIBLE_DEVICES=0 bash detector/run_advisor_final.sh run --method ewc --seed 6
CUDA_VISIBLE_DEVICES=1 bash detector/run_advisor_final.sh run --method mas --seed 6
```

Use explicit `--method` and `--seed` when redistributing work. Without `--seed`,
a method worker visits all its seeds sequentially and may later reach a cell
assigned elsewhere. The cell lock rejects duplicate execution; it is not a work
queue. Check existing workers first. Logical CUDA device 0 can correspond to the
physical GPU selected by `CUDA_VISIBLE_DEVICES`.

For different seeds or settings, register a new output directory:

```bash
export DETECTOR_FINAL_OUTPUT=detector/work/advisor_final_v2
bash detector/run_advisor_final.sh plan --seeds 9 10 11
bash detector/run_advisor_final.sh run
```

These seed numbers are examples, not a record that they are unused. Never edit
saved fingerprints to make changed code or settings pass an old plan.

## Resume an interrupted cell

Inspect the log and confirm that no worker still owns the cell, then run:

```bash
bash detector/run_advisor_final.sh run --method ewc --seed 6 --retry-failed
```

Verified completed stages are reused. Exclusive outputs of an incomplete stage
move to `interrupted/`, and that stage restarts from the beginning. This is not
epoch-level checkpoint recovery; shared datasets are not deleted.

## Summarise and package

After all workers finish:

```bash
bash detector/run_advisor_final.sh summarize
python -B -c 'import json; d=json.load(open("detector/work/advisor_final_v1/analysis/summary.json")); print("status:", d["status"]); print("missing:", d["missing"])'
bash detector/run_advisor_final.sh package
```

`status: complete` and `missing: []` indicate completed stages, not a performance
target. Packaging rebuilds the summary and checks registered code and completed
artifacts. The printed archive is named
`detector/work/advisor_final_v1_analysis_<UTC timestamp>.tar.gz`.

For an interim diagnostic export:

```bash
bash detector/run_advisor_final.sh package --allow-incomplete
```

The package includes the plan, summaries, states, environments, logs, frozen
configuration JSON, predictions, bag decisions, feature metadata/manifests,
label-free results, historical descriptors and small accuracy matrices. It
excludes datasets, checkpoints, noise, full feature matrices and joblib models.
A SHA-256 manifest accompanies it. This is an analysis package, not a full
reproduction archive or an atomic snapshot of active workers.

### NumPy serialization in older registered versions

If an old summary fails with `Object of type int64 is not JSON serializable`,
leave its fingerprinted Python files unchanged. A newer JSON writer exists, but
replacing it inside that run would invalidate the plan.

If present on the server, use the standalone compatibility wrapper:

```bash
bash detector/package_advisor_compat.sh --allow-incomplete
```

Only this additional wrapper needs copying if absent. It converts NumPy values
inside the CPU packaging process while retaining source and artifact checks.

## Result locations

| Location | Contents |
| --- | --- |
| analysis/summary.json | Completion, per-configuration results, cross-seed summaries and ablation differences |
| analysis/report.md | Supervised, label-free, ablation and forgetting tables |
| analysis/supervised.csv | Supervised measurements |
| analysis/paired_ablation_deltas.csv | Within-seed variant-minus-full comparisons |
| &lt;method&gt;/seed&lt;seed&gt;/frozen_supervised/ | Features, models and historical calibration |
| &lt;method&gt;/seed&lt;seed&gt;/task9/&lt;attack&gt;/evaluation/ | Sample predictions, bag decisions and label-free tests |
| &lt;method&gt;/seed&lt;seed&gt;/task9/&lt;attack&gt;/effectiveness/ | Full-task clean, poison and uniform accuracy matrices |
| &lt;method&gt;/seed&lt;seed&gt;/logs/ | Stage logs |

Analyse ranking, frozen-threshold transfer, sparse-poisoning detection, ablations
and clean-reference mismatch separately. Bags share originals and do not add
independent model/attack runs. Random perturbations can cause forgetting; report
their alerts separately from clean FPR. Forgetting controls do not establish
the benefit of detector-based filtering.
