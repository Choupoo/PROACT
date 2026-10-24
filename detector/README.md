# Detecting Forgetting Attacks in Continual Learning

This directory contains detector experiments built around PROACT and BrainWash.
The supervised detector uses predicted-class gradients to score images before
the continual learner trains on an incoming task. Separate label-free tests
compare incoming feature statistics with historical inversion references.

## Current experiment

Use [ADVISOR_FINAL_RUN.md](ADVISOR_FINAL_RUN.md) for the registered CIFAR-100
study: five CL methods, seeds 6–8, four BrainWash configurations, feature
ablations, Rank/MMD comparisons and forgetting controls.

For the full-data EWC extension to miniImageNet and tinyImageNet, use
[IMAGENET_EXTENSION.md](IMAGENET_EXTENSION.md) and `run_advisor_imagenet.sh`.
It has separate plans and output directories, one worker per GPU, checked
prepared-data formats, shared controls and recorded stage timings. The
CIFAR-100 entry point remains separate; do not change a running plan's source.

Run commands from the PROACT root on the GPU server:

```bash
cd /home/p.zhang/PROACT
conda activate proact38
python -B -m detector.doctor --require-cuda

# Register a new plan, or verify the existing matching plan.
bash detector/run_advisor_final.sh plan

# Inspect one cell before starting GPU work.
bash detector/run_advisor_final.sh run --method ewc --seed 6 --dry-run
```

For an already registered study, keep its source code, environment and settings
unchanged. Do not copy newer code into a running experiment. The run guide
covers parallel workers, stage recovery and packaging.

## Method at a glance

| Component | Current protocol |
| --- | --- |
| Benchmark | Split CIFAR-100; ten tasks with ten classes each, fixed class order |
| Source training | Task 1; 3,000 clean and 3,000 BrainWash views from 3,000 originals |
| Supervised model | StandardScaler and L2 logistic regression, fitted on source training data only |
| Inputs | 80 backbone tensor-gradient norms, entropy, confidence, margin and activation norm |
| Gradient target | Predicted class from a fixed, untrained incoming-task head |
| Calibration | Separate clean validation and reserve splits from Tasks 1 and 4; fixed `history_mad` rule |
| Final evaluation | Task 9, without detector fitting or recalibration |
| Dataset decision | Count image scores above a sample cutoff; alert at the calibrated count cutoff |
| Label-free comparison | Rank and MMD using historical inversion references |

Tasks are zero-indexed. Each method/seed has its own detector and numerical
thresholds. This is transfer across tasks within each method, not one EWC
classifier applied unchanged to every method. Incoming true-class labels are
not inputs to current feature extraction or scoring.

Each simulated dataset contains 150 distinct originals from a 500-image test
pool. Originals recur across simulations: the rates describe that finite-pool
experiment, not independent tasks or a deployment error guarantee. Logistic
scores and bootstrap p-values are not calibrated poisoning probabilities.
Full-task forgetting controls measure attack damage, not the benefit of
filtering with the detector.

## Reports and methodology

Reports describe artifacts available on their stated dates, not live progress.

| Document | Scope |
| --- | --- |
| [24 October report](results/result-24-10-2026.md) | Complete five-implementation, three-seed CIFAR-100 export; ablations, label-free tests and AFEC/ANCL implementation qualification |
| [18 October report](results/result-18-10-2026.md) | EWC seeds 6–8 and MAS/RWalk seeds 6–7; ablations, Rank/MMD and forgetting |
| [13 October report](results/result-13-10-2026.md) | Earlier partial export: EWC, MAS and RWalk seed 6 |
| [4 October report](results/result-04-10-2026.md) | Predicted-class correction, seed 5 threshold pilot and detailed Rank explanation |
| [3 October methodology](results/methodology-03-10-2026.md) | Pilot design, label access, aggregation and mixture construction |
| [2 October report](results/result-02-10-2026.md) | Historical true-class granularity study and separate Rank checks |
| [21 September report](results/result-21-09-2026.md) | Historical within-task extended-feature study |
| [Initial baseline](results/result.md) | Historical three-feature within-task study |
| [Background chapter](results/background_draft_2026-10-20/README.md) | LaTeX draft, bibliography and integration |

The September and early October supervised studies used true-class information.
Their measurements remain historical evidence, not results for the corrected
inference protocol.

## Earlier workflows

These documents support reproduction of earlier stages, not alternative
defaults for the current experiment.

| Document | Purpose |
| --- | --- |
| [ADVISOR_REVISION.md](ADVISOR_REVISION.md) | Predicted-class feature and threshold pilot |
| [GRANULARITY_STUDY.md](GRANULARITY_STUDY.md) | Eight global/stage/module/tensor comparisons using historical features |
| [MEETING3_PROTOCOL.md](MEETING3_PROTOCOL.md), [MEETING3_RUN.md](MEETING3_RUN.md) | Original source-to-target design and commands |
| [MEETING3_RESULT_AUDIT.md](MEETING3_RESULT_AUDIT.md) | Evidence available at the 24 September audit |
| [MEETING3_EMAIL_DRAFT.md](MEETING3_EMAIL_DRAFT.md) | Archived, unsent meeting follow-up |
| [REVISION_RUN.md](REVISION_RUN.md) | Supervised gradient-shape and historical shape-MMD comparisons |
| [UNSUPERVISED_ADAPTATION.md](UNSUPERVISED_ADAPTATION.md) | Rank definition and standalone evaluation |
| [UNSUPERVISED_THESIS.md](UNSUPERVISED_THESIS.md) | Historical MMD/Rank/local-shape study and seed checks |
| [THESIS_CLOSEOUT.md](THESIS_CLOSEOUT.md) | Historical random-negative ablation and fixed Rank rechecks |
| [THESIS_WRITEUP.md](THESIS_WRITEUP.md) | Thesis structure and use of evidence |
| [VALIDATION.md](VALIDATION.md) | Dated local validation record |

Legacy `bootstrap`, `pipeline`, `reassess` and `predict` modules remain for earlier
artifacts. Use their saved configurations and `--help` when reproducing them.
In particular, `predict.py` does not support current advisor or Rank bundles.
A true-class CSV cannot become predicted-class features by deleting or renaming
columns; re-extraction is required.

## Environment and artifacts

Use the server environment already validated for PROACT. Do not upgrade its
PyTorch/CUDA stack or install newer generic requirements into `proact38` without
checking compatibility. `doctor --require-cuda` checks dependencies and CUDA
availability, not scientific validity.

For local implementation checks:

```bash
python -B -m unittest discover -s detector/tests -q
```

Synthetic tests exercise formulas, split isolation, provenance and command flow;
their scores are not CIFAR-100 results. Dated environments and test counts are
recorded in [VALIDATION.md](VALIDATION.md).

Keep CSVs with their metadata and models with their checksum files. Load only
trusted pickle/joblib artifacts: CPU mapping and checksums do not make untrusted
deserialization safe. Generated outputs belong under `detector/work/`; reports
and writing material belong under `detector/results/`. Upstream programs are
called through adapters rather than edited by the detector workflow.
