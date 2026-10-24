# Historical label-free method comparison

This study compares MMD, Rank and local gradient shape using historical inversion
references only. It precedes the current advisor study; see
[README.md](README.md) for the experiment sequence.

## Methods

| Method | Signal | Main limitation |
| --- | --- | --- |
| Original RFF-MMD | Joint descriptor distribution | Task and synthetic-to-real shifts can alert |
| Rank | Pairwise Kendall tau-a dependence | Marginal changes can be invisible; sparse changes are diluted |
| Local shape kNN and count | Per-image relative stage gradients, then an outlier count | Historical shapes and count calibration may not transfer |

All three see the same bags and view assignments. They are not combined by an
OR/AND rule or selected using test performance. Rank is defined in
[UNSUPERVISED_ADAPTATION.md](UNSUPERVISED_ADAPTATION.md).

For local shape, five nonnegative stage norms become `z = g / ||g||_2`.
This deterministic per-image transform discards overall magnitude without fitting
an incoming distribution. Each historical task is split 50% fit, 25% threshold
and 25% calibration, with disjoint originals.

The score is the mean Euclidean distance to k=5 historical neighbours, divided
by sqrt(2), giving a value in [0,1]. Historical threshold data set a conservative
1% empirical tail cutoff; separate calibration data set the bag count cutoff.
Zero-gradient shapes are unsupported.

The count model uses a Clopper–Pearson bound and binomial tail as a historical
working model. Dependent inversions, task heterogeneity and synthetic-to-real
shift prevent interpreting this as a guaranteed 5% incoming FPR. Background:
[nearest neighbours](https://scikit-learn.org/1.0/modules/generated/sklearn.neighbors.NearestNeighbors.html)
and [binomial intervals](https://docs.scipy.org/doc/scipy-1.15.3/reference/generated/scipy.stats._result_classes.BinomTestResult.proportion_ci.html).

## A. Reuse existing features

Use a fresh directory and the intended source version; preserve old artifacts.

```bash
conda activate proact38
cd /home/p.zhang/PROACT

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
python -B -u -m detector.unsupervised_study run \
  --source-run detector/work/full_v2_seed0_proact38 \
  --output-dir detector/work/unsupervised_thesis_seed0
```

The source must contain complete predicted/reference CSVs and matching metadata,
not just reports. MMD is refitted with its original defaults into the new
directory. No supervised detector or true-class extraction is run.

Defaults are 150 images, 100 bags per scenario and requested modification
fractions 1%, 5%, 10%, 25%, 50% and 100%, plus clean controls. Actual counts are
2, 8, 15, 38, 75 and 150. Reusing inspected seed 0 is exploratory.

The output contains `models/legacy/`, `models/rank/`, `models/local/`,
`historical_audit.json`, `evaluation_metrics.json`, `report.md`,
configuration, environment and state. Audits exclude the held-out historical
task. For analysis, provide report, metrics, audit, configuration and state.

## B. Model/attack-seed checks

Changing bag seeds alone is not a new model/attack experiment. The following
historical plan generated seeds 1/2; check whether they have already been used
before registering any new study.

```bash
python -B -m detector.unsupervised_study plan \
  --output-dir detector/work/unsupervised_confirm_v1 \
  --seeds 1 2
```

Register on the GPU server. This writes `protocol.json` and `commands.sh`
without downloading data or starting training. After checking the plan:

```bash
bash detector/work/unsupervised_confirm_v1/commands.sh
```

The script generates checkpoints, inversions, attacks and effectiveness
controls; extracts predicted reference/benchmark features; and compares the
three methods. Relative dataset paths stay inside each artifact directory.
Code/settings are fingerprinted. `prospective_declared` records a declaration,
not proof of experimental independence.

These repeats still share a CIFAR pool and class order. Report them as
model/attack-seed checks rather than independent populations. On failure,
inspect the stage log and generated commands. Bootstrap can run missing stages
with `--stage`; do not overwrite completed artifacts or bypass fingerprints.

## C. Summarise

```bash
python -B -m detector.unsupervised_study summarize \
  --runs detector/work/unsupervised_confirm_v1/study_seed1 \
         detector/work/unsupervised_confirm_v1/study_seed2 \
  --output-dir detector/work/unsupervised_confirm_summary_v1
```

Summary files weight runs equally and report across-run sample SD. One run has
no estimable across-run SD. Duplicate checkpoint/inversion/attack combinations
are not new runs; incompatible protocols and synthetic/real studies cannot be
silently pooled.

The discussion should address reference mismatch, sensitivity lost through
normalisation, clean FPR, detection at each fraction and unsupported coverage.
Neither a high alert rate nor no clean alerts alone establishes useful detection.
Detector-based filtering benefits require a separate intervention experiment.
