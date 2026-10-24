# Rank: historical rank-dependence comparison

Rank is a label-free dataset test. It uses neither labelled clean/poison training
examples nor a known-clean incoming subset. Supervised models and original MMD
artifacts remain separate. Current study results are indexed in
[README.md](README.md); these commands reproduce the earlier standalone workflow.

## Statistical profile

Each image supplies ten predicted-class descriptors: global gradient norm, five
stage norms, entropy, confidence, margin and activation norm. Each historical
task has a separate inversion reference.

For each of the 45 feature pairs, Kendall tau-a averages

```text
h(i,j) = sign(x[i,a] - x[j,a]) * sign(x[i,b] - x[j,b])
```

over distinct image pairs; ties contribute zero. Strictly increasing column-wise
transformations preserve these correlations. Historical references alone select
pairs with nondegenerate jackknife variation in every task, then freeze them.

Let `hbar_i` be the average kernel value with image i fixed, and tau its overall
average. The implementation uses:

```text
centered_pseudovalue_i = 2(n-1)/(n-2) * (hbar_i - tau)
bootstrap_error = sum_i(normal_i * centered_pseudovalue_i) / sqrt(n(n-1))
observed = max_pair abs(tau_incoming - tau_reference)
bootstrap_null = max_pair abs(error_incoming - error_reference)
p_approx = (1 + count(bootstrap_null >= observed)) / (1 + draws)
p_union = max_task p_approx(task)
```

A draw shares image multipliers across pairs to retain their joint variation;
incoming and reference multipliers are independent. The maximum-p rule alerts
only if every historical comparison rejects at alpha=0.05. The null allows at
least one matching historical profile; failure to reject is not proof of safety.

This is approximate Gaussian multiplier bootstrap inference, not an exact
permutation test. It requires independent observations and nondegenerate
first-order variation, neither guaranteed for inversion samples. See
[Chen (2018)](https://arxiv.org/abs/1610.00032) for related U-statistic
bootstrap theory, not a validation theorem for this particular detector.

## Limitations

A clean task unlike every reference can alert. Conversely, marginal changes
preserving pairwise ordering, changes confined to higher-order dependence, or
sparse contamination may be missed. Matching any reference prevents an alert,
which can reduce power. Constant or otherwise unsupported features produce no
decision and are reported through coverage, not counted as correctly clean.
P-values measure disagreement with the reference statistics, not poisoning
probabilities or a guaranteed clean FPR.

## Standalone evaluation

Use the intended code before starting a new run, preserving previous outputs.
These commands reuse features; they do not run PROACT training, inversion,
attacks or supervised fitting.

```bash
conda activate proact38
cd /home/p.zhang/PROACT

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
python -B -m detector.unsupervised_adapt run \
  --source-run detector/work/full_v2_seed0_proact38 \
  --output-dir detector/work/unsupervised_rank_v1 --dry-run

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
python -B -u -m detector.unsupervised_adapt run \
  --source-run detector/work/full_v2_seed0_proact38 \
  --output-dir detector/work/unsupervised_rank_v1
```

Inputs:

```text
full_v2_seed0_proact38/
  reference_features.csv + reference_features.metadata.json
  predicted_features.csv + predicted_features.metadata.json
  unsupervised/unsupervised_bundle.joblib + .joblib.sha256.json
```

Reference identities use `taskN:sampleM`. Both feature files must use predicted
labels and matching frozen defender/head/inversion provenance. Rank freezes
before benchmark values are read. Labels only construct and score offline bags.

Defaults: alpha=0.05, 499 draws, 150 images and 100 bags per scenario. Requested
1%, 5% and 25% become 2/150, 8/150 and 38/150. Rank and MMD share bags.
A requested nonzero fraction rounded to zero at small bag sizes is skipped.
Output directories must be new; failed results are retained rather than overwritten.

Outputs include `rank/rank_bundle.joblib` and checksum,
`historical_audit.json`, `evaluation_metrics.json`, `report.md`,
`run_config.json`, `run_environment.json` and `run_state.json`. The audit
holds out historical tasks, not incoming clean examples. Provide the report,
metrics, audit and state for initial analysis.

## One incoming dataset

Extract a single-view predicted feature file, not a multi-view benchmark, then:

```bash
python -B -m detector.unsupervised_adapt predict \
  --bundle detector/work/unsupervised_rank_v1/rank/rank_bundle.joblib \
  --features detector/work/incoming/predicted_features.csv \
  --output detector/work/incoming/rank_prediction.json
```

At least 16 images are required. Standalone prediction deterministically
subsamples above the frozen incoming limit and reports the actual size;
research evaluation does not silently truncate bags.

Compare clean alerts, poison detection, random alerts and coverage together.
No alerts on either clean or poisoned data indicate lost sensitivity, not a
successful detector. This method was developed after earlier test failures;
reuse of those data is exploratory, even if the results improve.
