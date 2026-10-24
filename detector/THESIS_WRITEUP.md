# Thesis structure and use of evidence

Working title: *Detecting Forgetting Attacks in Continual Learning*.

Use the dated [report index](README.md#reports-and-methodology) to identify the
protocol behind each measurement. The September true-class studies and the
October predicted-class studies should not be pooled as one experiment.

## Research questions

1. Can predicted-class gradient features distinguish clean and BrainWash-poisoned
   images before the incoming task is learned?
2. Do parameter-tensor norms retain more useful information than stage norms?
3. Can thresholds calibrated on earlier tasks support decisions on a later task?
4. What can historical inversion references detect without labelled clean/poison
   examples or a known-clean incoming calibration set?

The first three concern a supervised detector fitted with source clean/poison
labels. The fourth concerns a separate label-free test. Neither uses incoming
image-class labels at inference in the corrected protocol.

## Chapter responsibilities

| Chapter | Material |
| --- | --- |
| Introduction | Forgetting attacks, motivation, questions and contribution scope |
| Background and Related Work | CL, forgetting, BrainWash, inversion, gradient descriptors and statistical testing |
| Problem Setting | Attacker access, clean-history assumption, available labels and task stages |
| Detection Methods | Fixed-head predicted gradients, tensor norms, LR, historical threshold rule and Rank |
| Experimental Setup | Splits, methods, datasets, attacks, seeds, optimisation, code provenance and simulations |
| Results and Discussion | Ranking, threshold transfer, bag decisions, ablations, forgetting, failures and limitations |
| Conclusion and Future Work | Supported findings and remaining research questions |

The background draft is under
[results/background_draft_2026-10-20/](results/background_draft_2026-10-20/README.md).
Explain original AFEC/ANCL algorithms separately from any issues found in their
upstream implementations.

## Evidence by experiment

| Source | Evidence | Interpretation |
| --- | --- | --- |
| Initial baseline and 21 September report | Within-task true-class features | Historical motivation, not corrected inference performance |
| 2 October report | Seeds 3/4 granularity and SHAP; separate label-free seeds 1/2 | Tensor/module ranking and source-to-target calibration failures |
| 4 October report | Seed 5 predicted-feature and threshold pilot | Method development, not multi-setting confirmation |
| 13 October report | EWC/MAS/RWalk seed 6 | Partial export of the fixed expansion |
| 18 October report | EWC seeds 6–8, MAS/RWalk seeds 6–7 | Later partial export, including weak-budget failures |
| Subsequent exports | Only complete, provenance-checked cells | Update tables when the corresponding evidence is available |

The 26 September shape comparison remains useful as a historical negative
control:

| Seed | Features | AUC | Clean FPR | Poison TPR | Clean bag alerts |
| --- | --- | --- | --- | --- | --- |
| 3 | portable | 0.987932 | 18.6% | 99.2% | 100% |
| 3 | extended | 0.987856 | 24.6% | 99.2% | 100% |
| 3 | shape | 0.987428 | 8.4% | 96.6% | 14.4% |
| 4 | portable | 0.961952 | 12.8% | 95.0% | 99.5% |
| 4 | extended | 0.961676 | 12.8% | 95.4% | 99.5% |
| 4 | shape | 0.964844 | 11.6% | 97.0% | 93.5% |

These values came from `revision_results_20260926_101113`. Shape changed several
feature families as well as scale; it did not isolate gradient magnitude as a
cause. Shape random-sample alerts were 57% and 51%.

The same historical seed 0 label-free comparison had 100% alerts for
MMD/local/shape-MMD in all scenarios. Rank observed 0/100 clean alerts and 0%–1%
random alerts, with the following poison rates:

| Actual poisoned fraction | 1.33% | 5.33% | 10% | 25.33% | 50% | 100% |
| --- | --- | --- | --- | --- | --- | --- |
| Rank detection | 0% | 0% | 2% | 5% | 31% | 96% |

Treat this as a dated observation, not the headline estimate: later seeds show
substantial variation. Historical shape-MMD accepted all nine held-out inversion
tasks while alerting on all real clean bags. This is consistent with reference
mismatch, but does not isolate inversion quality, class shift or model adaptation
as the sole cause.

## Reporting rules

Report each method/seed and actual poisoned count, not just its best AUC or an
average hiding failures. State how ablations are retrained and recalibrated.
SHAP concerns fitted log-odds, not causal effects of layers.

Distinguish sample ranking, fixed-threshold decisions and dataset decisions.
A low clean alert rate without poison sensitivity is not useful detection.
Historical MAD and binomial count calculations do not guarantee target FPR.
Randomly perturbed images may cause forgetting and should have their own alert
measurements, rather than automatically count as harmless negatives.

A bag contains distinct originals, but bags share a finite pool and attack.
Cross-seed SD describes model/attack repeats; it does not turn bags into
independent tasks. An observed zero alert count is not a population-zero claim.

The agreed expansion is full CL-method/attack coverage on CIFAR-100 and EWC
with all four attacks on miniImageNet and tinyImageNet. Keep full training data;
any reduction in seed count should be documented. Report incomplete work as
pending, not as a negative result.

Detection experiments do not establish that filtering preserves past accuracy.
That intervention, new task orders and broader attack families remain separate
claims requiring separate evidence.
