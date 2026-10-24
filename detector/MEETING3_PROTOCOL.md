# Meeting 3 protocol: source-to-target detection

Historical protocol based on `meeting3_transcript.pdf`, particularly 14:24–17:45.
It predates the removal of true-class dependence. Server commands are in
[MEETING3_RUN.md](MEETING3_RUN.md); the unsent correspondence is retained in
[MEETING3_EMAIL_DRAFT.md](MEETING3_EMAIL_DRAFT.md).

## Research question

Can a detector trained on an early task's clean and artificially poisoned images
transfer to a later task more reliably than tests against historical inversion
references? This is supervised source-to-target detection without target
poisoning labels, not fully unsupervised learning.

The MMD, Rank and local-shape comparisons remain separate. Original MMD already
uses detector descriptors rather than complete backbone embeddings.

## Original design

- Fixed CIFAR-100 class order; ten tasks with ten classes each. Task indices are
  zero-based: source Task 1 follows clean Task 0; target Task 9 follows Tasks 0–8.
- Source and target classes are disjoint, but the later backbone has legitimately
  learned the source clean classes/images. Entire training histories are not
  independent.
- Historical training remains clean. Source poisoning trains the detector; the
  attacked source model is not propagated into the later defender.
- Seeds 3/4 supplement earlier observations on seeds 0–2. The same image pool,
  class order and previously studied target task are reused.
- EWC uses 20 epochs per task, inversion 10,000 iterations, and BrainWash
  5,000 epochs at budget 0.3.

The original main branch uses `ground_truth` class labels for CE/gradients.
It excludes target clean/poison labels from fitting and calibration, but is not
class-label-free. `predicted` is a separate option; current studies require it.
The incoming head is fixed-seed initialised and untrained on that task.

## Splits and classifier

Each task has 5,000 originals: 3,000 train, 500 validation, 500 test and 1,000
reserve, corresponding to 300/50/50/100 per class. Clean, poison and norm-matched
random views of an original remain in one split.

StandardScaler and logistic regression use source train clean/poison only.
Source validation-clean sets a sample cutoff targeting 5% empirical FPR;
separate reserve-clean calibrates the count cutoff for 150-image datasets.
Models and thresholds are saved before target evaluation. Target labels serve
only offline metrics and mixture construction.

| Feature set | Dimensions | Inputs |
| --- | ---: | --- |
| portable | 10 | Five stage norms, four uncertainty features and activation norm |
| extended | 16 | Loss, global norm, historical cosine, three task-cosine summaries, four uncertainty features, activation norm and five stage norms |

The uncertainty group here includes true-class probability. Parameter and module
norms are also extracted for analysis. Portable scores omit inversion-derived
cosines, but attack generation and the extended control still use inversions.
At source Task 1, task-cosine min/max/mean coincide because only Task 0 exists;
Task 9 has nine reference directions.

## Feature interpretation

The frozen LR uses exact interventional linear SHAP in log-odds space:

```text
phi_j = beta_j * (standardized_x_j - mean_training_standardized_x_j)
```

The background comes from source training; validation is the default explanation
split. Attributions sum to each prediction's logit. Family importance sums mean
absolute feature attributions rather than jointly cancelling them. Correlated
features can substitute for one another; attribution is not causal evidence.
See [SHAP LinearExplainer](https://shap.readthedocs.io/en/latest/generated/shap.LinearExplainer.html).

`--ablations` compares groups on source validation without changing the registered
model. Test-split explanations are descriptive and cannot be combined with model
selection.

## Evaluation and assumptions

Report sample AUC, frozen-cutoff FPR/TPR, random alerts, dataset curves and paired
clean/poison full-task forgetting. Dataset simulations use 150 distinct
originals, 1,000 bags per requested fraction of 0%, 1%, 5%, 10%, 25%, 50% and 100%;
record the rounded counts.

Source count calibration assumes a stable clean alert probability and does not
guarantee target FPR. Bags reuse a finite pool. Attack optimisation sees the
incoming training pool, including detector-test originals: this is detector
split isolation, not an unseen-image attack-generation experiment. Forgetting
controls do not test detector-based filtering.

## Provenance and recovery

`transfer_upstream` adapts the dataset in a subprocess while calling unmodified
PROACT scripts. It constructs all ten tasks before exposing a task prefix;
reducing `tasknum` to two would incorrectly change the classes per task.
Effectiveness training continues on one incoming task only.

Source and target checkpoints must differ and advance in task index. Class
order, protocol, head seed and label mode must match; image UIDs include task
index. Source/configuration and successful-output hashes are registered.
Recovery checks those hashes and the environment. Failed steps are not silently
overwritten, and epoch-level recovery is not supported.
