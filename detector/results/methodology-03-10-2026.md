# Detector methodology: feature correction and threshold pilot

This note describes the revised implementation and a proposed small pilot. It does not report new GPU results. Following the latest feedback, the multi-method, multi-attack experiments and feature ablations are deferred until we agree on the methodology.

## 1. What the detector is allowed to know

There are two different kinds of labels in this experiment:

- **Image-class labels**, such as the CIFAR class of an image. The incoming detector must not use these.
- **Clean/poison labels**, which are available for the controlled source experiment and are used to train the supervised detector. They are not available to the detector at deployment. In offline evaluation, they are used to construct mixtures and calculate metrics.

The previous supervised results used true-class gradients. Removing only the true-class probability would not correct this. Those results remain historical diagnostics and cannot establish the performance of the corrected detector.

### Corrected feature extraction

For each incoming image, I run the frozen defender in evaluation mode and take the class with the highest predicted probability:

$$
\hat y(x)=\arg\max_c p_\theta(c\mid x),\qquad
\ell(x)=-\log p_\theta(\hat y(x)\mid x).
$$

The predicted class is treated as a fixed target for differentiation. For each backbone weight or bias tensor, I calculate the Euclidean norm of its gradient:

$$
g_j(x)=\left\|\nabla_{\theta_j}\ell(x)\right\|_2.
$$

This is one norm per tensor, as requested, not one feature per scalar weight. The classifier-head parameters are excluded. There is no optimizer step, and the model and BatchNorm statistics remain unchanged.

The corrected full supervised model uses the tensor norms and four additional descriptors:

| Feature | Definition |
| --- | --- |
| Entropy | $-\sum_c p_c\log p_c$; spread of the predicted class distribution |
| Confidence | $\max_c p_c$; the largest predicted probability |
| Margin | Difference between the largest and second-largest probabilities |
| Activation norm | L2 norm of the representation entering the incoming-task classifier head |

The new inference CSV omits `true_class_probability`. The detector also excludes loss and historical-gradient cosine features. The predicted-target loss is used to obtain gradients, not as a classifier input. This avoids adding both confidence and its deterministic transform, negative log confidence, to the model. Additional gradient summaries stored in the CSV are diagnostic descriptors, not automatically selected inputs.

The incoming head is initialized with a fixed seed at the pre-training checkpoint. It has **not** been trained using incoming-task labels. Its confidence should therefore not be described as calibrated classification confidence. Whether these predicted-gradient descriptors remain useful must be checked by rerunning the detector.

Image-class labels still appear in offline, class-stratified split construction and in the original CL training benchmark. They are not passed to predicted-gradient extraction or to the deployment scoring function. Historical inversion labels may be used to construct legacy reference-gradient diagnostics; those diagnostics are not inputs to either revised detector.

## 2. Supervised decisions: from images to a dataset

The supervised detector first maps each image's descriptors to a score using a source-fitted StandardScaler and logistic regression. This is a sample-level risk score, not a calibrated deployment probability of poisoning.

For an incoming dataset of $n$ images, the image scores are thresholded and counted:

$$
K=\sum_{i=1}^{n}\mathbf{1}\{s(x_i)\geq t\}.
$$

The dataset triggers an alert when $K\geq k$. Thus there are **two thresholds**: the sample threshold $t$, and the suspicious-image count $k$. A sample ROC-AUC alone does not establish whether either threshold works after task transfer.

### Proposed threshold pilot

The default pilot uses Split CIFAR-100, EWC and reckless BrainWash with an L-infinity budget of 0.3. Tasks are numbered from zero. The detector is fitted on Task 1; Task 4 is a development task; Task 9 is reserved for the final frozen evaluation in this pilot. This is a follow-up on a previously studied benchmark, not an independent new population.

Each task's original images have disjoint train, validation, test and reserve splits. All views of the same original stay in the same split. The source classifier is fitted only on source-training clean and poisoned views.

I have implemented three candidate threshold rules:

1. **Source quantile:** use source validation-clean scores to set $t$, targeting at most 5% empirical sample alerts. Ties are handled conservatively.
2. **Historical quantiles:** compute the same cutoff separately on source and development validation-clean samples, then use the largest cutoff.
3. **Historical median/MAD:** for each calibration task, compute median plus three scaled MADs on clipped logit scores. Use the larger of this cutoff and that task's clean quantile, then take the largest cutoff across tasks. This is a robust heuristic, not a distribution-free guarantee.

After fixing each sample threshold, I use the separate reserve-clean images to estimate the clean sample-alert probability. A one-sided Clopper–Pearson upper bound and a binomial tail determine the dataset count cutoff. For the historical rules, I use the most conservative task-wise bound. The nominal 5% error budget is split equally between estimation and the count-test tail. These calculations rely on assumptions about independent images and stable alarm probabilities; they do not guarantee 5% false positives on a shifted target task.

The candidate rule is selected using the development task's test split: first require both sample and dataset clean-alert rates to be at most 5%, then compare mean dataset detection at 10%, 25%, 50% and 100% contamination. If no rule meets the clean-error criterion, the run explicitly records that failure. It does not describe the chosen fallback as a successful calibration.

After selection, the scaler, classifier, sample cutoff and count cutoff are frozen. Task 9 does not change them. Results for all three frozen rules are retained for transparency; the target results must not be used to select another winner. A useful rule needs both an acceptable clean-alert rate and nontrivial detection power, not simply a very high threshold that never alerts.

Historical known-clean calibration is an assumption of this **supervised** pilot. It is not part of the strictly label-free Rank method below.

## 3. Rank: a direct dataset-level test

Rank does **not** classify each image and then vote. It extracts a descriptor vector for each image, constructs a statistical profile of the whole dataset, and compares that profile with historical references.

The descriptors are the global backbone gradient norm, five stage gradient norms, entropy, confidence, margin and activation norm: ten features in total. Gradients use the predicted class. Neither true image-class labels nor clean/poison labels are inputs to the test.

For every pair of features, I calculate Kendall's tau-a across the images in the dataset. Ten features give at most 45 feature pairs. For features $a$ and $b$, this measures whether their values tend to increase or decrease together across images:

$$
\hat\tau_{ab}(D)=\frac{2}{n(n-1)}\sum_{i<j}
\operatorname{sign}(z_{ia}-z_{ja})\operatorname{sign}(z_{ib}-z_{jb}).
$$

Tied values contribute zero. The resulting vector of correlations is the dataset's profile; it is not a vector of sample poison predictions.

For Task 9, each historical task from 0 to 8 supplies a separate reference made from inversion images. Reference and incoming images are passed through the **same frozen Task 9 defender and incoming head**. The test keeps at most 256 inversion images per historical task and retains feature pairs with nondegenerate jackknife variation in every historical reference. This selection uses historical references only.

For each historical task $h$, the test statistic is the largest absolute change among the retained correlations:

$$
T_h(D)=\max_{(a,b)\in A}\left|\hat\tau_{ab}(D)-\hat\tau_{ab}(R_h)\right|.
$$

I use 499 Gaussian multiplier draws of centered jackknife pseudovalues to approximate its null distribution. The per-task value is $(1+\text{number of bootstrap statistics at least as large as }T_h)/500$. This is an **approximate p-value**, not an exact permutation test or a poisoning probability.

The final value is the maximum across historical tasks:

$$
p_{\mathrm{combined}}(D)=\max_h p_h(D).
$$

The dataset alerts only when this value is at most 0.05, meaning every historical reference rejects a matching dependence profile. For example, per-task values of 0.01, 0.02 and 0.20 do not trigger an alert; the third reference is not rejected. Failure to reject does not prove that the dataset is clean.

If required incoming features are constant, the implementation reports an unsupported test instead of silently accepting the dataset. The report includes test coverage.

There are important limitations. A clean incoming task can differ from all inversion references, causing false alerts. Conversely, an attack may alter feature magnitudes without changing their rank relationships, so Rank can miss it. The test detects a specific kind of distribution shift; it does not establish that every detected shift is malicious.

## 4. Why the evaluation contains multiple “datasets”

There is one underlying incoming task, not many independently collected tasks. In the current split, the task has 5,000 training originals: 3,000 in the detector-training split, 500 in validation, 500 in test and 1,000 in reserve. These numbers describe the offline benchmark partition, not data that must all be available to a deployed detector.

For the final evaluation, I use only the 500 test originals. Each original has a clean view, a BrainWash-poisoned view and a random-perturbation control view. The new pilot uses uniform perturbations at the same L-infinity budget for the random control.

BrainWash is optimized on the incoming task's training images before this detector evaluation. “Test originals” means held out from detector fitting, not unseen by attack generation. I do not optimize a separate attack for each bag. This distinction matters when interpreting generalization.

One **simulated incoming dataset**, or bag, is constructed as follows:

1. Draw 150 distinct original-image IDs without replacement from the test pool.
2. For a requested contamination fraction $r$, calculate $m=\lfloor150r+0.5\rfloor$.
3. Use the poisoned view of $m$ selected originals and the clean view of the remaining $150-m$. A noise-control bag uses random-perturbed views instead of poisoned views.
4. Pass only the selected descriptor rows to the detector. Each original appears once in that bag; its clean and poisoned views never appear together.

At a requested 1% contamination, this gives 2 poisoned images out of 150, or about 1.33%. The report records both the requested and realized fractions. A clean bag contains no modified images.

The evaluator repeats this sampling 100 times per condition, using requested fractions of 0%, 1%, 5%, 10%, 25%, 50% and 100%. This estimates alert frequency under the specified finite-pool mixture simulation. It does **not** create 100 independent CIFAR tasks, 100 independently generated attacks or 100 training seeds. Originals can recur across bags, and all poisoned views in a run come from the same attack artifact. Bags are therefore dependent as experimental replicates.

Rank and MMD are compared on identical bags within their evaluator. The supervised evaluator has its own deterministic bag sequence, so its bag index must not be treated as paired with the unsupervised evaluator's bag index. Comparisons between threshold rules within the supervised evaluator use the same seed and bags.

For a real incoming dataset of 150 images, the detector makes **one** dataset-level decision. Repeated bag construction is only an offline sensitivity analysis. Different dataset sizes would require a separately defined count calibration and evaluation; this implementation does not silently reuse the 150-image count cutoff.

## 5. What I propose to do next

First, re-extract predicted-gradient features, check that no true-class feature enters inference, and run the small threshold pilot. Existing compatible checkpoints, attack artifacts and inversion images can be reused; the old true-class feature CSVs cannot be repaired by deleting or renaming columns.

I will report clean sample and dataset false-positive rates alongside detection at every contamination fraction, including failures. Random-perturbation alerts will be reported as noise sensitivity, not automatically counted as false positives on harmless data.

The broader CL-method and attack matrix, extra forgetting controls and feature-removal ablations remain deferred. Before running them, I would like to agree on the feature definition, threshold rule and interpretation of the simulated dataset evaluation.
