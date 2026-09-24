# Draft only — not sent

Subject: Follow-up on feature analysis and cross-task poisoning detection

Dear Prof. Carta,

Thank you for today's discussion. I have prepared the following implementation and experimental protocol; the new GPU experiments are still pending.

First, I will analyze the frozen supervised detector using feature ablations and exact interventional linear SHAP values for its logistic-regression logit. The SHAP background uses only the training partition. I will report individual and layer/stage-level contributions, feature correlations, and representative true/false positives and negatives. I am also checking which exact model and split produced the 86.4% figure discussed in the meeting, rather than attributing it to a feature set without the corresponding artifacts.

Second, following your suggestion, I will train a detector on clean and artificially poisoned data from the second task and apply it, without refitting, to the tenth task. In the code these are zero-based task indices 1 and 9. The CIFAR-100 partition remains ten tasks with ten classes each. The source scaler, classifier, sample threshold and dataset-level count threshold are frozen before target evaluation; no target poisoning labels or target-clean calibration samples are used to fit them.

For terminology, this is supervised source-to-target transfer with no target poisoning labels, rather than fully unsupervised learning. The primary protocol assumes the incoming training set has its ordinary class labels; a predicted-class variant can be evaluated separately. The historical model is assumed clean. I will retain the earlier strictly label-free methods as controls rather than relabel their results.

I have pre-specified two descriptor sets: layer/stage gradient norms, uncertainty measures and activation norm as the primary set, and the full sixteen-feature set including historical gradient similarities as a control. One implementation detail worth clarifying is that the existing MMD baseline already compares these low-dimensional descriptors, not the full activation embeddings.

The new runs will report sample ROC-AUC, clean false-positive rate, poisoning detection rate, random-perturbation controls, dataset-level detection across poisoning fractions, and paired attack-effectiveness measurements. I will report all outcomes, including failures: source calibration does not guarantee a low false-positive rate after task or backbone shift, and repeated simulated bags are not independent datasets.

Best regards,
Pan Zhang
