# Archived email draft — not sent

This draft records the original Meeting 3 plan. Its assumption that incoming
class labels are available was superseded by the predicted-class protocol.
It is retained as correspondence history, not a current methods description.

Subject: Follow-up on feature analysis and cross-task detection

Dear Prof. Carta,

Thank you for the discussion. I will first check which model and split produced
the 86.4% figure, then use ablations and linear SHAP to examine its features.
The SHAP background will come from the detector's training split.

I will also train a detector on clean and artificially poisoned images from
Task 1 and test it on Task 9 without refitting or recalibration. These are
zero-based task indices. I plan to compare the stage-gradient, uncertainty and
activation features with the full sixteen-feature set. The current MMD baseline
already compares detector descriptors, not full activation embeddings.

In this original plan, the supervised branch uses image-class labels to extract
features, but no target clean/poison labels to fit or calibrate the detector.
The predicted-class variant is a separate option. Historical-reference tests
remain a separate label-free comparison.

I will report sample ranking, frozen-threshold errors, random-perturbation alerts,
dataset detection across contamination fractions and attack-induced forgetting.
The repeated bags reuse images, so I will not present them as independent tasks.

Best regards,
Pan Zhang
