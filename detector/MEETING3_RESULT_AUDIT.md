# Meeting 3 evidence audit — 24 September 2026

This is a dated read-only audit, not a new experiment. It examined repository
reports and the local downloaded results. The detailed record was saved under
`detector/work/meeting3_local_audit/`.

## Verified at the time

- `results/metrics.json` used the old three-feature
  `clean_supervised_baseline_v1` protocol: test ROC-AUC 0.880064, poison TPR
  44.8% and clean FPR 4.4%. It did not identify the meeting's newer model.
- `config.proact38.json` selected `extended`, including activation and historical
  cosine features. That configuration alone did not establish that a reduced
  model had run.
- Downloaded seed directories mainly contained label-free reports/configuration,
  not the full supervised CSV, frozen bundle and checksums needed for explanation.
- MMD already compared detector descriptors, not full backbone embeddings.

## Unresolved at the time

The artifacts did not identify the model, split, feature set and threshold
behind the meeting's 86.4% figure. This was missing provenance, not evidence that
the figure was wrong. Aggregate metrics also could not reconstruct SHAP rankings.

The follow-up protocol and commands are retained in
[MEETING3_PROTOCOL.md](MEETING3_PROTOCOL.md) and [MEETING3_RUN.md](MEETING3_RUN.md).
Later dated reports provide subsequent results; they do not change what was
available during this audit.
