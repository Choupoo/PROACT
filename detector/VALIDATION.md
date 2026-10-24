# Local validation record

This is a dated development record. Counts and timings below describe the checks
performed then, not a new test run during documentation editing. Synthetic
scores test implementation behaviour, not CIFAR-100 detection performance.

The local environment was Python 3.12 / torch 2.9 unless noted. Python 3.8
syntax parsing and mocked older interfaces are not equivalent to executing the
complete server workflow.

## Test history

| Date | Change | Full-suite tests passed | Runtime when recorded | Python files parsed as 3.8 |
| --- | --- | ---: | ---: | ---: |
| 26 September 2026 | Gradient-granularity study | 169 | 104.657 s | 46 |
| 26 September 2026 | Closeout ablation and Rank rechecks | 164 | 84.264 s | 44 |
| 26 September 2026 | Supervised shape and historical shape-MMD | 158 | Not recorded here | 42 |
| 24 September 2026 | Meeting 3 transfer and explanations | 145 | Not recorded here | 39 |
| 20 September 2026 | Historical Rank adaptation | 114 | Not recorded here | 28 |
| 20 September 2026 | Count calibration and reference audit | 96 | Not recorded here | 25 |
| 19 September 2026 | Older PyTorch compatibility | 82 | Not recorded here | Not recorded here |
| 15 September 2026 | Initial detector | 78 | Not recorded here | 20 |

Ruff and Git whitespace checks passed for the listed development revisions.
Shell syntax checks were also recorded for the granularity and closeout entries.
The initial notebook's four code cells passed syntax parsing, and eleven CLI
help commands exited normally.

## Coverage by revision

### Gradient granularity — 26 September

Five new tests covered source schema freezing, missing fine-grained inputs,
head exclusion, test/random isolation and rejection of target schema mismatch.
A synthetic CLI integration exercised eight groups, two seeds and two tasks,
including SHAP, paired differences and provenance. At that point the local
machine lacked the full real-feature CSVs for the comparison. The later dated
reports, not these tests, contain its measured results.

### Closeout — 26 September

Six new tests covered source random-negative mapping and weights, split
isolation, compatibility, SHAP background and cross-run summaries. The synthetic
integration ran six supervised comparisons, two historical-reference rechecks
and the combined summary. Previously observed seeds remained rechecks.

### Shape comparisons — 26 September

Thirteen new tests covered scale invariance, source test/random isolation,
SHAP transformation, historical kernel-fit/bank separation, ignored labels,
max-p aggregation, whole-bag changes, zero gradients, provenance and both CLIs.
After frozen-model and synthetic/real provenance checks were added, those
thirteen tests passed again. A target CSV failure test confirmed that the model
was already frozen and the run state recorded failure.

The read-only Meeting 3 audit observed target sample FPR 12.8%–24.6% and clean
bag alerts 99.5%–100%. The report package lacked full feature CSVs, so the new
shape models had not yet been fitted on those real inputs during this check.

### Meeting 3 — 24 September

Seventeen new tests covered the fixed ten-task partition, future heads and EWC
Fisher buffers, model/attack identity, source test/random exclusion, target
calibration exclusion, transfer contracts, SHAP permutation/additivity checks,
synthetic CLI integration, command generation and completed-stage hashes.

The read-only artifact audit was saved under `work/meeting3_local_audit/`.
It could not identify the complete provenance of the meeting's 86.4% figure.
Synthetic fitting, explanations and source/target evaluation ran under
`work/meeting3_cpu_smoke_v2/`; the earlier failed smoke directory was retained.
Plots were inspected. A local fontconfig configuration avoided cache symlinks
conflicting with output-integrity rules.

### Rank — 20 September

Eighteen new tests covered tau kernels, exact leave-one-out pseudovalues,
monotone invariance and blind spots, ignored labels, identity checks, constants,
max-p aggregation, task holdout and unchanged model/source files. The synthetic
`unsupervised_adapt run` integration verified that freezing precedes benchmark
reading, dry-run leaves no output and existing directories are rejected.

A separate 50-trial synthetic check used two historical tasks of 128 rows and
150 incoming rows, 499 draws and alpha=0.05. The test `descriptors` generator
used historical seeds `10000 + 2*trial + task`, incoming seed `20000 + trial`,
and trials 0–49:

| Synthetic condition | Alerts |
| --- | ---: |
| Clean | 1/50 |
| Whole-bag monotone transform: norms *30+100, confidence/margin cubed | 1/50 |
| First-feature latent sign reversal | 50/50 |
| Replace only the first 15/150 rows | 20/50 |

These checks retained the method's weak sparse-change response. They were not
CIFAR-100/BrainWash measurements or evidence of a guaranteed clean FPR.

### Count calibration — 20 September

Tests covered count boundaries and error budgets, counting each calibration
original once, legacy bundles, test isolation, validation-only reference audits,
CSV integrity and reassessment commands.

The synthetic `demo` at `work/cpu_demo_calibration_v2/` and dry/actual
`reassess` at `work/cpu_reassessment_v2/` completed. New models used
`count_bound`; legacy LR/top-tail behaviour remained available. The MMD audit
retained raw alerts and recorded an undetermined poisoning decision/abstention.
Abstention was not counted as improved detection.

### Older PyTorch — 19 September

User-supplied server logs reported Python 3.8.20, torch 1.10.1, torchvision
0.11.2, working CUDA and 75 passes / 3 errors in the original 78 tests.
Two errors involved unsupported `weights_only`; one involved cross-dtype
`torch.equal`.

The fix checks the explicit loading signature before passing
`weights_only=False`, preserves CPU mapping and validates indices through
same-dtype round trips. Four regressions covered old/new interfaces, propagation
of unrelated TypeError and integer/fractional/nonfinite index handling.
They reproduced the errors before the fix; the resulting local suite passed 82
tests. `config.proact38.json` reused existing artifacts in a new output directory,
and its pipeline dry-run completed.

These were local compatibility checks, not direct execution on the server.
Trusted pickle loading remains a security requirement.

### Initial detector — 15 September

Pipeline/bootstrap dry-runs and the complete synthetic demo at
`work/cpu_demo_final/` completed, including feature analysis, frozen models,
evaluation, reports and legacy prediction CLIs. Tests used small models plus
actual PROACT ResNet parameter grouping, mocked checkpoint dependencies,
Fisher-buffer regressions and CPU tensor-storage loading.

The recorded machine used Python 3.12.7, torch 2.9.0, NumPy 1.26.4, pandas 2.3.3
and scikit-learn 1.7.2. Numerical imports and torch worked; CUDA was unavailable.
The installed torchvision 0.21.0 import subprocess exited with signal 11.
No real upstream training, inversion or attacks were performed locally, and no
external environment changes were made to repair that setup.

## Interpreting this record

All development changes described above were confined to `detector/`.
Passing implementation tests does not establish attack sensitivity, threshold
transfer or reference validity. For real results, use the dated reports linked
from [README.md](README.md). The current server guide is
[ADVISOR_FINAL_RUN.md](ADVISOR_FINAL_RUN.md).
