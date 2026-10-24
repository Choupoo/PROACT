# Runtime audit

Sum of recorded subprocess wall durations, not measured GPU kernel time. Parallel cells overlap.

| Method | Seed | Total stage wall hours | Attack hours | Continuation hours |
| --- | --- | ---: | ---: | ---: |
| afec_ewc | seed6 | 69.52 | 52.62 | 7.44 |
| afec_ewc | seed7 | 70.21 | 52.06 | 7.64 |
| afec_ewc | seed8 | 69.83 | 52.05 | 7.45 |
| ancl_ewc | seed6 | 65.29 | 52.04 | 4.92 |
| ancl_ewc | seed7 | 65.75 | 52.21 | 4.99 |
| ancl_ewc | seed8 | 65.00 | 51.95 | 4.95 |
| ewc | seed6 | 60.59 | 52.03 | 3.18 |
| ewc | seed7 | 62.04 | 52.24 | 3.18 |
| ewc | seed8 | 61.96 | 52.17 | 3.30 |
| mas | seed6 | 60.86 | 52.20 | 3.27 |
| mas | seed7 | 62.37 | 52.04 | 3.24 |
| mas | seed8 | 63.06 | 52.26 | 3.25 |
| rwalk | seed6 | 62.35 | 51.86 | 4.27 |
| rwalk | seed7 | 64.36 | 52.05 | 4.27 |
| rwalk | seed8 | 64.48 | 52.19 | 4.22 |

## Repeated work removed from the new plan

- Legacy target clean continuation runs four times per cell; the new plan runs it once.
- Legacy uniform continuation runs twice per budget; the new plan runs it once per budget.
- Legacy clean/random feature measurements repeat across attacks; the new plan reuses matching frozen-model controls.
- ImageNet attack evaluation now loads and partitions prepared arrays once per subprocess, with fresh dataset wrappers and unchanged RNG resets.

## Work retained

- Six distinct BrainWash artifacts remain: Task 1 and Task 4 reckless/0.3, plus four Task 9 attacks.
- Different stage checkpoints require their own inversions and feature references.
- Attack epochs (5000), victim epochs (100), full images and inversion iterations (2000) are unchanged.
- Pseudo-gradient model creation remains in the unmodified upstream attack loop; no unvalidated functional rewrite is enabled.
