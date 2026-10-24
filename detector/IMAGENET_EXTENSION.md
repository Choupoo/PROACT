# Full-data EWC extension

This extension tests EWC on miniImageNet and tinyImageNet, with four BrainWash
attack settings, nine retrained feature groups and frozen-threshold evaluation.
All new code and outputs remain under `detector/`. The upstream PROACT files
and the completed CIFAR-100 results are not modified.

## Protocol

| Item | miniImageNet | tinyImageNet |
| --- | --- | --- |
| Classes / resolution | 100 / 84 × 84 | 200 / 64 × 64 |
| Task partition | 10 tasks, 10 classes each | 20 tasks, 10 classes each |
| Source / calibration tasks | 1 / 1 and 4 | 1 / 1 and 4 |
| Final evaluation task | 9 | 19 |
| Default seeds | 6 and 7 | 6 and 7 |
| Full training images per class | 500 | 500 |
| Test images per class | 100 | 50 labelled validation images |

Task numbers are zero-based. The task counts follow Section 5.1 of the
[BrainWash paper](https://openaccess.thecvf.com/content/CVPR2024/papers/Abbasi_BrainWash_A_Poisoning_Attack_to_Forget_in_Continual_Learning_CVPR_2024_paper.pdf).
The earlier ImageNet runner incorrectly used 10 tasks of 20 classes for
tinyImageNet. Version 3 corrects this before the new experiments are run.
Old ImageNet plans are incompatible: create a new output directory after
uploading the code. The default is `detector/work/advisor_imagenet_2seeds_v3`.

The detector-specific source/calibration protocol is not an exact reproduction
of every BrainWash paper setting. It retains 100 victim epochs, 2000 inversion
iterations and 5000 attack epochs, with EWC lambda 500000. Final attacks are
reckless/cautious at each L-infinity budget 0.1 and 0.3. No images, classes or
epochs are removed to claim a speedup.

Both datasets train a separate logistic regression from 3000 clean and 3000
poisoned source views (300 originals per class). Another 50 originals per class
are used for validation, 50 for test and 100 for reserve. These detector splits
do not reduce the 500-per-class CL training and attack workload. The full
detector uses 80 backbone parameter-tensor gradient norms plus four context
features, without image-class labels or true-class probability at extraction.
The `history_mad` rule is prespecified; its numerical cutoffs are fitted on
earlier tasks and frozen before the target. This is within-dataset task
transfer, not a CIFAR-trained detector applied to another dataset without
refitting. Rank/MMD are not rerun in this extension.

## Download and prepare the missing data

The preparation command downloads the archives linked from the
[authors' dataset repository](https://huggingface.co/datasets/mintlabvandy/BrainWash-CVPR24/tree/main/datasets).
It resolves and saves an immutable repository revision before downloading.
It uses HTTPS with normal certificate verification, retries interrupted
transfers and resumes supported byte ranges. Archive CRCs, SHA-256 hashes,
class counts, image shapes and pixel values are checked. Optional
`--mini-sha256` and `--tiny-sha256` accept externally obtained archive hashes;
a locally computed hash alone does not authenticate external content.

Run once on the server, after uploading version 3:

```bash
cd /home/p.zhang/PROACT
conda activate proact38
CUDA_VISIBLE_DEVICES="" python -B -u -m detector.prepare_imagenet
```

No GPU job starts here. Archives go under `detector/data_downloads/`; prepared
data and preparation records go under `detector/data_prepared/`. These
directories are Git-ignored. Allow at least 20 GiB free disk for preparation.
NPZ loading and full-data training require substantial host memory; allow
roughly 32 GB per worker plus headroom and check actual usage on the server.

Preparation preserves the four miniImageNet NPY files byte-for-byte, including
their existing training/test split and row order. It does not construct a new
500/100 split from unpartitioned images. If the archive does not contain
exactly one complete `train_x.npy`, `train_y.npy`, `test_x.npy`,
`test_y.npy` set, it stops and asks for archive inspection.

For tinyImageNet, all official training images are retained and the annotated
validation set is used for testing. Classes and original filenames are sorted,
images are converted to RGB float32 NCHW in [0,1], and no resizing is performed.
The unlabelled test set is not substituted for the labelled validation set.
The original images are not moved. The NPZ and provenance sidecar record the
conversion and class mapping.

The extractor rejects absolute/traversing paths, symlinks, duplicate members
and oversized archives. Failed conversions use isolated temporary directories
and do not promote incomplete data. Existing untracked outputs are never
overwritten. A lock prevents concurrent preparation into the same cache/data
root. Rerunning the same command verifies and reuses prepared data.

For trusted archives already downloaded elsewhere, avoid another transfer:

```bash
CUDA_VISIBLE_DEVICES="" python -B -u -m detector.prepare_imagenet \
  --mini-archive /absolute/path/to/miniImagenet.zip \
  --tiny-archive /absolute/path/to/tiny-imagenet-200.zip
```

Use `--dataset mini` or `--dataset tiny` to prepare only one dataset.
A certificate, network or archive-format error must be resolved rather than
bypassed; do not disable TLS verification or class-count checks.

## Register and launch on two GPUs

After preparation finishes:

```bash
python -B -m detector.doctor --require-cuda
bash detector/run_advisor_imagenet.sh plan --seeds 6 7
```

Plan creation hashes and validates the full arrays; it may take several minutes.
It is not training. If prepared files are elsewhere, supply `--mini-data`
and `--tiny-data`. Use the same `--output-dir` on every workflow command
when overriding the default.

In a GPU 0 tmux session:

```bash
tmux new -s imagenet_mini
cd /home/p.zhang/PROACT
conda activate proact38
CUDA_VISIBLE_DEVICES=0 bash detector/run_advisor_imagenet.sh run --dataset mini
```

Detach with Ctrl+B, then D. In another terminal:

```bash
tmux new -s imagenet_tiny
cd /home/p.zhang/PROACT
conda activate proact38
CUDA_VISIBLE_DEVICES=1 bash detector/run_advisor_imagenet.sh run --dataset tiny
```

Each worker runs seeds 6 and 7 sequentially for its assigned dataset. These are
independent single-GPU workers, not distributed execution of one attack.
Each sees its physical GPU as local `cuda:0`. Do not start another worker for
the same dataset/seed. Check that the two GPUs are allocated to you before use.

Before launching, `run --dataset tiny --dry-run` prints the registered commands.
The tiny target commands must contain incoming task 19 and `--tasknum 20`.
Never upload changed source files into a running registered experiment.

## Efficiency, progress and recovery

Each dataset/seed has 30 stages instead of the legacy 36. Clean target
continuation is trained once; uniform continuation is trained once per budget.
Matching clean features and same-budget random controls are reused only under
the same data, checkpoint, inversion and extraction identities. Poison features
are extracted separately. Each subprocess loads the arrays once; fresh dataset
wrappers prevent evaluation shuffling from altering cached inputs. The adapter
passes the requested seed explicitly, rather than accepting the upstream
ImageNet loader call's implicit zero seed.

Six distinct attacks remain per cell: two development attacks and four target
attacks. The higher-order attack loop is unchanged. The archived CIFAR-100 EWC
seed 6 took 60.59 stage wall-hours, including 52.03 in attacks (about 86%).
Removing duplicate controls cannot imply a several-fold overall speedup.
ImageNet runtime must be measured; CIFAR timings are not a completion forecast.

CPU-only progress and packaging:

```bash
CUDA_VISIBLE_DEVICES="" bash detector/run_advisor_imagenet.sh status
CUDA_VISIBLE_DEVICES="" bash detector/run_advisor_imagenet.sh summarize
CUDA_VISIBLE_DEVICES="" bash detector/run_advisor_imagenet.sh package
```

After interruption, rerun the original dataset command with `--retry-failed`.
Completed stages are hash-verified and skipped; an interrupted stage is
archived and restarted. This is stage recovery, not epoch-level resume.
A stored running flag may be stale after a crash: inspect the process and log.

Send both `mini_analysis_*.tar.gz` and `tiny_analysis_*.tar.gz` from the
output root. They contain plans, registrations, timings, logs, predictions,
bag decisions, ablations and forgetting matrices, with a file-hash manifest.
Large datasets, checkpoints, noise and joblib files stay on the server.
For a partial diagnostic package, use `package --allow-incomplete`.

## Validation limits

Synthetic CPU tests cover download/range handling, archive safety, full-data
conversion, immutable outputs, task-19 contracts, loader RNG/tensor equivalence,
native-resolution features, frozen-target evaluation and packaging. They do
not establish that the full remote archives or a V100 experiment have run
successfully. Inspect the server's preparation and environment checks first.

Report failed frozen cutoffs as well as successful settings. Two seeds offer
limited evidence of variability, not a population guarantee. Repeated bags
share a finite image pool and are not independent datasets. Forgetting controls
do not establish that filtering improves CL. Upstream AFEC/ANCL qualifications
remain unchanged; this extension uses EWC only.
