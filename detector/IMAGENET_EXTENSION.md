# Full-data EWC extension

This run implements the agreed extension to miniImageNet and tinyImageNet:
EWC, four BrainWash settings, feature-removal ablations and frozen-threshold
evaluation. It does not repeat the completed CIFAR-100 grid. Only files under
`detector/` are changed; the original PROACT scripts are executed through
detector-owned dataset adapters.

## Registered scope

| Item | Setting |
| --- | --- |
| Datasets | Full prepared miniImageNet and tinyImageNet training splits |
| CL method | EWC, lambda 500000 |
| Default seeds | One seed, 6, per dataset; increase at plan creation if needed |
| Fixed task partition | Ten tasks; 10 classes per miniImageNet task, 20 per tinyImageNet task |
| Detector fit | Task 1, original/BrainWash pairs from the training split |
| Calibration | Clean validation and reserve splits from Tasks 1 and 4 |
| Final target | Task 9; no target fitting or recalibration |
| Gradient target | Fixed incoming-head prediction; no true-class probability feature |
| Full detector | 80 backbone parameter-tensor norms and four context features |
| Ablations | Nine feature groups, each retrained and recalibrated on the same source/history splits |
| Threshold rule | Prespecified `history_mad`; numeric cutoffs fitted within each dataset |
| Target attacks | reckless/cautious, each with L-infinity budget 0.1/0.3 |
| Training | 100 victim epochs, 2000 inversion iterations, 5000 attack epochs |
| Dataset simulations | 150 distinct originals per bag, 100 repeated bags per rate |

The detector-training splits use 300 originals per class, with 50 validation,
50 test and 100 reserve originals. All 500 images per class remain in the
continual-learning/attack workload. Thus miniImageNet fits the logistic model
on 3000 clean and 3000 poisoned views; tinyImageNet uses 6000 of each. These
detector splits are not a reduction of the CL training dataset.

This is a within-dataset source-to-later-task generalization test: each dataset
gets its own source-trained logistic model and historical calibration. It is
not a CIFAR-trained detector transferred to ImageNet without refitting.
Rank/MMD are not rerun in this extension; their existing results remain separate.
One seed per dataset is an initial resource-conscious comparison, not a measure
of seed-to-seed robustness. Do not pick a new seed based on its target results.

## Prepared data and validation

No dataset download or destructive preprocessing is performed by this runner.
Use the same prepared full-data formats as PROACT:

- miniImageNet: `train_x.npy`, `train_y.npy`, `test_x.npy`, `test_y.npy` in one
  directory. Images are uint8 NHWC, 84 x 84 x 3. Labels are global integer IDs
  0--99. Each class has 500 training and 100 test images.
- tinyImageNet: one NPZ containing `train_data`, `train_lbls`, `tst_data`,
  `tst_lbls`. Images are floating-point NCHW, 3 x 64 x 64, in [0,1]. Labels
  are global integer IDs 0--199. Each class has 500 training and 50 test images.

Registration checks dimensions, pixel values, class counts and SHA-256 hashes.
Reduced or incomplete arrays are rejected before training. If your files use
a different encoding or split, stop and check the preparation; do not bypass
the checks or rename another dataset. Counts and hashes establish consistency
with the registered files, not proof of their external provenance.

The adapter retains original row order, uses task-local labels without
modifying source arrays, corrects the miniImageNet input-size metadata to 84,
and retains seeded RNG resets. The adapter explicitly passes the registered
seed to the ImageNet loader: the original ImageNet training call omits this
argument and otherwise resets the RNGs to zero, regardless of the requested
seed. This correction applies only to the new ImageNet plans. A subprocess loads and
partitions the arrays once. Fresh dataset wrappers on subsequent calls prevent
an evaluation shuffle from changing the cached inputs.

Plan creation reads large files and can take several minutes. Two workers
also need substantial host RAM: allow roughly 32 GB per worker (64 GB total),
plus capacity for the OS and other users. Actual peak requirements depend on
the NPZ encoding and upstream allocations. Check free RAM and disk before
starting; cached data are process-local and not shared between GPUs.

## Work removed, and work retained

For each new dataset/seed, the schedule has 30 stages rather than the legacy
36-stage expanded cell:

- Clean target continuation is trained once, rather than four times.
- Uniform continuation is trained once per budget, rather than once per mode.
  The full-data uniform draw does not depend on BrainWash's poison values or
  permutation. Tests compare the actual upstream perturbation branch.
- Clean features are reused under the same checkpoint and extraction seeds.
  Random-control features are reused only at the same budget. Dataset,
  checkpoint, inversion and extraction identities must match; poison features
  are always extracted separately. Reuse is recorded in the feature sidecar.
- Repeated attack-time evaluation no longer rereads and repartitions the full
  ImageNet arrays on every evaluation call.

There are still six distinct attacks: Task 1 reckless/0.3, Task 4 reckless/0.3,
and the four target attacks. Checkpoints and inversion references from different
stages are not interchangeable. The attack's higher-order gradient loop and
model-copy operations are unchanged. Neither fewer images nor fewer epochs
are used to claim an efficiency gain.

For the archived CIFAR-100 EWC seed 6, recorded stage wall time was 60.59 hours:
52.03 hours in attack generation (about 86%), 4.75 in victim training, 3.18 in
continuation training, and the remainder in inversion, extraction and analysis.
Consequently, removing duplicate controls alone cannot yield a several-fold
speedup. These are subprocess durations, not measured GPU kernel time. New
ImageNet timings must be measured; CIFAR timings do not establish a completion
date for larger images.

To audit a completed server run without using a GPU:

```bash
cd /home/p.zhang/PROACT
conda activate proact38
CUDA_VISIBLE_DEVICES="" python -B -m detector.runtime_audit \
  --run-dir detector/work/advisor_final_v1 \
  --output-dir detector/work/runtime_audit_20261024
```

## Start on two GPUs

Upload this code before creating the new plans. Do not overwrite detector source
files while an older registered experiment is still running: its fingerprint
will no longer match. The new output root is `detector/work/advisor_imagenet_v1`.
It does not overwrite `advisor_final_v1` or reuse its CIFAR artifacts.

First, in a server terminal, select the environment and check the data paths.
Replace the two example paths below if the prepared arrays are elsewhere.

```bash
cd /home/p.zhang/PROACT
conda activate proact38
python -B -m detector.doctor --require-cuda
ls data/miniImagenet/train_x.npy data/miniImagenet/train_y.npy \
   data/miniImagenet/test_x.npy data/miniImagenet/test_y.npy \
   data/tiny_imagenet.npz

bash detector/run_advisor_imagenet.sh plan \
  --mini-data /home/p.zhang/PROACT/data/miniImagenet \
  --tiny-data /home/p.zhang/PROACT/data/tiny_imagenet.npz \
  --seeds 6
```

If `ls` or registration fails, resolve the data preparation first. A plan is not
a completed experiment. Repeating the exact plan command verifies and reuses
the matching plans; differing inputs require a new output directory.

GPU 0, first tmux session:

```bash
tmux new -s imagenet_mini
cd /home/p.zhang/PROACT
conda activate proact38
CUDA_VISIBLE_DEVICES=0 bash detector/run_advisor_imagenet.sh run --dataset mini
```

Detach with `Ctrl+B`, release both keys, then press `D`. In a second terminal,
use GPU 1 for tinyImageNet:

```bash
tmux new -s imagenet_tiny
cd /home/p.zhang/PROACT
conda activate proact38
CUDA_VISIBLE_DEVICES=1 bash detector/run_advisor_imagenet.sh run --dataset tiny
```

This is two independent single-GPU experiments, not distributed training of
one attack. Each worker sees its assigned physical GPU as local `cuda:0`.
Check `nvidia-smi` and each tmux session; never start another worker for the same
dataset/seed. A method/seed lock rejects duplicate workers. GPU memory sufficiency
and throughput on V100 still require the real run; CPU tests cannot verify them.

For inspection before launching, `run --dataset mini --dry-run` prints the
registered commands without CUDA work. Use the corresponding flag for tiny.

## Progress, interruption and packaging

Progress checks do not need a GPU:

```bash
CUDA_VISIBLE_DEVICES="" bash detector/run_advisor_imagenet.sh status
```

The output reports completed stages and measured wall time by category. A
recorded `running` state can remain after a crash; also check the actual process
and log. Stage counts are not time percentages.

After an interruption, return to the same environment, source version and
dataset files, then use the assigned GPU:

```bash
CUDA_VISIBLE_DEVICES=0 bash detector/run_advisor_imagenet.sh run \
  --dataset mini --retry-failed
```

Completed stages are hash-verified and skipped. Only the interrupted stage is
archived and restarted; this is stage-level recovery, **not epoch-level resume**.
The archive remains under the cell's `interrupted/` folder. Do not run plan
creation again with different settings, or change the code mid-run.

After both workers finish:

```bash
CUDA_VISIBLE_DEVICES="" bash detector/run_advisor_imagenet.sh summarize
CUDA_VISIBLE_DEVICES="" bash detector/run_advisor_imagenet.sh package
```

Send both `mini_analysis_*.tar.gz` and `tiny_analysis_*.tar.gz` from
`detector/work/advisor_imagenet_v1/`. They include plans, data registration,
timings, logs, predictions, bag decisions, ablations and forgetting matrices,
with a SHA-256 manifest. Datasets, noise, checkpoints and joblib models are
excluded; keep them on the server for reproducibility. For a partial diagnostic
package, add `--allow-incomplete`; it remains explicitly incomplete.

## Checks and limitations

### Optional explanation and incoming-image prediction

`explain_detector` now accepts the inference schema without
`true_class_probability`, including tinyImageNet's 20-class tasks. It uses the
frozen model's original source CSV for its training background and verifies that
CSV's identity. This explains the fitted logistic logit; it does not select new
features or retune the detector. For example, after the miniImageNet run:

```bash
python -B -m detector.explain_detector \
  --features detector/work/advisor_imagenet_v1/mini/ewc/seed6/task1/reckless_0p3/features/features.csv \
  --bundle detector/work/advisor_imagenet_v1/mini/ewc/seed6/frozen_supervised/ablations/inference_full/bundle.joblib \
  --output-dir detector/work/advisor_imagenet_v1/mini/source_explanation
```

For these bundles, use the study's already registered and historically
calibrated ablations, not this helper's legacy `--ablations` option. The latter
is explicitly rejected because its source-only calibration would change the
comparison protocol.

`predict` also accepts the new transfer bundle. Supply an NPZ whose `images`
array contains exactly 150 images (or the size recorded in the frozen bundle),
floating-point NCHW in [0,1], at the native resolution. Incoming class labels,
clean/poison labels and benchmark triplets are not required. A `targets` array,
if present, is never accessed by this new prediction route. Historical inversion
labels remain part of the reference-gradient computation.

```bash
python -B -m detector.predict \
  --bundle detector/work/advisor_imagenet_v1/mini/ewc/seed6/frozen_supervised/ablations/inference_full/bundle.joblib \
  --checkpoint detector/work/advisor_imagenet_v1/mini/ewc/seed6/task9/victim/checkpoint.pkl \
  --inversion-dir detector/work/advisor_imagenet_v1/mini/ewc/seed6/task9/inversions \
  --input-npz /path/to/incoming_images.npz \
  --output-dir detector/work/advisor_imagenet_v1/mini/incoming_prediction
```

Use the incoming Task 9 checkpoint, not the source Task 1 checkpoint. The new
route checks dataset, CL method, registered seed and task chronology, resets
the incoming head with the frozen head seed, and uses the unchanged sample and
count cutoffs. It exports `prediction.json`, per-image scores and descriptor
provenance. It does not fit or calibrate on the incoming batch. The caller is
responsible for supplying matching clean historical references and distinct
incoming originals. An alert is not a calibrated poisoning probability.

The old prediction routes remain available for their original legacy bundles.
These helper fixes do not require rerunning completed detector experiments.
However, source fingerprints are strict: do not upload this update into an
already running experiment. If an older plan has been created but not run,
register a new plan under a new `--output-dir` after updating the code.

### Validation status

The test suite covers native-resolution features, 20-class heads, class-balanced
full-data checks, loader tensor/RNG equivalence, cache identity, frozen-target
evaluation, shared controls, incomplete packaging and atomic state writes.
On 24 October 2026, all 215 CPU tests passed with Python 3.12 and PyTorch 2.9;
Python 3.8 syntax, shell syntax and static checks also passed. This is not a
runtime certification of the server's Python 3.8/PyTorch 1.10/CUDA stack. Run
the environment doctor on that server before starting either worker.
The stored source-validation metrics are recomputed at the final historical
threshold instead of retaining metrics from the earlier source-only cutoff.
The diagnostic global gradient norm now accumulates in FP64: the long FP32
reduction could otherwise fail the stage/global consistency check through
rounding error. The 80 tensor-norm feature calculations are unchanged.

No new GPU results are supplied with this implementation. Report frozen-cutoff
failures as well as successful settings. High ROC-AUC does not imply a reliable
dataset decision. Repeated bags share a finite image pool and are not independent
datasets. Forgetting controls do not establish that filtering improves CL.

Known AFEC/ANCL issues in the upstream implementation are not silently patched
by this EWC extension. The old results and their qualifications remain intact.
