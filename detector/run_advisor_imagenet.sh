#!/usr/bin/env bash
# Full-data extension only. Existing CIFAR-100 plans are not modified or resumed.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd -- "$SCRIPT_DIR/.."
export PYTHONDONTWRITEBYTECODE=1
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
PYTHON_BIN="${DETECTOR_PYTHON:-python}"
"$PYTHON_BIN" -B -u -m detector.advisor_imagenet "$@"
