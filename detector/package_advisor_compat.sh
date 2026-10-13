#!/usr/bin/env bash
# Package a registered running study without editing its fingerprinted Python.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd -- "$SCRIPT_DIR/.."

PYTHON_BIN="${DETECTOR_PYTHON:-python}"
OUTPUT_DIR="${DETECTOR_FINAL_OUTPUT:-detector/work/advisor_final_v1}"
export CUDA_VISIBLE_DEVICES=""
export PYTHONDONTWRITEBYTECODE=1
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

"$PYTHON_BIN" -B -u - "$OUTPUT_DIR" "$@" <<'PY'
import json
import sys

import numpy as np

original_default = json.JSONEncoder.default


def numpy_default(self, value):
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return original_default(self, value)


# This conversion exists only in this process. Source and artifact checks stay
# enabled, and allow_nan=False still rejects non-finite numerical results.
json.JSONEncoder.default = numpy_default
sys.argv = ["detector.advisor_final", "package", "--output-dir", sys.argv[1]] + sys.argv[2:]

from detector.advisor_final import main

main()
PY
