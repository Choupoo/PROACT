#!/usr/bin/env bash
# Run from any location, with the GPU server's proact38 Python selected.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd -- "$SCRIPT_DIR/.."

PYTHON_BIN="${DETECTOR_PYTHON:-python}"
export PYTHONDONTWRITEBYTECODE=1
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
OUTPUT_DIR="${DETECTOR_FINAL_OUTPUT:-detector/work/advisor_final_v1}"
ACTION="${1:-run}"
if [[ $# -gt 0 ]]; then shift; fi

case "$ACTION" in
  plan)
    "$PYTHON_BIN" -B -u -m detector.advisor_final plan --output-dir "$OUTPUT_DIR" "$@"
    ;;
  run)
    if [[ ! -f "$OUTPUT_DIR/plan.json" ]]; then
      "$PYTHON_BIN" -B -u -m detector.advisor_final plan --output-dir "$OUTPUT_DIR"
    fi
    "$PYTHON_BIN" -B -u -m detector.advisor_final run --output-dir "$OUTPUT_DIR" "$@"
    if [[ $# -eq 0 ]]; then
      "$PYTHON_BIN" -B -u -m detector.advisor_final summarize --output-dir "$OUTPUT_DIR"
    else
      printf 'Cell command finished. After all workers finish, run: bash detector/run_advisor_final.sh summarize\n'
    fi
    ;;
  summarize|package)
    "$PYTHON_BIN" -B -u -m detector.advisor_final "$ACTION" --output-dir "$OUTPUT_DIR" "$@"
    ;;
  *)
    printf 'Usage: bash detector/run_advisor_final.sh [plan|run|summarize|package] [options]\n' >&2
    exit 2
    ;;
esac
