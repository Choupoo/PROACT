#!/usr/bin/env bash
set -euo pipefail
detector_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$detector_dir/.."
if (( $# > 1 )); then
  echo "Usage: bash detector/run_granularity.sh [new-output-directory]" >&2
  exit 2
fi
result_root="${1:-detector/work/granularity_$(date +%Y%m%d_%H%M%S)}"
source_run="${THESIS_TRANSFER_RUN:-detector/work/meeting3_transfer_v1}"
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python -B -u -m detector.granularity_study \
  --source-run "$source_run" --output-dir "$result_root" --dry-run
python -B -u -m detector.granularity_study \
  --source-run "$source_run" --output-dir "$result_root"
echo "Completed: $result_root"
echo "Read granularity_summary.md, shap_summary.md and professor_update.md."
