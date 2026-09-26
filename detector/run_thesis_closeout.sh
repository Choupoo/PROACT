#!/usr/bin/env bash
# Run from the activated PROACT environment. No downloads or upstream training.
set -euo pipefail

detector_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$detector_dir/.."
if (( $# > 1 )); then
  echo "Usage: bash detector/run_thesis_closeout.sh [new-output-directory]" >&2
  exit 2
fi
result_root="${1:-detector/work/thesis_closeout_$(date +%Y%m%d_%H%M%S)}"
transfer_source="${THESIS_TRANSFER_RUN:-detector/work/meeting3_transfer_v1}"
rank_source1="${THESIS_RANK_SOURCE1:-detector/work/unsupervised_confirm_v1/source_seed1}"
rank_source2="${THESIS_RANK_SOURCE2:-detector/work/unsupervised_confirm_v1/source_seed2}"
if [[ -e "$result_root" ]]; then
  echo "Output already exists; choose a new directory: $result_root" >&2
  exit 2
fi

# Check every source before starting either experiment.
for seed in 3 4; do
  for task in 1 9; do
    for filename in features.csv features.metadata.json; do
      required="$transfer_source/seed$seed/features_task$task/$filename"
      if [[ ! -f "$required" ]]; then
        echo "Missing full source feature artifact: $required" >&2
        echo "Set THESIS_TRANSFER_RUN to the original server feature run." >&2
        exit 2
      fi
    done
  done
done
for source in "$rank_source1" "$rank_source2"; do
  for filename in reference_features.csv reference_features.metadata.json predicted_features.csv predicted_features.metadata.json; do
    if [[ ! -f "$source/$filename" ]]; then
      echo "Missing full source feature artifact: $source/$filename" >&2
      echo "Set THESIS_RANK_SOURCE1/THESIS_RANK_SOURCE2 to the original feature directories." >&2
      exit 2
    fi
  done
done

export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python -B -m unittest discover -s detector/tests -q
python -B -u -m detector.revision_study supervised \
  --source-run "$transfer_source" --output-dir "$result_root/supervised" \
  --seeds 3 4 --negative-policies clean_only clean_and_random --explanations
python -B -u -m detector.thesis_closeout rank \
  --source-runs "$rank_source1" "$rank_source2" --output-dir "$result_root/rank"
python -B -u -m detector.thesis_closeout summarize \
  --supervised-run "$result_root/supervised" --rank-run "$result_root/rank" \
  --output-dir "$result_root/tables"
echo "Completed experiments and thesis tables: $result_root"
echo "Inspect tables/thesis_summary.md; successful execution is not a performance or graduation guarantee."
