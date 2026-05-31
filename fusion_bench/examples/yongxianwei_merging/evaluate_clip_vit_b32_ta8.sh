#!/usr/bin/env bash
# Evaluate all thirteen yongxianwei_merging methods on the
# CLIPVisionModelPool/clip-vit-base-patch32_TA8_model_only benchmark using
# CLIPVisionModelTaskPool/clip-vit-classification_TA8.
#
# Usage:
#   bash examples/yongxianwei_merging/evaluate_clip_vit_b32_ta8.sh
#
# Each method's JSON report is written under
# outputs/yongxianwei_merging/reports/clip-vit-base-patch32_TA8/<method>.json
# and re-runs are skipped if the report already exists.

set -euo pipefail

METHODS=(ccam ccam_v2 ccam_v3 ccam_v4 ccam_v5 ccam_v6 amm wudi wudi_plus iwudi swudi aswudi swudi_align)
ROOT_DIR="outputs/yongxianwei_merging"
BENCHMARK="clip-vit-base-patch32_TA8"

mkdir -p "${ROOT_DIR}/reports/${BENCHMARK}" "${ROOT_DIR}/logs"

for method in "${METHODS[@]}"; do
  report_path="${ROOT_DIR}/reports/${BENCHMARK}/${method}.json"
  if [ -f "${report_path}" ]; then
    echo "Skip ${method}: ${report_path} exists"
    continue
  fi

  echo "=== Running ${method} ==="
  fusion_bench \
    seed=42 \
    method=yongxianwei_merging/${method} \
    modelpool=CLIPVisionModelPool/clip-vit-base-patch32_TA8_model_only \
    taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TA8 \
    fabric.loggers.root_dir="${ROOT_DIR}/logs" \
    fabric.loggers.name="${BENCHMARK}/${method}" \
    report_save_path="${report_path}"
done

echo "All evaluations complete. Reports under: ${ROOT_DIR}/reports/${BENCHMARK}/"
