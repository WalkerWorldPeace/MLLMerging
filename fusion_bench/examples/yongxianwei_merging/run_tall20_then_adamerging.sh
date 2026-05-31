#!/bin/bash
REPO_ROOT="${REPO_ROOT:-$(git rev-parse --show-toplevel 2>/dev/null || pwd)}"
# Build SWUDI / ASWUDI anchors on CLIP-ViT-B/32 TALL20 and run residual
# AdaMerging on top of each (lr=1e-4, init=0, 200 steps), mirroring the
# B/32 / B/16 / L/14 TA8 swudi_then_adamerging recipe (PLAN §9.6, §10.2).
#
# Output reports:
#   outputs/yongxianwei_merging/reports/clip-vit-base-patch32_TALL20/
#       swudi_then_adamerging_full_lr1e4_s200.json
#       aswudi_then_adamerging_full_lr1e4_s200.json
#
# Logs:
#   outputs/yongxianwei_merging/logs/clip-vit-base-patch32_TALL20/adamerging_chain/

set -euo pipefail

ROOT="${REPO_ROOT}"
cd "$ROOT"

source "${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
conda activate fusionbench
export HF_HOME=$ROOT/.cache/huggingface
export TOKENIZERS_PARALLELISM=false

MERGED_DIR="$ROOT/outputs/yongxianwei_merging/merged_clip"
REPORT_DIR="$ROOT/outputs/yongxianwei_merging/reports/clip-vit-base-patch32_TALL20"
LOG_DIR="$ROOT/outputs/yongxianwei_merging/logs/clip-vit-base-patch32_TALL20/adamerging_chain"
mkdir -p "$MERGED_DIR" "$REPORT_DIR" "$LOG_DIR"

run_anchor () {
  local METHOD="$1"   # swudi or aswudi
  local NAME="${METHOD}_b32_tall20"
  local VISION_OUT="$MERGED_DIR/${NAME}"
  local FULL_OUT="$MERGED_DIR/${NAME}_full"
  local MERGE_LOG="$LOG_DIR/${NAME}_merge.log"
  local SPLICE_LOG="$LOG_DIR/${NAME}_splice.log"
  local ADAMERGE_LOG="$LOG_DIR/${NAME}_then_adamerging.log"
  local ANCHOR_REPORT="$REPORT_DIR/${NAME}_anchor.json"
  local CHAIN_REPORT="$REPORT_DIR/${METHOD}_then_adamerging_full_lr1e4_s200.json"

  echo "==== [$METHOD] Stage 1: merge anchor → $VISION_OUT ===="
  if [[ -d "$VISION_OUT" && -f "$VISION_OUT/model.safetensors" ]]; then
    echo "anchor exists, skipping merge"
  else
    fusion_bench seed=42 \
      method=yongxianwei_merging/${METHOD} \
      modelpool=CLIPVisionModelPool/clip-vit-base-patch32_TALL20 \
      taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TALL20 \
      merged_model_save_path="$VISION_OUT" \
      report_save_path="$ANCHOR_REPORT" \
      > "$MERGE_LOG" 2>&1
  fi

  echo "==== [$METHOD] Stage 2: splice into full CLIPModel ===="
  if [[ -d "$FULL_OUT" && -f "$FULL_OUT/model.safetensors" ]]; then
    echo "full anchor exists, skipping splice"
  else
    python "$ROOT/examples/yongxianwei_merging/splice_clip_vision_into_full.py" \
      --base openai/clip-vit-base-patch32 \
      --vision "$VISION_OUT" \
      --out   "$FULL_OUT" \
      > "$SPLICE_LOG" 2>&1
  fi

  echo "==== [$METHOD] Stage 3: AdaMerging on top (lr=1e-4, steps=200, init=0) ===="
  fusion_bench seed=42 \
    method=adamerging/clip \
    method.lr=1e-4 \
    method.max_steps=200 \
    method.init_values=0 \
    modelpool=CLIPVisionModelPool/clip-vit-base-patch32_TALL20 \
    modelpool.models._pretrained_="$FULL_OUT" \
    taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TALL20 \
    report_save_path="$CHAIN_REPORT" \
    > "$ADAMERGE_LOG" 2>&1

  echo "==== [$METHOD] done. report: $CHAIN_REPORT ===="
  python -c "import json; d=json.load(open('$CHAIN_REPORT')); print('avg=', d['average']['accuracy'])"
}

run_anchor swudi
run_anchor aswudi

echo "=== ALL DONE ==="
ls -la "$REPORT_DIR" | grep -i adamerging
