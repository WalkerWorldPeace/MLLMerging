#!/bin/bash
# Qwen2-VL OptMerge RefCOCO eval via lmms-eval
# (matches fusion_bench/README.md §3.4.3)
set -e

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "$0")" && git rev-parse --show-toplevel)}"

# Resolve conda init script: honor $CONDA_SH, else probe common locations.
CONDA_SH="${CONDA_SH:-}"
if [ -z "$CONDA_SH" ]; then
  for c in "$HOME/miniconda3/etc/profile.d/conda.sh" "$HOME/anaconda3/etc/profile.d/conda.sh" \
           "/opt/conda/etc/profile.d/conda.sh" "/data/miniconda3/etc/profile.d/conda.sh"; do
    [ -f "$c" ] && CONDA_SH="$c" && break
  done
fi
[ -f "$CONDA_SH" ] || { echo "ERROR: cannot find conda.sh; set \$CONDA_SH" >&2; exit 1; }
source "$CONDA_SH"
conda activate fusionbench_lmms

export HF_HOME=${REPO_ROOT}/.cache/huggingface
export TOKENIZERS_PARALLELISM=false
# Optional: load HF_TOKEN etc. for gated repos if present.
[ -f "${REPO_ROOT}/fusion_bench/.env.local" ] && source "${REPO_ROOT}/fusion_bench/.env.local"

cd ${REPO_ROOT}/lmms-eval
OUTDIR=${REPO_ROOT}/outputs/yongxianwei_merging/mllm/eval/lmms_eval/optmerge_s10
mkdir -p "$OUTDIR"

accelerate launch --num_processes=4 --main_process_port=12347 \
  -m lmms_eval \
  --model qwen2_vl \
  --model_args=pretrained=${REPO_ROOT}/outputs/yongxianwei_merging/mllm/merged/optmerge_s10,max_pixels=2359296,use_flash_attention_2=True \
  --tasks refcoco_bbox_rec_val,refcoco+_bbox_rec_val,refcocog_bbox_rec_val \
  --batch_size 1 --log_samples \
  --log_samples_suffix mllm_merge_optmerge_s10 \
  --output_path "$OUTDIR"
