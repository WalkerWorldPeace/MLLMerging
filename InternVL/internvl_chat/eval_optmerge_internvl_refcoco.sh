#!/bin/bash
# InternVL OptMerge RefCOCO eval via native InternVL evaluate_grounding.py
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

cd ${REPO_ROOT}/InternVL/internvl_chat
export PYTHONPATH="$(pwd):${PYTHONPATH}"

OUTROOT=${REPO_ROOT}/outputs/yongxianwei_merging/mllm/merged
RESDIR=${REPO_ROOT}/outputs/yongxianwei_merging/mllm/eval/refcoco
TAG=internvl_optmerge_s01
mkdir -p "$RESDIR/$TAG"

torchrun \
  --nnodes=1 --node_rank=0 --master_addr=127.0.0.1 \
  --nproc_per_node=4 --master_port=63671 \
  eval/refcoco/evaluate_grounding.py \
  --checkpoint "$OUTROOT/$TAG" \
  --datasets refcoco_val,refcoco+_val,refcocog_val \
  --dynamic --out-dir "$RESDIR/$TAG"
