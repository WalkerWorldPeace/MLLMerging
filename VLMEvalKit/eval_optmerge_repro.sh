#!/bin/bash
# VLMEvalKit 7-task eval for OptMerge reproduction
# Runs both InternVL2_5-1B + Qwen2-VL-7B optmerge checkpoints.
# Math metric: paper geometry subset (applied later via results.py).
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
export LMUData=${REPO_ROOT}/.cache/LMUData
export TOKENIZERS_PARALLELISM=false
# Optional: load HF_TOKEN etc. for gated repos if present.
[ -f "${REPO_ROOT}/fusion_bench/.env.local" ] && source "${REPO_ROOT}/fusion_bench/.env.local"

cd ${REPO_ROOT}/VLMEvalKit

DATASETS="MathVista_MINI MathVision_MINI TextVQA_VAL OCRVQA_TESTCORE VizWiz GQA_TestDev_Balanced ChartQA_TEST"
WORKDIR=${REPO_ROOT}/outputs/yongxianwei_merging/mllm/eval/vlmevalkit

torchrun --nproc-per-node=4 --master-port=29541 run.py \
  --data $DATASETS \
  --model merge_optmerge_s10 merge_internvl_optmerge_s01 \
  --verbose \
  --work-dir "$WORKDIR"
