#!/usr/bin/env bash
REPO_ROOT="${REPO_ROOT:-$(git rev-parse --show-toplevel 2>/dev/null || pwd)}"
# Evaluate ASWUDI-merged InternVL2_5-1B (scale=0.1, participation_sqrt) on the
# 5 OptMerge "integrated tasks" (Table 10 in the conference version):
#   MMMU_DEV_VAL, DocVQA_VAL, ScienceQA_TEST, AI2D_TEST, InfoVQA_VAL
#
# All 5 datasets are rule-based (MCQ exact-matching or ANLS); no GPT judge
# required, so this run does not consume OpenAI quota.
#
# Hardware: 3 GPUs visible -> CUDA_VISIBLE_DEVICES=0,2,3, torchrun nproc=3.

set -euo pipefail

source "${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
# Note: fusionbench_mllm is currently transformers==5.8.0 (drifted from
# CLAUDE.md's pinned 4.45.2). VLMEvalKit needs transformers<5 for
# AutoModelForVision2Seq. fusionbench_lmms still has 4.45.2 + torch 2.5.1+cu124
# and a working vlmeval install, so we use it for InternVL VLMEvalKit runs.
conda activate fusionbench_lmms

export HF_HOME=${REPO_ROOT}/.cache/huggingface
export LMUData=${REPO_ROOT}/.cache/LMUData
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=0,2,3

cd ${REPO_ROOT}/VLMEvalKit

DATASETS="MMMU_DEV_VAL DocVQA_VAL ScienceQA_TEST AI2D_TEST InfoVQA_VAL"
MODEL=merge_internvl_aswudi_psqrt_s01
WORKDIR=${REPO_ROOT}/outputs/yongxianwei_merging/mllm/eval/vlmevalkit

torchrun --nproc-per-node=3 --master-port=29503 run.py \
  --data ${DATASETS} \
  --model ${MODEL} \
  --verbose \
  --work-dir ${WORKDIR}
