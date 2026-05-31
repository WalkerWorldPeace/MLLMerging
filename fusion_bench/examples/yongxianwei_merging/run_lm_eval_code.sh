#!/bin/bash
REPO_ROOT="${REPO_ROOT:-$(git rev-parse --show-toplevel 2>/dev/null || pwd)}"
# Evaluate a merged Llama-3.2-3B model on humaneval_plus + mbpp_plus via lm_eval 0.4.11.
# These require --confirm_run_unsafe_code and HF_ALLOW_CODE_EVAL=1 because
# the metric executes the model-generated Python code to compute pass@1.
#
# usage:
#   CUDA_VISIBLE_DEVICES=0 bash run_lm_eval_code.sh <merged_model_dir>

set -euo pipefail

MODEL_PATH="${1:?usage: $0 <merged_model_dir>}"
METHOD="$(basename "${MODEL_PATH%/}")"
ROOT="${REPO_ROOT}"
OUT_DIR="${ROOT}/outputs/llama32_3b_mergebench/eval_results/${METHOD}"
LOG_DIR="${ROOT}/outputs/llama32_3b_mergebench/logs"
mkdir -p "$OUT_DIR" "$LOG_DIR"

export HF_ALLOW_CODE_EVAL=1

TASKS=(humaneval_plus mbpp_plus)

for TASK in "${TASKS[@]}"; do
  TASK_OUT="${OUT_DIR}/${TASK}"
  if compgen -G "${TASK_OUT}/*/results_*.json" > /dev/null 2>&1; then
    echo "[skip] ${METHOD}/${TASK}: result exists"
    continue
  fi
  echo "[run] ${METHOD}/${TASK} on CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
  lm-eval run \
    --model hf \
    --model_args "pretrained=${MODEL_PATH},dtype=bfloat16" \
    --tasks "${TASK}" \
    --batch_size 8 \
    --confirm_run_unsafe_code \
    --output_path "${TASK_OUT}" \
    > "${LOG_DIR}/eval_${METHOD}_${TASK}.log" 2>&1 \
    && echo "[ok]   ${METHOD}/${TASK}" \
    || { echo "[FAIL] ${METHOD}/${TASK} - see log"; }
done
echo "=== ${METHOD} code eval done ==="