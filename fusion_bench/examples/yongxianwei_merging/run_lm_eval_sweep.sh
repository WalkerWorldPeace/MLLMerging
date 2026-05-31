#!/bin/bash
REPO_ROOT="${REPO_ROOT:-$(git rev-parse --show-toplevel 2>/dev/null || pwd)}"
# Evaluate one sweep-merged Llama-3.2-3B on the 8 MergeBench-style tasks.
# Writes results to outputs/llama32_3b_mergebench/sweep/eval_results/<name>/<task>/...
#
# Combines legacy lm_eval CLI (6 non-code tasks) and new lm-eval run CLI
# (humaneval_plus, mbpp_plus). Skips tasks that already have a result JSON.
#
# Multilingual coverage is selected via LANG_GROUP env var:
#   LANG_GROUP=fr   (default, legacy): m_mmlu_fr, arc_fr, hellaswag_fr only
#   LANG_GROUP=all  (MergeBench parity): fr/es/de/ru × m_mmlu/arc/hellaswag (12)
#
# usage:
#   CUDA_VISIBLE_DEVICES=0 bash run_lm_eval_sweep.sh <merged_model_dir>
#   LANG_GROUP=all CUDA_VISIBLE_DEVICES=0 bash run_lm_eval_sweep.sh <merged_model_dir>

set -u

MODEL_PATH="${1:?usage: $0 <merged_model_dir>}"
NAME="$(basename "${MODEL_PATH%/}")"
ROOT="${REPO_ROOT}"
OUT_DIR="${ROOT}/outputs/llama32_3b_mergebench/sweep/eval_results/${NAME}"
LOG_DIR="${ROOT}/outputs/llama32_3b_mergebench/sweep/logs"
mkdir -p "$OUT_DIR" "$LOG_DIR"

export HF_ALLOW_CODE_EVAL=1

LANG_GROUP="${LANG_GROUP:-fr}"
case "$LANG_GROUP" in
  fr)
    ML_TASKS=(m_mmlu_fr arc_fr hellaswag_fr)
    ;;
  all)
    ML_TASKS=(
      m_mmlu_fr m_mmlu_es m_mmlu_de m_mmlu_ru
      arc_fr    arc_es    arc_de    arc_ru
      hellaswag_fr hellaswag_es hellaswag_de hellaswag_ru
    )
    ;;
  *)
    echo "[err] LANG_GROUP=$LANG_GROUP not in {fr,all}" >&2
    exit 2
    ;;
esac

NONCODE_TASKS=(gsm8k_cot ifeval truthfulqa "${ML_TASKS[@]}")
CODE_TASKS=(humaneval_plus mbpp_plus)

for TASK in "${NONCODE_TASKS[@]}"; do
  TASK_OUT="${OUT_DIR}/${TASK}"
  if compgen -G "${TASK_OUT}/*/results_*.json" > /dev/null 2>&1; then
    echo "[skip] ${NAME}/${TASK}: exists"
    continue
  fi
  echo "[run] ${NAME}/${TASK} (non-code) on CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
  lm-eval run \
    --model hf \
    --model_args "pretrained=${MODEL_PATH},dtype=bfloat16" \
    --tasks "${TASK}" \
    --batch_size 8 \
    --output_path "${TASK_OUT}" \
    > "${LOG_DIR}/eval_${NAME}_${TASK}.log" 2>&1 \
    && echo "[ok]   ${NAME}/${TASK}" \
    || echo "[FAIL] ${NAME}/${TASK}"
done

for TASK in "${CODE_TASKS[@]}"; do
  TASK_OUT="${OUT_DIR}/${TASK}"
  if compgen -G "${TASK_OUT}/*/results_*.json" > /dev/null 2>&1; then
    echo "[skip] ${NAME}/${TASK}: exists"
    continue
  fi
  echo "[run] ${NAME}/${TASK} (code) on CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
  lm-eval run \
    --model hf \
    --model_args "pretrained=${MODEL_PATH},dtype=bfloat16" \
    --tasks "${TASK}" \
    --batch_size 8 \
    --confirm_run_unsafe_code \
    --output_path "${TASK_OUT}" \
    > "${LOG_DIR}/eval_${NAME}_${TASK}.log" 2>&1 \
    && echo "[ok]   ${NAME}/${TASK}" \
    || echo "[FAIL] ${NAME}/${TASK}"
done

echo "=== ${NAME} full eval done ==="