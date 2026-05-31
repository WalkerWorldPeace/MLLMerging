#!/bin/bash
REPO_ROOT="${REPO_ROOT:-$(git rev-parse --show-toplevel 2>/dev/null || pwd)}"
# Evaluate a single merged Llama-3.2-3B model on the selected MergeBench-style
# lm-eval tasks. The non-code tasks (gsm8k_cot, ifeval, truthfulqa, multilingual)
# are run here; humaneval_plus / mbpp_plus are run separately via
# run_lm_eval_code.sh because they require lm_eval >= 0.4.8 + transformers 5.x.
#
# Multilingual coverage is selected via LANG_GROUP env var:
#   LANG_GROUP=fr   (default, legacy): m_mmlu_fr, arc_fr, hellaswag_fr only
#   LANG_GROUP=all  (MergeBench parity): fr/es/de/ru × m_mmlu/arc/hellaswag (12)
#
# usage:
#   CUDA_VISIBLE_DEVICES=0 bash run_lm_eval_tasks.sh <merged_model_dir>
#   LANG_GROUP=all CUDA_VISIBLE_DEVICES=0 bash run_lm_eval_tasks.sh <merged_model_dir>
#
# output:
#   outputs/llama32_3b_mergebench/eval_results/<method>/<task>/<pretrained-path>/results_*.json

set -euo pipefail

MODEL_PATH="${1:?usage: $0 <merged_model_dir>}"
METHOD="$(basename "${MODEL_PATH%/}")"
ROOT="${REPO_ROOT}"
OUT_DIR="${ROOT}/outputs/llama32_3b_mergebench/eval_results/${METHOD}"
LOG_DIR="${ROOT}/outputs/llama32_3b_mergebench/logs"
mkdir -p "$OUT_DIR" "$LOG_DIR"

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

TASKS=(
  gsm8k_cot      # math (generative, chain-of-thought)
  ifeval         # instruction following
  truthfulqa     # safety / truthfulness
  "${ML_TASKS[@]}"   # multilingual MC (MMLU/ARC/HellaSwag) per LANG_GROUP
)

for TASK in "${TASKS[@]}"; do
  TASK_OUT="${OUT_DIR}/${TASK}"
  # lm_eval saves into <output_path>/<sanitized-model-path>/; skip if non-empty
  if compgen -G "${TASK_OUT}/*/results_*.json" > /dev/null 2>&1; then
    echo "[skip] ${METHOD}/${TASK}: result exists"
    continue
  fi
  echo "[run] ${METHOD}/${TASK} on CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
  lm_eval \
    --model_args "pretrained=${MODEL_PATH},dtype=bfloat16" \
    --tasks "${TASK}" \
    --batch_size 8 \
    --output_path "${TASK_OUT}" \
    > "${LOG_DIR}/eval_${METHOD}_${TASK}.log" 2>&1 \
    && echo "[ok]   ${METHOD}/${TASK}" \
    || { echo "[FAIL] ${METHOD}/${TASK} - see log"; }
done
echo "=== ${METHOD} done ==="
