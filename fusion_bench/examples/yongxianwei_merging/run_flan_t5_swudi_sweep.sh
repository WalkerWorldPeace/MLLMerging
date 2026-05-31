#!/bin/bash
REPO_ROOT="${REPO_ROOT:-$(git rev-parse --show-toplevel 2>/dev/null || pwd)}"
# SWUDI (t, r) sweep on Flan-T5-base GLUE LoRA r=16, to see whether any
# (t, r) combination can close the gap to ASWUDI 0.8298.
# Existing data: SWUDI (t=1300, r=0.65) = 0.8030; r<0.5 all collapse.
# This sweep: vary t AND high r.
#
# Usage: bash run_flan_t5_swudi_sweep.sh

set -u
ROOT="${REPO_ROOT}"
REPORT_DIR="${ROOT}/outputs/yongxianwei_merging/reports/flan-t5-base_glue_lora16"
LOG_DIR="${ROOT}/outputs/yongxianwei_merging/reports/flan-t5-base_glue_lora16/logs"
mkdir -p "$REPORT_DIR" "$LOG_DIR"

# (tag, t, r)
CONFIGS=(
  "t300_r065|300|0.65"
  "t500_r065|500|0.65"
  "t3000_r065|3000|0.65"
  "t1300_r080|1300|0.80"
  "t1300_r095|1300|0.95"
  "t300_r080|300|0.80"
  "t300_r095|300|0.95"
)

GPU=0
for cfg in "${CONFIGS[@]}"; do
  IFS='|' read -r TAG T R <<< "$cfg"
  NAME="swudi_${TAG}"
  OUT="${REPORT_DIR}/${NAME}.json"
  LOG="${LOG_DIR}/${NAME}.log"

  if [[ -f "${OUT}" ]]; then
    echo "[skip] ${NAME} exists"
    GPU=$((GPU+1)); continue
  fi
  echo "[run] GPU${GPU} ${NAME} (t=${T}, r=${R})"
  nohup bash -c "source \"\${CONDA_SH:-\$HOME/miniconda3/etc/profile.d/conda.sh}\" && conda activate fusionbench && export HF_HOME=${ROOT}/.cache/huggingface && export TOKENIZERS_PARALLELISM=false && cd ${ROOT} && CUDA_VISIBLE_DEVICES=${GPU} fusion_bench seed=42 \
    method=yongxianwei_merging/swudi \
    method.method_kwargs.exp_time=${T} \
    method.method_kwargs.truncate_rank_ratio=${R} \
    modelpool=Seq2SeqLMPool/flan-t5-base_glue_lora16 \
    taskpool=flan-t5_glue_text_generation \
    taskpool.num_workers=0 \
    report_save_path=${OUT}" \
    > "${LOG}" 2>&1 &
  GPU=$((GPU+1))
done

wait
echo "=== all done at $(date +%H:%M:%S) ==="
