#!/usr/bin/env bash
REPO_ROOT="${REPO_ROOT:-$(git rev-parse --show-toplevel 2>/dev/null || pwd)}"
# RefCOCO/+/g native eval for the 4 SWUDI scale-sweep checkpoints
# produced by run_internvl_swudi_scale_sweep_merge.sh.
#
# Eval cadence: 4-GPU torchrun on InternVL's native evaluate_grounding.py
# (P@1 = ACC@0.5, IoU>=0.5). Mirrors mllmerging.md §8.6 and the OptMerge
# eval_optmerge_internvl_refcoco.sh template.
#
# Robustness: each (ckpt, dataset) combo runs in its own torchrun. If any
# combo crashes (e.g. transient CUDA error), the rest still finish. Successful
# combos are detected by the per-dataset *.json output and skipped on re-run.
#
# Wall-clock: ~25-35 min per (ckpt, dataset) × 12 combos ≈ ~5-6 hr.

set -uo pipefail   # NOTE: no -e — we want to keep going past torchrun failures.

source "${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
conda activate fusionbench_lmms

export HF_HOME=${REPO_ROOT}/.cache/huggingface
export TOKENIZERS_PARALLELISM=false

# Pull HF_TOKEN etc. for any gated repos (matches eval_optmerge_internvl_refcoco.sh).
if [[ -f ${REPO_ROOT}/.env.local ]]; then
  source ${REPO_ROOT}/.env.local
fi

# Run from the InternVL working tree that contains
#   data/refcoco/{refcoco,refcoco+,refcocog}_val.jsonl
#   data/coco/train2014/
# evaluate_grounding.py opens 'data/refcoco/*.jsonl' relative to CWD, so we must
# run from a tree where those files exist. By default we use the in-repo mirror
# at MLLMerging/InternVL/internvl_chat; override via INTERNVL_DATA_ROOT if your
# RefCOCO data lives elsewhere (see mllmerging.md §8.6).
cd "${INTERNVL_DATA_ROOT:-${REPO_ROOT}/InternVL/internvl_chat}"
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

OUTROOT=${REPO_ROOT}/outputs/yongxianwei_merging/mllm/merged
RESDIR=${REPO_ROOT}/outputs/yongxianwei_merging/mllm/eval/refcoco

# master_port chosen to avoid collision with eval_optmerge_internvl_refcoco.sh (63671).
PORT=${PORT:-63680}
NPROC=${NPROC:-4}

failed_combos=()

# refcoco_val_*.json contains 'refcoco_val_<ts>.json'; refcoco+_val_*.json
# requires globbing literal '+' so we use compgen -G with the bare pattern.
already_have() {
  local outdir="$1"; local ds="$2"
  compgen -G "$outdir/${ds}_*.json" >/dev/null
}

run_one() {
  local tag="$1"
  local ds="$2"
  local ckpt="$OUTROOT/$tag"
  local outdir="$RESDIR/$tag"

  if [[ ! -d "$ckpt" ]]; then
    echo "[ERROR] missing merged ckpt: $ckpt" >&2
    failed_combos+=("$tag/$ds:missing-ckpt")
    return
  fi
  mkdir -p "$outdir"

  if already_have "$outdir" "$ds"; then
    echo "[skip] $tag/$ds already has ${ds}_*.json under $outdir"
    return
  fi

  echo "==================================================================="
  echo "[eval] $tag  ds=$ds  ckpt=$ckpt  out=$outdir  port=$PORT"
  echo "==================================================================="

  set +e
  torchrun \
    --nnodes=1 --node_rank=0 --master_addr=127.0.0.1 \
    --nproc_per_node="$NPROC" --master_port="$PORT" \
    eval/refcoco/evaluate_grounding.py \
    --checkpoint "$ckpt" \
    --datasets "$ds" \
    --dynamic --out-dir "$outdir"
  rc=$?
  set -e 2>/dev/null || true   # don't actually re-enable -e

  PORT=$((PORT + 1))

  if [[ $rc -ne 0 ]]; then
    echo "[WARN] $tag/$ds failed with rc=$rc — continuing with the next combo"
    failed_combos+=("$tag/$ds:rc=$rc")
    # Some torchrun crashes leave child python procs holding GPU memory. Try
    # to clean them up so the next torchrun starts on a clean slate.
    pkill -9 -f "evaluate_grounding.py" 2>/dev/null || true
    sleep 3
  fi
}

# Order: priority-1 ckpt fully, then priority-2, etc.
TAGS=(
  internvl_swudi_t300_r085_s01
  internvl_swudi_t200_r085_s01
  internvl_swudi_t300_r065_s01
  internvl_swudi_t500_r065_s01
)
DATASETS=(refcoco_val refcoco+_val refcocog_val)

for tag in "${TAGS[@]}"; do
  for ds in "${DATASETS[@]}"; do
    run_one "$tag" "$ds"
  done
done

echo
echo "==================================================================="
echo "Sweep done. Per-ckpt P@1 summaries under: $RESDIR/internvl_swudi_*_s01/merged_*.txt"
if (( ${#failed_combos[@]} > 0 )); then
  echo "Failed combos (${#failed_combos[@]}):"
  for c in "${failed_combos[@]}"; do echo "  - $c"; done
  exit 1
else
  echo "All 12 (ckpt, dataset) combos succeeded."
fi
