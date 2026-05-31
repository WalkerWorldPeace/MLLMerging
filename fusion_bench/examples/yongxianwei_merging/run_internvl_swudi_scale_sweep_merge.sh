#!/usr/bin/env bash
REPO_ROOT="${REPO_ROOT:-$(git rev-parse --show-toplevel 2>/dev/null || pwd)}"
# SWUDI low-cost closed-form scale sweep on InternVL2_5-1B.
#
# Why: mllmerging.md only has SWUDI s03 (scale=0.3, r=0.85, t=300) on InternVL,
# which lands at 3-RefCOCO=57.37 — but ASWUDI's sweet spot scale is 0.1, not
# 0.3, so the SWUDI vs ASWUDI grounding gap (~7 pt in §1.2 / §6.3 of
# mllmerging.md) is confounded with scale + rank choice. This sweep runs 4
# closed-form SWUDI merges all at scale=0.1, varying truncate_rank_ratio (r)
# and exp_time (t), to give SWUDI a fair chance:
#
#   priority | tag                              | scale | r    | t   | purpose
#   ---------+----------------------------------+-------+------+-----+--------
#   1        | internvl_swudi_t300_r085_s01     | 0.1   | 0.85 | 300 | direct scale fix of old SWUDI s03
#   2        | internvl_swudi_t200_r085_s01     | 0.1   | 0.85 | 200 | softer than t=300
#   3        | internvl_swudi_t300_r065_s01     | 0.1   | 0.65 | 300 | r close to ASWUDI mean K/D2=0.615
#   4        | internvl_swudi_t500_r065_s01     | 0.1   | 0.65 | 500 | low rank + harder mask, mimic ASWUDI saturated subspace
#
# Eval: only InternVL RefCOCO/+/g (mllmerging.md §6.2 shows InternVL 7-task is
# scale-insensitive, so we save ~hours by skipping VLMEvalKit). Eval is run by
# a separate script run_internvl_swudi_scale_sweep_refcoco.sh.
#
# Hardware: single GPU (GPU 0). Per mllmerging.md §7.1, SWUDI merge wall-clock
# is ~8 s; cephfs loading of base + 5 experts dominates at ~19 min/run, so 4
# sequential runs ≈ 76 min total.
#
# Env: fusionbench_lmms (transformers 4.45.2). fusionbench_mllm has been
# upgraded to 5.8.0 which breaks the InternVL trust_remote_code path
# (see mllmerging.md §8.4 / §7.3 notes).

set -euo pipefail

source "${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
conda activate fusionbench_lmms

export HF_HOME=${REPO_ROOT}/.cache/huggingface
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

cd ${REPO_ROOT}/LLaMA-Factory

OUTROOT=${REPO_ROOT}/outputs/yongxianwei_merging/mllm/merged

# Locate InternVL2_5-1B base snapshot once (used to copy custom-code files
# into every merged dir, so that AutoModel.from_pretrained(trust_remote_code=
# True) works downstream).
BASE_HUB=$HF_HOME/hub/models--OpenGVLab--InternVL2_5-1B/snapshots
if [[ ! -d "$BASE_HUB" ]]; then
  echo "[ERROR] InternVL2_5-1B base snapshot dir not found at $BASE_HUB" >&2
  exit 1
fi
BASE_SNAP="$BASE_HUB/$(ls "$BASE_HUB" | head -1)"
REMOTE_FILES=(
  configuration_intern_vit.py
  configuration_internvl_chat.py
  conversation.py
  modeling_intern_vit.py
  modeling_internvl_chat.py
  preprocessor_config.json
  configuration.json
)

run_one() {
  local tag="$1"
  local scale="$2"
  local r="$3"
  local t="$4"
  local out="$OUTROOT/$tag"

  if [[ -f "$out/model.safetensors" ]]; then
    echo "[skip] $tag already has model.safetensors at $out"
  else
    echo "==================================================================="
    echo "[merge] $tag  scale=$scale r=$r t=$t  -> $out"
    echo "==================================================================="
    python run_merge_internvl.py \
      --method swudi \
      --output_path "$out" \
      --scaling_coefficient "$scale" \
      --truncate_rank_ratio "$r" \
      --exp_time "$t"
  fi

  # Copy InternVL remote-code files in (idempotent).
  echo "[remote-code] copying $(printf "%s," "${REMOTE_FILES[@]}") into $out"
  for f in "${REMOTE_FILES[@]}"; do
    if [[ -f "$BASE_SNAP/$f" ]]; then
      cp -L "$BASE_SNAP/$f" "$out/$f"
    else
      echo "[WARN] base snapshot missing $f at $BASE_SNAP" >&2
    fi
  done
}

# Order = priority order from the user's task spec.
run_one internvl_swudi_t300_r085_s01 0.1 0.85 300
run_one internvl_swudi_t200_r085_s01 0.1 0.85 200
run_one internvl_swudi_t300_r065_s01 0.1 0.65 300
run_one internvl_swudi_t500_r065_s01 0.1 0.65 500

echo
echo "All 4 SWUDI scale-sweep merges complete."
echo "Merged ckpts under: $OUTROOT/internvl_swudi_*_s01"
echo "Next step: bash examples/yongxianwei_merging/run_internvl_swudi_scale_sweep_refcoco.sh"
