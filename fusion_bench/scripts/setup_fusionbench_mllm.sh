#!/bin/bash
REPO_ROOT="${REPO_ROOT:-$(git rev-parse --show-toplevel 2>/dev/null || pwd)}"
# Rebuild fusionbench_mllm conda env after server reboot wiped envs/.
# Pinned versions reproduced from outputs/.../mllm/logs/env_setup.log:
#   - python 3.11.15
#   - torch 2.5.1+cu124
#   - transformers 4.45.2
#   - editable: vlmeval-0.1.0 (MLLMerging/VLMEvalKit), lmms_eval-0.3.0 (MLLMerging/lmms-eval)
#   - qwen_vl_utils 0.0.14, timm 1.0.27, einops 0.8.2
# Retry fix merged: do not pin numpy/opencv here; let lmms-eval/VLMEvalKit resolve them.

set -eo pipefail

ENV_NAME=fusionbench_mllm
ROOT=${REPO_ROOT}
LOG=$ROOT/outputs/yongxianwei_merging/mllm/logs/env_setup_v2.log
# Optional pip index URL. Leave empty to use default PyPI; set e.g.
#   PIP_MIRROR="https://mirrors.cloud.tencent.com/pypi/simple/"      # CN tencent
#   PIP_MIRROR="https://pypi.tuna.tsinghua.edu.cn/simple/"           # CN tsinghua
# to speed up downloads from China.
PIP_MIRROR="${PIP_MIRROR:-}"
PIP_INDEX_FLAG=${PIP_MIRROR:+-i "$PIP_MIRROR"}

mkdir -p "$(dirname "$LOG")"

echo "=== fusionbench_mllm rebuild START $(date '+%Y-%m-%d %H:%M:%S') ==="

# Step 1: ensure no stale env, then create
# Locate conda init script: respect $CONDA_SH override, fall back to common paths.
CONDA_SH="${CONDA_SH:-}"
if [[ -z "$CONDA_SH" ]]; then
    for cand in \
        "$HOME/miniconda3/etc/profile.d/conda.sh" \
        "$HOME/anaconda3/etc/profile.d/conda.sh" \
        "/opt/conda/etc/profile.d/conda.sh" \
        "/data/miniconda3/etc/profile.d/conda.sh"; do
        if [[ -f "$cand" ]]; then CONDA_SH="$cand"; break; fi
    done
fi
[[ -f "$CONDA_SH" ]] || { echo "ERROR: cannot find conda.sh; set \$CONDA_SH" >&2; exit 1; }
source "$CONDA_SH"
echo "--- Step 1: conda create -n $ENV_NAME python=3.11 ---"
conda env list | grep -q "$ENV_NAME " && {
    echo "Removing existing $ENV_NAME ..."
    conda env remove -y -n $ENV_NAME
}
conda create -y -n $ENV_NAME python=3.11.15

# Step 2: activate
conda activate $ENV_NAME
echo "Python: $(python --version)"
echo "Pip: $(pip --version)"

# Step 3: torch + cuda (PyTorch official wheels)
echo "--- Step 2: torch 2.5.1+cu124 (~3 GB) ---"
pip install --no-cache-dir torch==2.5.1 torchvision==0.20.1 \
    --index-url https://download.pytorch.org/whl/cu124

# Step 4: transformers stack + InternVL/Qwen2-VL deps (Tencent mirror is fast)
# Do not pin numpy/opencv here: lmms-eval and VLMEvalKit have conflicting constraints,
# so the editable installs below should resolve the final compatible versions.
echo "--- Step 3: transformers 4.45.2 + ML deps (Tencent mirror, no numpy/opencv pin) ---"
pip install --no-cache-dir $PIP_INDEX_FLAG \
    transformers==4.45.2 \
    accelerate==1.13.0 \
    tokenizers==0.20.3 \
    safetensors \
    huggingface_hub \
    qwen_vl_utils==0.0.14 \
    av==17.0.1 \
    timm==1.0.27 \
    sentencepiece==0.1.99 \
    einops==0.8.2 \
    Pillow \
    scipy \
    matplotlib \
    pandas \
    tqdm \
    rich \
    omegaconf \
    pyyaml \
    decord==0.6.0 \
    pycocoevalcap==1.2 \
    pycocotools==2.0.11

# Step 5: editable lmms-eval
echo "--- Step 4: lmms-eval editable ---"
cd $ROOT/lmms-eval
pip install --no-cache-dir $PIP_INDEX_FLAG -e .

# Step 6: editable VLMEvalKit
echo "--- Step 5: VLMEvalKit editable ---"
cd $ROOT/VLMEvalKit
pip install --no-cache-dir $PIP_INDEX_FLAG -e .

# Step 7: smoke test
echo "--- Step 6: smoke imports + CUDA check ---"
python - <<'PY'
import torch, transformers, numpy
print(f"torch        {torch.__version__}")
print(f"cuda avail   {torch.cuda.is_available()}")
print(f"#gpu         {torch.cuda.device_count()}")
print(f"transformers {transformers.__version__}")
print(f"numpy        {numpy.__version__}")
import vlmeval
print(f"vlmeval      OK ({vlmeval.__file__})")
import qwen_vl_utils
print(f"qwen_vl_utils OK ({qwen_vl_utils.__version__ if hasattr(qwen_vl_utils,'__version__') else 'unknown'})")
import timm
print(f"timm         {timm.__version__}")
PY

echo "=== rebuild rc=$? END $(date '+%Y-%m-%d %H:%M:%S') ==="
