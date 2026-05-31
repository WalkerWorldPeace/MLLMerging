<div align="center">

# Closed-Form Spectral Regularization for Multi-Task Model Merging

**WUDI / SWUDI / ASWUDI spectral filtering family**

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](./LICENSE)
[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5.1-orange.svg)](https://pytorch.org/)

</div>

This repository extends the
OptMerge framework (ICLR 2026) with a unified closed-form spectral solver and
its zero-hyperparameter adaptive variant, and ships reproducible pipelines for
**CLIP-ViT, Flan-T5, Llama-3.2** and **multimodal LLM** merging.

---

## 1. Installation

We maintain four parallel conda environments to handle different `transformers`
version pins required by each benchmark. Do **not** mix versions inside a
single env.

| Env | `transformers` | `torch` | Used for |
|---|---:|---:|---|
| `fusionbench` | `4.46.3` | `2.5.1+cu124` | CLIP-ViT (B/32, B/16, L/14), Flan-T5 GLUE LoRA, all `yongxianwei_merging` algorithms |
| `fusionbench_t58` | `5.8.0` | `2.5.1+cu124` | Llama-3.2-3B MergeBench merging + `lm_eval==0.4.11` |
| `fusionbench_mllm` | `4.45.2` | `2.5.1+cu124` | Qwen2-VL / InternVL2.5 merging + VLMEvalKit |
| `fusionbench_lmms` | `4.45.2` | `2.5.1+cu124` | Qwen2-VL RefCOCO via `lmms-eval` |

> **Layout note.** This `README.md` lives in the `fusion_bench/` subdirectory of
> the `MLLMerging` repo. The Python package, Hydra `config/`, `examples/`,
> `scripts/` and `tests/` are all under `fusion_bench/`; the multimodal
> sub-projects (`LLaMA-Factory/`, `VLMEvalKit/`, `lmms-eval/`, `InternVL/`) are
> siblings of `fusion_bench/` at the repo root. Run all `fusion_bench` CLI /
> `pip` / `scripts` commands from inside `fusion_bench/`.

```bash
# Clone and enter the FusionBench package (lives in the fusion_bench/ subdir)
git clone https://github.com/WalkerWorldPeace/MLLMerging.git
cd MLLMerging/fusion_bench

# Primary env
conda create -n fusionbench python=3.11 -y
conda activate fusionbench
pip install -e .
pip install "torch==2.5.1" torchvision \
    --index-url https://download.pytorch.org/whl/cu124 --force-reinstall
pip install "transformers==4.46.3" "lm_eval==0.4.11" immutabledict langdetect

# Llama env (clone primary, only swap transformers)
conda create -n fusionbench_t58 --clone fusionbench -y
conda activate fusionbench_t58
pip install "transformers==5.8.0" "lm_eval==0.4.11" immutabledict langdetect

# MLLM env (one-shot script)
bash scripts/setup_fusionbench_mllm.sh   # builds fusionbench_mllm + fusionbench_lmms
```

> Why version-pin matters: Flan-T5 GLUE LoRA loses ≈9 points average
> accuracy on `transformers >= 5.x` because of changes to encoder-decoder
> `generate()` defaults; Llama 3 + `lm_eval 0.4.11` requires `transformers
> >=5.x`; Qwen2-VL / InternVL pin 4.45.2 to match VLMEvalKit and the original
> remote-code paths.

### Common environment variables

```bash
export REPO_ROOT="$(git rev-parse --show-toplevel)"   # the MLLMerging repo root
export FB_ROOT=${REPO_ROOT}/fusion_bench               # the FusionBench package subdir
export HF_HOME=${REPO_ROOT}/.cache/huggingface
export TOKENIZERS_PARALLELISM=false
# Llama gated repos: put HF_TOKEN in ${FB_ROOT}/.env.local (gitignored), then
source ${FB_ROOT}/.env.local
export HF_ALLOW_CODE_EVAL=1   # for humaneval_plus / mbpp_plus
# MLLM eval data:
export LMUData=${REPO_ROOT}/.cache/LMUData
```

---

## 2. Hydra CLI

All `fusion_bench` algorithms expose a Hydra entry point with the same
signature; reproduction commands in §3 are concrete instances of (run from
`${FB_ROOT}` so Hydra finds `config/` and relative `outputs/` paths resolve):

```bash
fusion_bench seed=42 \
    method=<method_path> \
    modelpool=<modelpool_path> \
    taskpool=<taskpool_path> \
    method.<key>=<override> \
    report_save_path=outputs/.../<tag>.json
```

Method config index (`config/method/yongxianwei_merging/`):

| Method | Config | Key knobs |
|---|---|---|
| WUDI | `wudi.yaml` | `iter_num=300, learning_rate=1e-5` |
| OptMerge | `wudi2.yaml` | `iter_num=300, lr=1e-5, truncate_rank_ratio=null` |
| SWUDI-soft | `iwudi.yaml` | `filter_type=exponential, exp_time=300` |
| SWUDI | `swudi.yaml` | `exp_time, truncate_rank_ratio` |
| ASWUDI | `aswudi.yaml` | `auto_rank_method=participation_sqrt` (default) |

---

## 3. Reproduction

### 3.1 CLIP-ViT (TA8 / TALL20)

```bash
conda activate fusionbench
cd ${FB_ROOT}

# B/32 TA8 — SWUDI (paper SOTA on B/32 TA8)
fusion_bench seed=42 \
  method=yongxianwei_merging/swudi \
  method.method_kwargs.exp_time=1300 \
  method.method_kwargs.truncate_rank_ratio=0.65 \
  modelpool=CLIPVisionModelPool/clip-vit-base-patch32_TA8_model_only \
  taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TA8 \
  report_save_path=outputs/reports/clip-vit-base-patch32_TA8/swudi.json

# B/32 TA8 — ASWUDI (zero-hparam)
fusion_bench seed=42 \
  method=yongxianwei_merging/aswudi \
  modelpool=CLIPVisionModelPool/clip-vit-base-patch32_TA8_model_only \
  taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TA8 \
  report_save_path=outputs/reports/clip-vit-base-patch32_TA8/aswudi.json

# B/16 TA8 — SWUDI (B/16-tuned)
fusion_bench seed=42 \
  method=yongxianwei_merging/swudi \
  method.method_kwargs.exp_time=1800 \
  method.method_kwargs.truncate_rank_ratio=0.65 \
  modelpool=CLIPVisionModelPool/clip-vit-base-patch16_TA8_model_only \
  taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TA8_B16 \
  report_save_path=outputs/reports/clip-vit-base-patch16_TA8/swudi.json

# L/14 TA8 — ASWUDI (paper SOTA on L/14)
fusion_bench seed=42 \
  method=yongxianwei_merging/aswudi \
  modelpool=CLIPVisionModelPool/clip-vit-large-patch14_TA8_model_only \
  taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TA8_L14 \
  report_save_path=outputs/reports/clip-vit-large-patch14_TA8/aswudi.json

# B/32 TALL20 — ASWUDI (paper SOTA on TALL20)
fusion_bench seed=42 \
  method=yongxianwei_merging/aswudi \
  modelpool=CLIPVisionModelPool/clip-vit-base-patch32_TALL20_model_only \
  taskpool=CLIPVisionModelTaskPool/clip-vit-classification_TALL20 \
  report_save_path=outputs/reports/clip-vit-base-patch32_TALL20/aswudi.json
```

### 3.2 Flan-T5-base GLUE LoRA r=16

```bash
conda activate fusionbench   # transformers MUST be 4.46.3
cd ${FB_ROOT}

# ASWUDI (paper SOTA on Flan-T5)
fusion_bench seed=42 \
  method=yongxianwei_merging/aswudi \
  modelpool=Seq2SeqLMPool/flan-t5-base_glue_lora16 \
  taskpool=flan-t5_glue_text_generation taskpool.num_workers=0 \
  report_save_path=outputs/reports/flan-t5-base_glue_lora16/aswudi.json

# Aggregate report
python examples/yongxianwei_merging/summarize_flan_t5_glue_reports.py
```

### 3.3 Llama-3.2-3B MergeBench

```bash
conda activate fusionbench_t58
cd ${FB_ROOT}
source ${FB_ROOT}/.env.local              # provides HF_TOKEN for gated Llama

# Llama-tuned SWUDI (paper SOTA on Llama-3.2-3B)
fusion_bench seed=42 \
  method=yongxianwei_merging/swudi \
  method.method_kwargs.exp_time=300 \
  method.method_kwargs.truncate_rank_ratio=0.85 \
  method.exclude_param_names_regex='[embed_tokens,lm_head]' \
  modelpool=CausalLMPool/mergebench/Llama-3.2-3B \
  merged_model_save_path=outputs/llama32_3b_mergebench/merged/swudi_t300_r085

# Eval (8 legacy fr-only tasks + 4 code tasks)
bash examples/yongxianwei_merging/run_lm_eval_tasks.sh \
    outputs/llama32_3b_mergebench/merged/swudi_t300_r085
bash examples/yongxianwei_merging/run_lm_eval_code.sh \
    outputs/llama32_3b_mergebench/merged/swudi_t300_r085

# Optional: 4-language family-mean (fr/es/de/ru)
LANG_GROUP=all bash examples/yongxianwei_merging/run_lm_eval_tasks.sh \
    outputs/llama32_3b_mergebench/merged/swudi_t300_r085

python examples/yongxianwei_merging/summarize_llama32_3b_reports.py
```

### 3.4 Multimodal LLM merging

We merge only the LLM-backbone 2-D matrices; vision tower, `embed_tokens`,
`lm_head`, `norm.*` and biases are kept from pretrained.

#### 3.4.1 Qwen2-VL-7B (5 LoRA experts: OCR / VQA / Geometry / Chart / Grounding)

```bash
conda activate fusionbench_mllm
cd ${REPO_ROOT}/LLaMA-Factory
OUTROOT=${REPO_ROOT}/outputs/yongxianwei_merging/mllm/merged

# ASWUDI
CUDA_VISIBLE_DEVICES=0 python run_merge.py \
  --method aswudi --auto_rank_method participation_sqrt \
  --scaling_coefficient 0.2 \
  --output_path ${OUTROOT}/aswudi_psqrt_s02

# SWUDI
CUDA_VISIBLE_DEVICES=0 python run_merge.py \
  --method swudi --exp_time 1300 --truncate_rank_ratio 0.01 \
  --scaling_coefficient 0.2 \
  --output_path ${OUTROOT}/swudi_t1300_r001_s02

# After merging Qwen2-VL: copy chat_template.json from any instruct expert
# (the base processor template makes apply_chat_template emit empty prompts).
QWEN_EXPERT=$HF_HOME/hub/models--yongxianwei--Qwen2-VL-7B-VQA/snapshots/$(
    ls $HF_HOME/hub/models--yongxianwei--Qwen2-VL-7B-VQA/snapshots | head -1)
cp -L "$QWEN_EXPERT/chat_template.json" \
      "${OUTROOT}/aswudi_psqrt_s02/chat_template.json"
```

#### 3.4.2 InternVL2.5-1B (5 full-FT experts)

```bash
conda activate fusionbench_mllm
cd ${REPO_ROOT}/LLaMA-Factory
OUTROOT=${REPO_ROOT}/outputs/yongxianwei_merging/mllm/merged

# SWUDI
CUDA_VISIBLE_DEVICES=0 python run_merge_internvl.py \
  --method swudi --exp_time 300 --truncate_rank_ratio 0.65 \
  --scaling_coefficient 0.1 \
  --output_path ${OUTROOT}/internvl_swudi_t300_r065_s01

# ASWUDI
CUDA_VISIBLE_DEVICES=0 python run_merge_internvl.py \
  --method aswudi --auto_rank_method participation_sqrt \
  --scaling_coefficient 0.1 \
  --output_path ${OUTROOT}/internvl_aswudi_psqrt_s01

# After merging InternVL: copy custom modeling files so that
# trust_remote_code=True can re-load the merged checkpoint.
MERGED=${OUTROOT}/internvl_swudi_t300_r065_s01
BASE_SNAP=$HF_HOME/hub/models--OpenGVLab--InternVL2_5-1B/snapshots/$(
    ls $HF_HOME/hub/models--OpenGVLab--InternVL2_5-1B/snapshots | head -1)
for f in configuration_intern_vit.py configuration_internvl_chat.py \
         conversation.py modeling_intern_vit.py modeling_internvl_chat.py \
         preprocessor_config.json configuration.json; do
  cp -L "$BASE_SNAP/$f" "$MERGED/$f"
done
```

#### 3.4.3 Evaluation

```bash
# 7-task VLMEvalKit (VizWiz / GQA / MathVista / MathVision / ChartQA / TextVQA / OCRVQA)
conda activate fusionbench_mllm
cd ${REPO_ROOT}/VLMEvalKit
DATASETS="MathVista_MINI MathVision_MINI TextVQA_VAL OCRVQA_TESTCORE \
          VizWiz GQA_TestDev_Balanced ChartQA_TEST"
torchrun --nproc-per-node=4 --master-port=29501 run.py \
    --data $DATASETS \
    --model merge_aswudi_psqrt_s02 merge_internvl_swudi_t300_r065_s01 \
    --verbose \
    --work-dir ${REPO_ROOT}/outputs/mllm_eval/vlmevalkit

# RefCOCO/+/g: Qwen2-VL via lmms-eval
conda activate fusionbench_lmms
cd ${REPO_ROOT}/lmms-eval
accelerate launch --num_processes=4 --main_process_port=12345 -m lmms_eval \
    --model qwen2_vl \
    --model_args pretrained=${OUTROOT}/aswudi_psqrt_s02,max_pixels=2359296 \
    --tasks refcoco_bbox_rec_val,refcoco+_bbox_rec_val,refcocog_bbox_rec_val \
    --batch_size 1 --log_samples \
    --output_path ${REPO_ROOT}/outputs/mllm_eval/lmms_eval/aswudi_psqrt_s02

# RefCOCO/+/g: InternVL native evaluator (P@1, IoU>=0.5)
cd ${REPO_ROOT}/InternVL/internvl_chat
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
torchrun --nnodes=1 --nproc_per_node=4 --master_port=63669 \
    eval/refcoco/evaluate_grounding.py \
    --checkpoint ${OUTROOT}/internvl_swudi_t300_r065_s01 \
    --datasets refcoco_val,refcoco+_val,refcocog_val --dynamic \
    --out-dir ${REPO_ROOT}/outputs/mllm_eval/refcoco/internvl_swudi_t300_r065_s01
```

Models registered in `VLMEvalKit/vlmeval/config.py` (at the repo root, one level
up from `fusion_bench/`); new merged checkpoints can be added via:

```python
'merge_my_qwen':     partial(Qwen2VLChat, model_path='/abs/path/to/merged',
                             min_pixels=256*28*28, max_pixels=1280*28*28),
'merge_my_internvl': partial(InternVLChat, model_path='/abs/path/to/merged',
                             version='V2.0'),
```

---

## 4. Repository Layout

```text
MLLMerging/                               # repo root (git toplevel = ${REPO_ROOT})
├── fusion_bench/                         # FusionBench package + configs (${FB_ROOT}); pip install -e . here
│   ├── fusion_bench/method/yongxianwei_merging/
│   │   ├── functional.py                 # canonical WUDI / SWUDI / ASWUDI implementation
│   │   ├── algorithm.py                  # FusionBench Hydra wrapper
│   │   └── task_vector.py                # task vector extraction & shape/dtype checks
│   ├── config/                           # Hydra configs (method, modelpool, taskpool, dataset)
│   │   ├── method/yongxianwei_merging/{wudi,iwudi,swudi,aswudi,wudi2}.yaml
│   │   ├── modelpool/{CLIPVisionModelPool,Seq2SeqLMPool,CausalLMPool}/...
│   │   └── taskpool/{CLIPVisionModelTaskPool,LMEvalHarnessTaskPool}/...
│   ├── examples/yongxianwei_merging/     # Reproduction launchers + diagnostics + summarizers
│   ├── scripts/                          # env setup, lm_eval helpers
│   └── tests/                            # `python -m unittest discover -v -s tests`
├── LLaMA-Factory/                        # Multimodal merging line (sibling of fusion_bench/)
│   ├── swudi_aswudi.py                   # standalone SWUDI / ASWUDI (math-identical to
│   │                                     # fusion_bench/fusion_bench/method/.../functional.py, no fb dep)
│   ├── run_merge.py                      # Qwen2-VL merging CLI
│   ├── run_merge_internvl.py             # InternVL2.5 merging CLI
│   └── run_merge_wudi2*.py               # WUDI2 / OptMerge baseline CLIs
├── VLMEvalKit/                           # 7-task evaluator
├── lmms-eval/                            # Qwen2-VL RefCOCO evaluator
└── InternVL/internvl_chat/               # InternVL native RefCOCO evaluator
```

> The `LLaMA-Factory/swudi_aswudi.py` standalone is intentionally a
> separate copy so that MLLM merging can be run without installing the full
> `fusion_bench` package. **It does NOT auto-sync** with
> `fusion_bench/fusion_bench/method/yongxianwei_merging/functional.py`; if you
> change the algorithm, port the change manually.