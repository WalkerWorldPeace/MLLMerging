#!/bin/bash

# Note: set the base path to the project base path
BASE_PATH="/path/to/base_path"
cd $BASE_PATH
# Note: set the base path to the user root or other path for cache
ROOT="/path/to/user_root"
MODEL_BASE="/path/to/vicuna-7b-v1.5"

current_time=$(date +"%Y-%m-%d %H:%M:%S")

export WANDB_API_KEY=""
export WANDB_MODE=""
export WANDB_ENTITY=""
export WANDB_PROJECT=""
export WANDB_RUN_NAME=""

DATA_FILE="data/train/PointLLM_brief_description_660K_filtered_mm.json"

HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
XDG_CACHE_HOME="$ROOT/.cache" \
HF_HOME="$ROOT/.cache/huggingface" \
PYTHONPATH=$BASE_PATH \
deepspeed --include localhost:0,1,2,3,4,5,6,7 --master_port 29952  modelcompose/train/train_multimodal.py \
    --deepspeed ./scripts/zero3.json \
    --model_name_or_path $MODEL_BASE \
    --freeze_backbone True \
    --version v1 \
    --data_path $DATA_FILE \
    --mm_point_encoder model/PointLLM/point_bert_v1.2.pt \
    --mm_point_projector_type mlp2x_gelu \
    --tune_mm_mlp_adapter True \
    --bf16 True \
    --output_dir ./checkpoints/modelcompose-point-pretrain \
    --num_train_epochs 3 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 2400 \
    --save_total_limit 1 \
    --learning_rate 2e-3 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb