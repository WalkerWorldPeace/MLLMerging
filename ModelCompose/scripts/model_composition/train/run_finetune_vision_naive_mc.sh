#!/bin/bash

# Note: set the base path to the project base path
BASE_PATH="/path/to/base_path"
cd $BASE_PATH
# Note: set the base path to the user root or other path for cache
ROOT="/path/to/user_root"
MODEL_BASE="/path/to/vicuna-7b-v1.5"
PRETRAIN_ADAPTER_PATH="checkpoints/modelcompose-vision-pretrain/mm_projector.bin"

current_time=$(date +"%Y-%m-%d %H:%M:%S")

export WANDB_API_KEY=""
export WANDB_MODE=""
export WANDB_ENTITY=""
export WANDB_PROJECT=""
export WANDB_RUN_NAME=""

DATA_FILE="data/train/llava_v1_5_mix665k_mm_vision_image_only.json"

# check_gpu_idle 0 1 2 3 4 5 6 7

HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
XDG_CACHE_HOME="$ROOT/.cache" \
HF_HOME="$ROOT/.cache/huggingface" \
PYTHONPATH=$BASE_PATH \
deepspeed --include localhost:0,1,2,3,4,5,6,7 --master_port 29952 modelcompose/train/train_multimodal.py \
    --lora_strategy same --lora_r 128 --lora_alpha 256 --mm_projector_lr 2e-5 \
    --deepspeed ./scripts/zero3.json \
    --model_name_or_path $MODEL_BASE \
    --version v1 \
    --data_path $DATA_FILE \
    --mm_vision_encoder model/clip-vit-large-patch14-336 \
    --mm_vision_select_layer -2 \
    --pretrain_mm_mlp_adapter $PRETRAIN_ADAPTER_PATH \
    --mm_projector_type mlp2x_gelu \
    --image_aspect_ratio pad \
    --group_by_modality_length False \
    --bf16 True \
    --output_dir ./checkpoints/modelcompose-vision-finetune-naive-mc \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50000 \
    --save_total_limit 1 \
    --learning_rate 2e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --dataloader_drop_last True \
    --lazy_preprocess True \
    --report_to wandb


    