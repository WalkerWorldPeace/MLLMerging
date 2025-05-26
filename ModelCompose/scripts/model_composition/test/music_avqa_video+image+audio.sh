#!/bin/bash

set -ex

BASE_PATH=$(cd "$(dirname "$0")"; pwd)
BASE_PATH=${BASE_PATH%%/scripts*}

cd $BASE_PATH

# gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
gpu_list=$1
CKPT=$2
MODEL_BASE=$3

IFS=',' read -ra GPULIST <<< "$gpu_list"

if [ -z $MODEL_BASE ]; then
  MODEL_BASE=None
fi

CHUNKS=${#GPULIST[@]}

# If $CKPT is a relative path, add "./checkpoints/" prefix
if [[ "$CKPT" != /* ]]; then
    CKPT="./checkpoints/$CKPT"
fi

# Extracting the last layer folder using basename
ANS_CKPT=$(basename $CKPT)

# Change this image+audio to image or audio in mono-modal evaluation
QUESTION_FILE=data/test/music-avqa-test_mm_video+image+audio.json
SCORE_FILE=./playground/data/eval/answers/$ANS_CKPT/MUSIC-AVQA/score_video+image+audio.txt


for IDX in $(seq 0 $((CHUNKS-1))); do
    COMMAND="TRANSFORMERS_VERBOSITY=error CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m modelcompose.eval.model_multimodal_qa_loader \
        --model-path $CKPT \
        --model-base $MODEL_BASE \
        --question-file $QUESTION_FILE \
        --answers-file ./playground/data/eval/answers/$ANS_CKPT/MUSIC-AVQA/${CHUNKS}_${IDX}.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --temperature 0 \
        --conv-mode vicuna_v1"

    if [ ${#GPULIST[@]} -gt 1 ]; then
        eval $COMMAND &
    else
        eval $COMMAND
    fi
done

wait

output_file=./playground/data/eval/answers/$ANS_CKPT/MUSIC-AVQA/merge_video+image+audio.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat ./playground/data/eval/answers/$ANS_CKPT/MUSIC-AVQA/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done
 
python modelcompose/eval/eval_music_avqa.py --answers data/test/music-avqa-test_mm_answer.json --output $output_file | tee $SCORE_FILE