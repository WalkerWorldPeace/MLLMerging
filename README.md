<div align='center'>

# Unifying Multimodal Large Language Model Capabilities and Modalities via Model Merging

</div>

## Checkpoints

You can find MLLM checkpoints at [🤗 Hugging Face collection](https://huggingface.co/collections/yongxianwei/mllmerging-6833cd681869c47bd0b5200a). The weights can also be automatically downloaded when running the model merging scripts below.

## QwenVL Merging

1. **Install the development version and dependencies:**
    ```bash
    cd LLaMA-Factory
    pip install -e ".[torch,metrics]" --no-build-isolation
    pip install qwen_vl_utils torchvision
    ```

2. **Select and modify the `merge_method` as needed, then run the merging script:**
    ```bash
    python model_merging.py
    ```

3. **To evaluate QwenVL on RefCOCO, RefCOCO+, and RefCOCOg:**

    - Prepare the evaluation environment:
        ```bash
        cd lmms-eval
        pip install -e .
        conda install openjdk=8
        ```
    - Download the datasets from Huggingface:
        - [RefCOCO](https://huggingface.co/datasets/lmms-lab/RefCOCO)
        - [RefCOCOplus](https://huggingface.co/datasets/lmms-lab/RefCOCOplus)
        - [RefCOCOg](https://huggingface.co/datasets/lmms-lab/RefCOCOg)
    - Run the evaluation:
        ```bash
        accelerate launch --num_processes=8 --main_process_port=12345 -m lmms_eval \
            --model qwen2_vl \
            --model_args=pretrained=merged_model_path,max_pixels=2359296 \
            --tasks refcoco_bbox_rec_val,refcoco+_bbox_rec_val,refcocog_bbox_rec_val \
            --batch_size 1 --log_samples --log_samples_suffix reproduce --output_path ./logs
        ```

---

## InternVL Merging

1. **Install dependencies:**
    ```bash
    cd InternVL
    pip install -r requirements.txt
    pip install timm
    ```

2. **Run the merging script:**
    ```bash
    cd internvl_chat
    python model_merging.py
    ```

3. **Prepare datasets for RefCOCO, RefCOCO+, and RefCOCOg:**
    ```bash
    # Create data directory and download annotation files
    mkdir -p data/refcoco && cd data/refcoco
    wget https://ofasys-wlcb.oss-cn-wulanchabu.aliyuncs.com/Qwen-VL/evaluation/refcoco/refcoco_val.jsonl
    wget https://ofasys-wlcb.oss-cn-wulanchabu.aliyuncs.com/Qwen-VL/evaluation/refcoco%2B/refcoco%2B_val.jsonl
    wget https://ofasys-wlcb.oss-cn-wulanchabu.aliyuncs.com/Qwen-VL/evaluation/refcocog/refcocog_val.jsonl
    
    # Download and unzip COCO images
    mkdir -p data/coco && cd data/coco
    wget http://images.cocodataset.org/zips/train2014.zip && unzip train2014.zip
    ```

4. **Run evaluation:**
    ```bash
    GPUS=8 bash evaluate.sh merged_model_path/ refcoco --dynamic
    ```

---

## Evaluating the Merged Model

1. **Install VLMEvalKit and configure evaluation:**
    ```bash
    cd VLMEvalKit
    pip install -e .
    ```
    - All VLMs are configured in `vlmeval/config.py`.  
    - Update the model path in `vlmeval/config.py` and select the model and evaluation datasets in `eval.sh`.

2. **Run evaluation:**
    ```bash
    bash eval.sh
    ```

3. **Summarize evaluation results:**

    To quickly summarize all evaluation results, you can run:
    ```bash
    python results.py outputs/merge_model_name
    ```

> **Note:** For reproducibility, use eager attention and load the model in float16.

---

## Modality Merging

1. **Install dependencies:**
    ```bash
    cd ModelCompose
    pip install -r requirements.txt
    ```

2. **Download required models and encoders:**
    - Pretrained LLM: [vicuna-7b-v1.5](https://huggingface.co/lmsys/vicuna-7b-v1.5)
    - Finetuned LoRAs for different modalities: [ModelCompose](https://huggingface.co/Adu2021/ModelCompose) (put in `checkpoints/`)
    - Encoders for different modalities:
        - `modelcompose/model/multimodal_encoder/beats`: [beats](https://huggingface.co/nsivaku/nithin_checkpoints)
        - `modelcompose/model/multimodal_encoder/clip-vit-large-patch14-336`: [clip-vit-large-patch14-336](https://huggingface.co/openai/clip-vit-large-patch14-336)
        - `modelcompose/model/multimodal_encoder/LanguageBind_Video_merge`: [LanguageBind_Video_merge](https://huggingface.co/LanguageBind/LanguageBind_Video_merge)

3. **Merge models:**
    ```bash
    python scripts/model_composition/merge_unimodal_modelcompose.py \
        checkpoints/multimodal-vicuna-7b-v1.5-video-naivemc \
        checkpoints/multimodal-vicuna-7b-v1.5-audio-naivemc \
        checkpoints/multimodal-vicuna-7b-v1.5-vision-naivemc \
        -o multimodal-checkpoint-name --strategy merge-ties
    ```
    - You can change the merging method with the `--strategy` argument.

4. **Evaluate the merged three-modality model:**

    - **AVQA:**
        ```bash
        bash scripts/model_composition/test/avqa.sh 0,1,2,3,4,5,6,7 multimodal-checkpoint-name video+image+audio checkpoints/vicuna-7b-v1.5
        ```
    - **MUSIC-AVQA:**
        ```bash
        bash scripts/model_composition/test/music_avqa_video+image+audio.sh 0,1,2,3,4,5,6,7 multimodal-checkpoint-name checkpoints/vicuna-7b-v1.5
        ```

---

## Acknowledgement

This project thanks the following open source communities for their contributions:

- [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)
- [InternVL](https://github.com/OpenGVLab/InternVL)
- [ModelCompose](https://github.com/THUNLP-MT/ModelCompose)
- [VLMEvalKit](https://github.com/open-compass/VLMEvalKit)
- [LMMs-Eval](https://github.com/EvolvingLMMs-Lab/lmms-eval)

Thanks to them for their contributions to the development of model training and evaluation tools!