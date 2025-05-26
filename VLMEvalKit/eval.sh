#!/bin/bash
datasets="MathVista_MINI MathVision_MINI TextVQA_VAL OCRVQA_TESTCORE VizWiz GQA_TestDev_Balanced COCO_VAL ChartQA_TEST"
models="InternVL2_5-1B"
torchrun --nproc-per-node=8 run.py --data $datasets --model $models --verbose
# AUTO_SPLIT=1 python run.py --data $datasets --model $models --verbose