#!/bin/bash

set -x

cd /path/to/CII-Bench/src
export PYTHONPATH=$(pwd)

python infer/infer.py --config config/config_cii.yaml --split CII --mode none cot domain emotion rhetoric --model_name Qwen2-VL-72B --output_dir results_cii--batch_size 100 --use_accel

