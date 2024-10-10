#!/bin/bash

set -x

cd /path/to/CII-Bench/src
export PYTHONPATH=$(pwd)

python3 infer/infer.py --config config/config_cii.yaml --split CII --mode none cot domain emotion rhetoric one-shot two-shot three-shot --model_name Qwen2-VL-7B --output_dir results_cii --batch_size 4
