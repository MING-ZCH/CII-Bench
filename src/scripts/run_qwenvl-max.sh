#!/bin/bash

set -x

cd /path/to/CII-Bench/src
export PYTHONPATH=$(pwd)

python infer/infer.py --config config/config_cii.yaml --split CII --mode none cot domain emotion rhetoric --model_name qwen-vl-max --output_dir results_cii --num_workers 8
