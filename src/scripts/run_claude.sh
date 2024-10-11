#!/bin/bash

set -x

cd /path/to/CII-Bench/src
export PYTHONPATH=$(pwd)

python infer/infer.py --config config/config_cii.yaml --split CII --mode none cot domain emotion rhetoric one-shot two-shot three-shot --model_name claude-3-5-sonnet-20240620 --output_dir results_cii --num_workers 16