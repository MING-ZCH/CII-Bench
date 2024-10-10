#!/bin/bash

set -x

# activate your conda environment

cd /path/to/CII-Bench/src
export PYTHONPATH=$(pwd)


python infer/infer.py --config config/config_cii.yaml --split CII --mode one cot domain emotion rhetoric one-shot two-shot three-shot --model_name gpt4o --output_dir results_cii --num_workers 8