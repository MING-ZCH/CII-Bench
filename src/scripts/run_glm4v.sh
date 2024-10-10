#!/bin/bash

set -x

cd /path/to/CII-Bench/src
export PYTHONPATH=$(pwd)

python infer/infer.py --config config/config_cii.yaml --split CII --mode none --model_name glm-4-v-plus --output_dir results_cii --num_workers 16
sleep 5
python infer/infer.py --config config/config_cii.yaml --split CII --mode cot --model_name glm-4-v-plus --output_dir results_cii --num_workers 16
sleep 5
python infer/infer.py --config config/config_cii.yaml --split CII --mode domain --model_name glm-4-v-plus --output_dir results_cii --num_workers 16
sleep 5
python infer/infer.py --config config/config_cii.yaml --split CII --mode emotion --model_name glm-4-v-plus --output_dir results_cii --num_workers 16
sleep 5
python infer/infer.py --config config/config_cii.yaml --split CII --mode rhetoric --model_name glm-4-v-plus --output_dir results_cii --num_workers 16
