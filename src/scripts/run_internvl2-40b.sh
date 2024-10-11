#!/bin/bash

set -x

cd /path/to/CII-Bench/src
export PYTHONPATH=$(pwd)

python infer/infer.py --config config/config_cii.yaml --split CII --mode none --model_name InternVL2-40B --output_dir results_cii --batch_size 100 --use_accel
sleep 5
python infer/infer.py --config config/config_cii.yaml --split CII --mode cot --model_name InternVL2-40B --output_dir results_cii --batch_size 100 --use_accel
sleep 5
python infer/infer.py --config config/config_cii.yaml --split CII --mode domain --model_name InternVL2-40B --output_dir results_cii --batch_size 100 --use_accel
sleep 5
python infer/infer.py --config config/config_cii.yaml --split CII --mode emotion --model_name InternVL2-40B --output_dir results_cii --batch_size 100 --use_accel
sleep 5
python infer/infer.py --config config/config_cii.yaml --split CII --mode rhetoric --model_name InternVL2-40B --output_dir results_cii --batch_size 100 --use_accel
sleep 5
python infer/infer.py --config config/config_cii.yaml --split CII --mode one-shot two-shot three-shot --model_name InternVL2-40B --output_dir results_cii --batch_size 100 --use_accel
