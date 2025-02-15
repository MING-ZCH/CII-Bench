#!/bin/bash
set -x

cd /path/to/CII-Bench/src
export PYTHONPATH=$(pwd)

python infer/infer.py --config config/config_cii.yaml --split CII --mode none --model_name MiniCPM-V-2.6 --output_dir results_cii --batch_size 4
sleep 5
python infer/infer.py --config config/config_cii.yaml --split CII --mode cot --model_name MiniCPM-V-2.6 --output_dir results_cii --batch_size 4
sleep 5
python infer/infer.py --config config/config_cii.yaml --split CII --mode domain --model_name MiniCPM-V-2.6 --output_dir results_cii --batch_size 4
sleep 5
python infer/infer.py --config config/config_cii.yaml --split CII --mode emotion --model_name MiniCPM-V-2.6 --output_dir results_cii --batch_size 4
sleep 5
python infer/infer.py --config config/config_cii.yaml --split CII --mode rhetoric --model_name MiniCPM-V-2.6 --output_dir results_cii --batch_size 4
