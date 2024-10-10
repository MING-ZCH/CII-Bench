#!/bin/bash

set -x

cd /path/to/CII-Bench/src
export PYTHONPATH=$(pwd)
python infer/infer.py --config config/config_cii.yaml --split CII --mode none --model_name cogvlm2-llama3-chinese-chat-19B --output_dir results_cii --batch_size 32
sleep 5
python infer/infer.py --config config/config_cii.yaml --split CII --mode cot --model_name cogvlm2-llama3-chinese-chat-19B --output_dir results_cii --batch_size 32
sleep 5
python infer/infer.py --config config/config_cii.yaml --split CII --mode domain --model_name cogvlm2-llama3-chinese-chat-19B --output_dir results_cii --batch_size 32
sleep 5
python infer/infer.py --config config/config_cii.yaml --split CII --mode emotion --model_name cogvlm2-llama3-chinese-chat-19B --output_dir results_cii --batch_size 32
sleep 5
python infer/infer.py --config config/config_cii.yaml --split CII --mode rhetoric --model_name cogvlm2-llama3-chinese-chat-19B --output_dir results_cii --batch_size 32