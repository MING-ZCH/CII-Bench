# -*- coding: utf-8 -*-

import json
import re
from collections import Counter
import argparse
import os
from prettytable import PrettyTable

def extract_option_labels(text, options=None):
    if isinstance(text, dict):
        return 'error'
    text = text.strip()
    # print(text)
    if text in ["A", "B", "C", "D", "E", "F", "G", "H"]:
        return text
    
    if "\n答案" in text:
        # 取\n答案后的内容
        text = text.split("\n答案")[-1]
        for char in ["A", "B", "C", "D", "E", "F", "G", "H"]:
            if char in text:
                return char
    if "\n**答案：" in text:
        # 取\n**答案：后的内容
        text = text.split("\n**答案：")[-1]
        for char in ["A", "B", "C", "D", "E", "F", "G", "H"]:
            if char in text:
                return char
    
    if isinstance(text, dict):
        return 'error'
    pattern = r"\(([A-F])\)"
    matches = re.findall(pattern, text)
    
    if not matches:
        pattern = r"\b([A-F])\b"
        matches = re.findall(pattern, text)
    # matches = False
    if matches:
        counter = Counter(matches)
        most_common = counter.most_common()
        max_count = most_common[0][1]
        candidates = [item for item in most_common if item[1] == max_count]
        return candidates[-1][0]
    else:
        if options:
            counter = Counter()
            for i, option in enumerate(options, start=1):
                label = chr(64 + i)
                option_stripped = option.strip()
                if option_stripped in text:
                    counter[label] += 1
                elif text in option:
                    counter[label] += 1
            if counter:
                most_common = counter.most_common()
                max_count = most_common[0][1]
                candidates = [item for item in most_common if item[1] == max_count]
                return candidates[-1][0]
    return None

def calculate_accuracy(file_path, save_dir):
    data = []
    acc = 0
    count = 0
    err = 0
    miss = 0
    
    with open(file_path, "r") as file:
        print(file_path)
        for line in file:
            
            data_ = json.loads(line)
            data.append(data_)
    for sample in data:
        # full_ques = sample["prompt"]["prompt"].split("\n")
        # answer_text = sample["answer"]
        # for ques in full_ques:
        #     if answer_text in ques:
        #         for char in ["A", "B", "C", "D", "E", "F"]:
        #             if char in ques:
        #                 sample["answer"] = char
        #                 break
        # sample["answer"] = sample["answer"]     
        
        # for ques in sample["questions"]:
        if sample["response"] != "":
            predict = extract_option_labels(sample["response"], sample.get("options"))
            sample["extracted_answer"] = predict
            if predict and sample["answer"] == predict:
                acc += 1
                sample["status"] = "correct"
            elif predict == None:
                miss += 1
                sample["status"] = "miss"
            elif predict == 'error':
                err += 1
                sample["status"] = "error"
            else:
                sample["status"] = "incorrect"
        count += 1
    
    accuracy = acc / count
    errors = err / count
    miss = miss / count

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, os.path.basename(file_path))
    with open(save_path, "w") as file:
        for sample in data:
            json.dump(sample, file, ensure_ascii=False)
            file.write("\n")
    
    return accuracy, errors, miss

def evaluate_all_files(output_dir, save_dir):
    results = PrettyTable()
    results.field_names = ["Model", "Split", "Mode", "Accuracy", "Errors", "Miss"]
    
    files = sorted(os.listdir(output_dir))
    for file_name in files:
        if file_name.endswith('.jsonl'):
            model_name, split, mode = file_name.split('_')
            mode = mode.replace('.jsonl', '')
            file_path = os.path.join(output_dir, file_name)
            accuracy, errors, miss = calculate_accuracy(file_path, save_dir)
            results.add_row([model_name, split, mode, f"{accuracy:.2%}", f"{errors:.2%}", f"{miss:.2%}"])
    
    print(results)

def main(args):
    if args.evaluate_all:
        evaluate_all_files(args.output_dir, args.save_dir)
    else:
        print(f"Model: {args.model_name}")
        results = PrettyTable()
        results.field_names = ["Mode", "Accuracy", "Errors", "Miss"]
        
        for mode in args.mode:
            file_name = f"{args.model_name}_{args.split}_{mode}.jsonl"
            file_path = os.path.join(args.output_dir, file_name)
            if os.path.exists(file_path):
                accuracy, errors, miss = calculate_accuracy(file_path, args.save_dir)
                results.add_row([mode, f"{accuracy:.2%}", f"{errors:.2%}", f"{miss:.2%}"])
            else:
                results.add_row([mode, "File does not exist", "File does not exist", "File does not exist"])
        
        print(results)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate accuracy for different modes and splits.")
    parser.add_argument('--model_name', type=str, default='yi-vl-6b-chat', help='Model name to use')
    parser.add_argument('--split', type=str, default='test', help='Data split to use')
    parser.add_argument('--mode', nargs='+', default=['none', 'cot', 'domain', 'emotion', 'rhetoric', '1-shot', '2-shot', '3-shot'], help='Modes to use for data loading, separated by space')
    parser.add_argument('--output_dir', type=str, default='results_cii', help='Directory to read result files from')
    parser.add_argument('--save_dir', type=str, default='results_cii_with_status', help='Directory to save result files with status')
    parser.add_argument('--evaluate_all', action='store_true', help='Evaluate all files in the output directory')
    
    args = parser.parse_args()
    main(args)
