import json
import re
from collections import Counter
import argparse
import os
from prettytable import PrettyTable

domain_map ={"中华传统文化":"CTC",
             "生活":"Life",
             "艺术":"Art",
             "社会": "Society",
             "政治":"Politics",
             "环境":"Env.",}

emotion_map = {"积极":"Positive",
               "消极":"Negative",
               "中性":"Neutral"}

rhetoric_map={"隐喻":"Metaphor",
              "夸张":"Exaggeration",
              "象征":"Symbol",
              "对比":"Contrast",
              "视觉错位":"Visual Dislocation",
              "拟人":"Personification",
              "类比":"Analogy",
              "对立":"Opposition",
              "其他":"Other"}

def evaluate_domain(output_dir):
    results = PrettyTable()
    results.field_names = ["Model", "Mode", "Overall", "生活", "艺术", "社会", "政治", "环境", "中国传统文化"]
    
    domain_list = ["生活", "艺术", "社会", "政治", "环境", "中华传统文化"]
    files = sorted(os.listdir(output_dir))
    for file_name in files:
        if file_name.endswith('.jsonl') and "shot" not in file_name:
            if "110b" in file_name:
                continue
            score_dict = {}
            overall = []
            model_name, split, mode = file_name.split('_')
            mode = mode.replace('.jsonl', '')
            file_path = os.path.join(output_dir, file_name)
            with open(file_path, "r", encoding='utf-8') as file:
                for line in file:
                    data = json.loads(line)
                    if data["domain"] not in score_dict:
                        score_dict[data["domain"]] = []
                    if data["status"] == "correct":
                        score_dict[data["domain"]].append(1)
                    else:
                        score_dict[data["domain"]].append(0)
                    if data["status"] == "correct":
                        overall.append(1)
                    else:
                        overall.append(0)
            
            for key in score_dict:
                score_dict[key] = sum(score_dict[key])/len(score_dict[key])
            # print(score_dict)
            overall_acc = sum(overall)/len(overall)
            if mode!="none":
                continue
            results.add_row([model_name, mode, f"{overall_acc:.2%}"]+ [f"{score_dict[k]:.2%}" for k in domain_list])
    
    print(results)
    return results.get_string()

def evaluate_emotion(output_dir):
    results = PrettyTable()
    results.field_names = ["Model", "Mode", "Overall", "积极", "消极", "中性"]
    
    files = sorted(os.listdir(output_dir))
    for file_name in files:
        if file_name.endswith('.jsonl') and "shot" not in file_name:
            if "110b" in file_name:
                continue
            score_dict = {}
            overall = []
            model_name, split, mode = file_name.split('_')
            mode = mode.replace('.jsonl', '')
            file_path = os.path.join(output_dir, file_name)
            with open(file_path, "r", encoding='utf-8') as file:
                for line in file:
                    data = json.loads(line)
                    if data["emotion"] not in score_dict:
                        score_dict[data["emotion"]] = []
                    if data["status"] == "correct":
                        score_dict[data["emotion"]].append(1)
                    else:
                        score_dict[data["emotion"]].append(0)
                    if data["status"] == "correct":
                        overall.append(1)
                    else:
                        overall.append(0)
            
            for key in score_dict:
                score_dict[key] = sum(score_dict[key])/len(score_dict[key])
            # print(score_dict)
            overall_acc = sum(overall)/len(overall)
            if mode!="none":
                continue
            results.add_row([model_name, mode, f"{overall_acc:.2%}", f"{score_dict['积极']:.2%}", f"{score_dict['消极']:.2%}", f"{score_dict['中性']:.2%}"])
    
    print(results)
    return results.get_string()
    
def evaluate_retoric(output_dir):
    results = PrettyTable()
    results.field_names = ["Model", "Mode", "Overall", "隐喻","夸张","象征","对比","视觉错位","拟人","类比","对立","其他"]
    
    files = sorted(os.listdir(output_dir))
    for file_name in files:
        if file_name.endswith('.jsonl') and "shot" not in file_name:
            if "110b" in file_name:
                continue
            score_dict = {}
            overall = []
            model_name, split, mode = file_name.split('_')
            mode = mode.replace('.jsonl', '')
            file_path = os.path.join(output_dir, file_name)
            with open(file_path, "r", encoding='utf-8') as file:
                for line in file:
                    data = json.loads(line)
                    if isinstance(data["rhetoric"], dict):
                        for r in data["rhetoric"]["choices"]:
                            if r not in score_dict:
                                score_dict[r] = []
                            if data["status"] == "correct":
                                score_dict[r].append(1)
                            else:
                                score_dict[r].append(0)
                    elif isinstance(data["rhetoric"], str):
                        if data["rhetoric"] not in score_dict:
                            score_dict[data["rhetoric"]] = []
                        if data["status"] == "correct":
                            score_dict[data["rhetoric"]].append(1)
                        else:
                            score_dict[data["rhetoric"]].append(0)
                    if data["status"] == "correct":
                        overall.append(1)
                    else:
                        overall.append(0)
            
            for key in score_dict:
                score_dict[key] = sum(score_dict[key])/len(score_dict[key])
            # print(score_dict)
            overall_acc = sum(overall)/len(overall)
            if mode!="none":
                continue
            results.add_row([model_name, mode, f"{overall_acc:.2%}"]+ [f"{score_dict[k]:.2%}" for k in rhetoric_map])
    
    print(results)
    return results.get_string()

def evaluate_image_type(output_dir):
    # similar to evaluate_retoric, but different keys
    results = PrettyTable()
    results.field_names = ["Model", "Mode", "Overall", "插画(Illustration)", "绘画(Painting)", "海报(Poster)", "单格漫画(Single-panel Comic)", "多格漫画(Multi-panel Comic)","梗图(Meme)"]
    files = sorted(os.listdir(output_dir))
    for file_name in files:
        if file_name.endswith('.jsonl') and "shot" not in file_name:
            if "110b" in file_name:
                continue
            score_dict = {}
            overall = []
            model_name, split, mode = file_name.split('_')
            mode = mode.replace('.jsonl', '')
            file_path = os.path.join(output_dir, file_name)
            with open(file_path, "r", encoding='utf-8') as file:
                for line in file:
                    data = json.loads(line)
                    if isinstance(data["image_type"], dict):
                        for r in data["image_type"]["choices"]:
                            if r not in score_dict:
                                score_dict[r] = []
                            if data["status"] == "correct":
                                score_dict[r].append(1)
                            else:
                                score_dict[r].append(0)
                    elif isinstance(data["image_type"], str):
                        if data["image_type"] not in score_dict:
                            score_dict[data["image_type"]] = []
                        if data["status"] == "correct":
                            score_dict[data["image_type"]].append(1)
                        else:
                            score_dict[data["image_type"]].append(0)
                    if data["status"] == "correct":
                        overall.append(1)
                    else:
                        overall.append(0)
            
            for key in score_dict:
                score_dict[key] = sum(score_dict[key])/len(score_dict[key])
            
            overall_acc = sum(overall)/len(overall)
            # print(score_dict)
            if mode!="none":
                continue
            results.add_row([model_name, mode, f"{overall_acc:.2%}"]+ [f"{score_dict[k]:.2%}" for k in ["插画(Illustration)", "绘画(Painting)", "海报(Poster)", "单格漫画(Single-panel Comic)", "多格漫画(Multi-panel Comic)","梗图(Meme)"]])
    print(results)
    return results.get_string()

def evaluate_difficulty(output_dir):
    results = PrettyTable()
    results.field_names = ["Model", "Mode", "Overall", "Easy", "Medium", "Hard"]
    
    files = sorted(os.listdir(output_dir))
    for file_name in files:
        if file_name.endswith('.jsonl') and "shot" not in file_name:
            if "110b" in file_name:
                continue
            score_dict = {}
            overall = []
            model_name, split, mode = file_name.split('_')
            mode = mode.replace('.jsonl', '')
            file_path = os.path.join(output_dir, file_name)
            with open(file_path, "r", encoding='utf-8') as file:
                for line in file:
                    data = json.loads(line)
                    if data["difficulty"] not in score_dict:
                        score_dict[data["difficulty"]] = []
                    if data["status"] == "correct":
                        score_dict[data["difficulty"]].append(1)
                    else:
                        score_dict[data["difficulty"]].append(0)
                    if data["status"] == "correct":
                        overall.append(1)
                    else:
                        overall.append(0)
            
            for key in score_dict:
                score_dict[key] = sum(score_dict[key])/len(score_dict[key])
            # print(score_dict)
            overall_acc = sum(overall)/len(overall)
            if mode!="none":
                continue
            results.add_row([model_name, mode, f"{overall_acc:.2%}", f"{score_dict['简单']:.2%}", f"{score_dict['中等']:.2%}", f"{score_dict['困难']:.2%}"])
    
    print(results)
    return results.get_string()

if __name__=="__main__":

    output_dir = "results_cii_with_status"
    domain_t = evaluate_domain(output_dir)
    emotion_t = evaluate_emotion(output_dir)
    retoric_t = evaluate_retoric(output_dir) 
    type_t = evaluate_image_type(output_dir)
    diff_t = evaluate_difficulty(output_dir)
    full_table = domain_t +"\n"+ emotion_t +"\n"+ retoric_t +"\n"+ type_t +"\n"+ diff_t
    
    with open("cii_results_table.txt", "w") as f:
        f.write(full_table)
