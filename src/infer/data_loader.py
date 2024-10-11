import json
import yaml
import os
import random
random.seed(42)

IMAGE_ROOT="/path/to/CII-Bench/"

def read_json_or_jsonl(data_path, split='', mapping_key=None):
    base_path = f'{data_path}/{split}'
    if os.path.exists(f'{base_path}.json'):
        file_path = f'{base_path}.json'
    elif os.path.exists(f'{base_path}.jsonl'):
        file_path = f'{base_path}.jsonl'
    else:
        raise FileNotFoundError(f"No JSON or JSONL file found of {base_path}.")
    
    data = []
    with open(file_path, 'r') as file:
        if file_path.endswith('.json'):
            json_data = json.load(file)
        elif file_path.endswith('.jsonl'):
            json_data = [json.loads(line) for line in file]
        for image_data in json_data:
            for item in image_data["questions"]:
                item["local_path"] = image_data["local_path"]
                for key, value in image_data["meta_data"].items():
                    item[key] = value
                data.append(item)

# Read the YAML template
def read_yaml(config='default'):
    with open(f'config/prompt/{config}.yaml', 'r') as yaml_file:
        return yaml.safe_load(yaml_file)

# Load the data
def load_data(split='CII', mode='none'):
    if split in ["CII"] and mode in ['one-shot','two-shot','three-shot']:
        template = read_yaml("none")
        samples = read_json_or_jsonl('../../data', "test") + read_json_or_jsonl('../../data', "dev")
        # Last three examples are used for few-shot examples
        examples = [exa for exa in samples if exa["id"] in ["dev-33","dev-34","dev-35"]]
        samples = [exa for exa in samples if exa["id"] not in ["dev-33","dev-34","dev-35"]]
        for sample in samples:
            if mode == 'one-shot':
                example = examples[:1]
            elif mode == 'two-shot':
                example = examples[:2]
            elif mode == 'three-shot':
                example = examples
            prompts = {"id":sample["id"],
                       "few-shot":True,
                       "conversations":[]}
            example = example + [sample]
            for turn_id, instance in enumerate(example):
                for turn in instance["questions"]:
                    question = template[f'prompt_format'][0].format(question=turn["full_question"])
                    answer = turn["answer"]
                    
                    image_path = IMAGE_ROOT + "/".join(turn['local_path'])
                    
                    if turn_id == len(example)-1:
                        prompt = {'prompt': question, 'images': [image_path], "id": sample["id"]+"-turn-"+str(turn_id)}
                    else:
                        prompt = {'prompt': question, 'images': [image_path], "response":answer, "id": sample["id"]+"-turn-"+str(turn_id)}
                        
                    prompts["conversations"].append(prompt)
            
            yield prompts, sample
            

    elif split in ['CII'] and mode in ['none', 'cot', 'domain', 'emotion', 'rhetoric']:
        if mode == 'none' or mode == 'cot':
            config = mode
        else:
            config = 'key-words'
        template = read_yaml(config)
        samples = read_json_or_jsonl('../../data', "test") + read_json_or_jsonl('../../data', "dev")
        for sample in samples:        
            image_path = IMAGE_ROOT + "/".join(sample['local_path'])
            if mode == 'none' or mode == 'cot':
                prompt = {'prompt': template[f'prompt_format'][0].format(question=sample["full_question"]), 'images': [image_path], "id": sample["id"]}

            elif mode == 'domain' or mode == 'emotion' or mode == 'rhetoric':
                key_words = sample["meta_data"][mode]['choices'] if isinstance(sample[mode], dict) else [sample["meta_data"][mode]]
                prompt = {'prompt': template[f'prompt_format'][0].format(key_words=",".join(key_words), question=sample["full_question"]), 'images': [image_path], "id": sample["id"]}
            else:
                raise ValueError(f"Invalid mode: {mode}")
            yield prompt, sample

if __name__ == '__main__':
    for prompt, sample in load_data('CII', 'none'):
        print(prompt)
        break