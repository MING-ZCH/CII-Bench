from transformers import AutoModelForCausalLM, AutoTokenizer
# import torch
import os
# torch.manual_seed(1234)

def load_model(model_name, model_args, use_accel=False):
    model_path = model_args.get('model_path_or_name')
    tp = model_args.get('tp', 8)
    model_components = {}
    if use_accel:
        model_components['use_accel'] = True
        pass
    else:
        model_components['use_accel'] = False
        model_components['model'] = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, device_map='auto').eval()
        model_components['tokenizer'] = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model_components['model_name'] = model_name
    return model_components

def infer(prompts, **kwargs):
    model = kwargs.get('model')
    tokenizer = kwargs.get('tokenizer', None)
    use_accel = kwargs.get('use_accel', False)

    if use_accel:
        pass
    else:
        responses = []
        for prompt in prompts:
            if isinstance(prompt, dict) and 'prompt' in prompt and 'images' in prompt:
                prompt_text = prompt['prompt']
                query = tokenizer.from_list_format([{'image': image} for image in prompt['images']] + [{'text': prompt_text}])
                response, history = model.chat(tokenizer, query=query,history=None, max_new_tokens=MAX_NEW_TOKEN)
                responses.append(response)   
            else:
                raise ValueError("Invalid prompts format")
    
    return responses

if __name__ == '__main__':
    pass
