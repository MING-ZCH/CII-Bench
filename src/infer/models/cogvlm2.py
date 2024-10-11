import torch
from PIL import Image

def load_model(model_name, model_args, use_accel=False):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    model_path = model_args.get('model_path_or_name')
    tp = model_args.get('tp', 8)
    model_components = {}
    if use_accel:
        model_components['use_accel'] = True
        pass
    else:
        model_components['use_accel'] = False
        model_components['model'] = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True).to('cuda').eval()
        model_components['tokenizer'] = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model_components['model_name'] = model_name

    return model_components


def infer(prompts, **kwargs):
    from transformers.image_utils import load_image
    model = kwargs.get('model')
    tokenizer = kwargs.get('tokenizer', None)
    use_accel = kwargs.get('use_accel', False)
    model_name = kwargs.get('model_name')

    if use_accel:
        pass
    else:
        responses = []
        for prompt in prompts:
            if isinstance(prompt, dict) and 'prompt' in prompt and 'images' in prompt:
                images = [Image.open(image).convert('RGB') for image in prompt['images']]
            else:
                raise ValueError("Invalid prompts format")
            
            input_by_model = model.build_conversation_input_ids(
                tokenizer,
                query=prompt['prompt'],
                history=None,
                images=[images[0]],
                template_version='chat'
            )
            inputs = {
                'input_ids': input_by_model['input_ids'].unsqueeze(0).to('cuda'),
                'token_type_ids': input_by_model['token_type_ids'].unsqueeze(0).to('cuda'),
                'attention_mask': input_by_model['attention_mask'].unsqueeze(0).to('cuda'),
                'images': [[input_by_model['images'][0].to('cuda').to(torch.bfloat16)]] if images[0] is not None else None,
            }
            gen_kwargs = {
                "max_new_tokens": 2048,
                "pad_token_id": 128002,  
            }
            with torch.no_grad():
                outputs = model.generate(**inputs, **gen_kwargs)
                outputs = outputs[:, inputs['input_ids'].shape[1]:]
                response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            responses.append(response)

    return responses

if __name__ == '__main__':
    pass