from transformers import AutoTokenizer, AutoModelForCausalLM
from lmdeploy import pipeline, TurbomindEngineConfig, GenerationConfig, ChatTemplateConfig
from lmdeploy.vl import load_image
from PIL import Image
from utils.vl_utils import make_interleave_content

def encode_image(image_path):
    # Open and encode the image
    try:
        with Image.open(image_path) as image:
            return image.convert("RGBA")
    except:
        print (f"error image encoder:{image_path}")
    
def load_model(model_name, model_args, use_accel=False):
    model_path = model_args.get('model_path_or_name')
    tp = model_args.get('tp', 8)
    model_components = {}
    if use_accel:
        model_components['use_accel'] = True
        # model_components['chat_template'] = get_chat_template_from_config(model_path)
        model_components['tokenizer'] = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model_components['model'] = pipeline(model_path,backend_config=TurbomindEngineConfig(tp=tp,session_len=8192))
        # model_components['model'] = LLM(model=model_path, tokenizer=model_path, gpu_memory_utilization=0.95, tensor_parallel_size=tp, trust_remote_code=True, disable_custom_all_reduce=True, enforce_eager=True)
        model_components['model_name'] = model_name
    else:
        print ('use_accel1 pending...')
    return model_components

def infer(prompts, **kwargs):
    model = kwargs.get('model')
    tokenizer = kwargs.get('tokenizer', None)
    # chat_template = kwargs.get('chat_template', None)
    model_name = kwargs.get('model_name', None)
    use_accel = kwargs.get('use_accel', False)
    max_new_tokens = 2048
    gen_config = GenerationConfig(max_new_tokens=max_new_tokens, temperature=0.8, top_p=0.95)

    inputs = []
    for prompt in prompts:
        text = prompt['prompt']
        images = []
        for _ in prompt['images']:
            images.append(encode_image(_))
        inputs.append((text,images))


    if use_accel:
        outputs = model(inputs, gen_config=gen_config)
        responses = []
        for output in outputs:
            response = output.text
            responses.append(response)
        

    return responses

if __name__ == '__main__':
    pass