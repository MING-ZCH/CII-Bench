from zhipuai import ZhipuAI
from utils.vl_utils import make_interleave_content

def load_model(model_name="GPT4", base_url="", api_key="", model=""):
    model_components = {}
    model_components['model_name'] = model_name
    model_components['model'] = model
    model_components['base_url'] = base_url
    model_components['api_key'] = api_key
    return model_components


def request_with_images(texts_or_image_paths, timeout=30, max_tokens=2000, base_url="", api_key="", model="gpt-4o", model_name=None):
    client =ZhipuAI(api_key=api_key)
    include_system = False
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": make_interleave_content(texts_or_image_paths),
            }  
        ],  
    )
    return response

def infer(prompts, **kwargs):
    model = kwargs.get('model')
    base_url = kwargs.get('base_url')
    api_key = kwargs.get('api_key')
    model_name = kwargs.get('model_name', None)

    if isinstance(prompts, dict) and 'images' in prompts:
        prompts, images = prompts['prompt'], prompts['images']
        images = ["<|image|>" + image for image in images]
        response = request_with_images([prompts, *images], base_url=base_url, api_key=api_key, model=model, model_name=model_name).choices[0].message.content
    
    return [response]

