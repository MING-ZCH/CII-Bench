import requests
from utils.vl_utils import make_interleave_content

def load_model(model_name="GPT4", base_url="", api_key="", model="gpt-4-turbo-preview"):
    model_components = {}
    model_components['model_name'] = model_name
    model_components['model'] = model
    model_components['base_url'] = base_url
    model_components['api_key'] = api_key
    return model_components

def request_with_interleave_content(interleave_content, timeout=60, base_url="", api_key="", model="", model_name=None):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
        
    payload = {
        "model": model,
        "messages": interleave_content,
        "max_tokens": 8192
        }
    
    response = requests.post("base_url", headers=headers, json=payload).json()
    response = response["choices"][0]["message"]["content"]
    return response

def infer(prompts, **kwargs):
    model = kwargs.get('model')
    base_url = kwargs.get('base_url')
    api_key = kwargs.get('api_key')
    model_name = kwargs.get('model_name', None)

   
    responses = []
    for data_id, prompt_set in enumerate(prompts):
        prompt_set = prompt_set["conversations"]
        messages = []
        for idx, data in enumerate(prompt_set):
            user_payload = {"role": "user", "content": None}
            question = data["prompt"]
        
            images = data["images"] if "images" in data.keys() else []
            images = [image for image in images if "<|image|>" in image else "<|image|>"+image]
            
            interleave_content = make_interleave_content([question] + images)
            user_payload["content"] = interleave_content
            
            messages.append(user_payload)
            
            if idx != len(prompt_set) - 1 and "response" in data.keys():
                assistant_payload = {"role": "assistant", "content": data["response"]}
                messages.append(assistant_payload)
        response = request_with_interleave_content(messages)
        responses.append(response)
    return responses

