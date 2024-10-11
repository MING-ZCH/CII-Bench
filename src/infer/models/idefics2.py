

def load_model(model_name, model_args, use_accel=False):
    from transformers import AutoProcessor, AutoModelForVision2Seq
    model_path = model_args.get('model_path_or_name')
    tp = model_args.get('tp', 8)
    model_components = {}
    if use_accel:
        model_components['use_accel'] = True
        pass
    else:
        model_components['use_accel'] = False
        model_components['model'] = AutoModelForVision2Seq.from_pretrained(model_path).to("cuda:0")
        model_components['processor'] = AutoProcessor.from_pretrained(model_path)
        model_components['model_name'] = model_name

    return model_components

def infer(prompts, **kwargs):
    from transformers.image_utils import load_image
    model = kwargs.get('model')
    model.config.use_cache = True
    processor = kwargs.get('processor', None)
    use_accel = kwargs.get('use_accel', False)
    model_name = kwargs.get('model_name')

    if use_accel:
        pass
    else:
        responses = []
        for prompt in prompts:
            if isinstance(prompt, dict) and 'prompt' in prompt and 'images' in prompt:
                prompt_text = prompt['prompt']
                images = [load_image(image) for image in prompt['images']]
            else:
                raise ValueError("Invalid prompts format")
            
            messages = [
                {
                    "role": "user",
                    "content": [{"type": "image"} for _ in range(len(images))]+ [{"type": "text", "text": prompt_text}] if images else []
                }
            ]
            
            generation_kwargs = {
                "max_new_tokens": 2048,
                "do_sample": False,
                "use_cache":True
            }
            prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
            inputs = processor(text=prompt, images=images, return_tensors="pt")
            inputs = {k: v.to("cuda:0") for k, v in inputs.items()}
            generated_ids = model.generate(**inputs, **generation_kwargs)
            response = processor.batch_decode(generated_ids[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            responses.append(response[0])
                                            
    return responses

if __name__ == '__main__':
    pass