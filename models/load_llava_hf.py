from transformers import LlavaProcessor, LlavaForConditionalGeneration
import torch
import requests
from typing import List
from PIL import Image 
from io import BytesIO
from .utils import load_images



default_generation_config = {
    "do_sample": False, 
    "max_new_tokens": 128, 
    "top_p": 1.0, 
    "temperature": 0.0
}


# load_model
print(f"[INFO] Load llavaforcontionalgeneration")
model_path = "llava_hf/llava-1.5-7b-hf"
model = LlavaForConditionalGeneration.from_pretrained(model_path, torch_dtype=torch.float16, low_cpu_mem_usage=True, device_map='cuda')
processor = LlavaProcessor.from_pretrained(model_path)



# model inference
def model_inference(question, image_path) -> str:
    question = question.replace("<image>", "")
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": question},
            ],
        },
    ]
    image = load_images([image_path])[0]
    conversation[0]['content'].append({"type": "image"})
    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    inputs = processor(text=prompt, return_tensors="pt", images=image).to("cuda", torch.float16)
    with torch.no_grad():
        generate_ids = model.generate(**inputs, **default_generation_config)
        output = processor.decode(generate_ids[0][len(inputs['input_ids'][0]):], skip_special_tokens=True, clean_up_tokenization_spaces=False)
    output = output.strip()
    return output