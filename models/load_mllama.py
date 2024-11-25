import torch  
from transformers import MllamaForConditionalGeneration, AutoProcessor
from .utils import load_images

default_generation_config = {
    "do_sample": False, 
    "max_new_tokens": 128, 
    "top_p": 1.0, 
    "temperature": 0.0
}

# load model
print(f"[INFO] Load MllamaForConditionalGeneration")
model_path = "/mnt/hwfile/trustai/huxuhao/models/Llama-3.2-11B-Vision-Instruct"
model = MllamaForConditionalGeneration.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map='auto').to("cuda")
processor = AutoProcessor.from_pretrained(model_path)

# model inference
def model_inference(question, image_path) -> str:
    image = load_images([image_path])[0]
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": question},
            ],
        },
    ]
  
    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    inputs = processor(text=prompt, return_tensors="pt", images=image, add_special_tokens=False).to("cuda")
    with torch.no_grad():
        generate_ids = model.generate(**inputs, **default_generation_config)
        output = processor.decode(generate_ids[0][len(inputs['input_ids'][0]):], skip_special_tokens=True, clean_up_tokenization_spaces=False)
    return output