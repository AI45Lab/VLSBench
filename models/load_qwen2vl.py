import torch  
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

default_generation_config = {
    "do_sample": False, 
    "max_new_tokens": 128, 
    "top_p": 1.0, 
    "temperature": 0.0
}

# load model
print(f"[INFO] Load qwenvl2forconditionalgeneration")
model_path = "/mnt/hwfile/trustai/huxuhao/models/Qwen2-VL-7B-Instruct"
model = Qwen2VLForConditionalGeneration.from_pretrained(model_path, torch_dtype=torch.bfloat16).to("cuda")
processor = AutoProcessor.from_pretrained(model_path)

# model inference
def model_inference(question, image_path) -> str:
    conversations = [    # qwen default a system message You are a helpful assistant.
        { 
            "role": "user",
            "content": 
            [
                {"type": "image", "image": image_path},
                {"type": "text", "text": question},
            ],
        },
    ]
    texts = processor.apply_chat_template(conversations, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(conversations)
    inputs = processor(
        text=texts,
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")
    output_ids = model.generate(**inputs, **default_generation_config)
    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
    output = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]
    return output