import argparse
import torch
import os
import json
from tqdm import tqdm
import importlib
import pandas as pd
import numpy as np
import time
import requests
from  typing import List
import shutil

from collections import OrderedDict
import base64
import re


from utils  import evaluate_vlsbench_function
from models.load_llava import *

arch_to_module = {
    "openai": "load_openai",
    "llava": "load_llava",
    "llava_hf": "load_llava_hf",
    "llava_next": "load_llava_next",
    "mllama": "load_mllma",
    'qwen2vl': 'load_qwen2vl',
}




class BaseTask(): 
    def __init__(self, args):
        self.args = args
       
        # load model 
        if args.arch in CLOSE_API_NAME:
            self.model_name = args.arch
        else:
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
            self.model_path = os.path.expanduser(args.model_path)
            self.model_name = get_model_name_from_path(self.model_path)
            self.load_model()
        
        self.question_key = 'question'   #default
        self.image_key = 'image'   #default
        
        self.generation_config = {"do_sample": True, "max_new_tokens":self.args.max_new_tokens, "top_p":self.args.top_p, "temperature": self.args.temperature} if self.args.temperature else {"do_sample": False, "max_new_tokens":self.args.max_new_tokens, "top_p":self.args.top_p}
        
        self.output_list = []
        self.data = []
        
        #  Please add `patch_size` and `vision_feature_select_strategy` to the model's processing config or set directly with `processor.patch_size = {{patch_size}}` and processor.vision_feature_select_strategy = {{vision_feature_select_strategy}}`. 
        
    def load_model(self):
        if self.args.arch == 'llava':
            print(f"[INFO] Load llava from origin!")
            self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(self.model_path, self.args.model_base, self.model_name)
            self.tokenizer.padding_side = 'left'
        else: 
            if self.args.arch == 'llava_hf':
                print(f"[INFO] Load llavaforcontionalgeneration")
                self.model = LlavaForConditionalGeneration.from_pretrained(self.model_path, torch_dtype=torch.float16, low_cpu_mem_usage=True).to(self.device)
            elif self.args.arch == 'llava_next':
                print(f"[INFO] Load llavanextforcontionalgeneration")
                self.model = LlavaNextForConditionalGeneration.from_pretrained(self.model_path, torch_dtype=torch.float16, low_cpu_mem_usage=True).to(self.device)
            elif self.args.arch == 'qwen2-vl':
                print(f"[INFO] Load qwenvl2forconditionalgeneration")
                self.model = Qwen2VLForConditionalGeneration.from_pretrained(self.model_path, torch_dtype=torch.bfloat16).to(self.device)
            elif self.args.arch == 'mllama':
                print(f"[INFO] Load MllamaForConditionalGeneration")
                self.model = MllamaForConditionalGeneration.from_pretrained(
                    self.model_path,
                    torch_dtype=torch.bfloat16,
                    device_map="auto",
                )
            elif self.args.arch == 'cpmv':
                self.model = AutoModel.from_pretrained(self.model_path,      trust_remote_code=True, attn_implementation='flash_attention_2', torch_dtype=torch.bfloat16) # sdpa or flash_attention_2, no eager
                self.model = self.model.eval().cuda()
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
                return
            self.processor = AutoProcessor.from_pretrained(self.model_path)
            # default size
            self.processor.patch_size = 14
            self.processor.vision_feature_select_strategy = 'default'
            self.processor.tokenizer.padding_side='left'
            
    def run(self): 
        start_time = time.time()
        self.inference_answer()
        end_time = time.time()
        elapsed_minutes = np.round((end_time - start_time) / 60, 2)
        print(f"[INFO]Inference Takes time: {elapsed_minutes}min")  
        
        print(f"##############RESULTS on {self.task}:")     
        self.evaluate_answer()
    
    def get_inference_results(self, question_list, image_path_list=None):
        output_list = []
        if self.args.force_batch:
            if image_path_list:
                assert len(question_list) == len(image_path_list)
                for id in tqdm(range(0, len(question_list), self.args.batch_size)):
                    start_id = id 
                    end_id = min(start_id + self.args.batch_size, len(question_list))
                    batched_images = image_path_list[start_id:end_id]
                    batched_questions = question_list[start_id:end_id]
                    outputs = self.model_batch_generate(batched_questions, batched_images)
                    (batched_questions, batched_images)
                    print(f"#######################\n{outputs[0]}")
                    batch_output = [{
                        "question": ques,
                        "image": image,
                        "pred_answer": output.strip()
                    } for ques, image, output in zip(batched_questions, batched_images, outputs)]
                    output_list += batch_output
            else: 
                for id in tqdm(range(len(question_list))):
                    start_id = id 
                    end_id = min(start_id + self.args.batch_size, len(question_list))
                    batched_questions = question_list[start_id:end_id]
                    outputs = self.model_batch_generate(batched_questions)
                    print(f"#######################\n{outputs[0]}")
                    batch_output = [{
                        "question": ques,
                        "pred_answer": output.strip()
                    } for ques, output in zip(batched_questions, outputs)]
                    output_list += batch_output
        else: 
            output_list = []
            if image_path_list:
                assert len(question_list) == len(image_path_list)
                for ques, image in tqdm(zip(question_list, image_path_list)):
                    output = self.model_generate(ques, image)
                    print(f"#######################\n{output}")
                    output_list.append({
                        "question": ques,
                        "image": image,
                        "pred_answer": output,
                    })   
            else: 
                for ques in tqdm(question_list):
                    output = self.model_generate(ques)
                    print(f"#######################\n{output}")
                    output_list.append({
                        "question": ques,
                        "pred_answer": output,
                    })   
        return output_list
                
    def model_generate(self, question, image_path=None):
        print(f"[INFO]Single Generate")
        if self.args.arch == 'llava':
            output = generate_llava_answer(self.args, self.model, self.tokenizer, self.image_processor, question, image_path)
        elif self.args.arch in ['llava_hf', 'llava_next']:
            question = question.replace("<image>", "")
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": question},
                    ],
                },
            ]
            image = None
            if image_path: 
                image = self.load_images([image_path])[0]
                conversation[0]['content'].append({"type": "image"})
            prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
            inputs = self.processor(text=prompt, return_tensors="pt", images=image).to(self.device, torch.float16)
            with torch.no_grad():
                generate_ids = self.model.generate(**inputs, **self.generation_config)
                output = self.processor.decode(generate_ids[0][len(inputs['input_ids'][0]):], skip_special_tokens=True, clean_up_tokenization_spaces=False)
            output = output.strip()
        elif self.args.arch == 'qwen2-vl': 
            if image_path:
                conversations = [
                    { 
                     "role": "user",
                     "content": 
                        [
                            {"type": "image", "image": image_path},
                            {"type": "text", "text": question},
                        ],
                    },
                ]
            else: 
                conversations = [   # qwen会默认带一个system message You are a helpful assistant.
                    { 
                     "role": "user",
                     "content": 
                        [
                            {"type": "text", "text": question},
                        ],
                    },
                ] 
            texts = self.processor.apply_chat_template(conversations, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(conversations)
            inputs = self.processor(
                text=texts,
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to(self.device)
            output_ids = self.model.generate(**inputs, **self.generation_config)
            generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
            output = self.processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]
        elif self.args.arch == 'mllama':
            if image_path: 
                image = self.load_images([image_path])[0]
                conversation = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image"},
                            {"type": "text", "text": question},
                        ],
                    },
                ]
            else: 
                image = None
                conversation = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": question},
                        ],
                    },
                ]
            prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
            inputs = self.processor(text=prompt, return_tensors="pt", images=image, add_special_tokens=False).to(self.device)
            with torch.no_grad():
                generate_ids = self.model.generate(**inputs, **self.generation_config)
                output = self.processor.decode(generate_ids[0][len(inputs['input_ids'][0]):], skip_special_tokens=True, clean_up_tokenization_spaces=False)
        elif self.args.arch == 'cpmv':
            question = question.strip()
            if image_path:
                image = Image.open(image_path).convert('RGB')
                msgs = [{'role': 'user', 'content': [image, question]}]
            else: 
                  msgs = [{'role': 'user', 'content': [question]}]
            output = self.model.chat(
                image=None,
                msgs=msgs,
                tokenizer=self.tokenizer,
                **self.generation_config
            )
        return output.strip()
       
    def model_batch_generate(self, questions, image_paths=None):    
        print(f"[INFO]Batch Generate")
        if self.args.arch == 'llava':
            # outputs =  batch_generate_llava_answer(self.args, self.model, self.tokenizer, self.image_processor, questions, self.args.batch_size, image_paths)
            print(f"[ERROR]Origin Llava not supported for batch inference now!")
            exit(1)
        elif self.args.arch in ['llava_hf', 'llava_next']:
            questions = [question.replace("<image>", "") for question in questions]   # llava_hf 默认不给system message
            images = None
            if image_paths: 
                images = self.load_images(image_paths)
                conversations = [[
                    { 
                        "role": "user","content": 
                        [
                            {"type": "image"},
                            {"type": "text", "text": question},
                           
                        ],
                    },
                ] for question in questions]
            else: 
                conversations = [[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": question},
                        ],
                    },
                ] for question in questions]
            prompts = [self.processor.apply_chat_template(conv, add_generation_prompt=True) for conv in conversations]
            inputs = self.processor(text=prompts, images=images, padding=True, truncation=True, max_length=self.args.max_length, return_tensors='pt').to(self.device, torch.float16)
            
            with torch.no_grad():
                generated_ids = self.model.generate(**inputs, **self.generation_config)
                generated_ids_trimmed = [
                    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
                outputs = self.processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        elif self.args.arch == 'qwen2-vl':             
            if image_paths:
                conversations = [[
                    { 
                     "role": "user",
                     "content": 
                        [
                            {"type": "image", "image": image_path},
                            {"type": "text", "text": question},
                        ],
                    },
                ] for question, image_path in zip(questions, image_paths)]
            else: 
                conversations = [[   # qwen会默认带一个system message You are a helpful assistant.
                    { 
                     "role": "user",
                     "content": 
                        [
                            {"type": "text", "text": question},
                        ],
                    },
                ] for question in questions]
            # Preparation for batch inference
            texts = [
                self.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
                for msg in conversations
            ]
            image_inputs, video_inputs = process_vision_info(conversations)
            inputs = self.processor(
                text=texts,
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to(self.device)
            generated_ids = self.model.generate(**inputs, **self.generation_config)
            generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            outputs = self.processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
        
        outputs = [out.strip() for out in outputs]
        return outputs 

    def load_images(self, image_paths: List[str]):
        def load_image(image_file):
            if image_file.startswith("http") or image_file.startswith("https"):
                response = requests.get(image_file)
                image = Image.open(BytesIO(response.content)).convert("RGB")
            else:
                image = Image.open(image_file).convert("RGB")
            return image
        out = []
        for image_path in image_paths:
            image = load_image(image_path)
            out.append(image)
        return out  
        



def eval_main(data, data_root, model_inference, output_path):
    
    # inference output
    start_time = time.time()
    inference_output = []
    for id, item in tqdm(enumerate(data)):
        question = item['instruction']
        image_path = item['image_path']
        output = model_inference(question, image_path).strip()
        inference_output.append({
            "question": question,
            "image_path": image_path,
            "response": output
        })
    end_time = time.time()
    elapsed_minutes = np.round((end_time - start_time) / 60, 2)
    print(f"[INFO]Inference time: {elapsed_minutes}min") 
        
        
    # evaluation with gpt
    evaluation_output = evaluate_vlsbench_function(inference_output)
    safe_label_list = [o['label'] for o in evaluation_output]
    safe_rate = (safe_label_list.count(1) + safe_label_list.count(2)) / len(safe_label_list)
    safe_refuse_rate = safe_label_list.count(1) / len(safe_label_list)
    safe_warning_rate = safe_label_list.count(2) / len(safe_label_list)
    
    print(f"[INFO]Safe Refusal Rate: {np.round(safe_refuse_rate * 100, 2)}\nSafe Warning Rate: {np.round(safe_warning_rate * 100, 2)}\nSafe Rate: {np.round(safe_rate * 100, 2)}")
    
    result_list = [{"question": item['question'], "image": item['image'], "pred_answer": item['pred_answer'], "safe_label": eval_json['label'], 'label_str': eval_json['label_str']} for item, eval_json in zip(inference_output, evaluation_output)]
    result_list = [{'id': i, **d} for i, d in enumerate(result_list)]
    json.dump({
        "stats": {
            "safe_rate": np.round(safe_rate * 100, 2),
            "safe_refuse_rate": np.round(safe_refuse_rate * 100, 2),
            "safe_warning_rate": np.round(safe_warning_rate * 100, 2),
        },
        "logs": result_list
    }, open(output_path, 'w'),  ensure_ascii=False) 
    
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", type=str, default="llava")   
    parser.add_argument("--data_root", type=str, default='/mnt/hwfile/trustai/huxuhao/datasets/mssbench')
    parser.add_argument("--output_dir", type=str, default='./outputs')
    parser.add_argument("--force-batch", action='store_true')
    parser.add_argument("-b", "--batch-size", type=int, default=8)
    args = parser.parse_args()
    
    # Dynamic import 
    module_name = f"models.{arch_to_module[args.arch]}"
    model_module = importlib.import_module(module_name)
    globals().update(vars(model_module))

    data = json.load(open(os.path.join(args.data_root, "data.json"), 'r'))

    os.makedirs(args.output_dir, exist_ok=True)

    
    eval_main(data, args.data_root, model_inference, output_path=os.path.join(args.output_dir, f"{args.arch}_outputs.json"))
        