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


from eval_utils  import evaluate_vlsbench_function

arch_to_module = {
    "openai": "load_openai",
    "llava": "load_llava",
    "llava_hf": "load_llava_hf",
    "llava_next": "load_llava_next",
    "mllama": "load_mllama",
    'qwen2vl': 'load_qwen2vl',
}

def eval_main(data, data_root, model_inference, output_path):
    
    # inference output
    start_time = time.time()
    inference_output = []
    for id, item in tqdm(enumerate(data)):
        question = item['instruction']
        image_path = os.path.join(data_root, item['image_path'])
        output = model_inference(question, image_path).strip()
        print(f"#######################\n{output}")
        inference_output.append({
            "question": question,
            "image_path": image_path,
            "pred_answer": output
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
    
    result_list = [{"question": item['question'], "image_path": item['image_path'], "pred_answer": item['pred_answer'], "safe_label": eval_json['label'], 'label_str': eval_json['label_str']} for item, eval_json in zip(inference_output, evaluation_output)]
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
    parser.add_argument("--arch", type=str, default="llava")   # if you specify the openai model, you need to specify the api name in load_openai.py
    parser.add_argument("--data_root", type=str, default='~/vlsbench')
    parser.add_argument("--output_dir", type=str, default='./outputs')
    args = parser.parse_args()
    
    # Dynamic import 
    module_name = f"models.{arch_to_module[args.arch]}"
    model_module = importlib.import_module(module_name)

    data = json.load(open(os.path.join(args.data_root, "data.json"), 'r'))
    data = data[:2]

    os.makedirs(args.output_dir, exist_ok=True)

    eval_main(data, args.data_root, model_module.model_inference, output_path=os.path.join(args.output_dir, f"{args.arch}_outputs.json"))
        