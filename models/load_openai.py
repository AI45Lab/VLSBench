from openai import OpenAI 
import os 
from  tqdm import tqdm
from typing import List
import time
from ..utils import encode_image, guess_image_type_from_base64

CLOSE_API_NAME = ['gpt4', 'gpt-4o', 'gpt-4o-mini', 'gpt-4-turbo','o1-mini', 'o1-preview', 'gemini-1.5-flash', 'gemini-1.5-pro']

model_name = "gpt-4o"  # local or you can specify the api name

api_key = ""
api_base = ""

if model_name == 'local':
    if "http_proxy" in os.environ:
        del os.environ["http_proxy"]
    if 'HTTP_PROXY' in os.environ:
        del os.environ["HTTP_PROXY"]
    client = OpenAI(api_key=api_key, base_url=api_base)
    model_name = client.models.list().data[0].id      
else: 
    assert model_name in CLOSE_API_NAME
    if not api_key and not api_base:    # not set, then default
        api_key = os.environ.get("OPENAI_API_KEY")
        api_base = f"https://api.openai.com/v1"
        os.environ['http_proxy'] = "http://10.1.20.57:23128"
        os.environ['https_proxy'] = "http://10.1.20.57:23128"
    client = OpenAI(api_key=api_key, base_url=api_base)
print(model_name)

gen_kwargs = {
    "max_tokens": 128, 
    "top_p": 1.0, 
    "temperature": 0.0
}

CALL_RETRY = 3
CALL_SLEEP = 2

def api_call(client: OpenAI, question, image_path):
    base64_image = encode_image(image_path)
    image_format = guess_image_type_from_base64(base64_image)
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": [
            {
                "type": "text",
                "text": question,
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:{image_format};base64,{base64_image}",
                },
            },
        ],
        }
    ]
    for _ in range(CALL_RETRY):
        try:
            completion = client.chat.completions.create(
                model=model_name, messages=messages, **gen_kwargs
            )
            output = completion.choices[0].message.content
            print(output)
            break
        except Exception as e:
            print(f"GPT_CALL Error: {model_name}:{e}")
            time.sleep(CALL_SLEEP)
            output = 'Sorry, but error occured'
            continue
    return output


def model_inference(question, image_path) -> str:
    output = api_call(client, question, image_path)
    print(f"####Model Output####\n{output}")
    return output




