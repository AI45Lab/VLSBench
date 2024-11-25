from typing import List  
import requests 
from io import BytesIO 
from PIL import Image 


def load_images(image_paths: List[str]):
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