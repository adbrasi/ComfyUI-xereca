import torch
from PIL import Image
import numpy as np
from math import gcd

class ClosestAspectRatio:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
            },
        }

    CATEGORY = "image/analysis"
    RETURN_TYPES = ("INT", "INT")
    RETURN_NAMES = ("width", "height")
    FUNCTION = "calculate_closest_aspect_ratio"

    def calculate_aspect_ratio(self, width, height):
        common_divisor = gcd(width, height)
        return width // common_divisor, height // common_divisor

    def calculate_closest_aspect_ratio(self, image):
        aspect_ratios = {
            (10, 24): (640, 1536),
            (5, 12): (640, 1536),
            (16, 28): (768, 1344),
            (4, 7): (768, 1344),
            (13, 19): (832, 1216),
            (14, 18): (896, 1152),
            (7, 9): (896, 1152),
            (1, 1): (1024, 1024),
            (18, 14): (1152, 896),
            (9, 7): (1152, 896),
            (19, 13): (1216, 832),
            (21, 12): (1344, 768),
            (7, 4): (1344, 768),
            (24, 10): (1536, 640),
            (12, 5): (1536, 640)
        }

        # Convert NHWC tensor to PIL Image
        img_array = image.squeeze(0).cpu().numpy() * 255.0
        img_pil = Image.fromarray(np.clip(img_array, 0, 255).astype(np.uint8))

        # Get image dimensions
        width, height = img_pil.size

        # Calculate aspect ratio
        aspect_ratio = self.calculate_aspect_ratio(width, height)

        # Find the closest aspect ratio
        closest_ratio = min(aspect_ratios.keys(), key=lambda x: (x[0] / x[1] - aspect_ratio[0] / aspect_ratio[1])**2)

        return aspect_ratios[closest_ratio]


import torch
import numpy as np
from PIL import Image, ImageOps
import os
import requests
import shutil
from rule34Py import rule34Py
import random
import time

class Rule34ImageFetcher:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "search_text": ("STRING", {"default": "monster girl"}),
                "is_random": ("BOOLEAN", {"default": True}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff})
            }
        }
    
    RETURN_TYPES = ("IMAGE", "MASK", "INT")
    RETURN_NAMES = ("image", "mask", "seed")
    FUNCTION = "fetch_image"
    CATEGORY = "image"

    def fetch_image(self, search_text, is_random, seed):
        random.seed(seed)
        r34Py = rule34Py()
        
        for attempt in range(3):  # Try up to 3 times
            try:
                if is_random:
                    lista_de_palavras = search_text.split()
                    search = r34Py.random_post(lista_de_palavras)
                    image_url = search.image
                    file_name = f"{search.id}_{int(time.time())}{os.path.splitext(search.image)[1]}"
                else:
                    search = r34Py.search([search_text], limit=1)
                    for result in search:
                        image_url = result.image
                        file_name = f"{result.id}_{int(time.time())}{os.path.splitext(result.image)[1]}"
                        break
                
                if not image_url:
                    raise ValueError("No image URL found")
                
                image_path = self.download_image(image_url, file_name)
                
                if image_path:
                    image, mask = self.load_image(image_path)
                    os.remove(image_path)  # Delete the image after loading
                    new_seed = random.randint(0, 0xffffffffffffffff)
                    return (image, mask, new_seed)
            
            except (requests.exceptions.MissingSchema, ValueError) as e:
                print(f"Attempt {attempt + 1} failed: {str(e)}")
                if attempt == 2:  # If this was the last attempt
                    print("All attempts failed. Returning blank image.")
                    new_seed = random.randint(0, 0xffffffffffffffff)
                    return (torch.zeros((1, 3, 64, 64)), torch.zeros((64, 64)), new_seed)
            
            except Exception as e:
                print(f"Unexpected error on attempt {attempt + 1}: {str(e)}")
                if attempt == 2:  # If this was the last attempt
                    print("All attempts failed. Returning blank image.")
                    new_seed = random.randint(0, 0xffffffffffffffff)
                    return (torch.zeros((1, 3, 64, 64)), torch.zeros((64, 64)), new_seed)

    def download_image(self, url, file_name):
        res = requests.get(url, stream=True)
        
        if res.status_code == 200:
            with open(file_name, 'wb') as f:
                shutil.copyfileobj(res.raw, f)
            print('Image successfully Downloaded: ', file_name)
            return file_name
        else:
            print('Image Couldn\'t be retrieved')
            return None

    def load_image(self, image_path):
        i = Image.open(image_path)
        i = ImageOps.exif_transpose(i)
        image = i.convert("RGB")
        image = np.array(image).astype(np.float32) / 255.0
        image = torch.from_numpy(image)[None,]
        if 'A' in i.getbands():
            mask = np.array(i.getchannel('A')).astype(np.float32) / 255.0
            mask = 1. - torch.from_numpy(mask)
        else:
            mask = torch.zeros((64,64), dtype=torch.float32, device="cpu")
        return (image, mask)

import os
import random
import torch
from PIL import Image, ImageOps
import numpy as np
import hashlib

class TextAnalysisNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "input_text": ("STRING", {"multiline": True}),
                "selected_words": ("STRING", {"default": "python;programming;code;algorithm"}),
                "folder_path": ("STRING", {"default": "/home/studio-lab-user/images"}),
                "recursive": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "STRING")
    RETURN_NAMES = ("image", "mask", "text")
    FUNCTION = "analyze_text"
    CATEGORY = "Text Analysis"

    def analyze_text(self, input_text, selected_words, folder_path, recursive):
        word_list = selected_words.split(';')
        matches = [word for word in word_list if word.lower() in input_text.lower()]
        
        txt_files = []
        image_files = []
        
        for root, dirs, files in os.walk(folder_path):
            if not recursive and root != folder_path:
                continue
            
            for file in files:
                file_path = os.path.join(root, file)
                if file.endswith('.txt'):
                    txt_files.append(file_path)
                elif file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                    image_files.append(file_path)
        
        if matches:
            best_match_files = []
            max_matches = 0
            
            for file_path in txt_files:
                with open(file_path, 'r') as f:
                    file_content = f.read()
                
                file_matches = [word for word in matches if word.lower() in file_content.lower()]
                
                if len(file_matches) > max_matches:
                    max_matches = len(file_matches)
                    best_match_files = [file_path]
                elif len(file_matches) == max_matches:
                    best_match_files.append(file_path)
            
            if best_match_files:
                chosen_file = random.choice(best_match_files)
                with open(chosen_file, 'r') as f:
                    text_content = f.read()
                
                image_path = self.find_corresponding_image(chosen_file)
                if not image_path:
                    image_path = random.choice(image_files) if image_files else None
            else:
                text_content = "No matching text files found."
                image_path = random.choice(image_files) if image_files else None
        else:
            text_content = "No matches found in the input text."
            image_path = random.choice(image_files) if image_files else None
        
        if image_path:
            try:
                image, mask = self.load_image(image_path)
                return (image, mask, text_content)
            except Exception as e:
                return (torch.zeros((1, 3, 64, 64)), torch.zeros((64, 64)), f"Error reading image file: {str(e)}")
        else:
            return (torch.zeros((1, 3, 64, 64)), torch.zeros((64, 64)), "No image found in the specified folder.")

    def find_corresponding_image(self, txt_file):
        base_name = os.path.splitext(txt_file)[0]
        image_extensions = ['.png', '.jpg', '.jpeg', '.gif', '.bmp']
        for ext in image_extensions:
            image_path = base_name + ext
            if os.path.exists(image_path):
                return image_path
        return None

    def load_image(self, image_path):
        i = Image.open(image_path)
        i = ImageOps.exif_transpose(i)
        image = i.convert("RGB")
        image = np.array(image).astype(np.float32) / 255.0
        image = torch.from_numpy(image)[None,]
        if 'A' in i.getbands():
            mask = np.array(i.getchannel('A')).astype(np.float32) / 255.0
            mask = 1. - torch.from_numpy(mask)
        else:
            mask = torch.zeros((64,64), dtype=torch.float32, device="cpu")
        return (image, mask)

    @classmethod
    def IS_CHANGED(s, input_text, selected_words, folder_path, recursive):
        m = hashlib.sha256()
        m.update(input_text.encode('utf-8'))
        m.update(selected_words.encode('utf-8'))
        m.update(folder_path.encode('utf-8'))
        m.update(str(recursive).encode('utf-8'))
        return m.digest().hex()

NODE_CLASS_MAPPINGS = {
    "TextAnalysisNode": TextAnalysisNode,
    "Rule34ImageFetcher": Rule34ImageFetcher,
    "ClosestAspectRatio": ClosestAspectRatio
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TextAnalysisNode": "Text Analysis",
    "Rule34ImageFetcher": "Rule34ImageFetcher",
    "ClosestAspectRatio": "Closest Aspect Ratio"
}