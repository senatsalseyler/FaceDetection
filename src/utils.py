import os
import cv2
import torch
import requests
from pathlib import Path

def download_weights():
    weights_url = "https://github.com/deepcam-cn/yolov5-face/releases/download/v1.0/yolov5n-face.pt"
    weights_path = os.path.join('models', 'yolov5n-face.pt')
    
    if not os.path.exists(weights_path):
        print(f"Downloading YOLOv5-face weights...")
        os.makedirs('models', exist_ok=True)
        response = requests.get(weights_url)
        with open(weights_path, 'wb') as f:
            f.write(response.content)
    return weights_path

def load_image(image_path):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at {image_path}")
    return cv2.imread(image_path)

def create_directories():
    dirs = ['models', 'data/test_images', 'data/validation_images']
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)