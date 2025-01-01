from setuptools import setup

# Clone and set up YOLOv5-face
import os
os.system('git clone https://github.com/deepcam-cn/yolov5-face.git models/yolov5')

# Download weights
import requests
weights_url = "https://github.com/deepcam-cn/yolov5-face/releases/download/v1.0/yolov5n-face.pt"
os.makedirs('models', exist_ok=True)
response = requests.get(weights_url)
with open('models/yolov5n-face.pt', 'wb') as f:
    f.write(response.content)