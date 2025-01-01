import torch
import numpy as np
from models.yolov5.models.experimental import attempt_load
from models.yolov5.utils.datasets import letterbox
from models.yolov5.utils.general import non_max_suppression_face

class FaceDetector:
    def __init__(self, weights_path, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.model = attempt_load(weights_path, map_location=device)
        self.model.eval()

    def preprocess_image(self, image):
        # Resize and pad image
        img = letterbox(image, new_shape=640)[0]
        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(self.device)
        img = img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        return img

    def detect_faces(self, image, conf_thres=0.3, iou_thres=0.5):
        # Preprocess image
        img = self.preprocess_image(image)
        
        # Inference
        with torch.no_grad():
            pred = self.model(img)[0]
        
        # Apply NMS
        pred = non_max_suppression_face(pred, conf_thres, iou_thres)
        
        # Process detections
        faces = []
        if len(pred[0]) > 0:
            # Rescale boxes from img_size to im0 size
            det = pred[0]
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], image.shape).round()
            
            # Convert to [x, y, w, h] format
            for *xyxy, conf, _ in det:
                x1, y1, x2, y2 = [int(x) for x in xyxy]
                w = x2 - x1
                h = y2 - y1
                faces.append([x1, y1, w, h, float(conf)])
                
        return faces