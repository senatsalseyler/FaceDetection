import cv2
import numpy as np

def draw_faces(image, faces):
    image_copy = image.copy()
    for face in faces:
        x, y, w, h, conf = face
        # Draw rectangle
        cv2.rectangle(image_copy, (int(x), int(y)), (int(x+w), int(y+h)), (0, 255, 0), 2)
        # Draw confidence
        label = f'{conf:.2f}'
        cv2.putText(image_copy, label, (int(x), int(y-10)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return image_copy