import cv2
import numpy as np
from insightface.app import FaceAnalysis

def run_gender_detection():
    # Initialize face analyzer
    app = FaceAnalysis(providers=['CPUExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(640, 640))
    
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detect and analyze faces
        faces = app.get(frame)
        
        for face in faces:
            # Get face information
            bbox = face.bbox.astype(int)
            gender = face.gender
            age = face.age
            
            # Set colors based on gender (Pink for female, Blue for male)
            color = (255, 192, 203) if gender == 0 else (255, 0, 0)
            
            # Draw bounding box
            cv2.rectangle(frame, 
                         (bbox[0], bbox[1]), 
                         (bbox[2], bbox[3]), 
                         color, 2)
            
            # Prepare label text
            gender_text = "Female" if gender == 0 else "Male"
            label = f"{gender_text} ({age:.0f}y)"
            
            # Add background for text
            (text_width, text_height), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(frame, 
                         (bbox[0], bbox[1]-30), 
                         (bbox[0] + text_width, bbox[1]), 
                         color, -1)
            
            # Add text
            cv2.putText(frame, label, 
                       (bbox[0], bbox[1]-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 
                       0.7, (255, 255, 255), 2)
        
        # Show the frame
        cv2.imshow('Gender Detection', frame)
        
        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Clean up
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_gender_detection()