from ultralytics import YOLO
import cv2
import numpy as np
import os

def detect_gender_live():
    # Load YOLOv8 model
    model = YOLO('yolov8n.pt')  # replace with your trained model path
    
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    
    while True:
        # Read frame
        ret, frame = cap.read()
        if not ret:
            break
            
        # Run detection on frame
        results = model(frame)
        
        # Process each detection
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Get box coordinates
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
                # Get confidence and class
                confidence = float(box.conf)
                class_id = int(box.cls)
                
                # Map class to gender (adjust based on your model's classes)
                gender = "Male" if class_id == 0 else "Female"
                
                # Draw box and label
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"{gender} {confidence:.2f}"
                cv2.putText(frame, label, (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        # Show the frame
        cv2.imshow('Gender Detection', frame)
        
        # Break loop with 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Clean up
    cap.release()
    cv2.destroyAllWindows()

def detect_gender_image(image_path):
    # Load YOLOv8 model
    model = YOLO('yolov8n.pt')  # replace with your trained model path
    
    # Read image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image at {image_path}")
        return
    
    # Run detection
    results = model(image)
    
    # Process each detection
    for result in results:
        boxes = result.boxes
        for box in boxes:
            # Get box coordinates
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            # Get confidence and class
            confidence = float(box.conf)
            class_id = int(box.cls)
            
            # Map class to gender (adjust based on your model's classes)
            gender = "Male" if class_id == 0 else "Female"
            
            # Draw box and label
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{gender} {confidence:.2f}"
            cv2.putText(image, label, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    # Show result
    cv2.imshow('Gender Detection', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # Save result
    output_path = 'output_' + os.path.basename(image_path)
    cv2.imwrite(output_path, image)
    print(f"Saved result to {output_path}")

if __name__ == "__main__":
    # Choose mode:
    
    # For webcam:
    detect_gender_live()
    
    # For single image:
    # image_path = "path/to/your/image.jpg"
    # detect_gender_image(image_path)