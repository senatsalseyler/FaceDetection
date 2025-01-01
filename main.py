import os
from src.detector import FaceDetector
from src.utils import load_image, create_directories, download_weights
from src.visualization import draw_faces

def main():
    # Create necessary directories
    create_directories()
    
    # Download weights if needed
    weights_path = download_weights()
    
    # Initialize detector
    detector = FaceDetector(weights_path)
    
    # Process single image
    image_path = os.path.join('data', 'test_images', 'test.jpg')
    image = load_image(image_path)
    
    # Detect faces
    faces = detector.detect_faces(image)
    
    # Draw results
    result_image = draw_faces(image, faces)
    
    # Display results
    cv2.imshow('Face Detection Result', result_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()