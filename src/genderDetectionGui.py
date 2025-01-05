import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import PIL.Image, PIL.ImageTk
from insightface.app import FaceAnalysis
import threading
from datetime import datetime
import os
import sys

class GenderDetectorApp:
    def __init__(self, window):
        self.window = window
        self.window.title("Gender Detection App")
        self.window.geometry("1200x800")
        
        # Initialize face analyzer
        try:
            self.app = FaceAnalysis(providers=['CPUExecutionProvider'])
            self.app.prepare(ctx_id=0, det_size=(640, 640))
        except Exception as e:
            messagebox.showerror("Error", f"Failed to initialize face analyzer: {str(e)}")
            sys.exit(1)
        
        # Video capture variables
        self.cap = None
        self.is_webcam_on = False
        self.current_image = None
        self.camera_index = 0
        
        self.create_gui()
        
    def create_gui(self):
        # Create main container
        main_container = ttk.Frame(self.window, padding="10")
        main_container.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Create video frame
        self.video_frame = ttk.Label(main_container)
        self.video_frame.grid(row=0, column=0, columnspan=2, padx=5, pady=5)
        
        # Create buttons frame
        button_frame = ttk.Frame(main_container)
        button_frame.grid(row=1, column=0, columnspan=2, pady=10)
        
        # Add camera selection
        ttk.Label(button_frame, text="Camera:").grid(row=0, column=0, padx=5)
        self.camera_var = tk.StringVar(value="0")
        camera_entry = ttk.Entry(button_frame, textvariable=self.camera_var, width=5)
        camera_entry.grid(row=0, column=1, padx=5)
        
        # Create buttons
        self.webcam_button = ttk.Button(button_frame, 
                                      text="Start Webcam", 
                                      command=self.toggle_webcam)
        self.webcam_button.grid(row=0, column=2, padx=5)
        
        self.image_button = ttk.Button(button_frame, 
                                     text="Load Image", 
                                     command=self.load_image)
        self.image_button.grid(row=0, column=3, padx=5)
        
        self.screenshot_button = ttk.Button(button_frame, 
                                          text="Take Screenshot", 
                                          command=self.take_screenshot)
        self.screenshot_button.grid(row=0, column=4, padx=5)
        
        # Create stats frame
        stats_frame = ttk.LabelFrame(main_container, text="Statistics", padding="10")
        stats_frame.grid(row=0, column=2, rowspan=2, padx=10, sticky=(tk.N, tk.S))
        
        self.stats_text = tk.Text(stats_frame, width=30, height=20)
        self.stats_text.pack(fill=tk.BOTH, expand=True)
        
        # Create output directory
        os.makedirs("screenshots", exist_ok=True)
        
    def toggle_webcam(self):
        if not self.is_webcam_on:
            try:
                camera_idx = int(self.camera_var.get())
                self.cap = cv2.VideoCapture(camera_idx)
                
                if not self.cap.isOpened():
                    raise ValueError(f"Could not open camera {camera_idx}")
                
                self.is_webcam_on = True
                self.webcam_button.config(text="Stop Webcam")
                self.update_webcam()
            except Exception as e:
                messagebox.showerror("Camera Error", 
                    f"Failed to start camera {camera_idx}.\nError: {str(e)}\n"
                    "Try a different camera index (0, 1, 2, etc.)")
                self.stop_webcam()
        else:
            self.stop_webcam()
            
    def stop_webcam(self):
        if self.cap is not None:
            self.cap.release()
        self.is_webcam_on = False
        self.webcam_button.config(text="Start Webcam")
        self.video_frame.config(image='')
        
    def update_webcam(self):
        if self.is_webcam_on:
            ret, frame = self.cap.read()
            if ret:
                self.current_image = frame.copy()
                frame = self.process_frame(frame)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = PIL.Image.fromarray(frame)
                
                # Resize to fit window while maintaining aspect ratio
                display_size = (800, 600)
                image.thumbnail(display_size, PIL.Image.LANCZOS)
                
                photo = PIL.ImageTk.PhotoImage(image=image)
                self.video_frame.config(image=photo)
                self.video_frame.image = photo
                
            self.window.after(10, self.update_webcam)
            
    def process_frame(self, frame):
        faces = self.app.get(frame)
        stats = {"male": 0, "female": 0}
        
        for face in faces:
            bbox = face.bbox.astype(int)
            gender = face.gender
            age = face.age
            
            if gender == 0:
                stats["female"] += 1
            else:
                stats["male"] += 1
            
            color = (255, 192, 203) if gender == 0 else (255, 0, 0)
            
            cv2.rectangle(frame, 
                         (bbox[0], bbox[1]), 
                         (bbox[2], bbox[3]), 
                         color, 2)
            
            gender_text = "Female" if gender == 0 else "Male"
            label = f"{gender_text} ({age:.0f}y)"
            
            (text_width, text_height), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(frame, 
                         (bbox[0], bbox[1]-30), 
                         (bbox[0] + text_width, bbox[1]), 
                         color, -1)
            
            cv2.putText(frame, label, 
                       (bbox[0], bbox[1]-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 
                       0.7, (255, 255, 255), 2)
        
        self.update_stats(stats)
        return frame
    
    def update_stats(self, stats):
        self.stats_text.delete(1.0, tk.END)
        self.stats_text.insert(tk.END, "Current Detection:\n\n")
        self.stats_text.insert(tk.END, f"Males detected: {stats['male']}\n")
        self.stats_text.insert(tk.END, f"Females detected: {stats['female']}\n")
        self.stats_text.insert(tk.END, f"Total faces: {stats['male'] + stats['female']}\n")
        
    def load_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.gif *.tiff")])
        
        if file_path:
            self.stop_webcam()
            image = cv2.imread(file_path)
            self.current_image = image.copy()
            
            if image is not None:
                processed_image = self.process_frame(image)
                processed_image = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)
                image = PIL.Image.fromarray(processed_image)
                
                display_size = (800, 600)
                image.thumbnail(display_size, PIL.Image.LANCZOS)
                
                photo = PIL.ImageTk.PhotoImage(image=image)
                self.video_frame.config(image=photo)
                self.video_frame.image = photo
                
    def take_screenshot(self):
        if self.current_image is not None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"screenshots/detection_{timestamp}.jpg"
            
            processed_image = self.process_frame(self.current_image.copy())
            cv2.imwrite(filename, processed_image)
            
            self.stats_text.insert(tk.END, f"\nScreenshot saved: {filename}\n")

def main():
    root = tk.Tk()  # Create root window first
    try:
        app = GenderDetectorApp(root)
        root.mainloop()
    except Exception as e:
        messagebox.showerror("Error", str(e))
        root.destroy()

if __name__ == "__main__":
    main()