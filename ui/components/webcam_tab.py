import tkinter as tk
from tkinter import ttk
import cv2
from PIL import Image, ImageTk
import threading
import queue
import time

class WebcamTab:
    def __init__(self, parent, on_model_load):
        self.parent = parent
        self.on_model_load = on_model_load
        self.webcam_engine = None
        self.webcam_running = False
        self.frame_queue = queue.Queue(maxsize=1)
        self.caption_queue = queue.Queue(maxsize=1)
        self.webcam_width = 640
        self.webcam_height = 480
        
        # Create frame
        self.frame = ttk.LabelFrame(parent, text="Webcam Feed")
        self.frame.pack(expand=True, fill='both', padx=10, pady=5)
        
        # Model selection frame
        self.setup_model_frame()
        
        # Status label
        self.webcam_status_label = ttk.Label(self.frame, text="Status: No model loaded")
        self.webcam_status_label.pack(fill='x', padx=5, pady=5)
        
        # Create video display
        self.setup_video_display()
        
        # Control buttons
        self.setup_control_buttons()
    
    def setup_model_frame(self):
        model_frame = ttk.Frame(self.frame)
        model_frame.pack(fill='x', padx=5, pady=5)
        
        ttk.Label(model_frame, text="Model:").pack(side='left', padx=5)
        
        # Add dropdown for cached models
        self.webcam_model_dropdown = ttk.Combobox(model_frame, values=[])
        self.webcam_model_dropdown.pack(side='left', expand=True, fill='x', padx=5)
        self.webcam_model_dropdown.bind('<<ComboboxSelected>>', 
            lambda e: self.webcam_model_input.delete(0, tk.END) or 
                     self.webcam_model_input.insert(0, self.webcam_model_dropdown.get()))
        
        # Add search entry
        ttk.Label(model_frame, text="Search:").pack(side='left', padx=5)
        self.webcam_model_input = ttk.Entry(model_frame)
        self.webcam_model_input.insert(0, "Salesforce/blip-image-captioning-base")
        self.webcam_model_input.pack(side='left', expand=True, fill='x', padx=5)
        
        self.load_webcam_model_button = ttk.Button(model_frame, text="Load Model", 
                                                 command=self.load_specified_webcam_model)
        self.load_webcam_model_button.pack(side='right', padx=5)
    
    def setup_video_display(self):
        # Create a canvas for the video display
        self.video_canvas = tk.Canvas(self.frame, width=self.webcam_width, height=self.webcam_height)
        self.video_canvas.pack(expand=True, fill='both', padx=5, pady=5)
        
        # Create a label for the video
        self.video_label = ttk.Label(self.video_canvas)
        self.video_label.place(relx=0.5, rely=0.5, anchor='center')
        
        # Create a label for captions
        self.caption_label = ttk.Label(
            self.video_canvas,
            text="",
            wraplength=self.webcam_width - 20,
            background='black',
            foreground='white',
            font=('Arial', 12)
        )
        self.caption_label.place(relx=0.5, rely=0.95, anchor='s')
    
    def setup_control_buttons(self):
        button_frame = ttk.Frame(self.frame)
        button_frame.pack(fill='x', padx=5, pady=5)
        
        self.start_button = ttk.Button(button_frame, text="Start Webcam", command=self.start_webcam)
        self.start_button.pack(side='left', padx=5)
        
        self.stop_button = ttk.Button(button_frame, text="Stop Webcam", command=self.stop_webcam, state='disabled')
        self.stop_button.pack(side='left', padx=5)
        
        # Initially disable webcam buttons
        self.start_button.config(state='disabled')
    
    def load_specified_webcam_model(self):
        """Load the webcam model specified in the input field."""
        model_name = self.webcam_model_input.get().strip()
        if not model_name:
            self.show_error("Please enter a model name.")
            return
        
        # Update status
        self.webcam_status_label.config(text=f"Status: Loading {model_name}...")
        self.load_webcam_model_button.config(state='disabled')
        self.webcam_model_input.config(state='disabled')
        self.webcam_model_dropdown.config(state='disabled')
        
        # Call the parent's model loading function
        self.on_model_load(model_name, self.on_model_loaded)
    
    def on_model_loaded(self, success, error=None):
        """Callback for when model loading is complete"""
        if success:
            self.webcam_status_label.config(text=f"Status: Model loaded successfully")
            self.start_button.config(state='normal')
        else:
            self.webcam_status_label.config(text="Status: Error loading model")
            self.show_error(f"Error loading webcam model: {error}")
        
        self.load_webcam_model_button.config(state='normal')
        self.webcam_model_input.config(state='normal')
        self.webcam_model_dropdown.config(state='normal')
    
    def start_webcam(self):
        if self.webcam_engine is None:
            self.show_error("Please load a model first")
            return
        
        self.webcam_running = True
        self.start_button.config(state='disabled')
        self.stop_button.config(state='normal')
        self.load_webcam_model_button.config(state='disabled')
        self.webcam_model_input.config(state='disabled')
        
        # Start webcam thread
        threading.Thread(target=self.webcam_thread, daemon=True).start()
        # Start caption thread
        threading.Thread(target=self.caption_thread, daemon=True).start()
    
    def stop_webcam(self):
        self.webcam_running = False
        self.start_button.config(state='normal')
        self.stop_button.config(state='disabled')
        self.load_webcam_model_button.config(state='normal')
        self.webcam_model_input.config(state='normal')
    
    def webcam_thread(self):
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.webcam_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.webcam_height)
        
        while self.webcam_running:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert frame to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Update frame queue
            if self.frame_queue.empty():
                self.frame_queue.put(rgb_frame)
            
            # Resize frame to fit canvas
            image = Image.fromarray(rgb_frame)
            photo = ImageTk.PhotoImage(image=image)
            
            # Update video display
            def update_ui():
                self.video_label.config(image=photo)
                self.video_label.image = photo
            self.parent.after(0, update_ui)
            
            time.sleep(0.03)  # ~30 FPS
        
        cap.release()
    
    def caption_thread(self):
        while self.webcam_running:
            if not self.frame_queue.empty():
                frame = self.frame_queue.get()
                try:
                    caption = self.webcam_engine.generate_caption(frame)
                    # Update caption
                    def update_caption():
                        self.caption_label.config(text=caption)
                    self.parent.after(0, update_caption)
                except Exception as e:
                    self.show_error(f"Error generating caption: {str(e)}")
                finally:
                    self.frame_queue.task_done()
            time.sleep(0.1)
    
    def show_error(self, message):
        """Show error message"""
        from tkinter import messagebox
        messagebox.showerror("Error", message)
    
    def update_model_dropdown(self, models):
        """Update the model dropdown with available models"""
        self.webcam_model_dropdown['values'] = models
        if models:
            self.webcam_model_dropdown.set(models[0])
            self.webcam_model_input.delete(0, tk.END)
            self.webcam_model_input.insert(0, models[0])
    
    def cleanup(self):
        """Clean up resources"""
        self.webcam_running = False
        if self.webcam_engine:
            self.webcam_engine = None 