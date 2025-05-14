#!/usr/bin/env python3
import os
import sys
from pathlib import Path
import torch
import cv2
import numpy as np
from PIL import Image
from diffusers import StableDiffusionPipeline
from datetime import datetime
import time
import threading
import gc

def cleanup_memory():
    """Clean up memory and cache"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

class WebcamDetectionEngine:
    def __init__(self, model_name: str, hf_token: str, output_dir: str = "generated_images"):
        """
        Initialize the webcam detection engine.
        
        Args:
            model_name: Name of the model to use
            hf_token: HuggingFace token for authentication
            output_dir: Directory to save generated images
        """
        # Validate model name and token
        if not model_name or not isinstance(model_name, str):
            raise ValueError("Invalid model name provided")
            
        if not hf_token or not isinstance(hf_token, str):
            raise ValueError("Invalid HuggingFace token provided")
            
        # Strip any whitespace from the token
        hf_token = hf_token.strip()
            
        # Check if token starts with 'hf_'
        if not hf_token.startswith('hf_'):
            raise ValueError(f"Invalid HuggingFace token format. Token should start with 'hf_'. Got: {hf_token[:10]}...")
            
        # Check token length
        if len(hf_token) < 20 or len(hf_token) > 100:
            raise ValueError(f"Token length seems incorrect. Expected 20-100 characters, got {len(hf_token)}")
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        self.output_dir = output_dir
        
        # Initialize pipeline
        print(f"\nLoading {model_name}... This may take a moment.")
        self.pipeline = StableDiffusionPipeline.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            use_safetensors=True,
            token=hf_token
        )
        
        # Move to GPU if available
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.pipeline = self.pipeline.to(self.device)
        print(f"Model loaded successfully! Using device: {self.device}")
        
        # Initialize variables for performance optimization
        self.frame_count = 0
        self.start_time = time.time()
        self.fps_display = 0
        self.last_generation = None
        self.generation_thread = None
    
    def preprocess_frame(self, frame):
        """Preprocess frame for model input"""
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Convert to PIL Image
        pil_image = Image.fromarray(rgb_frame)
        
        return pil_image
    
    def generate_from_frame(self, frame):
        """Generate image based on the current frame"""
        # Preprocess frame
        pil_image = self.preprocess_frame(frame)
        
        # Generate image
        with torch.no_grad():
            image = self.pipeline(pil_image).images[0]
        
        # Save image
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.output_dir}/generated_{timestamp}.png"
        image.save(filename)
        
        return filename
    
    def run(self, device_id: int = 0):
        """
        Run the webcam detection engine.
        
        Args:
            device_id: Webcam device index
        """
        # Initialize webcam
        cap = cv2.VideoCapture(device_id)
        if not cap.isOpened():
            print("Error: Could not open webcam")
            return
        
        # Set lower resolution for better performance
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        # Get webcam properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        print(f"\nWebcam initialized: {width}x{height} @ {fps:.1f} FPS")
        print("\nPress 'q' to quit")
        print("Press 'g' to generate an image from the current frame")
        print("="*50)
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Calculate FPS
                self.frame_count += 1
                if self.frame_count % 30 == 0:
                    end_time = time.time()
                    self.fps_display = 30 / (end_time - self.start_time)
                    self.start_time = time.time()
                
                # Display FPS
                cv2.putText(frame, f"FPS: {self.fps_display:.1f}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Display last generation info
                if self.last_generation:
                    cv2.putText(frame, f"Last saved: {self.last_generation}", (10, 60),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Display frame
                cv2.imshow('Webcam Detection', frame)
                
                # Wait for key press
                key = cv2.waitKey(1)
                if key == ord('q'):
                    break
                elif key == ord('g'):
                    # Generate image in a separate thread
                    if self.generation_thread is None or not self.generation_thread.is_alive():
                        self.generation_thread = threading.Thread(
                            target=lambda: setattr(self, 'last_generation', self.generate_from_frame(frame))
                        )
                        self.generation_thread.start()
                
                # Clean up memory periodically
                if self.frame_count % 100 == 0:
                    cleanup_memory()
        
        finally:
            cap.release()
            cv2.destroyAllWindows()

def main():
    # ===== CONFIGURE YOUR MODEL HERE =====
    MODEL_NAME = "runwayml/stable-diffusion-v1-5"  # Change this to your desired model
    HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
    if not HF_TOKEN:
        print("Warning: HUGGINGFACE_TOKEN environment variable not set. Model downloads may fail.")
    # =====================================
    
    try:
        engine = WebcamDetectionEngine(MODEL_NAME, HF_TOKEN)
        engine.run()
    except Exception as e:
        print(f"\nError: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 