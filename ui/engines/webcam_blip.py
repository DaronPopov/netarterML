#!/usr/bin/env python3
import os
import sys
from pathlib import Path
import torch
import cv2
import numpy as np
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from datetime import datetime
import time
import threading
import gc
from queue import Queue
from OPENtransformer.arm64_engine.core.asm.kernels.vision.vision_kernels_asm import VisionKernelsASM
from OPENtransformer.arm64_engine.core.asm.kernels.vision.vision_transformer_simd import VisionTransformerSIMD

def validate_model_and_token(model_name: str, hf_token: str) -> bool:
    """
    Validate the model name and token before attempting to load.
    
    Args:
        model_name: Name of the model to validate
        hf_token: HuggingFace token to validate
        
    Returns:
        bool: True if validation passes, False otherwise
    """
    if not model_name or not isinstance(model_name, str):
        print("Error: Invalid model name provided")
        return False
        
    if not hf_token or not isinstance(hf_token, str):
        print("Error: Invalid HuggingFace token provided")
        return False
        
    # Check if token starts with 'hf_'
    if not hf_token.startswith('hf_'):
        print(f"Error: Invalid HuggingFace token format. Token should start with 'hf_'. Got: {hf_token[:10]}...")
        return False
        
    # Check token length
    if len(hf_token) < 20 or len(hf_token) > 100:
        print(f"Error: Token length seems incorrect. Expected 20-100 characters, got {len(hf_token)}")
        return False
        
    return True

def check_and_download_model(model_name: str, hf_token: str) -> bool:
    """
    Check if model exists locally, if not download it.
    
    Args:
        model_name: Name of the model to check/download
        hf_token: HuggingFace token for authentication
        
    Returns:
        bool: True if model is available (either locally or downloaded), False otherwise
    """
    try:
        print(f"\nChecking model: {model_name}")
        
        # Try to load the model
        processor = BlipProcessor.from_pretrained(model_name, token=hf_token)
        model = BlipForConditionalGeneration.from_pretrained(model_name, token=hf_token)
        
        print(f"Model {model_name} is available locally.")
        return True
        
    except Exception as e:
        print(f"\nModel {model_name} not found locally. Attempting to download...")
        try:
            # Try downloading the model
            processor = BlipProcessor.from_pretrained(model_name, token=hf_token)
            model = BlipForConditionalGeneration.from_pretrained(model_name, token=hf_token)
            print(f"Successfully downloaded model: {model_name}")
            return True
        except Exception as download_error:
            print(f"\nError downloading model: {str(download_error)}")
            print("Please check:")
            print("1. The model name is correct")
            print("2. Your HuggingFace token is valid")
            print("3. You have sufficient disk space")
            print("4. You have a stable internet connection")
            return False

def cleanup_memory():
    """Clean up memory and cache"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

class WebcamBlipEngine:
    def __init__(self, model_name: str, hf_token: str):
        """
        Initialize the webcam BLIP engine.
        
        Args:
            model_name: Name of the BLIP model to use
            hf_token: HuggingFace token for authentication
        """
        # Initialize BLIP model and processor
        print("\nLoading BLIP model...")
        self.processor = BlipProcessor.from_pretrained(model_name, token=hf_token)
        self.model = BlipForConditionalGeneration.from_pretrained(model_name, token=hf_token)
        
        # Initialize Vision Transformer with SIMD
        print("Initializing Vision Transformer with SIMD...")
        self.vision_model = VisionTransformerSIMD(
            image_size=224,
            patch_size=16,
            num_channels=3,
            embed_dim=768,
            num_heads=12,
            num_layers=12,
            num_classes=1000
        )
        
        # Initialize SIMD kernels
        self.simd_kernels = VisionKernelsASM()
        
        # Move to GPU if available
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
        self.model.eval()
        print(f"Model loaded successfully! Using device: {self.device}")
        
        # Initialize variables for performance optimization
        self.frame_count = 0
        self.start_time = time.time()
        self.fps_display = 0
        self.last_caption = ""
        self.caption_queue = Queue(maxsize=1)
        self.running = True
        
        # Start caption generation thread
        self.caption_thread = threading.Thread(target=self._caption_generation_loop)
        self.caption_thread.daemon = True
        self.caption_thread.start()
    
    def preprocess_frame(self, frame):
        """Preprocess frame for model input using SIMD-optimized kernels"""
        # Convert BGR to RGB using SIMD
        rgb_frame = self.simd_kernels.bgr_to_rgb(frame)
        
        # Convert to float32 and scale to [0, 1]
        rgb_frame = rgb_frame.astype(np.float32) / 255.0
        
        # Normalize using SIMD
        normalized = self.simd_kernels.normalize_image(rgb_frame)
        
        # Get visual features using Vision Transformer
        features = self.vision_model.extract_features(normalized)
        
        # Convert back to uint8 for BLIP
        normalized = (normalized * 255).clip(0, 255).astype(np.uint8)
        
        # Convert to PIL Image
        pil_image = Image.fromarray(normalized, mode='RGB')
        
        return pil_image, features
    
    def generate_caption(self, frame):
        """Generate caption for the current frame"""
        # Preprocess frame with SIMD
        pil_image, features = self.preprocess_frame(frame)
        
        # Process image with BLIP
        inputs = self.processor(images=pil_image, return_tensors="pt").to(self.device)
        
        # Generate caption
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_length=50,
                num_beams=5,
                length_penalty=1.0
            )
            caption = self.processor.decode(generated_ids[0], skip_special_tokens=True)
        
        return caption
    
    def _caption_generation_loop(self):
        """Background thread for continuous caption generation"""
        while self.running:
            if not self.caption_queue.empty():
                frame = self.caption_queue.get()
                try:
                    caption = self.generate_caption(frame)
                    self.last_caption = caption
                except Exception as e:
                    print(f"Error generating caption: {e}")
                finally:
                    self.caption_queue.task_done()
            time.sleep(0.01)  # Small sleep to prevent CPU overload
    
    def run(self, device_id: int = 0):
        """
        Run the webcam BLIP engine.
        
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
                
                # Update caption queue with new frame
                if self.caption_queue.empty():
                    self.caption_queue.put(frame.copy())
                
                # Display current caption
                if self.last_caption:
                    # Split caption into multiple lines if too long
                    words = self.last_caption.split()
                    lines = []
                    current_line = ""
                    for word in words:
                        if len(current_line + " " + word) < 40:
                            current_line += " " + word
                        else:
                            lines.append(current_line.strip())
                            current_line = word
                    if current_line:
                        lines.append(current_line.strip())
                    
                    # Display each line
                    for i, line in enumerate(lines):
                        cv2.putText(frame, line, (10, 60 + i*30),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Display frame
                cv2.imshow('Webcam BLIP Captioning', frame)
                
                # Wait for key press
                key = cv2.waitKey(1)
                if key == ord('q'):
                    break
                
                # Clean up memory periodically
                if self.frame_count % 100 == 0:
                    cleanup_memory()
        
        finally:
            self.running = False
            self.caption_thread.join()
            cap.release()
            cv2.destroyAllWindows()

def run_webcam_blip(model_name: str, hf_token: str):
    """
    Run the webcam BLIP engine with the given model and token.
    
    Args:
        model_name: Name of the model to use
        hf_token: HuggingFace token for authentication
    """
    # Validate model name and token first
    if not validate_model_and_token(model_name, hf_token):
        print("\nValidation failed. Please check your model name and token.")
        return
    
    # Check and download model if needed
    if not check_and_download_model(model_name, hf_token):
        print("\nFailed to load or download the model. Please check the error messages above.")
        return
    
    try:
        engine = WebcamBlipEngine(model_name, hf_token)
        engine.run()
    except Exception as e:
        print(f"\nError: {str(e)}")
        sys.exit(1)

def main():
    # ===== CONFIGURE YOUR MODEL HERE =====
    MODEL_NAME = "Salesforce/blip-image-captioning-base"  # Change this to your desired model
    HF_TOKEN = "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"      # Replace with your token
    # =====================================
    
    run_webcam_blip(MODEL_NAME, HF_TOKEN)

if __name__ == "__main__":
    main() 