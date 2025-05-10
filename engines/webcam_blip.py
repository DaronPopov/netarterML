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

# Add project root to Python path
project_root = str(Path(__file__).absolute().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import SIMD kernels
from OPENtransformer.arm64_engine.core.asm.kernels.vision.vision_kernels_asm import VisionKernelsASM, build_vit_kernels
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
        try:
            # Build SIMD kernels first
            build_vit_kernels()
            
            # Initialize Vision Transformer
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
            print("SIMD kernels initialized successfully!")
        except Exception as e:
            print(f"Warning: Failed to initialize SIMD kernels: {e}")
            print("Falling back to standard image processing...")
            self.vision_model = None
            self.simd_kernels = None
        
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
        """Preprocess frame for model input"""
        if self.simd_kernels is not None:
            try:
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
            except Exception as e:
                print(f"Warning: SIMD processing failed: {e}")
                print("Falling back to standard image processing...")
                self.simd_kernels = None
                self.vision_model = None
        
        # Fallback to standard processing
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_frame)
        return pil_image, None
    
    def generate_caption(self, frame):
        """Generate caption for the current frame"""
        # Preprocess frame
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
            device_id: Camera device ID (default: 0)
        """
        cap = cv2.VideoCapture(device_id)
        if not cap.isOpened():
            print(f"Error: Could not open camera {device_id}")
            return
        
        print("\nStarting webcam feed...")
        print("Press 'q' to quit")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame")
                break
            
            # Update frame count and FPS
            self.frame_count += 1
            elapsed_time = time.time() - self.start_time
            if elapsed_time >= 1.0:
                self.fps_display = self.frame_count / elapsed_time
                self.frame_count = 0
                self.start_time = time.time()
            
            # Add FPS to frame
            cv2.putText(frame, f"FPS: {self.fps_display:.1f}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Add caption to frame
            if self.last_caption:
                cv2.putText(frame, self.last_caption, (10, frame.shape[0] - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Display frame
            cv2.imshow('Webcam BLIP', frame)
            
            # Add frame to caption queue
            if self.caption_queue.empty():
                self.caption_queue.put(frame.copy())
            
            # Check for quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Cleanup
        self.running = False
        cap.release()
        cv2.destroyAllWindows()
        cleanup_memory()

def run_webcam_blip(model_name: str, hf_token: str):
    """
    Run the webcam BLIP engine with the specified model.
    
    Args:
        model_name: Name of the BLIP model to use
        hf_token: HuggingFace token for authentication
    """
    if not validate_model_and_token(model_name, hf_token):
        return
    
    if not check_and_download_model(model_name, hf_token):
        return
    
    engine = WebcamBlipEngine(model_name, hf_token)
    engine.run()

def main():
    # ===== CONFIGURE YOUR MODEL HERE =====
    MODEL_NAME = "Salesforce/blip-image-captioning-base"
    HF_TOKEN = os.environ.get('HUGGINGFACE_TOKEN')
    
    if not HF_TOKEN:
        print("Error: HUGGINGFACE_TOKEN environment variable not set")
        sys.exit(1)
    
    run_webcam_blip(MODEL_NAME, HF_TOKEN)

if __name__ == "__main__":
    main() 