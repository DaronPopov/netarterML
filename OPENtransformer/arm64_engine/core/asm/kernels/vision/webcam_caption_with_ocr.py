#!/usr/bin/env python3
import os
import sys
from pathlib import Path
from huggingface_hub import login

# Add the project root to Python path
project_root = str(Path(__file__).parent.parent.parent.parent.parent.parent.absolute())
sys.path.insert(0, project_root)

import cv2
import torch
import numpy as np
import time
import threading
import gc
import warnings
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
from OPENtransformer.arm64_engine.core.asm.kernels.vision.vision_kernels_asm import VisionKernelsASM
from OPENtransformer.arm64_engine.core.asm.kernels.vision.vision_transformer_simd import VisionTransformerSIMD

def cleanup_memory():
    """Clean up memory and cache"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

class ImageCaptioningAndOCREngine:
    def __init__(self, offline_mode=False):
        # Check for Hugging Face token
        hf_token = os.getenv('HF_API_TOKEN') or os.getenv('HUGGINGFACE_TOKEN')
        if not hf_token and not offline_mode:
            print("Error: Hugging Face API token is required. Please set HF_API_TOKEN or HUGGINGFACE_TOKEN environment variable.")
            sys.exit(1)
        
        # Login to Hugging Face if token is available
        if hf_token:
            try:
                login(token=hf_token)
            except Exception as e:
                print(f"Warning: Could not login to Hugging Face: {e}")
                print("Continuing in offline mode...")
                offline_mode = True
        
        # Initialize BLIP model and processor for captioning and text recognition
        print("\nLoading BLIP model...")
        if offline_mode:
            local_model_path = Path("models/vision/blip-image-captioning-base")
            if local_model_path.exists():
                print(f"   • Using local model from: {local_model_path}")
                self.processor = BlipProcessor.from_pretrained(str(local_model_path))
                self.model = BlipForConditionalGeneration.from_pretrained(str(local_model_path))
            else:
                print("   • Error: No local model found in models/vision/blip-image-captioning-base")
                print("   • Please ensure the model is downloaded and available locally")
                sys.exit(1)
        else:
            self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
            self.model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
        
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
        
        # Move model to GPU if available
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
        self.model.eval()
        
        print(f"Model loaded successfully! Using device: {self.device}")
    
    def preprocess_image(self, frame):
        """Preprocess image using SIMD-optimized kernels"""
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
        
        return normalized, features
    
    def generate_caption(self, frame):
        """Generate image caption with visual features"""
        # Preprocess image
        processed, features = self.preprocess_image(frame)
        
        # Convert to PIL Image
        pil_image = Image.fromarray(processed, mode='RGB')
        
        # Process image with BLIP
        inputs = self.processor(images=pil_image, return_tensors="pt").to(self.device)
        
        # Generate caption using the BLIP model
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_length=50
            )
            caption = self.processor.decode(generated_ids[0], skip_special_tokens=True)
        
        return caption
    
    def extract_text(self, frame):
        """Extract text from the image using BLIP model with SIMD preprocessing"""
        # Preprocess image
        processed, _ = self.preprocess_image(frame)
        
        # Convert to PIL Image
        pil_image = Image.fromarray(processed, mode='RGB')
        
        # Process image with BLIP
        inputs = self.processor(images=pil_image, return_tensors="pt").to(self.device)
        
        # Generate text using BLIP model
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_length=50,
                num_beams=5,
                length_penalty=1.0
            )
            text = self.processor.decode(generated_ids[0], skip_special_tokens=True)
        
        return text

def main():
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Webcam Captioning with OCR')
    parser.add_argument('--device', type=int, default=0, help='Webcam device index')
    parser.add_argument('--display-time', type=float, default=0.0, help='Time to display each frame')
    parser.add_argument('--offline', action='store_true', help='Run in offline mode using local models')
    args = parser.parse_args()
    
    # Initialize captioning and OCR engine
    engine = ImageCaptioningAndOCREngine(offline_mode=args.offline)
    
    # Initialize webcam with reduced resolution
    cap = cv2.VideoCapture(args.device)
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
    
    # Initialize variables for performance optimization
    frame_count = 0
    start_time = time.time()
    fps_display = 0
    last_caption = ""
    last_text = ""
    caption_thread = None
    text_thread = None
    last_processed_frame = None
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Calculate FPS
            frame_count += 1
            if frame_count % 30 == 0:
                end_time = time.time()
                fps_display = 30 / (end_time - start_time)
                start_time = time.time()
            
            # Display FPS
            cv2.putText(frame, f"FPS: {fps_display:.1f}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Process frame in separate threads
            if caption_thread is None or not caption_thread.is_alive():
                caption_thread = threading.Thread(target=lambda: setattr(engine, 'last_caption', engine.generate_caption(frame)))
                caption_thread.start()
            
            if text_thread is None or not text_thread.is_alive():
                text_thread = threading.Thread(target=lambda: setattr(engine, 'last_text', engine.extract_text(frame)))
                text_thread.start()
            
            # Display results
            if hasattr(engine, 'last_caption'):
                cv2.putText(frame, f"Caption: {engine.last_caption}", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            if hasattr(engine, 'last_text'):
                cv2.putText(frame, f"Text: {engine.last_text}", (10, 90),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Display frame
            cv2.imshow('Webcam Captioning with OCR', frame)
            
            # Wait for key press
            key = cv2.waitKey(1)
            if key == ord('q'):
                break
            
            # Clean up memory periodically
            if frame_count % 100 == 0:
                cleanup_memory()
    
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 