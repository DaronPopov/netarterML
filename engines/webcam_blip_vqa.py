#!/usr/bin/env python3
import os
import sys
from pathlib import Path
import torch
import cv2
import numpy as np
from PIL import Image
from transformers import ViltProcessor, ViltForQuestionAnswering
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
    """
    if not model_name or not isinstance(model_name, str):
        print("Error: Invalid model name provided")
        return False
        
    if not hf_token or not isinstance(hf_token, str):
        print("Error: Invalid HuggingFace token provided")
        return False
        
    if not hf_token.startswith('hf_'):
        print(f"Error: Invalid HuggingFace token format. Token should start with 'hf_'. Got: {hf_token[:10]}...")
        return False
        
    if len(hf_token) < 20 or len(hf_token) > 100:
        print(f"Error: Token length seems incorrect. Expected 20-100 characters, got {len(hf_token)}")
        return False
        
    return True

def check_and_download_model(model_name: str, hf_token: str) -> bool:
    """
    Check if model exists locally, if not download it.
    """
    try:
        print(f"\nChecking model: {model_name}")
        processor = ViltProcessor.from_pretrained(model_name, token=hf_token)
        model = ViltForQuestionAnswering.from_pretrained(model_name, token=hf_token)
        print(f"Model {model_name} is available locally.")
        return True
    except Exception as e:
        print(f"\nModel {model_name} not found locally. Attempting to download...")
        try:
            processor = ViltProcessor.from_pretrained(model_name, token=hf_token)
            model = ViltForQuestionAnswering.from_pretrained(model_name, token=hf_token)
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

class WebcamViltEngine:
    def __init__(self, model_name: str, hf_token: str):
        """
        Initialize the webcam ViLT engine.
        """
        print("\nLoading ViLT model...")
        self.processor = ViltProcessor.from_pretrained(model_name, token=hf_token)
        self.model = ViltForQuestionAnswering.from_pretrained(model_name, token=hf_token)
        
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
        
        # Initialize variables
        self.frame_count = 0
        self.start_time = time.time()
        self.fps_display = 0
        self.last_answer = ""
        self.current_question = "What do you see in this image?"
        self.answer_queue = Queue(maxsize=1)
        self.running = True
        
        # Start answer generation thread
        self.answer_thread = threading.Thread(target=self._answer_generation_loop)
        self.answer_thread.daemon = True
        self.answer_thread.start()
    
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
                
                # Convert back to uint8 for ViLT
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
    
    def generate_answer(self, frame, question):
        """Generate answer for the current frame and question"""
        # Preprocess frame
        pil_image, features = self.preprocess_frame(frame)
        
        # Process image and question with ViLT
        encoding = self.processor(pil_image, question, return_tensors="pt").to(self.device)
        
        # Generate answer
        with torch.no_grad():
            outputs = self.model(**encoding)
            logits = outputs.logits
            idx = logits.argmax(-1).item()
            answer = self.model.config.id2label[idx]
        
        return answer
    
    def _answer_generation_loop(self):
        """Background thread for continuous answer generation"""
        while self.running:
            if not self.answer_queue.empty():
                frame, question = self.answer_queue.get()
                try:
                    answer = self.generate_answer(frame, question)
                    self.last_answer = answer
                except Exception as e:
                    print(f"Error generating answer: {e}")
                finally:
                    self.answer_queue.task_done()
            time.sleep(0.01)
    
    def run(self, device_id: int = 0):
        """
        Run the webcam ViLT engine.
        """
        cap = cv2.VideoCapture(device_id)
        if not cap.isOpened():
            print(f"Error: Could not open camera {device_id}")
            return
        
        print("\nStarting webcam feed...")
        print("Press 'n' to enter a new question")
        print("Press 'q' to quit")
        
        # Start a separate thread for terminal input
        def terminal_input_thread():
            while self.running:
                if input("\nPress Enter to ask a new question (or 'q' to quit): ").lower() == 'q':
                    self.running = False
                    break
                new_question = input("Enter your question: ").strip()
                if new_question:
                    self.current_question = new_question
                    print(f"New question set: {self.current_question}")
        
        # Start the terminal input thread
        input_thread = threading.Thread(target=terminal_input_thread)
        input_thread.daemon = True
        input_thread.start()
        
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
            
            # Create a copy of the frame for drawing
            display_frame = frame.copy()
            
            # Add FPS to frame
            cv2.putText(display_frame, f"FPS: {self.fps_display:.1f}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Add question and answer to frame
            cv2.putText(display_frame, f"Q: {self.current_question}", (10, frame.shape[0] - 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(display_frame, f"A: {self.last_answer}", (10, frame.shape[0] - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Display frame
            cv2.imshow('Webcam ViLT VQA', display_frame)
            
            # Add frame to answer queue
            if self.answer_queue.empty():
                self.answer_queue.put((frame.copy(), self.current_question))
            
            # Check for quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Cleanup
        self.running = False
        cap.release()
        cv2.destroyAllWindows()
        cleanup_memory()

def run_webcam_vilt(model_name: str, hf_token: str):
    """
    Run the webcam ViLT engine with the specified model.
    """
    if not validate_model_and_token(model_name, hf_token):
        return
    
    if not check_and_download_model(model_name, hf_token):
        return
    
    engine = WebcamViltEngine(model_name, hf_token)
    engine.run()

def main():
    # ===== CONFIGURE YOUR MODEL HERE =====
    MODEL_NAME = "dandelin/vilt-b32-finetuned-vqa"
    HF_TOKEN = os.environ.get("HUGGINGFACE_TOKEN")
    if not HF_TOKEN:
        print("Error: HUGGINGFACE_TOKEN environment variable not set")
        sys.exit(1)
    
    run_webcam_vilt(MODEL_NAME, HF_TOKEN)

if __name__ == "__main__":
    main() 