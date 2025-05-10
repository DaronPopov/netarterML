#!/usr/bin/env python3
"""
Real-time webcam captioning using BLIP model and SIMD optimization
"""

import os
import sys
import cv2
import torch
import numpy as np
from pathlib import Path
import argparse
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

# Add the project root to Python path
project_root = str(Path(__file__).parent.parent.parent.absolute())
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import the Vision API
from vision_api import VisionAPI

class WebcamCaptioner:
    """Real-time webcam captioning with BLIP model"""
    
    def __init__(self, model_name: str = "Salesforce/blip-image-captioning-base"):
        """
        Initialize the webcam captioner
        
        Args:
            model_name: Name of the BLIP model to use
        """
        # Initialize BLIP model and processor
        print("\nLoading BLIP model...")
        self.processor = BlipProcessor.from_pretrained(model_name)
        self.model = BlipForConditionalGeneration.from_pretrained(model_name)
        
        # Move to GPU if available
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
        self.model.eval()
        print(f"Model loaded successfully! Using device: {self.device}")
        
        # Initialize Vision API for SIMD processing
        self.vision_api = VisionAPI()
        
        # Performance tracking
        self.frame_count = 0
        self.start_time = None
        self.fps = 0
        
    def process_frame(self, frame: np.ndarray) -> str:
        """
        Process a single frame and generate caption
        
        Args:
            frame: Input frame as numpy array
            
        Returns:
            Generated caption
        """
        # Preprocess frame with SIMD if available
        pil_image, _ = self.vision_api.preprocess_frame(frame)
        
        # Generate caption
        inputs = self.processor(images=pil_image, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_length=50)
            caption = self.processor.decode(outputs[0], skip_special_tokens=True)
            
        return caption
        
    def run(self, camera_id: int = 0, window_name: str = "Webcam Captioning"):
        """
        Run real-time webcam captioning
        
        Args:
            camera_id: Camera device ID
            window_name: Name of the display window
        """
        # Open webcam
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open camera {camera_id}")
            
        # Set FPS
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        # Initialize performance tracking
        self.start_time = cv2.getTickCount()
        
        try:
            while True:
                # Read frame
                ret, frame = cap.read()
                if not ret:
                    print("Failed to grab frame")
                    break
                    
                # Process frame and generate caption
                caption = self.process_frame(frame)
                
                # Update FPS
                self.frame_count += 1
                if self.frame_count % 30 == 0:
                    end_time = cv2.getTickCount()
                    self.fps = 30 * cv2.getTickFrequency() / (end_time - self.start_time)
                    self.start_time = end_time
                
                # Draw caption and FPS
                cv2.putText(frame, caption, (10, 30),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"FPS: {self.fps:.1f}", (10, 60),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Show frame
                cv2.imshow(window_name, frame)
                
                # Check for quit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
        finally:
            cap.release()
            cv2.destroyAllWindows()

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Real-time webcam captioning")
    parser.add_argument("--model", type=str, 
                      default="Salesforce/blip-image-captioning-base",
                      help="BLIP model to use")
    parser.add_argument("--camera", type=int, default=0,
                      help="Camera device ID")
    return parser.parse_args()

def main():
    """Main function"""
    # Parse arguments
    args = parse_args()
    
    # Initialize captioner
    captioner = WebcamCaptioner(model_name=args.model)
    
    try:
        # Run webcam captioning
        print("\nStarting webcam captioning...")
        print("Press 'q' to quit")
        captioner.run(camera_id=args.camera)
        
    except KeyboardInterrupt:
        print("\nProcessing stopped by user")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 