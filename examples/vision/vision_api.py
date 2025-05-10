#!/usr/bin/env python3
"""
Vision API for real-time webcam transformer processing with SIMD optimization
"""

import os
import sys
import cv2
import torch
import numpy as np
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Union
from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image

# Add the project root to Python path
project_root = str(Path(__file__).parent.parent.parent.absolute())
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import SIMD kernels
from OPENtransformer.arm64_engine.core.asm.kernels.vision.vision_kernels_asm import VisionKernelsASM, build_vit_kernels
from OPENtransformer.arm64_engine.core.asm.kernels.vision.vision_transformer_simd import VisionTransformerSIMD

class VisionAPI:
    """API for vision processing with SIMD-optimized transformer models"""
    
    def __init__(self, model_name: str = "microsoft/resnet-50"):
        """
        Initialize the vision API
        
        Args:
            model_name: Name of the model to use
        """
        self.model_name = model_name
        self.processor = None
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize SIMD components
        try:
            # Build SIMD kernels
            build_vit_kernels()
            
            # Initialize Vision Transformer with SIMD
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
        
    def load_model(self):
        """Load the vision model and processor"""
        try:
            print(f"Loading model: {self.model_name}")
            self.processor = AutoImageProcessor.from_pretrained(self.model_name)
            self.model = AutoModelForImageClassification.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()
            print("Model loaded successfully")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
            
    def preprocess_frame(self, frame: np.ndarray) -> Tuple[Image.Image, Optional[np.ndarray]]:
        """
        Preprocess frame using SIMD optimization if available
        
        Args:
            frame: Input frame as numpy array
            
        Returns:
            Tuple of (PIL Image, Optional features)
        """
        if self.simd_kernels is not None and self.vision_model is not None:
            try:
                # Convert BGR to RGB using SIMD
                rgb_frame = self.simd_kernels.bgr_to_rgb(frame)
                
                # Convert to float32 and scale to [0, 1]
                rgb_frame = rgb_frame.astype(np.float32) / 255.0
                
                # Normalize using SIMD
                normalized = self.simd_kernels.normalize_image(rgb_frame)
                
                # Get visual features using Vision Transformer
                features = self.vision_model.extract_features(normalized)
                
                # Convert back to uint8 for model input
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
            
    def process_frame(self, frame: np.ndarray) -> Dict:
        """
        Process a single frame with SIMD optimization
        
        Args:
            frame: Input frame as numpy array
            
        Returns:
            Dictionary containing processing results
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
            
        # Preprocess frame with SIMD if available
        pil_image, features = self.preprocess_frame(frame)
        
        # Process with model
        inputs = self.processor(images=pil_image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Run inference
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = torch.nn.functional.softmax(logits, dim=-1)
            
        # Get top predictions
        top_probs, top_indices = torch.topk(probs[0], 5)
        
        # Convert to dictionary
        results = {
            "predictions": [
                {
                    "label": self.model.config.id2label[idx.item()],
                    "probability": prob.item()
                }
                for prob, idx in zip(top_probs, top_indices)
            ]
        }
        
        # Add SIMD features if available
        if features is not None:
            results["simd_features"] = features.tolist()
        
        return results
        
    def process_webcam(self, 
                      camera_id: int = 0,
                      window_name: str = "Vision Transformer",
                      display: bool = True,
                      max_fps: int = 30) -> None:
        """
        Process webcam feed in real-time with SIMD optimization
        
        Args:
            camera_id: Camera device ID
            window_name: Name of the display window
            display: Whether to display the feed
            max_fps: Maximum frames per second
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
            
        # Open webcam
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open camera {camera_id}")
            
        # Set FPS
        cap.set(cv2.CAP_PROP_FPS, max_fps)
        
        try:
            while True:
                # Read frame
                ret, frame = cap.read()
                if not ret:
                    print("Failed to grab frame")
                    break
                    
                # Process frame with SIMD
                results = self.process_frame(frame)
                
                # Display results
                if display:
                    # Draw predictions on frame
                    for i, pred in enumerate(results["predictions"]):
                        text = f"{pred['label']}: {pred['probability']:.2f}"
                        cv2.putText(frame, text, (10, 30 + i*30),
                                  cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    
                    # Show frame
                    cv2.imshow(window_name, frame)
                    
                    # Check for quit
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                        
        finally:
            cap.release()
            cv2.destroyAllWindows()
            
    def save_frame(self, frame: np.ndarray, output_path: str) -> None:
        """
        Save a processed frame
        
        Args:
            frame: Frame to save
            output_path: Path to save the frame
        """
        cv2.imwrite(output_path, frame)
        print(f"Frame saved to {output_path}")

def main():
    """Example usage of the Vision API with SIMD optimization"""
    # Initialize API
    api = VisionAPI()
    
    try:
        # Load model
        api.load_model()
        
        # Process webcam feed
        print("\nStarting webcam processing with SIMD optimization...")
        print("Press 'q' to quit")
        api.process_webcam()
        
    except KeyboardInterrupt:
        print("\nProcessing stopped by user")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 