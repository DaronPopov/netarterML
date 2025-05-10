"""
Medical image analysis engine for the AI Studio application.
"""

import os
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification

class MedicalImageEngine:
    def __init__(self, model_name, hf_token=None):
        """Initialize the medical image analysis engine.
        
        Args:
            model_name (str): The model name to use
            hf_token (str, optional): HuggingFace token for authentication
        """
        self.model_name = model_name
        self.hf_token = hf_token
        self.processor = None
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    def load_model(self):
        """Load the model and processor."""
        try:
            self.processor = AutoImageProcessor.from_pretrained(
                self.model_name,
                token=self.hf_token,
                trust_remote_code=True
            )
            self.model = AutoModelForImageClassification.from_pretrained(
                self.model_name,
                token=self.hf_token,
                trust_remote_code=True,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
            ).to(self.device)
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def analyze_image(self, image_path):
        """Analyze a medical image.
        
        Args:
            image_path (str): Path to the image file
            
        Returns:
            tuple: (PIL.Image, str) The processed image and analysis results
        """
        if not self.model or not self.processor:
            raise RuntimeError("Model not loaded")
        
        try:
            # Load and preprocess image
            image = Image.open(image_path).convert('RGB')
            inputs = self.processor(images=image, return_tensors="pt").to(self.device)
            
            # Get predictions
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probs = torch.nn.functional.softmax(logits, dim=-1)
                pred_idx = torch.argmax(probs).item()
                confidence = probs[0][pred_idx].item()
            
            # Get label
            label = self.model.config.id2label[pred_idx]
            
            # Format results
            results = f"Analysis Results:\n"
            results += f"Prediction: {label}\n"
            results += f"Confidence: {confidence:.2%}\n"
            
            return image, results
            
        except Exception as e:
            print(f"Error analyzing image: {e}")
            return None, f"Error analyzing image: {str(e)}"
    
    def cleanup(self):
        """Clean up resources."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache() 