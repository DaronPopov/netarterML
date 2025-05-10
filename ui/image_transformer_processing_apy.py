from transformers import AutoImageProcessor, AutoModelForImageClassification, AutoTokenizer
import os
import sys
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from OPENtransformer.arm64_engine.core.asm.kernels.vision import vision_kernels_asm

# Add Hugging Face token
os.environ["HUGGING_FACE_HUB_TOKEN"] = "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"

# Add the project root to Python path
project_root = str(Path(__file__).parent.parent.absolute())
if project_root not in sys.path:
    sys.path.insert(0, project_root)

class SIMDVisionProcessor:
    def __init__(self):
        self.simd_kernels = vision_kernels_asm.VisionKernelsASM()
        
    def preprocess_with_simd(self, image_array):
        """Apply SIMD-optimized preprocessing"""
        try:
            # Convert to float32 and scale to [0, 1]
            image_array = image_array.astype(np.float32) / 255.0
            
            # Apply SIMD-optimized normalization
            normalized_image = self.simd_kernels.normalize_image(image_array)
            
            # Convert back to uint8 for PIL
            normalized_image = (normalized_image * 255).clip(0, 255).astype(np.uint8)
            return normalized_image
        except Exception as e:
            print(f"Error in SIMD preprocessing: {e}")
            return image_array

class MedicalImageAnalyzer:
    def __init__(self):
        self.setup_inference()
        
    def setup_inference(self):
        """Setup offline inference system with SIMD backend"""
        try:
            # Initialize SIMD processor
            self.simd_processor = SIMDVisionProcessor()
            
            # Initialize device
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
            print(f"\n📊 System Information:")
            print(f"   • System: SIMD-Optimized Medical Inference")
            print(f"   • Device: {self.device}")
            print(f"   • Backend: ARM64 SIMD ASM")
            
            # Initialize model and processor
            self.initialize_model()
            
            print("SIMD backend system initialized successfully")
            
        except Exception as e:
            print(f"Error initializing backend system: {e}")
            raise

    def initialize_model(self):
        """Initialize medical imaging model"""
        try:
            # Use pneumonia X-ray specific model
            model_name = "pawlo2013/vit-pneumonia-x-ray_3_class"
            
            # Check for local model
            local_model_path = Path("medical_imaging_project/models/medical/vit-pneumonia-x-ray")
            if local_model_path.exists():
                print(f"   • Using local model from: {local_model_path}")
                self.processor = AutoImageProcessor.from_pretrained(str(local_model_path))
                self.model = AutoModelForImageClassification.from_pretrained(str(local_model_path))
            else:
                print(f"   • Local model not found, downloading from Hugging Face...")
                print(f"   • Model: {model_name}")
                print(f"   • Task: Pneumonia X-ray Classification")
                print(f"   • Classes: No Pneumonia, Bacterial, Viral")
                
                # Create model directory
                local_model_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Download and save model
                self.processor = AutoImageProcessor.from_pretrained(model_name)
                self.model = AutoModelForImageClassification.from_pretrained(model_name)
                
                # Save model locally
                print(f"   • Saving model to: {local_model_path}")
                self.processor.save_pretrained(str(local_model_path))
                self.model.save_pretrained(str(local_model_path))
            
            # Move model to device and set to evaluation mode
            self.model.to(self.device)
            self.model.eval()
            
        except Exception as e:
            print(f"\n❌ Error loading model: {str(e)}")
            raise

    def preprocess_image(self, image_input):
        """Preprocess image using SIMD-optimized pipeline"""
        try:
            # Handle different input types
            if isinstance(image_input, str):
                image = Image.open(image_input)
            elif isinstance(image_input, Image.Image):
                image = image_input
            else:
                raise ValueError("Input must be either a file path or a PIL Image object")
            
            # Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Convert to numpy array for SIMD processing
            image_array = np.array(image)
            
            # Apply SIMD-optimized preprocessing
            processed_image = self.simd_processor.preprocess_with_simd(image_array)
            
            # Convert back to PIL Image for the transformer model
            processed_pil = Image.fromarray(processed_image.astype('uint8'))
            
            # Process with the transformer's processor
            inputs = self.processor(
                images=processed_pil,
                return_tensors="pt",
                do_resize=True,
                size=224,  # ViT model uses 224x224 input size
                resample=Image.BICUBIC,
                do_normalize=True
            )
            
            # Move inputs to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            return inputs, image
            
        except Exception as e:
            print(f"Error preprocessing image: {e}")
            raise

    def analyze_image(self, image_path):
        """Analyze image using the SIMD-optimized backend"""
        try:
            # Preprocess with SIMD optimizations
            inputs, original_image = self.preprocess_image(image_path)
            
            # Run inference
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probabilities = torch.nn.functional.softmax(logits, dim=-1)
            
            # Get predictions
            predictions = {
                self.model.config.id2label[i]: float(prob)
                for i, prob in enumerate(probabilities[0])
            }
            
            # Get the highest probability prediction
            max_pred = max(predictions.items(), key=lambda x: x[1])
            
            return {
                'prediction': max_pred[0],
                'confidence': max_pred[1],
                'probabilities': list(predictions.values()),
                'class_names': list(predictions.keys())
            }
            
        except Exception as e:
            print(f"Error analyzing image: {e}")
            return None
            
    def __del__(self):
        """Cleanup resources"""
        if hasattr(self, 'model'):
            del self.model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

def main():
    try:
        # Initialize the medical image analyzer
        analyzer = MedicalImageAnalyzer()
        
        # List of X-ray images to analyze
        image_paths = [
            "local_images/normal_1.jpg",
            "local_images/normal_2.jpg",
            "local_images/normal_3.jpg",
            "local_images/normal_4.jpg",
            "local_images/normal_5.jpg",
            "local_images/pneumonia_1.jpg",
            "local_images/pneumonia_2.jpg"
        ]
        
        # Analyze each image
        for image_path in image_paths:
            print(f"\n🔍 Analyzing X-ray image: {image_path}")
            results = analyzer.analyze_image(image_path)
            
            if results:
                print("\n📊 Analysis Results:")
                print(f"   • Prediction: {results['prediction']}")
                print(f"   • Confidence: {results['confidence']:.2%}")
                print("\n   Probability Distribution:")
                for class_name, prob in zip(results['class_names'], results['probabilities']):
                    print(f"   • {class_name}: {prob:.2%}")
            else:
                print(f"Failed to analyze the image: {image_path}")
            
    except Exception as e:
        print(f"Error running analysis: {e}")
        raise

if __name__ == "__main__":
    main() 