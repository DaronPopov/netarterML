#!/usr/bin/env python3
"""
Multimodal pipeline combining image generation and captioning
Shows step-by-step generation with real-time captions
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
from diffusers import StableDiffusionPipeline
import time

# Add the project root to Python path
project_root = str(Path(__file__).parent.parent.parent.absolute())
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import the Vision API from the vision directory
sys.path.insert(0, str(Path(__file__).parent.parent / "vision"))
from vision_api import VisionAPI

class MultimodalPipeline:
    """Pipeline combining image generation and captioning"""
    
    def __init__(self, 
                 diffusion_model: str = "runwayml/stable-diffusion-v1-5",
                 caption_model: str = "Salesforce/blip-image-captioning-base"):
        """
        Initialize the multimodal pipeline
        
        Args:
            diffusion_model: Name of the diffusion model to use
            caption_model: Name of the BLIP model to use
        """
        # Initialize diffusion model
        print("\nLoading diffusion model...")
        self.diffusion_pipe = StableDiffusionPipeline.from_pretrained(
            diffusion_model,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )
        if torch.cuda.is_available():
            self.diffusion_pipe = self.diffusion_pipe.to("cuda")
        print("Diffusion model loaded successfully!")
        
        # Initialize captioning model
        print("\nLoading captioning model...")
        self.caption_processor = BlipProcessor.from_pretrained(caption_model)
        self.caption_model = BlipForConditionalGeneration.from_pretrained(caption_model)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.caption_model = self.caption_model.to(self.device)
        self.caption_model.eval()
        print("Captioning model loaded successfully!")
        
        # Initialize Vision API for SIMD processing
        self.vision_api = VisionAPI()
        
        # Create output directory
        self.output_dir = "generated_images"
        os.makedirs(self.output_dir, exist_ok=True)
        
    def generate_step(self, prompt: str, step: int, total_steps: int = 5) -> Image.Image:
        """
        Generate an image for a specific step
        
        Args:
            prompt: Text prompt for generation
            step: Current step number
            total_steps: Total number of steps
            
        Returns:
            Generated image
        """
        # Calculate steps for this generation
        steps = int(50 * (step / total_steps))  # Scale steps based on progress
        
        # Generate image
        image = self.diffusion_pipe(
            prompt,
            num_inference_steps=steps,
            guidance_scale=7.5
        ).images[0]
        
        return image
        
    def caption_image(self, image: Image.Image) -> str:
        """
        Generate caption for an image
        
        Args:
            image: Input image
            
        Returns:
            Generated caption
        """
        # Convert to numpy for SIMD processing
        frame = np.array(image)
        
        # Preprocess with SIMD if available
        pil_image, _ = self.vision_api.preprocess_frame(frame)
        
        # Generate caption
        inputs = self.caption_processor(images=pil_image, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.caption_model.generate(**inputs, max_length=50)
            caption = self.caption_processor.decode(outputs[0], skip_special_tokens=True)
            
        return caption
        
    def run_pipeline(self, prompt: str, total_steps: int = 5):
        """
        Run the complete multimodal pipeline
        
        Args:
            prompt: Text prompt for generation
            total_steps: Total number of generation steps
        """
        print(f"\nStarting multimodal pipeline with prompt: {prompt}")
        print(f"Total steps: {total_steps}")
        
        # Create window for display
        window_name = "Multimodal Pipeline"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        
        try:
            for step in range(1, total_steps + 1):
                print(f"\nStep {step}/{total_steps}")
                
                # Generate image for this step
                start_time = time.time()
                image = self.generate_step(prompt, step, total_steps)
                gen_time = time.time() - start_time
                
                # Generate caption
                start_time = time.time()
                caption = self.caption_image(image)
                caption_time = time.time() - start_time
                
                # Save image
                timestamp = int(time.time())
                image_path = os.path.join(self.output_dir, f"step_{step}_{timestamp}.png")
                image.save(image_path)
                
                # Convert to OpenCV format for display
                frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                
                # Add text overlay
                cv2.putText(frame, f"Step {step}/{total_steps}", (10, 30),
                          cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame, f"Generation: {gen_time:.1f}s", (10, 60),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"Caption: {caption_time:.1f}s", (10, 90),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Split caption into multiple lines if needed
                words = caption.split()
                lines = []
                current_line = []
                for word in words:
                    current_line.append(word)
                    if len(' '.join(current_line)) > 40:  # Max characters per line
                        lines.append(' '.join(current_line[:-1]))
                        current_line = [word]
                if current_line:
                    lines.append(' '.join(current_line))
                
                # Display caption
                for i, line in enumerate(lines):
                    cv2.putText(frame, line, (10, 120 + i*30),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Show frame
                cv2.imshow(window_name, frame)
                
                # Wait for key press
                key = cv2.waitKey(0) & 0xFF
                if key == ord('q'):
                    break
                    
        finally:
            cv2.destroyAllWindows()

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Multimodal pipeline")
    parser.add_argument("--prompt", type=str, 
                      default="a beautiful sunset over mountains, digital art",
                      help="Text prompt for image generation")
    parser.add_argument("--steps", type=int, default=5,
                      help="Number of generation steps")
    parser.add_argument("--diffusion-model", type=str,
                      default="runwayml/stable-diffusion-v1-5",
                      help="Diffusion model to use")
    parser.add_argument("--caption-model", type=str,
                      default="Salesforce/blip-image-captioning-base",
                      help="Captioning model to use")
    return parser.parse_args()

def main():
    """Main function"""
    # Parse arguments
    args = parse_args()
    
    # Initialize pipeline
    pipeline = MultimodalPipeline(
        diffusion_model=args.diffusion_model,
        caption_model=args.caption_model
    )
    
    try:
        # Run pipeline
        pipeline.run_pipeline(args.prompt, args.steps)
        
    except KeyboardInterrupt:
        print("\nProcessing stopped by user")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 