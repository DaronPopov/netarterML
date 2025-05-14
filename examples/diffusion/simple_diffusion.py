#!/usr/bin/env python3
"""
Simple diffusion image generation demo using the diffusion part of the multimodal pipeline
"""

import os
import sys
import torch
from pathlib import Path
import argparse
from diffusers import DiffusionPipeline

# Add the project root to Python path
project_root = str(Path(__file__).parent.parent.parent.absolute())
if project_root not in sys.path:
    sys.path.insert(0, project_root)

def generate_image(prompt, output_path, model_name="Heartsync/NSFW-Uncensored", num_steps=10):
    """
    Generate an image using Stable Diffusion
    
    Args:
        prompt: Text prompt for image generation
        output_path: Path to save the generated image
        model_name: Name of the diffusion model to use
        num_steps: Number of denoising steps (default: 10)
    """
    try:
        print(f"Loading model: {model_name}")
        pipe = DiffusionPipeline.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            use_safetensors=True,
            variant="fp16" if torch.cuda.is_available() else None,
            requires_safety_checker=False
        )
        
        if torch.cuda.is_available():
            pipe = pipe.to("cuda")
        print("Model loaded successfully!")

        print(f"Generating image for prompt: {prompt}")
        image = pipe(prompt, num_inference_steps=num_steps).images[0]
        image.save(output_path)
        print(f"Image saved to {output_path}")
        return True
    except Exception as e:
        print(f"Error generating image: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Simple diffusion image generation")
    parser.add_argument("--prompt", required=True, help="Text prompt for image generation")
    parser.add_argument("--output", default="generated_images/simple_diffusion.png", help="Output path for the generated image")
    parser.add_argument("--model", default="Heartsync/NSFW-Uncensored", help="Diffusion model to use")
    parser.add_argument("--steps", type=int, default=10, help="Number of denoising steps (default: 10)")
    args = parser.parse_args()

    # Ensure output directory exists
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    generate_image(args.prompt, args.output, args.model, args.steps)

if __name__ == "__main__":
    main() 