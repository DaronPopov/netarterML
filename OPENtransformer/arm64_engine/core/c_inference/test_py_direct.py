#!/usr/bin/env python3
"""
Direct test of the Python diffusion interface without C integration.
"""

import os
import sys
import time
import argparse
from PIL import Image
import numpy as np

# Add parent directory to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../..")))

# Import the EasyImage module
from OPENtransformer.easy_image import EasyImage

def main():
    """Test image generation directly using Python."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Test image generation using Python")
    parser.add_argument("--prompt", type=str, default="a futuristic city with flying cars",
                        help="Text prompt for image generation")
    parser.add_argument("--steps", type=int, default=5,
                        help="Number of inference steps")
    parser.add_argument("--guidance", type=float, default=7.5,
                        help="Guidance scale")
    parser.add_argument("--size", type=int, default=512,
                        help="Output image size")
    parser.add_argument("--output", type=str, default="py_direct_output.png",
                        help="Output image filename")
    parser.add_argument("--model", type=str, default="stabilityai/stable-diffusion-2-1",
                        help="Model to use for generation")
    args = parser.parse_args()
    
    print("Testing direct Python image generation")
    
    # Create image generator
    model_name = args.model
    print(f"Using model: {model_name}")
    
    generator = EasyImage(
        model_name=model_name,
        model_type="stable-diffusion",
        offline_mode=True
    )
    
    # Setup the generator
    print("Setting up image generator...")
    if not generator.setup():
        print("Failed to set up image generator")
        return
    
    # Set parameters
    generator.num_inference_steps = args.steps
    generator.guidance_scale = args.guidance
    generator.image_size = args.size
    
    # Generate an image
    prompt = args.prompt
    print(f"Generating image with prompt: '{prompt}'")
    
    start_time = time.time()
    images = generator.generate(prompt)
    generation_time = time.time() - start_time
    
    if not images:
        print("No images were generated")
        return
    
    # Save the image
    output_path = args.output
    images[0].save(output_path)
    print(f"Image saved to {output_path}")
    print(f"Generation took {generation_time:.2f} seconds")

if __name__ == "__main__":
    main() 