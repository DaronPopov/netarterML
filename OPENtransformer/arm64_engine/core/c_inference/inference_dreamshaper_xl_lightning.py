#!/usr/bin/env python3
"""
Streamlined inference script for Dreamshaper XL Lightning
Uses existing model without attempting to download or convert
"""

import os
import sys
import time
import random
import argparse
from pathlib import Path
from easy_diffusion_api import EasyDiffusionAPI

def run_inference(prompt, 
                  output_path="generated_images/output.png", 
                  steps=6, 
                  width=728, 
                  height=728, 
                  guidance=7.0, 
                  seed=None):
    """
    Run inference with existing Dreamshaper XL Lightning model
    Does not attempt to download or reconvert the model
    """
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Create API instance without downloading
    api = EasyDiffusionAPI()
    print(f"Using resolution: {width}x{height}, steps: {steps}")
    
    # Check if model exists
    model_id = "dreamshaper-xl-lightning"
    model_path = os.path.join("models", model_id)
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        print("Please run download_dreamshaper_xl_lightning.py first")
        return False
    
    # Just set the active model - the models should already be registered in config
    print(f"Setting active model to: {model_id}")
    if not api.set_active_model(model_id):
        print(f"Error: Could not set active model to {model_id}")
        print("Available models:", [m['id'] for m in api.list_models()])
        return False
    
    # Generate a random seed if None is provided
    if seed is None:
        seed = random.randint(1, 1000000)
        print(f"Using random seed: {seed}")
    
    # Run inference
    print(f"Generating image for prompt: {prompt}")
    start_time = time.time()
    
    result = api.generate_image(
        prompt=prompt,
        steps=steps,
        width=width,
        height=height,
        guidance=guidance,
        seed=seed,
        output_path=output_path
    )
    
    elapsed = time.time() - start_time
    
    if result:
        print(f"Generation successful! Image saved to: {output_path}")
        print(f"Generation took {elapsed:.2f} seconds")
        return True
    else:
        print("Generation failed")
        return False

def main():
    parser = argparse.ArgumentParser(description="Fast inference with Dreamshaper XL Lightning model")
    parser.add_argument("--prompt", type=str, default="a detailed portrait of a cyberpunk character with neon lights, ultra realistic, cinematic lighting, 8k", 
                        help="Prompt to generate")
    parser.add_argument("--output", type=str, default="generated_images/dreamshaper_xl_lightning_out.png",
                        help="Output image path")
    parser.add_argument("--steps", type=int, default=6,
                        help="Number of inference steps (default: 6)")
    parser.add_argument("--width", type=int, default=728,
                        help="Image width (default: 728)")
    parser.add_argument("--height", type=int, default=728,
                        help="Image height (default: 728)")
    parser.add_argument("--guidance", type=float, default=7.0,
                        help="Guidance scale (default: 7.0)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed (default: random)")
    
    args = parser.parse_args()
    
    run_inference(
        prompt=args.prompt,
        output_path=args.output,
        steps=args.steps,
        width=args.width,
        height=args.height,
        guidance=args.guidance,
        seed=args.seed
    )

if __name__ == "__main__":
    main() 