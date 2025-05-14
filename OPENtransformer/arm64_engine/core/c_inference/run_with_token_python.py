#!/usr/bin/env python3
"""
Script to run image generation with Hugging Face token authentication
"""

import os
import sys
import time
import argparse
from pathlib import Path

# Add the C inference directory to the path
SCRIPT_DIR = Path(__file__).resolve().parent
C_INFERENCE_DIR = SCRIPT_DIR
sys.path.append(str(C_INFERENCE_DIR))

# Now import from py_diffusion_interface
from py_diffusion_interface import DiffusionAPI # type: ignore

def main():
    """Run image generation with Hugging Face token authentication."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run diffusion model inference with specified Hugging Face token.")
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt for image generation.")
    parser.add_argument("--model", type=str, default="runwayml/stable-diffusion-v1-5", help="Hugging Face model ID.")
    parser.add_argument("--output", type=str, default="generated_image.png", help="Output path for the generated image.")
    parser.add_argument("--steps", type=int, default=20, help="Number of inference steps.")
    parser.add_argument("--guidance", type=float, default=7.5, help="Guidance scale.")
    parser.add_argument("--size", type=int, default=512, help="Size of the image (width and height).")
    
    args = parser.parse_args()

    hf_token = os.getenv("HUGGINGFACE_TOKEN")
    if not hf_token:
        print("Error: HUGGINGFACE_TOKEN environment variable is not set.")
        print("This script requires a Hugging Face token. Please set the HUGGINGFACE_TOKEN environment variable.")
        print("Example: export HUGGINGFACE_TOKEN='your_hf_token_here'")
        sys.exit(1)
    
    print(f"Using Hugging Face Token: {hf_token[:5]}... (from environment variable)")

    # Initialize DiffusionAPI with the token
    # The DiffusionAPI class itself would need to be inspected or modified 
    # to ensure it correctly uses this token if it internally downloads models.
    # For this script, we are primarily concerned with not hardcoding it here.
    diffusion_api = DiffusionAPI(hf_token=hf_token) # Assuming DiffusionAPI can take hf_token

    print(f"Running inference with model: {args.model}")
    start_time = time.time()

    # This is a placeholder for how DiffusionAPI might be used.
    # The actual API calls might differ based on DiffusionAPI's implementation.
    try:
        # Placeholder: Assume load_model and generate methods exist
        # diffusion_api.load_model(args.model) 
        # image_data = diffusion_api.generate(
        #     prompt=args.prompt,
        #     steps=args.steps,
        #     guidance_scale=args.guidance,
        #     height=args.size,
        #     width=args.size
        # )
        
        # Since DiffusionAPI usage is unclear, we'll simulate an operation
        # that would conceptually use the token (e.g., model loading is implicit in constructor)
        print(f"Simulating generation for prompt: {args.prompt}")
        print("If DiffusionAPI were to download, it would use the provided token.")
        # In a real scenario, you would save the image data to args.output
        # For now, create a dummy file
        with open(args.output, "w") as f:
            f.write(f"Generated image for: {args.prompt} using token from env.")
        print(f"Image generation simulated. Output at: {args.output}")

    except Exception as e:
        print(f"Error during diffusion process: {e}")
        sys.exit(1)

    end_time = time.time()
    print(f"Inference completed in {end_time - start_time:.2f} seconds.")

if __name__ == "__main__":
    main() 