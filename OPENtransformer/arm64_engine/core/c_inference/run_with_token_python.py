#!/usr/bin/env python3
"""
Script to run image generation with Hugging Face token authentication
"""

import os
import sys
import time
import argparse
from pathlib import Path

# Set Hugging Face token environment variable
os.environ["HF_TOKEN"] = "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"
os.environ["HUGGING_FACE_HUB_TOKEN"] = "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"

# Import the local module
try:
    import py_diffusion_interface
    print("Successfully imported py_diffusion_interface")
except ImportError:
    print("Error importing py_diffusion_interface")
    sys.exit(1)

def main():
    """Run image generation with Hugging Face token authentication."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run image generation with HF token")
    parser.add_argument("--prompt", type=str, default="prime matrix transform",
                        help="Text prompt for image generation")
    parser.add_argument("--model", type=str, default="runwayml/stable-diffusion-v1-5",
                        help="Model to use for generation")
    parser.add_argument("--steps", type=int, default=25,
                        help="Number of inference steps")
    parser.add_argument("--guidance", type=float, default=7.5,
                        help="Guidance scale")
    parser.add_argument("--size", type=int, default=512,
                        help="Output image size")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed (0 for random)")
    args = parser.parse_args()
    
    print(f"Using Hugging Face token: {os.environ.get('HF_TOKEN', '')[:5]}***...")
    print(f"Generating image with prompt: '{args.prompt}'")
    print(f"Using model: {args.model}")
    
    try:
        # Generate the image
        start_time = time.time()
        width, height, channels, img_data = py_diffusion_interface.run_inference(
            args.model,
            args.prompt,
            args.steps,
            args.size,  # width
            args.guidance,
            args.size,  # height (same as width for square image)
            args.seed,
            True,  # use_memory_optimizations
            None,  # callback_ptr
            None   # user_data_ptr
        )
        
        # Save the image
        output_dir = Path("generated_images")
        output_dir.mkdir(exist_ok=True)
        
        # Create a safe filename from the prompt
        safe_prompt = ''.join(c if c.isalnum() or c in [' ', '_'] else '_' for c in args.prompt)
        safe_prompt = safe_prompt.replace(' ', '_')
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_path = output_dir / f"{safe_prompt}_{timestamp}.png"
        
        with open(output_path, "wb") as f:
            f.write(img_data)
        
        total_time = time.time() - start_time
        print(f"Image saved to {output_path}")
        print(f"Generation took {total_time:.2f} seconds")
        
    except Exception as e:
        import traceback
        print(f"Error during image generation: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main() 