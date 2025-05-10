"""
Simple script for image generation using C inference engine
"""

import os
import sys
import time
import platform
import subprocess
import argparse
from pathlib import Path

# Add the C inference directory to the path
project_root = Path(__file__).resolve().parent.parent.parent
c_inference_path = project_root / "OPENtransformer" / "arm64_engine" / "core" / "c_inference"

# Set Hugging Face token
os.environ["HF_TOKEN"] = "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"
os.environ["HUGGING_FACE_HUB_TOKEN"] = "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"

def generate_image(
    prompt, 
    model_path="runwayml/stable-diffusion-v1-5",
    height=512,
    width=512, 
    num_inference_steps=25, 
    guidance_scale=7.5,
    seed=0
):
    """Generate an image using the C inference engine executable."""
    
    print(f"\n=== Generating Image with C Inference Engine ===")
    print(f"Prompt: {prompt}")
    print(f"Model: {model_path}")
    print(f"Size: {width}x{height}, Steps: {num_inference_steps}, Guidance: {guidance_scale}")
    print(f"Using Hugging Face token: {os.environ.get('HF_TOKEN', '')[:5]}***...")
    
    if seed == 0:
        import random
        seed = random.randint(1, 2147483647)
        print(f"Using random seed: {seed}")
    else:
        print(f"Using seed: {seed}")
    
    start_time = time.time()
    
    # Prepare output directory
    output_dir = c_inference_path / "generated_images"
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a safe filename from the prompt
    safe_filename = "".join(c for c in prompt if c.isalnum() or c in " _-").strip()
    safe_filename = safe_filename.replace(" ", "_")
    
    # Define output path
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_filename = f"{safe_filename}_{timestamp}.png"
    output_path = output_dir / output_filename
    
    # Call the C test executable directly
    test_executable = c_inference_path / "test_diffusion"
    
    if not test_executable.exists():
        print(f"Error: Test executable not found at {test_executable}")
        sys.exit(1)
        
    cmd = [
        str(test_executable),
        "--prompt", prompt,
        "--model", model_path,
        "--steps", str(num_inference_steps),
        "--width", str(width),
        "--height", str(height),
        "--guidance", str(guidance_scale),
        "--seed", str(seed),
        "--output", str(output_path)
    ]
    
    print(f"Running command: {' '.join(cmd)}")
    
    # Use the run_test.sh script instead of directly calling the executable
    run_test_script = c_inference_path / "run_test.sh"
    if run_test_script.exists():
        # Format parameters for the shell script
        # Looking at the error, we need to use only supported arguments
        # test_py_direct.py accepts: --prompt, --steps, --guidance, --size, --output, --model
        cmd = [
            str(run_test_script),
            prompt,
            f"--model={model_path}",
            f"--steps={num_inference_steps}",
            f"--guidance={guidance_scale}",
            f"--size={width}"  # Assuming width and height are the same (square image)
        ]
        print(f"Running script: {' '.join(cmd)}")
    
    # Set environment variables for Hugging Face token
    env = os.environ.copy()
    env["HF_TOKEN"] = "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"
    env["HUGGING_FACE_HUB_TOKEN"] = "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"
    
    # Run the command
    try:
        result = subprocess.run(cmd, cwd=str(c_inference_path), check=True, env=env)
    except subprocess.CalledProcessError as e:
        print(f"Error running C inference engine: {e}")
        sys.exit(1)
        
    total_time = time.time() - start_time
    print(f"\nImage generation completed in {total_time:.2f} seconds")
    
    # Look for the most recent image file in the generated_images directory
    image_files = list(output_dir.glob("*.png"))
    if not image_files:
        print("No image files found")
        sys.exit(1)
        
    latest_image = max(image_files, key=os.path.getctime)
    print(f"Image generated: {latest_image}")
    
    # Copy the image to the current directory for easy access
    local_output_path = f"generated_image_{timestamp}.png"
    
    with open(latest_image, "rb") as src:
        with open(local_output_path, "wb") as dst:
            dst.write(src.read())
    
    print(f"Image copied to: {local_output_path}")
    
    # Open the image
    print("\nOpening generated image...")
    if platform.system() == "Darwin":  # macOS
        subprocess.run(["open", local_output_path])
    elif platform.system() == "Windows":
        os.startfile(local_output_path)
    else:  # Linux
        subprocess.run(["xdg-open", local_output_path])

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='C Inference Engine Image Generator')
    parser.add_argument('--prompt', type=str, required=True, help='Text prompt for image generation')
    parser.add_argument('--model', type=str, default="runwayml/stable-diffusion-v1-5", 
                        help='Model ID or path (default: runwayml/stable-diffusion-v1-5)')
    parser.add_argument('--height', type=int, default=512, help='Image height')
    parser.add_argument('--width', type=int, default=512, help='Image width')
    parser.add_argument('--steps', type=int, default=25, help='Number of inference steps')
    parser.add_argument('--guidance', type=float, default=7.5, help='Guidance scale')
    parser.add_argument('--seed', type=int, default=0, help='Random seed (0 for random)')
    return parser.parse_args()

def main():
    args = parse_args()
    
    try:
        generate_image(
            prompt=args.prompt,
            model_path=args.model,
            height=args.height,
            width=args.width,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance,
            seed=args.seed
        )
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 