#!/usr/bin/env python3
"""
Script to generate multiple human portraits with different attributes
"""

import os
import sys
import time
import random
import subprocess
from pathlib import Path

# Get token from environment variable
hf_token = os.environ.get("HF_TOKEN")
if not hf_token:
    print("Warning: HF_TOKEN environment variable not set")
    print("You can set it using 'export HF_TOKEN=your_token_here'")
    exit(1)

def main():
    # Model to use
    model_path = "models/dreamlike-photoreal-2"
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}. Please download it first using download_realistic.py")
        sys.exit(1)
    
    # Create output directory
    output_dir = Path("generated_images/portraits")
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Portrait configurations
    portrait_configs = [
        {
            "gender": "male",
            "age": "young",
            "hair": "dark brown",
            "eyes": "blue",
            "attire": "business suit with blue tie",
            "style": "professional corporate headshot"
        },
        {
            "gender": "female",
            "age": "middle-aged",
            "hair": "blonde",
            "eyes": "green",
            "attire": "formal business attire with blazer",
            "style": "LinkedIn profile photo"
        },
        {
            "gender": "male",
            "age": "elderly",
            "hair": "gray",
            "eyes": "brown",
            "attire": "casual sweater",
            "style": "warm natural lighting"
        },
        {
            "gender": "female",
            "age": "young adult",
            "hair": "red",
            "eyes": "hazel",
            "attire": "casual professional outfit",
            "style": "outdoor natural setting"
        }
    ]
    
    for i, config in enumerate(portrait_configs):
        # Build prompt
        prompt = f"photorealistic portrait of a {config['age']} {config['gender']} with {config['hair']} hair and {config['eyes']} eyes, wearing {config['attire']}, {config['style']}, high quality, detailed facial features, 8k, high detail"
        
        # Output filename
        filename = f"portrait_{i+1}_{config['gender']}_{config['age']}.png"
        output_path = output_dir / filename
        
        print(f"\n=== Generating Portrait {i+1}/{len(portrait_configs)} ===")
        print(f"Prompt: {prompt}")
        
        # Generate image
        start_time = time.time()
        
        # Use Python command to generate the image
        cmd = [
            "./venv/bin/python", 
            "test_py_direct.py",
            "--prompt", prompt,
            "--model", model_path,
            "--steps", "30",
            "--output", str(output_path)
        ]
        
        try:
            result = subprocess.run(cmd, check=True)
            
            # If successful
            generation_time = time.time() - start_time
            print(f"Portrait generated in {generation_time:.2f} seconds: {output_path}")
        except subprocess.CalledProcessError as e:
            print(f"Error generating portrait: {e}")
            continue
    
    print("\nPortrait generation complete!")

if __name__ == "__main__":
    main() 