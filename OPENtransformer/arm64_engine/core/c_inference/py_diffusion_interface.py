#!/usr/bin/env python3
"""
Python interface for C-based diffusion inference

This module provides a clean interface between the C wrapper and the Python
diffusion implementation, handling all necessary conversions and execution.
"""

import os
import sys
import time
import numpy as np
from PIL import Image
import io
import torch
import random

# Import diffusers
try:
    from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline
    print("Successfully imported diffusers")
except ImportError as e:
    print(f"Error importing diffusers: {e}")
    print("Please install with: pip install diffusers torch")
    sys.exit(1)

# Try to import c_callback module (which is created by the C code)
try:
    import c_callback
    print("Successfully imported c_callback module")
except ImportError:
    # Create a dummy callback for testing within Python
    print("Creating dummy c_callback module")
    class DummyCallback:
        @staticmethod
        def c_progress_callback(step, total_steps, step_time, callback_ptr=None, user_data_ptr=None):
            print(f"Step {step}/{total_steps}, time: {step_time:.3f}s")
    
    c_callback = DummyCallback()

# Global model cache to avoid reloading models
_model_cache = {}

def is_sdxl_model(model_path):
    """
    Check if the model is an SDXL model based on its name/path
    """
    sdxl_identifiers = [
        "sdxl", 
        "stable-diffusion-xl", 
        "sd-xl", 
        "dreamshaper-xl",
        "juggernaut-xl"
    ]
    
    model_name = model_path.lower()
    return any(identifier in model_name for identifier in sdxl_identifiers)

def run_inference(
    model_path,
    prompt,
    num_inference_steps,
    width,
    guidance_scale,
    height,
    seed,
    use_memory_optimizations,
    callback_ptr,
    user_data_ptr
):
    """
    Run diffusion inference with the given parameters
    
    Args:
        model_path: Path to the model or model ID
        prompt: Text prompt for image generation
        num_inference_steps: Number of inference steps
        width: Output image width
        guidance_scale: Guidance scale
        height: Output image height
        seed: Random seed
        use_memory_optimizations: Whether to use memory optimizations
        callback_ptr: Pointer to C callback function
        user_data_ptr: User data pointer for the callback
        
    Returns:
        Tuple of (width, height, channels, image_data)
    """
    
    # Set random seed if specified
    if seed != 0:
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.cuda.manual_seed_all(seed)
        print(f"Set random seed to {seed}")

    # Create progress callback
    def progress_callback(step, total_steps, step_time):
        c_callback.c_progress_callback(
            step, total_steps, step_time, callback_ptr, user_data_ptr)
    
    print(f"Python interface: Generating image with prompt: '{prompt}'")
    print(f"Using model: {model_path}")
    print(f"Steps: {num_inference_steps}, Guidance: {guidance_scale}, Size: {width}x{height}")
    
    try:
        # Check if the model is SDXL
        is_xl_model = is_sdxl_model(model_path)
        model_key = f"{model_path}_{is_xl_model}"
        
        # Get or create model instance
        if model_key in _model_cache:
            print("Using cached model")
            pipe = _model_cache[model_key]
        else:
            print(f"Loading model {model_path}")
            # Choose the appropriate pipeline based on model type
            if is_xl_model:
                print("Using SDXL pipeline for this model")
                pipe = StableDiffusionXLPipeline.from_pretrained(
                    model_path,
                    torch_dtype=torch.float32,
                    use_safetensors=True
                )
            else:
                print("Using standard Stable Diffusion pipeline")
                pipe = StableDiffusionPipeline.from_pretrained(
                    model_path,
                    torch_dtype=torch.float32,
                    use_safetensors=True
                )
            
            pipe = pipe.to("cpu")
            _model_cache[model_key] = pipe
        
        # Create a callback for diffusers
        def pipe_callback(step, timestep, latents):
            # Calculate step time (approximate)
            step_time = 0.1  # Placeholder
            # Report progress
            progress_callback(step, num_inference_steps, step_time)
            return True
            
        # Generate the image
        start_time = time.time()
        
        # For SDXL models, we need to set different parameters
        if is_xl_model:
            negative_prompt = "low quality, bad quality, blurry, pixelated"
            image = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                height=height,
                width=width,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                callback=pipe_callback,
                callback_steps=1
            ).images[0]
        else:
            image = pipe(
                prompt=prompt,
                height=height,
                width=width,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                callback=pipe_callback,
                callback_steps=1
            ).images[0]
        
        # Convert PIL image to bytes
        img_byte_array = io.BytesIO()
        image.save(img_byte_array, format='PNG')
        img_bytes = img_byte_array.getvalue()
        
        # Get image dimensions
        width, height = image.size
        channels = 3  # RGB
        
        # Report time
        total_time = time.time() - start_time
        print(f"Image generation completed in {total_time:.2f} seconds")
        
        # Return the result
        return (width, height, channels, img_bytes)
        
    except Exception as e:
        import traceback
        print(f"Error during image generation: {e}")
        traceback.print_exc()
        raise

# For testing from command line
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python py_diffusion_interface.py 'prompt'")
        sys.exit(1)
    
    prompt = sys.argv[1]
    model = "runwayml/stable-diffusion-v1-5"
    
    width, height, channels, img_data = run_inference(
        model, prompt, 5, 512, 7.5, 512, 42, True, None, None
    )
    
    # Save the image
    with open("output.png", "wb") as f:
        f.write(img_data)
    
    print(f"Image saved to output.png ({width}x{height})") 