#!/usr/bin/env python3
"""
Example using the Dreamshaper XL Lightning model
This model delivers fast, high-quality results with fewer inference steps
"""

import os
import sys
import time
from pathlib import Path

# Import the Easy Diffusion API
from easy_diffusion_api import EasyDiffusionAPI

def generate_with_dreamshaper_xl_lightning():
    """Example of loading and using the Dreamshaper XL Lightning model"""
    
    print("\n=== Dreamshaper XL Lightning Example ===")
    
    # Create the API instance
    api = EasyDiffusionAPI()
    
    # Register the Dreamshaper XL Lightning model
    api.register_model("dreamshaper-xl-lightning", "Lykon/dreamshaper-xl-lightning")
    
    # Make it the active model
    api.set_active_model("dreamshaper-xl-lightning")
    
    # Ensure output directory exists
    os.makedirs("generated_images", exist_ok=True)
    
    # Generate a high-quality image with lower resolution and fewer steps
    output_path = "generated_images/dreamshaper_xl_lightning_example.png"
    
    # The Lightning model can work with very few steps
    result = api.generate_image(
        prompt="a detailed portrait of a cyberpunk character with neon lights, ultra realistic, cinematic lighting, 8k, trending on artstation",
        steps=6,  # Ultra-fast generation with only 6 steps
        width=728,  # Lower resolution for faster generation
        height=728,
        guidance=7.0,  # Guidance scale for prompt adherence
        output_path=output_path
    )
    
    if result:
        print(f"Dreamshaper XL Lightning image generated successfully: {output_path}")
    else:
        print("Dreamshaper XL Lightning image generation failed")
    
    # Try generating a landscape image with different aspect ratio but same lower resolution
    output_path = "generated_images/dreamshaper_xl_lightning_landscape.png"
    
    result = api.generate_image(
        prompt="a magical landscape with waterfalls and floating islands, fantasy art, detailed, vibrant colors, cinematic, epic",
        steps=6,  # Ultra-fast generation with only 6 steps
        width=728,
        height=546,  # Maintain aspect ratio at lower resolution
        guidance=7.5,
        output_path=output_path
    )
    
    if result:
        print(f"Dreamshaper XL Lightning landscape image generated successfully: {output_path}")
    else:
        print("Dreamshaper XL Lightning landscape image generation failed")

if __name__ == "__main__":
    generate_with_dreamshaper_xl_lightning() 