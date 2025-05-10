#!/usr/bin/env python3
"""
Example usage of the Easy Diffusion API
"""

import os
import sys
import time
from pathlib import Path

# Import the Easy Diffusion API
from easy_diffusion_api import EasyDiffusionAPI

def example_basic_usage():
    """Basic usage example: register, activate and generate"""
    
    print("\n=== Basic Usage Example ===")
    
    # Create the API instance
    api = EasyDiffusionAPI()
    
    # Register Dreamlike Photoreal 2.0 model - known for photorealism
    api.register_model("dreamlike-photoreal", "dreamlike-art/dreamlike-photoreal-2.0")
    
    # Make it the active model
    api.set_active_model("dreamlike-photoreal")
    
    # Generate an image
    output_path = "generated_images/example_dreamlike_basic.png"
    result = api.generate_image(
        prompt="photo, a beautiful landscape with mountains and a lake at sunset, high quality, detailed",
        steps=20,  # Fewer steps for faster generation
        output_path=output_path
    )
    
    if result:
        print(f"Image generated successfully: {output_path}")
    else:
        print("Image generation failed")

def example_multiple_models():
    """Example using multiple models"""
    
    print("\n=== Multiple Models Example ===")
    
    # Create the API instance
    api = EasyDiffusionAPI()
    
    # Register multiple models - using Dreamlike Photoreal as main model
    models = [
        ("dreamlike-photoreal", "dreamlike-art/dreamlike-photoreal-2.0"),
        ("sd-v1-5", "runwayml/stable-diffusion-v1-5"),
        ("sd-v2", "stabilityai/stable-diffusion-2-1")
    ]
    
    for model_id, model_path in models:
        api.register_model(model_id, model_path)
    
    # List registered models
    model_list = api.list_models()
    print("Registered models:")
    for model in model_list:
        print(f"  - {model['id']}: {model['path']}")
    
    # Generate with specific models
    prompts = [
        "photo, a photorealistic portrait of a smiling young woman, perfect skin, studio lighting",
        "A futuristic city with flying cars and neon lights",
        "An abstract painting with vibrant colors"
    ]
    
    for i, (model_id, _) in enumerate(models):
        if i >= len(prompts):
            break
            
        print(f"\nGenerating with model: {model_id}")
        output_path = f"generated_images/example_model_{model_id}.png"
        
        result = api.generate_image(
            prompt=prompts[i],
            model_id=model_id,
            steps=20,
            output_path=output_path
        )
        
        if result:
            print(f"Image generated with {model_id}: {output_path}")
        else:
            print(f"Failed to generate with {model_id}")

def example_download_and_use():
    """Example downloading a model and using it"""
    
    print("\n=== Download and Use Example ===")
    
    # Create the API instance
    api = EasyDiffusionAPI()
    
    # Model to download - Dreamlike Photoreal 2.0
    model_id = "dreamlike-art/dreamlike-photoreal-2.0"
    
    # Download the model
    print(f"Downloading model: {model_id}")
    success = api.download_model(model_id)
    
    if not success:
        print("Failed to download model, check your Hugging Face token")
        print("You can set it using 'export HF_TOKEN=your_token_here'")
        return
    
    # Register with a shorter name
    api.register_model("dreamlike", model_id)
    
    # Activate the model
    api.set_active_model("dreamlike")
    
    # Generate an image
    output_path = "generated_images/example_dreamlike_downloaded.png"
    result = api.generate_image(
        prompt="photo, a cyberpunk scene with a person wearing a neon jacket, realistic, high quality",
        steps=20,
        output_path=output_path
    )
    
    if result:
        print(f"Image generated successfully: {output_path}")
    else:
        print("Image generation failed")

def example_programmatic_api():
    """Example using the API programmatically"""
    
    print("\n=== Programmatic API Example ===")
    
    # Create the API instance with a custom cache directory
    cache_dir = Path("custom_models")
    api = EasyDiffusionAPI(cache_dir=str(cache_dir))
    
    # Register the Dreamlike Photoreal model
    api.register_model(
        model_id="dreamlike-photoreal",
        model_path="dreamlike-art/dreamlike-photoreal-2.0"
    )
    
    # Set it as active
    api.set_active_model("dreamlike-photoreal")
    
    # Generate multiple images with different prompts
    prompts = [
        "photo, a colorful parrot sitting on a branch, detailed feathers, nature photography",
        "photo, a starry night over a mountain range, milky way visible, high resolution",
        "photo, a medieval castle on a hill with clouds, dramatic lighting, cinematic"
    ]
    
    # Batch generation
    generated_files = []
    for i, prompt in enumerate(prompts):
        output_path = f"generated_images/dreamlike_batch_{i+1}.png"
        
        # Generate with different seeds for variety
        result = api.generate_image(
            prompt=prompt,
            seed=i+1000,  # Use different seeds
            output_path=output_path
        )
        
        if result:
            generated_files.append(output_path)
            print(f"Generated image {i+1}/{len(prompts)}: {output_path}")
        else:
            print(f"Failed to generate image {i+1}/{len(prompts)}")
    
    print(f"Successfully generated {len(generated_files)}/{len(prompts)} images")

def example_sdxl_model():
    """Example of loading and using a SDXL model"""
    
    print("\n=== SDXL Model Example ===")
    
    # Create the API instance
    api = EasyDiffusionAPI()
    
    # Register the SDXL base model - higher resolution and quality than SD 1.5/2.1
    api.register_model("sdxl-base", "stabilityai/stable-diffusion-xl-base-1.0")
    
    # Make it the active model
    api.set_active_model("sdxl-base")
    
    # Generate a higher resolution image (SDXL works best at 1024x1024)
    output_path = "generated_images/sdxl_example.png"
    
    result = api.generate_image(
        prompt="an astronaut riding a horse on mars, highly detailed, cinematic lighting, 8k, trending on artstation",
        steps=30,  # More steps for better quality
        width=1024,  # SDXL works best at 1024x1024
        height=1024,
        guidance=9.0,  # Higher guidance for more prompt adherence
        output_path=output_path
    )
    
    if result:
        print(f"SDXL image generated successfully: {output_path}")
    else:
        print("SDXL image generation failed")
    
    # Try another SDXL model - Dreamshaper SDXL
    api.register_model("dreamshaper-xl", "lykon/dreamshaper-xl-1-0")
    api.set_active_model("dreamshaper-xl")
    
    # Generate with Dreamshaper SDXL
    output_path = "generated_images/dreamshaper_xl_example.png"
    
    result = api.generate_image(
        prompt="a mystical forest with glowing plants and magical creatures, fantasy art, detailed, vibrant colors",
        steps=25,
        width=1024,
        height=768,  # Different aspect ratio
        guidance=8.0,
        output_path=output_path
    )
    
    if result:
        print(f"Dreamshaper XL image generated successfully: {output_path}")
    else:
        print("Dreamshaper XL image generation failed")
        
    # Try the faster Dreamshaper XL Lightning model - optimized for speed
    api.register_model("dreamshaper-xl-lightning", "Lykon/dreamshaper-xl-lightning")
    api.set_active_model("dreamshaper-xl-lightning")
    
    # Generate with Dreamshaper XL Lightning - ultra-fast with lower resolution
    output_path = "generated_images/dreamshaper_xl_lightning_example.png"
    
    result = api.generate_image(
        prompt="a futuristic robot in a cyberpunk city, detailed, neon lights, reflections, cinematic, 8k",
        steps=6,  # Ultra-fast generation with only 6 steps
        width=728,  # Lower resolution for faster generation
        height=728,
        guidance=7.0,  # Slightly lower guidance works well with Lightning
        output_path=output_path
    )
    
    if result:
        print(f"Dreamshaper XL Lightning image generated successfully: {output_path}")
    else:
        print("Dreamshaper XL Lightning image generation failed")

def run_all_examples():
    """Run all examples in sequence"""
    
    # Ensure output directory exists
    os.makedirs("generated_images", exist_ok=True)
    
    # Run the examples
    try:
        example_basic_usage()
        example_multiple_models()
        example_download_and_use()
        example_programmatic_api()
        example_sdxl_model()  # Add the SDXL example
        
        print("\nAll examples completed successfully!")
    except Exception as e:
        import traceback
        print(f"Error running examples: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    # Run a specific example if specified
    if len(sys.argv) > 1:
        example_name = sys.argv[1]
        if example_name == "basic":
            example_basic_usage()
        elif example_name == "multiple":
            example_multiple_models()
        elif example_name == "download":
            example_download_and_use()
        elif example_name == "programmatic":
            example_programmatic_api()
        elif example_name == "sdxl":  # Add new command line option
            example_sdxl_model()
        else:
            print(f"Unknown example: {example_name}")
            print("Available examples: basic, multiple, download, programmatic, sdxl")
    else:
        # Run all examples
        run_all_examples() 