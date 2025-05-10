#!/usr/bin/env python3
"""
Example usage of the Easy Diffusion API
"""

import os
import sys
import time
from pathlib import Path

# Import the Easy Diffusion API
from OPENtransformer import EasyDiffusionAPI

def example_basic_usage():
    """Basic usage example: register, activate and generate"""
    
    print("\n=== Basic Usage Example ===")
    
    # Create the API instance
    api = EasyDiffusionAPI()
    
    # Register Stable Diffusion v1.5
    api.register_model("sd-v1-5", "runwayml/stable-diffusion-v1-5")
    
    # Make it the active model
    api.set_active_model("sd-v1-5")
    
    # Generate an image
    output_path = "generated_images/example_basic.png"
    result = api.generate_image(
            prompt="A beautiful landepscape with mountains and a lake at sunset",
            steps=20,  # Fewer stes for faster generation
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
    
    # Register multiple models
    models = [
        ("sd-v1-5", "runwayml/stable-diffusion-v1-5"),
        ("dreamlike", "dreamlike-art/dreamlike-photoreal-2.0"),
        ("dreamshaper", "dreamshaper/dreamshaper-xl-turbo")
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
        "A futuristic city with flying cars and neon lights",
        "A photorealistic portrait of a smiling young woman",
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
    
    # Model to download
    model_id = "runwayml/stable-diffusion-v1-5"
    
    # Download the model
    print(f"Downloading model: {model_id}")
    success = api.download_model(model_id)
    
    if not success:
        print("Failed to download model, check your Hugging Face token")
        print("You can set it using 'export HF_TOKEN=your_token_here'")
        return
    
    # Register with a shorter name
    api.register_model("sd-v1-5", model_id)
    
    # Activate the model
    api.set_active_model("sd-v1-5")
    
    # Generate an image
    output_path = "generated_images/example_downloaded.png"
    result = api.generate_image(
        prompt="A cyberpunk scene with a person wearing a neon jacket",
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
    
    # Register a model with a custom path
    api.register_model(
        model_id="my-local-model",
        model_path="models/my_fine_tuned_model"
    )
    
    # Generate multiple images with different prompts
    prompts = [
        "A colorful parrot sitting on a branch",
        "A starry night over a mountain range",
        "A medieval castle on a hill with clouds"
    ]
    
    # Batch generation
    generated_files = []
    for i, prompt in enumerate(prompts):
        output_path = f"generated_images/batch_{i+1}.png"
        
        # Generate with different seeds for variety
        result = api.generate_image(
            prompt=prompt,
            model_id="my-local-model",
            seed=i+1000,  # Use different seeds
            output_path=output_path
        )
        
        if result:
            generated_files.append(output_path)
            print(f"Generated image {i+1}/{len(prompts)}: {output_path}")
        else:
            print(f"Failed to generate image {i+1}/{len(prompts)}")
    
    print(f"Successfully generated {len(generated_files)}/{len(prompts)} images")

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
        else:
            print(f"Unknown example: {example_name}")
            print("Available examples: basic, multiple, download, programmatic")
    else:
        # Run all examples
        run_all_examples() 