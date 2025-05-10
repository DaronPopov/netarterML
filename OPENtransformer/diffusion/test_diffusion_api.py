#!/usr/bin/env python3
import argparse
from engines.diffusion_api import DiffusionAPI
import time

def main():
    parser = argparse.ArgumentParser(description='Test Stable Diffusion API with optimized kernels')
    parser.add_argument('--prompt', type=str, default="a mystical ancient temple emerging from a cosmic nebula, with time-folded architecture and ethereal light streams",
                      help='Text prompt for image generation')
    parser.add_argument('--negative-prompt', type=str, default=None,
                      help='Negative prompt for image generation')
    parser.add_argument('--height', type=int, default=512,
                      help='Image height')
    parser.add_argument('--width', type=int, default=512,
                      help='Image width')
    parser.add_argument('--steps', type=int, default=15,
                      help='Number of inference steps')
    parser.add_argument('--guidance-scale', type=float, default=8.5,
                      help='Guidance scale')
    parser.add_argument('--seed', type=int, default=None,
                      help='Random seed for reproducibility')
    parser.add_argument('--model', type=str, default="runwayml/stable-diffusion-v1-5",
                      help='Model ID to use')
    
    args = parser.parse_args()
    
    print("\n=== Starting Optimized Stable Diffusion Test ===\n")
    print("Parameters:")
    print(f"Model: {args.model}")
    print(f"Prompt: {args.prompt}")
    print(f"Negative prompt: {args.negative_prompt}")
    print(f"Image size: {args.height}x{args.width}")
    print(f"Steps: {args.steps}")
    print(f"Guidance scale: {args.guidance_scale}")
    print(f"Seed: {args.seed}\n")
    
    # Initialize API
    api = DiffusionAPI(model_id=args.model)
    api.load_model()
    
    # Generate image
    print("\nGenerating image...")
    image = api.generate(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        height=args.height,
        width=args.width,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        seed=args.seed
    )
    
    # Save image
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_path = f"generated_image_{timestamp}.png"
    image.save(output_path)
    print(f"\nImage saved: {output_path}")
    
    # Display image immediately
    print("\nDisplaying image...")
    image.show()
    
    # Print performance stats
    stats = api.get_performance_stats()
    print("\nPerformance Statistics:")
    print(f"Generation time: {stats['average_time']:.2f}s")
    print(f"Total time: {stats['total_time']:.2f}s")
    print(f"Number of generations: {stats['num_generations']}")
    
    print("\n=== Generation Complete ===")

if __name__ == "__main__":
    main() 