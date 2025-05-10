"""
Script to run text-to-video generation with SIMD optimizations.
"""

from text_to_video_pipeline import TextToVideoPipeline
import os

def main():
    # Initialize the pipeline with optimized kernels
    pipeline = TextToVideoPipeline(
        model_id="CompVis/stable-diffusion-v1-4",  # Smaller model
        device="cpu"
    )
    
    # Generate video from text prompt
    text_prompt = "A beautiful sunset over the ocean, waves gently crashing on the shore"
    frames = pipeline.generate(
        text_prompt=text_prompt,
        num_frames=8,  # Generate 8 frames
        frame_size=(256, 256),  # Smaller frame size
        num_inference_steps=4,  # Fewer steps for faster generation
        guidance_scale=7.0  # Slightly lower guidance scale
    )
    
    # Save the generated video
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "generated_video.gif")
    pipeline.save_video(frames, output_path)
    print(f"Video saved to {output_path}")

if __name__ == "__main__":
    main() 