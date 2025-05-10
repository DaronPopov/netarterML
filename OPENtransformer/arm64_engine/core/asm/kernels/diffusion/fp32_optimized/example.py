"""
Example script demonstrating text-to-video generation using the optimized pipeline.
"""

import argparse
from text_to_video_pipeline import TextToVideoPipeline

def main():
    parser = argparse.ArgumentParser(description="Generate video from text description")
    parser.add_argument("--prompt", type=str, required=True, help="Text description of the video to generate")
    parser.add_argument("--output", type=str, default="output.mp4", help="Output video file path")
    parser.add_argument("--num-frames", type=int, default=16, help="Number of frames to generate")
    parser.add_argument("--height", type=int, default=256, help="Height of each frame")
    parser.add_argument("--width", type=int, default=256, help="Width of each frame")
    parser.add_argument("--num-steps", type=int, default=50, help="Number of denoising steps")
    parser.add_argument("--guidance-scale", type=float, default=7.5, help="Classifier-free guidance scale")
    parser.add_argument("--seed", type=int, help="Random seed for reproducibility")
    parser.add_argument("--model", type=str, default="damo-vilab/text-to-video-ms-1.7b", help="HuggingFace model ID")
    parser.add_argument("--device", type=str, default="cpu", help="Device to run the model on")
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = TextToVideoPipeline(
        model_id=args.model,
        device=args.device
    )
    
    # Generate video
    print(f"Generating video for prompt: {args.prompt}")
    video = pipeline.generate(
        text_prompt=args.prompt,
        num_frames=args.num_frames,
        frame_size=(args.height, args.width),
        num_inference_steps=args.num_steps,
        guidance_scale=args.guidance_scale,
        seed=args.seed
    )
    
    # Save video
    print(f"Saving video to: {args.output}")
    pipeline.save_video(video, args.output)
    print("Done!")

if __name__ == "__main__":
    main() 