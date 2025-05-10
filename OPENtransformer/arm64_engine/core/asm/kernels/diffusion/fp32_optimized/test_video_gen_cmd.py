"""
Test script for video generation using ARM-optimized diffusion kernels
"""

import torch
from diffusers import TextToVideoSDPipeline
from diffusers.utils import export_to_video
import numpy as np
import time
import warnings
import subprocess
import platform
import os
import argparse
import sys
import logging
from huggingface_hub import snapshot_download
from OPENtransformer.arm64_engine.core.asm.kernels.vision.diffusion.diffusion_kernels import DiffusionKernels
from OPENtransformer.arm64_engine.core.asm.kernels.vision.diffusion.temporal_attention_kernel_asm import TemporalAttentionKernelASM
from OPENtransformer.arm64_engine.core.asm.kernels.vision.diffusion.noise_scheduling_kernel_asm import NoiseSchedulingKernelASM
from OPENtransformer.arm64_engine.core.asm.kernels.vision.diffusion.frame_interpolation_kernel_asm import FrameInterpolationKernelASM
warnings.filterwarnings("ignore")

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Define model paths
MODEL_ID = "damo-vilab/text-to-video-ms-1.7b"
CACHE_DIR = os.path.expanduser("~/.cache/huggingface/hub")

def download_model(model_id=MODEL_ID):
    """Download the model files if they don't exist."""
    print("\nChecking model files...")
    try:
        # Download model snapshot
        snapshot_download(
            repo_id=model_id,
            local_dir=os.path.join(CACHE_DIR, f"models--{model_id.replace('/', '--')}"),
            local_dir_use_symlinks=False
        )
        print("Model files downloaded successfully")
        return True
    except Exception as e:
        print(f"Error downloading model: {str(e)}")
        return False

def check_model_files():
    """Check if model files exist locally."""
    model_path = os.path.join(CACHE_DIR, f"models--{MODEL_ID.replace('/', '--')}")
    if not os.path.exists(model_path):
        return False
    return True

class VideoGenerator:
    """Pipeline using ARM-optimized diffusion kernels for text-to-video generation."""
    
    def __init__(self, model_id=MODEL_ID, device="cpu"):
        print("\nInitializing video generation pipeline...")
        
        # Initialize ARM-optimized kernels
        print("Initializing ARM-optimized kernels...")
        self.diffusion_kernels = DiffusionKernels()
        self.temporal_attention = TemporalAttentionKernelASM()
        self.noise_scheduler = NoiseSchedulingKernelASM()
        self.frame_interpolator = FrameInterpolationKernelASM()
        
        # Ensure model is downloaded
        if not check_model_files():
            print("Model files not found. Downloading...")
            if not download_model(model_id):
                raise RuntimeError("Failed to download model files")
        
        try:
            # Initialize pipeline with explicit CPU settings
            self.pipe = TextToVideoSDPipeline.from_pretrained(
                model_id,
                torch_dtype=torch.float32,
                use_safetensors=True,
                cache_dir=CACHE_DIR
            )
            
            # Convert all models to float32
            print("Converting models to float32...")
            self.pipe.unet = self.pipe.unet.to(torch.float32)
            self.pipe.vae = self.pipe.vae.to(torch.float32)
            self.pipe.text_encoder = self.pipe.text_encoder.to(torch.float32)
            
            # Move to CPU and enable optimizations
            print("Moving models to CPU...")
            self.pipe = self.pipe.to("cpu")
            self.pipe.enable_vae_slicing()
            
            print("Model loaded successfully")
            self.device = "cpu"
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            raise
        
    def _apply_arm_optimizations(self, frames, timesteps):
        """Apply ARM-optimized operations to the frames."""
        try:
            logger.debug(f"Input frames shape: {frames.shape}")
            
            # Convert frames to numpy for ARM kernels
            if isinstance(frames, torch.Tensor):
                frames_np = frames.cpu().numpy()
            else:
                frames_np = np.array(frames)
            
            logger.debug(f"Converted frames shape: {frames_np.shape}")
            
            # Apply temporal attention
            attention_weights = np.ones((frames_np.shape[0], frames_np.shape[1])) / frames_np.shape[1]
            logger.debug(f"Attention weights shape: {attention_weights.shape}")
            
            try:
                frames_np = self.temporal_attention.apply_temporal_attention(frames_np, attention_weights)
                logger.debug("Temporal attention applied successfully")
            except Exception as e:
                logger.error(f"Error in temporal attention: {str(e)}")
                raise
            
            # Apply noise scheduling
            try:
                betas = self.noise_scheduler.get_beta_schedule(
                    schedule_type='cosine',
                    beta_start=1e-4,
                    beta_end=0.02,
                    timesteps=timesteps
                )
                logger.debug(f"Generated beta schedule with shape: {betas.shape}")
            except Exception as e:
                logger.error(f"Error in noise scheduling: {str(e)}")
                raise
            
            # Apply frame interpolation for smoother transitions
            interpolated_frames = []
            try:
                for i in range(len(frames_np) - 1):
                    frame1 = frames_np[i]
                    frame2 = frames_np[i + 1]
                    logger.debug(f"Processing frames {i} and {i+1} with shapes: {frame1.shape}, {frame2.shape}")
                    
                    interpolated = self.frame_interpolator.interpolate(
                        frame1, frame2, factor=0.5
                    )
                    interpolated_frames.append(frame1)
                    interpolated_frames.append(interpolated)
                interpolated_frames.append(frames_np[-1])
                logger.debug(f"Generated {len(interpolated_frames)} interpolated frames")
            except Exception as e:
                logger.error(f"Error in frame interpolation: {str(e)}")
                raise
            
            # Convert back to torch tensor
            result = torch.from_numpy(np.array(interpolated_frames))
            logger.debug(f"Final output shape: {result.shape}")
            return result
            
        except Exception as e:
            logger.error(f"Error in ARM optimizations: {str(e)}")
            raise
        
    def __call__(
        self,
        prompt,
        num_frames=16,
        num_inference_steps=10,
        height=256,
        width=256,
        guidance_scale=7.5,
        negative_prompt=None
    ):
        try:
            # Generate video frames
            print("\nGenerating frames...")
            video_frames = self.pipe(
                prompt,
                num_inference_steps=num_inference_steps,
                num_frames=num_frames,
                height=height,
                width=width,
                guidance_scale=guidance_scale,
                negative_prompt=negative_prompt
            ).frames
            
            # Apply ARM optimizations
            print("Applying ARM optimizations...")
            video_frames = self._apply_arm_optimizations(video_frames, num_inference_steps)
            
            # Export frames to video file
            print("Converting frames to video...")
            output_path = f"generated_video_{int(time.time())}.mp4"
            export_to_video(video_frames, output_path)
            
            return output_path
        except Exception as e:
            print(f"Error during generation: {str(e)}")
            raise

def run_test(prompt, pipeline=None):
    """Run test for video generation."""
    print("\n=== Starting Video Generation Test ===\n")
    
    # Test parameters - reduced for faster output
    num_frames = 8  # Reduced from 16
    num_inference_steps = 5  # Reduced from 10
    height = 128  # Reduced from 256
    width = 128  # Reduced from 256
    guidance_scale = 7.5
    
    print("Parameters:")
    print(f"Prompt: {prompt}")
    print(f"Number of frames: {num_frames}")
    print(f"Steps: {num_inference_steps}")
    print(f"Resolution: {width}x{height}")
    print(f"Guidance scale: {guidance_scale}\n")
    
    # Initialize pipeline if not provided
    if pipeline is None:
        print("Initializing pipeline...")
        try:
            pipeline = VideoGenerator()
            print("Pipeline initialized successfully")
        except Exception as e:
            print(f"Error initializing pipeline: {str(e)}")
            return None
    
    # Generate video
    print("\nGenerating video...")
    start_time = time.time()
    
    output_path = pipeline(
        prompt=prompt,
        num_frames=num_frames,
        num_inference_steps=num_inference_steps,
        height=height,
        width=width,
        guidance_scale=guidance_scale
    )
    
    total_time = time.time() - start_time
    print(f"\nGeneration time: {total_time:.2f}s")
    print(f"Video saved: {output_path}")
    
    # Open the video
    print("\nOpening generated video...")
    if platform.system() == "Darwin":  # macOS
        subprocess.run(["open", output_path])
    elif platform.system() == "Windows":
        os.startfile(output_path)
    else:  # Linux
        subprocess.run(["xdg-open", output_path])
    
    print("\n=== Generation Completed ===")
    return pipeline

def main():
    print("\n=== Text-to-Video Generator ===")
    print("Type 'quit' or 'exit' to end the session")
    print("Type 'clear' to clear the screen")
    print("Type 'status' to check model status")
    print("Enter your prompts one at a time to generate videos\n")
    
    # Initialize pipeline once
    pipeline = None
    
    while True:
        try:
            prompt = input("\nEnter your prompt (or 'quit' to exit): ").strip()
            
            if prompt.lower() in ['quit', 'exit']:
                print("\nGoodbye!")
                break
            elif prompt.lower() == 'clear':
                os.system('cls' if os.name == 'nt' else 'clear')
                continue
            elif prompt.lower() == 'status':
                if check_model_files():
                    print("\nModel files found locally - can work offline")
                else:
                    print("\nModel files not found - will download on first run")
                continue
            elif not prompt:
                print("Please enter a valid prompt")
                continue
            
            # Run the generation
            pipeline = run_test(prompt, pipeline)
            
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"\nAn error occurred: {str(e)}")
            continue

if __name__ == "__main__":
    main() 