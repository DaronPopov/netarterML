"""
Text-to-video generation pipeline using optimized ARM64 kernels.
"""

import torch
from diffusers import StableDiffusionPipeline
import numpy as np
from PIL import Image

from .video_decoder_kernel_asm import VideoDecoderKernelASM
from .cross_attention_kernel_asm import CrossAttentionKernelASM
from .memory_efficient_attention_kernel_asm import MemoryEfficientAttentionKernelASM
from .latent_space_projection_kernel_asm import LatentSpaceProjectionKernelASM

class TextToVideoPipeline:
    def __init__(
        self,
        model_id: str = "CompVis/stable-diffusion-v1-4",
        device: str = "cpu"
    ):
        """Initialize the text-to-video generation pipeline.
        
        Args:
            model_id: HuggingFace model ID (default: CompVis/stable-diffusion-v1-4)
            device: Device to run the model on
        """
        self.model_id = model_id
        self.device = device
        self.dtype = torch.float32  # Use FP32 for CPU
        
        # Initialize base pipeline
        self.pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=self.dtype,
            safety_checker=None  # Disable safety checker for speed
        ).to(device)
        
        # Initialize optimized kernels
        self.video_decoder = VideoDecoderKernelASM()
        self.cross_attention = CrossAttentionKernelASM()
        self.memory_efficient_attention = MemoryEfficientAttentionKernelASM()
        self.latent_projection = LatentSpaceProjectionKernelASM()
        
        # Enable memory efficient attention
        self.pipe.enable_attention_slicing()

    def generate(
        self,
        text_prompt: str,
        num_frames: int = 8,
        frame_size: tuple = (256, 256),
        num_inference_steps: int = 4,
        guidance_scale: float = 7.0
    ):
        """Generate a video from text prompt.
        
        Args:
            text_prompt: Text description of the desired video
            num_frames: Number of frames to generate
            frame_size: Size of each frame (height, width)
            num_inference_steps: Number of denoising steps
            guidance_scale: Guidance scale for classifier-free guidance
            
        Returns:
            Generated video frames as a list of PIL Images
        """
        # Generate initial latents for each frame
        all_latents = []
        for _ in range(num_frames):
            latents = self.pipe(
                prompt=text_prompt,
                height=frame_size[0],
                width=frame_size[1],
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                output_type="latent"
            ).images
            all_latents.append(latents.detach())  # Detach tensors
        
        # Stack latents into video shape
        latents_np = np.stack([l.cpu().numpy() for l in all_latents], axis=1)  # [1, num_frames, C, H, W]
        
        # Get text embeddings
        text_input = self.pipe.tokenizer(
            text_prompt,
            padding="max_length",
            max_length=self.pipe.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt"
        )
        with torch.no_grad():  # Disable gradient computation
            text_embeddings = self.pipe.text_encoder(text_input.input_ids.to(self.device))[0]
            text_embeddings = text_embeddings.cpu().numpy()
        
        # Reshape tensors for cross-attention
        batch_size, num_frames, channels, height, width = latents_np.shape
        latents_flat = latents_np.reshape(batch_size, num_frames, -1)  # [B, T, C*H*W]
        text_embeddings_flat = text_embeddings.reshape(batch_size, -1, text_embeddings.shape[-1])  # [B, L, D]
        
        # Project latents to match text embedding dimension
        latent_dim = latents_flat.shape[-1]
        text_dim = text_embeddings_flat.shape[-1]
        projection_matrix = np.random.randn(latent_dim, text_dim).astype(np.float32) / np.sqrt(latent_dim)
        latents_projected = np.matmul(latents_flat, projection_matrix)  # [B, T, D]
        
        # Apply cross-attention between frames
        video_latents = self.cross_attention.apply_cross_attention(text_embeddings_flat, latents_projected)
        
        # Project back to original latent dimension
        video_latents = np.matmul(video_latents, projection_matrix.T)  # [B, T, C*H*W]
        
        # Reshape back to video shape
        video_latents = video_latents.reshape(batch_size, num_frames, channels, height, width)
        
        # Apply memory-efficient attention for temporal consistency
        video_latents = self.memory_efficient_attention.apply_attention(video_latents)
        
        # Project latents to final space
        video_latents = self.latent_projection.project(video_latents)
        
        # Decode latents to pixel space using our optimized decoder
        video_frames = self.video_decoder.decode_video(video_latents)
        
        # Convert to PIL images
        pil_frames = []
        for i in range(num_frames):
            frame = (video_frames[0, i] * 255).astype(np.uint8).transpose(1, 2, 0)  # [H, W, C]
            pil_frames.append(Image.fromarray(frame))
        
        return pil_frames

    def save_video(self, frames, output_path: str) -> None:
        """Save generated frames as a video file.
        
        Args:
            frames: List of PIL Images
            output_path: Path to save the video file
        """
        if len(frames) > 0:
            frames[0].save(
                output_path, 
                save_all=True,
                append_images=frames[1:],
                duration=100,  # 100ms per frame = 10fps
                loop=0
            ) 