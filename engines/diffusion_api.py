#!/usr/bin/env python3
import os
import sys
import torch
import numpy as np
from pathlib import Path
from typing import Optional, Dict, List, Union, Tuple
from diffusers import StableDiffusionPipeline
from PIL import Image
import time

# Add the project root to Python path
project_root = str(Path(__file__).parent.parent.absolute())
sys.path.insert(0, project_root)

# Import optimized kernels
from OPENtransformer.arm64_engine.core.asm.kernels.diffusion.fp32_optimized.cross_attention_kernel_asm import CrossAttentionKernelASM
from OPENtransformer.arm64_engine.core.asm.kernels.diffusion.fp32_optimized.memory_efficient_attention_kernel_asm import MemoryEfficientAttentionKernelASM
from OPENtransformer.arm64_engine.core.asm.kernels.diffusion.fp32_optimized.latent_space_projection_kernel_asm import LatentSpaceProjectionKernelASM
from OPENtransformer.arm64_engine.core.asm.kernels.diffusion.fp32_optimized.diffusion_process_kernel_asm import DiffusionProcessKernelASM
from OPENtransformer.arm64_engine.core.asm.kernels.diffusion.fp32_optimized.diffusion_layer_norm_asm import DiffusionLayerNormASM
from OPENtransformer.arm64_engine.core.asm.kernels.diffusion.fp32_optimized.diffusion_feed_forward_asm import DiffusionFeedForwardASM
from OPENtransformer.arm64_engine.core.asm.kernels.diffusion.fp32_optimized.noise_scheduling_kernel_asm import NoiseSchedulingKernelASM

class DiffusionAPI:
    """API for running Stable Diffusion with optimized kernels"""
    
    def __init__(self, model_id: str = "runwayml/stable-diffusion-v1-5"):
        """Initialize the API with model ID"""
        self.model_id = model_id
        self.pipe = None
        self.device = "cpu"
        
        # Initialize optimized kernels
        self.cross_attention = CrossAttentionKernelASM()
        self.memory_efficient_attention = MemoryEfficientAttentionKernelASM()
        self.latent_projection = LatentSpaceProjectionKernelASM()
        self.diffusion_process = DiffusionProcessKernelASM()
        self.layer_norm = DiffusionLayerNormASM()
        self.feed_forward = DiffusionFeedForwardASM()
        self.noise_scheduler = NoiseSchedulingKernelASM()
        
        # Performance tracking
        self.total_inference_time = 0
        self.num_generations = 0
        
    def load_model(self, device: str = "cpu") -> None:
        """Load the Stable Diffusion model"""
        print(f"\nLoading model {self.model_id}...")
        
        self.device = device
        self.pipe = StableDiffusionPipeline.from_pretrained(
            self.model_id,
            torch_dtype=torch.float32,
            use_safetensors=True
        ).to(device)
        
        print("Model loaded successfully")
        print(f"UNet config: {self.pipe.unet.config}")
        
    def _process_latents(self, latents: torch.Tensor, text_embeddings: torch.Tensor, timestep: float) -> torch.Tensor:
        """Process latents using optimized kernels"""
        # Convert to numpy for SIMD processing
        latents_np = latents.detach().cpu().numpy().astype(np.float32)
        text_embeddings_np = text_embeddings.detach().cpu().numpy().astype(np.float32)
        
        # Get dimensions
        batch_size, channels, height, width = latents_np.shape
        seq_len = height * width
        hidden_size = text_embeddings_np.shape[-1]
        
        print(f"\nDebug - Shapes:")
        print(f"latents_np: {latents_np.shape}")
        print(f"text_embeddings_np: {text_embeddings_np.shape}")
        print(f"seq_len: {seq_len}, hidden_size: {hidden_size}")
        
        # Project latents
        latents_flat = latents_np.reshape(batch_size * seq_len, channels)
        print(f"latents_flat: {latents_flat.shape}")
        
        conv_weights = self.pipe.unet.conv_in.weight.detach().cpu().numpy()
        print(f"conv_weights: {conv_weights.shape}")
        
        # Project to hidden size
        projection = np.random.randn(channels, hidden_size).astype(np.float32) / np.sqrt(channels)
        latents_projected = np.matmul(latents_flat, projection)
        print(f"latents_projected: {latents_projected.shape}")
        
        # Reshape projected latents for attention
        latents_projected = latents_projected.reshape(batch_size, seq_len, hidden_size)
        print(f"latents_projected reshaped: {latents_projected.shape}")
        
        # Apply attention
        attention_output = self.cross_attention.compute_attention(
            query=latents_projected,
            key=text_embeddings_np,
            value=text_embeddings_np,
            scale_factor=1.0 / np.sqrt(hidden_size)
        )
        print(f"attention_output: {attention_output.shape}")
        
        # Project back to original channel size
        projection_back = np.random.randn(hidden_size, channels).astype(np.float32) / np.sqrt(hidden_size)
        output = np.matmul(attention_output.reshape(batch_size * seq_len, hidden_size), projection_back)
        print(f"output after back projection: {output.shape}")
        
        # Reshape and convert back to torch
        output = output.reshape(batch_size, channels, height, width)
        print(f"final_output: {output.shape}\n")
        
        return torch.from_numpy(output).to(self.device).to(torch.float32)
        
    def generate(
        self,
        prompt: str,
        negative_prompt: Optional[str] = None,
        height: int = 512,
        width: int = 512,
        num_inference_steps: int = 25,
        guidance_scale: float = 7.5,
        seed: Optional[int] = None
    ) -> Image.Image:
        """Generate an image using the optimized pipeline"""
        if self.pipe is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
            
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
            
        start_time = time.time()
        
        # Encode text
        text_inputs = self.pipe.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.pipe.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_embeddings = self.pipe.text_encoder(text_inputs.input_ids.to(self.device))[0]
        
        # Handle negative prompt
        if negative_prompt is None:
            negative_prompt = ""
        uncond_input = self.pipe.tokenizer(
            negative_prompt,
            padding="max_length",
            max_length=text_inputs.input_ids.shape[-1],
            return_tensors="pt"
        )
        uncond_embeddings = self.pipe.text_encoder(uncond_input.input_ids.to(self.device))[0]
        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
        
        # Initialize latents
        latents = torch.randn(
            (1, self.pipe.unet.config.in_channels, height // 8, width // 8),
            device=self.device,
            dtype=torch.float32
        )
        
        # Set timesteps
        self.pipe.scheduler.set_timesteps(num_inference_steps)
        latents = latents * self.pipe.scheduler.init_noise_sigma
        
        # Denoising loop
        for t in self.pipe.scheduler.timesteps:
            # Expand latents for classifier free guidance
            latent_model_input = torch.cat([latents] * 2)
            
            # Process through optimized kernels
            noise_pred = self._process_latents(latent_model_input, text_embeddings, t)
            
            # Perform guidance
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
            
            # Scheduler step
            latents = self.pipe.scheduler.step(noise_pred, t, latents).prev_sample
            
        # Decode latents
        with torch.no_grad():
            image = self.pipe.vae.decode(latents / self.pipe.vae.config.scaling_factor).sample
            
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()
        image = (image * 255).round().astype("uint8")
        image = Image.fromarray(image[0])
        
        # Update performance tracking
        self.total_inference_time += time.time() - start_time
        self.num_generations += 1
        
        return image
        
    def get_performance_stats(self) -> Dict[str, float]:
        """Get performance statistics"""
        if self.num_generations == 0:
            return {
                "average_time": 0.0,
                "total_time": 0.0,
                "num_generations": 0
            }
            
        return {
            "average_time": self.total_inference_time / self.num_generations,
            "total_time": self.total_inference_time,
            "num_generations": self.num_generations
        }
        
    def clear_stats(self) -> None:
        """Clear performance statistics"""
        self.total_inference_time = 0
        self.num_generations = 0 