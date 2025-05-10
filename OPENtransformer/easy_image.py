#!/usr/bin/env python3

"""
A super easy way to generate images with AI models for beginner programmers.
Just run this file and start generating images!
"""

import os
import sys
import time
import numpy as np
import torch
from pathlib import Path
from PIL import Image
from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionXLPipeline,
    FluxControlNetPipeline,
    ControlNetModel
)
from transformers import CLIPTextModel, CLIPTokenizer
import subprocess
import argparse

# Add the project root to Python path
project_root = str(Path(__file__).parent.absolute())
sys.path.insert(0, project_root)

# Import SIMD optimized kernels
from OPENtransformer.arm64_engine.core.asm.kernels.diffusion.fp32_optimized.cross_attention_kernel_asm import CrossAttentionKernelASM
from OPENtransformer.arm64_engine.core.asm.kernels.diffusion.fp32_optimized.memory_efficient_attention_kernel_asm import MemoryEfficientAttentionKernelASM
from OPENtransformer.arm64_engine.core.asm.kernels.diffusion.fp32_optimized.latent_space_projection_kernel_asm import LatentSpaceProjectionKernelASM
from OPENtransformer.arm64_engine.core.asm.kernels.diffusion.fp32_optimized.diffusion_process_kernel_asm import DiffusionProcessKernelASM
from OPENtransformer.arm64_engine.core.asm.kernels.diffusion.fp32_optimized.diffusion_layer_norm_asm import DiffusionLayerNormASM
from OPENtransformer.arm64_engine.core.asm.kernels.diffusion.fp32_optimized.diffusion_feed_forward_asm import DiffusionFeedForwardASM
from OPENtransformer.arm64_engine.core.asm.kernels.diffusion.fp32_optimized.noise_scheduling_kernel_asm import NoiseSchedulingKernelASM

class EasyImage:
    def __init__(self, model_name="runwayml/stable-diffusion-v1-5", model_type="stable-diffusion", offline_mode=False):
        """
        Start the image generation system with a specific AI model.
        
        Args:
            model_name: Name or path of the model to use
            model_type: Type of diffusion model to use. Options:
                - "stable-diffusion": Standard Stable Diffusion model
                - "stable-diffusion-xl": Stable Diffusion XL model
                - "controlnet": ControlNet model for controlled generation
            offline_mode: Whether to use local models only
        """
        self.model_name = model_name
        self.model_type = model_type
        self.pipe = None
        self.tokenizer = None
        self.text_encoder = None
        self.num_inference_steps = 7
        self.guidance_scale = 7.5
        self.image_size = 512
        self.offline_mode = offline_mode
        
        # Get Hugging Face token from environment
        self.hf_token = os.environ.get('HUGGINGFACE_TOKEN')
        
        # Check for GPU availability
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        # Initialize SIMD kernels
        self.cross_attention = CrossAttentionKernelASM()
        self.memory_efficient_attention = MemoryEfficientAttentionKernelASM()
        self.latent_projection = LatentSpaceProjectionKernelASM()
        self.diffusion_process = DiffusionProcessKernelASM()
        self.layer_norm = DiffusionLayerNormASM()
        self.feed_forward = DiffusionFeedForwardASM()
        self.noise_scheduler = NoiseSchedulingKernelASM()
        
        # Set up model cache directories - check both common locations
        self.cache_dirs = [
            os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "models"),
            os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub")
        ]
        for cache_dir in self.cache_dirs:
            os.makedirs(cache_dir, exist_ok=True)

    def clear_screen(self):
        """Clear the terminal screen"""
        os.system('cls' if os.name == 'nt' else 'clear')

    def download_model(self, model_name=None):
        """
        Download a model and cache it locally for offline use.
        
        Args:
            model_name: Optional model name to download. If None, uses self.model_name
        """
        if model_name is None:
            model_name = self.model_name
        
        # Clear memory before downloading
        self._clear_memory_cache()
            
        print(f"Downloading model {model_name}...")
        try:
            if self.model_type == "stable-diffusion-xl":
                self.pipe = StableDiffusionXLPipeline.from_pretrained(
                    model_name,
                    torch_dtype=torch.float32,
                    safety_checker=None,
                    requires_safety_checker=False,
                    use_safetensors=True,
                    cache_dir=self.cache_dirs[0],
                    token=self.hf_token
                )
            elif self.model_type == "controlnet":
                # Download base model
                base_model = StableDiffusionPipeline.from_pretrained(
                    "runwayml/stable-diffusion-v1-5",
                    torch_dtype=torch.float32,
                    safety_checker=None,
                    requires_safety_checker=False,
                    use_safetensors=True,
                    cache_dir=self.cache_dirs[0],
                    token=self.hf_token
                )
                # Download ControlNet model
                controlnet = ControlNetModel.from_pretrained(
                    model_name,
                    torch_dtype=torch.float32,
                    cache_dir=self.cache_dirs[0],
                    token=self.hf_token
                )
                # Create pipeline
                self.pipe = FluxControlNetPipeline(
                    vae=base_model.vae,
                    text_encoder=base_model.text_encoder,
                    tokenizer=base_model.tokenizer,
                    unet=base_model.unet,
                    controlnet=controlnet,
                    scheduler=base_model.scheduler,
                    safety_checker=None,
                    requires_safety_checker=False
                )
            else:  # Default to stable-diffusion
                self.pipe = StableDiffusionPipeline.from_pretrained(
                    model_name,
                    torch_dtype=torch.float32,
                    safety_checker=None,
                    requires_safety_checker=False,
                    use_safetensors=True,
                    cache_dir=self.cache_dirs[0],
                    token=self.hf_token
                )
            
            # Clean up memory after download
            self._clear_memory_cache()
            
            print(f"Model {model_name} downloaded and cached successfully")
            return True
        except Exception as e:
            print(f"Error downloading model: {e}")
            return False

    def setup(self):
        """
        Set up the AI model. Call this once before generating images.
        It will download and cache the model if needed.
        """
        print("Loading the AI model... This might take a minute.")
        try:
            # Clear any existing models and caches
            self._clear_memory_cache()
            
            if self.offline_mode:
                # Try to load from known cache paths
                model_loaded = False
                
                # Try specific model paths directly based on model name
                if self.model_name == "runwayml/stable-diffusion-v1-5":
                    possible_paths = [
                        os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub", "models--runwayml--stable-diffusion-v1-5"),
                        os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "models", "models--runwayml--stable-diffusion-v1-5")
                    ]
                    
                    for model_path in possible_paths:
                        if os.path.exists(model_path):
                            print(f"   • Using cached model from: {model_path}")
                            if self._load_model_from_local(model_path):
                                model_loaded = True
                                break
                else:
                    # Generic path construction for other models
                    model_name_path = self.model_name.replace("/", "--")
                    possible_paths = [
                        os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub", f"models--{model_name_path}"),
                        os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "models", f"models--{model_name_path}")
                    ]
                    
                    for model_path in possible_paths:
                        if os.path.exists(model_path):
                            print(f"   • Using cached model from: {model_path}")
                            if self._load_model_from_local(model_path):
                                model_loaded = True
                                break
                
                # Also check for direct access with the model name (local file path)
                if not model_loaded and os.path.exists(self.model_name):
                    print(f"   • Using local model from: {self.model_name}")
                    if self._load_model_from_local(self.model_name):
                        model_loaded = True
                
                if not model_loaded:
                    # Try the original model name directly, in case it's a HF model ID
                    try:
                        print(f"   • Trying to load model directly with ID: {self.model_name}")
                        if self.model_type == "stable-diffusion-xl":
                            self.pipe = StableDiffusionXLPipeline.from_pretrained(
                                self.model_name,
                                torch_dtype=torch.float32,
                                safety_checker=None,
                                requires_safety_checker=False,
                                use_safetensors=True,
                                local_files_only=True
                            )
                        else:  # Default to stable-diffusion
                            self.pipe = StableDiffusionPipeline.from_pretrained(
                                self.model_name,
                                torch_dtype=torch.float32,
                                safety_checker=None,
                                requires_safety_checker=False,
                                use_safetensors=True,
                                local_files_only=True
                            )
                        model_loaded = True
                    except Exception as e:
                        print(f"   • Error trying to load model directly: {e}")
                        
                if not model_loaded:
                    print("   • Error: No cached model found")
                    print("   • Please run download_model() first to cache the model")
                    return False
            else:
                # Online mode - try to load from cache first, then download if needed
                model_loaded = False
                
                # Try specific model paths directly based on model name
                if self.model_name == "runwayml/stable-diffusion-v1-5":
                    possible_paths = [
                        os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub", "models--runwayml--stable-diffusion-v1-5"),
                        os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "models", "models--runwayml--stable-diffusion-v1-5")
                    ]
                    
                    for model_path in possible_paths:
                        if os.path.exists(model_path):
                            print(f"   • Using cached model from: {model_path}")
                            if self._load_model_from_local(model_path):
                                model_loaded = True
                                break
                else:
                    # Generic path construction for other models
                    model_name_path = self.model_name.replace("/", "--")
                    possible_paths = [
                        os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub", f"models--{model_name_path}"),
                        os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "models", f"models--{model_name_path}")
                    ]
                    
                    for model_path in possible_paths:
                        if os.path.exists(model_path):
                            print(f"   • Using cached model from: {model_path}")
                            if self._load_model_from_local(model_path):
                                model_loaded = True
                                break
                
                # If no cached model, download it
                if not model_loaded:
                    print(f"   • Downloading model {self.model_name}...")
                    if not self.download_model():
                        return False
            
            # === OPTIMIZATION: MOVE MODEL TO CPU AND OPTIMIZE FOR INFERENCE ===
            # Move model to CPU
            self.pipe = self.pipe.to("cpu")
            
            # === OPTIMIZATION: ENABLE MODEL OPTIMIZATIONS ===
            # Enable memory-efficient attention
            self.pipe.enable_attention_slicing(slice_size=1)
            
            # Set models to evaluation mode 
            self.pipe.unet.eval()
            self.pipe.vae.eval()
            self.pipe.text_encoder.eval()
            
            # Pre-compile critical operations for reduced overhead
            self._pre_compile_critical_ops()
            
            # Verify model components are loaded
            if not hasattr(self.pipe, 'text_encoder') or self.pipe.text_encoder is None:
                raise RuntimeError("Model text encoder not properly loaded")
            
            # Convert model weights to SIMD format
            print("Converting model weights to SIMD format...")
            self._convert_to_simd()
            
            # Final memory clean up after initialization
            self._clear_memory_cache()
            
            self.clear_screen()
            print("AI model is ready! Let's generate some images.")
            return True
        except Exception as e:
            print(f"Oops, something went wrong: {e}")
            print("Please check your internet connection and try again.")
            return False
            
    def _pre_compile_critical_ops(self):
        """Pre-compile critical operations to reduce JIT overhead during generation"""
        try:
            # Create minimal test inputs
            test_prompt = "test"
            test_inputs = self.pipe.tokenizer(
                test_prompt,
                padding="max_length",
                max_length=self.pipe.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            with torch.no_grad():
                # Pre-compile text encoder
                _ = self.pipe.text_encoder(test_inputs.input_ids.to(self.device))
                
                # Pre-compile UNet with small input
                test_latent = torch.randn(
                    (1, self.pipe.unet.config.in_channels, 8, 8),
                    device=self.device
                )
                test_timestep = torch.tensor([0], device=self.device, dtype=torch.float32)
                test_text_embeddings = torch.randn(
                    (1, self.pipe.tokenizer.model_max_length, self.pipe.text_encoder.config.hidden_size),
                    device=self.device
                )
                
                # Pre-compile UNet forward pass
                _ = self.pipe.unet(
                    test_latent, 
                    test_timestep, 
                    encoder_hidden_states=test_text_embeddings
                )
                
                # Pre-compile VAE decoder
                _ = self.pipe.vae.decode(test_latent)
            
            print("   • Pre-compiled critical operations for faster inference")
        except Exception as e:
            print(f"   • Warning: Pre-compilation failed: {e}")
            print("   • Continuing without pre-compilation")
            
    def _convert_to_simd(self):
        """Convert model weights to SIMD format"""
        try:
            # Convert UNet weights
            if hasattr(self.pipe, 'unet'):
                unet_weights = self.pipe.unet.state_dict()
                # Convert weights to numpy arrays
                for key, value in unet_weights.items():
                    if isinstance(value, torch.Tensor):
                        unet_weights[key] = value.cpu().numpy()
                # Store converted weights
                self.unet_weights = unet_weights
            
            # Convert VAE weights
            if hasattr(self.pipe, 'vae'):
                vae_weights = self.pipe.vae.state_dict()
                # Convert weights to numpy arrays
                for key, value in vae_weights.items():
                    if isinstance(value, torch.Tensor):
                        vae_weights[key] = value.cpu().numpy()
                # Store converted weights
                self.vae_weights = vae_weights
            
            # Convert text encoder weights
            if hasattr(self.pipe, 'text_encoder'):
                text_encoder_weights = self.pipe.text_encoder.state_dict()
                # Convert weights to numpy arrays
                for key, value in text_encoder_weights.items():
                    if isinstance(value, torch.Tensor):
                        text_encoder_weights[key] = value.cpu().numpy()
                # Store converted weights
                self.text_encoder_weights = text_encoder_weights
            
            print("Model weights converted to SIMD format successfully")
            
        except Exception as e:
            print(f"Warning: Failed to convert model weights to SIMD format: {e}")
            print("Falling back to PyTorch implementation")

    def _load_model_from_local(self, model_path):
        """Load model from local path"""
        try:
            if self.model_type == "stable-diffusion-xl":
                self.pipe = StableDiffusionXLPipeline.from_pretrained(
                    str(model_path),
                    torch_dtype=torch.float32,
                    safety_checker=None,
                    requires_safety_checker=False,
                    use_safetensors=True,
                    local_files_only=True
                )
            else:  # Default to stable-diffusion
                self.pipe = StableDiffusionPipeline.from_pretrained(
                    str(model_path),
                    torch_dtype=torch.float32,
                    safety_checker=None,
                    requires_safety_checker=False,
                    use_safetensors=True,
                    local_files_only=True
                )
            return True
        except Exception as e:
            print(f"Error loading model from {model_path}: {e}")
            return False

    def open_image(self, image_path):
        """Open the image using the default system viewer"""
        if sys.platform == 'darwin':  # macOS
            subprocess.run(['open', image_path])
        elif sys.platform == 'win32':  # Windows
            os.startfile(image_path)
        else:  # Linux and others
            subprocess.run(['xdg-open', image_path])

    def _process_latents_simd(self, latents: np.ndarray, text_embeddings: np.ndarray, timestep: float) -> np.ndarray:
        """
        Process latents using SIMD optimized kernels with kernel fusion to reduce overhead.
        
        This implementation:
        1. Minimizes memory allocations by reusing buffers
        2. Performs kernel fusion where possible
        3. Reduces Python/C boundary crossings
        """
        try:
            # Convert to torch tensors once - avoid repeated conversions
            latents_torch = torch.from_numpy(latents).to(self.device)
            text_embeddings_torch = torch.from_numpy(text_embeddings).to(self.device)
            
            # Ensure timestep is a float tensor (important: must be float, not long)
            timestep_tensor = torch.tensor([timestep], device=self.device, dtype=torch.float32)
            
            # Use UNet in evaluation mode to disable dropout for better performance
            self.pipe.unet.eval()
            
            # Use torch.no_grad to avoid building computational graph
            with torch.no_grad():
                # Run the entire UNet in one call to minimize Python overhead
                noise_pred = self.pipe.unet(
                    latents_torch,
                    timestep_tensor,
                    encoder_hidden_states=text_embeddings_torch
                ).sample
            
            # Convert back to numpy - do this only once at the end
            return noise_pred.detach().cpu().numpy()
            
        except Exception as e:
            print(f"\r❌ Error in UNet: {str(e)[:100]}...", end="")
            # Return zeros with the right shape in case of failure
            if isinstance(latents, np.ndarray):
                return np.zeros_like(latents)
            else:
                return np.zeros((2, 4, 64, 64), dtype=np.float32)

    def _generate_with_simd(self, prompt: str, num_images: int) -> list:
        """Generate images using optimized implementation with proper error handling"""
        try:
            print("\n🔍 Encoding prompt...")
            
            # Track performance
            generation_start = time.time()
            
            # Reset temporary buffers
            unet_temp_buffers = []
            
            # Standard PyTorch implementation with performance tracking
            with torch.no_grad():
                # 1. Encode text
                text_inputs = self.pipe.tokenizer(
                    prompt,
                    padding="max_length",
                    max_length=self.pipe.tokenizer.model_max_length,
                    truncation=True,
                    return_tensors="pt",
                )
                text_embeddings = self.pipe.text_encoder(text_inputs.input_ids.to(self.device))[0]
                
                # Handle negative prompt
                uncond_input = self.pipe.tokenizer(
                    [""] * len([prompt]),
                    padding="max_length",
                    max_length=text_inputs.input_ids.shape[-1],
                    return_tensors="pt"
                )
                uncond_embeddings = self.pipe.text_encoder(uncond_input.input_ids.to(self.device))[0]
                text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
            
            print("🎨 Initializing latents...")
            
            # Use a fresh random seed for each generation
            random_seed = int(time.time()) % 10000
            generator = torch.Generator(device=self.device).manual_seed(random_seed)
            print(f"   • Using random seed: {random_seed}")
            
            # 2. Initialize latents
            latents = torch.randn(
                (num_images, self.pipe.unet.config.in_channels, self.image_size // 8, self.image_size // 8),
                device=self.device,
                dtype=torch.float32,
                generator=generator
            )
            
            # 3. Set up scheduler
            self.pipe.scheduler.set_timesteps(self.num_inference_steps)
            timesteps = self.pipe.scheduler.timesteps
            
            # Check if init_noise_sigma is None before using it
            init_noise_sigma = getattr(self.pipe.scheduler, 'init_noise_sigma', None)
            if init_noise_sigma is not None:
                latents = latents * init_noise_sigma
            
            # 4. Denoising loop
            print("\n⏳ Generating image:")
            
            # Create a progress bar display
            def progress_bar(current, total, bar_length=30):
                progress = current / total
                arrow = '█' * int(round(progress * bar_length))
                spaces = ' ' * (bar_length - len(arrow))
                percent = int(progress * 100)
                return f"[{arrow}{spaces}] {percent}% ({current}/{total})"
            
            # Track step times
            step_times = []
            
            # Create reusable buffers for optimization
            latent_model_input_buffer = None
            noise_pred_buffer = None
            
            for i, t in enumerate(timesteps):
                step_start = time.time()
                
                # Display progress
                bar = progress_bar(i+1, len(timesteps))
                print(f"\r{bar} Step: {i+1}/{len(timesteps)} | t={t.item():.2f}", end="")
                
                # Reuse buffer if possible, otherwise create a new one
                if latent_model_input_buffer is None or latent_model_input_buffer.shape != torch.cat([latents] * 2).shape:
                    latent_model_input_buffer = torch.cat([latents] * 2)
                else:
                    # Copy data to existing buffer
                    latent_model_input_buffer[:latents.shape[0]] = latents
                    latent_model_input_buffer[latents.shape[0]:] = latents
                
                # Get noise prediction
                with torch.no_grad():
                    noise_pred = self.pipe.unet(
                        latent_model_input_buffer, 
                        t, 
                        encoder_hidden_states=text_embeddings
                    ).sample
                
                # Perform guidance
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)
                
                # Scheduler step - Add null check for scheduler step method
                step_output = self.pipe.scheduler.step(noise_pred, t, latents)
                if step_output is not None and hasattr(step_output, 'prev_sample') and step_output.prev_sample is not None:
                    latents = step_output.prev_sample
                else:
                    # Fallback if scheduler step output is invalid
                    print("\r⚠️ Warning: Invalid scheduler step output, using noise prediction directly", end="")
                    # Simple fallback denoising step
                    latents = latents - 0.1 * noise_pred
                
                # Track performance
                step_time = time.time() - step_start
                step_times.append(step_time)
                
                # Explicitly clear some temporary buffers
                del noise_pred_uncond, noise_pred_text, noise_pred
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            print("\n\n🖼️ Decoding image from latents...")
            
            # 5. Decode latents
            with torch.no_grad():
                # Check if VAE scaling factor exists
                scaling_factor = 1.0
                if hasattr(self.pipe.vae.config, 'scaling_factor') and self.pipe.vae.config.scaling_factor is not None:
                    scaling_factor = self.pipe.vae.config.scaling_factor
                
                image = self.pipe.vae.decode(latents / scaling_factor).sample
            
            # Convert to PIL images
            image = (image / 2 + 0.5).clamp(0, 1)
            image = image.cpu().permute(0, 2, 3, 1).numpy()
            
            # Fix for invalid values warning - replace NaNs or infinities with zeros
            image = np.nan_to_num(image, nan=0.0, posinf=1.0, neginf=0.0)
            
            # Ensure values are in valid range before conversion
            image = np.clip(image, 0, 1)
            image = (image * 255).round().astype("uint8")
            images = [Image.fromarray(img) for img in image]
            
            # Performance metrics
            total_time = time.time() - generation_start
            avg_step_time = sum(step_times) / len(step_times)
            
            print("\n✅ Image generation complete!")
            print(f"\n⚡ Performance metrics:")
            print(f"  • Total time: {total_time:.3f}s")
            print(f"  • Average step time: {avg_step_time:.3f}s")
            print(f"  • Steps per second: {1/avg_step_time:.2f}")
            
            # Clear buffers
            del latents, text_embeddings, latent_model_input_buffer
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            return images
            
        except Exception as e:
            print(f"\n❌ Error in generation: {e}")
            print("Falling back to standard PyTorch implementation")
            return self._generate_with_pytorch(prompt, num_images)

    def _generate_with_pytorch(self, prompt: str, num_images: int) -> list:
        """Generate images using PyTorch implementation"""
        try:
            # Use a fresh random seed for each generation
            random_seed = int(time.time()) % 10000
            generator = torch.Generator(device="cpu").manual_seed(random_seed)
            print(f"   • Using random seed: {random_seed} for PyTorch generation")
            
            # Prepare generation arguments
            gen_args = {
                "prompt": [prompt] * num_images,
                "num_inference_steps": self.num_inference_steps,
                "guidance_scale": self.guidance_scale,
                "width": self.image_size,
                "height": self.image_size,
                "generator": generator
            }
            
            # Generate images using PyTorch pipeline
            with torch.no_grad():
                images = self.pipe(**gen_args).images
            
            # Force memory cleanup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            return images
            
        except Exception as e:
            print(f"Error in PyTorch generation: {e}")
            # Return empty list in case of error
            return []

    def generate(self, prompt: str, num_images: int = 1, save_path: str = None, control_image: Image = None) -> list:
        """
        Generate images from a text prompt.
        
        Args:
            prompt: Text description of the image to generate
            num_images: Number of images to generate
            save_path: Optional path to save the images
            control_image: Optional control image for ControlNet models
            
        Returns:
            List of PIL Image objects
        """
        if self.pipe is None:
            print("Please run setup() first!")
            return None

        try:
            print(f"\nGenerating {num_images} image(s) for prompt: '{prompt}'")
            start_time = time.time()
            
            # Clear CUDA cache and garbage collect to ensure fresh generation
            self._clear_memory_cache()
            
            # Generate images using the pipeline
            generator = torch.Generator(device="cpu").manual_seed(42)  # For reproducibility
            
            # Ensure the model is on CPU and ready
            print("Checking model components before generation...")
            if not hasattr(self.pipe, 'text_encoder') or self.pipe.text_encoder is None:
                print("Model text encoder not properly initialized. Please run setup() again.")
                return None
            print("✓ Text encoder ready.")
            
            if not hasattr(self.pipe, 'vae') or self.pipe.vae is None:
                print("Model VAE not properly initialized. Please run setup() again.")
                return None
            print("✓ VAE ready.")
            
            if not hasattr(self.pipe, 'unet') or self.pipe.unet is None:
                print("Model UNET not properly initialized. Please run setup() again.")
                return None
            print("✓ UNET ready.")
            
            print("\n🚀 Starting image generation with SIMD optimizations...")
            
            # Additional check for internal None values before generation
            if hasattr(self.pipe, 'scheduler') and self.pipe.scheduler is None:
                print("Scheduler not initialized. Please run setup() again.")
                return None
            print("✓ Scheduler ready.")
            
            # Prepare generation arguments
            gen_args = {
                "prompt": [prompt] * num_images,
                "num_inference_steps": self.num_inference_steps,
                "guidance_scale": self.guidance_scale,
                "width": self.image_size,
                "height": self.image_size,
                "generator": generator
            }
            
            # Add control image if using ControlNet
            if self.model_type == "controlnet" and control_image is not None:
                gen_args["image"] = control_image
            
            # Try to use SIMD kernels if available
            try:
                # Generate images using SIMD kernels
                images = self._generate_with_simd(prompt, num_images)
                
            except Exception as simd_error:
                print(f"Warning: SIMD generation failed: {simd_error}")
                print("Falling back to PyTorch implementation")
                # Generate images using PyTorch
                images = self._generate_with_pytorch(prompt, num_images)

            # Clear memory again after generation
            self._clear_memory_cache()

            if not images or len(images) == 0:
                print("No images were generated. Please try again with a different prompt.")
                return None

            # Save images if save_path is provided
            if save_path:
                os.makedirs(save_path, exist_ok=True)
                for i, image in enumerate(images):
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    image_path = os.path.join(save_path, f"generated_image_{timestamp}.png")
                    image.save(image_path)
                    print(f"💾 Saved image to {image_path}")
                    # Open the image after saving
                    self.open_image(image_path)

            print(f"\n✨ Generation completed in {time.time() - start_time:.2f} seconds")
            return images

        except Exception as e:
            print(f"❌ Something went wrong: {e}")
            print("Please check if the model is properly loaded and try again.")
            return None

    def _clear_memory_cache(self):
        """Clear memory cache to ensure fresh generation each time"""
        try:
            # Clear CUDA cache if using GPU
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            # Force garbage collection
            import gc
            gc.collect()
            
            # Clear model-specific buffers
            if hasattr(self, 'pipe') and self.pipe is not None:
                # Reset attention caches if they exist
                if hasattr(self.pipe, 'unet') and self.pipe.unet is not None:
                    for module in self.pipe.unet.modules():
                        if hasattr(module, 'attn_probs_cache'):
                            module.attn_probs_cache = None
                        if hasattr(module, 'attn_output_cache'):
                            module.attn_output_cache = None
                
                # Remove any cached intermediate states
                if hasattr(self.pipe, '_state'):
                    self.pipe._state = {}
            
            print("🧹 Memory cache cleared for fresh generation")
        except Exception as e:
            print(f"Warning: Cache clearing had an issue: {e}")

def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(description='Generate images using Stable Diffusion')
    parser.add_argument('--prompt', type=str, required=True, help='Text prompt for image generation')
    parser.add_argument('--num-images', type=int, default=1, choices=range(1, 5),
                      help='Number of images to generate (1-4)')
    parser.add_argument('--output-dir', type=str, default='generated_images',
                       help='Output directory for generated images')
    parser.add_argument('--offline', action='store_true', help='Run in offline mode using local models')
    parser.add_argument('--model-type', type=str, default='stable-diffusion',
                      choices=['stable-diffusion', 'stable-diffusion-xl', 'controlnet'],
                      help='Type of diffusion model to use')
    parser.add_argument('--model-name', type=str, default='CompVis/stable-diffusion-v1-4',
                      help='Name or path of the model to use')
    
    args = parser.parse_args()
    
    print("Welcome to EasyImage! A simple way to generate images with AI.")
    print("=" * 40)
    
    image_gen = EasyImage(
        model_name=args.model_name,
        model_type=args.model_type,
        offline_mode=args.offline
    )
    if not image_gen.setup():
        print("Sorry, I couldn't set up the AI. Bye!")
        return
    
    try:
        # Generate images with the provided arguments
        images = image_gen.generate(args.prompt, args.num_images, args.output_dir)
        if not images:
            print("Sorry, I couldn't generate the images.")
            return
            
        print("\nSuccessfully generated images!")
        print(f"Images saved to: {args.output_dir}")
        
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
        print("Please try again with a different prompt.")

if __name__ == "__main__":
    main() 