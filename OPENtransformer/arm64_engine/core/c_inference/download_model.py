#!/usr/bin/env python3
"""
Script to download and cache the Stable Diffusion model.
"""

import os
import sys
import time
import torch
import numpy as np
from diffusers import StableDiffusionPipeline
import logging
import argparse
from tqdm import tqdm
import json
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def convert_to_binary_format(pipe, model_path):
    """
    Convert model components to binary format for C/ASM inference
    
    Args:
        pipe: The StableDiffusionPipeline instance
        model_path: Path to save the binary files
    """
    print("Converting model components to binary format for C/ASM inference...")
    
    try:
        # Create component directories
        os.makedirs(os.path.join(model_path, "unet"), exist_ok=True)
        os.makedirs(os.path.join(model_path, "vae"), exist_ok=True)
        os.makedirs(os.path.join(model_path, "text_encoder"), exist_ok=True)
        
        # Save UNet weights in binary format
        print("Converting UNet weights...")
        unet_weights = {}
        state_dict = pipe.unet.state_dict()
        for k, v in tqdm(state_dict.items(), desc="Converting UNet weights"):
            unet_weights[k] = v.cpu().numpy().astype(np.float32)
        
        with open(os.path.join(model_path, "unet.bin"), "wb") as f:
            np.savez(f, **unet_weights)
        
        # Save VAE weights in binary format
        print("Converting VAE weights...")
        vae_weights = {}
        state_dict = pipe.vae.state_dict()
        for k, v in tqdm(state_dict.items(), desc="Converting VAE weights"):
            vae_weights[k] = v.cpu().numpy().astype(np.float32)
        
        with open(os.path.join(model_path, "vae.bin"), "wb") as f:
            np.savez(f, **vae_weights)
        
        # Save text encoder weights in binary format
        print("Converting text encoder weights...")
        text_encoder_weights = {}
        state_dict = pipe.text_encoder.state_dict()
        for k, v in tqdm(state_dict.items(), desc="Converting text encoder weights"):
            text_encoder_weights[k] = v.cpu().numpy().astype(np.float32)
        
        with open(os.path.join(model_path, "text_encoder.bin"), "wb") as f:
            np.savez(f, **text_encoder_weights)
        
        # Save configuration metadata for C inference
        print("Saving model configuration...")
        config = {
            "model_type": "stable-diffusion",
            "model_name": Path(model_path).name,
            "unet": {
                "in_channels": pipe.unet.config.in_channels,
                "out_channels": pipe.unet.config.out_channels,
                "sample_size": pipe.unet.config.sample_size
            },
            "vae": {
                "latent_channels": pipe.vae.config.latent_channels,
                "scaling_factor": pipe.vae.config.scaling_factor
            },
            "text_encoder": {
                "hidden_size": pipe.text_encoder.config.hidden_size,
                "num_attention_heads": pipe.text_encoder.config.num_attention_heads,
                "vocab_size": pipe.text_encoder.config.vocab_size
            }
        }
        
        with open(os.path.join(model_path, "model_config.json"), "w") as f:
            json.dump(config, f, indent=2)
        
        # Create a marker file to indicate binary conversion is complete
        with open(os.path.join(model_path, "binary_conversion_complete"), "w") as f:
            f.write("1")
        
        print("Binary conversion complete!")
        return True
        
    except Exception as e:
        import traceback
        print(f"Error during binary conversion: {e}")
        traceback.print_exc()
        return False

def download_model(model_id="runwayml/stable-diffusion-v1-5", output_dir="models"):
    """Download and cache the Stable Diffusion model."""
    try:
        # Create models directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Determine local model path
        model_name = model_id.split("/")[-1] if "/" in model_id else model_id
        model_path = os.path.join(output_dir, model_name)
        
        # Check if the binary conversion is already complete
        if os.path.exists(os.path.join(model_path, "binary_conversion_complete")):
            logger.info(f"Model {model_id} already exists and is converted to binary format.")
            return True
        
        # Check if model directory exists but not converted yet
        if os.path.exists(model_path) and not os.path.exists(os.path.join(model_path, "binary_conversion_complete")):
            logger.info(f"Model directory {model_path} exists but binary conversion not complete.")
            
            try:
                logger.info(f"Loading existing model from {model_path}...")
                pipe = StableDiffusionPipeline.from_pretrained(
                    model_path,
                    torch_dtype=torch.float32,
                    use_safetensors=True
                )
                return convert_to_binary_format(pipe, model_path)
            except Exception as e:
                logger.error(f"Error loading existing model: {e}")
                logger.info("Will try downloading fresh...")
        
        # Create model directory if needed
        os.makedirs(model_path, exist_ok=True)
        
        logger.info(f"Downloading model {model_id}...")
        
        # Check if model is SD 2.1 or other variant
        pipe = None
        if "stabilityai/stable-diffusion-2-1" in model_id:
            logger.info("Using Stable Diffusion 2.1 configuration...")
            pipe = StableDiffusionPipeline.from_pretrained(
                model_id,
                torch_dtype=torch.float32,
                use_safetensors=True,
                variant="fp16" if torch.cuda.is_available() else None
            )
        else:
            # Default configuration for other models
            pipe = StableDiffusionPipeline.from_pretrained(
                model_id,
                torch_dtype=torch.float32,
                use_safetensors=True
            )
        
        # Save the model locally
        logger.info(f"Saving model to {model_path}...")
        pipe.save_pretrained(model_path)
        
        # Convert to binary format for C inference
        logger.info(f"Converting model to binary format for C inference...")
        convert_success = convert_to_binary_format(pipe, model_path)
        
        if convert_success:
            logger.info("Model download and conversion complete!")
            return True
        else:
            logger.error("Failed to convert model to binary format")
            return False
        
    except Exception as e:
        logger.error(f"Error downloading/converting model: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    parser = argparse.ArgumentParser(description="Download and convert Stable Diffusion models")
    parser.add_argument("--model", type=str, default="runwayml/stable-diffusion-v1-5",
                        help="Model ID on Hugging Face (default: runwayml/stable-diffusion-v1-5)")
    parser.add_argument("--output", type=str, default="models",
                        help="Output directory (default: models)")
    parser.add_argument("--force-reconvert", action="store_true",
                        help="Force reconversion to binary format even if already done")
    
    args = parser.parse_args()
    
    # Handle force reconvert flag
    if args.force_reconvert:
        model_path = os.path.join(args.output, args.model.split("/")[-1] if "/" in args.model else args.model)
        binary_marker = os.path.join(model_path, "binary_conversion_complete")
        if os.path.exists(binary_marker):
            os.remove(binary_marker)
            logger.info(f"Forcing reconversion of model at {model_path}")
    
    success = download_model(args.model, args.output)
    
    if success:
        # Register with the Easy Diffusion API
        try:
            from easy_diffusion_api import EasyDiffusionAPI
            api = EasyDiffusionAPI()
            
            model_name = args.model.split("/")[-1] if "/" in args.model else args.model
            api.register_model(model_name, args.model)
            
            logger.info(f"Model '{model_name}' registered in the API")
        except Exception as e:
            logger.warning(f"Note: Couldn't register model with API: {e}")
            
        logger.info("Model download and conversion completed successfully")
        return True
    else:
        logger.error("Model download or conversion failed")
        return False

if __name__ == "__main__":
    main() 