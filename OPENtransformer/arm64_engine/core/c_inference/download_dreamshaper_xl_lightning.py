#!/usr/bin/env python3
"""
Script to download and cache the Dreamshaper XL Lightning model.
This is an optimized SDXL model that produces high-quality results with fewer steps.
"""

import os
import sys
import time
import torch
import numpy as np
from diffusers import DiffusionPipeline, AutoencoderKL
import logging
import argparse
from tqdm import tqdm
import json
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def convert_sdxl_to_binary_format(pipe, model_path):
    """
    Convert SDXL model components to binary format for C/ASM inference
    
    Args:
        pipe: The DiffusionPipeline instance
        model_path: Path to save the binary files
    """
    print("Converting SDXL model components to binary format for C/ASM inference...")
    
    try:
        # Create component directories
        os.makedirs(os.path.join(model_path, "unet"), exist_ok=True)
        os.makedirs(os.path.join(model_path, "vae"), exist_ok=True)
        os.makedirs(os.path.join(model_path, "text_encoder"), exist_ok=True)
        os.makedirs(os.path.join(model_path, "text_encoder_2"), exist_ok=True)
        
        # Save UNet weights in binary format (most important)
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
        print("Converting text encoder 1 weights...")
        text_encoder_weights = {}
        state_dict = pipe.text_encoder.state_dict()
        for k, v in tqdm(state_dict.items(), desc="Converting text encoder 1 weights"):
            text_encoder_weights[k] = v.cpu().numpy().astype(np.float32)
        
        with open(os.path.join(model_path, "text_encoder.bin"), "wb") as f:
            np.savez(f, **text_encoder_weights)
            
        # Save text encoder 2 weights in binary format (SDXL specific)
        print("Converting text encoder 2 weights...")
        text_encoder_2_weights = {}
        state_dict = pipe.text_encoder_2.state_dict()
        for k, v in tqdm(state_dict.items(), desc="Converting text encoder 2 weights"):
            text_encoder_2_weights[k] = v.cpu().numpy().astype(np.float32)
        
        with open(os.path.join(model_path, "text_encoder_2.bin"), "wb") as f:
            np.savez(f, **text_encoder_2_weights)
        
        # Save configuration metadata for C inference
        print("Saving model configuration...")
        config = {
            "model_type": "sdxl",  # Mark as SDXL type
            "model_name": Path(model_path).name,
            "unet": {
                "in_channels": pipe.unet.config.in_channels,
                "out_channels": pipe.unet.config.out_channels,
                "sample_size": pipe.unet.config.sample_size,
                "cross_attention_dim": pipe.unet.config.cross_attention_dim
            },
            "vae": {
                "latent_channels": pipe.vae.config.latent_channels,
                "scaling_factor": pipe.vae.config.scaling_factor
            },
            "text_encoder": {
                "hidden_size": pipe.text_encoder.config.hidden_size,
                "num_attention_heads": pipe.text_encoder.config.num_attention_heads,
                "vocab_size": pipe.text_encoder.config.vocab_size
            },
            "text_encoder_2": {
                "hidden_size": pipe.text_encoder_2.config.hidden_size,
                "num_attention_heads": pipe.text_encoder_2.config.num_attention_heads,
                "vocab_size": pipe.text_encoder_2.config.vocab_size
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

def download_model(model_id="Lykon/dreamshaper-xl-lightning", output_dir="models", name=None):
    """Download and cache the Dreamshaper XL Lightning model."""
    try:
        # Create models directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Use provided name or extract from model_id
        if name:
            model_name = name
        else:
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
                pipe = DiffusionPipeline.from_pretrained(
                    model_path,
                    torch_dtype=torch.float32
                )
                return convert_sdxl_to_binary_format(pipe, model_path)
            except Exception as e:
                logger.error(f"Error loading existing model: {e}")
                logger.info("Will try downloading fresh...")
        
        # Create model directory if needed
        os.makedirs(model_path, exist_ok=True)
        
        logger.info(f"Downloading model {model_id}...")
        
        # Download and load the model - Dreamshaper XL Lightning is an SDXL model
        logger.info("Using SDXL Lightning configuration...")
        pipe = DiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float32,
            use_safetensors=True
        )
        
        # Save the model locally
        logger.info(f"Saving model to {model_path}...")
        pipe.save_pretrained(model_path)
        
        # Convert to binary format for C inference
        logger.info(f"Converting model to binary format for C inference...")
        convert_success = convert_sdxl_to_binary_format(pipe, model_path)
        
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
    parser = argparse.ArgumentParser(description="Download and convert the Dreamshaper XL Lightning model")
    parser.add_argument("--model", type=str, default="Lykon/dreamshaper-xl-lightning",
                        help="Model ID on Hugging Face (default: Lykon/dreamshaper-xl-lightning)")
    parser.add_argument("--output", type=str, default="models",
                        help="Output directory (default: models)")
    parser.add_argument("--name", type=str, default="dreamshaper-xl-lightning",
                        help="Local name for the model (default: dreamshaper-xl-lightning)")
    parser.add_argument("--force-reconvert", action="store_true",
                        help="Force reconversion to binary format even if already done")
    
    args = parser.parse_args()
    
    # Handle force reconvert flag
    if args.force_reconvert:
        model_path = os.path.join(args.output, args.name)
        binary_marker = os.path.join(model_path, "binary_conversion_complete")
        if os.path.exists(binary_marker):
            os.remove(binary_marker)
            logger.info(f"Forcing reconversion of model at {model_path}")
    
    success = download_model(args.model, args.output, args.name)
    
    if success:
        # Register with the Easy Diffusion API
        try:
            from easy_diffusion_api import EasyDiffusionAPI
            api = EasyDiffusionAPI()
            
            api.register_model(args.name, args.model)
            
            logger.info(f"Model '{args.name}' registered in the API")
        except Exception as e:
            logger.warning(f"Note: Couldn't register model with API: {e}")
            
        logger.info("Model download and conversion completed successfully")
        return True
    else:
        logger.error("Model download or conversion failed")
        return False

if __name__ == "__main__":
    main() 