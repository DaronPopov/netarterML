#!/usr/bin/env python3
"""
Script to download and prepare SDXL models for inference
"""

import os
import sys
import torch
import numpy as np
from diffusers import StableDiffusionXLPipeline
import argparse
from pathlib import Path
import json
from tqdm import tqdm

# Set Hugging Face token environment variable
os.environ["HF_TOKEN"] = "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"
os.environ["HUGGING_FACE_HUB_TOKEN"] = "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"

def convert_to_binary_format(pipe, model_path):
    """
    Convert model components to binary format for C inference
    
    Args:
        pipe: The StableDiffusionXLPipeline instance
        model_path: Path to save the binary files
    """
    print("Converting model components to binary format for C/ASM inference...")
    
    try:
        # Create component directories
        os.makedirs(os.path.join(model_path, "unet"), exist_ok=True)
        os.makedirs(os.path.join(model_path, "vae"), exist_ok=True)
        os.makedirs(os.path.join(model_path, "text_encoder"), exist_ok=True)
        os.makedirs(os.path.join(model_path, "text_encoder_2"), exist_ok=True)
        
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
        
        # Save text encoder 2 weights (SDXL specific)
        print("Converting text encoder 2 weights (SDXL)...")
        text_encoder_2_weights = {}
        state_dict = pipe.text_encoder_2.state_dict()
        for k, v in tqdm(state_dict.items(), desc="Converting text encoder 2 weights"):
            text_encoder_2_weights[k] = v.cpu().numpy().astype(np.float32)
        
        with open(os.path.join(model_path, "text_encoder_2.bin"), "wb") as f:
            np.savez(f, **text_encoder_2_weights)
        
        # Save configuration metadata for C inference
        print("Saving model configuration...")
        config = {
            "model_type": "sdxl",
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

def download_and_convert_model(model_id="stabilityai/stable-diffusion-xl-base-1.0", local_name=None):
    """
    Download and convert an SDXL model for inference
    
    Args:
        model_id: Hugging Face model ID or path
        local_name: Local name to save the model under (defaults to last part of model_id)
    """
    if local_name is None:
        local_name = model_id.split('/')[-1]
    
    model_path = f"models/{local_name}"
    
    print(f"Using Hugging Face token: {os.environ.get('HF_TOKEN', '')[:5]}***...")
    
    # Check if the binary conversion is already complete
    if os.path.exists(os.path.join(model_path, "binary_conversion_complete")):
        print(f"Model {model_id} already exists and is converted to binary format.")
        return True
    
    # Check if model directory exists but not converted yet
    if os.path.exists(model_path) and not os.path.exists(os.path.join(model_path, "binary_conversion_complete")):
        print(f"Model directory {model_path} exists but binary conversion not complete.")
        
        try:
            print(f"Loading existing model from {model_path}...")
            pipe = StableDiffusionXLPipeline.from_pretrained(
                model_path,
                torch_dtype=torch.float32,
                use_safetensors=True
            )
            return convert_to_binary_format(pipe, model_path)
        except Exception as e:
            print(f"Error loading existing model: {e}")
            print("Will try downloading fresh...")
    
    # Create model directory if it doesn't exist
    os.makedirs(model_path, exist_ok=True)
    print(f"Downloading SDXL model {model_id} to {model_path}...")
    
    # Download the model
    try:
        pipe = StableDiffusionXLPipeline.from_pretrained(
            model_id, 
            torch_dtype=torch.float32,
            use_auth_token=os.environ.get("HF_TOKEN"),
            use_safetensors=True
        )
        
        # Save the model locally
        print(f"Saving model to {model_path}...")
        pipe.save_pretrained(model_path)
        print(f"Model saved to {model_path}")
        
        # Convert to binary format for C inference
        return convert_to_binary_format(pipe, model_path)
        
    except Exception as e:
        import traceback
        print(f"Error downloading model: {e}")
        traceback.print_exc()
        return False

def main():
    parser = argparse.ArgumentParser(description="Download SDXL models for inference")
    parser.add_argument("--model", type=str, default="stabilityai/stable-diffusion-xl-base-1.0",
                        help="Model ID on Hugging Face or local path")
    parser.add_argument("--name", type=str, default=None,
                        help="Local name to save the model under")
    parser.add_argument("--force-reconvert", action="store_true",
                        help="Force reconversion to binary format even if already done")
    
    args = parser.parse_args()
    
    # Create models directory
    os.makedirs("models", exist_ok=True)
    
    # Handle force reconvert flag
    if args.force_reconvert:
        model_path = f"models/{args.name if args.name else args.model.split('/')[-1]}"
        binary_marker = os.path.join(model_path, "binary_conversion_complete")
        if os.path.exists(binary_marker):
            os.remove(binary_marker)
            print(f"Forcing reconversion of model at {model_path}")
    
    # Download the model
    success = download_and_convert_model(args.model, args.name)
    
    if success:
        print("SDXL model download and conversion completed successfully!")
        
        # Register with the Easy Diffusion API
        try:
            from easy_diffusion_api import EasyDiffusionAPI
            api = EasyDiffusionAPI()
            
            model_id = args.name if args.name else args.model.split('/')[-1]
            api.register_model(model_id, args.model)
            api.set_active_model(model_id)
            
            print(f"Model '{model_id}' registered and set as active in the API")
        except Exception as e:
            print(f"Note: Couldn't register model with API: {e}")
    else:
        print("Failed to download or convert the model")
        sys.exit(1)

if __name__ == "__main__":
    print("Starting SDXL model download and conversion...")
    main()
    print("Done!") 