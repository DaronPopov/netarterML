#!/usr/bin/env python3
"""
Script to download and prepare Dreamshaper v8 model
"""

import os
import sys
import torch
from diffusers import StableDiffusionPipeline
import numpy as np
from tqdm import tqdm

# Set Hugging Face token environment variable
os.environ["HF_TOKEN"] = "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"
os.environ["HUGGING_FACE_HUB_TOKEN"] = "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"

def download_and_convert_model():
    model_id = "Lykon/dreamshaper-8"
    model_path = "models/dreamshaper-8"
    
    print(f"Using Hugging Face token: {os.environ.get('HF_TOKEN', '')[:5]}***...")
    
    if not os.path.exists(model_path):
        os.makedirs(model_path, exist_ok=True)
        print(f"Downloading Dreamshaper v8 model to {model_path}...")
        
        # Download the model
        pipe = StableDiffusionPipeline.from_pretrained(
            model_id, 
            torch_dtype=torch.float32,
            use_auth_token=os.environ.get("HF_TOKEN")
        )
        
        # Save the model locally
        pipe.save_pretrained(model_path)
        print(f"Model saved to {model_path}")
        
        # Convert to float32 numpy arrays for C
        print("Converting model components to binary format...")
        
        # Convert UNet weights
        unet_weights = {}
        state_dict = pipe.unet.state_dict()
        for k, v in tqdm(state_dict.items(), desc="Converting UNet weights"):
            unet_weights[k] = v.cpu().numpy().astype(np.float32)
        
        # Convert VAE weights
        vae_weights = {}
        state_dict = pipe.vae.state_dict()
        for k, v in tqdm(state_dict.items(), desc="Converting VAE weights"):
            vae_weights[k] = v.cpu().numpy().astype(np.float32)
        
        # Convert text encoder weights
        text_encoder_weights = {}
        state_dict = pipe.text_encoder.state_dict()
        for k, v in tqdm(state_dict.items(), desc="Converting text encoder weights"):
            text_encoder_weights[k] = v.cpu().numpy().astype(np.float32)
        
        # Save weights in binary format for C
        print("Saving weights in binary format...")
        with open(os.path.join(model_path, "unet.bin"), "wb") as f:
            np.savez(f, **unet_weights)
        
        with open(os.path.join(model_path, "vae.bin"), "wb") as f:
            np.savez(f, **vae_weights)
        
        with open(os.path.join(model_path, "text_encoder.bin"), "wb") as f:
            np.savez(f, **text_encoder_weights)
        
        print("Model download and conversion complete!")
    else:
        print(f"Model directory {model_path} already exists.")
        
        # Check if binary files exist
        if not all(os.path.exists(os.path.join(model_path, f)) for f in ["unet.bin", "vae.bin", "text_encoder.bin"]):
            print("Binary model files are missing, downloading and converting...")
            # Remove directory and restart
            import shutil
            shutil.rmtree(model_path)
            download_and_convert_model()

if __name__ == "__main__":
    print("Starting Dreamshaper v8 model download...")
    download_and_convert_model()
    print("Done!") 