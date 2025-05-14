#!/usr/bin/env python3
"""
Script to download and prepare Dreamshaper v8 model
"""

import os
import sys
from huggingface_hub import hf_hub_download
from pathlib import Path
import torch
from diffusers import StableDiffusionPipeline
import numpy as np
from tqdm import tqdm

# Ensure the project root is in the Python path
project_root = Path(__file__).resolve().parent.parent.parent.parent.parent
sys.path.append(str(project_root))

# Configuration
MODEL_NAME = "Lykon/dreamshaper-8"
MODEL_DIR = project_root / "models" / "dreamshaper-8"

# Ensure model directory exists
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Get Hugging Face token from environment variable
HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
if not HF_TOKEN:
    print("Error: HUGGINGFACE_TOKEN environment variable not set.")
    print("Please set it before running the script: export HUGGINGFACE_TOKEN='your_token_here'")
    # sys.exit(1) # Optionally exit if token is crucial

# Files to download for the main model
MODEL_FILES = [
    "model_index.json",
    "scheduler/scheduler_config.json",
    "text_encoder/config.json",
    "text_encoder/pytorch_model.bin",
    "tokenizer/merges.txt",
    "tokenizer/special_tokens_map.json",
    "tokenizer/tokenizer_config.json",
    "tokenizer/vocab.json",
    "unet/config.json",
    "unet/diffusion_pytorch_model.bin",
    "vae/config.json",
    "vae/diffusion_pytorch_model.bin",
    "dreamshaper-8.safetensors"
]

def download_file(repo_id, filename, local_dir, token):
    """Downloads a file from Hugging Face Hub."""
    print(f"Downloading {filename} from {repo_id}...")
    try:
        # Determine subfolder from filename if present
        subfolder = ""
        file_to_dl = filename
        if "/" in filename:
            parts = filename.split("/")
            subfolder = "/".join(parts[:-1])
            file_to_dl = parts[-1]
            dl_target_dir = Path(local_dir) / subfolder
            dl_target_dir.mkdir(parents=True, exist_ok=True)
        else:
            dl_target_dir = Path(local_dir)

        hf_hub_download(
            repo_id=repo_id,
            filename=filename, # Use original filename for repo path
            local_dir=dl_target_dir,
            local_dir_use_symlinks=False,
            token=token
        )
        print(f"Successfully downloaded {filename} to {dl_target_dir}")
    except Exception as e:
        print(f"Error downloading {filename}: {e}")

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

def main():
    print("Starting download of Dreamshaper 8 model...")

    # Download main model files
    for file_path_in_repo in MODEL_FILES:
        download_file(MODEL_NAME, file_path_in_repo, MODEL_DIR, HF_TOKEN)

    print("Download process completed.")

if __name__ == "__main__":
    main() 