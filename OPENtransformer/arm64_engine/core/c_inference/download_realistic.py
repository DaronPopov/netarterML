#!/usr/bin/env python3
"""
Script to download and prepare PhotoReal V2 model, known for realistic people
"""

import os
import sys
import torch
from diffusers import StableDiffusionPipeline
import numpy as np
from tqdm import tqdm
from huggingface_hub import hf_hub_download
from pathlib import Path

# Ensure the project root is in the Python path
project_root = Path(__file__).resolve().parent.parent.parent.parent.parent
sys.path.append(str(project_root))

# Configuration
MODEL_NAME = "SG161222/Realistic_Vision_V5.1_noVAE"
VAE_NAME = "stabilityai/sd-vae-ft-mse"
MODEL_DIR = project_root / "models" / "Realistic_Vision_V5.1"
VAE_DIR = MODEL_DIR / "vae"

# Ensure model and VAE directories exist
MODEL_DIR.mkdir(parents=True, exist_ok=True)
VAE_DIR.mkdir(parents=True, exist_ok=True)

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
    "v1-5-pruned.safetensors"
]

# Files to download for the VAE
VAE_FILES = [
    "config.json",
    "diffusion_pytorch_model.bin"
]

def download_file(repo_id, filename, local_dir, token):
    """Downloads a file from Hugging Face Hub."""
    print(f"Downloading {filename} from {repo_id}...")
    try:
        hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=local_dir,
            local_dir_use_symlinks=False,
            token=token
        )
        print(f"Successfully downloaded {filename}")
    except Exception as e:
        print(f"Error downloading {filename}: {e}")

def main():
    print("Starting download of Realistic Vision V5.1 model and VAE...")

    # Download main model files
    for file in MODEL_FILES:
        # Determine the subfolder (if any)
        subfolder = ""
        if "/" in file:
            parts = file.split("/")
            subfolder = "/".join(parts[:-1])
            file_to_download = parts[-1]
            target_dir = MODEL_DIR / subfolder
            target_dir.mkdir(parents=True, exist_ok=True)
        else:
            file_to_download = file
            target_dir = MODEL_DIR
        
        download_file(MODEL_NAME, file, target_dir, HF_TOKEN)

    # Download VAE files
    for file in VAE_FILES:
        download_file(VAE_NAME, file, VAE_DIR, HF_TOKEN)

    print("Download process completed.")

if __name__ == "__main__":
    main() 