#!/usr/bin/env python3
"""
Script to download and prepare Stable Diffusion 2.1 model
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
MODEL_NAME = "stabilityai/stable-diffusion-2-1-base"
MODEL_DIR = project_root / "models" / "stable-diffusion-2-1-base"

# Ensure model directory exists
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Get Hugging Face token from environment variable
HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
if not HF_TOKEN:
    print("Error: HUGGINGFACE_TOKEN environment variable not set.")
    print("Please set it before running the script: export HUGGINGFACE_TOKEN='your_token_here'")
    # sys.exit(1) # Optionally exit if token is crucial

# Files to download (based on a typical SD 2.1 base structure)
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
    "v2-1_512-ema-pruned.safetensors" # Common single-file checkpoint if used
]

def download_file(repo_id, filename, local_dir, token):
    """Downloads a file from Hugging Face Hub."""
    print(f"Downloading {filename} from {repo_id}...")
    try:
        # Determine subfolder from filename if present
        subfolder = ""
        if "/" in filename:
            parts = filename.split("/")
            subfolder = "/".join(parts[:-1])
            file_to_dl = parts[-1]
            dl_target_dir = Path(local_dir) / subfolder
            dl_target_dir.mkdir(parents=True, exist_ok=True)
        else:
            file_to_dl = filename
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

def main():
    print(f"Starting download of {MODEL_NAME} model...")

    # Download main model files
    for file_path_in_repo in MODEL_FILES:
        # Check if the file is the .safetensors file, which might be at the root
        if file_path_in_repo == "v2-1_512-ema-pruned.safetensors":
            # For safetensors, download to the MODEL_DIR root, not a subfolder
             download_file(MODEL_NAME, file_path_in_repo, MODEL_DIR, HF_TOKEN)
        else:
            download_file(MODEL_NAME, file_path_in_repo, MODEL_DIR, HF_TOKEN)

    print("Download process completed.")

if __name__ == "__main__":
    main() 