#!/usr/bin/env python3
"""
Easy Diffusion API - A simple API for hot-swapping diffusion models

This API provides an easy interface to work with the C/ASM backend inference 
for diffusion models. It allows loading models either from Hugging Face using
a token or from a local path.
"""

import os
import sys
import time
import argparse
import json
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Union, Any

# Add the current directory to the Python path to ensure imports work
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

class DiffusionModel:
    """Class representing a loaded diffusion model"""
    
    def __init__(self, model_id: str, model_path: Optional[str] = None):
        """
        Initialize a diffusion model reference
        
        Args:
            model_id: Identifier for the model
            model_path: Path to local model weights if available
        """
        self.model_id = model_id
        self.model_path = model_path or model_id
        self.is_loaded = False
        
    def __repr__(self) -> str:
        return f"DiffusionModel({self.model_id}, loaded={self.is_loaded})"


class EasyDiffusionAPI:
    """
    Easy-to-use API for diffusion model management and inference
    """
    
    def __init__(self, cache_dir: Optional[str] = None, use_hf_token: bool = True):
        """
        Initialize the diffusion API
        
        Args:
            cache_dir: Directory to cache models (defaults to ./models)
            use_hf_token: Whether to use the Hugging Face token from environment
        """
        self.cache_dir = cache_dir or os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
        self.models: Dict[str, DiffusionModel] = {}
        self.active_model: Optional[DiffusionModel] = None
        self.config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "diffusion_config.json")
        
        # Create cache directory if it doesn't exist
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Set Hugging Face token if requested
        if use_hf_token:
            self._set_hf_token()
        
        # Load saved configuration if available
        self._load_config()
    
    def _set_hf_token(self):
        """Set the Hugging Face token from environment or token file"""
        
        # Check environment variable first
        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        
        # If not found, try to read from token file
        if not token:
            token_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".hf_token")
            if os.path.exists(token_path):
                with open(token_path, "r") as f:
                    token = f.read().strip()
        
        # If found, set the environment variables
        if token:
            os.environ["HF_TOKEN"] = token
            os.environ["HUGGING_FACE_HUB_TOKEN"] = token
            print(f"Hugging Face token set: {token[:5]}***...")
        else:
            print("Warning: No Hugging Face token found.")
            print("Set HF_TOKEN environment variable or create a .hf_token file.")
    
    def _load_config(self):
        """Load saved configuration if available"""
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, "r") as f:
                    config = json.load(f)
                
                # Load models
                for model_id, model_data in config.get("models", {}).items():
                    self.models[model_id] = DiffusionModel(
                        model_id=model_id,
                        model_path=model_data.get("path")
                    )
                
                # Set active model if it exists
                active_model_id = config.get("active_model")
                if active_model_id and active_model_id in self.models:
                    self.active_model = self.models[active_model_id]
                    self.active_model.is_loaded = True
                
                print(f"Loaded {len(self.models)} models from configuration")
                
            except Exception as e:
                print(f"Error loading configuration: {e}")
    
    def _save_config(self):
        """Save current configuration"""
        config = {
            "models": {
                model_id: {
                    "path": model.model_path
                } for model_id, model in self.models.items()
            },
            "active_model": self.active_model.model_id if self.active_model else None
        }
        
        try:
            with open(self.config_path, "w") as f:
                json.dump(config, f, indent=2)
            print("Configuration saved successfully")
        except Exception as e:
            print(f"Error saving configuration: {e}")
    
    def register_model(self, model_id: str, model_path: Optional[str] = None) -> DiffusionModel:
        """
        Register a model for use with the API
        
        Args:
            model_id: Identifier for the model
            model_path: Path to local model weights if available
        
        Returns:
            The registered model object
        """
        model = DiffusionModel(model_id=model_id, model_path=model_path)
        self.models[model_id] = model
        self._save_config()
        return model
    
    def set_active_model(self, model_id: str) -> bool:
        """
        Set the active model for inference
        
        Args:
            model_id: Identifier for the model to activate
        
        Returns:
            True if successful, False otherwise
        """
        if model_id not in self.models:
            print(f"Model '{model_id}' is not registered")
            return False
        
        self.active_model = self.models[model_id]
        self.active_model.is_loaded = True
        self._save_config()
        print(f"Activated model: {model_id}")
        return True
    
    def list_models(self) -> List[Dict[str, Any]]:
        """
        List all registered models
        
        Returns:
            List of model information dictionaries
        """
        return [
            {
                "id": model_id,
                "path": model.model_path,
                "active": model == self.active_model,
                "loaded": model.is_loaded
            }
            for model_id, model in self.models.items()
        ]
    
    def generate_image(
        self,
        prompt: str,
        model_id: Optional[str] = None,
        steps: int = 25,
        guidance: float = 7.5,
        width: int = 512,
        height: int = 512,
        seed: int = 0,
        output_path: Optional[str] = None
    ) -> Optional[Tuple[int, int, int, bytes]]:
        """
        Generate an image using the active model or specified model
        
        Args:
            prompt: Text prompt for image generation
            model_id: Optional model ID to use (uses active model if None)
            steps: Number of inference steps
            guidance: Guidance scale
            width: Output image width
            height: Output image height
            seed: Random seed (0 = random)
            output_path: Path to save the image (optional)
        
        Returns:
            Tuple of (width, height, channels, image_data) or None if failed
        """
        try:
            from py_diffusion_interface import run_inference
        except ImportError as e:
            print(f"Error importing py_diffusion_interface: {e}")
            print("This function requires the C/ASM backend. Please ensure it is installed and available.")
            return None
        
        # Determine which model to use
        model = None
        if model_id is not None:
            if model_id in self.models:
                model = self.models[model_id]
            else:
                print(f"Model '{model_id}' is not registered")
                return None
        else:
            model = self.active_model
        
        # Check if we have a model
        if model is None:
            print("No active model. Please set an active model first.")
            return None
        
        try:
            # Generate the image
            start_time = time.time()
            print(f"Generating image with model: {model.model_id}")
            print(f"Prompt: {prompt}")
            
            # Call the C/ASM backend through the Python interface
            result = run_inference(
                model.model_path,
                prompt,
                steps,
                width,
                guidance,
                height,
                seed,
                True,  # use_memory_optimizations
                None,  # callback_ptr
                None   # user_data_ptr
            )
            
            if result is None:
                print("Image generation failed")
                return None
            
            width, height, channels, img_data = result
            
            # Save the image if requested
            if output_path:
                output_dir = os.path.dirname(os.path.abspath(output_path))
                os.makedirs(output_dir, exist_ok=True)
                
                with open(output_path, "wb") as f:
                    f.write(img_data)
                print(f"Image saved to {output_path}")
            
            total_time = time.time() - start_time
            print(f"Generation completed in {total_time:.2f} seconds")
            
            # Update model status
            model.is_loaded = True
            
            return result
            
        except Exception as e:
            import traceback
            print(f"Error during image generation: {e}")
            traceback.print_exc()
            return None
    
    def download_model(self, model_id: str, force: bool = False) -> bool:
        """
        Download a model from Hugging Face
        
        Args:
            model_id: Hugging Face model ID
            force: Whether to force re-download if already exists
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Check if we have a token
            token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
            if not token:
                print("No Hugging Face token found. Set HF_TOKEN or use set_hf_token().")
                return False
            
            # Import the download module
            sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
            from download_model import download_model
            
            # Download the model
            print(f"Downloading model {model_id}...")
            success = download_model(model_id=model_id, output_dir=self.cache_dir)
            
            if success:
                # Register the model with the local path
                model_path = os.path.join(self.cache_dir, os.path.basename(model_id))
                self.register_model(model_id=model_id, model_path=model_path)
                print(f"Model {model_id} downloaded and registered")
                return True
            else:
                print(f"Failed to download model {model_id}")
                return False
                
        except Exception as e:
            import traceback
            print(f"Error downloading model: {e}")
            traceback.print_exc()
            return False
            

def main():
    """Command-line interface for the Easy Diffusion API"""
    parser = argparse.ArgumentParser(description="Easy Diffusion API")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # List models
    list_parser = subparsers.add_parser("list", help="List registered models")
    
    # Register model
    register_parser = subparsers.add_parser("register", help="Register a model")
    register_parser.add_argument("model_id", help="Model ID or name")
    register_parser.add_argument("--path", help="Path to local model weights")
    
    # Set active model
    activate_parser = subparsers.add_parser("activate", help="Set the active model")
    activate_parser.add_argument("model_id", help="Model ID to activate")
    
    # Generate image
    generate_parser = subparsers.add_parser("generate", help="Generate an image")
    generate_parser.add_argument("prompt", help="Text prompt for image generation")
    generate_parser.add_argument("--model", help="Model ID to use (uses active model if not specified)")
    generate_parser.add_argument("--steps", type=int, default=25, help="Number of inference steps")
    generate_parser.add_argument("--guidance", type=float, default=7.5, help="Guidance scale")
    generate_parser.add_argument("--width", type=int, default=512, help="Output image width")
    generate_parser.add_argument("--height", type=int, default=512, help="Output image height")
    generate_parser.add_argument("--seed", type=int, default=0, help="Random seed (0 = random)")
    generate_parser.add_argument("--output", help="Path to save the image")
    
    # Download model
    download_parser = subparsers.add_parser("download", help="Download a model from Hugging Face")
    download_parser.add_argument("model_id", help="Hugging Face model ID")
    download_parser.add_argument("--force", action="store_true", help="Force re-download if already exists")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Create API instance
    api = EasyDiffusionAPI()
    
    # Execute command
    if args.command == "list":
        models = api.list_models()
        print(f"Registered models ({len(models)}):")
        for model in models:
            active = " (ACTIVE)" if model["active"] else ""
            loaded = " (LOADED)" if model["loaded"] else ""
            print(f"  - {model['id']}{active}{loaded}: {model['path']}")
    
    elif args.command == "register":
        model = api.register_model(args.model_id, args.path)
        print(f"Registered model: {model.model_id}")
        if args.path:
            print(f"  Path: {args.path}")
    
    elif args.command == "activate":
        success = api.set_active_model(args.model_id)
        if success:
            print(f"Activated model: {args.model_id}")
        else:
            print(f"Failed to activate model: {args.model_id}")
            sys.exit(1)
    
    elif args.command == "generate":
        result = api.generate_image(
            prompt=args.prompt,
            model_id=args.model,
            steps=args.steps,
            guidance=args.guidance,
            width=args.width,
            height=args.height,
            seed=args.seed,
            output_path=args.output
        )
        
        if result is None:
            print("Image generation failed")
            sys.exit(1)
    
    elif args.command == "download":
        success = api.download_model(args.model_id, args.force)
        if success:
            print(f"Successfully downloaded model: {args.model_id}")
        else:
            print(f"Failed to download model: {args.model_id}")
            sys.exit(1)
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main() 