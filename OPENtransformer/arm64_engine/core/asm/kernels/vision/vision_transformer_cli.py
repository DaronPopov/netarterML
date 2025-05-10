#!/usr/bin/env python3

import argparse
import numpy as np
import logging
from pathlib import Path
import sys
import os

# Add the project root to Python path
project_root = str(Path(__file__).parent.parent.parent.parent.parent.absolute())
sys.path.insert(0, project_root)

from OPENtransformer.arm64_engine.core.asm.kernels.vision.vision_transformer import VisionTransformer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_model(args):
    """Create a new Vision Transformer model"""
    model = VisionTransformer(
        image_size=args.image_size,
        patch_size=args.patch_size,
        num_channels=args.num_channels,
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        num_classes=args.num_classes,
        dropout=args.dropout,
        use_2d_pos_emb=args.use_2d_pos_emb
    )
    logger.info(f"Created Vision Transformer model with {args.num_layers} layers")
    return model

def process_image(model, image_path):
    """Process a single image through the model"""
    try:
        # Load and preprocess image
        # Note: This is a placeholder - you'll need to implement proper image loading
        # and preprocessing based on your requirements
        image = np.random.randn(1, model.num_channels, model.image_size, model.image_size).astype(np.float32)
        
        # Run forward pass
        output = model.forward(image)
        
        # Get top predictions
        top_k = 5
        top_indices = np.argsort(output[0])[-top_k:][::-1]
        top_probs = output[0][top_indices]
        
        logger.info(f"Top {top_k} predictions for {image_path}:")
        for idx, prob in zip(top_indices, top_probs):
            logger.info(f"Class {idx}: {prob:.4f}")
            
    except Exception as e:
        logger.error(f"Error processing image {image_path}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Vision Transformer Command Line Interface")
    
    # Model configuration arguments
    parser.add_argument("--image-size", type=int, default=224, help="Input image size")
    parser.add_argument("--patch-size", type=int, default=16, help="Size of patches")
    parser.add_argument("--num-channels", type=int, default=3, help="Number of input channels")
    parser.add_argument("--embed-dim", type=int, default=768, help="Model dimension")
    parser.add_argument("--num-heads", type=int, default=12, help="Number of attention heads")
    parser.add_argument("--num-layers", type=int, default=12, help="Number of transformer layers")
    parser.add_argument("--num-classes", type=int, default=1000, help="Number of output classes")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    parser.add_argument("--use-2d-pos-emb", action="store_true", help="Use 2D positional embeddings")
    
    # Command subparsers
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Create model command
    create_parser = subparsers.add_parser("create", help="Create a new model")
    create_parser.add_argument("--save", type=str, help="Path to save model weights")
    
    # Process image command
    process_parser = subparsers.add_parser("process", help="Process an image")
    process_parser.add_argument("image_path", type=str, help="Path to the image to process")
    process_parser.add_argument("--model-path", type=str, help="Path to load model weights from")
    
    # Load model command
    load_parser = subparsers.add_parser("load", help="Load model weights")
    load_parser.add_argument("model_path", type=str, help="Path to load model weights from")
    
    args = parser.parse_args()
    
    if args.command == "create":
        model = create_model(args)
        if args.save:
            model.save(args.save)
            logger.info(f"Model saved to {args.save}")
            
    elif args.command == "process":
        if args.model_path:
            model = VisionTransformer()
            model.load(args.model_path)
            logger.info(f"Loaded model from {args.model_path}")
        else:
            model = create_model(args)
            
        process_image(model, args.image_path)
        
    elif args.command == "load":
        model = VisionTransformer()
        model.load(args.model_path)
        logger.info(f"Loaded model from {args.model_path}")
        
    else:
        parser.print_help()

if __name__ == "__main__":
    main() 