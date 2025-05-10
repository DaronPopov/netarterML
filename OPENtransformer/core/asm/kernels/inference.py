import logging
from OPENtransformer.core.asm.kernels.transformer import Transformer
import argparse
import json
import os
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description='Run inference with the transformer model')
    parser.add_argument('--model_path', type=str, required=True, help='Path to converted model directory')
    parser.add_argument('--prompt', type=str, default='The quick brown fox', help='Input prompt for generation')
    parser.add_argument('--max_length', type=int, default=100, help='Maximum length of generated text')
    parser.add_argument('--temperature', type=float, default=0.7, help='Sampling temperature')
    parser.add_argument('--top_k', type=int, default=50, help='Number of top tokens to consider')
    
    args = parser.parse_args()
    
    try:
        # Load model configuration
        config_path = os.path.join(args.model_path, 'config.json')
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found at {config_path}")
            
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Initialize model
        logger.info("Loading converted model...")
        model = Transformer(
            d_model=config['d_model'],
            n_heads=config['n_heads'],
            n_layers=config['n_layers'],
            vocab_size=config['vocab_size']
        )
        
        # Load model weights and embeddings
        model.load(args.model_path)
        
        # Generate text
        logger.info(f"Generating text from prompt: {args.prompt}")
        generated_text = model.generate(
            prompt=args.prompt,
            max_length=args.max_length,
            temperature=args.temperature,
            top_k=args.top_k
        )
        
        logger.info("\nGenerated text:")
        logger.info("-" * 50)
        logger.info(generated_text)
        logger.info("-" * 50)
        
    except Exception as e:
        logger.error(f"Error during inference: {str(e)}")
        raise

if __name__ == "__main__":
    main() 