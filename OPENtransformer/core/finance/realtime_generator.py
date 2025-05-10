import torch
import numpy as np
from typing import List, Optional
import time
import sys
from datetime import datetime
from .transformer_inference import TransformerModel, TransformerInference

class RealtimeGenerator:
    def __init__(self, model_path: str):
        """Initialize the real-time generator with a transformer model."""
        self.transformer = TransformerInference(model_path)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def generate_stream(
        self,
        prompt: str,
        max_tokens: int = 100,
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.9,
        stream_delay: float = 0.05
    ):
        """
        Generate text in real-time, streaming each token as it's generated.
        
        Args:
            prompt: Initial text prompt
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (higher = more random)
            top_k: Top-k sampling parameter
            top_p: Top-p sampling parameter
            stream_delay: Delay between token generations (seconds)
        """
        # Convert prompt to tensor and ensure float dtype
        input_ids = torch.tensor([ord(c) for c in prompt], dtype=torch.float32).to(self.device)
        
        print(f"\nGenerating from prompt: {prompt}\n")
        print("Generated text: ", end="", flush=True)
        
        for _ in range(max_tokens):
            # Get model predictions
            with torch.no_grad():
                outputs = self.transformer.model(input_ids)
                next_token_logits = outputs[:, -1, :] / temperature
                
                # Apply top-k and top-p sampling
                top_k_logits, top_k_indices = torch.topk(next_token_logits, min(top_k, next_token_logits.size(-1)))
                probs = torch.softmax(top_k_logits, dim=-1)
                
                # Sample from the filtered distribution
                next_token_idx = torch.multinomial(probs, 1)
                next_token = top_k_indices[0, next_token_idx[0]]
                
                # Convert token to character and print
                char = chr(int(next_token.item()))
                print(char, end="", flush=True)
                
                # Append token to input (ensure float dtype)
                next_token = next_token.unsqueeze(0).float()
                input_ids = torch.cat([input_ids, next_token])
                
                # Add delay for real-time effect
                time.sleep(stream_delay)
                
                # Check for end of sequence or special tokens
                if char in ['\n', '.', '!', '?']:
                    break
        
        print("\n")

def main():
    """Main function to run the real-time generator."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Real-time text generation with transformer model")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model weights")
    parser.add_argument("--prompt", type=str, default="Once upon a time", help="Initial text prompt")
    parser.add_argument("--max_tokens", type=int, default=100, help="Maximum number of tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--top_k", type=int, default=50, help="Top-k sampling parameter")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p sampling parameter")
    parser.add_argument("--stream_delay", type=float, default=0.05, help="Delay between token generations")
    
    args = parser.parse_args()
    
    generator = RealtimeGenerator(args.model_path)
    generator.generate_stream(
        prompt=args.prompt,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        stream_delay=args.stream_delay
    )

if __name__ == "__main__":
    main() 