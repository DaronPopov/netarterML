#!/usr/bin/env python3
"""
Script to convert model weights from safetensors/pt to binary format for C inference.
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path
import logging
from safetensors import safe_open

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def convert_weights(input_path, output_path):
    """Convert model weights to binary format."""
    try:
        # Load state dict
        logger.info(f"Loading weights from {input_path}")
        
        # Handle both safetensors and pytorch formats
        if input_path.endswith('.safetensors'):
            tensors = {}
            with safe_open(input_path, framework="pt", device="cpu") as f:
                for key in f.keys():
                    tensors[key] = f.get_tensor(key)
            state_dict = tensors
        else:
            state_dict = torch.load(input_path, map_location='cpu')
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Open output file
        with open(output_path, 'wb') as f:
            # Write number of tensors
            num_tensors = len(state_dict)
            f.write(np.array(num_tensors, dtype=np.int64).tobytes())
            
            # Write each tensor
            for name, tensor in state_dict.items():
                # Convert to float32 numpy array
                array = tensor.cpu().numpy().astype(np.float32)
                
                # Write metadata
                ndim = len(array.shape)
                f.write(np.array(ndim, dtype=np.int32).tobytes())
                f.write(np.array(array.shape, dtype=np.int32).tobytes())
                
                # Write name
                name_bytes = name.encode('utf-8')
                f.write(np.array(len(name_bytes), dtype=np.uint64).tobytes())
                f.write(name_bytes)
                
                # Write tensor data
                f.write(array.tobytes())
                
                logger.info(f"Converted tensor {name} with shape {array.shape}")
        
        logger.info(f"Successfully converted weights to {output_path}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to convert weights: {e}")
        return False

def main():
    if len(sys.argv) != 3:
        print("Usage: convert_weights.py <input_path> <output_path>")
        sys.exit(1)
    
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    
    if not os.path.exists(input_path):
        print(f"Error: Input file {input_path} does not exist")
        sys.exit(1)
    
    if convert_weights(input_path, output_path):
        sys.exit(0)
    else:
        sys.exit(1)

if __name__ == "__main__":
    main() 