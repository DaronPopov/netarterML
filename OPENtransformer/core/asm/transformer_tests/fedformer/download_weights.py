import os
import torch
import requests
from tqdm import tqdm
import logging
from pathlib import Path
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def download_file(url: str, destination: str, chunk_size: int = 8192):
    """Download a file with progress bar."""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(destination, 'wb') as f, tqdm(
        desc=os.path.basename(destination),
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as pbar:
        for data in response.iter_content(chunk_size=chunk_size):
            size = f.write(data)
            pbar.update(size)

def format_weights_for_kernel(weights: dict) -> dict:
    """Format model weights for kernel usage."""
    formatted_weights = {}
    
    # Process each layer's weights
    for name, param in weights.items():
        try:
            # Convert to numpy array
            if isinstance(param, torch.Tensor):
                param = param.detach().cpu().numpy()
            
            # Handle different types of layers
            if 'weight' in name:
                if len(param.shape) == 2:  # Linear layer weights
                    formatted_weights[name] = param.T  # Transpose for kernel format
                elif len(param.shape) == 3:  # Conv1d layer weights
                    formatted_weights[name] = param.transpose(1, 2, 0)  # [out_channels, kernel_size, in_channels]
            elif 'bias' in name:
                formatted_weights[name] = param
            
            # Handle attention weights
            if 'attention' in name:
                if 'weight' in name:
                    formatted_weights[name] = param.transpose(0, 1)  # [num_heads, head_dim]
        except Exception as e:
            logger.warning(f"Error processing weight {name}: {str(e)}")
            continue
    
    return formatted_weights

def download_fedformer_weights():
    """Download and format FEDformer model weights for kernel usage."""
    # Create weights directory if it doesn't exist
    weights_dir = Path(__file__).parent / 'weights'
    weights_dir.mkdir(exist_ok=True)
    
    # Model weights URL (using the official FEDformer repository)
    weights_url = "https://huggingface.co/MAZiqing/FEDformer/resolve/main/fedformer_financial.pt"
    
    weights_path = weights_dir / 'fedformer_weights.pt'
    kernel_weights_path = weights_dir / 'fedformer_kernel_weights.npz'
    
    if kernel_weights_path.exists():
        logger.info(f"Kernel-formatted weights already exist at {kernel_weights_path}")
        return kernel_weights_path
    
    try:
        # Download weights if not exists
        if not weights_path.exists():
            logger.info("Downloading FEDformer model weights...")
            download_file(weights_url, str(weights_path))
            logger.info(f"Successfully downloaded weights to {weights_path}")
        
        # Load and format weights
        logger.info("Loading and formatting weights for kernel...")
        weights = torch.load(weights_path, map_location='cpu', weights_only=False)
        
        # Extract model state dict
        if isinstance(weights, dict) and 'model_state_dict' in weights:
            weights = weights['model_state_dict']
        elif isinstance(weights, dict):
            weights = weights
        else:
            raise ValueError("Invalid weights format")
        
        # Format weights for kernel
        formatted_weights = format_weights_for_kernel(weights)
        
        # Save formatted weights
        np.savez_compressed(kernel_weights_path, **formatted_weights)
        logger.info(f"Successfully saved kernel-formatted weights to {kernel_weights_path}")
        
        return kernel_weights_path
        
    except Exception as e:
        logger.error(f"Error processing weights: {str(e)}")
        raise

def verify_kernel_weights(weights_path: str):
    """Verify the kernel-formatted weights."""
    try:
        # Load the weights
        weights = np.load(weights_path)
        
        # Check for required weight keys
        required_keys = [
            'enc_embedding.0.weight',
            'enc_embedding.0.bias',
            'dec_embedding.0.weight',
            'dec_embedding.0.bias',
            'freq_decomp.freq_decomp.0.0.weight',
            'freq_decomp.freq_decomp.0.0.bias'
        ]
        
        missing_keys = [key for key in required_keys if key not in weights]
        if missing_keys:
            raise ValueError(f"Missing required weights: {missing_keys}")
        
        logger.info("Successfully verified kernel-formatted weights")
        return True
    except Exception as e:
        logger.error(f"Error verifying kernel weights: {str(e)}")
        return False

if __name__ == "__main__":
    try:
        # Download and format weights
        kernel_weights_path = download_fedformer_weights()
        
        # Verify weights
        if verify_kernel_weights(kernel_weights_path):
            logger.info("Model weights are ready for kernel usage")
        else:
            logger.error("Failed to verify kernel-formatted weights")
    except Exception as e:
        logger.error(f"Failed to process model weights: {str(e)}") 