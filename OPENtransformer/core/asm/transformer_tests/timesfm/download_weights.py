import os
import torch
import requests
from tqdm import tqdm
import logging
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoConfig
from huggingface_hub import hf_hub_download
from finlib.models.timesfm.architecture import TimesFMConfig, TimesFMModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def download_timesfm_model(model_name="google/timesfm-2.0-500m-pytorch"):
    """Download TimesFM model and weights."""
    try:
        # Create models directory if it doesn't exist
        models_dir = Path(__file__).parent / 'weights'
        models_dir.mkdir(exist_ok=True)
        
        logger.info(f"Downloading TimesFM model from {model_name}")
        
        # Download model configuration
        config = TimesFMConfig.from_pretrained(model_name)
        
        # Download model weights
        model = TimesFMModel.from_pretrained(
            model_name,
            config=config,
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        
        # Save model locally
        save_path = models_dir / 'timesfm'
        logger.info(f"Saving model to {save_path}")
        model.save_pretrained(save_path)
        config.save_pretrained(save_path)
        
        logger.info("Successfully downloaded and saved TimesFM model")
        return save_path
        
    except Exception as e:
        logger.error(f"Error downloading TimesFM model: {str(e)}")
        raise

def verify_model(model_path: Path):
    """Verify the downloaded model."""
    try:
        # Try loading the model
        config = TimesFMConfig.from_pretrained(model_path)
        model = TimesFMModel.from_pretrained(
            model_path,
            config=config,
            trust_remote_code=True
        )
        
        # Test with dummy input
        batch_size, seq_len = 1, 100
        dummy_input = torch.randint(0, config.vocab_size, (batch_size, seq_len))
        
        # Generate predictions
        with torch.no_grad():
            output = model(dummy_input)
        
        logger.info("Successfully verified model")
        logger.info(f"Model output shape: {output.shape}")
        return True
        
    except Exception as e:
        logger.error(f"Error verifying model: {str(e)}")
        return False

if __name__ == "__main__":
    try:
        # Download model
        model_path = download_timesfm_model()
        
        # Verify model
        if verify_model(model_path):
            logger.info("TimesFM model is ready to use")
        else:
            logger.error("Failed to verify TimesFM model")
            
    except Exception as e:
        logger.error(f"Failed to download model: {str(e)}") 