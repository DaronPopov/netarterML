import os
import torch
import numpy as np
import json
from diffusers import StableDiffusionPipeline
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def save_tensor_metadata(state_dict, name, output_dir):
    """Save tensor metadata including shapes and data types."""
    metadata = {}
    for key, tensor in state_dict.items():
        metadata[key] = {
            'shape': list(tensor.shape),
            'dtype': str(tensor.dtype),
            'requires_grad': tensor.requires_grad
        }
    
    with open(os.path.join(output_dir, f"{name}_metadata.json"), 'w') as f:
        json.dump(metadata, f, indent=2)

def convert_model(model_dir="models", output_dir="models"):
    """Convert Hugging Face model files to binary format."""
    try:
        logger.info("Loading model from %s...", model_dir)
        pipe = StableDiffusionPipeline.from_pretrained(model_dir)
        
        # Convert UNet weights and save metadata
        logger.info("Converting UNet weights...")
        unet_state = pipe.unet.state_dict()
        save_tensor_metadata(unet_state, "unet", output_dir)
        unet_weights = torch.cat([p.flatten() for p in unet_state.values()]).numpy()
        unet_weights.astype(np.float32).tofile(os.path.join(output_dir, "unet.bin"))
        
        # Convert VAE weights and save metadata
        logger.info("Converting VAE weights...")
        vae_state = pipe.vae.state_dict()
        save_tensor_metadata(vae_state, "vae", output_dir)
        vae_weights = torch.cat([p.flatten() for p in vae_state.values()]).numpy()
        vae_weights.astype(np.float32).tofile(os.path.join(output_dir, "vae.bin"))
        
        # Convert text encoder weights and save metadata
        logger.info("Converting text encoder weights...")
        text_encoder_state = pipe.text_encoder.state_dict()
        save_tensor_metadata(text_encoder_state, "text_encoder", output_dir)
        text_encoder_weights = torch.cat([p.flatten() for p in text_encoder_state.values()]).numpy()
        text_encoder_weights.astype(np.float32).tofile(os.path.join(output_dir, "text_encoder.bin"))
        
        # Save model configuration
        logger.info("Saving model configuration...")
        config = {
            'unet': {
                'in_channels': pipe.unet.config.in_channels,
                'out_channels': pipe.unet.config.out_channels,
                'block_out_channels': pipe.unet.config.block_out_channels,
                'layers_per_block': pipe.unet.config.layers_per_block,
                'attention_head_dim': pipe.unet.config.attention_head_dim
            },
            'vae': {
                'latent_channels': pipe.vae.config.latent_channels,
                'scaling_factor': pipe.vae.config.scaling_factor
            },
            'text_encoder': {
                'hidden_size': pipe.text_encoder.config.hidden_size,
                'intermediate_size': pipe.text_encoder.config.intermediate_size,
                'num_attention_heads': pipe.text_encoder.config.num_attention_heads,
                'num_hidden_layers': pipe.text_encoder.config.num_hidden_layers
            },
            'scheduler': {
                'num_train_timesteps': pipe.scheduler.config.num_train_timesteps,
                'beta_start': pipe.scheduler.config.beta_start,
                'beta_end': pipe.scheduler.config.beta_end,
                'beta_schedule': pipe.scheduler.config.beta_schedule
            }
        }
        with open(os.path.join(output_dir, "model_config.json"), 'w') as f:
            json.dump(config, f, indent=2)
        
        # Save tensor shapes for each component
        logger.info("Saving tensor shapes...")
        shapes = {
            'unet': {name: list(param.shape) for name, param in pipe.unet.named_parameters()},
            'vae': {name: list(param.shape) for name, param in pipe.vae.named_parameters()},
            'text_encoder': {name: list(param.shape) for name, param in pipe.text_encoder.named_parameters()}
        }
        with open(os.path.join(output_dir, "tensor_shapes.json"), 'w') as f:
            json.dump(shapes, f, indent=2)
        
        logger.info("Model conversion completed successfully")
        return True
        
    except Exception as e:
        logger.error("Error converting model: %s", e)
        logger.error("Full traceback:", exc_info=True)
        return False

if __name__ == "__main__":
    success = convert_model()
    if success:
        logger.info("Model conversion completed successfully")
    else:
        logger.error("Model conversion failed") 