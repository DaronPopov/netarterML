import numpy as np
import torch
import logging
from typing import Dict, Any, Optional, Tuple, List
from pathlib import Path
import json
import pickle
import os

logger = logging.getLogger("finlib.model_converter")

class ModelConverter:
    """Converts custom transformer models to the fused transformer format."""
    
    def __init__(self, model_config: Optional[Dict[str, Any]] = None):
        """
        Initialize the model converter.
        
        Args:
            model_config: Optional configuration dictionary for the model
        """
        self.model_config = model_config or {}
        self.converted_weights = {}
        self.model_metadata = {}
        
    def load_pytorch_model(self, model_path: str) -> torch.nn.Module:
        """
        Load a PyTorch model from a file.
        
        Args:
            model_path: Path to the PyTorch model file
            
        Returns:
            Loaded PyTorch model
        """
        try:
            if model_path.endswith('.pt') or model_path.endswith('.pth'):
                # Add our custom classes to safe globals
                from finlib.model_converter.test_converter import DummyTransformer, DummyTransformerLayer
                torch.serialization.add_safe_globals([DummyTransformer, DummyTransformerLayer])
                
                # Load with weights_only=False since we trust our own model
                model = torch.load(model_path, weights_only=False)
            elif model_path.endswith('.bin'):
                # Handle HuggingFace format
                from transformers import AutoModel
                model = AutoModel.from_pretrained(model_path)
            else:
                raise ValueError(f"Unsupported model format: {model_path}")
            return model
        except Exception as e:
            logger.error(f"Error loading PyTorch model: {e}")
            raise
            
    def extract_transformer_weights(self, model: torch.nn.Module) -> Dict[str, np.ndarray]:
        """
        Extract transformer weights from a PyTorch model.
        
        Args:
            model: PyTorch model to extract weights from
            
        Returns:
            Dictionary of extracted weights
        """
        weights = {}
        try:
            # Handle our dummy model structure
            if hasattr(model, 'transformer'):
                layers = model.transformer
            else:
                layers = model
                
            # Extract layer norm weights
            for i, layer in enumerate(layers):
                weights[f'layer_{i}_ln1_gamma'] = layer.norm1.weight.detach().numpy()
                weights[f'layer_{i}_ln1_beta'] = layer.norm1.bias.detach().numpy()
                weights[f'layer_{i}_ln2_gamma'] = layer.norm2.weight.detach().numpy()
                weights[f'layer_{i}_ln2_beta'] = layer.norm2.bias.detach().numpy()
                
                # Extract attention weights
                weights[f'layer_{i}_qkv_weights'] = layer.attn.qkv.weight.detach().numpy()
                weights[f'layer_{i}_attn_output_weights'] = layer.attn.proj.weight.detach().numpy()
                
                # Extract feed-forward weights
                weights[f'layer_{i}_ff1_weights'] = layer.mlp[0].weight.detach().numpy()
                weights[f'layer_{i}_ff2_weights'] = layer.mlp[2].weight.detach().numpy()
                
            return weights
        except Exception as e:
            logger.error(f"Error extracting transformer weights: {e}")
            raise
            
    def convert_weights(self, weights: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Convert weights to the format expected by the fused transformer op."""
        converted = {}
        
        # Get number of layers from weights
        layer_indices = sorted(set(
            int(key.split('_')[1]) for key in weights.keys() 
            if key.startswith('layer_') and 'qkv_weights' in key
        ))
        
        # Convert each layer's weights
        for i in layer_indices:
            # Attention weights
            converted[f'layer_{i}_qkv_weights'] = weights[f'layer_{i}_qkv_weights']
            converted[f'layer_{i}_qkv_bias'] = weights[f'layer_{i}_qkv_bias']
            converted[f'layer_{i}_out_weights'] = weights[f'layer_{i}_out_weights']
            converted[f'layer_{i}_out_bias'] = weights[f'layer_{i}_out_bias']
            
            # Layer normalization weights
            converted[f'layer_{i}_ln1_weight'] = weights[f'layer_{i}_ln1_weight']
            converted[f'layer_{i}_ln1_bias'] = weights[f'layer_{i}_ln1_bias']
            converted[f'layer_{i}_ln2_weight'] = weights[f'layer_{i}_ln2_weight']
            converted[f'layer_{i}_ln2_bias'] = weights[f'layer_{i}_ln2_bias']
            
            # Feed-forward weights
            converted[f'layer_{i}_ff1_weight'] = weights[f'layer_{i}_ff1_weight']
            converted[f'layer_{i}_ff1_bias'] = weights[f'layer_{i}_ff1_bias']
            converted[f'layer_{i}_ff2_weight'] = weights[f'layer_{i}_ff2_weight']
            converted[f'layer_{i}_ff2_bias'] = weights[f'layer_{i}_ff2_bias']
        
        # Word embeddings and layer normalization
        converted['word_embeddings'] = weights['word_embeddings']
        converted['word_embeddings_layernorm_weight'] = weights['word_embeddings_layernorm_weight']
        converted['word_embeddings_layernorm_bias'] = weights['word_embeddings_layernorm_bias']
        converted['lm_head_weights'] = weights['lm_head_weights']
        converted['ln_f_weight'] = weights['ln_f_weight']
        converted['ln_f_bias'] = weights['ln_f_bias']
        
        return converted
            
    def save_converted_model(self, output_dir: str, weights: Dict[str, np.ndarray], metadata: Dict[str, Any]):
        """
        Save converted model weights and metadata.
        
        Args:
            output_dir: Directory to save the converted model
            weights: Dictionary of converted weights
            metadata: Model metadata
        """
        try:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Save weights
            weights_path = output_path / "weights.npz"
            np.savez(weights_path, **weights)
            
            # Save metadata
            metadata_path = output_path / "metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
                
            logger.info(f"Saved converted model to {output_dir}")
        except Exception as e:
            logger.error(f"Error saving converted model: {e}")
            raise
            
    def convert_model(self, model_path: str, output_dir: str, model_config: Optional[Dict[str, Any]] = None) -> None:
        """
        Convert a custom transformer model to the fused format.
        
        Args:
            model_path: Path to the input model
            output_dir: Directory to save the converted model
            model_config: Optional model configuration
        """
        try:
            # Update model config if provided
            if model_config:
                self.model_config.update(model_config)
                
            # Load and convert model
            model = self.load_pytorch_model(model_path)
            weights = self.extract_transformer_weights(model)
            converted_weights = self.convert_weights(weights)
            
            # Extract metadata
            metadata = {
                "model_type": "transformer",
                "d_model": converted_weights['layer_0_qkv_weights'].shape[1],
                "n_heads": converted_weights['layer_0_qkv_weights'].shape[0] // (3 * converted_weights['layer_0_qkv_weights'].shape[1]),
                "n_layers": len(set(k.split('_')[1] for k in converted_weights.keys())),
                "config": self.model_config
            }
            
            # Save converted model
            self.save_converted_model(output_dir, converted_weights, metadata)
            
        except Exception as e:
            logger.error(f"Error converting model: {e}")
            raise 