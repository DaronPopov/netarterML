import logging
from typing import Dict, Any, Optional, Union
from pathlib import Path
from transformers import AutoModel, AutoConfig, PreTrainedModel
import torch
import numpy as np
from .converter import ModelConverter
from .loader import ConvertedModelLoader
from .architecture_adapter import ArchitectureAdapter

logger = logging.getLogger("finlib.model_converter")

class HuggingFaceConverter:
    """High-level interface for converting Hugging Face models to custom ASM transformer format."""
    
    def __init__(self, model_config: Optional[Dict[str, Any]] = None):
        """
        Initialize the Hugging Face converter.
        
        Args:
            model_config: Optional configuration dictionary for the model
        """
        self.model_config = model_config or {}
        self.converter = ModelConverter(model_config)
        self.model = None
        self.config = None
        self.adapter = None
        self.weights = {}
        self.metadata = {}
        self.fused_op = None
        
    def load_model(self, model_name_or_path: str, model_type: Optional[str] = None) -> None:
        """
        Load a Hugging Face model and its configuration.
        
        Args:
            model_name_or_path: Name of the model on Hugging Face Hub or local path
            model_type: Optional model type to specify (e.g., 'bert', 'gpt2', etc.)
        """
        try:
            logger.info(f"Loading model from {model_name_or_path}")
            
            # Load model configuration
            self.config = AutoConfig.from_pretrained(model_name_or_path)
            
            # Load model
            if model_type:
                # Use specific model class if specified
                from transformers import AutoModelForCausalLM, AutoModelForMaskedLM
                model_class = {
                    'gpt2': AutoModelForCausalLM,
                    'bert': AutoModelForMaskedLM,
                    'bloom': AutoModelForCausalLM,
                    # Add more model types as needed
                }.get(model_type.lower(), AutoModel)
            else:
                model_class = AutoModel
                
            self.model = model_class.from_pretrained(
                model_name_or_path,
                config=self.config,
                trust_remote_code=True
            )
            
            # Move model to CPU for conversion
            self.model.to('cpu')
            self.model.eval()
            
            # Initialize architecture adapter
            self.adapter = ArchitectureAdapter(model_type or self._infer_model_type())
            
            logger.info("Successfully loaded Hugging Face model")
            
        except Exception as e:
            logger.error(f"Error loading Hugging Face model: {e}")
            raise
            
    def _infer_model_type(self) -> str:
        """Infer the model type from the model class or configuration."""
        if hasattr(self.model, 'config'):
            model_type = self.model.config.model_type.lower()
            if model_type in ['bloom', 'bert', 'gpt2', 'llama']:
                return model_type
        return 'default'
            
    def extract_transformer_weights(self) -> Dict[str, np.ndarray]:
        """
        Extract transformer weights from the loaded Hugging Face model.
        
        Returns:
            Dictionary of extracted weights
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
            
        try:
            # Convert state dict to numpy arrays
            state_dict = {k: v.detach().numpy() for k, v in self.model.state_dict().items()}
            
            # Debug: Print state dict keys
            logger.info("Model state dict keys:")
            for key in state_dict.keys():
                logger.info(f"- {key}")
            
            # Use architecture adapter to map weights
            adapted_weights = self.adapter.adapt_weights(self.model, state_dict)
            
            return adapted_weights
            
        except Exception as e:
            logger.error(f"Error extracting transformer weights: {e}")
            raise
            
    def convert_model(self, model_name_or_path: str, output_dir: str, 
                     model_type: Optional[str] = None,
                     model_config: Optional[Dict[str, Any]] = None) -> None:
        """
        Convert a Hugging Face model to the custom ASM transformer format.
        
        Args:
            model_name_or_path: Name of the model on Hugging Face Hub or local path
            output_dir: Directory to save the converted model
            model_type: Optional model type to specify
            model_config: Optional model configuration
        """
        try:
            # Update model config if provided
            if model_config:
                self.model_config.update(model_config)
                
            # Load the model
            self.load_model(model_name_or_path, model_type)
            
            # Extract weights
            weights = self.extract_transformer_weights()
            
            # Convert weights to fused format
            converted_weights = self.converter.convert_weights(weights)
            
            # Extract metadata
            metadata = {
                "model_type": "transformer",
                "d_model": self.config.hidden_size,
                "n_heads": self.config.num_attention_heads,
                "n_layers": self.config.num_hidden_layers,
                "config": self.model_config,
                "original_model": model_name_or_path,
                "original_config": self.config.to_dict()
            }
            
            # Save converted model
            self.converter.save_converted_model(output_dir, converted_weights, metadata)
            
            logger.info(f"Successfully converted model to {output_dir}")
            
        except Exception as e:
            logger.error(f"Error converting model: {e}")
            raise
            
    def load_converted_model(self, model_dir: str, builder_func) -> ConvertedModelLoader:
        """
        Load a converted model for inference.
        
        Args:
            model_dir: Directory containing the converted model
            builder_func: Function to build and JIT compile assembly code
            
        Returns:
            Loaded model in ConvertedModelLoader format
        """
        try:
            loader = ConvertedModelLoader(model_dir)
            loader.load_model(builder_func)
            return loader
        except Exception as e:
            logger.error(f"Error loading converted model: {e}")
            raise

    def get_lm_head_weights(self):
        """Get the language model head weights for logits calculation."""
        if "lm_head_weights" not in self.weights:
            raise ValueError("LM head weights not found in converted model")
        return self.weights["lm_head_weights"] 