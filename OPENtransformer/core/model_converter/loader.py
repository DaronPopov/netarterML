import numpy as np
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from finlib.core.asm.kernels.fused_transformer_op import create_fully_fused_transformer_op
import os
import ctypes

logger = logging.getLogger("finlib.model_converter")

class ConvertedModelLoader:
    """Loads converted transformer models into the fused transformer format."""
    
    def __init__(self, model_dir, builder_func=None):
        """
        Initialize the model loader.
        
        Args:
            model_dir: Directory containing the converted model
            builder_func: Function to build and JIT compile assembly code
        """
        self.model_dir = Path(model_dir)
        self.weights = {}
        self.metadata = {}
        self.fused_op = None
        self.builder_func = builder_func
        self._load_model()
        
    def _load_model(self):
        """Load the converted model weights and metadata."""
        # Load weights
        weights_path = self.model_dir / "weights.npz"
        if not weights_path.exists():
            raise FileNotFoundError(f"Weights file not found at {weights_path}")
        self.weights = dict(np.load(weights_path))

        # Load metadata
        metadata_path = self.model_dir / "metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found at {metadata_path}")
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)

        # Initialize fused transformer operation
        if self.builder_func is not None:
            self.fused_op = create_fully_fused_transformer_op(self.builder_func)

    def load_weights(self) -> Dict[str, np.ndarray]:
        """
        Load converted weights from the model directory.
        
        Returns:
            Dictionary of loaded weights
        """
        try:
            return self.weights
        except Exception as e:
            logger.error(f"Error loading weights: {e}")
            raise
            
    def load_metadata(self) -> Dict[str, Any]:
        """
        Load model metadata from the model directory.
        
        Returns:
            Dictionary of model metadata
        """
        try:
            return self.metadata
        except Exception as e:
            logger.error(f"Error loading metadata: {e}")
            raise
            
    def initialize_fused_op(self, builder_func) -> None:
        """
        Initialize the fused transformer operation.
        
        Args:
            builder_func: Function to build and JIT compile assembly code
        """
        try:
            # If we already have a fused op, just return
            if self.fused_op is not None:
                return

            # Create the fused op with the builder function
            fused_op = builder_func(None, None)  # Pass None since we're using pre-compiled op
            
            # If the op was created but doesn't have argtypes, create a wrapper
            if fused_op is not None and not hasattr(fused_op, 'argtypes'):
                logger.info("Creating wrapper for fused op without argtypes")
                
                def wrapped_fused_op(input_ptr, output_ptr, *args):
                    """Wrapper that handles the function call without argtypes."""
                    try:
                        # Call the raw function
                        result = fused_op(input_ptr, output_ptr, *args)
                        return result if result is not None else 0
                    except Exception as e:
                        logger.error(f"Error in wrapped fused op: {e}")
                        raise
                
                self.fused_op = wrapped_fused_op
            else:
                self.fused_op = fused_op
            
            # Verify the op was created successfully
            if self.fused_op is None:
                logger.error("Fused transformer operation returned None")
                raise RuntimeError("Failed to create fused transformer operation")
                
            logger.info("Successfully initialized fused transformer operation")
            
        except Exception as e:
            logger.error(f"Error initializing fused op: {e}")
            raise RuntimeError(f"Failed to create fused transformer operation: {str(e)}")
            
    def get_layer_weights(self, layer_idx: int) -> Tuple[np.ndarray, ...]:
        """
        Get weights for a specific transformer layer.
        
        Args:
            layer_idx: Index of the layer to get weights for
            
        Returns:
            Tuple of weights for the layer (qkv_weights, out_weights, ff1_weight, ff2_weight,
            ln1_weight, ln1_bias, ln2_weight, ln2_bias)
        """
        try:
            return (
                self.weights[f'layer_{layer_idx}_qkv_weights'],
                self.weights[f'layer_{layer_idx}_out_weights'],
                self.weights[f'layer_{layer_idx}_ff1_weight'],
                self.weights[f'layer_{layer_idx}_ff2_weight'],
                self.weights[f'layer_{layer_idx}_ln1_weight'],
                self.weights[f'layer_{layer_idx}_ln1_bias'],
                self.weights[f'layer_{layer_idx}_ln2_weight'],
                self.weights[f'layer_{layer_idx}_ln2_bias']
            )
        except KeyError as e:
            logger.error(f"Missing weights for layer {layer_idx}: {e}")
            raise
            
    def get_model_config(self) -> Dict[str, Any]:
        """
        Get the model configuration.
        
        Returns:
            Dictionary of model configuration
        """
        return self.metadata.get('config', {})
        
    def get_model_dimensions(self):
        """Get the model dimensions from metadata."""
        return {
            "d_model": int(self.metadata["d_model"]),
            "n_heads": int(self.metadata["n_heads"]),
            "n_layers": int(self.metadata["n_layers"]),
        }
        
    def word_embeddings(self, input_ids: np.ndarray) -> np.ndarray:
        """
        Convert input IDs to embeddings using the word embeddings matrix.
        
        Args:
            input_ids: Input token IDs of shape (batch_size, seq_len)
            
        Returns:
            Embedded tokens of shape (batch_size, seq_len, d_model)
        """
        try:
            # Get embeddings
            embeddings = self.weights['word_embeddings'][input_ids]
            
            # Apply layer normalization if available
            if 'word_embeddings_layernorm_weight' in self.weights:
                gamma = self.weights['word_embeddings_layernorm_weight']
                beta = self.weights['word_embeddings_layernorm_bias']
                
                # Layer normalization
                mean = np.mean(embeddings, axis=-1, keepdims=True)
                variance = np.var(embeddings, axis=-1, keepdims=True)
                normalized = (embeddings - mean) / np.sqrt(variance + 1e-5)
                embeddings = gamma * normalized + beta
                
            return embeddings
            
        except Exception as e:
            logger.error(f"Error in word embeddings: {e}")
            raise
        
    def load_model(self, builder_func) -> None:
        """
        Load the complete converted model.
        
        Args:
            builder_func: Function to build and JIT compile assembly code
        """
        try:
            # Load weights and metadata
            self.load_weights()
            self.load_metadata()
            
            # Initialize fused operation
            self.initialize_fused_op(builder_func)
            
            logger.info(f"Successfully loaded model from {self.model_dir}")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise

    def get_word_embeddings(self):
        """Get the word embedding weights."""
        return self.weights["word_embeddings"]

    def get_lm_head_weights(self):
        """Get the language model head weights."""
        return self.weights["lm_head_weights"]

    def get_ln_f_weights(self):
        """Get the final layer normalization weights."""
        return (
            self.weights["ln_f_weight"],
            self.weights["ln_f_bias"],
        )

    def get_next_token_logits(self, input_ids: np.ndarray) -> np.ndarray:
        """
        Get logits for the next token prediction.
        
        Args:
            input_ids: Input token IDs of shape (batch_size, seq_len)
            
        Returns:
            Logits for next token prediction of shape (batch_size, vocab_size)
        """
        try:
            # Get model dimensions
            dimensions = self.get_model_dimensions()
            d_model = dimensions["d_model"]
            n_heads = dimensions["n_heads"]
            n_layers = dimensions["n_layers"]
            batch_size, seq_len = input_ids.shape
            
            # Get word embeddings
            hidden_states = self.word_embeddings(input_ids)
            
            # Process through transformer layers
            for layer_idx in range(n_layers):
                # Get layer weights
                weights = self.get_layer_weights(layer_idx)
                
                # Create output tensor
                output = np.zeros_like(hidden_states)
                
                # Apply fused transformer layer
                self.fused_op(
                    hidden_states.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                    output.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                    *[w.ctypes.data_as(ctypes.POINTER(ctypes.c_float)) for w in weights],
                    batch_size,
                    seq_len,
                    d_model,
                    n_heads
                )
                
                # Update hidden states
                hidden_states = output
            
            # Apply final layer normalization
            ln_f_weight, ln_f_bias = self.get_ln_f_weights()
            mean = np.mean(hidden_states, axis=-1, keepdims=True)
            variance = np.var(hidden_states, axis=-1, keepdims=True)
            hidden_states = (hidden_states - mean) / np.sqrt(variance + 1e-5)
            hidden_states = ln_f_weight * hidden_states + ln_f_bias
            
            # Get logits from language model head
            lm_head_weights = self.get_lm_head_weights()
            logits = np.matmul(hidden_states[:, -1, :], lm_head_weights.T)
            
            return logits
            
        except Exception as e:
            logger.error(f"Error getting next token logits: {e}")
            raise 