import numpy as np
import logging
from typing import Dict, Any, Optional, Tuple, List, Union
from transformers import PreTrainedModel
import torch

logger = logging.getLogger("finlib.model_converter")

class ArchitectureAdapter:
    """Maps different transformer architectures to our kernel's expected format."""
    
    def __init__(self, model_type: str):
        """
        Initialize the architecture adapter.
        
        Args:
            model_type: Type of transformer model (e.g., 'bloom', 'bert', 'gpt2')
        """
        self.model_type = model_type.lower()
        self.architecture_mappings = {
            'bloom': self._map_bloom_architecture,
            'bert': self._map_bert_architecture,
            'gpt2': self._map_gpt2_architecture,
            'llama': self._map_llama_architecture,
            'default': self._map_default_architecture
        }
        
    def adapt_weights(self, model: PreTrainedModel, state_dict: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Adapt model weights to match our kernel's expected format.
        
        Args:
            model: The Hugging Face model
            state_dict: Model's state dictionary
            
        Returns:
            Adapted weights in our kernel's format
        """
        try:
            # Get the appropriate mapping function
            mapper = self.architecture_mappings.get(self.model_type, self.architecture_mappings['default'])
            
            # Map the weights
            adapted_weights = mapper(model, state_dict)
            
            # Validate the adapted weights
            self._validate_adapted_weights(adapted_weights)
            
            return adapted_weights
            
        except Exception as e:
            logger.error(f"Error adapting weights: {e}")
            raise
            
    def _map_bloom_architecture(self, model: PreTrainedModel, state_dict: Dict[str, Union[torch.Tensor, np.ndarray]]) -> Dict[str, np.ndarray]:
        """Map BLOOM architecture weights to our expected format."""
        adapted = {}
        
        # Helper function to convert tensors to numpy arrays
        def to_numpy(x):
            if isinstance(x, torch.Tensor):
                return x.numpy()
            return x
        
        # Map word embeddings and layer normalization weights
        adapted['word_embeddings'] = to_numpy(state_dict['transformer.word_embeddings.weight'])
        adapted['word_embeddings_layernorm_weight'] = to_numpy(state_dict['transformer.word_embeddings_layernorm.weight'])
        adapted['word_embeddings_layernorm_bias'] = to_numpy(state_dict['transformer.word_embeddings_layernorm.bias'])
        adapted['lm_head_weights'] = to_numpy(state_dict['lm_head.weight'])
        adapted['ln_f_weight'] = to_numpy(state_dict['transformer.ln_f.weight'])
        adapted['ln_f_bias'] = to_numpy(state_dict['transformer.ln_f.bias'])
        
        # Get number of layers from state dict
        layer_indices = sorted(set(
            int(key.split('.')[2]) for key in state_dict.keys() 
            if key.startswith('transformer.h.') and key.endswith('.weight')
        ))
        
        # Map each layer's weights
        for layer_idx in layer_indices:
            # Layer normalization weights
            adapted[f'layer_{layer_idx}_ln1_weight'] = to_numpy(state_dict[f'transformer.h.{layer_idx}.input_layernorm.weight'])
            adapted[f'layer_{layer_idx}_ln1_bias'] = to_numpy(state_dict[f'transformer.h.{layer_idx}.input_layernorm.bias'])
            adapted[f'layer_{layer_idx}_ln2_weight'] = to_numpy(state_dict[f'transformer.h.{layer_idx}.post_attention_layernorm.weight'])
            adapted[f'layer_{layer_idx}_ln2_bias'] = to_numpy(state_dict[f'transformer.h.{layer_idx}.post_attention_layernorm.bias'])
            
            # Attention weights
            adapted[f'layer_{layer_idx}_qkv_weights'] = to_numpy(state_dict[f'transformer.h.{layer_idx}.self_attention.query_key_value.weight'])
            adapted[f'layer_{layer_idx}_qkv_bias'] = to_numpy(state_dict[f'transformer.h.{layer_idx}.self_attention.query_key_value.bias'])
            adapted[f'layer_{layer_idx}_out_weights'] = to_numpy(state_dict[f'transformer.h.{layer_idx}.self_attention.dense.weight'])
            adapted[f'layer_{layer_idx}_out_bias'] = to_numpy(state_dict[f'transformer.h.{layer_idx}.self_attention.dense.bias'])
            
            # Feed-forward weights
            adapted[f'layer_{layer_idx}_ff1_weight'] = to_numpy(state_dict[f'transformer.h.{layer_idx}.mlp.dense_h_to_4h.weight'])
            adapted[f'layer_{layer_idx}_ff1_bias'] = to_numpy(state_dict[f'transformer.h.{layer_idx}.mlp.dense_h_to_4h.bias'])
            adapted[f'layer_{layer_idx}_ff2_weight'] = to_numpy(state_dict[f'transformer.h.{layer_idx}.mlp.dense_4h_to_h.weight'])
            adapted[f'layer_{layer_idx}_ff2_bias'] = to_numpy(state_dict[f'transformer.h.{layer_idx}.mlp.dense_4h_to_h.bias'])
        
        return adapted
        
    def _map_bert_architecture(self, model: PreTrainedModel, state_dict: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Map BERT architecture weights to our kernel format."""
        adapted = {}
        n_layers = model.config.num_hidden_layers
        
        for i in range(n_layers):
            prefix = f'encoder.layer.{i}.'
            
            # Map attention weights
            q = state_dict[f'{prefix}attention.self.query.weight']
            k = state_dict[f'{prefix}attention.self.key.weight']
            v = state_dict[f'{prefix}attention.self.value.weight']
            adapted[f'layer_{i}_qkv_weights'] = np.concatenate([q, k, v], axis=0)
            
            # Map attention output
            adapted[f'layer_{i}_attn_output_weights'] = state_dict[f'{prefix}attention.output.dense.weight']
            
            # Map feed-forward weights
            adapted[f'layer_{i}_ff1_weights'] = state_dict[f'{prefix}intermediate.dense.weight']
            adapted[f'layer_{i}_ff2_weights'] = state_dict[f'{prefix}output.dense.weight']
            
            # Map layer norms
            adapted[f'layer_{i}_ln1_gamma'] = state_dict[f'{prefix}attention.output.LayerNorm.weight']
            adapted[f'layer_{i}_ln1_beta'] = state_dict[f'{prefix}attention.output.LayerNorm.bias']
            adapted[f'layer_{i}_ln2_gamma'] = state_dict[f'{prefix}output.LayerNorm.weight']
            adapted[f'layer_{i}_ln2_beta'] = state_dict[f'{prefix}output.LayerNorm.bias']
            
        return adapted
        
    def _map_gpt2_architecture(self, model: PreTrainedModel, state_dict: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Map GPT-2 architecture weights to our kernel format."""
        adapted = {}
        n_layers = model.config.n_layer
        
        for i in range(n_layers):
            prefix = f'h.{i}.'
            
            # Map attention weights
            q = state_dict[f'{prefix}attn.c_attn.weight'][:model.config.n_embd]
            k = state_dict[f'{prefix}attn.c_attn.weight'][model.config.n_embd:2*model.config.n_embd]
            v = state_dict[f'{prefix}attn.c_attn.weight'][2*model.config.n_embd:]
            adapted[f'layer_{i}_qkv_weights'] = np.concatenate([q, k, v], axis=0)
            
            # Map attention output
            adapted[f'layer_{i}_attn_output_weights'] = state_dict[f'{prefix}attn.c_proj.weight']
            
            # Map feed-forward weights
            adapted[f'layer_{i}_ff1_weights'] = state_dict[f'{prefix}mlp.c_fc.weight']
            adapted[f'layer_{i}_ff2_weights'] = state_dict[f'{prefix}mlp.c_proj.weight']
            
            # Map layer norms
            adapted[f'layer_{i}_ln1_gamma'] = state_dict[f'{prefix}ln_1.weight']
            adapted[f'layer_{i}_ln1_beta'] = state_dict[f'{prefix}ln_1.bias']
            adapted[f'layer_{i}_ln2_gamma'] = state_dict[f'{prefix}ln_2.weight']
            adapted[f'layer_{i}_ln2_beta'] = state_dict[f'{prefix}ln_2.bias']
            
        return adapted
        
    def _map_llama_architecture(self, model: PreTrainedModel, state_dict: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Map LLaMA architecture weights to our kernel format."""
        adapted = {}
        n_layers = model.config.num_hidden_layers
        
        for i in range(n_layers):
            prefix = f'model.layers.{i}.'
            
            # Map attention weights
            q = state_dict[f'{prefix}self_attn.q_proj.weight']
            k = state_dict[f'{prefix}self_attn.k_proj.weight']
            v = state_dict[f'{prefix}self_attn.v_proj.weight']
            adapted[f'layer_{i}_qkv_weights'] = np.concatenate([q, k, v], axis=0)
            
            # Map attention output
            adapted[f'layer_{i}_attn_output_weights'] = state_dict[f'{prefix}self_attn.o_proj.weight']
            
            # Map feed-forward weights
            adapted[f'layer_{i}_ff1_weights'] = state_dict[f'{prefix}mlp.gate_proj.weight']
            adapted[f'layer_{i}_ff2_weights'] = state_dict[f'{prefix}mlp.down_proj.weight']
            
            # Map layer norms
            adapted[f'layer_{i}_ln1_gamma'] = state_dict[f'{prefix}input_layernorm.weight']
            adapted[f'layer_{i}_ln1_beta'] = state_dict[f'{prefix}input_layernorm.bias']
            adapted[f'layer_{i}_ln2_gamma'] = state_dict[f'{prefix}post_attention_layernorm.weight']
            adapted[f'layer_{i}_ln2_beta'] = state_dict[f'{prefix}post_attention_layernorm.bias']
            
        return adapted
        
    def _map_default_architecture(self, model: PreTrainedModel, state_dict: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Default mapping that tries to infer the architecture from the state dict keys."""
        adapted = {}
        
        # Try to find layer count from state dict
        layer_indices = set()
        for key in state_dict.keys():
            if 'layer.' in key:
                idx = int(key.split('layer.')[1].split('.')[0])
                layer_indices.add(idx)
        n_layers = max(layer_indices) + 1 if layer_indices else 0
        
        if n_layers == 0:
            raise ValueError("Could not determine number of layers from state dict")
            
        # Try to find attention weights
        for i in range(n_layers):
            # Look for QKV weights
            qkv_keys = [k for k in state_dict.keys() if f'layer.{i}' in k and any(x in k for x in ['query', 'key', 'value', 'qkv'])]
            if qkv_keys:
                # Found QKV weights
                if len(qkv_keys) == 3:  # Separate Q, K, V
                    q = state_dict[qkv_keys[0]]
                    k = state_dict[qkv_keys[1]]
                    v = state_dict[qkv_keys[2]]
                    adapted[f'layer_{i}_qkv_weights'] = np.concatenate([q, k, v], axis=0)
                else:  # Combined QKV
                    qkv = state_dict[qkv_keys[0]]
                    head_size = qkv.shape[0] // 3
                    adapted[f'layer_{i}_qkv_weights'] = qkv
                    
            # Look for attention output
            attn_out_keys = [k for k in state_dict.keys() if f'layer.{i}' in k and any(x in k for x in ['output', 'proj'])]
            if attn_out_keys:
                adapted[f'layer_{i}_attn_output_weights'] = state_dict[attn_out_keys[0]]
                
            # Look for feed-forward weights
            ff_keys = [k for k in state_dict.keys() if f'layer.{i}' in k and any(x in k for x in ['mlp', 'ffn', 'intermediate'])]
            if len(ff_keys) >= 2:
                adapted[f'layer_{i}_ff1_weights'] = state_dict[ff_keys[0]]
                adapted[f'layer_{i}_ff2_weights'] = state_dict[ff_keys[1]]
                
            # Look for layer norms
            ln_keys = [k for k in state_dict.keys() if f'layer.{i}' in k and any(x in k for x in ['norm', 'layernorm'])]
            if len(ln_keys) >= 2:
                adapted[f'layer_{i}_ln1_gamma'] = state_dict[ln_keys[0]]
                adapted[f'layer_{i}_ln1_beta'] = state_dict[ln_keys[1]]
                adapted[f'layer_{i}_ln2_gamma'] = state_dict[ln_keys[2]]
                adapted[f'layer_{i}_ln2_beta'] = state_dict[ln_keys[3]]
                
        return adapted
        
    def _validate_adapted_weights(self, adapted_weights: Dict[str, np.ndarray]) -> None:
        """Validate that all required weights are present."""
        # Required weights for word embeddings
        required_weights = [
            'word_embeddings',
            'word_embeddings_layernorm_weight',
            'word_embeddings_layernorm_bias',
            'lm_head_weights',
            'ln_f_weight',
            'ln_f_bias'
        ]
        
        # Get number of layers from adapted weights
        layer_indices = sorted(set(
            int(key.split('_')[1]) for key in adapted_weights.keys() 
            if key.startswith('layer_') and 'qkv_weights' in key
        ))
        
        # Required weights for each layer
        layer_weights = [
            'qkv_weights',
            'qkv_bias',
            'out_weights',
            'out_bias',
            'ln1_weight',
            'ln1_bias',
            'ln2_weight',
            'ln2_bias',
            'ff1_weight',
            'ff1_bias',
            'ff2_weight',
            'ff2_bias'
        ]
        
        # Add layer-specific weights to required list
        for layer_idx in layer_indices:
            for weight in layer_weights:
                required_weights.append(f'layer_{layer_idx}_{weight}')
        
        # Check that all required weights are present
        for weight in required_weights:
            if weight not in adapted_weights:
                raise ValueError(f"Missing required weight: {weight}")
            
        # Check that all weights have the expected type
        for weight, value in adapted_weights.items():
            if not isinstance(value, np.ndarray):
                raise ValueError(f"Weight {weight} has incorrect type: {type(value)}")
            
        # Check that all weights have valid shapes
        for weight, value in adapted_weights.items():
            if value.ndim == 0 or 0 in value.shape:
                raise ValueError(f"Weight {weight} has invalid shape: {value.shape}")
            
        # Check that all weights have valid values
        for weight, value in adapted_weights.items():
            if np.isnan(value).any():
                raise ValueError(f"Weight {weight} contains NaN values")
            if np.isinf(value).any():
                raise ValueError(f"Weight {weight} contains infinite values") 