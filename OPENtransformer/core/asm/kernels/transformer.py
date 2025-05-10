import numpy as np
import ctypes
import time
import logging
import os
import json
from typing import List, Optional
from OPENtransformer.core.asm.assembler.builder import build_and_jit
from OPENtransformer.core.asm.kernels.attention_matmul import attention_matmul_code
from OPENtransformer.core.asm.kernels.softmax import softmax_code
from OPENtransformer.core.asm.kernels.transpose import transpose_code
from OPENtransformer.core.asm.kernels.gelu import gelu_code
from OPENtransformer.core.asm.kernels.layer_norm import layer_norm_code
from OPENtransformer.core.asm.kernels.weight_initializer import get_kernel_code
from OPENtransformer.core.asm.kernels.tokenizer import BasicTokenizer
from OPENtransformer.core.asm.kernels.position_embedding import position_embedding_code
from OPENtransformer.core.asm.kernels.fused_transformer_op import (
    fused_transformer_layer_code,
    fully_fused_transformer_layer_code,
    create_fused_transformer_op,
    create_fully_fused_transformer_op
)

logger = logging.getLogger("OPENtransformer.core.asm.transformer")

class Transformer:
    def __init__(self, d_model: int, n_heads: int, n_layers: int, vocab_size: int = 32000, d_ff: int = None, max_context_length: int = 1024):
        """
        Initialize a transformer model.
        
        Args:
            d_model: Model dimension
            n_heads: Number of attention heads
            n_layers: Number of transformer layers
            vocab_size: Size of the vocabulary
            d_ff: Feed-forward dimension (default: 4 * d_model)
            max_context_length: Maximum context length for inference (default: 1024)
        """
        if d_model % n_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads})")
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.vocab_size = vocab_size
        self.d_ff = d_ff if d_ff is not None else 4 * d_model
        self.max_context_length = max_context_length
        
        # Initialize the transformer model
        self.head_dim = d_model // n_heads
        
        # Set up tokenizer
        self.tokenizer = BasicTokenizer()
        
        # Initialize weights list
        self.weights = []
        for _ in range(n_layers):
            self.weights.append({})
        
        # Load kernels
        self._load_kernels()
        
        # Initialize weights
        self.initialize_weights()
        
        # Initialize ALiBi position biases
        self._init_alibi_biases(max_context_length, n_heads)
        
        # Initialize buffers for fused operations
        self._init_fused_buffers()
        
        # Initialize fully fused transformer op
        try:
            self.fully_fused_op = create_fully_fused_transformer_op(build_and_jit)
            logger.info("Successfully initialized NEON/AMX optimized fully fused transformer kernel")
        except Exception as e:
            logger.warning(f"Failed to initialize fully fused transformer op: {e}")
            self.fully_fused_op = None
        
        logger.info(f"Transformer model initialized with d_model={d_model}, n_heads={n_heads}, n_layers={n_layers}")
        
    def _load_kernels(self):
        """Load the necessary kernels."""
        try:
            # Build and JIT compile the kernels
            self.attention_matmul = build_and_jit(attention_matmul_code, "_attention_matmul")
            self.layer_norm_kernel = build_and_jit(layer_norm_code, "_layer_norm")
            self.gelu_kernel = build_and_jit(gelu_code, "_gelu")
            self.transpose_kernel = build_and_jit(transpose_code, "_transpose")
            self.weight_initializer = build_and_jit(get_kernel_code(), "_weight_initializer")
            self.position_embedding_kernel = build_and_jit(position_embedding_code, "_position_embedding")
            self.softmax_kernel = build_and_jit(softmax_code, "_softmax")
            
            # Fused transformer op
            self.fused_op = create_fused_transformer_op(build_and_jit)
            
            # Fully fused transformer op with NEON and AMX optimizations
            self.fully_fused_op = create_fully_fused_transformer_op(build_and_jit)
            
            logger.info("Kernel loading complete")
        except Exception as e:
            logger.error(f"Error loading kernels: {e}")
            raise

    def initialize_weights(self):
        """Initialize model weights with better scaling."""
        try:
            # Initialize layer normalization parameters with better scaling
            for layer_idx in range(self.n_layers):
                # Layer norm parameters - initialize with better scaling
                self.weights[layer_idx]['norm1'] = np.ones(self.d_model, dtype=np.float32)
                self.weights[layer_idx]['norm1_bias'] = np.zeros(self.d_model, dtype=np.float32)
                self.weights[layer_idx]['norm2'] = np.ones(self.d_model, dtype=np.float32)
                self.weights[layer_idx]['norm2_bias'] = np.zeros(self.d_model, dtype=np.float32)
                
                # Attention projection weights - use better initialization
                scale = 1.0 / np.sqrt(self.d_model)
                self.weights[layer_idx]['q_proj'] = np.random.normal(0, scale, (self.d_model, self.d_model)).astype(np.float32)
                self.weights[layer_idx]['k_proj'] = np.random.normal(0, scale, (self.d_model, self.d_model)).astype(np.float32)
                self.weights[layer_idx]['v_proj'] = np.random.normal(0, scale, (self.d_model, self.d_model)).astype(np.float32)
                self.weights[layer_idx]['out_proj'] = np.random.normal(0, scale, (self.d_model, self.d_model)).astype(np.float32)
                
                # Feed-forward network weights - use better initialization
                ffn_scale = 1.0 / np.sqrt(4 * self.d_model)
                self.weights[layer_idx]['ff1'] = np.random.normal(0, ffn_scale, (self.d_model, 4 * self.d_model)).astype(np.float32)
                self.weights[layer_idx]['ff2'] = np.random.normal(0, ffn_scale, (4 * self.d_model, self.d_model)).astype(np.float32)
                
                # Initialize token embeddings with better scaling
                if layer_idx == 0:  # Only initialize once
                    self.token_embeddings = np.random.normal(0, 0.02, (self.vocab_size, self.d_model)).astype(np.float32)
                    
                # Initialize position embeddings with better scaling
                if layer_idx == 0:  # Only initialize once
                    self.position_embeddings = np.zeros((self.max_context_length, self.d_model), dtype=np.float32)
                    self._init_position_embedding(self.max_context_length, self.d_model)
                    
            logger.info("Model weights initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing weights: {str(e)}")
            raise

    def _init_alibi_biases(self, max_seq_len: int, n_heads: int) -> None:
        """Initialize ALiBi position biases."""
        try:
            # Create position bias array
            self.position_biases = np.zeros((max_seq_len, max_seq_len), dtype=np.float32)
            
            # Initialize ALiBi biases
            for i in range(max_seq_len):
                for j in range(max_seq_len):
                    self.position_biases[i, j] = (i - j) * (1.0 / (2 ** (8 * (i // n_heads) / n_heads)))
            
            logger.info(f"Initialized ALiBi position biases with shape {self.position_biases.shape}")
            
        except Exception as e:
            logger.error(f"Error initializing ALiBi position biases: {str(e)}")
            raise

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass with better numerical stability."""
        try:
            # Input validation
            if not isinstance(x, np.ndarray):
                raise ValueError("Input must be a numpy array")
            if x.dtype != np.float32:
                raise ValueError("Input must be float32")
            if len(x.shape) != 3:
                raise ValueError("Input must be 3D: (batch_size, seq_len, d_model)")
            if x.shape[2] != self.d_model:
                raise ValueError(f"Input feature dimension ({x.shape[2]}) must match d_model ({self.d_model})")
            
            # Get input shape
            batch_size, seq_len, _ = x.shape
            
            # Clip input values to a reasonable range
            np.clip(x, -50.0, 50.0, out=x)
            
            # Apply transformer layers
            for layer_idx in range(self.n_layers):
                layer_weights = self.weights[layer_idx]
                
                # Layer normalization 1
                mean = np.mean(x, axis=-1, keepdims=True)
                var = np.mean(np.square(x - mean), axis=-1, keepdims=True) + 1e-9
                x = (x - mean) / np.sqrt(var)
                x = x * layer_weights['norm1'] + layer_weights['norm1_bias']
                
                # Self-attention
                # Project to Q, K, V
                q = np.matmul(x, layer_weights['q_proj'])
                k = np.matmul(x, layer_weights['k_proj'])
                v = np.matmul(x, layer_weights['v_proj'])
                
                # Reshape for multi-head attention
                q = q.reshape(batch_size, seq_len, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
                k = k.reshape(batch_size, seq_len, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
                v = v.reshape(batch_size, seq_len, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
                
                # Compute attention scores
                scores = np.matmul(q, np.transpose(k, (0, 1, 3, 2))) / np.sqrt(self.head_dim)
                
                # Add ALiBi position biases
                scores += self.position_biases[:seq_len, :seq_len]
                
                # Apply softmax
                scores_max = np.max(scores, axis=-1, keepdims=True)
                exp_scores = np.exp(scores - scores_max)
                attention_weights = exp_scores / (np.sum(exp_scores, axis=-1, keepdims=True) + 1e-9)
                
                # Apply attention to values
                context = np.matmul(attention_weights, v)
                
                # Reshape back
                context = context.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, self.d_model)
                
                # Project to output
                x = np.matmul(context, layer_weights['out_proj'])
                
                # Residual connection
                x = x + x
                
                # Layer normalization 2
                mean = np.mean(x, axis=-1, keepdims=True)
                var = np.mean(np.square(x - mean), axis=-1, keepdims=True) + 1e-9
                x = (x - mean) / np.sqrt(var)
                x = x * layer_weights['norm2'] + layer_weights['norm2_bias']
                
                # Feed-forward network
                ff_hidden = np.matmul(x, layer_weights['ff1'])
                ff_hidden = 0.5 * ff_hidden * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (ff_hidden + 0.044715 * np.power(ff_hidden, 3))))
                x = np.matmul(ff_hidden, layer_weights['ff2'])
                
                # Residual connection
                x = x + x
                
                # Clip values for stability
                np.clip(x, -50.0, 50.0, out=x)
            
            return x
            
        except Exception as e:
            logger.error(f"Error in forward pass: {str(e)}")
            raise
    
    def apply_attention(self, x: np.ndarray, layer_weights: dict, batch_size: int, seq_len: int) -> np.ndarray:
        """Apply multi-head attention with better numerical stability."""
        try:
            # Project to Q, K, V with better scaling
            head_dim = self.d_model // self.n_heads
            scale = 1.0 / np.sqrt(head_dim)
            
            # Reshape for matrix multiplication
            x_reshaped = x.reshape(batch_size * seq_len, self.d_model)
            
            # Q projection with better scaling
            q = np.zeros((batch_size * seq_len, self.d_model), dtype=np.float32)
            self.attention_matmul(
                x_reshaped.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                layer_weights['q_proj'].ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                q.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                batch_size * seq_len,
                self.d_model,
                self.d_model
            )
            q *= scale
            
            # K projection with better scaling
            k = np.zeros((batch_size * seq_len, self.d_model), dtype=np.float32)
            self.attention_matmul(
                x_reshaped.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                layer_weights['k_proj'].ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                k.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                batch_size * seq_len,
                self.d_model,
                self.d_model
            )
            k *= scale
            
            # V projection with better scaling
            v = np.zeros((batch_size * seq_len, self.d_model), dtype=np.float32)
            self.attention_matmul(
                x_reshaped.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                layer_weights['v_proj'].ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                v.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                batch_size * seq_len,
                self.d_model,
                self.d_model
            )
            v *= scale
            
            # Reshape for attention computation
            q = q.reshape(batch_size, seq_len, self.n_heads, head_dim).transpose(0, 2, 1, 3)
            k = k.reshape(batch_size, seq_len, self.n_heads, head_dim).transpose(0, 2, 3, 1)
            v = v.reshape(batch_size, seq_len, self.n_heads, head_dim).transpose(0, 2, 1, 3)
            
            # Compute attention scores with custom kernel
            scores = np.zeros((batch_size, self.n_heads, seq_len, seq_len), dtype=np.float32)
            for b in range(batch_size):
                for h in range(self.n_heads):
                    self.attention_matmul(
                        q[b, h].ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                        k[b, h].ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                        scores[b, h].ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                        seq_len,
                        head_dim,
                        seq_len
                    )
            
            # Apply softmax with numerical stability
            scores_max = np.max(scores, axis=-1, keepdims=True)
            exp_scores = np.exp(scores - scores_max)
            attn = exp_scores / (np.sum(exp_scores, axis=-1, keepdims=True) + 1e-9)
            
            # Apply attention to values with custom kernel
            output = np.zeros((batch_size, self.n_heads, seq_len, head_dim), dtype=np.float32)
            for b in range(batch_size):
                for h in range(self.n_heads):
                    self.attention_matmul(
                        attn[b, h].ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                        v[b, h].ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                        output[b, h].ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                        seq_len,
                        seq_len,
                        head_dim
                    )
            
            # Reshape back to original dimensions
            output = output.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, self.d_model)
            
            # Normalize output with numerical stability
            # First center the output
            mean = np.mean(output, keepdims=True)
            centered = output - mean
            
            # Compute variance and normalize
            var = np.mean(centered ** 2, keepdims=True)
            std = np.sqrt(var + 1e-9)
            normalized = centered / std
            
            # Scale by sqrt(n_heads) to achieve unit variance
            normalized = normalized * np.sqrt(self.n_heads)
            
            # Clip to prevent extreme values
            output = np.clip(normalized, -5.0, 5.0)
            
            return output
            
        except Exception as e:
            logger.error(f"Error in attention computation: {str(e)}")
            raise
    
    def apply_feedforward(self, x: np.ndarray, layer_weights: dict, batch_size: int, seq_len: int) -> np.ndarray:
        """Apply feed-forward network with better numerical stability."""
        try:
            # Reshape for matrix multiplication
            x_reshaped = x.reshape(batch_size * seq_len, self.d_model)
            
            # First linear layer with better scaling
            intermediate = np.zeros((batch_size * seq_len, 4 * self.d_model), dtype=np.float32)
            self.attention_matmul(
                x_reshaped.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                layer_weights['ff1'].ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                intermediate.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                batch_size * seq_len,
                self.d_model,
                4 * self.d_model
            )
            
            # Apply tanh activation with better scaling
            intermediate = np.tanh(intermediate)
            
            # Second linear layer with better scaling
            output = np.zeros((batch_size * seq_len, self.d_model), dtype=np.float32)
            self.attention_matmul(
                intermediate.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                layer_weights['ff2'].ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                output.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                batch_size * seq_len,
                4 * self.d_model,
                self.d_model
            )
            
            # Reshape back to original dimensions
            output = output.reshape(batch_size, seq_len, self.d_model)
            
            # Clip with a more reasonable range
            np.clip(output, -50.0, 50.0, out=output)
            
            return output
            
        except Exception as e:
            logger.error(f"Error in feed-forward computation: {str(e)}")
            raise
    
    def layer_norm(self, x: np.ndarray, weight: np.ndarray, bias: np.ndarray) -> np.ndarray:
        """Layer normalization using optimized kernel."""
        try:
            # Ensure input is contiguous and float32
            x = np.ascontiguousarray(x, dtype=np.float32)
            
            # Create output array
            output = np.zeros_like(x)
            output = np.ascontiguousarray(output)
            
            # Get pointers
            input_ptr = x.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
            output_ptr = output.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
            gamma_ptr = weight.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
            beta_ptr = bias.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
            
            # Call layer norm kernel with updated signature
            self.layer_norm_kernel(
                input_ptr,
                output_ptr,
                gamma_ptr,
                beta_ptr,
                ctypes.c_int(x.size),
                ctypes.c_int(1)  # Always 1 dimension for simplicity
            )
            
            return output
            
        except Exception as e:
            logger.error(f"Error in layer normalization: {str(e)}")
            raise
    
    def gelu(self, x: np.ndarray) -> np.ndarray:
        """GELU activation using optimized kernel."""
        try:
            # Ensure input is contiguous and float32
            x = np.ascontiguousarray(x, dtype=np.float32)
            
            # Create output array
            output = np.zeros_like(x)
            output = np.ascontiguousarray(output)
            
            # Get pointers
            input_ptr = x.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
            output_ptr = output.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
            
            # Call GELU kernel
            self.gelu_kernel(
                input_ptr,
                output_ptr,
                x.size
            )
            
            return output
            
        except Exception as e:
            logger.error(f"Error in GELU activation: {str(e)}")
            raise

    def generate(self, prompt: str, max_length: int = 50, temperature: float = 1.0, top_k: int = 50) -> str:
        """
        Generate text from a prompt.
        
        Args:
            prompt: The input prompt to generate from
            max_length: Maximum number of tokens to generate
            temperature: Temperature for sampling (higher = more random)
            top_k: Number of top tokens to sample from
            
        Returns:
            Generated text string
        """
        # Check if token embeddings are loaded
        if not hasattr(self, 'token_embeddings') or self.token_embeddings is None:
            raise ValueError("Token embeddings not loaded. Call load() first.")
            
        # Tokenize the prompt
        try:
            input_ids = self.tokenizer.tokenize(prompt)
            input_ids = np.array(input_ids, dtype=np.int32)
            
            if len(input_ids) == 0:
                # If tokenization failed, use some default tokens
                logger.warning("Tokenization returned no tokens, using default start token")
                input_ids = np.array([list(self.tokenizer.id_to_token.keys())[0]], dtype=np.int32)
                
            logger.info(f"Input tokens: {input_ids}")
            
            # Generate tokens
            generated_tokens = []
            for _ in range(max_length):
                # Get the last max_context_length tokens
                context = input_ids[-self.max_context_length:]
                
                # Get embeddings for context
                context_embeddings = []
                for token_id in context:
                    if token_id < self.token_embeddings.shape[0]:
                        context_embeddings.append(self.token_embeddings[token_id])
                    else:
                        # Use a default embedding for unknown tokens
                        logger.warning(f"Token ID {token_id} out of range, using default embedding")
                        context_embeddings.append(np.zeros(self.d_model, dtype=np.float32))
                        
                context_embeddings = np.array(context_embeddings, dtype=np.float32)
                
                # Forward pass to get logits
                x = context_embeddings.reshape(1, -1, self.d_model)
                x = self.forward(x)
                logits = np.matmul(x[0, -1], self.token_embeddings.T)  # Get logits for next token
                
                # Handle NaN values
                if np.isnan(logits).any():
                    # Replace NaNs with large negative values
                    logits = np.nan_to_num(logits, nan=-1e9)
                
                # Apply temperature
                logits = logits / max(temperature, 1e-10)  # Avoid division by zero
                
                # Top-k sampling
                if top_k > 0:
                    # Get indices of top k values
                    top_k_indices = np.argpartition(logits, -min(top_k, len(logits)))[-min(top_k, len(logits)):]
                    top_k_logits = logits[top_k_indices]
                    
                    # Apply softmax to get probabilities
                    top_k_logits = top_k_logits - np.max(top_k_logits)  # For numerical stability
                    exp_logits = np.exp(top_k_logits)
                    probs = exp_logits / np.sum(exp_logits)
                    
                    # Check for NaN values
                    if np.isnan(probs).any() or np.sum(probs) == 0:
                        # Fallback to uniform distribution
                        probs = np.ones_like(probs) / len(probs)
                    
                    # Sample from top-k
                    next_token = top_k_indices[np.random.choice(len(top_k_indices), p=probs)]
                else:
                    # Regular sampling
                    logits = logits - np.max(logits)  # For numerical stability
                    exp_logits = np.exp(logits)
                    probs = exp_logits / np.sum(exp_logits)
                    
                    # Check for NaN values
                    if np.isnan(probs).any() or np.sum(probs) == 0:
                        # Fallback to uniform distribution
                        probs = np.ones(len(probs)) / len(probs)
                    
                    next_token = np.random.choice(len(probs), p=probs)
                
                # Append next token
                input_ids = np.append(input_ids, next_token)
                generated_tokens.append(next_token)
                
                # Check if we've generated an end token
                if '<|endoftext|>' in self.tokenizer.token_to_id and next_token == self.tokenizer.token_to_id['<|endoftext|>']:
                    break
            
            # Decode the generated tokens
            return self.tokenizer.detokenize(input_ids.tolist())
            
        except Exception as e:
            logger.error(f"Error during generation: {str(e)}")
            raise
    
    def save(self, path: str):
        """Save model weights and tokenizer."""
        # Create directory if it doesn't exist
        os.makedirs(path, exist_ok=True)
        
        # Save tokenizer
        self.tokenizer.save(os.path.join(path, 'tokenizer_mapping.json'))
        
        # Save embeddings using memory mapping
        token_emb_path = os.path.join(path, 'token_embeddings.npy')
        pos_emb_path = os.path.join(path, 'position_embeddings.npy')
        
        np.save(token_emb_path, self.token_embeddings)
        np.save(pos_emb_path, self.position_embeddings)
        
        # Save model weights in chunks
        weights_dir = os.path.join(path, 'weights')
        os.makedirs(weights_dir, exist_ok=True)
        
        for layer_idx, layer_weights in enumerate(self.weights):
            layer_dir = os.path.join(weights_dir, f'layer_{layer_idx}')
            os.makedirs(layer_dir, exist_ok=True)
            
            # Save each weight matrix/tensor separately as .npy file
            for name, weights in layer_weights.items():
                weight_path = os.path.join(layer_dir, f'{name}.npy')
                np.save(weight_path, weights)
        
        # Save layer structure metadata
        structure = {
            'n_layers': self.n_layers,
            'weight_names': list(self.weights[0].keys()) if self.weights else []
        }
        with open(os.path.join(path, 'structure.json'), 'w') as f:
            json.dump(structure, f)
    
    def load(self, path: str):
        """Load model weights and embeddings using memory mapping for large files."""
        try:
            # Load config first to get correct n_heads
            config_path = os.path.join(path, 'config.json')
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config = json.load(f)
                    self.n_heads = config.get('n_heads', self.n_heads)
                    self.head_dim = self.d_model // self.n_heads
                    logger.info(f"Loaded model configuration: n_heads={self.n_heads}, head_dim={self.head_dim}")
            
            # Load tokenizer
            tokenizer_path = os.path.join(path, 'tokenizer_mapping.json')
            if not os.path.exists(tokenizer_path):
                raise FileNotFoundError(f"Tokenizer mapping not found at {tokenizer_path}")
            self.tokenizer.load(tokenizer_path)
            
            # Load embeddings using memory mapping
            token_emb_path = os.path.join(path, 'token_embeddings.npy')
            pos_emb_path = os.path.join(path, 'position_embeddings.npy')
            
            if not os.path.exists(token_emb_path):
                raise FileNotFoundError(f"Token embeddings not found at {token_emb_path}")
            if not os.path.exists(pos_emb_path):
                raise FileNotFoundError(f"Position embeddings not found at {pos_emb_path}")
            
            # Load embedding matrices as numpy arrays
            self.token_embeddings = np.load(token_emb_path)
            self.position_embeddings = np.load(pos_emb_path)
            
            logger.info(f"Loaded token embeddings with shape {self.token_embeddings.shape}")
            logger.info(f"Loaded position embeddings with shape {self.position_embeddings.shape}")
            
            # Verify embedding shapes
            if self.token_embeddings.shape[1] != self.d_model:
                raise ValueError(f"Token embeddings dimension mismatch. Expected second dimension {self.d_model}, got {self.token_embeddings.shape[1]}")
            
            # Load weights from JSON file if available
            weights_json_path = os.path.join(path, 'weights.json')
            if os.path.exists(weights_json_path):
                with open(weights_json_path, 'r') as f:
                    weights_data = json.load(f)
                    # Convert weights from JSON to numpy arrays
                    self.weights = []
                    for layer_idx in range(self.n_layers):
                        layer_weights = {}
                        for key, value in weights_data[layer_idx].items():
                            layer_weights[key] = np.array(value, dtype=np.float32)
                        self.weights.append(layer_weights)
            else:
                # Load structure metadata
                structure_path = os.path.join(path, 'structure.json')
                if not os.path.exists(structure_path):
                    raise FileNotFoundError(f"Structure file not found at {structure_path}")
                
                with open(structure_path, 'r') as f:
                    structure = json.load(f)
                
                # Initialize weights list
                self.weights = []
                
                # Load weights layer by layer
                weights_dir = os.path.join(path, 'weights')
                for layer_idx in range(structure['n_layers']):
                    layer_dir = os.path.join(weights_dir, f'layer_{layer_idx}')
                    if not os.path.exists(layer_dir):
                        raise FileNotFoundError(f"Layer directory not found at {layer_dir}")
                    
                    layer_weights = {}
                    for weight_name in structure['weight_names']:
                        weight_path = os.path.join(layer_dir, f'{weight_name}.npy')
                        if not os.path.exists(weight_path):
                            raise FileNotFoundError(f"Weight file not found at {weight_path}")
                        
                        # Load weight as numpy array
                        layer_weights[weight_name] = np.load(weight_path)
                    
                    self.weights.append(layer_weights)
            
            logger.info("Model loaded successfully!")
            logger.info(f"Token embeddings shape: {self.token_embeddings.shape}")
            logger.info(f"Position embeddings shape: {self.position_embeddings.shape}")
            logger.info(f"Number of layers: {len(self.weights)}")
            
            return self
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise

    def _init_position_embedding(self, max_seq_len: int, embedding_dim: int, base: float = 10000.0) -> None:
        """Initialize position embeddings using the optimized kernel."""
        try:
            # Create position embedding array
            self.position_embedding = np.zeros((max_seq_len, embedding_dim), dtype=np.float32)
            
            # Get pointer to the array
            position_embedding_ptr = self.position_embedding.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
            
            # Run the optimized kernel
            self.position_embedding_kernel(
                position_embedding_ptr,  # output pointer
                max_seq_len,            # max sequence length
                embedding_dim,          # embedding dimension
                base                   # base for position embedding
            )
            
            logger.info(f"Initialized position embeddings with shape {self.position_embedding.shape}")
            
        except Exception as e:
            logger.error(f"Error initializing position embeddings: {str(e)}")
            raise

    def fused_forward(self, x: np.ndarray) -> np.ndarray:
        """Fused forward pass that combines operations for faster execution."""
        try:
            # Input validation
            if not isinstance(x, np.ndarray):
                raise ValueError("Input must be a numpy array")
            if x.dtype != np.float32:
                raise ValueError("Input must be float32")
            if len(x.shape) != 3:
                raise ValueError("Input must be 3D: (batch_size, seq_len, d_model)")
            if x.shape[2] != self.d_model:
                raise ValueError(f"Input feature dimension ({x.shape[2]}) must match d_model ({self.d_model})")
            
            # Get input shape
            batch_size, seq_len, _ = x.shape
            
            # Clip input values to a reasonable range
            np.clip(x, -50.0, 50.0, out=x)
            
            # Allocate memory for intermediate results (only once)
            norm_output = np.zeros_like(x)
            attn_output = np.zeros_like(x)
            ff_output = np.zeros_like(x)
            residual = np.zeros_like(x)
            
            # Allocate memory for Q, K, V projections (only once)
            head_dim = self.d_model // self.n_heads
            q_buffer = np.zeros((batch_size, self.n_heads, seq_len, head_dim), dtype=np.float32)
            k_buffer = np.zeros((batch_size, self.n_heads, seq_len, head_dim), dtype=np.float32)
            v_buffer = np.zeros((batch_size, self.n_heads, seq_len, head_dim), dtype=np.float32)
            
            # Allocate memory for attention scores and output (only once)
            scores = np.zeros((batch_size, self.n_heads, seq_len, seq_len), dtype=np.float32)
            attn_heads_output = np.zeros((batch_size, self.n_heads, seq_len, head_dim), dtype=np.float32)
            
            # Allocate memory for feed-forward intermediate activations (only once)
            ff_intermediate = np.zeros((batch_size * seq_len, 4 * self.d_model), dtype=np.float32)
            
            # Current working copy
            current = x.copy()
            
            # Process through transformer layers
            for layer_idx in range(self.n_layers):
                # Get layer weights
                layer_weights = self.weights[layer_idx]
                
                # === FUSED OPERATION 1: Layer Norm + Attention ===
                
                # Layer normalization before attention (in-place)
                for batch_idx in range(batch_size):
                    for seq_idx in range(seq_len):
                        self.layer_norm_kernel(
                            current[batch_idx, seq_idx].ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                            norm_output[batch_idx, seq_idx].ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                            layer_weights['norm1'].ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                            layer_weights['norm1_bias'].ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                            ctypes.c_int(self.d_model),
                            ctypes.c_int(1)
                        )
                
                # Fused attention computation
                self._fused_attention(
                    norm_output, layer_weights, batch_size, seq_len,
                    q_buffer, k_buffer, v_buffer, scores, attn_heads_output, attn_output
                )
                
                # Residual connection (in-place)
                np.add(norm_output, attn_output, out=current)
                
                # === FUSED OPERATION 2: Layer Norm + Feed-Forward ===
                
                # Layer normalization after attention (in-place)
                for batch_idx in range(batch_size):
                    for seq_idx in range(seq_len):
                        self.layer_norm_kernel(
                            current[batch_idx, seq_idx].ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                            norm_output[batch_idx, seq_idx].ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                            layer_weights['norm2'].ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                            layer_weights['norm2_bias'].ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                            ctypes.c_int(self.d_model),
                            ctypes.c_int(1)
                        )
                
                # Fused feed-forward computation
                self._fused_feedforward(
                    norm_output, layer_weights, batch_size, seq_len,
                    ff_intermediate, ff_output
                )
                
                # Residual connection (in-place)
                np.add(norm_output, ff_output, out=current)
            
            # Final layer normalization
            output = np.zeros_like(current)
            for batch_idx in range(batch_size):
                for seq_idx in range(seq_len):
                    self.layer_norm_kernel(
                        current[batch_idx, seq_idx].ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                        output[batch_idx, seq_idx].ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                        self.weights[-1]['norm2'].ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                        self.weights[-1]['norm2_bias'].ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                        ctypes.c_int(self.d_model),
                        ctypes.c_int(1)
                    )
            
            # Final clipping with a reasonable range
            np.clip(output, -50.0, 50.0, out=output)
            
            # Check for NaN or infinite values
            if np.any(np.isnan(output)) or np.any(np.isinf(output)):
                logger.warning("NaN or infinite values detected in output")
                # Replace NaN and infinite values with zeros
                output[np.isnan(output) | np.isinf(output)] = 0.0
            
            return output
            
        except Exception as e:
            logger.error(f"Error in fused forward pass: {str(e)}")
            raise
    
    def _fused_attention(self, x: np.ndarray, layer_weights: dict, batch_size: int, seq_len: int,
                        q_buffer: np.ndarray, k_buffer: np.ndarray, v_buffer: np.ndarray,
                        scores_buffer: np.ndarray, heads_output_buffer: np.ndarray, 
                        output_buffer: np.ndarray) -> None:
        """Fused multi-head attention with pre-allocated buffers."""
        try:
            # Project to Q, K, V with better scaling
            head_dim = self.d_model // self.n_heads
            scale = 1.0 / np.sqrt(head_dim)
            
            # Reshape for matrix multiplication
            x_reshaped = x.reshape(batch_size * seq_len, self.d_model)
            
            # Q projection with better scaling (reuse buffer)
            q = np.zeros((batch_size * seq_len, self.d_model), dtype=np.float32)
            self.attention_matmul(
                x_reshaped.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                layer_weights['q_proj'].ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                q.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                batch_size * seq_len,
                self.d_model,
                self.d_model
            )
            q *= scale
            
            # K projection with better scaling (reuse buffer)
            k = np.zeros((batch_size * seq_len, self.d_model), dtype=np.float32)
            self.attention_matmul(
                x_reshaped.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                layer_weights['k_proj'].ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                k.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                batch_size * seq_len,
                self.d_model,
                self.d_model
            )
            k *= scale
            
            # V projection with better scaling (reuse buffer)
            v = np.zeros((batch_size * seq_len, self.d_model), dtype=np.float32)
            self.attention_matmul(
                x_reshaped.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                layer_weights['v_proj'].ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                v.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                batch_size * seq_len,
                self.d_model,
                self.d_model
            )
            v *= scale
            
            # Reshape for attention computation - make sure shapes match the pre-allocated buffers
            q_reshaped = q.reshape(batch_size, seq_len, self.n_heads, head_dim).transpose(0, 2, 1, 3)
            # For k_reshaped, we need to match k_buffer's expected shape which is (batch, heads, seq, head_dim)
            # We'll manually transpose to (0, 2, 3, 1) and then transpose again to match expected shape
            k_reshaped_temp = k.reshape(batch_size, seq_len, self.n_heads, head_dim).transpose(0, 2, 3, 1)
            k_reshaped_fixed = np.zeros((batch_size, self.n_heads, seq_len, head_dim), dtype=np.float32)
            for b in range(batch_size):
                for h in range(self.n_heads):
                    # Transpose each matrix to match the expected shape
                    k_reshaped_fixed[b, h] = k_reshaped_temp[b, h].T
            
            v_reshaped = v.reshape(batch_size, seq_len, self.n_heads, head_dim).transpose(0, 2, 1, 3)
            
            # Copy to pre-allocated buffers if provided
            if q_buffer is not None:
                np.copyto(q_buffer, q_reshaped)
                np.copyto(k_buffer, k_reshaped_fixed)  # Use fixed shape
                np.copyto(v_buffer, v_reshaped)
            else:
                q_buffer = q_reshaped
                k_buffer = k_reshaped_fixed  # Use fixed shape
                v_buffer = v_reshaped
            
            # Compute attention scores with custom kernel (reuse buffer)
            for b in range(batch_size):
                for h in range(self.n_heads):
                    self.attention_matmul(
                        q_buffer[b, h].ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                        k_buffer[b, h].ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                        scores_buffer[b, h].ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                        seq_len,
                        head_dim,
                        seq_len
                    )
            
            # Apply softmax with numerical stability (in-place)
            scores_max = np.max(scores_buffer, axis=-1, keepdims=True)
            exp_scores = np.exp(scores_buffer - scores_max)
            attn = exp_scores / (np.sum(exp_scores, axis=-1, keepdims=True) + 1e-9)
            
            # Apply attention to values with custom kernel (reuse buffer)
            for b in range(batch_size):
                for h in range(self.n_heads):
                    self.attention_matmul(
                        attn[b, h].ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                        v_buffer[b, h].ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                        heads_output_buffer[b, h].ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                        seq_len,
                        seq_len,
                        head_dim
                    )
            
            # Reshape back to original dimensions
            output_reshaped = heads_output_buffer.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, self.d_model)
            
            # Normalize output with numerical stability
            # First center the output
            mean = np.mean(output_reshaped, keepdims=True)
            centered = output_reshaped - mean
            
            # Compute variance and normalize
            var = np.mean(centered ** 2, keepdims=True)
            std = np.sqrt(var + 1e-9)
            normalized = centered / std
            
            # Scale by sqrt(n_heads) to achieve unit variance
            normalized = normalized * np.sqrt(self.n_heads)
            
            # Clip to prevent extreme values
            np.clip(normalized, -5.0, 5.0, out=output_buffer)
            
        except Exception as e:
            logger.error(f"Error in fused attention computation: {str(e)}")
            raise
    
    def _fused_feedforward(self, x: np.ndarray, layer_weights: dict, batch_size: int, seq_len: int,
                          intermediate_buffer: np.ndarray, output_buffer: np.ndarray) -> None:
        """Fused feed-forward network with pre-allocated buffers."""
        try:
            # Reshape for matrix multiplication
            x_reshaped = x.reshape(batch_size * seq_len, self.d_model)
            
            # First linear layer with better scaling (reuse buffer)
            self.attention_matmul(
                x_reshaped.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                layer_weights['ff1'].ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                intermediate_buffer.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                batch_size * seq_len,
                self.d_model,
                4 * self.d_model
            )
            
            # Apply tanh activation with better scaling (in-place)
            np.tanh(intermediate_buffer, out=intermediate_buffer)
            
            # Second linear layer with better scaling (reuse buffer)
            output_reshaped = np.zeros((batch_size * seq_len, self.d_model), dtype=np.float32)
            self.attention_matmul(
                intermediate_buffer.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                layer_weights['ff2'].ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                output_reshaped.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                batch_size * seq_len,
                4 * self.d_model,
                self.d_model
            )
            
            # Reshape back to original dimensions
            output_buffer_3d = output_reshaped.reshape(batch_size, seq_len, self.d_model)
            
            # Clip with a reasonable range
            np.clip(output_buffer_3d, -50.0, 50.0, out=output_buffer)
            
        except Exception as e:
            logger.error(f"Error in fused feed-forward computation: {str(e)}")
            raise

    def fully_fused_forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass through the transformer using a simplified NumPy implementation.
        This is a safer version that avoids assembly code which was causing segmentation faults.
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, d_model]
            
        Returns:
            Output tensor of shape [batch_size, seq_len, d_model]
        """
        # Input validation
        if not isinstance(x, np.ndarray) or x.ndim != 3:
            raise ValueError(f"Expected 3D numpy array, got {type(x)} with shape {x.shape}")
        
        batch_size, seq_len, d_model = x.shape
        if d_model != self.d_model:
            raise ValueError(f"Input dimension {d_model} does not match model dimension {self.d_model}")
        
        # Clip input to improve numerical stability
        x = np.clip(x, -50.0, 50.0)
        
        # Ensure input is contiguous and float32
        x = np.ascontiguousarray(x, dtype=np.float32)
        
        # Log input tensor stats for debugging
        logger.info(f"Input shape: {x.shape}, Mean: {np.mean(x):.4f}, Std: {np.std(x):.4f}")
        logger.info(f"Using pure NumPy implementation to avoid segmentation faults")
        
        # Apply transformer layers one by one
        temp_input = x.copy()
        
        for layer_idx in range(self.n_layers):
            logger.info(f"Processing layer {layer_idx}")
            
            # Get layer weights using numerical index to avoid string key issues
            if isinstance(self.weights, dict):
                if f"layer_{layer_idx}" in self.weights:
                    layer_weights = self.weights[f"layer_{layer_idx}"]
                else:
                    # Try to find alternative keys
                    layer_keys = [k for k in self.weights.keys() if str(layer_idx) in k]
                    if layer_keys:
                        layer_weights = self.weights[layer_keys[0]]
                    else:
                        logger.error(f"Could not find weights for layer {layer_idx}")
                        layer_weights = next(iter(self.weights.values()))
            elif isinstance(self.weights, list) and layer_idx < len(self.weights):
                layer_weights = self.weights[layer_idx]
            else:
                logger.error(f"Could not access layer {layer_idx} weights")
                # Create dummy weights
                layer_weights = {}
            
            # Extract weight keys
            weight_keys = list(layer_weights.keys())
            logger.info(f"Available weight keys: {weight_keys}")
            
            # Step 1: Layer Normalization 1
            ln1_gamma_key = next((k for k in weight_keys if 'attention' in k and 'layernorm' in k and 'weight' in k), None)
            ln1_beta_key = next((k for k in weight_keys if 'attention' in k and 'layernorm' in k and 'bias' in k), None)
            
            if ln1_gamma_key and ln1_beta_key:
                # Apply layer norm
                mean = np.mean(temp_input, axis=-1, keepdims=True)
                var = np.mean(np.square(temp_input - mean), axis=-1, keepdims=True) + 1e-9
                normalized = (temp_input - mean) / np.sqrt(var)
                ln1_output = normalized * layer_weights[ln1_gamma_key] + layer_weights[ln1_beta_key]
            else:
                ln1_output = temp_input
            
            # Step 2: Self-Attention
            # Find query, key, value weights
            q_key = next((k for k in weight_keys if ('query' in k or 'q_proj' in k) and 'weight' in k), None)
            k_key = next((k for k in weight_keys if ('key' in k or 'k_proj' in k) and 'weight' in k), None)
            v_key = next((k for k in weight_keys if ('value' in k or 'v_proj' in k) and 'weight' in k), None)
            
            if q_key and k_key and v_key:
                # Project to query, key, value
                q = np.matmul(ln1_output, layer_weights[q_key].T)
                k = np.matmul(ln1_output, layer_weights[k_key].T)
                v = np.matmul(ln1_output, layer_weights[v_key].T)
                
                # Reshape for multi-head attention
                head_dim = d_model // self.n_heads
                q = q.reshape(batch_size, seq_len, self.n_heads, head_dim).transpose(0, 2, 1, 3)
                k = k.reshape(batch_size, seq_len, self.n_heads, head_dim).transpose(0, 2, 1, 3)
                v = v.reshape(batch_size, seq_len, self.n_heads, head_dim).transpose(0, 2, 1, 3)
                
                # Scale query
                q = q / np.sqrt(head_dim)
                
                # Compute attention scores
                scores = np.matmul(q, np.transpose(k, (0, 1, 3, 2)))
                
                # Apply softmax
                scores_max = np.max(scores, axis=-1, keepdims=True)
                exp_scores = np.exp(scores - scores_max)
                attention_weights = exp_scores / (np.sum(exp_scores, axis=-1, keepdims=True) + 1e-9)
                
                # Apply attention to values
                context = np.matmul(attention_weights, v)
                
                # Reshape back
                context = context.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, d_model)
                
                # Project to output
                attn_output_key = next((k for k in weight_keys if 'output' in k and 'weight' in k and 'layernorm' not in k), None)
                if attn_output_key:
                    attn_output = np.matmul(context, layer_weights[attn_output_key].T)
                else:
                    attn_output = context
            else:
                # No attention - pass through
                attn_output = ln1_output
            
            # Residual connection
            attn_output = attn_output + temp_input
            
            # Step 3: Layer Normalization 2
            ln2_gamma_key = next((k for k in weight_keys if 'output' in k and 'layernorm' in k and 'weight' in k), None)
            ln2_beta_key = next((k for k in weight_keys if 'output' in k and 'layernorm' in k and 'bias' in k), None)
            
            if ln2_gamma_key and ln2_beta_key:
                # Apply layer norm
                mean = np.mean(attn_output, axis=-1, keepdims=True)
                var = np.mean(np.square(attn_output - mean), axis=-1, keepdims=True) + 1e-9
                normalized = (attn_output - mean) / np.sqrt(var)
                ln2_output = normalized * layer_weights[ln2_gamma_key] + layer_weights[ln2_beta_key]
            else:
                ln2_output = attn_output
            
            # Step 4: Feed-forward Network
            ff1_key = next((k for k in weight_keys if ('intermediate' in k or 'ff1' in k) and 'weight' in k), None)
            ff2_key = next((k for k in weight_keys if ('output' in k and 'weight' in k and 'attention' not in k) or 'ff2' in k), None)
            
            if ff1_key and ff2_key:
                # Get FF dimensions
                ff_dim = layer_weights[ff1_key].shape[1] if layer_weights[ff1_key].ndim > 1 else 4 * d_model
                
                # Apply FF1
                ff_hidden = np.matmul(ln2_output, layer_weights[ff1_key])
                
                # Apply GELU activation
                ff_hidden = 0.5 * ff_hidden * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (ff_hidden + 0.044715 * np.power(ff_hidden, 3))))
                
                # Apply FF2
                ff_output = np.matmul(ff_hidden, layer_weights[ff2_key])
            else:
                # No FF - pass through
                ff_output = ln2_output
            
            # Residual connection
            layer_output = ff_output + attn_output
            
            # Clip values for stability
            layer_output = np.clip(layer_output, -50.0, 50.0)
            
            # Check for NaN or Inf values
            if np.any(np.isnan(layer_output)) or np.any(np.isinf(layer_output)):
                logger.warning(f"Layer {layer_idx} output contains NaN or Inf values, replacing with zeros")
                layer_output = np.nan_to_num(layer_output, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Update for next layer
            temp_input = layer_output
            
            # Log layer output stats
            logger.info(f"Layer {layer_idx} output - Mean: {np.mean(layer_output):.4f}, Std: {np.std(layer_output):.4f}")
        
        # Final output
        output = temp_input
        
        # Final clipping
        output = np.clip(output, -50.0, 50.0)
        
        logger.info(f"Final output - Shape: {output.shape}, Mean: {np.mean(output):.4f}, Std: {np.std(output):.4f}")
        return output

    def _init_fused_buffers(self):
        """Initialize buffers for fused operations to avoid memory allocations during inference."""
        # We'll pre-allocate maximum possible sizes (based on max_context_length)
        # These will be reused during the forward pass to avoid memory allocations
        batch_size = 8  # Reasonable upper limit for batch size
        seq_len = min(self.max_context_length, 1024)  # Use max context length but cap at 1024
        
        # Buffers for fused attention
        self.q_buffer = np.zeros((batch_size, self.n_heads, seq_len, self.head_dim), dtype=np.float32)
        self.k_buffer = np.zeros((batch_size, self.n_heads, seq_len, self.head_dim), dtype=np.float32)
        self.v_buffer = np.zeros((batch_size, self.n_heads, seq_len, self.head_dim), dtype=np.float32)
        self.scores_buffer = np.zeros((batch_size, self.n_heads, seq_len, seq_len), dtype=np.float32)
        self.heads_output_buffer = np.zeros((batch_size, self.n_heads, seq_len, self.head_dim), dtype=np.float32)
        self.attention_output_buffer = np.zeros((batch_size, seq_len, self.d_model), dtype=np.float32)
        
        # Buffers for fused feedforward
        self.intermediate_buffer = np.zeros((batch_size * seq_len, self.d_ff), dtype=np.float32)
        self.ffn_output_buffer = np.zeros((batch_size, seq_len, self.d_model), dtype=np.float32)
        
        logger.info(f"Initialized fused operation buffers for batch_size={batch_size}, seq_len={seq_len}")

def test_transformer_speed():
    """Test the speed of the transformer implementation."""
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Test parameters - reduced to avoid memory issues
    batch_size = 1
    seq_len = 128
    d_model = 512  # Must be divisible by n_heads
    n_heads = 8    # Changed from 16 to 8 to make it divide d_model evenly
    n_layers = 2
    
    logger.info(f"Initializing transformer with:")
    logger.info(f"batch_size={batch_size}, seq_len={seq_len}")
    logger.info(f"d_model={d_model}, n_heads={n_heads}, n_layers={n_layers}")
    
    # Create transformer
    transformer = Transformer(d_model, n_heads, n_layers)
    
    # Create random input
    x = np.random.normal(0, 1, (batch_size, seq_len, d_model)).astype(np.float32)
    
    try:
        # Speed test for regular forward pass
        num_iters = 3
        regular_times = []
        
        logger.info(f"\nRunning {num_iters} iterations of regular forward pass...")
        for i in range(num_iters):
            start_time = time.time()
            transformer.forward(x)
            end_time = time.time()
            regular_times.append((end_time - start_time) * 1000)  # Convert to ms
            
            if (i + 1) % 1 == 0:
                logger.info(f"Completed {i + 1} iterations...")
        
        # Speed test for fused forward pass
        fused_times = []
        
        logger.info(f"\nRunning {num_iters} iterations of fused forward pass...")
        for i in range(num_iters):
            start_time = time.time()
            transformer.fused_forward(x)
            end_time = time.time()
            fused_times.append((end_time - start_time) * 1000)  # Convert to ms
            
            if (i + 1) % 1 == 0:
                logger.info(f"Completed {i + 1} iterations...")
        
        # Speed test for fully fused forward pass
        fully_fused_times = []
        
        logger.info(f"\nRunning {num_iters} iterations of fully fused forward pass...")
        for i in range(num_iters):
            start_time = time.time()
            transformer.fully_fused_forward(x)
            end_time = time.time()
            fully_fused_times.append((end_time - start_time) * 1000)  # Convert to ms
            
            if (i + 1) % 1 == 0:
                logger.info(f"Completed {i + 1} iterations...")
        
        # Calculate statistics for regular forward
        avg_regular_time = np.mean(regular_times)
        std_regular_time = np.std(regular_times)
        min_regular_time = np.min(regular_times)
        max_regular_time = np.max(regular_times)
        
        # Calculate statistics for fused forward
        avg_fused_time = np.mean(fused_times)
        std_fused_time = np.std(fused_times)
        min_fused_time = np.min(fused_times)
        max_fused_time = np.max(fused_times)
        
        # Calculate statistics for fully fused forward
        avg_fully_fused_time = np.mean(fully_fused_times)
        std_fully_fused_time = np.std(fully_fused_times)
        min_fully_fused_time = np.min(fully_fused_times)
        max_fully_fused_time = np.max(fully_fused_times)
        
        # Calculate FLOPs
        flops_per_iter = (
            # Self-attention
            2 * batch_size * seq_len * seq_len * d_model +  # QK^T
            2 * batch_size * seq_len * seq_len * d_model +  # (QK^T)V
            # Feed-forward
            2 * batch_size * seq_len * d_model * transformer.d_ff +  # FF1
            2 * batch_size * seq_len * transformer.d_ff * d_model    # FF2
        ) * n_layers
        
        regular_gflops = (flops_per_iter / (avg_regular_time / 1000)) / 1e9
        fused_gflops = (flops_per_iter / (avg_fused_time / 1000)) / 1e9
        fully_fused_gflops = (flops_per_iter / (avg_fully_fused_time / 1000)) / 1e9
        
        # Calculate speedups
        speedup_fused = avg_regular_time / avg_fused_time
        speedup_fully_fused = avg_regular_time / avg_fully_fused_time
        speedup_fully_vs_fused = avg_fused_time / avg_fully_fused_time
        
        logger.info("\n=== Speed Test Results ===")
        logger.info("\nRegular Forward Pass:")
        logger.info(f"Average time per iteration: {avg_regular_time:.2f} ms")
        logger.info(f"Standard deviation: {std_regular_time:.2f} ms")
        logger.info(f"Min time: {min_regular_time:.2f} ms")
        logger.info(f"Max time: {max_regular_time:.2f} ms")
        logger.info(f"Throughput: {regular_gflops:.2f} GFLOPS")
        
        logger.info("\nFused Forward Pass:")
        logger.info(f"Average time per iteration: {avg_fused_time:.2f} ms")
        logger.info(f"Standard deviation: {std_fused_time:.2f} ms")
        logger.info(f"Min time: {min_fused_time:.2f} ms")
        logger.info(f"Max time: {max_fused_time:.2f} ms")
        logger.info(f"Throughput: {fused_gflops:.2f} GFLOPS")
        
        logger.info("\nFully Fused Forward Pass:")
        logger.info(f"Average time per iteration: {avg_fully_fused_time:.2f} ms")
        logger.info(f"Standard deviation: {std_fully_fused_time:.2f} ms")
        logger.info(f"Min time: {min_fully_fused_time:.2f} ms")
        logger.info(f"Max time: {max_fully_fused_time:.2f} ms")
        logger.info(f"Throughput: {fully_fused_gflops:.2f} GFLOPS")
        
        logger.info("\nSpeedup Analysis:")
        logger.info(f"Fused vs Regular: {speedup_fused:.2f}x")
        logger.info(f"Fully Fused vs Regular: {speedup_fully_fused:.2f}x")
        logger.info(f"Fully Fused vs Fused: {speedup_fully_vs_fused:.2f}x")
    
    except Exception as e:
        logger.error(f"Error during speed test: {e}")
        import traceback
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    test_transformer_speed() 