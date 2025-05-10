import numpy as np
import ctypes
import logging
from typing import Tuple, Optional
import gc
import torch

from OPENtransformer.arm64_engine.core.asm.kernels.vision.vision_kernels_asm import build_vit_kernels

logger = logging.getLogger(__name__)

class VisionTransformerSIMD:
    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 16,
        num_channels: int = 3,
        embed_dim: int = 768,
        num_heads: int = 12,
        num_layers: int = 12,
        num_classes: int = 1000,
        dropout: float = 0.1,
    ):
        """
        Initialize a SIMD-optimized Vision Transformer model.
        """
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.dropout = dropout
        self.training = True  # Add training flag
        
        # Calculate number of patches
        self.num_patches = ((image_size - patch_size) // patch_size + 1) ** 2
        
        # Calculate head dimension
        self.head_dim = embed_dim // num_heads
        
        # Initialize weights with normalization
        self.patch_embedding_weights = np.random.normal(0, 0.02, (patch_size * patch_size * num_channels, embed_dim))
        self.patch_embedding_weights = self.patch_embedding_weights / np.linalg.norm(self.patch_embedding_weights, axis=0, keepdims=True)
        
        self.position_embeddings = np.random.normal(0, 0.02, (1, (image_size // patch_size) ** 2 + 1, embed_dim))
        self.position_embeddings = self.position_embeddings / np.linalg.norm(self.position_embeddings, axis=2, keepdims=True)
        
        self.qkv_weights = np.random.normal(0, 0.02, (num_layers, 3, num_heads, embed_dim // num_heads, embed_dim))
        self.qkv_weights = self.qkv_weights / np.linalg.norm(self.qkv_weights, axis=3, keepdims=True)
        
        self.mlp_weights1 = np.random.normal(0, 0.02, (num_layers, embed_dim * 4, embed_dim))
        self.mlp_weights1 = self.mlp_weights1 / np.linalg.norm(self.mlp_weights1, axis=1, keepdims=True)
        
        self.mlp_weights2 = np.random.normal(0, 0.02, (num_layers, embed_dim, embed_dim * 4))
        self.mlp_weights2 = self.mlp_weights2 / np.linalg.norm(self.mlp_weights2, axis=0, keepdims=True)
        
        self.output_weights = np.random.normal(0, 0.02, (embed_dim, num_classes))
        self.output_weights = self.output_weights / np.linalg.norm(self.output_weights, axis=0, keepdims=True)
        
        # Layer normalization parameters
        self.layer_norm1 = np.ones((num_layers, embed_dim))
        self.layer_norm2 = np.ones((num_layers, embed_dim))
        self.layer_norm3 = np.ones(embed_dim)
        
        # Attention scaling factor
        self.attention_scale = 1.0 / np.sqrt(embed_dim // num_heads)
        
        # Build SIMD kernels
        self.kernels = build_vit_kernels()
        
        logger.info(f"SIMD Vision Transformer initialized with {num_layers} layers")
    
    def train(self):
        """Set the model to training mode."""
        self.training = True
    
    def eval(self):
        """Set the model to evaluation mode."""
        self.training = False
    
    def _patch_embedding(self, x):
        """Apply SIMD-optimized patch embedding."""
        batch_size = x.shape[0]
        patches = np.zeros((batch_size, self.num_patches, self.embed_dim), dtype=np.float32)
        
        for b in range(batch_size):
            self.kernels['patch_embedding'](
                x[b].ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                patches[b].ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                self.patch_size,
                self.patch_size,  # stride = patch_size
                self.num_channels,
                self.embed_dim,
                self.image_size,
                self.image_size,
                self.patch_embedding_weights.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
            )
        
        # Add CLS token
        cls_token = np.random.normal(0, 0.02, (batch_size, 1, self.embed_dim)).astype(np.float32)
        patches = np.concatenate([cls_token, patches], axis=1)
        
        return patches
    
    def _attention(self, x, layer_idx):
        """Multi-head attention with SIMD optimization"""
        try:
            # Layer normalization
            x = x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-6) * self.layer_norm1[layer_idx]
            
            # Reshape input for multi-head attention
            batch_size, seq_len, _ = x.shape
            
            # Project to Q, K, V
            qkv = x @ self.qkv_weights[layer_idx].reshape(self.embed_dim, 3 * self.embed_dim)
            qkv = qkv.reshape(batch_size, seq_len, 3, self.num_heads, self.embed_dim // self.num_heads)
            qkv = np.transpose(qkv, (2, 0, 3, 1, 4))
            q, k, v = qkv[0], qkv[1], qkv[2]
            
            # Scale attention scores
            attn = (q @ k.transpose(0, 1, 3, 2)) * self.attention_scale
            
            # Subtract max for numerical stability
            attn_max = np.max(attn, axis=-1, keepdims=True)
            attn = attn - attn_max
            
            # Clip attention scores
            attn = np.clip(attn, -30, 30)
            
            # Apply softmax with small epsilon
            exp_attn = np.exp(attn)
            exp_attn = np.where(np.isfinite(exp_attn), exp_attn, 1e-9)
            attn = exp_attn / (np.sum(exp_attn, axis=-1, keepdims=True) + 1e-9)
            
            # Apply dropout
            if self.dropout > 0:
                mask = np.random.binomial(1, 1-self.dropout, attn.shape)
                attn = attn * mask / (1-self.dropout)
            
            # Compute output
            out = attn @ v
            out = np.transpose(out, (0, 2, 1, 3))
            out = out.reshape(batch_size, seq_len, self.embed_dim)
            
            # Layer normalization
            out = out / (np.linalg.norm(out, axis=-1, keepdims=True) + 1e-6) * self.layer_norm2[layer_idx]
            
            return out
            
        except Exception as e:
            logger.error(f"Error in attention computation: {str(e)}")
            raise
        finally:
            # Clean up intermediate tensors
            if 'qkv' in locals():
                del qkv
            if 'q' in locals():
                del q
            if 'k' in locals():
                del k
            if 'v' in locals():
                del v
            if 'attn' in locals():
                del attn
            if 'exp_attn' in locals():
                del exp_attn
            gc.collect()
    
    def _mlp(self, x, layer_idx):
        """Apply SIMD-optimized MLP/feed-forward layer."""
        batch_size, seq_len, _ = x.shape
        
        # Layer normalization
        x = x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-6) * self.layer_norm2[layer_idx]
        
        # First linear layer
        x = x @ self.mlp_weights1[layer_idx].T
        x = np.maximum(x, 0)  # ReLU
        
        # Apply dropout
        if self.dropout > 0:
            mask = np.random.binomial(1, 1-self.dropout, x.shape)
            x = x * mask / (1-self.dropout)
        
        # Second linear layer
        x = x @ self.mlp_weights2[layer_idx].T
        
        # Layer normalization
        x = x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-6) * self.layer_norm1[layer_idx]
        
        return x
    
    def forward(self, x):
        """Forward pass with improved memory management and input preprocessing"""
        try:
            # Input validation and preprocessing
            if not isinstance(x, np.ndarray):
                raise ValueError("Input must be a numpy array")
            if x.ndim != 4:
                raise ValueError(f"Input must be 4D tensor, got {x.ndim}D")
            if x.shape[1] != self.num_channels:
                raise ValueError(f"Input must have {self.num_channels} channels, got {x.shape[1]}")
            if x.shape[2] != self.image_size or x.shape[3] != self.image_size:
                raise ValueError(f"Input must be {self.image_size}x{self.image_size}, got {x.shape[2]}x{x.shape[3]}")
            
            # Ensure contiguous memory layout
            x = np.ascontiguousarray(x)
            
            # Clip input values
            x = np.clip(x, -10, 10)
            
            # Patch embedding
            x = self._patch_embedding(x)
            
            # Add position embeddings
            x = x + self.position_embeddings
            
            # Process transformer layers
            for i in range(self.num_layers):
                # Attention block
                residual = x
                x = self._attention(x, i)
                x = residual + x
                
                # MLP block
                residual = x
                x = self._mlp(x, i)
                x = residual + x
                
                # Force memory cleanup after each layer
                gc.collect()
            
            # Final layer normalization
            x = x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-6) * self.layer_norm3
            
            # Classification head
            x = np.mean(x, axis=1)  # Global average pooling
            x = x @ self.output_weights
            
            return x
            
        except Exception as e:
            logger.error(f"Error in forward pass: {str(e)}")
            raise
        finally:
            gc.collect()

    def extract_features(self, x):
        """Extract visual features from the transformer.
        
        Args:
            x: Input tensor of shape (batch_size, channels, height, width)
            
        Returns:
            features: Extracted features of shape (batch_size, embed_dim)
        """
        # Convert to numpy if needed
        if isinstance(x, torch.Tensor):
            x = x.cpu().numpy()
        
        # Ensure input is float32
        x = x.astype(np.float32)
        
        # Get batch size
        batch_size = x.shape[0]
        
        # Reshape to (batch_size, channels, height, width)
        if len(x.shape) == 3:
            x = x.reshape(1, *x.shape)
        
        # Apply patch embedding
        x = self._patch_embedding(x)
        
        # Add position embeddings
        x = x + self.position_embeddings
        
        # Apply transformer layers
        for i in range(self.num_layers):
            x = self._attention(x, i)
        
        # Apply final layer norm
        x = x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-6) * self.layer_norm3
        
        # Extract class token features
        features = x[:, 0, :]  # Shape: (batch_size, embed_dim)
        
        return features 