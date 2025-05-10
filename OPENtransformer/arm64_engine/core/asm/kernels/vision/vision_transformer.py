import numpy as np
import ctypes
import logging
import os
from typing import Tuple, Optional
import pickle

from OPENtransformer.arm64_engine.core.asm.kernels.transformer import TransformerLayer
from OPENtransformer.arm64_engine.core.asm.kernels.vision.patch_embedding import patch_embedding
from OPENtransformer.arm64_engine.core.asm.kernels.vision.vision_kernels_asm import conv2d, max_pool2d, batch_norm
from OPENtransformer.arm64_engine.core.asm.kernels.layer_norm import LayerNorm, layer_norm_code
from OPENtransformer.core.asm.assembler.builder import build_and_jit

logger = logging.getLogger(__name__)

class MLP:
    def __init__(self, input_dim, hidden_dim, output_dim):
        """Simple MLP for classification head"""
        self.weights1 = np.random.randn(input_dim, hidden_dim).astype(np.float32)
        self.weights2 = np.random.randn(hidden_dim, output_dim).astype(np.float32)
        self.bias1 = np.zeros(hidden_dim, dtype=np.float32)
        self.bias2 = np.zeros(output_dim, dtype=np.float32)
        self.shape = (input_dim, output_dim)  # Add shape attribute
    
    def forward(self, x):
        """Forward pass through MLP"""
        x = np.dot(x, self.weights1) + self.bias1
        x = np.maximum(x, 0)  # ReLU
        x = np.dot(x, self.weights2) + self.bias2
        return x
    
    def get_weights(self):
        """Get all weights"""
        return {
            'weights1': self.weights1,
            'weights2': self.weights2,
            'bias1': self.bias1,
            'bias2': self.bias2
        }
    
    def set_weights(self, weights):
        """Set all weights"""
        self.weights1 = weights['weights1']
        self.weights2 = weights['weights2']
        self.bias1 = weights['bias1']
        self.bias2 = weights['bias2']

class VisionTransformer:
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
        use_2d_pos_emb: bool = True
    ):
        """
        Initialize a Vision Transformer model.
        
        Args:
            image_size: Input image size (assumed square)
            patch_size: Size of patches to extract
            num_channels: Number of input channels (3 for RGB)
            embed_dim: Model dimension
            num_heads: Number of attention heads
            num_layers: Number of transformer layers
            num_classes: Number of output classes
            dropout: Dropout rate
            use_2d_pos_emb: Whether to use 2D positional embeddings
        """
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.dropout = dropout
        self.use_2d_pos_emb = use_2d_pos_emb
        
        # Calculate number of patches
        self.num_patches = ((image_size - patch_size) // patch_size + 1) ** 2
        
        # Initialize patch embedding weights
        self.patch_embedding_weights = np.random.normal(
            0, 0.02, 
            (embed_dim, num_channels * patch_size * patch_size)
        ).astype(np.float32)
        
        # Initialize class token
        self.class_token = np.random.randn(1, 1, embed_dim).astype(np.float32)
        
        # Initialize position embeddings
        # Add 1 for the class token
        if use_2d_pos_emb:
            # Create 2D positional embeddings
            grid_h = np.arange(image_size // patch_size)
            grid_w = np.arange(image_size // patch_size)
            grid = np.stack(np.meshgrid(grid_h, grid_w), axis=0)
            grid = grid.reshape(2, -1).T
            
            # Create sinusoidal embeddings for each position
            position_embeddings = np.random.normal(
                0, 0.02, 
                (self.num_patches + 1, embed_dim)
            ).astype(np.float32)
            for i in range(embed_dim // 2):
                position_embeddings[1:, 2*i] = np.sin(grid[:, 0] / (10000 ** (2*i/embed_dim)))
                position_embeddings[1:, 2*i+1] = np.cos(grid[:, 0] / (10000 ** (2*i/embed_dim)))
        else:
            # Use 1D positional embeddings
            position_embeddings = np.random.normal(
                0, 0.02, 
                (self.num_patches + 1, embed_dim)
            ).astype(np.float32)
            for i in range(embed_dim // 2):
                position_embeddings[:, 2*i] = np.sin(np.arange(self.num_patches + 1) / (10000 ** (2*i/embed_dim)))
                position_embeddings[:, 2*i+1] = np.cos(np.arange(self.num_patches + 1) / (10000 ** (2*i/embed_dim)))
        
        self.position_embeddings = position_embeddings
        
        # Initialize transformer layers
        self.transformer_layers = [
            TransformerLayer(embed_dim, num_heads, embed_dim * 4, dropout)
            for _ in range(num_layers)
        ]
        
        # Initialize MLP head
        self.mlp_head = MLP(embed_dim, embed_dim * 4, num_classes)
        
        # Initialize layer normalization
        self.layer_norm = LayerNorm(embed_dim)
        
        logger.info(f"Vision Transformer initialized with {num_layers} layers, {num_heads} heads, and {embed_dim} dimensions")
    
    def patch_embedding(self, x):
        """
        Extract patches from input image and project them through learned weights.
        
        Args:
            x: Input image of shape (batch_size, channels, height, width)
            
        Returns:
            patches: Extracted and projected patches of shape (batch_size, num_patches, embed_dim)
        """
        # Reshape input to (batch_size, height, width, channels)
        x = np.transpose(x, (0, 2, 3, 1))
        
        # Extract patches and project through weights
        patches, _ = patch_embedding(
            x,
            self.patch_size,
            self.patch_size,  # stride = patch_size
            self.patch_embedding_weights
        )
        
        return patches
    
    def forward(self, x):
        # x shape: (batch_size, num_channels, height, width)
        batch_size = x.shape[0]
        
        # Reshape input into patches
        # Resulting shape: (batch_size, num_patches, embed_dim)
        x = self.patch_embedding(x)
        
        # Expand class token to match batch size
        # Shape: (batch_size, 1, embed_dim)
        class_token = np.repeat(self.class_token, batch_size, axis=0)
        
        # Concatenate class token with patch embeddings
        # Shape: (batch_size, num_patches + 1, embed_dim)
        x = np.concatenate([class_token, x], axis=1)
        
        # Add position embeddings
        # Shape: (1, num_patches + 1, embed_dim)
        x = x + self.position_embeddings[np.newaxis, :, :]
        
        # Apply transformer layers
        for layer in self.transformer_layers:
            x = layer(x)
        
        # Apply layer norm
        x = self.layer_norm(x)
        
        # Extract class token representation
        # Shape: (batch_size, embed_dim)
        x = x[:, 0]
        
        # Apply MLP head
        # Shape: (batch_size, num_classes)
        x = self.mlp_head(x)
        
        return x
    
    def save(self, path):
        """Save the model weights to a file"""
        try:
            # Convert numpy arrays to lists for serialization
            weights = {
                'patch_embedding_weights': self.patch_embedding_weights.tolist(),
                'class_token': self.class_token.tolist(),
                'position_embeddings': self.position_embeddings.tolist(),
                'transformer_layers': [layer.get_weights() for layer in self.transformer_layers],
                'mlp_head': self.mlp_head.get_weights(),
                'layer_norm': self.layer_norm.get_weights()
            }
            
            with open(path, 'wb') as f:
                pickle.dump(weights, f)
            
            logger.info(f"Model weights saved to {path}")
        except Exception as e:
            logger.error(f"Failed to save model weights: {e}")
    
    def load(self, path):
        """Load model weights from a file"""
        try:
            with open(path, 'rb') as f:
                weights = pickle.load(f)
            
            # Load weights
            self.patch_embedding_weights = np.array(weights['patch_embedding_weights'])
            self.class_token = np.array(weights['class_token'])
            self.position_embeddings = np.array(weights['position_embeddings'])
            
            for i, layer_weights in enumerate(weights['transformer_layers']):
                self.transformer_layers[i].set_weights(layer_weights)
            
            self.mlp_head.set_weights(weights['mlp_head'])
            self.layer_norm.set_weights(weights['layer_norm'])
            
            logger.info(f"Model weights loaded from {path}")
        except Exception as e:
            logger.error(f"Failed to load model weights: {e}") 