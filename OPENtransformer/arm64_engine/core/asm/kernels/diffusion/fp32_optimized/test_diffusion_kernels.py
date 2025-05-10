"""
Unit tests for diffusion kernels.
"""

import unittest
import numpy as np
from OPENtransformer.arm64_engine.core.asm.kernels.vision.diffusion.diffusion_layer_norm_asm import DiffusionLayerNormASM
from OPENtransformer.arm64_engine.core.asm.kernels.vision.diffusion.diffusion_feed_forward_asm import DiffusionFeedForwardASM

class TestDiffusionKernels(unittest.TestCase):
    """Test cases for diffusion kernels."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Set random seed for reproducibility
        np.random.seed(42)
        
        # Set dimensions
        self.batch_size = 2
        self.num_frames = 3
        self.channels = 4
        self.height = 5
        self.width = 6
        self.hidden_dim = 8
        
        # Generate random inputs
        self.x = np.random.randn(self.batch_size, self.num_frames, self.channels, self.height, self.width).astype(np.float32)
        self.gamma = np.random.randn(self.channels, self.height, self.width).astype(np.float32)
        self.beta = np.random.randn(self.channels, self.height, self.width).astype(np.float32)
        
        # Generate random weights and biases
        input_size = self.channels * self.height * self.width
        self.w1 = np.random.randn(self.hidden_dim, input_size).astype(np.float32)
        self.b1 = np.random.randn(self.hidden_dim).astype(np.float32)
        self.w2 = np.random.randn(input_size, self.hidden_dim).astype(np.float32)
        self.b2 = np.random.randn(input_size).astype(np.float32)
        
        # Initialize kernels
        self.layer_norm = DiffusionLayerNormASM()
        self.feed_forward = DiffusionFeedForwardASM()
    
    def test_layer_norm_initialization(self):
        """Test layer normalization kernel initialization."""
        self.assertTrue(hasattr(self.layer_norm, '_asm_available'))
    
    def test_layer_norm_output_shape(self):
        """Test layer normalization output shape."""
        output = self.layer_norm.apply_layer_norm(
            self.x,
            self.gamma,
            self.beta,
            epsilon=1e-5
        )
        self.assertEqual(output.shape, self.x.shape)
    
    def test_layer_norm_normalization(self):
        """Test layer normalization statistics."""
        output = self.layer_norm.apply_layer_norm(
            self.x,
            self.gamma,
            self.beta,
            epsilon=1e-5
        )
        
        # Check mean and variance
        mean = np.mean(output, axis=(2, 3, 4), keepdims=True)
        var = np.var(output, axis=(2, 3, 4), keepdims=True)
        
        np.testing.assert_allclose(mean, 0, atol=1e-5)
        np.testing.assert_allclose(var, 1, atol=1e-5)
    
    def test_layer_norm_edge_cases(self):
        """Test layer normalization with edge cases."""
        # Test with all zeros input
        x_zeros = np.zeros_like(self.x)
        output = self.layer_norm.apply_layer_norm(
            x_zeros,
            self.gamma,
            self.beta,
            epsilon=1e-5
        )
        
        # Output should be equal to beta when input is zero
        beta_broadcast = np.broadcast_to(self.beta, self.x.shape)
        np.testing.assert_allclose(output, beta_broadcast, atol=1e-5)
    
    def test_feed_forward_initialization(self):
        """Test feed forward kernel initialization."""
        self.assertTrue(hasattr(self.feed_forward, '_asm_available'))
    
    def test_feed_forward_output_shape(self):
        """Test feed forward output shape."""
        output = self.feed_forward.apply_feed_forward(self.x, self.w1, self.b1, self.w2, self.b2)
        self.assertEqual(output.shape, self.x.shape)
    
    def test_feed_forward_activation(self):
        """Test feed forward activation."""
        output = self.feed_forward.apply_feed_forward(self.x, self.w1, self.b1, self.w2, self.b2)
        
        # Check that output is not just a linear transformation
        x_flat = self.x.reshape(self.batch_size * self.num_frames, -1)
        linear_output = np.dot(x_flat, self.w1.T) + self.b1
        linear_output = np.dot(linear_output, self.w2.T) + self.b2
        linear_output = linear_output.reshape(self.x.shape)
        
        self.assertFalse(np.allclose(output, linear_output))
    
    def test_feed_forward_edge_cases(self):
        """Test feed forward with edge cases."""
        # Test with all zeros input
        x_zeros = np.zeros_like(self.x)
        output = self.feed_forward.apply_feed_forward(x_zeros, self.w1, self.b1, self.w2, self.b2)
        
        # Output should be equal to b2 when input is zero
        x_shape = x_zeros.shape
        b2_reshaped = self.b2.reshape(1, 1, self.channels, self.height, self.width)
        b2_broadcast = np.broadcast_to(b2_reshaped, x_shape)
        np.testing.assert_allclose(output, b2_broadcast, atol=1e-5)
    
    def test_layer_norm_fallback(self):
        """Test layer normalization fallback to NumPy."""
        self.layer_norm._asm_available = False
        output = self.layer_norm.apply_layer_norm(
            self.x,
            self.gamma,
            self.beta,
            epsilon=1e-5
        )
        self.assertEqual(output.shape, self.x.shape)
    
    def test_feed_forward_fallback(self):
        """Test feed forward fallback to NumPy."""
        self.feed_forward._asm_available = False
        output = self.feed_forward.apply_feed_forward(self.x, self.w1, self.b1, self.w2, self.b2)
        self.assertEqual(output.shape, self.x.shape)

if __name__ == '__main__':
    unittest.main() 