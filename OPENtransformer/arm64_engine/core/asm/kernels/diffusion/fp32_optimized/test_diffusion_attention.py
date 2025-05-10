"""
Unit tests for diffusion attention kernels.
"""

import unittest
import numpy as np
from OPENtransformer.arm64_engine.core.asm.kernels.vision.diffusion.diffusion_kernels_asm import DiffusionKernelsASM

class TestDiffusionAttention(unittest.TestCase):
    """Test cases for diffusion attention kernels."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Set random seed for reproducibility
        np.random.seed(42)
        
        # Set dimensions
        self.batch_size = 2
        self.seq_len = 4
        self.num_heads = 8
        self.head_dim = 64
        
        # Generate random inputs
        self.query = np.random.randn(self.batch_size, self.seq_len, self.num_heads, self.head_dim).astype(np.float32)
        self.key = np.random.randn(self.batch_size, self.seq_len, self.num_heads, self.head_dim).astype(np.float32)
        self.value = np.random.randn(self.batch_size, self.seq_len, self.num_heads, self.head_dim).astype(np.float32)
        
        # Initialize kernel
        self.kernel = DiffusionKernelsASM()
    
    def test_attention_initialization(self):
        """Test attention kernel initialization."""
        self.assertTrue(hasattr(self.kernel, '_attention_kernel'))
    
    def test_attention_output_shape(self):
        """Test attention output shape."""
        output = self.kernel.apply_attention(self.query, self.key, self.value)
        self.assertEqual(output.shape, self.query.shape)
    
    def test_attention_scale_factor(self):
        """Test attention with different scale factors."""
        # Test with scale factor = 1.0
        output1 = self.kernel.apply_attention(self.query, self.key, self.value, scale_factor=1.0)
        
        # Test with scale factor = 0.5
        output2 = self.kernel.apply_attention(self.query, self.key, self.value, scale_factor=0.5)
        
        # Outputs should be different
        self.assertFalse(np.allclose(output1, output2))
    
    def test_attention_edge_cases(self):
        """Test attention with edge cases."""
        # Test with all zeros input
        query_zeros = np.zeros_like(self.query)
        key_zeros = np.zeros_like(self.key)
        value_zeros = np.zeros_like(self.value)
        
        output = self.kernel.apply_attention(query_zeros, key_zeros, value_zeros)
        
        # Output should be zeros
        np.testing.assert_allclose(output, 0, atol=1e-5)
    
    def test_attention_fallback(self):
        """Test attention fallback to NumPy."""
        self.kernel._asm_available = False
        output = self.kernel.apply_attention(self.query, self.key, self.value)
        self.assertEqual(output.shape, self.query.shape)
    
    def test_attention_numerical_stability(self):
        """Test attention numerical stability."""
        # Create large values to test numerical stability
        query_large = self.query * 1000
        key_large = self.key * 1000
        value_large = self.value * 1000
        
        output = self.kernel.apply_attention(query_large, key_large, value_large)
        
        # Check that output is finite
        self.assertTrue(np.all(np.isfinite(output)))
        
        # Check that output is within reasonable range
        self.assertTrue(np.all(np.abs(output) < 1e6))

if __name__ == '__main__':
    unittest.main() 