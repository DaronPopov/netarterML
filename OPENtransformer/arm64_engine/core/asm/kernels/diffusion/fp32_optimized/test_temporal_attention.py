"""
Test cases for the temporal attention kernel.
"""

import unittest
import numpy as np
from OPENtransformer.arm64_engine.core.asm.kernels.vision.diffusion.temporal_attention_kernel_asm import TemporalAttentionKernelASM

class TestTemporalAttentionKernel(unittest.TestCase):
    """Test cases for TemporalAttentionKernelASM."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.kernel = TemporalAttentionKernelASM()
        
    def test_initialization(self):
        """Test kernel initialization."""
        self.assertIsNotNone(self.kernel)
        self.assertTrue(hasattr(self.kernel, '_asm_available'))
        
    def test_small_batch_attention(self):
        """Test temporal attention with small batch size."""
        # Create test data
        batch_size = 2
        num_frames = 4
        channels = 3
        height = 8
        width = 8
        
        frames = np.random.randn(batch_size, num_frames, channels, height, width).astype(np.float32)
        attention_weights = np.random.randn(batch_size, num_frames).astype(np.float32)
        
        # Normalize attention weights
        attention_weights = attention_weights / np.sum(attention_weights, axis=1, keepdims=True)
        
        # Apply temporal attention
        output = self.kernel.apply_temporal_attention(frames, attention_weights)
        
        # Check output shape
        self.assertEqual(output.shape, (batch_size, channels, height, width))
        
        # Check value range
        self.assertTrue(np.all(np.isfinite(output)))
        
    def test_large_batch_attention(self):
        """Test temporal attention with large batch size."""
        # Create test data
        batch_size = 16
        num_frames = 8
        channels = 3
        height = 32
        width = 32
        
        frames = np.random.randn(batch_size, num_frames, channels, height, width).astype(np.float32)
        attention_weights = np.random.randn(batch_size, num_frames).astype(np.float32)
        
        # Normalize attention weights
        attention_weights = attention_weights / np.sum(attention_weights, axis=1, keepdims=True)
        
        # Apply temporal attention
        output = self.kernel.apply_temporal_attention(frames, attention_weights)
        
        # Check output shape
        self.assertEqual(output.shape, (batch_size, channels, height, width))
        
        # Check value range
        self.assertTrue(np.all(np.isfinite(output)))
        
    def test_attention_weights_normalization(self):
        """Test that attention weights are properly normalized."""
        # Create test data
        batch_size = 2
        num_frames = 4
        channels = 3
        height = 8
        width = 8
        
        # Create frames with constant value
        frames = np.ones((batch_size, num_frames, channels, height, width), dtype=np.float32)
        
        # Create normalized attention weights
        attention_weights = np.ones((batch_size, num_frames), dtype=np.float32)
        attention_weights = attention_weights / np.sum(attention_weights, axis=1, keepdims=True)
        
        # Apply temporal attention
        output = self.kernel.apply_temporal_attention(frames, attention_weights)
        
        # Check that output values are close to 1.0 (since input frames are 1.0 and weights sum to 1.0)
        self.assertTrue(np.allclose(output, 1.0, rtol=1e-5, atol=1e-5))
        
    def test_edge_cases(self):
        """Test edge cases and error handling."""
        # Test with empty frames
        with self.assertRaises(ValueError):
            self.kernel.apply_temporal_attention(
                np.zeros((0, 4, 3, 8, 8), dtype=np.float32),
                np.zeros((0, 4), dtype=np.float32)
            )
            
        # Test with mismatched batch sizes
        with self.assertRaises(ValueError):
            self.kernel.apply_temporal_attention(
                np.zeros((2, 4, 3, 8, 8), dtype=np.float32),
                np.zeros((3, 4), dtype=np.float32)  # Mismatched batch size
            )
            
        # Test with mismatched num_frames
        with self.assertRaises(ValueError):
            self.kernel.apply_temporal_attention(
                np.zeros((2, 4, 3, 8, 8), dtype=np.float32),
                np.zeros((2, 5), dtype=np.float32)  # Mismatched num_frames
            )
            
    def test_numpy_fallback(self):
        """Test NumPy fallback implementation."""
        # Force fallback by setting _asm_available to False
        self.kernel._asm_available = False
        
        # Create test data
        batch_size = 2
        num_frames = 4
        channels = 3
        height = 8
        width = 8
        
        frames = np.random.randn(batch_size, num_frames, channels, height, width).astype(np.float32)
        attention_weights = np.random.randn(batch_size, num_frames).astype(np.float32)
        
        # Normalize attention weights
        attention_weights = attention_weights / np.sum(attention_weights, axis=1, keepdims=True)
        
        # Apply temporal attention
        output = self.kernel.apply_temporal_attention(frames, attention_weights)
        
        # Check output shape
        self.assertEqual(output.shape, (batch_size, channels, height, width))
        
        # Check value range
        self.assertTrue(np.all(np.isfinite(output)))
        
    def test_attention_weights_sum(self):
        """Test that attention weights sum to 1.0 for each batch."""
        # Create test data
        batch_size = 2
        num_frames = 4
        channels = 3
        height = 8
        width = 8
        
        frames = np.random.randn(batch_size, num_frames, channels, height, width).astype(np.float32)
        attention_weights = np.random.randn(batch_size, num_frames).astype(np.float32)
        
        # Normalize attention weights
        attention_weights = attention_weights / np.sum(attention_weights, axis=1, keepdims=True)
        
        # Verify weights sum to 1.0
        self.assertTrue(np.allclose(np.sum(attention_weights, axis=1), 1.0, rtol=1e-5, atol=1e-5))
        
        # Apply temporal attention
        output = self.kernel.apply_temporal_attention(frames, attention_weights)
        
        # Check output shape and values
        self.assertEqual(output.shape, (batch_size, channels, height, width))
        self.assertTrue(np.all(np.isfinite(output)))

    def test_asm_vs_numpy_output(self):
        """Test that ASM and NumPy implementations produce identical results."""
        # Create test data with specific patterns for easier debugging
        batch_size = 2
        num_frames = 4
        channels = 3
        height = 8
        width = 8
        
        # Create frames with a pattern
        frames = np.zeros((batch_size, num_frames, channels, height, width), dtype=np.float32)
        for b in range(batch_size):
            for f in range(num_frames):
                for c in range(channels):
                    frames[b, f, c] = (b + 1) * (f + 1) * (c + 1)
        
        # Create attention weights
        attention_weights = np.ones((batch_size, num_frames), dtype=np.float32)
        attention_weights = attention_weights / np.sum(attention_weights, axis=1, keepdims=True)
        
        # Get ASM output
        asm_output = self.kernel.apply_temporal_attention(frames, attention_weights)
        
        # Get NumPy output
        self.kernel._asm_available = False
        numpy_output = self.kernel.apply_temporal_attention(frames, attention_weights)
        
        # Print detailed debugging information
        print("\nDetailed Output Analysis:")
        print("Input frames shape:", frames.shape)
        print("Input frames range:", [np.min(frames), np.max(frames)])
        print("Attention weights shape:", attention_weights.shape)
        print("Attention weights:", attention_weights)
        
        print("\nOutput Statistics:")
        print("ASM output shape:", asm_output.shape)
        print("ASM output range:", [np.min(asm_output), np.max(asm_output)])
        print("NumPy output shape:", numpy_output.shape)
        print("NumPy output range:", [np.min(numpy_output), np.max(numpy_output)])
        
        # Find locations where outputs differ significantly
        diff_mask = np.abs(asm_output - numpy_output) > 1e-5
        if np.any(diff_mask):
            print("\nLocations where outputs differ significantly:")
            diff_indices = np.where(diff_mask)
            for i in range(min(10, len(diff_indices[0]))):  # Print first 10 differences
                b, c, h, w = [idx[i] for idx in diff_indices]
                print(f"Position (b={b}, c={c}, h={h}, w={w}):")
                print(f"  ASM value: {asm_output[b,c,h,w]:.6f}")
                print(f"  NumPy value: {numpy_output[b,c,h,w]:.6f}")
                print(f"  Difference: {abs(asm_output[b,c,h,w] - numpy_output[b,c,h,w]):.6f}")
        
        print("\nOverall Statistics:")
        print(f"Max absolute difference: {np.max(np.abs(asm_output - numpy_output)):.6f}")
        print(f"Mean absolute difference: {np.mean(np.abs(asm_output - numpy_output)):.6f}")
        print(f"Number of differing elements: {np.sum(diff_mask)}")
        
        # Compare outputs with a slightly higher tolerance
        self.assertTrue(np.allclose(asm_output, numpy_output, rtol=1e-4, atol=1e-4),
                       "ASM and NumPy outputs differ significantly")

if __name__ == '__main__':
    unittest.main() 