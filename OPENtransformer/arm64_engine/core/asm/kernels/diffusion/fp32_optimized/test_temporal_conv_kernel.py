"""
Unit tests for temporal convolution kernel.
"""

import unittest
import numpy as np
from OPENtransformer.arm64_engine.core.asm.kernels.vision.diffusion.temporal_conv_kernel_asm import TemporalConvKernelASM

class TestTemporalConvKernel(unittest.TestCase):
    """Test cases for temporal convolution kernel."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Set random seed for reproducibility
        np.random.seed(42)
        
        # Set dimensions
        self.batch_size = 2
        self.num_frames = 8
        self.channels = 4  # Multiple of 4 for SIMD
        self.height = 8    # Multiple of 4 for SIMD
        self.width = 8     # Multiple of 4 for SIMD
        self.kernel_size = 3
        
        # Generate random input with proper alignment
        self.x = np.ascontiguousarray(
            np.random.randn(
                self.batch_size,
                self.num_frames,
                self.channels,
                self.height,
                self.width
            ).astype(np.float32)
        )
        
        # Generate random kernel with proper alignment
        self.kernel = np.ascontiguousarray(
            np.random.randn(self.kernel_size).astype(np.float32)
        )
        
        # Initialize kernel
        self.temporal_conv = TemporalConvKernelASM()
        
    def test_initialization(self):
        """Test kernel initialization."""
        self.assertTrue(hasattr(self.temporal_conv, '_asm_available'))
        self.assertTrue(hasattr(self.temporal_conv, '_temporal_conv_kernel'))
        
    def test_basic_convolution(self):
        """Test basic temporal convolution."""
        # Apply convolution
        output = self.temporal_conv.apply_temporal_conv(
            self.x,
            self.kernel,
            stride=1,
            padding=0
        )
        
        # Check output shape
        expected_frames = self.num_frames - self.kernel_size + 1
        self.assertEqual(output.shape, (
            self.batch_size,
            expected_frames,
            self.channels,
            self.height,
            self.width
        ))
        
        # Check output values
        numpy_output = self.temporal_conv._numpy_apply_temporal_conv(
            self.x,
            self.kernel,
            stride=1,
            padding=0
        )
        np.testing.assert_allclose(output, numpy_output, rtol=1e-5)
        
    def test_stride(self):
        """Test temporal convolution with stride."""
        stride = 2
        
        # Apply convolution
        output = self.temporal_conv.apply_temporal_conv(
            self.x,
            self.kernel,
            stride=stride,
            padding=0
        )
        
        # Check output shape
        expected_frames = (self.num_frames - self.kernel_size) // stride + 1
        self.assertEqual(output.shape, (
            self.batch_size,
            expected_frames,
            self.channels,
            self.height,
            self.width
        ))
        
        # Check output values
        numpy_output = self.temporal_conv._numpy_apply_temporal_conv(
            self.x,
            self.kernel,
            stride=stride,
            padding=0
        )
        np.testing.assert_allclose(output, numpy_output, rtol=1e-5)
        
    def test_padding(self):
        """Test temporal convolution with padding."""
        padding = 1
        
        # Apply convolution
        output = self.temporal_conv.apply_temporal_conv(
            self.x,
            self.kernel,
            stride=1,
            padding=padding
        )
        
        # Check output shape
        expected_frames = self.num_frames + 2 * padding - self.kernel_size + 1
        self.assertEqual(output.shape, (
            self.batch_size,
            expected_frames,
            self.channels,
            self.height,
            self.width
        ))
        
        # Check output values
        numpy_output = self.temporal_conv._numpy_apply_temporal_conv(
            self.x,
            self.kernel,
            stride=1,
            padding=padding
        )
        np.testing.assert_allclose(output, numpy_output, rtol=1e-5)
        
    def test_empty_input(self):
        """Test handling of empty input."""
        with self.assertRaises(ValueError):
            self.temporal_conv.apply_temporal_conv(
                np.array([]),
                self.kernel
            )
            
    def test_invalid_input_shape(self):
        """Test handling of invalid input shape."""
        with self.assertRaises(ValueError):
            self.temporal_conv.apply_temporal_conv(
                np.random.randn(2, 3, 4),  # Invalid shape
                self.kernel
            )
            
    def test_invalid_kernel_shape(self):
        """Test handling of invalid kernel shape."""
        with self.assertRaises(ValueError):
            self.temporal_conv.apply_temporal_conv(
                self.x,
                np.random.randn(2, 3)  # Invalid shape
            )
            
    def test_large_input(self):
        """Test with large input dimensions."""
        # Set large dimensions
        batch_size = 4
        num_frames = 32
        channels = 64
        height = 64
        width = 64
        
        # Generate large input with proper alignment
        x = np.ascontiguousarray(
            np.random.randn(
                batch_size,
                num_frames,
                channels,
                height,
                width
            ).astype(np.float32)
        )
        
        # Apply convolution
        output = self.temporal_conv.apply_temporal_conv(
            x,
            self.kernel,
            stride=1,
            padding=0
        )
        
        # Check output shape
        expected_frames = num_frames - self.kernel_size + 1
        self.assertEqual(output.shape, (
            batch_size,
            expected_frames,
            channels,
            height,
            width
        ))
        
    def test_edge_cases(self):
        """Test edge cases."""
        # Test with single frame
        x = np.ascontiguousarray(
            np.random.randn(1, 1, 4, 8, 8).astype(np.float32)  # SIMD-friendly dimensions
        )
        output = self.temporal_conv.apply_temporal_conv(
            x,
            self.kernel,
            stride=1,
            padding=0
        )
        self.assertEqual(output.shape, (1, 0, 4, 8, 8))  # No output frames possible
        
        # Test with kernel size equal to number of frames
        x = np.ascontiguousarray(
            np.random.randn(1, 3, 4, 8, 8).astype(np.float32)  # SIMD-friendly dimensions
        )
        kernel = np.ascontiguousarray(
            np.random.randn(3).astype(np.float32)
        )
        output = self.temporal_conv.apply_temporal_conv(
            x,
            kernel,
            stride=1,
            padding=0
        )
        self.assertEqual(output.shape, (1, 1, 4, 8, 8))  # Single output frame
        
if __name__ == '__main__':
    unittest.main() 