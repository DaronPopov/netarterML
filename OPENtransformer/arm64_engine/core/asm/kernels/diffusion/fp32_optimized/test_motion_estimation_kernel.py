"""
Unit tests for the motion estimation kernel.
"""

import unittest
import numpy as np
from OPENtransformer.arm64_engine.core.asm.kernels.vision.diffusion.motion_estimation_kernel_asm import MotionEstimationKernelASM

class TestMotionEstimationKernel(unittest.TestCase):
    """Test cases for the motion estimation kernel."""
    
    def setUp(self):
        """Set up test fixtures."""
        np.random.seed(42)
        
        # Initialize dimensions
        self.height = 32
        self.width = 32
        self.block_size = 8
        self.search_range = 4
        self.smoothness_weight = 0.1
        
        # Create test frames with 3 channels (RGB)
        self.frame1 = np.random.randn(self.height, self.width, 3).astype(np.float32)
        self.frame2 = self.frame1 + 0.1 * np.random.randn(self.height, self.width, 3).astype(np.float32)
        
        # Initialize kernel
        self.kernel = MotionEstimationKernelASM()
    
    def test_initialization(self):
        """Test kernel initialization."""
        self.assertIsNotNone(self.kernel)
        self.assertTrue(hasattr(self.kernel, '_motion_estimation_kernel'))
        self.assertTrue(hasattr(self.kernel, '_asm_available'))
    
    def test_basic_motion_estimation(self):
        """Test basic motion estimation between two frames."""
        flow = self.kernel.estimate_motion(
            self.frame1,
            self.frame2,
            self.block_size,
            self.search_range,
            self.smoothness_weight
        )
        
        self.assertEqual(flow.shape, (*self.frame1.shape[:2], 2))
        self.assertTrue(np.all(np.isfinite(flow)))
    
    def test_empty_input(self):
        """Test motion estimation with empty input."""
        with self.assertRaises(ValueError):
            self.kernel.estimate_motion(
                np.zeros((0, 0, 3), dtype=np.float32),
                np.zeros((0, 0, 3), dtype=np.float32),
                self.block_size, self.search_range,
                self.smoothness_weight
            )
    
    def test_shape_mismatch(self):
        """Test motion estimation with mismatched frame shapes."""
        frame2_wrong_shape = np.random.randn(16, 16, 3).astype(np.float32)
        with self.assertRaises(ValueError):
            self.kernel.estimate_motion(
                self.frame1, frame2_wrong_shape,
                self.block_size, self.search_range,
                self.smoothness_weight
            )
    
    def test_invalid_dimensions(self):
        """Test motion estimation with invalid input dimensions."""
        # Wrong number of dimensions
        invalid_frame = np.random.randn(32, 32).astype(np.float32)
        
        with self.assertRaises(ValueError):
            self.kernel.estimate_motion(
                invalid_frame,
                invalid_frame,
                self.block_size,
                self.search_range,
                self.smoothness_weight
            )
            
        # Mismatched dimensions
        frame1 = np.random.randn(32, 32, 3).astype(np.float32)
        frame2 = np.random.randn(64, 64, 3).astype(np.float32)
        
        with self.assertRaises(ValueError):
            self.kernel.estimate_motion(
                frame1,
                frame2,
                self.block_size,
                self.search_range,
                self.smoothness_weight
            )
    
    def test_large_input(self):
        """Test motion estimation with large input dimensions."""
        height, width = 256, 256
        large_frame1 = np.random.randn(height, width, 3).astype(np.float32)
        large_frame2 = large_frame1 + 0.1 * np.random.randn(height, width, 3).astype(np.float32)
        
        flow = self.kernel.estimate_motion(
            large_frame1,
            large_frame2,
            self.block_size,
            self.search_range,
            self.smoothness_weight
        )
        
        self.assertEqual(flow.shape, (height, width, 2))
        self.assertTrue(np.all(np.isfinite(flow)))
    
    def test_edge_cases(self):
        """Test motion estimation with edge cases."""
        # Single pixel frame
        single_pixel_frame = np.random.randn(1, 1, 3).astype(np.float32)
        
        with self.assertRaises(ValueError):
            flow = self.kernel.estimate_motion(
                single_pixel_frame,
                single_pixel_frame,
                1,
                1,
                self.smoothness_weight
            )
            
        # Empty frame
        empty_frame = np.zeros((0, 0, 3), dtype=np.float32)
        
        with self.assertRaises(ValueError):
            flow = self.kernel.estimate_motion(
                empty_frame,
                empty_frame,
                self.block_size,
                self.search_range,
                self.smoothness_weight
            )
    
    def test_smoothness_weight(self):
        """Test effect of smoothness weight on motion estimation."""
        # Zero smoothness weight
        flow_zero_smooth = self.kernel.estimate_motion(
            self.frame1,
            self.frame2,
            self.block_size,
            self.search_range,
            0.0
        )
        
        # High smoothness weight
        flow_high_smooth = self.kernel.estimate_motion(
            self.frame1,
            self.frame2,
            self.block_size,
            self.search_range,
            1.0
        )
        
        # Higher smoothness weight should result in smoother flow field
        self.assertTrue(np.mean(np.abs(np.diff(flow_high_smooth))) <
                       np.mean(np.abs(np.diff(flow_zero_smooth))))
    
    def test_consistency(self):
        """Test consistency of motion estimation results."""
        # Forward flow
        flow1 = self.kernel.estimate_motion(
            self.frame1,
            self.frame2,
            self.block_size,
            self.search_range,
            self.smoothness_weight
        )
        
        # Backward flow
        flow2 = self.kernel.estimate_motion(
            self.frame2,
            self.frame1,
            self.block_size,
            self.search_range,
            self.smoothness_weight
        )
        
        # Flows should be approximately opposite
        self.assertTrue(np.allclose(flow1, -flow2, rtol=0.1, atol=0.1))

if __name__ == '__main__':
    unittest.main() 