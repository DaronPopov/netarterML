"""
Unit tests for the noise scheduling kernel.
"""

import unittest
import numpy as np
from OPENtransformer.arm64_engine.core.asm.kernels.vision.diffusion.noise_scheduling_kernel_asm import NoiseSchedulingKernelASM

class TestNoiseSchedulingKernel(unittest.TestCase):
    """Test cases for the noise scheduling kernel."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.kernel = NoiseSchedulingKernelASM()
        
    def test_linear_schedule(self):
        """Test linear beta schedule generation."""
        # Test parameters
        beta_start = 0.0001
        beta_end = 0.02
        timesteps = 1000
        
        # Generate schedule
        betas = self.kernel.get_beta_schedule('linear', beta_start, beta_end, timesteps)
        
        # Check shape
        self.assertEqual(betas.shape, (timesteps,))
        
        # Check values
        self.assertAlmostEqual(betas[0], beta_start, places=4)
        self.assertAlmostEqual(betas[-1], beta_end, places=4)
        
        # Check monotonicity
        self.assertTrue(np.all(np.diff(betas) > 0))
        
    def test_cosine_schedule(self):
        """Test cosine beta schedule generation."""
        # Test parameters
        beta_start = 0.0001
        beta_end = 0.02
        timesteps = 1000
        
        # Generate schedule
        betas = self.kernel.get_beta_schedule('cosine', beta_start, beta_end, timesteps)
        
        # Check shape
        self.assertEqual(betas.shape, (timesteps,))
        
        # Check value range
        self.assertTrue(np.all(betas >= 0.0001))
        self.assertTrue(np.all(betas <= 0.9999))
        
        # Check values are reasonable
        self.assertGreater(np.mean(betas), 0.0001)
        self.assertLess(np.mean(betas), 0.9999)
        
    def test_invalid_schedule_type(self):
        """Test handling of invalid schedule type."""
        with self.assertRaises(ValueError):
            self.kernel.get_beta_schedule('invalid', 0.0001, 0.02, 1000)
            
    def test_zero_timesteps(self):
        """Test handling of zero timesteps."""
        with self.assertRaises(RuntimeError):
            self.kernel.get_beta_schedule('linear', 0.0001, 0.02, 0)
            
    def test_negative_timesteps(self):
        """Test handling of negative timesteps."""
        with self.assertRaises(RuntimeError):
            self.kernel.get_beta_schedule('linear', 0.0001, 0.02, -1000)
            
    def test_invalid_beta_range(self):
        """Test handling of invalid beta range."""
        with self.assertRaises(RuntimeError):
            self.kernel.get_beta_schedule('linear', 0.02, 0.0001, 1000)  # beta_start > beta_end
            
    def test_numpy_fallback(self):
        """Test NumPy fallback implementation."""
        # Force fallback by setting _asm_available to False
        self.kernel._asm_available = False
        
        # Test parameters
        beta_start = 0.0001
        beta_end = 0.02
        timesteps = 1000
        
        # Generate schedule
        betas = self.kernel.get_beta_schedule('linear', beta_start, beta_end, timesteps)
        
        # Check shape
        self.assertEqual(betas.shape, (timesteps,))
        
        # Check values
        self.assertAlmostEqual(betas[0], beta_start, places=4)
        self.assertAlmostEqual(betas[-1], beta_end, places=4)
        
        # Check monotonicity
        self.assertTrue(np.all(np.diff(betas) > 0))
        
if __name__ == '__main__':
    unittest.main() 