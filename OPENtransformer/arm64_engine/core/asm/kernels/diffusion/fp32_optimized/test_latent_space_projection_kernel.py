"""
Test script for the Latent Space Projection Kernel.
Tests various aspects of the kernel including basic functionality,
different dimensions, and performance.
"""

import numpy as np
import time
from latent_space_projection_kernel_asm import LatentSpaceProjectionKernelASM

def run_test():
    """Run all test cases for the latent space projection kernel."""
    print("Testing Latent Space Projection Kernel...")
    
    # Initialize kernel
    kernel = LatentSpaceProjectionKernelASM()
    
    # Test 1: Basic functionality
    print("\nTest 1: Basic functionality")
    batch_size = 2
    input_dim = 64
    output_dim = 32
    
    # Create random input tensors
    input_latent = np.random.randn(batch_size, input_dim).astype(np.float32)
    projection_matrix = np.random.randn(output_dim, input_dim).astype(np.float32)
    
    # Compute projection
    output_latent = kernel.project_latent_space(input_latent, projection_matrix)
    
    # Check output shape
    assert output_latent.shape == (batch_size, output_dim), \
        f"Expected shape {(batch_size, output_dim)}, got {output_latent.shape}"
    print("✓ Shape preservation verified")
    
    # Check output range
    assert np.all(np.isfinite(output_latent)), "Output contains non-finite values"
    print("✓ Output range verified")
    
    # Test 2: Different dimensions
    print("\nTest 2: Different dimensions")
    dimension_pairs = [
        (32, 64),
        (64, 128),
        (128, 256),
        (256, 512)
    ]
    
    for input_dim, output_dim in dimension_pairs:
        # Create input tensors with current dimensions
        input_latent = np.random.randn(batch_size, input_dim).astype(np.float32)
        projection_matrix = np.random.randn(output_dim, input_dim).astype(np.float32)
        
        output_latent = kernel.project_latent_space(input_latent, projection_matrix)
        
        # Verify output shape
        assert output_latent.shape == (batch_size, output_dim), \
            f"Expected shape {(batch_size, output_dim)}, got {output_latent.shape}"
        
        # Verify output range
        assert np.all(np.isfinite(output_latent)), \
            f"Output contains non-finite values for dimensions {input_dim}->{output_dim}"
        
    print(f"✓ Tested dimension pairs: {dimension_pairs}")
    
    # Test 3: Different scale factors
    print("\nTest 3: Different scale factors")
    scale_factors = [0.1, 1.0, 10.0]
    
    for scale in scale_factors:
        output_latent = kernel.project_latent_space(
            input_latent, projection_matrix,
            scale_factor=scale
        )
        
        # Verify output range
        assert np.all(np.isfinite(output_latent)), \
            f"Output contains non-finite values for scale factor {scale}"
        
    print(f"✓ Tested scale factors: {scale_factors}")
    
    # Test 4: Edge cases
    print("\nTest 4: Edge cases")
    
    # Test with all zeros
    input_latent = np.zeros((batch_size, input_dim), dtype=np.float32)
    projection_matrix = np.zeros((output_dim, input_dim), dtype=np.float32)
    
    output_latent = kernel.project_latent_space(input_latent, projection_matrix)
    assert np.allclose(output_latent, 0), "Output should be zero for zero inputs"
    print("✓ Zero inputs handled correctly")
    
    # Test with all ones
    input_latent = np.ones((batch_size, input_dim), dtype=np.float32)
    projection_matrix = np.ones((output_dim, input_dim), dtype=np.float32)
    
    output_latent = kernel.project_latent_space(input_latent, projection_matrix)
    assert np.all(np.isfinite(output_latent)), "Output contains non-finite values for all-ones inputs"
    print("✓ All-ones inputs handled correctly")
    
    # Test 5: Performance
    print("\nTest 5: Performance")
    # Use larger dimensions for performance test
    batch_size = 4
    input_dim = 512
    output_dim = 256
    
    input_latent = np.random.randn(batch_size, input_dim).astype(np.float32)
    projection_matrix = np.random.randn(output_dim, input_dim).astype(np.float32)
    
    # Warm-up run
    kernel.project_latent_space(input_latent, projection_matrix)
    
    # Measure performance
    num_runs = 10
    start_time = time.time()
    
    for _ in range(num_runs):
        kernel.project_latent_space(input_latent, projection_matrix)
    
    end_time = time.time()
    avg_time = (end_time - start_time) * 1000 / num_runs  # Convert to milliseconds
    
    print(f"Average time per projection: {avg_time:.2f} ms")
    print(f"Input dimension: {input_dim}, Output dimension: {output_dim}")
    
    print("\nAll tests completed successfully!")

if __name__ == "__main__":
    run_test() 