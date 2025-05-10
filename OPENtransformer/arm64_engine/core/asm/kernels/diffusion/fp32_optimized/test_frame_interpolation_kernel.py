"""
Test script for frame interpolation kernel.
Tests various aspects of frame interpolation including different frame sizes,
interpolation factors, and edge cases.
"""

import numpy as np
from frame_interpolation_kernel_asm import FrameInterpolationKernelASM

def run_test():
    """Run comprehensive tests for frame interpolation kernel."""
    # Initialize kernel
    kernel = FrameInterpolationKernelASM()
    
    print("Starting Frame Interpolation Kernel Tests...")
    
    # Test 1: Basic Functionality
    print("\nTest 1: Basic Functionality")
    frame1 = np.random.uniform(0, 1, (64, 64, 3)).astype(np.float32)
    frame2 = np.random.uniform(0, 1, (64, 64, 3)).astype(np.float32)
    
    # Test with factor 0.5 (midpoint)
    result = kernel.interpolate(frame1, frame2, 0.5)
    print(f"- Shape matches: {result.shape == frame1.shape}")
    print(f"- Output range: [{result.min():.3f}, {result.max():.3f}]")
    print(f"- Mean difference from average: {np.abs(result - (frame1 + frame2) / 2).mean():.6f}")
    
    # Test 2: Different Interpolation Factors
    print("\nTest 2: Different Interpolation Factors")
    factors = [0.0, 0.25, 0.75, 1.0]
    for factor in factors:
        result = kernel.interpolate(frame1, frame2, factor)
        if factor == 0.0:
            diff_from_expected = np.abs(result - frame1).mean()
        elif factor == 1.0:
            diff_from_expected = np.abs(result - frame2).mean()
        else:
            expected = frame1 * (1 - factor) + frame2 * factor
            diff_from_expected = np.abs(result - expected).mean()
        print(f"Factor {factor}:")
        print(f"- Mean difference from expected: {diff_from_expected:.6f}")
    
    # Test 3: Different Frame Sizes
    print("\nTest 3: Different Frame Sizes")
    sizes = [(32, 32, 3), (128, 128, 3), (256, 128, 3)]
    for size in sizes:
        frame1 = np.random.uniform(0, 1, size).astype(np.float32)
        frame2 = np.random.uniform(0, 1, size).astype(np.float32)
        result = kernel.interpolate(frame1, frame2, 0.5)
        print(f"Size {size}:")
        print(f"- Shape matches: {result.shape == size}")
        print(f"- Output range: [{result.min():.3f}, {result.max():.3f}]")
    
    # Test 4: Edge Cases
    print("\nTest 4: Edge Cases")
    # Test with identical frames
    frame1 = np.random.uniform(0, 1, (64, 64, 3)).astype(np.float32)
    result = kernel.interpolate(frame1, frame1, 0.5)
    print("Identical frames:")
    print(f"- Mean difference from input: {np.abs(result - frame1).mean():.6f}")
    
    # Test with extreme values
    frame1 = np.zeros((64, 64, 3), dtype=np.float32)
    frame2 = np.ones((64, 64, 3), dtype=np.float32)
    result = kernel.interpolate(frame1, frame2, 0.5)
    print("Zero to One interpolation:")
    print(f"- Mean value (should be ~0.5): {result.mean():.3f}")
    
    # Test 5: Performance Test
    print("\nTest 5: Performance Test")
    import time
    frame1 = np.random.uniform(0, 1, (256, 256, 3)).astype(np.float32)
    frame2 = np.random.uniform(0, 1, (256, 256, 3)).astype(np.float32)
    
    start_time = time.time()
    for _ in range(100):
        result = kernel.interpolate(frame1, frame2, 0.5)
    elapsed = time.time() - start_time
    print(f"Time for 100 interpolations (256x256): {elapsed:.3f} seconds")
    print(f"Average time per interpolation: {elapsed/100*1000:.2f} ms")
    
    print("\nAll tests completed successfully!")

if __name__ == "__main__":
    run_test() 