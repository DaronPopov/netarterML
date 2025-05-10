"""
Test script for video post-processing kernel.
Tests various aspects of video enhancement including sharpness,
contrast, and noise reduction operations.
"""

import numpy as np
from video_post_processing_kernel_asm import VideoPostProcessingKernelASM

def run_test():
    """Run comprehensive tests for video post-processing kernel."""
    # Initialize kernel
    kernel = VideoPostProcessingKernelASM()
    
    print("Starting Video Post-Processing Kernel Tests...")
    
    # Test 1: Basic Functionality
    print("\nTest 1: Basic Functionality")
    input_video = np.random.uniform(0, 1, (2, 4, 64, 64, 3)).astype(np.float32)
    
    # Test with default parameters
    result = kernel.process_video(input_video)
    print(f"- Shape matches: {result.shape == input_video.shape}")
    print(f"- Output range: [{result.min():.3f}, {result.max():.3f}]")
    print(f"- Mean difference from input: {np.abs(result - input_video).mean():.6f}")
    
    # Test 2: Different Enhancement Factors
    print("\nTest 2: Different Enhancement Factors")
    factors = [
        (1.0, 1.0, 1.0),    # No enhancement
        (1.5, 1.0, 1.0),    # Sharpness only
        (1.0, 1.5, 1.0),    # Contrast only
        (1.0, 1.0, 0.5),    # Noise reduction only
        (1.5, 1.5, 0.5)     # Combined enhancement
    ]
    
    for sharp, cont, noise in factors:
        result = kernel.process_video(
            input_video,
            sharpness_factor=sharp,
            contrast_factor=cont,
            noise_reduction_factor=noise
        )
        print(f"\nFactors (sharp={sharp}, cont={cont}, noise={noise}):")
        print(f"- Output range: [{result.min():.3f}, {result.max():.3f}]")
        print(f"- Mean value: {result.mean():.3f}")
        print(f"- Std dev: {result.std():.3f}")
    
    # Test 3: Different Video Sizes
    print("\nTest 3: Different Video Sizes")
    sizes = [
        (1, 2, 32, 32, 3),    # Small video
        (2, 4, 128, 128, 3),  # Medium video
        (1, 8, 256, 128, 3)   # Large video
    ]
    
    for size in sizes:
        input_video = np.random.uniform(0, 1, size).astype(np.float32)
        result = kernel.process_video(input_video)
        print(f"\nSize {size}:")
        print(f"- Shape matches: {result.shape == size}")
        print(f"- Output range: [{result.min():.3f}, {result.max():.3f}]")
    
    # Test 4: Edge Cases
    print("\nTest 4: Edge Cases")
    # Test with all zeros
    zeros = np.zeros((2, 4, 64, 64, 3), dtype=np.float32)
    result = kernel.process_video(zeros)
    print("All zeros:")
    print(f"- All zeros preserved: {np.all(result == 0)}")
    
    # Test with all ones
    ones = np.ones((2, 4, 64, 64, 3), dtype=np.float32)
    result = kernel.process_video(ones)
    print("All ones:")
    print(f"- All ones preserved: {np.all(result == 1)}")
    
    # Test with extreme values
    extreme = np.random.uniform(0, 2, (2, 4, 64, 64, 3)).astype(np.float32)
    result = kernel.process_video(extreme)
    print("Extreme values:")
    print(f"- Clipped to [0,1]: {np.all((result >= 0) & (result <= 1))}")
    
    # Test 5: Performance Test
    print("\nTest 5: Performance Test")
    import time
    input_video = np.random.uniform(0, 1, (2, 8, 256, 256, 3)).astype(np.float32)
    
    start_time = time.time()
    for _ in range(10):
        result = kernel.process_video(input_video)
    elapsed = time.time() - start_time
    print(f"Time for 10 processing runs (2x8x256x256x3): {elapsed:.3f} seconds")
    print(f"Average time per run: {elapsed/10*1000:.2f} ms")
    
    print("\nAll tests completed successfully!")

if __name__ == "__main__":
    run_test() 