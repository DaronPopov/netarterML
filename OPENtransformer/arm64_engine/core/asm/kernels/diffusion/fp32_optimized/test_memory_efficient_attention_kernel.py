"""
Test script for the Memory-Efficient Attention Kernel.
Tests various aspects of the kernel including basic functionality,
different window sizes, varying sequence lengths, and performance.
"""

import numpy as np
import time
from memory_efficient_attention_kernel_asm import MemoryEfficientAttentionKernelASM

def run_test():
    """Run all test cases for the memory-efficient attention kernel."""
    print("Testing Memory-Efficient Attention Kernel...")
    
    # Initialize kernel with default parameters
    kernel = MemoryEfficientAttentionKernelASM()
    
    # Test 1: Basic functionality
    print("\nTest 1: Basic functionality")
    batch_size = 2
    num_heads = 4
    seq_len = 32
    head_dim = 64
    
    # Create random input tensors
    query = np.random.randn(batch_size, num_heads, seq_len, head_dim).astype(np.float32)
    key = np.random.randn(batch_size, num_heads, seq_len, head_dim).astype(np.float32)
    value = np.random.randn(batch_size, num_heads, seq_len, head_dim).astype(np.float32)
    
    # Compute attention
    output = kernel.compute_attention(query, key, value)
    
    # Check output shape
    assert output.shape == (batch_size, num_heads, seq_len, head_dim), \
        f"Expected shape {(batch_size, num_heads, seq_len, head_dim)}, got {output.shape}"
    print("✓ Shape preservation verified")
    
    # Check output range
    assert np.all(np.isfinite(output)), "Output contains non-finite values"
    print("✓ Output range verified")
    
    # Test 2: Different window sizes
    print("\nTest 2: Different window sizes")
    window_sizes = [16, 32, 64, 128]
    
    for window_size in window_sizes:
        kernel = MemoryEfficientAttentionKernelASM(window_size=window_size)
        output = kernel.compute_attention(query, key, value)
        
        # Verify output shape
        assert output.shape == (batch_size, num_heads, seq_len, head_dim), \
            f"Expected shape {(batch_size, num_heads, seq_len, head_dim)}, got {output.shape}"
        
        # Verify output range
        assert np.all(np.isfinite(output)), f"Output contains non-finite values for window size {window_size}"
        
    print(f"✓ Tested window sizes: {window_sizes}")
    
    # Test 3: Varying sequence lengths
    print("\nTest 3: Varying sequence lengths")
    seq_lengths = [16, 32, 64, 128]
    
    for seq_len in seq_lengths:
        # Create input tensors with current sequence length
        query = np.random.randn(batch_size, num_heads, seq_len, head_dim).astype(np.float32)
        key = np.random.randn(batch_size, num_heads, seq_len, head_dim).astype(np.float32)
        value = np.random.randn(batch_size, num_heads, seq_len, head_dim).astype(np.float32)
        
        output = kernel.compute_attention(query, key, value)
        
        # Verify output shape
        assert output.shape == (batch_size, num_heads, seq_len, head_dim), \
            f"Expected shape {(batch_size, num_heads, seq_len, head_dim)}, got {output.shape}"
        
        # Verify output range
        assert np.all(np.isfinite(output)), f"Output contains non-finite values for sequence length {seq_len}"
        
    print(f"✓ Tested sequence lengths: {seq_lengths}")
    
    # Test 4: Edge cases
    print("\nTest 4: Edge cases")
    
    # Test with all zeros
    query = np.zeros((batch_size, num_heads, seq_len, head_dim), dtype=np.float32)
    key = np.zeros((batch_size, num_heads, seq_len, head_dim), dtype=np.float32)
    value = np.zeros((batch_size, num_heads, seq_len, head_dim), dtype=np.float32)
    
    output = kernel.compute_attention(query, key, value)
    assert np.allclose(output, 0), "Output should be zero for zero inputs"
    print("✓ Zero inputs handled correctly")
    
    # Test with all ones
    query = np.ones((batch_size, num_heads, seq_len, head_dim), dtype=np.float32)
    key = np.ones((batch_size, num_heads, seq_len, head_dim), dtype=np.float32)
    value = np.ones((batch_size, num_heads, seq_len, head_dim), dtype=np.float32)
    
    output = kernel.compute_attention(query, key, value)
    assert np.all(np.isfinite(output)), "Output contains non-finite values for all-ones inputs"
    print("✓ All-ones inputs handled correctly")
    
    # Test 5: Performance
    print("\nTest 5: Performance")
    # Use larger dimensions for performance test
    batch_size = 1
    num_heads = 8
    seq_len = 256
    head_dim = 64
    
    query = np.random.randn(batch_size, num_heads, seq_len, head_dim).astype(np.float32)
    key = np.random.randn(batch_size, num_heads, seq_len, head_dim).astype(np.float32)
    value = np.random.randn(batch_size, num_heads, seq_len, head_dim).astype(np.float32)
    
    # Warm-up run
    kernel.compute_attention(query, key, value)
    
    # Measure performance
    num_runs = 10
    start_time = time.time()
    
    for _ in range(num_runs):
        kernel.compute_attention(query, key, value)
    
    end_time = time.time()
    avg_time = (end_time - start_time) * 1000 / num_runs  # Convert to milliseconds
    
    print(f"Average time per attention computation: {avg_time:.2f} ms")
    print(f"Sequence length: {seq_len}, Head dimension: {head_dim}")
    
    print("\nAll tests completed successfully!")

if __name__ == "__main__":
    run_test() 