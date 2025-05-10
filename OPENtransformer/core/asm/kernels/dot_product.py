from OPENtransformer.core.asm.assembler.builder import build_and_jit
import ctypes
import numpy as np
import logging
import time

logger = logging.getLogger("OPENtransformer.asm.dot_product")

dot_product_code = """
.section __TEXT,__text,regular,pure_instructions

// Data section for constants
.section __DATA,__data
.align 4
.const_0: .float 0.0

.section __TEXT,__text
.globl _dot_product
.align 2
_dot_product:
    // Save callee-saved registers and link register
    stp x29, x30, [sp, -32]!
    mov x29, sp

    // x0: pointer to vector A
    // x1: pointer to vector B
    // x2: pointer to result
    // w3: length of vectors

    // Load constant 0.0
    adrp x9, .const_0@PAGE
    add x9, x9, .const_0@PAGEOFF
    ldr s0, [x9]              // s0 = 0.0 for accumulator

    // Check if length is 0
    cmp w3, #0
    b.le dot_product_end

    // Calculate number of full SIMD blocks (4 floats per block)
    lsr w4, w3, #2          // w4 = length / 4
    and w5, w3, #3          // w5 = length % 4 (remainder)

    // Process full SIMD blocks
    cbz w4, process_remainder  // Skip if no full blocks

    // Initialize SIMD accumulator
    dup v0.4s, wzr           // Initialize accumulator with zeros

simd_loop:
    // Load 4 floats from each vector
    ld1 {v1.4s}, [x0], #16   // Load 4 floats from A, post-increment
    ld1 {v2.4s}, [x1], #16   // Load 4 floats from B, post-increment

    // Multiply and accumulate
    fmla v0.4s, v1.4s, v2.4s

    // Decrement counter and continue if not zero
    subs w4, w4, #1
    b.ne simd_loop

    // Horizontal add of SIMD accumulator
    faddp v1.4s, v0.4s, v0.4s   // Add pairs within v0
    faddp v0.2s, v1.2s, v1.2s   // Add remaining pairs
    mov s0, v0.s[0]             // Move result to scalar register

process_remainder:
    // Process remaining elements
    cbz w5, store_result

remainder_loop:
    // Load single float from each vector
    ldr s1, [x0], #4
    ldr s2, [x1], #4

    // Multiply and accumulate
    fmadd s0, s1, s2, s0

    // Decrement counter and continue if not zero
    subs w5, w5, #1
    b.ne remainder_loop

store_result:
    // Store final result
    str s0, [x2]

dot_product_end:
    // Restore callee-saved registers and return
    ldp x29, x30, [sp], 32
    ret
"""

def build_kernel():
    """Build and return the dot product kernel."""
    return build_and_jit(dot_product_code, "_dot_product")

def test_dot_product_kernel():
    """Test the dot product kernel with various input sizes."""
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger.info("Testing dot product kernel")
    
    # Build the kernel
    kernel = build_kernel()
    
    # Test cases with different sizes
    test_sizes = [1, 2, 3, 4, 8, 15, 16, 31, 32, 64, 128]
    all_passed = True
    
    for size in test_sizes:
        # Create test vectors
        a = np.random.randn(size).astype(np.float32)
        b = np.random.randn(size).astype(np.float32)
        result = np.zeros(1, dtype=np.float32)
        
        # Calculate expected result using numpy
        expected = np.dot(a, b)
        
        # Call the kernel
        kernel(a.ctypes.data, b.ctypes.data, result.ctypes.data, size)
        
        # Check if results match
        is_correct = np.allclose(result[0], expected, rtol=1e-6)
        all_passed &= is_correct
        
        logger.info(f"\nTest with size {size}:")
        logger.info(f"Vector A: {a}")
        logger.info(f"Vector B: {b}")
        logger.info(f"Expected: {expected}")
        logger.info(f"Got: {result[0]}")
        logger.info(f"Result: {'✓' if is_correct else '✗'}")
        
        if not is_correct:
            logger.error(f"Test failed for size {size}")
            logger.error(f"Absolute difference: {abs(result[0] - expected)}")
            logger.error(f"Relative difference: {abs(result[0] - expected) / abs(expected)}")
    
    if all_passed:
        logger.info("\nAll tests passed! ✓")
    else:
        logger.error("\nSome tests failed! ✗")
    
    return kernel

def benchmark_dot_product():
    """Benchmark the dot product kernel against numpy."""
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger.info("\nBenchmarking dot product kernel")
    
    # Build the kernel
    kernel = build_kernel()
    
    # Test sizes (powers of 2 from 2^6 to 2^15)
    sizes = [2**i for i in range(6, 16)]
    
    logger.info("\nSize      Assembly(µs)    Numpy(µs)    Speedup")
    logger.info("-" * 50)
    
    for size in sizes:
        # Create random test vectors
        a = np.random.randn(size).astype(np.float32)
        b = np.random.randn(size).astype(np.float32)
        result = np.zeros(1, dtype=np.float32)
        
        # Warm up
        for _ in range(5):
            kernel(a.ctypes.data, b.ctypes.data, result.ctypes.data, size)
            np.dot(a, b)
        
        # Benchmark assembly kernel
        start_time = time.perf_counter()
        iterations = 1000
        for _ in range(iterations):
            kernel(a.ctypes.data, b.ctypes.data, result.ctypes.data, size)
        asm_time = (time.perf_counter() - start_time) * 1e6 / iterations  # Convert to microseconds
        
        # Benchmark numpy
        start_time = time.perf_counter()
        for _ in range(iterations):
            np.dot(a, b)
        numpy_time = (time.perf_counter() - start_time) * 1e6 / iterations  # Convert to microseconds
        
        speedup = numpy_time / asm_time if asm_time > 0 else 0
        
        logger.info(f"{size:8d}  {asm_time:10.2f}µs  {numpy_time:10.2f}µs  {speedup:8.2f}x")
    
    return kernel

if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Run tests
    kernel = test_dot_product_kernel()
    
    # Run benchmarks
    benchmark_dot_product() 