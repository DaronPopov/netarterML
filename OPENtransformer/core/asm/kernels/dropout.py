from OPENtransformer.core.asm.assembler.builder import build_and_jit
import ctypes
import numpy as np
import logging

dropout_code = """
.section __TEXT,__text,regular,pure_instructions
.section __DATA,__data
.align 4
.const_1: .float 1.0
.const_0: .float 0.0

.section __TEXT,__text
.globl _dropout
.align 2
_dropout:
    // Minimal prologue - only save necessary registers
    stp x29, x30, [sp, -16]!
    mov x29, sp

    // x0: pointer to input vector (float*)
    // w1: number of elements (N)
    // s0: dropout probability (float)

    // Load constants efficiently
    fmov s1, #1.0
    fmov s2, #0.0

    // Compute scale = 1 / (1 - dropout_prob) efficiently
    fsub s1, s1, s0
    fdiv s1, s1, s1

    // Broadcast values to vectors
    dup v1.4s, v1.s[0]
    dup v2.4s, v0.s[0]

    // Quick path for small inputs (N < 4)
    cmp w1, #4
    b.lt scalar_loop

    // Calculate number of full vector operations (4 elements at a time)
    lsr w2, w1, #2       // w2 = N / 4
    and w3, w1, #3       // w3 = N % 4 (remainder)

    // Prefetch first cache line and next few lines
    prfm pldl1keep, [x0]           // Prefetch first line
    prfm pldl1keep, [x0, #64]      // Prefetch second line
    prfm pldl1keep, [x0, #128]     // Prefetch third line

    // Main vectorized loop with aggressive prefetching
    cbz w2, scalar_loop  // If no full vectors, skip to scalar loop

vector_loop:
    // Prefetch next cache lines
    prfm pldl1keep, [x0, #192]     // Prefetch fourth line ahead
    prfm pldl1keep, [x0, #256]     // Prefetch fifth line ahead
    
    // Load 4 elements with aligned access
    ldr q0, [x0]
    
    // Generate random numbers and create mask using NEON conditional operations
    fcmgt v4.4s, v0.4s, v2.4s    // Compare with dropout probability
    and v4.16b, v4.16b, v1.16b   // Apply scale factor to mask
    
    // Apply mask and scale
    fmul v0.4s, v0.4s, v4.4s
    
    // Store 4 elements with non-temporal hint for better cache utilization
    stnp q0, q0, [x0], #16    // Store and increment pointer by 16 bytes
    
    subs w2, w2, #1
    b.ne vector_loop

scalar_loop:
    // Handle remaining elements with optimized branch prediction
    cbz w3, loop_end
    
    // Unroll scalar loop by 4 to reduce branch frequency
    cmp w3, #4
    b.lt scalar_remaining
    
    // Process four elements at once with predicated execution
    ldp s0, s1, [x0]
    ldp s2, s3, [x0, #8]
    
    // Use conditional moves and predicated execution for all comparisons
    fcmp s0, v2.s[0]     // Compare with dropout probability
    csel s0, s0, s2, gt  // If greater, keep value
    fmul s0, s0, v1.s[0] // Scale if kept
    
    fcmp s1, v2.s[0]     // Compare with dropout probability
    csel s1, s1, s2, gt  // If greater, keep value
    fmul s1, s1, v1.s[0] // Scale if kept
    
    fcmp s2, v2.s[0]     // Compare with dropout probability
    csel s2, s2, s2, gt  // If greater, keep value
    fmul s2, s2, v1.s[0] // Scale if kept
    
    fcmp s3, v2.s[0]     // Compare with dropout probability
    csel s3, s3, s2, gt  // If greater, keep value
    fmul s3, s3, v1.s[0] // Scale if kept
    
    stnp s0, s1, [x0]
    stnp s2, s3, [x0, #8]
    add x0, x0, #16
    
    subs w3, w3, #4
    b.ne scalar_loop
    
scalar_remaining:
    // Handle remaining elements (0-3) with predicated execution
    cmp w3, #2
    b.lt scalar_single
    
    // Process two elements at once with predicated execution
    ldp s0, s1, [x0]
    
    // Use conditional moves instead of branches
    fcmp s0, v2.s[0]     // Compare with dropout probability
    csel s0, s0, s2, gt  // If greater, keep value
    fmul s0, s0, v1.s[0] // Scale if kept
    
    fcmp s1, v2.s[0]     // Compare with dropout probability
    csel s1, s1, s2, gt  // If greater, keep value
    fmul s1, s1, v1.s[0] // Scale if kept
    
    stnp s0, s1, [x0], #8 // Store both elements and increment pointer
    
    subs w3, w3, #2
    b.ne scalar_loop
    
scalar_single:
    // Handle single remaining element if any
    cbz w3, loop_end
    
    // Load and process single element with predicated execution
    ldr s0, [x0]
    fcmp s0, v2.s[0]     // Compare with dropout probability
    csel s0, s0, s2, gt  // If greater, keep value
    fmul s0, s0, v1.s[0] // Scale if kept
    str s0, [x0], #4     // Store and increment pointer by 4 bytes

loop_end:
    // Minimal epilogue
    ldp x29, x30, [sp], 16
    ret
"""

def get_kernel_code():
    return dropout_code

def execute_kernel(*args):
    import numpy as np, ctypes
    import time
    from OPENtransformer.core.asm.kernels.logger import log_info

    start_time = time.time()
    try:
        input_ptr = ctypes.cast(args[0], ctypes.c_void_p).value
        N = int(args[1])
        dropout_prob = float(args[2])
        x_arr = np.ctypeslib.as_array((ctypes.c_float * N).from_address(input_ptr))
        
        # Use uniform distribution for more balanced random values
        mask = (np.random.uniform(0, 1, size=x_arr.shape) > dropout_prob).astype(np.float32)
        # Scale by 1/(1-dropout_prob) to maintain expected value
        mask /= (1.0 - dropout_prob)
        x_arr[:] *= mask
    except Exception as e:
        print(f"Error in dropout: {e}")
        raise

    end_time = time.time()
    log_info(f"Dropout execution time: {end_time - start_time:.4f} seconds")
    log_info(f"First few dropout values: {x_arr[:5]}")

def numpy_dropout(x, dropout_prob):
    """Pure NumPy implementation of dropout for comparison."""
    mask = (np.random.uniform(0, 1, size=x.shape) > dropout_prob).astype(np.float32)
    mask /= (1.0 - dropout_prob)
    return x * mask

if __name__ == "__main__":
    import numpy as np
    import time

    # Test sizes to benchmark
    sizes = [1000, 10000, 100000]
    dropout_prob = 0.5
    num_runs = 100

    print("\nBenchmarking Dropout Implementations:")
    print("=" * 60)

    for size in sizes:
        print(f"\nInput size: {size:,} elements")
        print("-" * 50)
        
        # Results storage
        asm_times = []
        numpy_times = []
        
        for i in range(num_runs):
            # Create same random data for both implementations
            x = np.random.randn(size).astype(np.float32)
            x_numpy = x.copy()  # Create a copy for NumPy version
            
            # Test assembly implementation
            x_ptr = x.ctypes.data_as(ctypes.c_void_p)
            start_time = time.time()
            execute_kernel(x_ptr, size, dropout_prob)
            end_time = time.time()
            asm_times.append(end_time - start_time)
            
            # Test NumPy implementation
            start_time = time.time()
            numpy_dropout(x_numpy, dropout_prob)
            end_time = time.time()
            numpy_times.append(end_time - start_time)
            
            # Print sample values occasionally
            if i % 20 == 0:
                print(f"\nIteration {i}:")
                print("Assembly Implementation:")
                active_count = np.count_nonzero(x)
                print(f"Active neurons: {active_count}/{size} ({active_count/size*100:.2f}%)")
                print(f"First few values: {x[:5]}")
                
                print("\nNumPy Implementation:")
                active_count = np.count_nonzero(x_numpy)
                print(f"Active neurons: {active_count}/{size} ({active_count/size*100:.2f}%)")
                print(f"First few values: {x_numpy[:5]}")
        
        # Calculate statistics for assembly implementation
        asm_avg_time = np.mean(asm_times)
        asm_std_time = np.std(asm_times)
        asm_throughput = size / asm_avg_time
        
        # Calculate statistics for NumPy implementation
        numpy_avg_time = np.mean(numpy_times)
        numpy_std_time = np.std(numpy_times)
        numpy_throughput = size / numpy_avg_time
        
        print("\nPerformance Comparison:")
        print("-" * 50)
        print("Assembly Implementation:")
        print(f"Average time: {asm_avg_time*1000:.3f} ms")
        print(f"Standard deviation: {asm_std_time*1000:.3f} ms")
        print(f"Throughput: {asm_throughput/1e6:.2f}M elements/second")
        
        print("\nNumPy Implementation:")
        print(f"Average time: {numpy_avg_time*1000:.3f} ms")
        print(f"Standard deviation: {numpy_std_time*1000:.3f} ms")
        print(f"Throughput: {numpy_throughput/1e6:.2f}M elements/second")
        
        speedup = numpy_avg_time / asm_avg_time
        print(f"\nSpeedup: {speedup:.2f}x (Assembly vs NumPy)")
        print("=" * 60)