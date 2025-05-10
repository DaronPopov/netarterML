from OPENtransformer.core.asm.assembler.builder import build_and_jit
import ctypes
import numpy as np
import logging
import time

logger = logging.getLogger("assembly_test.transpose")

transpose_code = """
.section __TEXT,__text,regular,pure_instructions

.section __TEXT,__text
.globl _transpose
.align 2
_transpose:
    // Save registers
    stp x29, x30, [sp, -32]!
    mov x29, sp

    // Parameters:
    // x0: pointer to input tensor
    // x1: pointer to output tensor
    // w2: number of rows
    // w3: number of columns

    // Save non-volatile registers
    stp x19, x20, [sp, -16]!
    stp x21, x22, [sp, -16]!
    stp x23, x24, [sp, -16]!
    stp x25, x26, [sp, -16]!
    stp x27, x28, [sp, -16]!
    stp d8, d9, [sp, -16]!
    stp d10, d11, [sp, -16]!
    stp d12, d13, [sp, -16]!
    stp d14, d15, [sp, -16]!

    // Process 4x4 blocks at a time
    mov w19, #0              // i = 0
1:  cmp w19, w2             // if i >= rows, exit
    b.ge 2f

    mov w20, #0              // j = 0
3:  cmp w20, w3             // if j >= cols, exit
    b.ge 4f

    // Load 4x4 block
    mov w21, #0              // k = 0
5:  cmp w21, #4             // if k >= 4, exit
    b.ge 6f

    mov w22, #0              // l = 0
7:  cmp w22, #4             // if l >= 4, exit
    b.ge 8f

    // Calculate input address
    add w23, w19, w21        // i + k
    add w24, w20, w22        // j + l
    mul w25, w23, w3         // (i + k) * cols
    add w25, w25, w24        // (i + k) * cols + (j + l)
    lsl w25, w25, #2         // byte offset
    add x26, x0, x25         // input base address
    ldr s8, [x26]            // Load element

    // Calculate output address
    add w23, w20, w22        // j + l
    add w24, w19, w21        // i + k
    mul w25, w23, w2         // (j + l) * rows
    add w25, w25, w24        // (j + l) * rows + (i + k)
    lsl w25, w25, #2         // byte offset
    add x26, x1, x25         // output base address
    str s8, [x26]            // Store element

    add w22, w22, #1         // l++
    b 7b
8:  add w21, w21, #1         // k++
    b 5b
6:  add w20, w20, #4         // j += 4
    b 3b
4:  add w19, w19, #4         // i += 4
    b 1b

2:  // Handle remaining rows
    cmp w19, w2              // if i >= rows, exit
    b.ge 9f

    mov w20, #0              // j = 0
10: cmp w20, w3              // if j >= cols, exit
    b.ge 11f

    // Calculate input address
    mul w21, w19, w3         // i * cols
    add w21, w21, w20        // i * cols + j
    lsl w21, w21, #2         // byte offset
    add x22, x0, x21         // input base address
    ldr s8, [x22]            // Load element

    // Calculate output address
    mul w21, w20, w2         // j * rows
    add w21, w21, w19        // j * rows + i
    lsl w21, w21, #2         // byte offset
    add x22, x1, x21         // output base address
    str s8, [x22]            // Store element

    add w20, w20, #1         // j++
    b 10b
11: add w19, w19, #1         // i++
    b 2b

9:  // Restore registers and return
    ldp d14, d15, [sp], #16
    ldp d12, d13, [sp], #16
    ldp d10, d11, [sp], #16
    ldp d8, d9, [sp], #16
    ldp x27, x28, [sp], #16
    ldp x25, x26, [sp], #16
    ldp x23, x24, [sp], #16
    ldp x21, x22, [sp], #16
    ldp x19, x20, [sp], #16
    ldp x29, x30, [sp], #32
    ret
"""

def get_kernel_code():
    logger.debug("Returning transpose assembly code")
    return transpose_code

def build_kernel():
    """Build the transpose kernel."""
    kernel = build_and_jit(transpose_code, "_transpose")
    if kernel:
        kernel.argtypes = [
            ctypes.POINTER(ctypes.c_float),  # Input tensor
            ctypes.POINTER(ctypes.c_float),  # Output tensor
            ctypes.c_int,     # Number of rows
            ctypes.c_int      # Number of columns
        ]
        kernel.restype = None
    return kernel

def transpose(input_mat: np.ndarray) -> np.ndarray:
    """
    Transpose a matrix using the assembly kernel.
    
    Args:
        input_mat: Input matrix to transpose
        
    Returns:
        Transposed matrix
    """
    # Ensure input is contiguous and float32
    input_mat = np.ascontiguousarray(input_mat, dtype=np.float32)
    
    # Create output matrix
    output_mat = np.zeros((input_mat.shape[1], input_mat.shape[0]), dtype=np.float32)
    
    # Get pointers to input and output matrices
    input_ptr = input_mat.ctypes.data_as(ctypes.c_void_p)
    output_ptr = output_mat.ctypes.data_as(ctypes.c_void_p)
    
    # Build and call kernel
    kernel = build_kernel()
    kernel(
        input_ptr,
        output_ptr,
        ctypes.c_int(input_mat.shape[0]),  # rows
        ctypes.c_int(input_mat.shape[1])   # columns
    )
    
    return output_mat

def test_transpose():
    """Test the transpose kernel with fused operations in a single pass."""
    # Test parameters
    sizes = [
        (4, 4),      # Small square matrix
        (8, 8),      # Medium square matrix
        (16, 16),    # Large square matrix
        (4, 8),      # Small rectangular matrix
        (8, 4),      # Small rectangular matrix (transposed)
        (16, 32),    # Large rectangular matrix
        (32, 16)     # Large rectangular matrix (transposed)
    ]
    
    for rows, cols in sizes:
        # Create test matrix
        input_mat = np.zeros((rows, cols), dtype=np.float32)
        for i in range(rows):
            for j in range(cols):
                input_mat[i, j] = i * cols + j
        
        # Run speed test with fused operations
        num_iters = 10
        times = []
        
        for _ in range(num_iters):
            start_time = time.perf_counter()
            
            # Fuse operations: transpose -> transpose back
            # This tests the kernel in a more realistic scenario
            t1 = transpose(input_mat)
            t2 = transpose(t1)
            
            end_time = time.perf_counter()
            times.append((end_time - start_time) / 2 * 1000)  # Average time per transpose
            
            # Verify correctness
            if not np.allclose(t2, input_mat):
                print(f"Error: Matrix {rows}x{cols} verification failed")
        
        # Print results
        avg_time = np.mean(times)
        print(f"Matrix {rows}x{cols}: {avg_time:.2f}ms")

if __name__ == "__main__":
    test_transpose() 