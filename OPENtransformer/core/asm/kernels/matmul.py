from ..assembler.builder import build_and_jit
from .logger import log_info
import ctypes
import numpy as np
import logging

logger = logging.getLogger("assembly_test.matmul")

matmul_code = """
.section __TEXT,__text,regular,pure_instructions
.globl _matmul
.align 2
_matmul:
    stp x29, x30, [sp, -32]!
    mov x29, sp

    // x0: pointer to matrix A (N x K)
    // x1: pointer to matrix B (K x M)
    // x2: pointer to matrix C (N x M)
    // w3: N (rows of A, rows of C)
    // w4: K (cols of A, rows of B)
    // w5: M (cols of B, cols of C)

    // i loop: i in w6
    mov w6, #0               // i = 0
i_loop_start:
    cmp w6, w3              // if i >= N, exit i loop
    b.ge i_loop_end

    // j loop: j in w7
    mov w7, #0               // j = 0
j_loop_start:
    cmp w7, w5              // if j >= M, exit j loop
    b.ge j_loop_end

    // Initialize C[i][j] = 0.0
    fmov s0, #0.0
    // Compute offset = i * M + j (in w8)
    mul w8, w6, w5          // w8 = i * M
    add w8, w8, w7          // w8 = i * M + j
    uxtw x8, w8             // convert to 64-bit
    add x8, x2, x8, lsl #2   // address of C[i][j]
    str s0, [x8]

    // k loop: k in w9
    mov w9, #0              // k = 0
k_loop_start:
    cmp w9, w4             // if k >= K, exit k loop
    b.ge k_loop_end

    // Compute A[i][k]:
    mul w10, w6, w4         // w10 = i * K
    add w10, w10, w9        // w10 = i * K + k
    uxtw x10, w10           // convert to 64-bit
    add x10, x0, x10, lsl #2 // address of A[i][k]
    ldr s1, [x10]

    // Compute B[k][j]:
    mul w11, w9, w5         // w11 = k * M
    add w11, w11, w7        // w11 = k * M + j
    uxtw x11, w11           // convert to 64-bit
    add x11, x1, x11, lsl #2 // address of B[k][j]
    ldr s2, [x11]

    // C[i][j] += A[i][k] * B[k][j]
    fmul s3, s1, s2
    // Recalculate address of C[i][j]:
    mul w8, w6, w5
    add w8, w8, w7
    uxtw x8, w8
    add x8, x2, x8, lsl #2
    ldr s0, [x8]
    fadd s0, s0, s3
    str s0, [x8]

    add w9, w9, #1         // k++
    b k_loop_start
k_loop_end:

    add w7, w7, #1         // j++
    b j_loop_start
j_loop_end:

    add w6, w6, #1         // i++
    b i_loop_start
i_loop_end:
    ldp x29, x30, [sp], 32
    ret
"""

def get_kernel_code():
    """
    Returns the ARM assembly code for matrix multiplication
    This function is called by build_and_load to get the assembly code
    """
    logger.debug("Returning matmul assembly code")
    return matmul_code

def test_matmul_kernel():
    """
    Test function to verify the matmul kernel works correctly
    """
    from ..assembler.builder import build_and_load
    
    logger.info("Testing matmul kernel")
    
    # Build and load the kernel
    kernel = build_and_load(matmul_code, "matmul")
    
    # Create test matrices
    a = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    b = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32)
    c = np.zeros((2, 2), dtype=np.float32)
    
    logger.debug(f"Test input A: {a}")
    logger.debug(f"Test input B: {b}")
    
    # Call the kernel
    kernel(a.ctypes.data, b.ctypes.data, c.ctypes.data, 2, 2, 2)
    
    # Calculate expected result
    expected = np.matmul(a, b)
    
    logger.info(f"Test result: {c}")
    logger.info(f"Expected: {expected}")
    
    # Check if the result is close to the expected
    return kernel, np.allclose(c, expected)

def execute_kernel(*args):
    import ctypes, numpy as np
    import time
    
    logger.info("Executing matmul kernel with numpy fallback (for reference only)")
    
    start_time = time.time()
    # args: A_ptr, B_ptr, C_ptr, N, K, M
    N, K, M = args[3], args[4], args[5]
    
    A_ptr = ctypes.cast(args[0], ctypes.POINTER(ctypes.c_float * (N * K))).contents
    B_ptr = ctypes.cast(args[1], ctypes.POINTER(ctypes.c_float * (K * M))).contents
    C_ptr = ctypes.cast(args[2], ctypes.POINTER(ctypes.c_float * (N * M))).contents
    
    A = np.frombuffer(A_ptr, dtype=np.float32).reshape((N, K))
    B = np.frombuffer(B_ptr, dtype=np.float32).reshape((K, M))
    C = np.frombuffer(C_ptr, dtype=np.float32).reshape((N, M))
    
    C[:] = A.dot(B)
    end_time = time.time()
    logger.info(f"Matmul execution time: {end_time - start_time:.4f} seconds")
    logger.debug(f"First few matmul values: {C.flatten()[:5]}")