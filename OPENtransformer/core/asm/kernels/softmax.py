from OPENtransformer.core.asm.assembler.builder import build_and_jit
import numpy as np
import ctypes
import time
import logging

logger = logging.getLogger("OPENtransformer.core.asm.softmax")

softmax_code = """
.section __TEXT,__text,regular,pure_instructions

// Data section for constants
.section __DATA,__data
.align 4
.float_neg_max: .float -3.402823466e+38  // Negative FLT_MAX
.const_0: .float 0.0
.const_1: .float 1.0
.const_2: .float 2.0
.const_ln2: .float 0.693147180559945  // ln(2)
.const_inv_ln2: .float 1.442695040888963  // 1/ln(2)
.const_epsilon: .float 0.000001  // Small value for numerical stability

// Taylor series coefficients for exp(x)
.const_1_div_2: .float 0.5       // 1/2!
.const_1_div_6: .float 0.166666666667  // 1/3!
.const_1_div_24: .float 0.041666666667  // 1/4!
.const_1_div_120: .float 0.008333333333  // 1/5!
.const_1_div_720: .float 0.001388888889  // 1/6!

.section __TEXT,__text
.globl _softmax
.align 2
_softmax:
    stp x29, x30, [sp, -32]!
    mov x29, sp

    // Save non-volatile registers
    stp x19, x20, [sp, -16]!
    stp x21, x22, [sp, -16]!
    stp d8, d9, [sp, -16]!
    stp d10, d11, [sp, -16]!
    stp d12, d13, [sp, -16]!
    stp d14, d15, [sp, -16]!

    // x0: pointer to input vector (float*)
    // w1: number of elements (N)
    // x2: pointer to output vector (float*)

    // Load constants
    adrp x9, .float_neg_max@PAGE
    add x9, x9, .float_neg_max@PAGEOFF
    ldr s0, [x9]              // Initialize max to smallest possible float
    
    adrp x9, .const_0@PAGE
    add x9, x9, .const_0@PAGEOFF
    ldr s3, [x9]              // s3 = 0.0 for sum initialization
    
    adrp x9, .const_1@PAGE
    add x9, x9, .const_1@PAGEOFF
    ldr s8, [x9]              // 1.0
    
    adrp x9, .const_ln2@PAGE
    add x9, x9, .const_ln2@PAGEOFF
    ldr s9, [x9]              // ln(2)
    
    adrp x9, .const_inv_ln2@PAGE
    add x9, x9, .const_inv_ln2@PAGEOFF
    ldr s10, [x9]             // 1/ln(2)
    
    adrp x9, .const_epsilon@PAGE
    add x9, x9, .const_epsilon@PAGEOFF
    ldr s11, [x9]             // epsilon
    
    // Load Taylor series coefficients
    adrp x9, .const_1_div_2@PAGE
    add x9, x9, .const_1_div_2@PAGEOFF
    ldr s12, [x9]             // 1/2!
    
    adrp x9, .const_1_div_6@PAGE
    add x9, x9, .const_1_div_6@PAGEOFF
    ldr s13, [x9]             // 1/3!
    
    adrp x9, .const_1_div_24@PAGE
    add x9, x9, .const_1_div_24@PAGEOFF
    ldr s14, [x9]             // 1/4!
    
    adrp x9, .const_1_div_120@PAGE
    add x9, x9, .const_1_div_120@PAGEOFF
    ldr s15, [x9]             // 1/5!
    
    adrp x9, .const_1_div_720@PAGE
    add x9, x9, .const_1_div_720@PAGEOFF
    ldr s16, [x9]             // 1/6!

    // Find max value
    mov x3, x0  // input pointer
    mov w4, w1  // N (number of elements)
    mov w5, #0  // loop counter

find_max_loop:
    cmp w5, w4
    b.ge find_max_end

    ldr s1, [x3, w5, UXTW #2]
    fcmp s1, s0
    b.le not_max
    fmov s0, s1
not_max:
    add w5, w5, #1
    b find_max_loop

find_max_end:
    // Compute exp(x - max) and sum
    mov x3, x0  // input pointer
    mov x4, x2  // output pointer
    fmov s2, s0  // max value
    mov w5, w1  // N
    mov w6, #0  // loop counter

exp_loop:
    cmp w6, w5
    b.ge exp_end

    ldr s1, [x3, w6, UXTW #2]
    fsub s4, s1, s2            // x - max

    // Range reduction for exp(x)
    fmul s5, s4, s10           // x/ln(2)
    fcvtzs w7, s5              // k = floor(x/ln(2))
    scvtf s6, w7               // float(k)
    fmul s7, s6, s9            // k*ln(2)
    fsub s8, s4, s7            // r = x - k*ln(2)
    
    // Calculate exp(r) using Taylor series
    fmov s17, s8               // result = 1.0
    
    // Add r term
    fadd s17, s17, s8          // result += r
    
    // r^2 term
    fmul s18, s8, s8           // r^2
    fmul s18, s18, s12         // r^2/2!
    fadd s17, s17, s18         // result += r^2/2!
    
    // r^3 term
    fmul s18, s18, s8          // r^3
    fmul s18, s18, s13         // r^3/3!
    fadd s17, s17, s18         // result += r^3/3!
    
    // r^4 term
    fmul s18, s18, s8          // r^4
    fmul s18, s18, s14         // r^4/4!
    fadd s17, s17, s18         // result += r^4/4!
    
    // r^5 term
    fmul s18, s18, s8          // r^5
    fmul s18, s18, s15         // r^5/5!
    fadd s17, s17, s18         // result += r^5/5!
    
    // r^6 term
    fmul s18, s18, s8          // r^6
    fmul s18, s18, s16         // r^6/6!
    fadd s17, s17, s18         // result += r^6/6!
    
    // Adjust for range reduction: exp(x) = exp(r) * 2^k
    fmov x8, d17               // move float bits to integer register
    lsl w7, w7, #23            // shift k into exponent position
    add x8, x8, x7             // add to exponent
    fmov d17, x8               // move back to float register
    
    // Add epsilon for stability
    fadd s17, s17, s11
    
    str s17, [x4, w6, UXTW #2]
    fadd s3, s3, s17           // sum += exp(x - max)

    add w6, w6, #1
    b exp_loop

exp_end:
    // Normalize
    mov x4, x2                 // output pointer
    fmov s6, s3                // sum
    mov w5, w1                 // N
    mov w6, #0                 // loop counter

normalize_loop:
    cmp w6, w5
    b.ge normalize_end

    ldr s5, [x4, w6, UXTW #2]
    fdiv s7, s5, s6            // output[i] / sum
    str s7, [x4, w6, UXTW #2]

    add w6, w6, #1
    b normalize_loop

normalize_end:
    // Restore non-volatile registers
    ldp d14, d15, [sp], #16
    ldp d12, d13, [sp], #16
    ldp d10, d11, [sp], #16
    ldp d8, d9, [sp], #16
    ldp x21, x22, [sp], #16
    ldp x19, x20, [sp], #16
    
    // Epilogue
    ldp x29, x30, [sp], #32
    ret
"""

def get_kernel_code():
    return softmax_code

def execute_kernel(*args):
    """Execute the softmax kernel.
    
    Args:
        input_ptr: Pointer to input array
        N: Number of elements
        output_ptr: Pointer to output array
    """
    start_time = time.time()
    try:
        input_ptr = args[0]
        N = args[1]
        output_ptr = args[2]
        
        if isinstance(input_ptr, ctypes.c_void_p):
            input_ptr = input_ptr.value
        if isinstance(output_ptr, ctypes.c_void_p):
            output_ptr = output_ptr.value
            
        # Build kernel if not already built
        kernel = build_and_jit(softmax_code, "_softmax")
        if not kernel:
            raise RuntimeError("Failed to build softmax kernel")
            
        # Set function types
        kernel.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_float)
        ]
        kernel.restype = None
        
        # Execute kernel
        kernel(input_ptr, N, output_ptr)
        
    except Exception as e:
        logger.error(f"Error in softmax: {e}")
        raise
        
    end_time = time.time()
    logger.debug(f"Softmax execution time: {end_time - start_time:.4f} seconds")
    
    # Debug output
    x_arr = np.ctypeslib.as_array((ctypes.c_float * N).from_address(input_ptr))
    out_arr = np.ctypeslib.as_array((ctypes.c_float * N).from_address(output_ptr))
    logger.debug(f"First few softmax values: {out_arr[:5]}")
    logger.debug(f"Sum of softmax values: {np.sum(out_arr)}")