from OPENtransformer.core.asm.assembler.builder import build_and_jit
import numpy as np
import ctypes
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

gelu_code = """
.section __TEXT,__text,regular,pure_instructions
.globl _gelu
.align 2

// Data section for constants
.section __DATA,__data
.align 4
.const_sqrt_2_pi: .float 0.7978845608028654  // sqrt(2/pi)
.const_1: .float 1.0
.const_05: .float 0.5
.const_0: .float 0.0
.const_c1: .float 0.044715  // cubic coefficient
.const_epsilon: .float 0.000001  // Small value for numerical stability

// Constants for exp Taylor series
.const_1_div_2: .float 0.5       // 1/2!
.const_1_div_6: .float 0.166666666667  // 1/3!
.const_1_div_24: .float 0.041666666667  // 1/4!
.const_1_div_120: .float 0.008333333333  // 1/5!
.const_1_div_720: .float 0.001388888889  // 1/6!

// Constants for range reduction
.const_ln2: .float 0.693147180559945  // ln(2)
.const_inv_ln2: .float 1.442695040888963  // 1/ln(2)
.const_2: .float 2.0

// Constants for handling extreme values
.const_large_x: .float 3.4  // Threshold for extreme values
.const_neg_large_x: .float -3.4

.section __TEXT,__text
_gelu:
    // Save registers
    stp x29, x30, [sp, -32]!
    mov x29, sp

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

    // x0: pointer to input vector
    // x1: pointer to output vector
    // w2: number of elements (N)

    // Load constants
    adrp x9, .const_sqrt_2_pi@PAGE
    add x9, x9, .const_sqrt_2_pi@PAGEOFF
    ldr s8, [x9]            // sqrt(2/pi)
    
    adrp x9, .const_1@PAGE
    add x9, x9, .const_1@PAGEOFF
    ldr s9, [x9]            // 1.0
    
    adrp x9, .const_05@PAGE
    add x9, x9, .const_05@PAGEOFF
    ldr s10, [x9]           // 0.5
    
    adrp x9, .const_c1@PAGE
    add x9, x9, .const_c1@PAGEOFF
    ldr s11, [x9]           // 0.044715
    
    adrp x9, .const_epsilon@PAGE
    add x9, x9, .const_epsilon@PAGEOFF
    ldr s12, [x9]           // epsilon
    
    // Load Taylor series constants
    adrp x9, .const_1_div_2@PAGE
    add x9, x9, .const_1_div_2@PAGEOFF
    ldr s13, [x9]           // 1/2!
    
    adrp x9, .const_1_div_6@PAGE
    add x9, x9, .const_1_div_6@PAGEOFF
    ldr s14, [x9]           // 1/3!
    
    adrp x9, .const_1_div_24@PAGE
    add x9, x9, .const_1_div_24@PAGEOFF
    ldr s15, [x9]           // 1/4!
    
    adrp x9, .const_1_div_120@PAGE
    add x9, x9, .const_1_div_120@PAGEOFF
    ldr s16, [x9]           // 1/5!
    
    adrp x9, .const_1_div_720@PAGE
    add x9, x9, .const_1_div_720@PAGEOFF
    ldr s17, [x9]           // 1/6!
    
    // Load range reduction constants
    adrp x9, .const_ln2@PAGE
    add x9, x9, .const_ln2@PAGEOFF
    ldr s18, [x9]           // ln(2)
    
    adrp x9, .const_inv_ln2@PAGE
    add x9, x9, .const_inv_ln2@PAGEOFF
    ldr s19, [x9]           // 1/ln(2)
    
    adrp x9, .const_2@PAGE
    add x9, x9, .const_2@PAGEOFF
    ldr s20, [x9]           // 2.0
    
    adrp x9, .const_large_x@PAGE
    add x9, x9, .const_large_x@PAGEOFF
    ldr s21, [x9]           // 3.4
    
    adrp x9, .const_neg_large_x@PAGE
    add x9, x9, .const_neg_large_x@PAGEOFF
    ldr s22, [x9]           // -3.4
    
    adrp x9, .const_0@PAGE
    add x9, x9, .const_0@PAGEOFF
    ldr s23, [x9]           // 0.0

    // Process each element
    mov x19, #0     // i = 0
loop:
    cmp x19, x2
    b.ge done

    // Load input value
    ldr s0, [x0, x19, lsl #2]    // x
    
    // Check for extreme values first
    fcmp s0, s21                // if x > 3.4
    b.gt large_positive
    
    fcmp s0, s22                // if x < -3.4
    b.lt large_negative
    
    // Regular GELU calculation
    // Calculate inner term: x + 0.044715 * x^3
    fmul s1, s0, s0              // x^2
    fmul s1, s1, s0              // x^3
    fmul s1, s1, s11             // 0.044715 * x^3
    fadd s1, s0, s1              // x + 0.044715 * x^3
    
    // Calculate z = sqrt(2/π) * (x + 0.044715 * x^3)
    fmul s1, s1, s8              // z = sqrt(2/pi) * (...)
    
    // Range reduction for exp(z)
    // First calculate k = floor(z/ln(2))
    fmul s2, s1, s19             // z/ln(2)
    fcvtzs w20, s2                // floor(z/ln(2))
    scvtf s3, w20                 // convert back to float
    
    // Calculate r = z - k*ln(2)
    fmul s4, s3, s18             // k*ln(2)
    fsub s4, s1, s4              // r = z - k*ln(2)
    
    // Calculate exp(r) using Taylor series
    fmov s5, s9                  // result = 1.0
    
    // Add r term
    fadd s5, s5, s4              // result += r
    
    // r^2 term
    fmul s6, s4, s4              // r^2
    fmul s7, s6, s13             // r^2/2!
    fadd s5, s5, s7              // result += r^2/2!
    
    // r^3 term
    fmul s7, s7, s4              // r^3
    fmul s7, s7, s14             // r^3/3!
    fadd s5, s5, s7              // result += r^3/3!
    
    // r^4 term
    fmul s7, s7, s4              // r^4
    fmul s7, s7, s15             // r^4/4!
    fadd s5, s5, s7              // result += r^4/4!
    
    // r^5 term
    fmul s7, s7, s4              // r^5
    fmul s7, s7, s16             // r^5/5!
    fadd s5, s5, s7              // result += r^5/5!
    
    // r^6 term
    fmul s7, s7, s4              // r^6
    fmul s7, s7, s17             // r^6/6!
    fadd s5, s5, s7              // result += r^6/6!
    
    // Adjust for the range reduction: exp(z) = exp(r) * 2^k
    // We do this by adding k to the exponent bits
    fmov x21, d5                  // move float bits to integer register
    lsl w20, w20, #23              // shift k into exponent position
    add x21, x21, x20               // add to exponent
    fmov d5, x21                  // move back to float register
    
    // Calculate exp(-z) using the same method
    fneg s1, s1                  // -z
    
    // Range reduction for exp(-z)
    fmul s2, s1, s19             // -z/ln(2)
    fcvtzs w20, s2                // floor(-z/ln(2))
    scvtf s3, w20                 // convert back to float
    
    // Calculate r = -z - k*ln(2)
    fmul s4, s3, s18             // k*ln(2)
    fsub s4, s1, s4              // r = -z - k*ln(2)
    
    // Calculate exp(r) using Taylor series
    fmov s6, s9                  // result = 1.0
    
    // Add r term
    fadd s6, s6, s4              // result += r
    
    // r^2 term
    fmul s7, s4, s4              // r^2
    fmul s7, s7, s13             // r^2/2!
    fadd s6, s6, s7              // result += r^2/2!
    
    // r^3 term
    fmul s7, s7, s4              // r^3
    fmul s7, s7, s14             // r^3/3!
    fadd s6, s6, s7              // result += r^3/3!
    
    // r^4 term
    fmul s7, s7, s4              // r^4
    fmul s7, s7, s15             // r^4/4!
    fadd s6, s6, s7              // result += r^4/4!
    
    // r^5 term
    fmul s7, s7, s4              // r^5
    fmul s7, s7, s16             // r^5/5!
    fadd s6, s6, s7              // result += r^5/5!
    
    // r^6 term
    fmul s7, s7, s4              // r^6
    fmul s7, s7, s17             // r^6/6!
    fadd s6, s6, s7              // result += r^6/6!
    
    // Adjust for the range reduction: exp(-z) = exp(r) * 2^k
    fmov x21, d6                  // move float bits to integer register
    lsl w20, w20, #23              // shift k into exponent position
    add x21, x21, x20               // add to exponent
    fmov d6, x21                  // move back to float register
    
    // Add epsilon to both exp(z) and exp(-z) for stability
    fadd s5, s5, s12             // exp(z) += epsilon
    fadd s6, s6, s12             // exp(-z) += epsilon
    
    // Calculate tanh(z) = (exp(z) - exp(-z)) / (exp(z) + exp(-z))
    fsub s7, s5, s6              // exp(z) - exp(-z)
    fadd s1, s5, s6              // exp(z) + exp(-z)
    fdiv s2, s7, s1              // tanh(z)
    
    // Calculate final GELU: 0.5 * x * (1 + tanh(z))
    fadd s2, s9, s2              // 1 + tanh(z)
    fmul s2, s0, s2              // x * (1 + tanh(z))
    fmul s2, s10, s2             // 0.5 * x * (1 + tanh(z))
    
    // Store result
    str s2, [x1, x19, lsl #2]
    
    b next_element
    
large_positive:
    // For large positive values, GELU approaches x
    str s0, [x1, x19, lsl #2]
    b next_element
    
large_negative:
    // For large negative values, GELU approaches 0
    str s23, [x1, x19, lsl #2]

next_element:
    add x19, x19, #1               // i++
    b loop

done:
    // Restore non-volatile registers
    ldp d14, d15, [sp], #16
    ldp d12, d13, [sp], #16
    ldp d10, d11, [sp], #16
    ldp d8, d9, [sp], #16
    ldp x27, x28, [sp], #16
    ldp x25, x26, [sp], #16
    ldp x23, x24, [sp], #16
    ldp x21, x22, [sp], #16
    ldp x19, x20, [sp], #16

    // Restore frame pointer and link register
    ldp x29, x30, [sp], #32
    ret
"""

def get_kernel_code():
    logger.debug("Returning GELU assembly code")
    return gelu_code

def execute_kernel(*args):
    # Try to use the compiled kernel first
    kernel = build_and_jit(gelu_code, "_gelu")
    if kernel:
        kernel(*args)
    else:
        # Fall back to numpy implementation
        N = args[2]
        # Convert integer addresses to arrays via ctypes
        in_arr = np.frombuffer((ctypes.c_float * N).from_address(args[0]), dtype=np.float32)
        out_arr = np.frombuffer((ctypes.c_float * N).from_address(args[1]), dtype=np.float32)
        out_arr[:] = 0.5 * in_arr * (1 + np.tanh(np.sqrt(2/np.pi) * (in_arr + 0.044715 * in_arr**3)))

def test_gelu_kernel():
    # Create test data
    N = 1024
    x = np.random.randn(N).astype(np.float32)
    y = np.zeros_like(x)

    # Get pointers to the arrays
    x_ptr = x.ctypes.data_as(ctypes.c_void_p)
    y_ptr = y.ctypes.data_as(ctypes.c_void_p)

    # Build and load the kernel
    gelu = build_and_jit(gelu_code, "_gelu")

    # Run kernel
    gelu(x_ptr, y_ptr, N)

    # Calculate expected result using numpy
    def numpy_gelu(x):
        return 0.5 * x * (1 + np.tanh(np.sqrt(2/np.pi) * (x + 0.044715 * x**3)))

    expected = numpy_gelu(x)

    # Compare results
    max_diff = np.max(np.abs(y - expected))
    mean_diff = np.mean(np.abs(y - expected))
    logging.info(f"Max difference: {max_diff}")
    logging.info(f"Mean difference: {mean_diff}")

    # Print first few values for comparison
    for i in range(min(5, N)):
        logging.info(f"Input: {x[i]:.8f}, ASM: {y[i]:.8f}, NumPy: {expected[i]:.8f}")

    # Test passes if max difference is small enough
    passed = max_diff < 0.01
    if passed:
        logging.info("Test passed!")
    else:
        logging.info("Test failed - differences too large")

    return gelu, passed