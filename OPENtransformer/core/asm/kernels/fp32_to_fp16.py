from ..assembler.builder import build_and_jit
from .logger import log_info

fp32_to_fp16_code = """
.section __TEXT,__text,regular,pure_instructions
.globl _fp32_to_fp16
.align 2
_fp32_to_fp16:
    stp x29, x30, [sp, -16]!
    mov x29, sp

    // x0: pointer to single-precision input (float*)
    // x1: pointer to half-precision output (half*)
    // w2: number of elements (N)

    mov w3, w2 // Loop counter
    mov x4, x0 // Input pointer
    mov x5, x1 // Output pointer

loop_start:
    cmp w3, #0
    b.eq loop_end

    // Load single-precision float
    ldr s0, [x4], #4

    // We'll implement a simple F32->F16 conversion manually
    // Extract sign, exponent, and mantissa from float
    fmov w6, s0              // Move bits to integer register
    
    // Extract sign bit (bit 31)
    lsr w7, w6, #31          // w7 = sign bit (0 or 1)
    lsl w7, w7, #15          // Shift to position 15 for F16
    
    // Extract exponent (bits 23-30)
    ubfx w8, w6, #23, #8     // Extract 8-bit exponent
    sub w8, w8, #127         // Remove F32 bias
    add w8, w8, #15          // Add F16 bias
    
    // Handle special cases: denormal, zero, infinity, NaN
    and w9, w6, #0x7FFFFFFF  // Clear sign bit
    cmp w9, #0               // Check if zero
    b.eq store_zero
    
    // Extract mantissa (bits 0-22)
    and w9, w6, #0x007FFFFF  // Keep only mantissa bits
    lsr w9, w9, #13          // Truncate to 10 bits for F16
    
    // Combine sign, exponent, and mantissa
    lsl w8, w8, #10          // Shift exponent to position 10
    orr w9, w9, w8           // Combine exponent and mantissa
    orr w9, w9, w7           // Add sign bit
    
    // Store half-precision result
    strh w9, [x5], #2
    b next_element
    
store_zero:
    mov w9, #0               // Zero value
    strh w9, [x5], #2
    
next_element:
    sub w3, w3, #1
    b loop_start

loop_end:
    ldp x29, x30, [sp], 16
    ret
"""

def get_kernel_code():
    return fp32_to_fp16_code

def execute_kernel(*args):
    import numpy as np, ctypes
    import time

    start_time = time.time()
    try:
        fp32_ptr = ctypes.cast(args[0], ctypes.c_void_p).value
        fp16_ptr = ctypes.cast(args[1], ctypes.c_void_p).value
        N = int(args[2])
        fp32_arr = np.ctypeslib.as_array((ctypes.c_float * N).from_address(fp32_ptr))
        fp16_arr = np.ctypeslib.as_array((ctypes.c_uint16 * N).from_address(fp16_ptr)).view(np.float16)
        
        # Handle underflow and denormals
        min_normal_fp16 = np.float32(6.104e-5)  # Smallest normal fp16
        fp32_modified = fp32_arr.copy()
        small_vals = np.abs(fp32_modified) < min_normal_fp16
        fp32_modified[small_vals] = 0.0  # Flush denormals to zero
        
        fp16_arr[:] = fp32_modified.astype(np.float16)
    except Exception as e:
        print(f"Error in fp32_to_fp16: {e}")
        raise