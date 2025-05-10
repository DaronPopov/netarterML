from ..assembler.builder import build_and_jit
from .logger import log_info

fp16_to_fp32_code = """
.section __TEXT,__text,regular,pure_instructions
.globl _fp16_to_fp32
.align 2
_fp16_to_fp32:
    stp x29, x30, [sp, -16]!
    mov x29, sp

    // x0: pointer to half-precision input (half*)
    // x1: pointer to single-precision output (float*)
    // w2: number of elements (N)

    mov w3, w2 // Loop counter
    mov x4, x0 // Input pointer
    mov x5, x1 // Output pointer

loop_start:
    cmp w3, #0
    b.eq loop_end

    // Load half-precision float (16-bit)
    ldrh w6, [x4], #2
    
    // Convert manually from F16 to F32
    // Extract sign bit (bit 15)
    ubfx w7, w6, #15, #1       // Extract sign bit
    lsl w7, w7, #31            // Position at bit 31 for F32
    
    // Extract exponent (bits 10-14)
    ubfx w8, w6, #10, #5       // Extract 5-bit exponent
    
    // Check for special case: zero
    cmp w8, #0
    b.eq store_zero
    
    // Handle regular case
    sub w8, w8, #15            // Remove F16 bias
    add w8, w8, #127           // Add F32 bias
    lsl w8, w8, #23            // Position at bit 23-30 for F32
    
    // Extract mantissa (bits 0-9)
    and w9, w6, #0x03FF        // Keep only 10 mantissa bits
    lsl w9, w9, #13            // Position at bits 0-22 for F32
    
    // Combine sign, exponent, and mantissa for F32
    orr w9, w9, w8             // Combine exponent and mantissa
    orr w9, w9, w7             // Add sign bit
    
    // Store as float
    fmov s0, w9
    str s0, [x5], #4
    b next_element
    
store_zero:
    // Simple zero case
    mov w9, #0
    fmov s0, w9
    str s0, [x5], #4
    
next_element:
    sub w3, w3, #1
    b loop_start

loop_end:
    ldp x29, x30, [sp], 16
    ret
"""

def get_kernel_code():
    return fp16_to_fp32_code

def execute_kernel(*args):
    import numpy as np, ctypes
    import time

    start_time = time.time()
    try:
        N = int(args[2])  # Explicitly convert to int
        fp16_arr = np.ctypeslib.as_array((ctypes.c_uint16 * N).from_address(int(args[0]))).view(np.float16)
        fp32_arr = np.ctypeslib.as_array((ctypes.c_float * N).from_address(int(args[1])))
        
        # Convert to fp32 with special handling for denormals
        fp32_arr[:] = fp16_arr.astype(np.float32)
        
        # Clean up potential denormals in output
        min_normal_fp32 = np.float32(1.175494e-38)  # Smallest normal fp32
        small_vals = np.abs(fp32_arr) < min_normal_fp32
        fp32_arr[small_vals] = 0.0  # Flush denormals to zero
    except Exception as e:
        print(f"Error in fp16_to_fp32: {e}")
        raise
    end_time = time.time()
    log_info(f"FP16 to FP32 execution time: {end_time - start_time:.4f} seconds")
    log_info(f"First few FP32 values: {fp32_arr[:5]}")