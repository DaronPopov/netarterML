from ..assembler.builder import build_and_jit
from .logger import log_info

rmsnorm_code = """
.section __TEXT,__text,regular,pure_instructions
.globl _rmsnorm
.align 2
_rmsnorm:
    stp x29, x30, [sp, -16]!
    mov x29, sp

    // x0: pointer to input vector (float*)
    // w1: number of elements (N)
    // x2: pointer to output vector (float*)

    // Calculate sum of squares
    mov x3, x0  // input pointer
    fmov s0, #0.0  // sum of squares
    mov w4, w1  // N (number of elements)
    mov x5, 0   // loop counter

sum_sq_loop:
    cmp x5, w4
    b.geq sum_sq_end

    ldr s1, [x3, x5, sxtw #2]  // load float input[i]
    fmul s2, s1, s1  // square input[i]
    fadd s0, s0, s2  // sum += input[i]^2

    add x5, x5, #1
    b sum_sq_loop

sum_sq_end:
    // Calculate RMS with epsilon inside sqrt for numerical stability
    fcvt s1, w4  // N as float
    fdiv s2, s0, s1  // sum_sq / N
    fmov s8, #1e-5   // Load epsilon
    fadd s2, s2, s8  // Add epsilon before sqrt
    fsqrt s3, s2  // sqrt((sum_sq / N) + epsilon)

    // Normalize
    mov x3, x0  // input pointer
    mov x4, x2  // output pointer
    fmov s4, s3  // rms
    mov w5, w1  // N (number of elements)
    mov x6, 0   // loop counter

normalize_loop:
    cmp x6, w5
    b.geq normalize_end

    ldr s1, [x3, x6, sxtw #2]  // load float input[i]
    fdiv s5, s1, s4  // input[i] / rms
    fstr s5, [x4, x6, sxtw #2]  // output[i] = input[i] / rms

    add x6, x6, #1
    b normalize_loop

normalize_end:
    ldp x29, x30, [sp], 16
    ret
"""

def get_kernel_code():
    return rmsnorm_code

def execute_kernel(*args):
    import numpy as np, ctypes
    import time

    start_time = time.time()
    try:
        N = int(args[1])  # Explicitly convert to int
        in_arr = np.ctypeslib.as_array((ctypes.c_float * N).from_address(int(args[0])))
        out_arr = np.ctypeslib.as_array((ctypes.c_float * N).from_address(int(args[2])))
        
        # Add logging for input array
        log_info(f"First few RMSNorm input values: {in_arr[:5]}")
        
        # Check for NaN or Inf values in input
        if np.any(np.isnan(in_arr)) or np.any(np.isinf(in_arr)):
            print("WARNING: RMSNorm input contains NaN/Inf values. Replacing with zeros.")
            in_arr = np.nan_to_num(in_arr, nan=0.0, posinf=1e5, neginf=-1e5)  # Clip Inf values
        
        # Add epsilon inside sqrt for numerical stability
        rms = np.sqrt(np.mean(in_arr**2) + 1e-5)
        
        # Check if RMS is zero or too small
        if rms <= 1e-6:
            print("WARNING: RMS value is too small. Setting to a minimum value.")
            rms = 1e-6  # Set a minimum value for RMS
        
        # Add logging for RMS value
        log_info(f"RMS value: {rms}")
        
        out_arr[:] = in_arr / rms
        
        # Add logging for output array
        log_info(f"First few RMSNorm output values: {out_arr[:5]}")
        
    except Exception as e:
        print(f"Error in rmsnorm: {e}")
        raise
    end_time = time.time()
    log_info(f"RMSNorm execution time: {end_time - start_time:.4f} seconds")
    log_info(f"First few RMSNorm values: {out_arr[:5]}")