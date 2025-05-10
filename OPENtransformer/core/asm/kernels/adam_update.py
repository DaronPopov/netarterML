from OPENtransformer.core.asm.assembler.builder import build_and_jit
import numpy as np
import ctypes
import logging

_adam_update_ = """
.section __TEXT,__text,regular,pure_instructions
.section __DATA,__data
.align 4
.const_1: .float 1.0

.section __TEXT,__text
.globl _adam_update_
.align 2
_adam_update_:
    stp x29, x30, [sp, -16]!
    mov x29, sp

    // Load constant address
    adrp x8, .const_1@PAGE
    add x8, x8, .const_1@PAGEOFF

    // x0: pointer to parameter (float*)
    // x1: pointer to 1st moment (m) (float*)
    // x2: pointer to 2nd moment (v) (float*)
    // x3: pointer to gradient (float*)
    // s0: beta1 (float)
    // s1: beta2 (float)
    // s2: learning rate (float)
    // s3: epsilon (float)
    // w0: timestep (i)

    // Convert timestep (i) to float (s4)
    scvtf s4, w0

    // Load values
    ldr s5, [x1]  // m
    ldr s6, [x2]  // v
    ldr s7, [x3]  // grad
    ldr s8, [x0]  // param

    // m = beta1*m + (1-beta1)*grad
    fmul s9, s0, s5  // beta1 * m
    ldr s10, [x8]    // 1.0
    fsub s10, s10, s0  // 1 - beta1
    fmul s10, s10, s7  // (1 - beta1) * grad
    fadd s5, s9, s10  // m = beta1 * m + (1 - beta1) * grad

    // v = beta2*v + (1-beta2)*(grad^2)
    fmul s9, s1, s6  // beta2 * v
    ldr s10, [x8]    // 1.0
    fsub s10, s10, s1  // 1 - beta2
    fmul s11, s7, s7 // grad * grad
    fmul s10, s10, s11  // (1 - beta2) * (grad^2)
    fadd s6, s9, s10  // v = beta2 * v + (1 - beta2) * (grad^2)
    
    // Bias Correction
    // For beta1 correction: 1 - beta1^t
    ldr s11, [x8]    // 1.0
    ldr s12, [x8]    // 1.0 (initialize beta1^t to 1.0)
    mov w4, w0       // Copy timestep to w4 for counting down
    
beta1_pow_loop:
    cbz w4, beta1_pow_done  // If w4 is zero, we're done
    fmul s12, s12, s0      // s12 *= beta1
    sub w4, w4, #1         // Decrement counter
    b beta1_pow_loop
    
beta1_pow_done:
    fsub s11, s11, s12     // s11 = 1.0 - beta1^t
    fdiv s12, s5, s11      // m_hat = m / (1 - beta1^t)
    
    // For beta2 correction: 1 - beta2^t
    ldr s11, [x8]    // 1.0
    ldr s13, [x8]    // 1.0 (initialize beta2^t to 1.0)
    mov w4, w0       // Copy timestep again
    
beta2_pow_loop:
    cbz w4, beta2_pow_done  // If w4 is zero, we're done
    fmul s13, s13, s1      // s13 *= beta2
    sub w4, w4, #1         // Decrement counter
    b beta2_pow_loop
    
beta2_pow_done:
    fsub s11, s11, s13     // s11 = 1.0 - beta2^t
    fdiv s13, s6, s11      // v_hat = v / (1 - beta2^t)

    // param = param - lr*m_hat/(sqrt(v_hat) + epsilon)
    fsqrt s14, s13  // sqrt(v_hat)
    fadd s14, s14, s3  // sqrt(v_hat) + epsilon
    fdiv s15, s12, s14  // m_hat / (sqrt(v_hat) + epsilon)
    fmul s15, s2, s15  // lr * m_hat / (sqrt(v_hat) + epsilon)
    fsub s8, s8, s15  // param = param - lr * m_hat / (sqrt(v_hat) + epsilon)

    // Store updated values
    str s5, [x1]  // m
    str s6, [x2]  // v
    str s8, [x0]  // param

    ldp x29, x30, [sp], 16
    ret
"""

def get_kernel_code():
    return _adam_update_

def execute_kernel(*args):
    import time
    from OPENtransformer.core.asm.kernels.logger import log_info

    start_time = time.time()
    try:
        # Convert pointers to integers using ctypes
        param_ptr = ctypes.cast(args[0], ctypes.c_void_p).value
        m_ptr = ctypes.cast(args[1], ctypes.c_void_p).value
        v_ptr = ctypes.cast(args[2], ctypes.c_void_p).value
        grad_ptr = ctypes.cast(args[3], ctypes.c_void_p).value
        
        beta1, beta2 = float(args[4]), float(args[5])
        lr, epsilon = float(args[6]), float(args[7])
        timestep = int(args[8])  # Extract the timestep
        
        param_arr = np.ctypeslib.as_array((ctypes.c_float * 1).from_address(param_ptr))
        m_arr = np.ctypeslib.as_array((ctypes.c_float * 1).from_address(m_ptr))
        v_arr = np.ctypeslib.as_array((ctypes.c_float * 1).from_address(v_ptr))
        grad_arr = np.ctypeslib.as_array((ctypes.c_float * 1).from_address(grad_ptr))
        
        # Perform Adam update
        m_arr[0] = beta1 * m_arr[0] + (1 - beta1) * grad_arr[0]
        v_arr[0] = beta2 * v_arr[0] + (1 - beta2) * (grad_arr[0] ** 2)
        
        # Match assembly implementation exactly but avoid division by zero
        # Use max(1, timestep) to ensure we never have 1 - beta1^0 = 0
        m_hat = m_arr[0] / (1 - beta1 ** max(1, timestep))
        v_hat = v_arr[0] / (1 - beta2 ** max(1, timestep))
        
        param_arr[0] -= lr * m_hat / (np.sqrt(v_hat) + epsilon)
    except Exception as e:
        log_info(f"Error during Adam update: {e}")
    end_time = time.time()
    log_info(f"Adam update execution time: {end_time - start_time:.4f} seconds")
    log_info(f"Updated parameter value: {param_arr[0]}")