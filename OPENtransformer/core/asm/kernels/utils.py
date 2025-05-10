import ctypes
import os
import numpy as np
import mmap
import sys

def execute_asm(code, *args):
    """
    Fallback implementation of the assembly kernels using NumPy.
    """
    # For weight_initializer: fill all elements with 1.0
    if "_weight_initializer" in code:
        input_ptr = args[0]
        rows = args[1]
        cols = args[2]
        array = np.ctypeslib.as_array(input_ptr, shape=(rows, cols))
        array.fill(1.0)
        return

    # For softmax: compute softmax using numpy
    elif "_softmax" in code:
        x_ptr, N, out_ptr = args
        x = np.ctypeslib.as_array((ctypes.c_float * N).from_address(x_ptr))
        out = np.ctypeslib.as_array((ctypes.c_float * N).from_address(out_ptr))
        e_x = np.exp(x - np.max(x))
        out[:] = e_x / e_x.sum()
        return

    # For matmul: compute matrix multiplication using numpy
    elif "_matmul" in code:
        A_ptr, B_ptr, C_ptr, N, K, M = args
        A = np.ctypeslib.as_array((ctypes.c_float * (N * K)).from_address(A_ptr)).reshape((N, K))
        B = np.ctypeslib.as_array((ctypes.c_float * (K * M)).from_address(B_ptr)).reshape((K, M))
        C = np.ctypeslib.as_array((ctypes.c_float * (N * M)).from_address(C_ptr)).reshape((N, M))
        C[:] = np.dot(A, B)
        return

    # For layer_norm: compute layer normalization using numpy
    elif "_layer_norm" in code:
        x_ptr, N, out_ptr = args
        x = np.ctypeslib.as_array((ctypes.c_float * N).from_address(x_ptr))
        out = np.ctypeslib.as_array((ctypes.c_float * N).from_address(out_ptr))
        mean = np.mean(x)
        var = np.var(x)
        out[:] = (x - mean) / np.sqrt(var + 1e-5)
        return

    # For gelu: compute GELU activation using numpy
    elif "_gelu" in code:
        x_ptr, out_ptr, N = args
        x = np.ctypeslib.as_array((ctypes.c_float * N).from_address(x_ptr))
        out = np.ctypeslib.as_array((ctypes.c_float * N).from_address(out_ptr))
        out[:] = 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))
        return

    # For fp32_to_fp16: convert float32 to float16 using numpy
    elif "_fp32_to_fp16" in code:
        fp32_ptr, fp16_ptr, N = args
        fp32_arr = np.ctypeslib.as_array((ctypes.c_float * N).from_address(fp32_ptr))
        fp16_arr = np.ctypeslib.as_array((ctypes.c_uint16 * N).from_address(fp16_ptr)).view(np.float16)
        fp16_arr[:] = fp32_arr.astype(np.float16)
        return

    # For fp16_to_fp32: convert float16 to float32 using numpy
    elif "_fp16_to_fp32" in code:
        fp16_ptr, fp32_ptr, N = args
        fp16_arr = np.ctypeslib.as_array((ctypes.c_uint16 * N).from_address(fp16_ptr)).view(np.float16)
        fp32_arr = np.ctypeslib.as_array((ctypes.c_float * N).from_address(fp32_ptr))
        fp32_arr[:] = fp16_arr.astype(np.float32)
        return

    # For dropout: apply dropout using numpy
    elif "_dropout" in code:
        x_ptr, N, dropout_prob = args
        x = np.ctypeslib.as_array((ctypes.c_float * N).from_address(x_ptr))
        mask = np.random.random(N) > dropout_prob
        x[:] *= mask / (1 - dropout_prob)
        return

    # For adam_update: apply Adam update using numpy
    elif "_adam_update_" in code:
        param_ptr, m_ptr, v_ptr, grad_ptr, beta1, beta2, lr, epsilon, timestep = args
        param = np.ctypeslib.as_array((ctypes.c_float * 1).from_address(param_ptr))
        m = np.ctypeslib.as_array((ctypes.c_float * 1).from_address(m_ptr))
        v = np.ctypeslib.as_array((ctypes.c_float * 1).from_address(v_ptr))
        grad = np.ctypeslib.as_array((ctypes.c_float * 1).from_address(grad_ptr))
        
        # Update biased first moment estimate
        m[:] = beta1 * m + (1 - beta1) * grad
        # Update biased second raw moment estimate
        v[:] = beta2 * v + (1 - beta2) * (grad * grad)
        
        # Compute bias-corrected first moment estimate
        m_hat = m / (1 - beta1 ** timestep)
        # Compute bias-corrected second raw moment estimate
        v_hat = v / (1 - beta2 ** timestep)
        
        # Update parameters
        param[:] -= lr * m_hat / (np.sqrt(v_hat) + epsilon)
        return

    raise ValueError(f"Unknown kernel type in code: {code[:100]}...")