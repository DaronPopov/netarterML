import numpy as np
import ctypes
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import finlib.asm.kernels.rmsnorm as rmsnorm_kernel
import finlib.asm.kernels.softmax as softmax_kernel
import finlib.asm.kernels.matmul as matmul_kernel


def execute_kernel(kernel, *args):
    """Execute a custom kernel with the given arguments."""
    # Convert arguments to ctypes
    c_args = []
    for arg in args:
        if isinstance(arg, (np.ndarray, int, float)):
            if isinstance(arg, np.ndarray):
                # Create a copy of the array to ensure memory safety
                arr_copy = np.array(arg, dtype=np.float32, copy=True)
                c_args.append(arr_copy.ctypes.data_as(ctypes.c_void_p))
            elif isinstance(arg, int):
                c_args.append(arg)  # Pass integers directly
            elif isinstance(arg, float):
                c_args.append(arg)  # Pass floats directly
        else:
            raise ValueError(f"Unsupported argument type: {type(arg)}")
    
    try:
        # Call the kernel
        kernel(*c_args)
    except Exception as e:
        print(f"Error executing kernel: {e}")
        raise
    finally:
        # Clean up any temporary arrays
        for arg in c_args:
            if hasattr(arg, '_objects'):
                del arg._objects

def to_device(data: np.ndarray, device: str = 'cpu') -> np.ndarray:
    """Move data to the specified device (CPU or GPU)."""
    if device == 'cpu':
        return data
    else:
        raise ValueError(f"Unsupported device: {device}")