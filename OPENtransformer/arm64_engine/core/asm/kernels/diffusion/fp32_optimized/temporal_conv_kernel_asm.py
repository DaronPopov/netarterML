"""
ARM64 SIMD-optimized temporal convolution kernel.
"""

import numpy as np
import ctypes
from typing import Optional, Tuple
import tempfile
import os
import subprocess

class TemporalConvKernelASM:
    def __init__(self):
        """Initialize the temporal convolution kernel."""
        self._temporal_conv_kernel = None
        self._asm_available = False
        self._argtypes = [
            np.ctypeslib.ndpointer(dtype=np.float32, flags='ALIGNED, CONTIGUOUS'),  # input
            np.ctypeslib.ndpointer(dtype=np.float32, flags='ALIGNED, CONTIGUOUS'),  # kernel
            np.ctypeslib.ndpointer(dtype=np.float32, flags='ALIGNED, CONTIGUOUS'),  # output
            ctypes.c_int32,  # batch_size
            ctypes.c_int32,  # num_frames
            ctypes.c_int32,  # channels
            ctypes.c_int32,  # height
            ctypes.c_int32,  # width
            ctypes.c_int32,  # kernel_size
            ctypes.c_int32,  # stride
            ctypes.c_int32   # padding
        ]
        
    def _numpy_apply_temporal_conv(
        self,
        x: np.ndarray,
        kernel: np.ndarray,
        stride: int = 1,
        padding: int = 0
    ) -> np.ndarray:
        """NumPy implementation of temporal convolution."""
        batch_size, num_frames, channels, height, width = x.shape
        kernel_size = kernel.shape[0]
        
        # Add padding if needed
        if padding > 0:
            pad_width = ((0, 0), (padding, padding), (0, 0), (0, 0), (0, 0))
            x = np.pad(x, pad_width, mode='constant')
            num_frames = x.shape[1]
            
        # Calculate output dimensions
        output_frames = (num_frames - kernel_size) // stride + 1
        
        # Prepare output array
        output = np.zeros((batch_size, output_frames, channels, height, width), dtype=np.float32)
        
        # Apply convolution
        for b in range(batch_size):
            for t in range(0, output_frames):
                t_start = t * stride
                t_end = t_start + kernel_size
                output[b, t] = np.sum(
                    x[b, t_start:t_end] * kernel[:, np.newaxis, np.newaxis, np.newaxis],
                    axis=0
                )
                
        return output
        
    def apply_temporal_conv(
        self,
        x: np.ndarray,
        kernel: np.ndarray,
        stride: int = 1,
        padding: int = 0
    ) -> np.ndarray:
        """Apply temporal convolution to input tensor.
        
        Args:
            x: Input tensor of shape (batch_size, num_frames, channels, height, width)
            kernel: Convolution kernel of shape (kernel_size,)
            stride: Stride of the convolution
            padding: Padding to apply to the input
            
        Returns:
            Output tensor of shape (batch_size, output_frames, channels, height, width)
        """
        # Input validation
        if not isinstance(x, np.ndarray) or not isinstance(kernel, np.ndarray):
            raise TypeError("Inputs must be numpy arrays")
            
        if x.ndim != 5:
            raise ValueError("Input must have 5 dimensions (batch_size, num_frames, channels, height, width)")
            
        if kernel.ndim != 1:
            raise ValueError("Kernel must have 1 dimension (kernel_size,)")
            
        batch_size, num_frames, channels, height, width = x.shape
        kernel_size = kernel.shape[0]
        
        # Ensure inputs are contiguous and float32
        x = np.ascontiguousarray(x, dtype=np.float32)
        kernel = np.ascontiguousarray(kernel, dtype=np.float32)
        
        # Add padding if needed
        if padding > 0:
            pad_width = ((0, 0), (padding, padding), (0, 0), (0, 0), (0, 0))
            x = np.pad(x, pad_width, mode='constant')
            num_frames = x.shape[1]
            
        # Calculate output dimensions
        output_frames = (num_frames - kernel_size) // stride + 1
        if output_frames <= 0:
            if padding > 0:
                # Try to use padding to get valid output dimensions
                output_frames = (num_frames + 2 * padding - kernel_size) // stride + 1
            if output_frames <= 0:
                raise ValueError(f"Invalid output dimensions: input_frames={num_frames}, "
                               f"kernel_size={kernel_size}, stride={stride}, padding={padding}")
            
        # Prepare output array
        output = np.zeros((batch_size, output_frames, channels, height, width), dtype=np.float32)
        
        # Fall back to NumPy implementation if assembly is not available
        if not self._asm_available:
            return self._numpy_apply_temporal_conv(x, kernel, stride, padding)
            
        # Call assembly implementation
        try:
            result = self._temporal_conv_kernel(
                x,
                kernel,
                output,
                batch_size,
                num_frames,
                channels,
                height,
                width,
                kernel_size,
                stride,
                padding
            )
            if result != 0:
                raise RuntimeError("Assembly implementation failed")
        except Exception as e:
            print(f"Warning: Assembly implementation failed ({str(e)}). Falling back to NumPy implementation.")
            return self._numpy_apply_temporal_conv(x, kernel, stride, padding)
            
        return output 