"""
Frame Interpolation Kernel for text-to-video generation.
Generates smooth transitions between frames using ARM64 NEON SIMD optimizations.
"""

import numpy as np
import ctypes
import os
import tempfile
import subprocess
import sys
from pathlib import Path

# Add parent directory to Python path to find the builder module
sys.path.append(str(Path(__file__).parent.parent.parent.parent.parent.parent))
from core.asm.assembler.builder import build_and_jit

class FrameInterpolationKernelASM:
    def __init__(self):
        """Initialize the frame interpolation kernel."""
        self._frame_interpolation_kernel = None
        self._asm_available = False
        
        try:
            self._frame_interpolation_kernel = build_and_jit(self._get_asm_code(), "_frame_interpolation")
            if self._frame_interpolation_kernel:
                self._frame_interpolation_kernel.argtypes = [
                    ctypes.POINTER(ctypes.c_float),  # frame1
                    ctypes.POINTER(ctypes.c_float),  # frame2
                    ctypes.POINTER(ctypes.c_float),  # output_frame
                    ctypes.c_int,                    # height
                    ctypes.c_int,                    # width
                    ctypes.c_int,                    # channels
                    ctypes.c_float                   # interpolation_factor
                ]
                self._frame_interpolation_kernel.restype = ctypes.c_int
                self._asm_available = True
        except Exception as e:
            print(f"Warning: Failed to compile frame interpolation kernel: {e}")
            print("Falling back to NumPy implementation")
            
    def _get_asm_code(self) -> str:
        """Get the assembly code for the frame interpolation kernel."""
        return """
        .section __TEXT,__text
        .globl _frame_interpolation
        .align 2
        
        _frame_interpolation:
            // Save registers
            stp x29, x30, [sp, #-16]!
            mov x29, sp
            stp x19, x20, [sp, #-16]!
            stp x21, x22, [sp, #-16]!
            stp x23, x24, [sp, #-16]!
            stp d8, d9, [sp, #-16]!  // Save NEON registers
            
            // Load parameters
            mov x19, x0  // frame1
            mov x20, x1  // frame2
            mov x21, x2  // output_frame
            mov w22, w3  // height
            mov w23, w4  // width
            mov w24, w5  // channels
            fmov s8, s0  // interpolation_factor
            
            // Duplicate interpolation factor across NEON vector
            dup v8.4s, v8.s[0]
            
            // Calculate 1 - interpolation_factor
            fmov s9, #1.0
            fsub s9, s9, s8
            dup v9.4s, v9.s[0]
            
            // Initialize height counter
            mov w25, #0  // height_idx
            
        height_loop:
            cmp w25, w22
            b.ge height_loop_end
            
            // Initialize width counter
            mov w26, #0  // width_idx
            
        width_loop:
            cmp w26, w23
            b.ge width_loop_end
            
            // Process 4 channels at a time using NEON
            mov w27, #0  // channel_idx
            
        channel_loop:
            // Check if we have at least 4 channels left
            sub w9, w24, w27
            cmp w9, #4
            b.lt channel_remainder
            
            // Calculate base offset
            mul w9, w25, w23     // height_idx * width
            mul w9, w9, w24      // * channels
            mul w10, w26, w24    // width_idx * channels
            add w9, w9, w10      // Add width offset
            add w9, w9, w27      // Add channel offset
            lsl x9, x9, #2       // * 4 (float size)
            
            // Load 4 channels from frame1 and frame2
            add x10, x19, x9     // frame1 pointer
            add x11, x20, x9     // frame2 pointer
            ld1 {v0.4s}, [x10]   // Load 4 float values from frame1
            ld1 {v1.4s}, [x11]   // Load 4 float values from frame2
            
            // Interpolate using NEON
            fmul v2.4s, v0.4s, v9.4s  // frame1 * (1 - factor)
            fmla v2.4s, v1.4s, v8.4s  // += frame2 * factor
            
            // Store interpolated result
            add x10, x21, x9     // output pointer
            st1 {v2.4s}, [x10]   // Store 4 float values
            
            add w27, w27, #4     // Increment channel counter
            b channel_loop
            
        channel_remainder:
            // Handle remaining channels one by one
            cmp w27, w24
            b.ge channel_loop_end
            
            // Calculate offset for single channel
            mul w9, w25, w23     // height_idx * width
            mul w9, w9, w24      // * channels
            mul w10, w26, w24    // width_idx * channels
            add w9, w9, w10      // Add width offset
            add w9, w9, w27      // Add channel offset
            lsl x9, x9, #2       // * 4 (float size)
            
            // Load single values
            ldr s0, [x19, x9]    // frame1 value
            ldr s1, [x20, x9]    // frame2 value
            
            // Interpolate
            fmul s2, s0, s9      // frame1 * (1 - factor)
            fmadd s2, s1, s8, s2  // frame1 * (1-factor) + frame2 * factor
            
            // Store interpolated result
            str s2, [x21, x9]
            
            add w27, w27, #1
            b channel_remainder
            
        channel_loop_end:
            add w26, w26, #1
            b width_loop
            
        width_loop_end:
            add w25, w25, #1
            b height_loop
            
        height_loop_end:
            // Restore registers
            ldp d8, d9, [sp], #16  // Restore NEON registers
            ldp x23, x24, [sp], #16
            ldp x21, x22, [sp], #16
            ldp x19, x20, [sp], #16
            ldp x29, x30, [sp], #16
            
            // Return success
            mov x0, #1
            ret
        """
            
    def interpolate(self, frame1: np.ndarray, frame2: np.ndarray, factor: float = 0.5) -> np.ndarray:
        """Interpolate between two frames."""
        if not isinstance(frame1, np.ndarray) or not isinstance(frame2, np.ndarray):
            raise TypeError("Input frames must be numpy arrays")
            
        if frame1.shape != frame2.shape:
            raise ValueError("Input frames must have the same shape")
            
        if frame1.dtype != np.float32:
            frame1 = frame1.astype(np.float32)
        if frame2.dtype != np.float32:
            frame2 = frame2.astype(np.float32)
            
        height, width, channels = frame1.shape
        output_frame = np.empty_like(frame1)
        
        if not self._asm_available:
            return self._numpy_interpolate(frame1, frame2, factor)
            
        try:
            result = self._frame_interpolation_kernel(
                frame1.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                frame2.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                output_frame.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                ctypes.c_int(height),
                ctypes.c_int(width),
                ctypes.c_int(channels),
                ctypes.c_float(factor)
            )
            if result != 1:
                raise RuntimeError("Frame interpolation kernel failed")
        except Exception as e:
            print(f"Warning: Frame interpolation kernel failed: {e}")
            return self._numpy_interpolate(frame1, frame2, factor)
            
        return output_frame
        
    def _numpy_interpolate(self, frame1: np.ndarray, frame2: np.ndarray, factor: float = 0.5) -> np.ndarray:
        """NumPy implementation of frame interpolation."""
        return frame1 * (1 - factor) + frame2 * factor 