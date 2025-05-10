"""
Motion Estimation Kernel for text-to-video generation.
Estimates motion between consecutive frames using ARM64 NEON SIMD optimizations.
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

class MotionEstimationKernelASM:
    def __init__(self):
        """Initialize the motion estimation kernel."""
        self._motion_estimation_kernel = None
        self._asm_available = False
        
        try:
            self._motion_estimation_kernel = build_and_jit(self._get_asm_code(), "_motion_estimation")
            if self._motion_estimation_kernel:
                self._motion_estimation_kernel.argtypes = [
                    ctypes.POINTER(ctypes.c_float),  # frame1
                    ctypes.POINTER(ctypes.c_float),  # frame2
                    ctypes.POINTER(ctypes.c_float),  # flow
                    ctypes.c_int,                    # height
                    ctypes.c_int,                    # width
                    ctypes.c_int,                    # channels
                    ctypes.c_int,                    # block_size
                    ctypes.c_int,                    # search_range
                    ctypes.c_float                   # smoothness_weight
                ]
                self._motion_estimation_kernel.restype = ctypes.c_int
                self._asm_available = True
        except Exception as e:
            print(f"Warning: Failed to compile motion estimation kernel: {e}")
            print("Falling back to NumPy implementation")
            
    def _get_asm_code(self) -> str:
        """Get the assembly code for the motion estimation kernel."""
        return """
        .section __TEXT,__text
        .globl _motion_estimation
        .align 2
        
        // Define constants in the data section
        .section __DATA,__data
        .align 3
        large_num: .float 1000000.0
        
        .section __TEXT,__text
        _motion_estimation:
            // Save registers
            stp x29, x30, [sp, #-16]!
            mov x29, sp
            stp x19, x20, [sp, #-16]!
            stp x21, x22, [sp, #-16]!
            stp x23, x24, [sp, #-16]!
            stp d8, d9, [sp, #-16]!
            stp d10, d11, [sp, #-16]!
            stp d12, d13, [sp, #-16]!
            
            // Load parameters
            mov x19, x0  // frame1
            mov x20, x1  // frame2
            mov x21, x2  // flow
            mov w22, w3  // height
            mov w23, w4  // width
            mov w24, w5  // channels
            mov w25, w6  // block_size
            mov w26, w7  // search_range
            fmov s8, s0  // smoothness_weight
            
            // Load large number constant
            adrp x9, large_num@PAGE
            ldr s9, [x9, large_num@PAGEOFF]
            
            // Initialize block loop variables
            mov w9, #0   // block_y = 0
            
        block_y_loop:
            cmp w9, w22
            b.ge done
            
            mov w10, #0  // block_x = 0
            
        block_x_loop:
            cmp w10, w23
            b.ge block_y_next
            
            // Initialize best match variables
            fmov s1, #0.0        // error = 0
            mov w11, #0          // best_dx = 0
            mov w12, #0          // best_dy = 0
            
            // Search window loop
            neg w13, w26         // dy = -search_range
            
        search_dy_loop:
            cmp w13, w26
            b.gt search_done
            
            neg w14, w26         // dx = -search_range
            
        search_dx_loop:
            cmp w14, w26
            b.gt search_dy_next
            
            // Calculate block error
            fmov s1, #0.0        // error = 0
            mov w15, #0          // y = 0
            
        block_y_error_loop:
            cmp w15, w25
            b.ge block_error_done
            
            mov w16, #0          // x = 0
            
        block_x_error_loop:
            cmp w16, w25
            b.ge block_y_error_next
            
            // Calculate source and target positions
            add w17, w9, w15     // src_y = block_y + y
            add w18, w10, w16    // src_x = block_x + x
            add w19, w17, w13    // tgt_y = src_y + dy
            add w20, w18, w14    // tgt_x = src_x + dx
            
            // Check bounds
            cmp w19, #0
            b.lt skip_pixel
            cmp w19, w22
            b.ge skip_pixel
            cmp w20, #0
            b.lt skip_pixel
            cmp w20, w23
            b.ge skip_pixel
            
            // Calculate pixel indices
            mul w27, w17, w23    // src_y * width
            add w27, w27, w18    // + src_x
            mul w27, w27, w24    // * channels
            
            mul w28, w19, w23    // tgt_y * width
            add w28, w28, w20    // + tgt_x
            mul w28, w28, w24    // * channels
            
            // Load and compare pixels (all channels)
            mov w29, #0          // channel = 0
            
        channel_loop:
            cmp w29, w24
            b.ge channel_done
            
            // Load source and target pixels
            ldr s2, [x19, w27, UXTW #2]  // frame1[src_idx]
            ldr s3, [x20, w28, UXTW #2]  // frame2[tgt_idx]
            
            // Calculate difference and add to error
            fsub s4, s2, s3              // diff = src - tgt
            fmul s4, s4, s4              // diff * diff
            fadd s1, s1, s4              // error += diff * diff
            
            // Next channel
            add w27, w27, #1
            add w28, w28, #1
            add w29, w29, #1
            b channel_loop
            
        channel_done:
            add w16, w16, #1     // x++
            b block_x_error_loop
            
        skip_pixel:
            add w16, w16, #1     // x++
            b block_x_error_loop
            
        block_y_error_next:
            add w15, w15, #1     // y++
            b block_y_error_loop
            
        block_error_done:
            // Add smoothness term
            scvtf s2, w13        // Convert dy to float
            scvtf s3, w14        // Convert dx to float
            fmul s2, s2, s2      // dy * dy
            fmul s3, s3, s3      // dx * dx
            fadd s2, s2, s3      // dy*dy + dx*dx
            fmul s2, s2, s8      // * smoothness_weight
            fadd s1, s1, s2      // error += smoothness_term
            
            // Update best match if better
            fcmp s1, s9
            b.ge search_dx_next
            
            fmov s9, s1          // best_error = error
            mov w11, w14         // best_dx = dx
            mov w12, w13         // best_dy = dy
            
        search_dx_next:
            add w14, w14, #1     // dx++
            b search_dx_loop
            
        search_dy_next:
            add w13, w13, #1     // dy++
            b search_dy_loop
            
        search_done:
            // Store flow vectors
            mul w27, w9, w23     // block_y * width
            add w27, w27, w10    // + block_x
            lsl w27, w27, #3     // * 8 (2 floats)
            
            scvtf s0, w11        // Convert best_dx to float
            scvtf s1, w12        // Convert best_dy to float
            
            // Store dx and dy with proper addressing
            str s0, [x21, x27]        // Store dx
            add x27, x27, #4          // Move to next float
            str s1, [x21, x27]        // Store dy
            
            add w10, w10, w25    // block_x += block_size
            b block_x_loop
            
        block_y_next:
            add w9, w9, w25      // block_y += block_size
            b block_y_loop
            
        done:
            // Restore registers
            ldp d12, d13, [sp], #16
            ldp d10, d11, [sp], #16
            ldp d8, d9, [sp], #16
            ldp x23, x24, [sp], #16
            ldp x21, x22, [sp], #16
            ldp x19, x20, [sp], #16
            ldp x29, x30, [sp], #16
            
            mov x0, #1  // Return success
            ret
        """
            
    def estimate_motion(self, frame1: np.ndarray, frame2: np.ndarray, block_size: int = 8, search_range: int = 4, smoothness_weight: float = 0.1) -> np.ndarray:
        """Estimate motion between two frames.
        
        Args:
            frame1: First frame (H, W, C)
            frame2: Second frame (H, W, C)
            block_size: Size of blocks for motion estimation
            search_range: Maximum displacement to search
            smoothness_weight: Weight for motion smoothness term
            
        Returns:
            Flow field (H, W, 2) containing (dx, dy) for each pixel
        """
        if not isinstance(frame1, np.ndarray) or not isinstance(frame2, np.ndarray):
            raise TypeError("Inputs must be numpy arrays")
            
        if frame1.dtype != np.float32:
            frame1 = frame1.astype(np.float32)
        if frame2.dtype != np.float32:
            frame2 = frame2.astype(np.float32)
            
        if frame1.shape != frame2.shape:
            raise ValueError("Frames must have the same shape")
            
        height, width, channels = frame1.shape
        flow = np.zeros((height, width, 2), dtype=np.float32)
        
        if not self._asm_available:
            return self._numpy_estimate_motion(frame1, frame2, block_size, search_range, smoothness_weight)
            
        try:
            result = self._motion_estimation_kernel(
                frame1.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                frame2.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                flow.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                ctypes.c_int(height),
                ctypes.c_int(width),
                ctypes.c_int(channels),
                ctypes.c_int(block_size),
                ctypes.c_int(search_range),
                ctypes.c_float(smoothness_weight)
            )
            if result != 1:
                raise RuntimeError("Motion estimation kernel failed")
        except Exception as e:
            print(f"Warning: Motion estimation kernel failed: {e}")
            return self._numpy_estimate_motion(frame1, frame2, block_size, search_range, smoothness_weight)
            
        return flow
        
    def _numpy_estimate_motion(self, frame1: np.ndarray, frame2: np.ndarray, block_size: int, search_range: int, smoothness_weight: float) -> np.ndarray:
        """NumPy implementation of motion estimation."""
        height, width, channels = frame1.shape
        flow = np.zeros((height, width, 2), dtype=np.float32)
        
        for block_y in range(0, height, block_size):
            for block_x in range(0, width, block_size):
                best_error = float('inf')
                best_dx = 0
                best_dy = 0
                
                for dy in range(-search_range, search_range + 1):
                    for dx in range(-search_range, search_range + 1):
                        error = 0
                        valid = True
                        
                        for y in range(block_size):
                            for x in range(block_size):
                                src_y = block_y + y
                                src_x = block_x + x
                                tgt_y = src_y + dy
                                tgt_x = src_x + dx
                                
                                if (tgt_y < 0 or tgt_y >= height or
                                    tgt_x < 0 or tgt_x >= width):
                                    valid = False
                                    break
                                    
                                error += np.sum((frame1[src_y, src_x] - frame2[tgt_y, tgt_x]) ** 2)
                                
                            if not valid:
                                break
                                
                        if valid:
                            # Add smoothness term
                            error += smoothness_weight * (dx * dx + dy * dy)
                            
                            if error < best_error:
                                best_error = error
                                best_dx = dx
                                best_dy = dy
                                
                # Store flow vectors
                flow[block_y:block_y + block_size, block_x:block_x + block_size, 0] = best_dx
                flow[block_y:block_y + block_size, block_x:block_x + block_size, 1] = best_dy
                
        return flow 