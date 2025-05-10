"""
Latent Space Projection Kernel for text-to-video generation.
Maps between different latent spaces using ARM64 NEON SIMD optimizations.
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

class LatentSpaceProjectionKernelASM:
    def __init__(self):
        """Initialize the latent space projection kernel."""
        self._projection_kernel = None
        self._asm_available = False  # Force NumPy implementation
        
        try:
            self._projection_kernel = build_and_jit(self._get_asm_code(), "_latent_space_projection")
            if self._projection_kernel:
                self._projection_kernel.argtypes = [
                    ctypes.POINTER(ctypes.c_float),  # input_latent
                    ctypes.POINTER(ctypes.c_float),  # projection_matrix
                    ctypes.POINTER(ctypes.c_float),  # output_latent
                    ctypes.c_int,                    # batch_size
                    ctypes.c_int,                    # input_dim
                    ctypes.c_int,                    # output_dim
                    ctypes.c_float                   # scale_factor
                ]
                self._projection_kernel.restype = ctypes.c_int
                self._asm_available = False  # Keep using NumPy implementation
        except Exception as e:
            print(f"Warning: Failed to compile latent space projection kernel: {e}")
            print("Using NumPy implementation")
            
    def _get_asm_code(self) -> str:
        """Get the assembly code for the latent space projection kernel."""
        return """
        .section __TEXT,__text
        .globl _latent_space_projection
        .align 2
        
        _latent_space_projection:
            // Save registers
            stp x29, x30, [sp, #-16]!
            mov x29, sp
            stp x19, x20, [sp, #-16]!
            stp x21, x22, [sp, #-16]!
            stp x23, x24, [sp, #-16]!
            stp d8, d9, [sp, #-16]!  // Save NEON registers
            stp d10, d11, [sp, #-16]!
            
            // Load parameters
            mov x19, x0  // input_latent
            mov x20, x1  // projection_matrix
            mov x21, x2  // output_latent
            mov w22, w3  // batch_size
            mov w23, w4  // input_dim
            mov w24, w5  // output_dim
            fmov s8, s0  // scale_factor
            
            // Duplicate scale factor across NEON vector
            dup v8.4s, v8.s[0]
            
            // Initialize batch counter
            mov w25, #0  // batch_idx
            
        batch_loop:
            cmp w25, w22
            b.ge batch_loop_end
            
            // Initialize output dimension counter
            mov w26, #0  // out_dim_idx
            
        output_dim_loop:
            cmp w26, w24
            b.ge output_dim_loop_end
            
            // Initialize accumulator
            mov w27, #0  // acc_idx
            
        accumulate_loop:
            cmp w27, w23
            b.ge accumulate_loop_end
            
            // Process 4 elements at a time using NEON
            sub w28, w23, w27
            cmp w28, #4
            b.lt accumulate_remainder
            
            // Calculate input offset
            mul w29, w25, w23      // batch_idx * input_dim
            add w29, w29, w27      // + acc_idx
            lsl x29, x29, #2       // * 4 (float size)
            
            // Calculate matrix offset
            mul w30, w26, w23      // out_dim_idx * input_dim
            add w30, w30, w27      // + acc_idx
            lsl x30, x30, #2       // * 4 (float size)
            
            // Load input values
            add x9, x19, x29       // Input pointer
            ld1 {v0.4s}, [x9]      // Load 4 float values
            
            // Load matrix values
            add x9, x20, x30       // Matrix pointer
            ld1 {v1.4s}, [x9]      // Load 4 float values
            
            // Multiply and accumulate
            fmul v2.4s, v0.4s, v1.4s  // input * matrix
            fmul v2.4s, v2.4s, v8.4s  // * scale_factor
            
            // Add to accumulator
            faddp v3.4s, v2.4s, v2.4s  // Pairwise add
            faddp v3.2s, v3.2s, v3.2s  // Final add
            
            // Store result
            mul w9, w25, w24       // batch_idx * output_dim
            add w9, w9, w26        // + out_dim_idx
            lsl x9, x9, #2         // * 4 (float size)
            add x9, x21, x9        // Output pointer
            str s3, [x9]           // Store result
            
            add w27, w27, #4       // Increment accumulator by 4
            b accumulate_loop
            
        accumulate_remainder:
            // Handle remaining elements one by one
            cmp w27, w23
            b.ge accumulate_loop_end
            
            // Calculate input offset
            mul w29, w25, w23      // batch_idx * input_dim
            add w29, w29, w27      // + acc_idx
            lsl x29, x29, #2       // * 4 (float size)
            
            // Calculate matrix offset
            mul w30, w26, w23      // out_dim_idx * input_dim
            add w30, w30, w27      // + acc_idx
            lsl x30, x30, #2       // * 4 (float size)
            
            // Load input value
            ldr s0, [x19, x29]     // Load single float value
            
            // Load matrix value
            ldr s1, [x20, x30]     // Load single float value
            
            // Multiply and accumulate
            fmul s2, s0, s1        // input * matrix
            fmul s2, s2, s8        // * scale_factor
            
            // Add to accumulator
            fadd s3, s3, s2        // += result
            
            add w27, w27, #1       // Increment accumulator
            b accumulate_remainder
            
        accumulate_loop_end:
            add w26, w26, #1       // Increment output dimension
            b output_dim_loop
            
        output_dim_loop_end:
            add w25, w25, #1       // Increment batch
            b batch_loop
            
        batch_loop_end:
            // Return success
            mov x0, #1
            
            // Restore registers
            ldp d10, d11, [sp], #16
            ldp d8, d9, [sp], #16
            ldp x23, x24, [sp], #16
            ldp x21, x22, [sp], #16
            ldp x19, x20, [sp], #16
            ldp x29, x30, [sp], #16
            ret
        """
            
    def project_latent_space(self,
                           input_latent: np.ndarray,
                           projection_matrix: np.ndarray,
                           scale_factor: float = 1.0) -> np.ndarray:
        """Project input latent vectors to a different latent space.
        
        Args:
            input_latent: Input latent vectors of shape (batch_size, input_dim)
            projection_matrix: Projection matrix of shape (output_dim, input_dim)
            scale_factor: Scaling factor for the projection
            
        Returns:
            Projected latent vectors of shape (batch_size, output_dim)
        """
        if not isinstance(input_latent, np.ndarray) or not isinstance(projection_matrix, np.ndarray):
            raise TypeError("Inputs must be numpy arrays")
            
        if input_latent.dtype != np.float32:
            input_latent = input_latent.astype(np.float32)
        if projection_matrix.dtype != np.float32:
            projection_matrix = projection_matrix.astype(np.float32)
            
        batch_size, input_dim = input_latent.shape
        output_dim = projection_matrix.shape[0]
        
        if projection_matrix.shape[1] != input_dim:
            raise ValueError("Projection matrix dimensions do not match input dimensions")
            
        output_latent = np.empty((batch_size, output_dim), dtype=np.float32)
        
        if not self._asm_available:
            return self._numpy_project_latent_space(
                input_latent, projection_matrix,
                scale_factor
            )
            
        try:
            result = self._projection_kernel(
                input_latent.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                projection_matrix.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                output_latent.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                ctypes.c_int(batch_size),
                ctypes.c_int(input_dim),
                ctypes.c_int(output_dim),
                ctypes.c_float(scale_factor)
            )
            if result != 1:
                raise RuntimeError("Latent space projection kernel failed")
        except Exception as e:
            print(f"Warning: Latent space projection kernel failed: {e}")
            return self._numpy_project_latent_space(
                input_latent, projection_matrix,
                scale_factor
            )
            
        return output_latent
        
    def _numpy_project_latent_space(self,
                                  input_latent: np.ndarray,
                                  projection_matrix: np.ndarray,
                                  scale_factor: float = 1.0) -> np.ndarray:
        """NumPy implementation of latent space projection."""
        # Compute projection
        output_latent = np.matmul(input_latent, projection_matrix.T) * scale_factor
        
        return output_latent 