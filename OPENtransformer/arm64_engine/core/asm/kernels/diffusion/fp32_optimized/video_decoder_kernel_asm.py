"""
Video Decoder Kernel for text-to-video generation.
Decodes latent video representation to pixel space using ARM64 NEON SIMD optimizations.
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

class VideoDecoderKernelASM:
    def __init__(self):
        """Initialize the video decoder kernel."""
        self._video_decoder_kernel = None
        self._asm_available = False
        
        try:
            self._video_decoder_kernel = build_and_jit(self._get_asm_code(), "_video_decoder")
            if self._video_decoder_kernel:
                self._video_decoder_kernel.argtypes = [
                    ctypes.POINTER(ctypes.c_float),  # latent_video
                    ctypes.POINTER(ctypes.c_float),  # output_video
                    ctypes.c_int,                    # batch_size
                    ctypes.c_int,                    # num_frames
                    ctypes.c_int,                    # height
                    ctypes.c_int,                    # width
                    ctypes.c_int,                    # channels
                    ctypes.c_float                   # scale_factor
                ]
                self._video_decoder_kernel.restype = ctypes.c_int
                self._asm_available = True
        except Exception as e:
            print(f"Warning: Failed to compile video decoder kernel: {e}")
            print("Falling back to NumPy implementation")
            
    def _get_asm_code(self) -> str:
        """Get the assembly code for the video decoder kernel."""
        return """
        .section __TEXT,__text
        .globl _video_decoder
        .align 2
        
        _video_decoder:
            // Save registers
            stp x29, x30, [sp, #-16]!
            mov x29, sp
            stp x19, x20, [sp, #-16]!
            stp x21, x22, [sp, #-16]!
            stp x23, x24, [sp, #-16]!
            stp d8, d9, [sp, #-16]!  // Save NEON registers
            
            // Load parameters
            mov x19, x0  // latent_video
            mov x20, x1  // output_video
            mov w21, w2  // batch_size
            mov w22, w3  // num_frames
            mov w23, w4  // height
            mov w24, w5  // width
            mov w25, w6  // channels
            fmov s8, s0  // scale_factor
            
            // Duplicate scale factor across NEON vector
            dup v8.4s, v8.s[0]
            
            // Initialize batch counter
            mov w26, #0  // batch_idx
            
        batch_loop:
            cmp w26, w21
            b.ge batch_loop_end
            
            // Initialize frame counter
            mov w27, #0  // frame_idx
            
        frame_loop:
            cmp w27, w22
            b.ge frame_loop_end
            
            // Initialize height counter
            mov w28, #0  // height_idx
            
        height_loop:
            cmp w28, w23
            b.ge height_loop_end
            
            // Initialize width counter
            mov w29, #0  // width_idx
            
        width_loop:
            cmp w29, w24
            b.ge width_loop_end
            
            // Process 4 channels at a time using NEON
            mov w30, #0  // channel_idx
            
        channel_loop:
            // Check if we have at least 4 channels left
            sub w9, w25, w30
            cmp w9, #4
            b.lt channel_remainder
            
            // Calculate base offset
            mul w9, w26, w22      // batch_idx * num_frames
            mul w9, w9, w23       // * height
            mul w9, w9, w24       // * width
            mul w9, w9, w25       // * channels
            
            // Add frame offset
            mul w10, w27, w23     // frame_idx * height
            mul w10, w10, w24     // * width
            mul w10, w10, w25     // * channels
            add w9, w9, w10       // Add to base offset
            
            // Add height offset
            mul w10, w28, w24     // height_idx * width
            mul w10, w10, w25     // * channels
            add w9, w9, w10       // Add to base offset
            
            // Add width and channel offset
            mul w10, w29, w25     // width_idx * channels
            add w10, w10, w30     // + channel_idx
            add w9, w9, w10       // Add to base offset
            
            // Convert to byte offset
            lsl x9, x9, #2        // * 4 (float size)
            
            // Load 4 channels at once using NEON
            add x10, x19, x9      // Input pointer
            ld1 {v0.4s}, [x10]    // Load 4 float values
            
            // Apply scale factor using NEON
            fmul v1.4s, v0.4s, v8.4s
            
            // Store 4 channels at once using NEON
            add x10, x20, x9      // Output pointer
            st1 {v1.4s}, [x10]    // Store 4 float values
            
            add w30, w30, #4      // Increment channel counter by 4
            b channel_loop
            
        channel_remainder:
            // Handle remaining channels one by one
            cmp w30, w25
            b.ge channel_loop_end
            
            // Calculate offset for single channel
            mul w9, w26, w22      // batch_idx * num_frames
            mul w9, w9, w23       // * height
            mul w9, w9, w24       // * width
            mul w9, w9, w25       // * channels
            
            // Add frame offset
            mul w10, w27, w23     // frame_idx * height
            mul w10, w10, w24     // * width
            mul w10, w10, w25     // * channels
            add w9, w9, w10       // Add to base offset
            
            // Add height offset
            mul w10, w28, w24     // height_idx * width
            mul w10, w10, w25     // * channels
            add w9, w9, w10       // Add to base offset
            
            // Add width and channel offset
            mul w10, w29, w25     // width_idx * channels
            add w10, w10, w30     // + channel_idx
            add w9, w9, w10       // Add to base offset
            
            // Convert to byte offset
            lsl x9, x9, #2        // * 4 (float size)
            
            // Load single value
            ldr s0, [x19, x9]
            
            // Apply scale factor
            fmul s0, s0, s8
            
            // Store single value
            str s0, [x20, x9]
            
            add w30, w30, #1
            b channel_remainder
            
        channel_loop_end:
            add w29, w29, #1
            b width_loop
            
        width_loop_end:
            add w28, w28, #1
            b height_loop
            
        height_loop_end:
            add w27, w27, #1
            b frame_loop
            
        frame_loop_end:
            add w26, w26, #1
            b batch_loop
            
        batch_loop_end:
            // Return success
            mov x0, #1
            
            // Restore registers
            ldp d8, d9, [sp], #16  // Restore NEON registers
            ldp x23, x24, [sp], #16
            ldp x21, x22, [sp], #16
            ldp x19, x20, [sp], #16
            ldp x29, x30, [sp], #16
            ret
        """
            
    def decode_video(self, latent_video: np.ndarray, scale_factor: float = 1.0) -> np.ndarray:
        """Decode latent video representation to pixel space."""
        if not isinstance(latent_video, np.ndarray):
            raise TypeError("Input must be a numpy array")
            
        if latent_video.dtype != np.float32:
            latent_video = latent_video.astype(np.float32)
            
        batch_size, num_frames, height, width, channels = latent_video.shape
        output_video = np.empty_like(latent_video)
        
        if not self._asm_available:
            return self._numpy_decode_video(latent_video, scale_factor)
            
        try:
            result = self._video_decoder_kernel(
                latent_video.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                output_video.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                ctypes.c_int(batch_size),
                ctypes.c_int(num_frames),
                ctypes.c_int(height),
                ctypes.c_int(width),
                ctypes.c_int(channels),
                ctypes.c_float(scale_factor)
            )
            if result != 1:
                raise RuntimeError("Video decoder kernel failed")
        except Exception as e:
            print(f"Warning: Video decoder kernel failed: {e}")
            return self._numpy_decode_video(latent_video, scale_factor)
            
        # Clip values to valid range
        np.clip(output_video, 0.0, 1.0, out=output_video)
        return output_video
        
    def _numpy_decode_video(self, latent_video: np.ndarray, scale_factor: float = 1.0) -> np.ndarray:
        """NumPy implementation of video decoding."""
        # Apply scale factor
        output_video = latent_video * scale_factor
        
        # Clip values to valid range
        output_video = np.clip(output_video, 0.0, 1.0)
        
        return output_video 