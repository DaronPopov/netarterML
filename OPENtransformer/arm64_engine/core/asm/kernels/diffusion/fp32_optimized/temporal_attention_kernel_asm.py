"""
ARM64 SIMD-optimized implementation of temporal attention for video diffusion.
"""

import numpy as np
import ctypes
from typing import Optional, Tuple
from OPENtransformer.core.asm.assembler.builder import build_and_jit

class TemporalAttentionKernelASM:
    """
    ARM64 SIMD-optimized implementation of temporal attention for video diffusion.
    
    This kernel processes temporal attention across video frames using NEON SIMD
    instructions for optimal performance on ARM64 architecture.
    """
    
    def __init__(self):
        """Initialize the temporal attention kernel."""
        self._asm_available = False
        try:
            self._temporal_attention_kernel = self._compile_temporal_attention()
            # Set function argument types
            self._temporal_attention_kernel.argtypes = [
                ctypes.POINTER(ctypes.c_float),  # frames
                ctypes.POINTER(ctypes.c_float),  # attention_weights
                ctypes.POINTER(ctypes.c_float),  # output
                ctypes.c_int,                    # batch_size
                ctypes.c_int,                    # num_frames
                ctypes.c_int,                    # channels
                ctypes.c_int,                    # height
                ctypes.c_int                     # width
            ]
            self._temporal_attention_kernel.restype = None
            self._asm_available = True
        except Exception as e:
            print(f"Warning: Failed to compile assembly kernels: {e}")
            print("Falling back to NumPy implementation")
    
    def _compile_temporal_attention(self):
        """Compile the temporal attention kernel."""
        asm = r"""
        .text
        .global _temporal_attention_asm
        _temporal_attention_asm:
            // x0=frames, x1=attention_weights, x2=output, x3=batch_size, x4=num_frames, x5=channels, x6=height, x7=width
            
            // Input validation
            cbz x0, err
            cbz x1, err
            cbz x2, err
            cbz x3, err
            cbz x4, err
            cbz x5, err
            cbz x6, err
            cbz x7, err
            
            // Save all callee-saved registers we'll use
            stp x29, x30, [sp, #-16]!
            stp x19, x20, [sp, #-16]!
            stp x21, x22, [sp, #-16]!
            stp x23, x24, [sp, #-16]!
            stp x25, x26, [sp, #-16]!
            stp x27, x28, [sp, #-16]!
            mov x29, sp
            
            // Save input parameters
            mov x19, x0     // frames
            mov x20, x1     // attention_weights
            mov x21, x2     // output
            mov x22, x3     // batch_size
            mov x23, x4     // num_frames
            mov x24, x5     // channels
            mov x25, x6     // height
            mov x26, x7     // width
            
            // Calculate strides (in bytes)
            mul x27, x25, x26       // height * width
            lsl x27, x27, #2        // * sizeof(float) = channel_stride
            mul x28, x27, x24       // channel_stride * channels = frame_stride
            
            // Initialize output to zeros using NEON
            mov x0, x21             // output pointer
            movi v31.4s, #0         // zero vector
            mov x1, x28             // total size per batch
            mul x1, x1, x22         // total size for all batches
            lsr x1, x1, #4          // divide by 16 (vector size)
            
        init_loop:
            cbz x1, init_done
            subs x1, x1, #1
            str q31, [x0], #16      // Store zeros
            b init_loop
            
        init_done:
            // Process each batch
            mov x22, x3             // restore batch_size
        batch_loop:
            cbz x22, done_success
            subs x22, x22, #1
            
            // Save batch pointers
            mov x0, x19    // current frames pointer
            mov x1, x20    // current weights pointer
            mov x2, x21    // current output pointer
            
            // Process each frame
            mov x4, x23    // reset num_frames counter
        frame_loop:
            cbz x4, next_batch
            subs x4, x4, #1
            
            // Load attention weight
            ldr s0, [x1], #4
            dup v0.4s, v0.s[0]      // Broadcast weight to vector
            
            // Process each channel
            mov x5, x24    // reset channels counter
            mov x10, x0    // save frame start pointer
            mov x11, x2    // save output start pointer
            
        channel_loop:
            cbz x5, frame_done
            subs x5, x5, #1
            
            // Process each row
            mov x6, x25    // reset height counter
            mov x12, x10   // current frame channel pointer
            mov x13, x11   // current output channel pointer
            
        height_loop:
            cbz x6, next_channel
            subs x6, x6, #1
            
            // Process pixels in groups of 4
            mov x7, x26    // reset width counter
            lsr x8, x7, #2 // divide by 4 for vector processing
            
            // Save row pointers
            mov x14, x12   // current frame row pointer
            mov x15, x13   // current output row pointer
            
        width_loop:
            cbz x8, width_remainder
            subs x8, x8, #1
            
            // Load 4 input pixels and multiply-add with weight
            ldr q1, [x14], #16      // Load input
            ldr q2, [x15]           // Load current output
            fmul v1.4s, v1.4s, v0.4s // Multiply by weight
            fadd v2.4s, v2.4s, v1.4s // Add to accumulator
            str q2, [x15], #16      // Store result
            
            b width_loop
            
        width_remainder:
            and x8, x7, #3  // remaining width % 4
            cbz x8, next_row
            
        remainder_loop:
            subs x8, x8, #1
            
            // Process one pixel at a time
            ldr s1, [x14], #4       // Load input
            ldr s2, [x15]           // Load current output
            fmul s1, s1, s0         // Multiply by weight
            fadd s2, s2, s1         // Add to accumulator
            str s2, [x15], #4       // Store result
            
            cbnz x8, remainder_loop
            
        next_row:
            // Move to next row
            add x12, x12, x26, lsl #2  // frame_ptr += width * sizeof(float)
            add x13, x13, x26, lsl #2  // output_ptr += width * sizeof(float)
            b height_loop
            
        next_channel:
            // Move to next channel
            add x10, x10, x27  // frame_ptr += channel_stride
            add x11, x11, x27  // output_ptr += channel_stride
            b channel_loop
            
        frame_done:
            // Move to next frame
            add x0, x0, x28   // frames += frame_stride
            mov x2, x21       // Reset output pointer to start of batch
            b frame_loop
            
        next_batch:
            // Update batch pointers
            mul x9, x28, x23    // frame_stride * num_frames
            add x19, x19, x9    // frames += frame_stride * num_frames (for entire batch)
            add x20, x20, x23, lsl #2  // weights += num_frames * sizeof(float)
            add x21, x21, x28   // output += frame_stride
            b batch_loop
            
        done_success:
            // Restore all saved registers
            ldp x27, x28, [sp], #16
            ldp x25, x26, [sp], #16
            ldp x23, x24, [sp], #16
            ldp x21, x22, [sp], #16
            ldp x19, x20, [sp], #16
            ldp x29, x30, [sp], #16
            mov x0, #0
            ret
            
        err:
            mov x0, #-1
            ret
        """
        return build_and_jit(asm, '_temporal_attention_asm')
    
    def apply_temporal_attention(self,
                               frames: np.ndarray,
                               attention_weights: np.ndarray) -> np.ndarray:
        """
        Apply temporal attention across video frames.
        
        Args:
            frames: Input frames of shape (batch_size, num_frames, channels, height, width)
            attention_weights: Attention weights of shape (batch_size, num_frames)
            
        Returns:
            Attended frames of shape (batch_size, channels, height, width)
            
        Raises:
            ValueError: If input shapes are invalid or empty
        """
        # Validate inputs
        if frames.size == 0 or attention_weights.size == 0:
            raise ValueError("Input arrays cannot be empty")
            
        if frames.shape[0] != attention_weights.shape[0]:
            raise ValueError("Batch sizes must match")
            
        if frames.shape[1] != attention_weights.shape[1]:
            raise ValueError("Number of frames must match attention weights")
            
        # Ensure inputs are contiguous and float32
        frames = np.ascontiguousarray(frames, dtype=np.float32)
        attention_weights = np.ascontiguousarray(attention_weights, dtype=np.float32)
        
        # Extract dimensions
        batch_size, num_frames, channels, height, width = frames.shape
        
        # Initialize output array to zeros
        output = np.zeros((batch_size, channels, height, width), dtype=np.float32)
        
        if self._asm_available:
            # Get pointers to the data
            frames_ptr = frames.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
            weights_ptr = attention_weights.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
            output_ptr = output.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
            
            # Call the assembly kernel
            self._temporal_attention_kernel(
                frames_ptr,
                weights_ptr,
                output_ptr,
                batch_size,
                num_frames,
                channels,
                height,
                width
            )
            
            return output
        else:
            return self._numpy_apply_temporal_attention(frames, attention_weights)
    
    def _numpy_apply_temporal_attention(self, frames, attention_weights):
        """NumPy implementation of temporal attention."""
        # Reshape attention weights for broadcasting
        attention_weights = attention_weights[:, :, None, None, None]
        
        # Apply attention weights
        attended = np.sum(frames * attention_weights, axis=1)
        
        return attended 