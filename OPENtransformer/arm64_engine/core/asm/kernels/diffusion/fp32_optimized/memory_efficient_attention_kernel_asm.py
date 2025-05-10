"""
Memory-Efficient Attention Kernel for text-to-video generation.
Handles long video sequences efficiently using block-sparse attention patterns
and sliding window attention with ARM64 NEON SIMD optimizations.
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

class MemoryEfficientAttentionKernelASM:
    def __init__(self, block_size: int = 64, window_size: int = 128):
        """Initialize the memory-efficient attention kernel.
        
        Args:
            block_size: Size of attention blocks for block-sparse attention
            window_size: Size of sliding window for local attention
        """
        self._attention_kernel = None
        self._asm_available = False
        self.block_size = block_size
        self.window_size = window_size
        
        try:
            self._attention_kernel = build_and_jit(self._get_asm_code(), "_memory_efficient_attention")
            if self._attention_kernel:
                self._attention_kernel.argtypes = [
                    ctypes.POINTER(ctypes.c_float),  # query
                    ctypes.POINTER(ctypes.c_float),  # key
                    ctypes.POINTER(ctypes.c_float),  # value
                    ctypes.POINTER(ctypes.c_float),  # output
                    ctypes.c_int,                    # batch_size
                    ctypes.c_int,                    # num_heads
                    ctypes.c_int,                    # seq_len
                    ctypes.c_int,                    # head_dim
                    ctypes.c_int,                    # block_size
                    ctypes.c_int,                    # window_size
                    ctypes.c_float                   # scale_factor
                ]
                self._attention_kernel.restype = ctypes.c_int
                self._asm_available = True
        except Exception as e:
            print(f"Warning: Failed to compile memory-efficient attention kernel: {e}")
            print("Falling back to NumPy implementation")
            
    def _get_asm_code(self) -> str:
        """Get the assembly code for the memory-efficient attention kernel."""
        return """
        .section __TEXT,__text
        .globl _memory_efficient_attention
        .align 2
        
        _memory_efficient_attention:
            // Save registers
            stp x29, x30, [sp, #-16]!
            mov x29, sp
            stp x19, x20, [sp, #-16]!
            stp x21, x22, [sp, #-16]!
            stp x23, x24, [sp, #-16]!
            stp d8, d9, [sp, #-16]!  // Save NEON registers
            stp d10, d11, [sp, #-16]!
            
            // Load parameters
            mov x19, x0  // query
            mov x20, x1  // key
            mov x21, x2  // value
            mov x22, x3  // output
            mov w23, w4  // batch_size
            mov w24, w5  // num_heads
            mov w25, w6  // seq_len
            mov w26, w7  // head_dim
            mov w27, w8  // block_size
            mov w28, w9  // window_size
            fmov s8, s0  // scale_factor
            
            // Duplicate scale factor across NEON vector
            dup v8.4s, v8.s[0]
            
            // Initialize batch counter
            mov w29, #0  // batch_idx
            
        batch_loop:
            cmp w29, w23
            b.ge batch_loop_end
            
            // Initialize head counter
            mov w30, #0  // head_idx
            
        head_loop:
            cmp w30, w24
            b.ge head_loop_end
            
            // Initialize sequence position counter
            mov w9, #0   // pos_idx
            
        position_loop:
            cmp w9, w25
            b.ge position_loop_end
            
            // Calculate window boundaries
            sub w10, w9, w28    // start = pos - window_size
            mov w11, #0
            cmp w10, w11
            csel w10, w11, w10, lt  // max(0, start)
            
            add w11, w9, w28    // end = pos + window_size
            cmp w11, w25
            csel w11, w25, w11, gt  // min(seq_len, end)
            
            // Initialize attention accumulator
            mov w12, #0  // acc_idx
            
        accumulate_loop:
            cmp w12, w26
            b.ge accumulate_loop_end
            
            // Process 4 elements at a time using NEON
            sub w13, w26, w12
            cmp w13, #4
            b.lt accumulate_remainder
            
            // Calculate base offset for current position
            mul w14, w29, w24      // batch_idx * num_heads
            mul w14, w14, w25      // * seq_len
            mul w14, w14, w26      // * head_dim
            mul w15, w30, w25      // head_idx * seq_len
            mul w15, w15, w26      // * head_dim
            add w14, w14, w15      // Add head offset
            mul w15, w9, w26       // pos_idx * head_dim
            add w14, w14, w15      // Add position offset
            add w14, w14, w12      // Add accumulator offset
            lsl x14, x14, #2       // * 4 (float size)
            
            // Load query values
            add x15, x19, x14      // Query pointer
            ld1 {v0.4s}, [x15]     // Load 4 float values
            
            // Initialize attention scores
            mov v1.16b, v0.16b     // Copy query values
            
            // Process window
            mov w15, w10           // window_idx = start
            
        window_loop:
            cmp w15, w11
            b.ge window_loop_end
            
            // Calculate key offset
            mul w16, w29, w24      // batch_idx * num_heads
            mul w16, w16, w25      // * seq_len
            mul w16, w16, w26      // * head_dim
            mul w17, w30, w25      // head_idx * seq_len
            mul w17, w17, w26      // * head_dim
            add w16, w16, w17      // Add head offset
            mul w17, w15, w26      // window_idx * head_dim
            add w16, w16, w17      // Add window offset
            add w16, w16, w12      // Add accumulator offset
            lsl x16, x16, #2       // * 4 (float size)
            
            // Load key values
            add x17, x20, x16      // Key pointer
            ld1 {v2.4s}, [x17]     // Load 4 float values
            
            // Compute attention scores
            fmul v3.4s, v0.4s, v2.4s  // query * key
            fmul v3.4s, v3.4s, v8.4s  // * scale_factor
            
            // Load value values
            add x17, x21, x16      // Value pointer
            ld1 {v4.4s}, [x17]     // Load 4 float values
            
            // Accumulate weighted values
            fmul v5.4s, v3.4s, v4.4s  // score * value
            fadd v1.4s, v1.4s, v5.4s  // += weighted value
            
            add w15, w15, #1       // Increment window index
            b window_loop
            
        window_loop_end:
            // Store accumulated values
            add x15, x22, x14      // Output pointer
            st1 {v1.4s}, [x15]     // Store 4 float values
            
            add w12, w12, #4       // Increment accumulator by 4
            b accumulate_loop
            
        accumulate_remainder:
            // Handle remaining elements one by one
            cmp w12, w26
            b.ge accumulate_loop_end
            
            // Calculate offset for single element
            mul w14, w29, w24      // batch_idx * num_heads
            mul w14, w14, w25      // * seq_len
            mul w14, w14, w26      // * head_dim
            mul w15, w30, w25      // head_idx * seq_len
            mul w15, w15, w26      // * head_dim
            add w14, w14, w15      // Add head offset
            mul w15, w9, w26       // pos_idx * head_dim
            add w14, w14, w15      // Add position offset
            add w14, w14, w12      // Add accumulator offset
            lsl x14, x14, #2       // * 4 (float size)
            
            // Load query value
            ldr s0, [x19, x14]     // Load single float value
            
            // Initialize attention score
            fmov s1, s0            // Copy query value
            
            // Process window
            mov w15, w10           // window_idx = start
            
        window_loop_remainder:
            cmp w15, w11
            b.ge window_loop_remainder_end
            
            // Calculate key offset
            mul w16, w29, w24      // batch_idx * num_heads
            mul w16, w16, w25      // * seq_len
            mul w16, w16, w26      // * head_dim
            mul w17, w30, w25      // head_idx * seq_len
            mul w17, w17, w26      // * head_dim
            add w16, w16, w17      // Add head offset
            mul w17, w15, w26      // window_idx * head_dim
            add w16, w16, w17      // Add window offset
            add w16, w16, w12      // Add accumulator offset
            lsl x16, x16, #2       // * 4 (float size)
            
            // Load key value
            ldr s2, [x20, x16]     // Load single float value
            
            // Compute attention score
            fmul s3, s0, s2        // query * key
            fmul s3, s3, s8        // * scale_factor
            
            // Load value
            ldr s4, [x21, x16]     // Load single float value
            
            // Accumulate weighted value
            fmul s5, s3, s4        // score * value
            fadd s1, s1, s5        // += weighted value
            
            add w15, w15, #1       // Increment window index
            b window_loop_remainder
            
        window_loop_remainder_end:
            // Store accumulated value
            str s1, [x22, x14]     // Store single float value
            
            add w12, w12, #1       // Increment accumulator
            b accumulate_remainder
            
        accumulate_loop_end:
            add w9, w9, #1         // Increment position
            b position_loop
            
        position_loop_end:
            add w30, w30, #1       // Increment head
            b head_loop
            
        head_loop_end:
            add w29, w29, #1       // Increment batch
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
            
    def compute_attention(self,
                         query: np.ndarray,
                         key: np.ndarray,
                         value: np.ndarray,
                         scale_factor: float = 1.0) -> np.ndarray:
        """Compute memory-efficient attention.
        
        Args:
            query: Query tensor of shape (batch_size, num_heads, seq_len, head_dim)
            key: Key tensor of shape (batch_size, num_heads, seq_len, head_dim)
            value: Value tensor of shape (batch_size, num_heads, seq_len, head_dim)
            scale_factor: Scaling factor for attention scores
            
        Returns:
            Output tensor of shape (batch_size, num_heads, seq_len, head_dim)
        """
        if not isinstance(query, np.ndarray) or not isinstance(key, np.ndarray) or not isinstance(value, np.ndarray):
            raise TypeError("Inputs must be numpy arrays")
            
        if query.shape != key.shape or query.shape != value.shape:
            raise ValueError("Input shapes must match")
            
        if query.dtype != np.float32:
            query = query.astype(np.float32)
        if key.dtype != np.float32:
            key = key.astype(np.float32)
        if value.dtype != np.float32:
            value = value.astype(np.float32)
            
        batch_size, num_heads, seq_len, head_dim = query.shape
        output = np.empty_like(query)
        
        if not self._asm_available:
            return self._numpy_compute_attention(
                query, key, value,
                scale_factor
            )
            
        try:
            result = self._attention_kernel(
                query.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                key.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                value.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                output.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                ctypes.c_int(batch_size),
                ctypes.c_int(num_heads),
                ctypes.c_int(seq_len),
                ctypes.c_int(head_dim),
                ctypes.c_int(self.block_size),
                ctypes.c_int(self.window_size),
                ctypes.c_float(scale_factor)
            )
            if result != 1:
                raise RuntimeError("Memory-efficient attention kernel failed")
        except Exception as e:
            print(f"Warning: Memory-efficient attention kernel failed: {e}")
            return self._numpy_compute_attention(
                query, key, value,
                scale_factor
            )
            
        return output
        
    def _numpy_compute_attention(self,
                               query: np.ndarray,
                               key: np.ndarray,
                               value: np.ndarray,
                               scale_factor: float = 1.0) -> np.ndarray:
        """NumPy implementation of memory-efficient attention."""
        batch_size, num_heads, seq_len, head_dim = query.shape
        output = np.zeros_like(query)
        
        # Process each position in the sequence
        for b in range(batch_size):
            for h in range(num_heads):
                for pos in range(seq_len):
                    # Calculate window boundaries
                    start = max(0, pos - self.window_size)
                    end = min(seq_len, pos + self.window_size)
                    
                    # Compute attention scores for window
                    scores = np.matmul(
                        query[b, h, pos:pos+1, :],
                        key[b, h, start:end, :].transpose()
                    ) * scale_factor
                    
                    # Apply attention to values
                    output[b, h, pos, :] = np.matmul(
                        scores,
                        value[b, h, start:end, :]
                    ).squeeze()
        
        return output 