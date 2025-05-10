"""
Cross-attention kernel for text-to-video generation.
"""

import numpy as np
import ctypes
import os
import tempfile
import subprocess

def build_and_jit(asm_code: str, entry_point: str):
    """Build and JIT compile ARM64 assembly code."""
    # Create temporary files
    with tempfile.NamedTemporaryFile(suffix='.s', delete=False) as asm_file:
        asm_file.write(asm_code.encode())
        asm_path = asm_file.name
        
    with tempfile.NamedTemporaryFile(suffix='.o', delete=False) as obj_file:
        obj_path = obj_file.name
        
    with tempfile.NamedTemporaryFile(suffix='.dylib', delete=False) as lib_file:
        lib_path = lib_file.name
        
    try:
        # Assemble
        print(f"\nAssembling {asm_path}...")
        result = subprocess.run(['as', '-v', '-arch', 'arm64', asm_path, '-o', obj_path], 
                              capture_output=True, text=True)
        if result.returncode != 0:
            print("Assembly error output:")
            print(result.stderr)
            print("\nAssembly code:")
            with open(asm_path, "r") as f:
                print(f.read())
            raise subprocess.CalledProcessError(result.returncode, result.args, result.stdout, result.stderr)
            
        # Check object file symbols
        print("\nObject file symbols:")
        nm_output = subprocess.check_output(['nm', '-g', obj_path]).decode()
        print(nm_output)
        
        # Get SDK path
        sdk_path = subprocess.check_output(['xcrun', '--show-sdk-path']).decode().strip()
        
        # Link using ld
        print(f"\nLinking to {lib_path}...")
        result = subprocess.run([
            'ld',
            '-dylib',
            '-arch', 'arm64',
            '-platform_version', 'macos', '14.0', '14.0',
            '-syslibroot', sdk_path,
            '-o', lib_path,
            obj_path,
            '-lSystem'
        ], capture_output=True, text=True)
        
        if result.returncode != 0:
            print("Linking error output:")
            print(result.stderr)
            raise subprocess.CalledProcessError(result.returncode, result.args, result.stdout, result.stderr)
        
        # Check library symbols
        print("\nLibrary symbols:")
        nm_output = subprocess.check_output(['nm', '-g', lib_path]).decode()
        print(nm_output)
        
        # Load library
        print(f"\nLoading library {lib_path}...")
        lib = ctypes.CDLL(lib_path)
        
        # List available symbols in the library
        print("\nAvailable symbols in library:")
        for sym in dir(lib):
            print(f"  {sym}")
            
        # Try to get the function
        try:
            func = getattr(lib, "_cross_attention")
            print(f"Successfully loaded function")
            
            # Set function signature
            func.argtypes = [
                ctypes.POINTER(ctypes.c_float),  # text_embeddings
                ctypes.POINTER(ctypes.c_float),  # video_features
                ctypes.POINTER(ctypes.c_float),  # output
                ctypes.c_int,                    # batch_size
                ctypes.c_int,                    # text_len
                ctypes.c_int,                    # video_len
                ctypes.c_int,                    # hidden_size
                ctypes.c_float,                  # scale_factor
                ctypes.c_float                   # dropout_prob
            ]
            func.restype = ctypes.c_int
            
            return func
        except AttributeError:
            print(f"Could not find function in library")
            return None
            
    except Exception as e:
        print(f"Failed to build and load assembly: {e}")
        return None
    finally:
        # Clean up temporary files
        os.unlink(asm_path)
        os.unlink(obj_path)
        os.unlink(lib_path)

class CrossAttentionKernelASM:
    def __init__(self):
        """Initialize the cross-attention kernel."""
        self._cross_attention_kernel = None
        self._asm_available = False
        
        try:
            self._cross_attention_kernel = build_and_jit(self._get_asm_code(), "_cross_attention")
            if self._cross_attention_kernel:
                self._cross_attention_kernel.argtypes = [
                    ctypes.POINTER(ctypes.c_float),  # text_embeddings
                    ctypes.POINTER(ctypes.c_float),  # video_features
                    ctypes.POINTER(ctypes.c_float),  # output
                    ctypes.c_int,                    # batch_size
                    ctypes.c_int,                    # text_len
                    ctypes.c_int,                    # video_len
                    ctypes.c_int,                    # hidden_size
                    ctypes.c_float,                  # scale_factor
                    ctypes.c_float                   # dropout_prob
                ]
                self._cross_attention_kernel.restype = ctypes.c_int
                self._asm_available = True
        except Exception as e:
            print(f"Warning: Failed to compile cross-attention kernel: {e}")
            print("Falling back to NumPy implementation")
            
    def _get_asm_code(self) -> str:
        """Get the assembly code for the cross-attention kernel."""
        return """
        .section __TEXT,__text
        .globl _cross_attention
        .align 2
        
        _cross_attention:
            // Save registers
            stp x29, x30, [sp, #-16]!
            mov x29, sp
            stp x19, x20, [sp, #-16]!
            stp x21, x22, [sp, #-16]!
            stp x23, x24, [sp, #-16]!
            
            // Load parameters
            mov x19, x0  // text_embeddings
            mov x20, x1  // video_features
            mov x21, x2  // output
            mov w22, w3  // batch_size
            mov w23, w4  // text_len
            mov w24, w5  // video_len
            mov w25, w6  // hidden_size
            fmov s8, s0  // scale_factor
            fmov s9, s1  // dropout_prob
            
            // Initialize batch counter
            mov w26, #0  // batch_idx
            
        batch_loop:
            cmp w26, w22
            b.ge batch_loop_end
            
            // Initialize text counter
            mov w27, #0  // text_idx
            
        text_loop:
            cmp w27, w23
            b.ge text_loop_end
            
            // Initialize video counter
            mov w28, #0  // video_idx
            
        video_loop:
            cmp w28, w24
            b.ge video_loop_end
            
            // Calculate dot product using SIMD
            mov w29, #0  // hidden_idx
            eor v0.16b, v0.16b, v0.16b  // dot_product = 0
            
        hidden_loop:
            cmp w29, w25
            b.ge hidden_loop_end
            
            // Calculate memory offsets
            lsl x12, x29, #2  // hidden_idx * 4
            
            // Load 4 elements at once using SIMD
            add x13, x19, x12
            add x14, x20, x12
            ld1 {v1.4s}, [x13]  // text elements
            ld1 {v2.4s}, [x14]  // video elements
            
            // Multiply and accumulate
            fmul v3.4s, v1.4s, v2.4s
            faddp v0.4s, v3.4s, v3.4s
            faddp s0, v0.2s
            
            add w29, w29, #4
            b hidden_loop
            
        hidden_loop_end:
            // Apply scale factor
            fmul s0, s0, s8
            
            // Apply dropout if probability > 0
            fcmp s9, #0.0
            b.eq store_output
            
            // Generate random number
            mrs x11, TPIDR_EL0
            add x11, x11, #1
            msr TPIDR_EL0, x11
            and w12, w11, #0x7fffffff
            ucvtf s10, w12
            fmov s11, #1.0
            fdiv s10, s10, s11
            
            // Apply dropout mask
            fcmp s10, s9
            b.gt store_output
            fmov s0, #0.0
            
        store_output:
            // Calculate output index
            mul w13, w26, w23      // batch_offset = batch_idx * text_len
            add w13, w13, w27      // + text_idx
            mul w13, w13, w24      // * video_len
            add w13, w13, w28      // + video_idx
            
            // Calculate memory offset
            lsl x14, x13, #2   // index * 4
            
            // Store attention score
            str s0, [x21, x14]
            
            add w28, w28, #1
            b video_loop
            
        video_loop_end:
            add w27, w27, #1
            b text_loop
            
        text_loop_end:
            add w26, w26, #1
            b batch_loop
            
        batch_loop_end:
            // Return success
            mov x0, #1
            
            // Restore registers
            ldp x23, x24, [sp], #16
            ldp x21, x22, [sp], #16
            ldp x19, x20, [sp], #16
            ldp x29, x30, [sp], #16
            ret
        """
            
    def apply_cross_attention(self, text_embeddings: np.ndarray, video_features: np.ndarray,
                            scale_factor: float = 1.0, dropout_prob: float = 0.0) -> np.ndarray:
        """Apply cross-attention between text embeddings and video features."""
        if not isinstance(text_embeddings, np.ndarray) or not isinstance(video_features, np.ndarray):
            raise TypeError("Inputs must be numpy arrays")
            
        if text_embeddings.dtype != np.float32 or video_features.dtype != np.float32:
            text_embeddings = text_embeddings.astype(np.float32)
            video_features = video_features.astype(np.float32)
            
        batch_size, text_len, hidden_size = text_embeddings.shape
        _, video_len, _ = video_features.shape
        
        if video_features.shape[0] != batch_size or video_features.shape[2] != hidden_size:
            raise ValueError("Input shapes are incompatible")
            
        output = np.empty((batch_size, text_len, video_len), dtype=np.float32)
        
        if not self._asm_available:
            return self._numpy_apply_cross_attention(text_embeddings, video_features, scale_factor, dropout_prob)
            
        try:
            result = self._cross_attention_kernel(
                text_embeddings.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                video_features.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                output.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                ctypes.c_int(batch_size),
                ctypes.c_int(text_len),
                ctypes.c_int(video_len),
                ctypes.c_int(hidden_size),
                ctypes.c_float(scale_factor),
                ctypes.c_float(dropout_prob)
            )
            if result != 1:
                raise RuntimeError("Cross-attention kernel failed")
        except Exception as e:
            print(f"Warning: Cross-attention kernel failed: {e}")
            return self._numpy_apply_cross_attention(text_embeddings, video_features, scale_factor, dropout_prob)
            
        return output
        
    def _numpy_apply_cross_attention(self, text_embeddings: np.ndarray, video_features: np.ndarray,
                                   scale_factor: float = 1.0, dropout_prob: float = 0.0) -> np.ndarray:
        """NumPy implementation of cross-attention."""
        # Calculate attention scores
        attention_scores = np.matmul(text_embeddings, video_features.transpose(0, 2, 1))
        attention_scores *= scale_factor
        
        # Apply softmax
        attention_scores = np.exp(attention_scores - np.max(attention_scores, axis=-1, keepdims=True))
        attention_scores /= np.sum(attention_scores, axis=-1, keepdims=True)
        
        # Apply dropout
        if dropout_prob > 0:
            dropout_mask = (np.random.random(attention_scores.shape) > dropout_prob).astype(np.float32)
            attention_scores *= dropout_mask
            attention_scores /= (1 - dropout_prob)  # Scale to maintain expected values
            
        return attention_scores

    def compute_attention(self,
                         query: np.ndarray,
                         key: np.ndarray,
                         value: np.ndarray,
                         scale_factor: float = 1.0,
                         dropout_prob: float = 0.0) -> np.ndarray:
        """Compute cross-attention between query and key-value pairs.
        
        Args:
            query: Query tensor of shape (batch_size, text_len, hidden_size)
            key: Key tensor of shape (batch_size, video_len, hidden_size)
            value: Value tensor of shape (batch_size, video_len, hidden_size)
            scale_factor: Scaling factor for attention scores
            dropout_prob: Dropout probability
            
        Returns:
            Output tensor of shape (batch_size, text_len, hidden_size)
        """
        # Validate inputs
        if not isinstance(query, np.ndarray) or not isinstance(key, np.ndarray) or not isinstance(value, np.ndarray):
            raise TypeError("Inputs must be numpy arrays")
            
        if query.shape[0] != key.shape[0] or key.shape != value.shape:
            raise ValueError("Input shapes are incompatible")
            
        if query.shape[2] != key.shape[2]:
            raise ValueError("Hidden sizes must match")
            
        # Compute attention scores and apply them to values
        attention_output = self.apply_cross_attention(
            text_embeddings=query,
            video_features=key,
            scale_factor=scale_factor,
            dropout_prob=dropout_prob
        )
        
        # Apply attention scores to values
        batch_size, text_len, video_len = attention_output.shape
        output = np.zeros((batch_size, text_len, query.shape[2]), dtype=np.float32)
        
        # Batch matrix multiplication between attention scores and values
        for b in range(batch_size):
            output[b] = np.matmul(attention_output[b], value[b])
            
        return output 