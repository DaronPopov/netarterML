import numpy as np
import ctypes
import sys
import os
from typing import Tuple
import logging
import gc
from pathlib import Path

# Add the project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))))
sys.path.insert(0, project_root)

from OPENtransformer.core.asm.assembler.builder import build_and_jit

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress linter warnings for undefined variables - these are defined at runtime or in assembly
_conv2d_kernel = None  # type: ignore
_max_pool2d_kernel = None  # type: ignore
_batch_norm_kernel = None  # type: ignore

class VisionKernelsASM:
    def __init__(self):
        # Load the compiled SIMD library
        lib_path = os.path.join(os.path.dirname(__file__), 'libvision_kernels.dylib')
        if not os.path.exists(lib_path):
            raise FileNotFoundError(f"SIMD library not found at {lib_path}")
        
        self.lib = ctypes.CDLL(lib_path)
        
        # Define function signatures
        self.lib.bgr_to_rgb_simd.argtypes = [
            ctypes.POINTER(ctypes.c_uint8),  # input
            ctypes.POINTER(ctypes.c_uint8),  # output
            ctypes.c_int,                    # width
            ctypes.c_int                     # height
        ]
        
        self.lib.normalize_image_simd.argtypes = [
            ctypes.POINTER(ctypes.c_float),  # input
            ctypes.POINTER(ctypes.c_float),  # output
            ctypes.c_int,                    # width
            ctypes.c_int,                    # height
            ctypes.c_int                     # channels
        ]
    
    def bgr_to_rgb(self, image):
        """Convert BGR to RGB using SIMD"""
        height, width = image.shape[:2]
        output = np.empty_like(image)
        
        # Ensure contiguous arrays
        input_arr = np.ascontiguousarray(image, dtype=np.uint8)
        output_arr = np.ascontiguousarray(output, dtype=np.uint8)
        
        # Get pointers to data
        input_ptr = input_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
        output_ptr = output_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
        
        # Call SIMD function
        self.lib.bgr_to_rgb_simd(input_ptr, output_ptr, width, height)
        
        return output_arr
    
    def normalize_image(self, image):
        """Normalize image using SIMD"""
        height, width, channels = image.shape
        output = np.empty_like(image, dtype=np.float32)
        
        # Ensure contiguous arrays
        input_arr = np.ascontiguousarray(image, dtype=np.float32)
        output_arr = np.ascontiguousarray(output, dtype=np.float32)
        
        # Get pointers to data
        input_ptr = input_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        output_ptr = output_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        
        # Call SIMD function
        self.lib.normalize_image_simd(input_ptr, output_ptr, width, height, channels)
        
        return output_arr

def ensure_contiguous(tensor: np.ndarray) -> np.ndarray:
    """Ensure tensor is contiguous in memory."""
    if not tensor.flags['C_CONTIGUOUS']:
        return np.ascontiguousarray(tensor)
    return tensor

def conv2d_code():
    return """
    .global _conv2d_simd
    .align 4
_conv2d_simd:
    // Save registers
    stp x29, x30, [sp, #-16]!
    mov x29, sp

    // Parameters:
    // x0: input data
    // x1: kernel data
    // x2: output data
    // x3: input height
    // x4: input width
    // x5: kernel height
    // x6: kernel width
    // x7: stride

    conv2d_height_loop:
        mov x8, #0  // width counter
        conv2d_width_loop:
            mov x9, #0  // kernel height counter
            conv2d_kernel_height_loop:
                mov x10, #0  // kernel width counter
                conv2d_kernel_width_loop:
                    // Load and multiply
                    ldr q0, [x0]
                    ldr q1, [x1]
                    fmul v2.4s, v0.4s, v1.4s
                    str q2, [x2]
                    
                    add x10, x10, #1
                    cmp x10, x6
                    b.lt conv2d_kernel_width_loop
                conv2d_kernel_width_loop_end:
                
                add x9, x9, #1
                cmp x9, x5
                b.lt conv2d_kernel_height_loop
            conv2d_kernel_height_loop_end:
            
            add x8, x8, x7
            cmp x8, x4
            b.lt conv2d_width_loop
        conv2d_width_loop_end:
        
        add x7, x7, #1
        cmp x7, x3
        b.lt conv2d_height_loop
    conv2d_height_loop_end:

    // Restore registers and return
    ldp x29, x30, [sp], #16
    ret
    """

def max_pool2d_code():
    return """
    .global _max_pool2d_simd
    .align 4
_max_pool2d_simd:
    // Save registers
    stp x29, x30, [sp, #-16]!
    mov x29, sp

    // Parameters:
    // x0: input data
    // x1: output data
    // x2: height
    // x3: width
    // x4: pool size
    // x5: stride
    // x6: channels

    mov x7, #0  // channel counter
    maxpool_channel_loop:
        mov x8, #0  // height counter
        maxpool_height_loop:
            mov x9, #0  // width counter
            maxpool_width_loop:
                // Load and compare 4 elements at a time
                ldr q0, [x0]
                fmax v1.4s, v0.4s, v0.4s
                str q1, [x1]
                
                add x9, x9, x5
                cmp x9, x3
                b.lt maxpool_width_loop
            maxpool_width_loop_end:
            
            add x8, x8, x5
            cmp x8, x2
            b.lt maxpool_height_loop
        maxpool_height_loop_end:
        
        add x7, x7, #1
        cmp x7, x6
        b.lt maxpool_channel_loop
    maxpool_channel_loop_end:

    // Restore registers and return
    ldp x29, x30, [sp], #16
    ret
    """

def batch_norm_code():
    return """
    .global _batch_norm_simd
    .align 4
_batch_norm_simd:
    // Save registers
    stp x29, x30, [sp, #-16]!
    mov x29, sp

    // Parameters:
    // x0: input data
    // x1: output data
    // x2: mean
    // x3: variance
    // x4: gamma
    // x5: beta
    // x6: epsilon
    // x7: size

    mov x8, #0  // counter
    bn_loop:
        // Load values
        ldr q0, [x0, x8]
        ldr q1, [x2, x8]  // mean
        ldr q2, [x3, x8]  // variance
        ldr q3, [x4, x8]  // gamma
        ldr q4, [x5, x8]  // beta
        
        // Compute (x - mean)
        fsub v5.4s, v0.4s, v1.4s
        
        // Compute 1/sqrt(variance + epsilon)
        fadd v6.4s, v2.4s, v6.4s  // variance + epsilon
        fsqrt v6.4s, v6.4s
        fdiv v6.4s, v6.4s, v6.4s  // reciprocal
        
        // Normalize
        fmul v7.4s, v5.4s, v6.4s
        
        // Scale and shift
        fmul v8.4s, v7.4s, v3.4s  // * gamma
        fadd v9.4s, v8.4s, v4.4s  // + beta
        
        // Store result
        str q9, [x1, x8]
        
        add x8, x8, #16
        cmp x8, x7
        b.lt bn_loop
    bn_loop_end:

    // Restore registers and return
    ldp x29, x30, [sp], #16
    ret
    """

def create_vision_kernels():
    """Create and return the vision kernel functions."""
    return {
        'conv2d': build_and_jit(conv2d_code(), "_conv2d"),
        'max_pool2d': build_and_jit(max_pool2d_code(), "_max_pool2d"),
        'batch_norm': build_and_jit(batch_norm_code(), "_batch_norm")
    }

def conv2d(input_tensor: np.ndarray, kernel: np.ndarray, stride: int = 1, padding: int = 0) -> np.ndarray:
    """Perform 2D convolution with memory management."""
    try:
        # Ensure input tensors are contiguous
        input_tensor = ensure_contiguous(input_tensor)
        kernel = ensure_contiguous(kernel)
        
        # Get dimensions
        batch_size, in_channels, in_height, in_width = input_tensor.shape
        out_channels, _, kernel_size, _ = kernel.shape
        
        # Calculate output dimensions
        out_height = (in_height - kernel_size + 2 * padding) // stride + 1
        out_width = (in_width - kernel_size + 2 * padding) // stride + 1
        
        # Allocate output tensor
        output = np.zeros((batch_size, out_channels, out_height, out_width), dtype=np.float32)
        output = ensure_contiguous(output)
        
        # Call SIMD kernel
        _conv2d_kernel(
            input_tensor.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            output.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            kernel.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            in_height, in_width, in_channels, out_channels,
            kernel_size, stride, padding
        )
        
        return output
        
    except Exception as e:
        logger.error(f"Error in conv2d: {str(e)}")
        raise
    finally:
        # Force garbage collection
        gc.collect()

def max_pool2d(input_tensor: np.ndarray, kernel_size: int = 2, stride: int = 2) -> np.ndarray:
    """Perform 2D max pooling with memory management."""
    try:
        # Ensure input tensor is contiguous
        input_tensor = ensure_contiguous(input_tensor)
        
        # Get dimensions
        batch_size, channels, in_height, in_width = input_tensor.shape
        
        # Calculate output dimensions
        out_height = (in_height - kernel_size) // stride + 1
        out_width = (in_width - kernel_size) // stride + 1
        
        # Allocate output tensor
        output = np.zeros((batch_size, channels, out_height, out_width), dtype=np.float32)
        output = ensure_contiguous(output)
        
        # Call SIMD kernel
        _max_pool2d_kernel(
            input_tensor.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            output.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            kernel_size, stride, in_height, in_width, channels
        )
        
        return output
        
    except Exception as e:
        logger.error(f"Error in max_pool2d: {str(e)}")
        raise
    finally:
        # Force garbage collection
        gc.collect()

def batch_norm(input_tensor: np.ndarray, gamma: np.ndarray, beta: np.ndarray, 
               mean: np.ndarray, variance: np.ndarray, epsilon: float = 1e-5) -> np.ndarray:
    """Perform batch normalization with memory management."""
    try:
        # Ensure all tensors are contiguous
        input_tensor = ensure_contiguous(input_tensor)
        gamma = ensure_contiguous(gamma)
        beta = ensure_contiguous(beta)
        mean = ensure_contiguous(mean)
        variance = ensure_contiguous(variance)
        
        # Get dimensions
        batch_size, channels, height, width = input_tensor.shape
        
        # Allocate output tensor
        output = np.zeros_like(input_tensor, dtype=np.float32)
        output = ensure_contiguous(output)
        
        # Call SIMD kernel
        _batch_norm_kernel(
            input_tensor.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            output.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            gamma.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            beta.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            mean.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            variance.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.c_float(epsilon),
            batch_size, channels, height, width
        )
        
        return output
        
    except Exception as e:
        logger.error(f"Error in batch_norm: {str(e)}")
        raise
    finally:
        # Force garbage collection
        gc.collect()

def get_kernel_code() -> str:
    """Get all vision kernel codes."""
    return conv2d_code() + "\n" + max_pool2d_code() + "\n" + batch_norm_code()

def vit_patch_embedding_simd_code():
    """Generate ARM64 NEON SIMD assembly code for patch embedding."""
    return """
    .section __TEXT,__text
    .globl _vit_patch_embedding_simd
    .align 2

    _vit_patch_embedding_simd:
        // x0: input ptr (float*)
        // x1: output ptr (float*)
        // x2: patch_size
        // x3: stride
        // x4: input_channels (C)
        // x5: embed_dim (D)
        // x6: height (H)
        // x7: width (W)
        // x8: weights ptr (float*)

        // Save frame and registers
        stp x29, x30, [sp, #-16]!
        mov x29, sp
        stp x19, x20, [sp, #-16]!
        stp x21, x22, [sp, #-16]!
        stp x23, x24, [sp, #-16]!
        
        // Calculate number of patches
        sub x9, x6, x2        // H - P
        add x9, x9, x3        // + S
        udiv x9, x9, x3       // / S = num_patches_h
        
        sub x10, x7, x2       // W - P
        add x10, x10, x3      // + S
        udiv x10, x10, x3     // / S = num_patches_w
        
        mul x11, x9, x10      // total patches = H_p * W_p
        
        // Calculate patch size in elements
        mul x12, x2, x2       // P * P
        mul x12, x12, x4      // * C = elements per patch
        
        // Process patches
        mov x13, #0           // patch counter
        
    patch_loop:
        cmp x13, x11
        b.ge patch_loop_end
        
        // Calculate patch position
        udiv x14, x13, x10    // patch_y = patch_idx / num_patches_w
        mul x15, x14, x3      // patch_y * stride
        msub x16, x14, x10, x13  // patch_x = patch_idx % num_patches_w
        mul x17, x16, x3      // patch_x * stride
        
        // Process elements in patch (4 at a time)
        mov x19, #0           // element counter
        lsr x20, x12, #2      // elements/4
        
    element_loop:
        cmp x19, x20
        b.ge element_loop_end
        
        // Load 4 input values
        add x21, x0, x15, lsl #2   // input_y offset
        add x21, x21, x17, lsl #2  // input_x offset
        add x21, x21, x19, lsl #4  // element offset
        ld1 {v0.4s}, [x21]
        
        // Load corresponding weights
        add x22, x8, x19, lsl #4   // weight offset
        ld1 {v1.4s}, [x22]
        
        // Multiply and accumulate
        fmul v2.4s, v0.4s, v1.4s
        
        // Store result
        add x23, x1, x13, lsl #2   // output patch offset
        add x23, x23, x19, lsl #4  // element offset
        st1 {v2.4s}, [x23]
        
        add x19, x19, #4
        b element_loop
        
    element_loop_end:
        add x13, x13, #1
        b patch_loop
        
    patch_loop_end:
        // Restore registers
        ldp x23, x24, [sp], #16
        ldp x21, x22, [sp], #16
        ldp x19, x20, [sp], #16
        ldp x29, x30, [sp], #16
        ret
    """

def vit_attention_simd_code():
    """Generate ARM64 NEON SIMD assembly code for multi-head attention."""
    return """
    .section __TEXT,__text
    .globl _vit_attention_simd
    .align 2

    _vit_attention_simd:
        // x0: query ptr (float*)
        // x1: key ptr (float*)
        // x2: value ptr (float*)
        // x3: output ptr (float*)
        // x4: seq_len
        // x5: head_dim
        // x6: num_heads
        
        // Save frame and registers
        stp x29, x30, [sp, #-16]!
        mov x29, sp
        stp x19, x20, [sp, #-16]!
        stp x21, x22, [sp, #-16]!
        
        // Calculate attention scores (Q * K^T)
        // Process 4 elements at a time
        lsr x7, x5, #2       // head_dim/4
        
        // Initialize loop counters
        mov x8, #0           // sequence position counter
        
    seq_loop:
        cmp x8, x4
        b.ge seq_loop_end
        
        mov x9, #0           // head counter
        
    head_loop:
        cmp x9, x6
        b.ge head_loop_end
        
        // Calculate offsets
        mul x10, x8, x5      // seq_pos * head_dim
        mul x11, x9, x5      // head_idx * head_dim
        add x12, x10, x11    // total offset
        
        // Process elements in chunks of 4
        mov x13, #0          // element counter
        
    element_loop:
        cmp x13, x7
        b.ge element_loop_end
        
        // Load Q and K vectors
        add x14, x0, x12, lsl #2  // query offset
        add x14, x14, x13, lsl #4
        ld1 {v0.4s}, [x14]
        
        add x15, x1, x12, lsl #2  // key offset
        add x15, x15, x13, lsl #4
        ld1 {v1.4s}, [x15]
        
        // Compute dot product
        fmul v2.4s, v0.4s, v1.4s
        faddp v2.4s, v2.4s, v2.4s
        faddp s2, v2.2s
        
        // Store result
        add x16, x3, x12, lsl #2  // output offset
        add x16, x16, x13, lsl #2
        str s2, [x16]
        
        add x13, x13, #4
        b element_loop
        
    element_loop_end:
        add x9, x9, #1
        b head_loop
        
    head_loop_end:
        add x8, x8, #1
        b seq_loop
        
    seq_loop_end:
        // Restore registers
        ldp x21, x22, [sp], #16
        ldp x19, x20, [sp], #16
        ldp x29, x30, [sp], #16
        ret
    """

def vit_mlp_simd_code():
    """Generate ARM64 NEON SIMD assembly code for MLP/feed-forward layers."""
    return """
    .section __TEXT,__text
    .globl _vit_mlp_simd
    .align 2
    
    _vit_mlp_simd:
        // x0: input ptr (float*)
        // x1: weights1 ptr (float*)
        // x2: weights2 ptr (float*)
        // x3: output ptr (float*)
        // x4: input_dim
        // x5: hidden_dim
        // x6: output_dim
        
        stp x29, x30, [sp, #-16]!
        stp x19, x20, [sp, #-16]!
        
        // Process 4 elements at a time
        lsr x7, x4, #2      // input_dim/4
        
        // Initialize zero vector for ReLU
        movi v3.4s, #0
        
    1:  // outer_loop
        // Load input and weights
        ld1 {v0.4s}, [x0], #16
        ld1 {v1.4s}, [x1], #16
        
        // Multiply and accumulate
        fmul v2.4s, v0.4s, v1.4s
        
        // Apply ReLU
        fmax v2.4s, v2.4s, v3.4s
        
        // Store result
        st1 {v2.4s}, [x3], #16
        
        subs x7, x7, #1
        b.ne 1b
        
        ldp x19, x20, [sp], #16
        ldp x29, x30, [sp], #16
        ret
    """

def build_vit_kernels():
    """Build all Vision Transformer SIMD kernels."""
    patch_embedding_kernel = build_and_jit(vit_patch_embedding_simd_code(), "_vit_patch_embedding_simd")
    attention_kernel = build_and_jit(vit_attention_simd_code(), "_vit_attention_simd")
    mlp_kernel = build_and_jit(vit_mlp_simd_code(), "_vit_mlp_simd")
    
    return {
        'patch_embedding': patch_embedding_kernel,
        'attention': attention_kernel,
        'mlp': mlp_kernel
    } 