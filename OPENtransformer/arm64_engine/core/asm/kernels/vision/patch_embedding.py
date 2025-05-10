import numpy as np
import ctypes
import sys
import os
from typing import Tuple

# Add the project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))))
sys.path.insert(0, project_root)

from OPENtransformer.core.asm.assembler.builder import build_and_jit

def patch_embedding_code() -> str:
    """Generate ARM64 assembly code for patch embedding operation."""
    return """
    .section __TEXT,__text,regular,pure_instructions
    .globl _patch_embedding
    .p2align 2

    _patch_embedding:
        // Input:
        // x0: input image pointer (float*)
        // x1: output patches pointer (float*)
        // x2: patch size (int)
        // x3: stride (int)
        // x4: input channels (int)
        // x5: output channels (int)
        // x6: height (int)
        // x7: width (int)
        // x8: projection weights pointer (float*)
        
        // Save registers
        stp x29, x30, [sp, #-16]!
        stp x19, x20, [sp, #-16]!
        stp x21, x22, [sp, #-16]!
        stp x23, x24, [sp, #-16]!
        
        // Calculate number of patches
        sub x9, x6, x2        // height - patch_size
        add x9, x9, x3        // + stride
        udiv x9, x9, x3       // / stride = num_patches_h
        
        sub x10, x7, x2       // width - patch_size
        add x10, x10, x3      // + stride
        udiv x10, x10, x3     // / stride = num_patches_w
        
        mul x11, x9, x10      // total patches = num_patches_h * num_patches_w
        
        // Calculate patch size in elements
        mul x25, x2, x2       // patch_size * patch_size
        mul x25, x25, x4      // * input_channels = total elements per patch
        
        // Initialize loop counters
        mov x12, #0           // patch index
        
    patch_loop:
        cmp x12, x11
        b.ge patch_loop_end
        
        // Calculate current patch position
        udiv x13, x12, x10    // row = patch_index / num_patches_w
        msub x14, x13, x10, x12  // col = patch_index % num_patches_w
        
        // Calculate base input offset for this patch
        mul x15, x13, x3      // row * stride
        mul x15, x15, x7      // * width
        mul x15, x15, x4      // * channels
        madd x15, x14, x3, x15  // + col * stride
        add x15, x15, x0      // + input pointer
        
        // Project patch through weights
        mov x16, #0           // output channel counter
        
    channel_loop:
        cmp x16, x5
        b.ge channel_loop_end
        
        // Initialize accumulator
        fmov s0, wzr
        
        // Loop through patch elements
        mov x17, #0           // element counter
        
    element_loop:
        cmp x17, x25
        b.ge element_loop_end
        
        // Load input value
        ldr s1, [x15, x17, lsl #2]  // load input value
        
        // Load weight
        mul x18, x16, x25     // output_channel * patch_elements
        add x18, x18, x17     // + element_index
        ldr s2, [x8, x18, lsl #2]  // load weight
        
        // Multiply and accumulate
        fmul s1, s1, s2
        fadd s0, s0, s1
        
        add x17, x17, #1
        b element_loop
        
    element_loop_end:
        // Normalize by patch size
        ucvtf s3, x25
        fdiv s0, s0, s3
        
        // Store result
        madd x18, x12, x5, x16  // output_index = patch_index * output_channels + output_channel
        str s0, [x1, x18, lsl #2]
        
        add x16, x16, #1
        b channel_loop
        
    channel_loop_end:
        add x12, x12, #1
        b patch_loop
        
    patch_loop_end:
        // Restore registers
        ldp x23, x24, [sp], #16
        ldp x21, x22, [sp], #16
        ldp x19, x20, [sp], #16
        ldp x29, x30, [sp], #16
        ret
    """

def create_patch_embedding_kernel():
    """Create and return the patch embedding kernel function."""
    return build_and_jit(patch_embedding_code(), "_patch_embedding")

def patch_embedding_python(image, patch_size, stride, projection_weights):
    """
    Extract patches from an image and project them through learned weights.
    
    Args:
        image: Input image of shape (batch_size, height, width, channels) or (height, width, channels)
        patch_size: Size of patches to extract (square patches)
        stride: Stride between patches
        projection_weights: Weights for projecting patches of shape (output_channels, patch_size * patch_size * channels)
    
    Returns:
        patches: Extracted and projected patches of shape (batch_size, num_patches, output_channels)
        (num_patches_h, num_patches_w): Number of patches in height and width dimensions
    """
    # Handle both batched and unbatched inputs
    if len(image.shape) == 4:
        batch_size, height, width, channels = image.shape
        is_batched = True
    else:
        height, width, channels = image.shape
        batch_size = 1
        is_batched = False
        image = image[np.newaxis, ...]
    
    # Calculate number of patches
    num_patches_h = (height - patch_size) // stride + 1
    num_patches_w = (width - patch_size) // stride + 1
    total_patches = num_patches_h * num_patches_w
    output_channels = projection_weights.shape[0]
    
    # Initialize output array
    patches = np.zeros((batch_size, total_patches, output_channels), dtype=np.float32)
    
    # Extract and process each patch
    for b in range(batch_size):
        patch_idx = 0
        for i in range(0, height - patch_size + 1, stride):
            for j in range(0, width - patch_size + 1, stride):
                # Extract patch
                patch = image[b, i:i+patch_size, j:j+patch_size, :].reshape(-1)
                
                # Normalize patch
                patch = (patch - patch.mean()) / (patch.std() + 1e-6)
                
                # Project patch through weights
                # Normalize weights by the square root of the input dimension
                norm_weights = projection_weights / np.sqrt(patch_size * patch_size * channels)
                patches[b, patch_idx] = np.dot(norm_weights, patch) / (patch_size * patch_size * channels)
                patch_idx += 1
    
    if not is_batched:
        patches = patches[0]  # Remove batch dimension for unbatched input
    
    return patches, (num_patches_h, num_patches_w)

def patch_embedding(image, patch_size, stride, projection_weights):
    """
    Wrapper function that currently uses the Python implementation.
    """
    return patch_embedding_python(image, patch_size, stride, projection_weights)

class PatchEmbedding:
    def __init__(self, patch_size: int, stride: int, projection_weights: np.ndarray):
        """
        Initialize the patch embedding layer.
        
        Args:
            patch_size: Size of patches to extract (square patches)
            stride: Stride between patches
            projection_weights: Weights for projecting patches of shape (output_channels, patch_size * patch_size * channels)
        """
        self.patch_size = patch_size
        self.stride = stride
        self.projection_weights = projection_weights
        
    def __call__(self, image: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int]]:
        """
        Extract patches from an image and project them through learned weights.
        
        Args:
            image: Input image of shape (batch_size, height, width, channels) or (height, width, channels)
            
        Returns:
            patches: Extracted and projected patches of shape (batch_size, num_patches, output_channels)
            (num_patches_h, num_patches_w): Number of patches in height and width dimensions
        """
        return patch_embedding(image, self.patch_size, self.stride, self.projection_weights) 