"""
This module imports and re-exports all kernel codes from their respective files.
"""

import logging

logger = logging.getLogger("OPENtransformer.core.asm.kernel_codes")

__all__ = [
    'attention_matmul_code',
    'gelu_code',
    'softmax_code',
    'position_embedding_code',
    'transformer_forward_code',
    'dropout_code',
    'dot_product_code',
    'transpose_code',
    'fp16_to_fp32_code',
    'fp32_to_fp16_code',
    'rmsnorm_code',
    'weight_initializer_code',
    'matmul_code',
    'kv_cache_update_code',
    'tokenizer_kernel_code'
]

def get_kernel_code(kernel_name: str) -> str:
    """Get the assembly code for a specific kernel."""
    if kernel_name == "layer_norm":
        from OPENtransformer.core.asm.kernels.layer_norm import get_kernel_code
        return get_kernel_code()
    else:
        raise ValueError(f"Unknown kernel: {kernel_name}") 