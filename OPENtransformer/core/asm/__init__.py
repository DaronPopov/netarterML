"""
Assembly module initialization
"""

from .kernels.layer_norm import get_kernel_code as get_layer_norm_code
from .kernels.matmul import get_kernel_code as get_matmul_code
from .assembler.builder import build_and_load

__all__ = [
    'get_layer_norm_code',
    'get_matmul_code',
    'build_and_load'
] 