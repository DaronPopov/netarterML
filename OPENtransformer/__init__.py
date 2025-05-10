"""
OPENtransformer - A comprehensive AI model framework
"""

__version__ = "0.1.0"

from .diffusion import EasyDiffusionAPI
from .chat import LLMAPI
from .vision import EasyImage
from .multimodal import MultimodalAnalysis

__all__ = [
    "EasyDiffusionAPI",
    "LLMAPI",
    "EasyImage",
    "MultimodalAnalysis",
]

"""
OPENtransformer package initialization
""" 