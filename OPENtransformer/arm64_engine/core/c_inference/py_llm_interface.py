"""
Python interface for the ARM64 LLM inference engine.
"""

import os
import sys
import ctypes
from typing import Optional, Tuple, List
import numpy as np

# Load the C library
try:
    lib_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "libllm_inference.so")
    llm_lib = ctypes.CDLL(lib_path)
    print("Successfully loaded LLM inference library")
except Exception as e:
    print(f"Error loading LLM inference library: {e}")
    sys.exit(1)

# Define C types
class LLMContext(ctypes.Structure):
    _fields_ = [
        ("model_path", ctypes.c_char_p),
        ("max_context_length", ctypes.c_int),
        ("use_simd", ctypes.c_bool),
        ("use_memory_optimizations", ctypes.c_bool),
        ("temperature", ctypes.c_float),
        ("top_p", ctypes.c_float),
        ("repetition_penalty", ctypes.c_float),
        ("num_beams", ctypes.c_int),
        ("max_new_tokens", ctypes.c_int),
    ]

# Function signatures
llm_lib.llm_init_context.argtypes = [ctypes.POINTER(LLMContext)]
llm_lib.llm_init_context.restype = ctypes.c_void_p

llm_lib.llm_free_context.argtypes = [ctypes.c_void_p]
llm_lib.llm_free_context.restype = None

llm_lib.llm_load_model.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
llm_lib.llm_load_model.restype = ctypes.c_bool

llm_lib.llm_generate.argtypes = [
    ctypes.c_void_p,
    ctypes.c_char_p,
    ctypes.c_int,
    ctypes.c_float,
    ctypes.c_float,
    ctypes.c_float,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.POINTER(ctypes.c_char_p)
]
llm_lib.llm_generate.restype = ctypes.c_bool

def run_llm_inference(
    model_path: str,
    prompt: str,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.9,
    repetition_penalty: float = 1.2,
    num_beams: int = 1,
    use_simd: bool = True,
    use_memory_optimizations: bool = True
) -> Optional[str]:
    """
    Run inference using the ARM64 LLM engine.
    
    Args:
        model_path: Path to the model weights
        prompt: Input prompt
        max_new_tokens: Maximum number of tokens to generate
        temperature: Sampling temperature
        top_p: Top-p sampling parameter
        repetition_penalty: Penalty for repeating tokens
        num_beams: Number of beams for beam search
        use_simd: Whether to use SIMD optimizations
        use_memory_optimizations: Whether to use memory optimizations
        
    Returns:
        Generated text or None if failed
    """
    # Initialize context
    ctx = LLMContext(
        model_path=model_path.encode('utf-8'),
        max_context_length=2048,
        use_simd=use_simd,
        use_memory_optimizations=use_memory_optimizations,
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        num_beams=num_beams,
        max_new_tokens=max_new_tokens
    )
    
    # Create context
    context = llm_lib.llm_init_context(ctypes.byref(ctx))
    if not context:
        print("Failed to initialize LLM context")
        return None
    
    try:
        # Load model
        if not llm_lib.llm_load_model(context, model_path.encode('utf-8')):
            print("Failed to load model")
            return None
        
        # Generate response
        response_ptr = ctypes.c_char_p()
        success = llm_lib.llm_generate(
            context,
            prompt.encode('utf-8'),
            max_new_tokens,
            temperature,
            top_p,
            repetition_penalty,
            num_beams,
            use_simd,
            ctypes.byref(response_ptr)
        )
        
        if not success:
            print("Failed to generate response")
            return None
        
        # Get response
        response = response_ptr.value.decode('utf-8')
        
        return response
        
    finally:
        # Free context
        llm_lib.llm_free_context(context) 