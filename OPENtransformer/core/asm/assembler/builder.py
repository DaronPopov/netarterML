import ctypes, mmap, os, subprocess, tempfile, time, sys
import numpy as np  # Add numpy import at the module level
import logging
from pathlib import Path
import re

logger = logging.getLogger(__name__)

def build_and_jit(asm_code: str, kernel_symbol: str):
    # Import all kernel codes from their respective files
    from OPENtransformer.core.asm.kernels.attention_matmul import attention_matmul_code
    from OPENtransformer.core.asm.kernels.layer_norm import layer_norm_code
    from OPENtransformer.core.asm.kernels.gelu import gelu_code
    from OPENtransformer.core.asm.kernels.softmax import softmax_code
    from OPENtransformer.core.asm.kernels.position_embedding import position_embedding_code
    from OPENtransformer.core.asm.kernels.transformer_forward import transformer_forward_code
    from OPENtransformer.core.asm.kernels.dropout import dropout_code
    from OPENtransformer.core.asm.kernels.dot_product import dot_product_code
    from OPENtransformer.core.asm.kernels.transpose import transpose_code
    from OPENtransformer.core.asm.kernels.fp16_to_fp32 import fp16_to_fp32_code
    from OPENtransformer.core.asm.kernels.fp32_to_fp16 import fp32_to_fp16_code
    from OPENtransformer.core.asm.kernels.rmsnorm import rmsnorm_code
    from OPENtransformer.core.asm.kernels.weight_initializer import weight_initializer_code
    from OPENtransformer.core.asm.kernels.matmul import matmul_code
    from OPENtransformer.core.asm.kernels.kv_cache_update import kv_cache_update_code
    from OPENtransformer.core.asm.kernels.tokenizer_kernel import tokenizer_kernel_code

    # Write asm_code to a temporary file.
    temp_dir = tempfile.mkdtemp()
    asm_path = os.path.join(temp_dir, "kernel.s")
    obj_path = os.path.join(temp_dir, "kernel.o")
    lib_path = os.path.join(temp_dir, "kernel.dylib" if sys.platform == 'darwin' else "kernel.so")
    
    with open(asm_path, "w") as f:
        f.write(asm_code)
    
    # Debugging: Display the assembly code
    print(f"Assembly code snippet to be compiled:")
    with open(asm_path, "r") as f:
        content = f.read()
        lines = content.split('\n')
        print('\n'.join(lines[:20]) + "\n...") # Print first 20 lines
    
    # Specifically look for the globl directive
    with open(asm_path, "r") as f:
        content = f.read()
        matches = re.findall(r'\.globl\s+([^\s]+)', content)
        print(f"Found .globl directives for symbols: {matches}")
    
    # Assemble to an object file for ARM64.
    try:
        result = subprocess.run(["as", "-v", "-arch", "arm64", asm_path, "-o", obj_path], 
                              capture_output=True, text=True)
        if result.returncode != 0:
            print("Assembly error output:")
            print(result.stderr)
            print("\nAssembly code:")
            with open(asm_path, "r") as f:
                print(f.read())
            raise subprocess.CalledProcessError(result.returncode, result.args, result.stdout, result.stderr)
    except subprocess.CalledProcessError as e:
        print("Assembly error output:")
        print(e.stderr)
        print("\nAssembly code:")
        with open(asm_path, "r") as f:
            print(f.read())
        raise

    # Create dynamic library
    try:
        if sys.platform == 'darwin':
            # Get SDK path for macOS
            sdk_path = subprocess.check_output([
                "xcrun",
                "--show-sdk-path"
            ]).decode().strip()
            
            # Use clang to create the dynamic library
            subprocess.check_call([
                "clang",
                "-dynamiclib",
                "-arch", "arm64",
                "-o", lib_path,
                obj_path,
                "-lSystem",
                "-isysroot", sdk_path
            ])
            print(f"Successfully created dynamic library: {lib_path}")
            print(f"Library contains: {subprocess.check_output(['nm', '-g', lib_path]).decode('utf-8')}")
        else:
            # For Linux/Unix, use gcc
            subprocess.check_call([
                "gcc",
                "-shared",
                "-o", lib_path,
                obj_path
            ])
            
        # Load the library using ctypes
        lib = ctypes.CDLL(lib_path)
        
        # Try to get the function with the exact symbol name
        try:
            # On macOS, the linker adds a leading underscore to all symbols
            # So we need to try both with and without the underscore
            symbol_to_try = kernel_symbol.lstrip('_')  # Remove any leading underscores
            try:
                func = getattr(lib, symbol_to_try)
                print(f"Successfully loaded function with symbol: {symbol_to_try}")
            except AttributeError:
                # Try with a leading underscore
                alt_symbol = '_' + symbol_to_try
                try:
                    func = getattr(lib, alt_symbol)
                    print(f"Successfully loaded function with alternative symbol: {alt_symbol}")
                except AttributeError:
                    raise AttributeError(f"Could not find symbol {symbol_to_try} or {alt_symbol} in library")
        except Exception as e:
            print(f"Error loading symbol: {e}")
            raise
        
        # Set appropriate function signatures for video decoder
        if kernel_symbol == "_video_decoder":
            func.argtypes = [
                ctypes.POINTER(ctypes.c_float),  # latent_video
                ctypes.POINTER(ctypes.c_float),  # output_video
                ctypes.c_int,                    # batch_size
                ctypes.c_int,                    # num_frames
                ctypes.c_int,                    # height
                ctypes.c_int,                    # width
                ctypes.c_int,                    # channels
                ctypes.c_float                   # scale_factor
            ]
            func.restype = ctypes.c_int
        
        return func
            
    except subprocess.CalledProcessError as e:
        print(f"Error creating dynamic library: {e}")
        print("Falling back to NumPy implementation")
        return None

import os
import subprocess
import tempfile
import ctypes
import platform
import uuid

def build_and_load(kernel_code, kernel_symbol):
    """Build and load a kernel from assembly code."""
    try:
        # Create temporary directory for build files
        with tempfile.TemporaryDirectory() as temp_dir:
            # Write kernel code to file
            kernel_s = os.path.join(temp_dir, "kernel.s")
            with open(kernel_s, "w") as f:
                f.write(kernel_code)
            
            # Assemble the kernel code
            obj_path = os.path.join(temp_dir, "kernel.o")
            subprocess.check_call([
                "as",
                "-arch", "arm64",
                "-o", obj_path,
                kernel_s
            ])
            
            # Get SDK path for macOS
            sdk_path = subprocess.check_output([
                "xcrun",
                "--show-sdk-path"
            ]).decode().strip()
            
            # Create a C file that references the symbols correctly for macOS
            c_wrapper_path = os.path.join(temp_dir, "wrapper.c")
            
            # The C wrapper will call the assembly function 
            with open(c_wrapper_path, "w") as f:
                f.write(f"""
                // C wrapper for assembly function
                #include <stdio.h>
                
                // Declare the assembly function (with leading underscore for macOS)
                extern void {kernel_symbol}(
                    float* latent_video,
                    float* output_video,
                    int batch_size,
                    int num_frames,
                    int height,
                    int width,
                    int channels,
                    float scale_factor
                );
                
                // Create a C function that can be called without leading underscore
                void wrapper_{kernel_symbol.lstrip('_').lstrip('_')}(
                    float* latent_video,
                    float* output_video,
                    int batch_size,
                    int num_frames,
                    int height,
                    int width,
                    int channels,
                    float scale_factor
                ) {{
                    // Forward the call to the assembly function
                    {kernel_symbol}(latent_video, output_video, batch_size, num_frames, height, width, channels, scale_factor);
                }}
                """)
            
            # Compile the C wrapper
            c_obj_path = os.path.join(temp_dir, "wrapper.o")
            subprocess.check_call([
                "clang",
                "-c", 
                "-o", c_obj_path,
                c_wrapper_path
            ])
            
            # Define lib_path before using it
            lib_path = os.path.join(temp_dir, f"lib{kernel_symbol[1:]}.dylib")
            
            # Link everything together using clang instead of ld
            subprocess.check_call([
                "clang",
                "-dynamiclib",
                "-o", lib_path,
                obj_path,
                c_obj_path,
                "-lSystem",
                "-syslibroot", sdk_path
            ])
            
            # Load the library
            lib = ctypes.CDLL(lib_path)
            
            # Get the wrapper function
            wrapper_func = getattr(lib, f"wrapper_{kernel_symbol.lstrip('_').lstrip('_')}")
            
            # Set argument types for video decoder
            wrapper_func.argtypes = [
                ctypes.POINTER(ctypes.c_float),  # latent_video
                ctypes.POINTER(ctypes.c_float),  # output_video
                ctypes.c_int,                    # batch_size
                ctypes.c_int,                    # num_frames
                ctypes.c_int,                    # height
                ctypes.c_int,                    # width
                ctypes.c_int,                    # channels
                ctypes.c_float                   # scale_factor
            ]
            wrapper_func.restype = None
            
            # Keep references to prevent garbage collection
            wrapper_func._lib = lib
            wrapper_func._lib_path = lib_path
            
            return wrapper_func
            
    except Exception as e:
        logger.error(f"Error building kernel: {str(e)}")
        return None

def get_kernel_code():
    """Get the kernel code."""
    return ""

def execute_kernel(*args):
    """Execute the kernel with the given arguments."""
    kernel = build_and_load(get_kernel_code(), "_kernel")
    if kernel:
        kernel(*args)
    else:
        raise RuntimeError("Failed to build and load kernel")

# Optional fallback for environments where compilation isn't possible
def get_fallback_implementation():
    return None

def run_with_timeout(func, *args, timeout=5):
    """Run a function with a timeout."""
    try:
        # Convert arguments
        converted_args = []
        for arg in args:
            if isinstance(arg, np.ndarray) and arg.dtype == np.float32:
                converted_args.append(arg.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))
            elif isinstance(arg, int):
                converted_args.append(ctypes.c_int(arg))
            elif isinstance(arg, float):
                converted_args.append(ctypes.c_float(arg))
            else:
                converted_args.append(arg)

        # Debugging: Log converted arguments
        logger.info(f"Converted arguments for kernel execution:")
        for i, arg in enumerate(converted_args):
            logger.info(f"Argument {i}: {arg}, type: {type(arg)}")

        # Execute the function
        result = func(*converted_args)

        # Debugging: Log result
        logger.info(f"Kernel execution result: {result}")

        return result
    except Exception as e:
        logger.error(f"Error in kernel execution: {str(e)}")
        raise