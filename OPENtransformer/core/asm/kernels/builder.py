import ctypes, mmap, os, subprocess, tempfile, time, sys
import numpy as np  # Add numpy import at the module level
import logging
from pathlib import Path
import re

logger = logging.getLogger(__name__)

def build_and_jit(asm_code: str, kernel_symbol: str):
    # Import all kernel codes from their respective files
    from .attention_matmul import attention_matmul_code
    from .layer_norm import layer_norm_code
    from .gelu import gelu_code
    from .softmax import softmax_code
    from .position_embedding import position_embedding_code
    from .transformer_forward import transformer_forward_code
    from .dropout import dropout_code
    from .dot_product import dot_product_code
    from .transpose import transpose_code
    from .fp16_to_fp32 import fp16_to_fp32_code
    from .fp32_to_fp16 import fp32_to_fp16_code
    from .rmsnorm import rmsnorm_code
    from .weight_initializer import weight_initializer_code
    from .matmul import matmul_code
    from .kv_cache_update import kv_cache_update_code
    from .tokenizer_kernel import tokenizer_kernel_code

    # Write asm_code to a temporary file.
    temp_dir = tempfile.mkdtemp()
    asm_path = os.path.join(temp_dir, "kernel.s")
    obj_path = os.path.join(temp_dir, "kernel.o")
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
    
    # Read the raw object file bytes.
    with open(obj_path, "rb") as f:
        obj_bytes = f.read()
    
    # Calculate aligned size
    page_size = os.sysconf("SC_PAGE_SIZE")
    size = len(obj_bytes)
    alloc_size = ((size + page_size - 1) // page_size) * page_size

    # Fall back to using dylib creation on macOS
    if sys.platform == 'darwin':
        lib_path = os.path.join(temp_dir, "kernel.dylib")
        
        print(f"Building library for symbol: {kernel_symbol}")
        print(f"Object file contains symbol: {subprocess.check_output(['nm', '-g', obj_path]).decode('utf-8')}")
        
        # Use clang to link the object file directly
        try:
            subprocess.check_call([
                "clang",
                "-dynamiclib",
                "-o", lib_path,
                obj_path,
                "-framework", "System"
            ])
            print(f"Successfully created dynamic library: {lib_path}")
            print(f"Library contains: {subprocess.check_output(['nm', '-g', lib_path]).decode('utf-8')}")
        except subprocess.CalledProcessError as e:
            print(f"Error creating dynamic library: {e}")
            raise
        
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
        
        # Set appropriate function signatures for transformer components
        if kernel_symbol == "_transformer_forward":
            func.argtypes = [
                ctypes.POINTER(ctypes.c_float),  # input tensor
                ctypes.POINTER(ctypes.c_float),  # output tensor
                ctypes.c_int,                    # batch_size
                ctypes.c_int,                    # seq_len
                ctypes.c_int,                    # d_model
                ctypes.c_int,                    # n_heads
                ctypes.c_int,                    # n_layers
                ctypes.POINTER(ctypes.POINTER(ctypes.c_float)),  # layer weights
                ctypes.POINTER(ctypes.c_float)   # position embeddings
            ]
            func.restype = None
        elif kernel_symbol == "_attention_matmul":
            func.argtypes = [
                ctypes.POINTER(ctypes.c_float),  # Q matrix
                ctypes.POINTER(ctypes.c_float),  # K matrix
                ctypes.POINTER(ctypes.c_float),  # Output matrix
                ctypes.c_int,                    # seq_len
                ctypes.c_int,                    # head_dim
                ctypes.c_int                     # num_heads
            ]
            func.restype = None
        elif kernel_symbol == "_layer_norm":
            func.argtypes = [
                ctypes.POINTER(ctypes.c_float),  # input (x0)
                ctypes.POINTER(ctypes.c_float),  # output (x1)
                ctypes.POINTER(ctypes.c_float),  # gamma weights (x2)
                ctypes.POINTER(ctypes.c_float),  # beta weights (x3)
                ctypes.c_int,                    # size (w4)
                ctypes.c_int                     # num_dims (w5)
            ]
            func.restype = None
        elif kernel_symbol == "_gelu":
            func.argtypes = [
                ctypes.POINTER(ctypes.c_float),  # input
                ctypes.POINTER(ctypes.c_float),  # output
                ctypes.c_int                     # size
            ]
            func.restype = None
        elif kernel_symbol == "_softmax":
            func.argtypes = [
                ctypes.POINTER(ctypes.c_float),  # input
                ctypes.POINTER(ctypes.c_float),  # output
                ctypes.c_int                     # size
            ]
            func.restype = None
        elif kernel_symbol == "_position_embedding":
            func.argtypes = [
                ctypes.POINTER(ctypes.c_float),  # output
                ctypes.c_int,                    # max_seq_len
                ctypes.c_int,                    # embedding_dim
                ctypes.c_float                   # base
            ]
            func.restype = None
        elif kernel_symbol == "_weight_initializer":
            func.argtypes = [
                ctypes.POINTER(ctypes.c_float),  # weights array
                ctypes.c_int,                    # features
                ctypes.c_int                     # classes
            ]
            func.restype = None
        elif kernel_symbol == "_fully_fused_transformer_layer":
            func.argtypes = [
                ctypes.POINTER(ctypes.c_float),  # input tensor
                ctypes.POINTER(ctypes.c_float),  # output tensor
                ctypes.POINTER(ctypes.c_float),  # layer_norm1_gamma
                ctypes.POINTER(ctypes.c_float),  # layer_norm1_beta
                ctypes.POINTER(ctypes.c_float),  # qkv_weights
                ctypes.POINTER(ctypes.c_float),  # attn_output_weights
                ctypes.POINTER(ctypes.c_float),  # layer_norm2_gamma
                ctypes.POINTER(ctypes.c_float),  # layer_norm2_beta
                ctypes.POINTER(ctypes.c_float),  # ff1_weights
                ctypes.POINTER(ctypes.c_float),  # ff2_weights
                ctypes.c_int,                    # batch_size
                ctypes.c_int,                    # seq_len
                ctypes.c_int,                    # d_model
                ctypes.c_int                     # n_heads
            ]
            func.restype = None
        
        # Create a wrapper that handles numpy arrays and argument conversion
        def wrapped_func(*args):
            try:
                # Convert numpy arrays to ctypes pointers
                converted_args = []
                for i, (arg, argtype) in enumerate(zip(args, func.argtypes)):
                    if isinstance(arg, np.ndarray):
                        if argtype == ctypes.POINTER(ctypes.c_float):
                            converted_args.append(arg.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))
                        elif argtype == ctypes.POINTER(ctypes.POINTER(ctypes.c_float)):
                            # Handle array of pointers to float arrays
                            ptr_array = (ctypes.POINTER(ctypes.c_float) * len(arg))()
                            for j, item in enumerate(arg):
                                if isinstance(item, np.ndarray):
                                    ptr_array[j] = item.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
                                else:
                                    ptr_array[j] = item
                            converted_args.append(ptr_array)
                        else:
                            converted_args.append(arg.ctypes.data_as(ctypes.c_void_p))
                    elif isinstance(arg, (int, np.integer)):
                        converted_args.append(ctypes.c_int(int(arg)))
                    elif isinstance(arg, (float, np.floating)):
                        converted_args.append(ctypes.c_float(float(arg)))
                    else:
                        converted_args.append(arg)
                
                # Execute the function
                if kernel_symbol == "_weight_initializer":
                    # For weight initializer, we need to handle the execution differently
                    from .weight_initializer import execute_kernel
                    execute_kernel(*converted_args)
                else:
                    result = func(*converted_args)
                    if result is not None:
                        return result
                
            except Exception as e:
                logger.error(f"Error executing kernel {kernel_symbol}: {str(e)}")
                raise
        
        return wrapped_func
    
    try:
        # First attempt: Try direct RWX mapping (least likely to work on modern macOS)
        flags = mmap.MAP_PRIVATE | mmap.MAP_ANON
        prot = mmap.PROT_READ | mmap.PROT_WRITE | mmap.PROT_EXEC
        mem = mmap.mmap(-1, alloc_size, flags=flags, prot=prot)
    except PermissionError:
        try:
            # Second attempt: Try creating RW mapping first
            flags = mmap.MAP_PRIVATE | mmap.MAP_ANON
            mem = mmap.mmap(-1, alloc_size, flags=flags,
                          prot=mmap.PROT_READ | mmap.PROT_WRITE)
        except PermissionError:
            # Final attempt: Try /dev/zero on Unix systems
            fd = os.open("/dev/zero", os.O_RDWR)
            try:
                mem = mmap.mmap(fd, alloc_size, flags=mmap.MAP_PRIVATE,
                              prot=mmap.PROT_READ | mmap.PROT_WRITE)
            finally:
                os.close(fd)

    # Write the object code
    mem.write(obj_bytes)
    
    # Get the memory address
    addr = ctypes.addressof(ctypes.c_char.from_buffer(mem))
    
    try:
        # Try to make the memory executable
        if sys.platform != 'darwin':
            libc = ctypes.CDLL(None)
            if libc.mprotect(ctypes.c_void_p(addr), alloc_size,
                           mmap.PROT_READ | mmap.PROT_EXEC) != 0:
                raise OSError("mprotect failed")
    except:
        # If we can't make it executable, fall back to using utils.execute_asm
        from utils import execute_asm
        return lambda *args: execute_asm(asm_code, *args)

    # Ensure loop termination in weight initializer
    if kernel_symbol == "_weight_initializer":
        if "b loop" not in asm_code:
            raise ValueError("Assembly code for weight_initializer is missing loop termination")

    # Create function prototype and return callable
    # Modify this section to handle arguments
    if kernel_symbol == "_weight_initializer":
        # Define prototype for weight_initializer(float*, int32, int32)
        prototype = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_int32, ctypes.c_int32)
    elif kernel_symbol == "_gelu":
        prototype = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int)
    elif kernel_symbol == "_matmul":
        prototype = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int)
    elif kernel_symbol == "_attention_matmul":
        prototype = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int)
    elif kernel_symbol == "_position_embedding":
        prototype = ctypes.CFUNCTYPE(None, 
            ctypes.POINTER(ctypes.c_float),  # output
            ctypes.c_int,                    # max_seq_len
            ctypes.c_int,                    # embedding_dim
            ctypes.c_float                   # base
        )
        func.restype = None
    elif kernel_symbol == "_attention_backprop":
        prototype = ctypes.CFUNCTYPE(ctypes.c_int,  # Return type is c_int for error codes
            ctypes.POINTER(ctypes.c_float),  # output_grad
            ctypes.POINTER(ctypes.c_float),  # q_mat
            ctypes.POINTER(ctypes.c_float),  # k_mat
            ctypes.POINTER(ctypes.c_float),  # v_mat
            ctypes.POINTER(ctypes.c_float),  # attention_scores
            ctypes.POINTER(ctypes.c_float),  # q_grad
            ctypes.POINTER(ctypes.c_float),  # k_grad
            ctypes.POINTER(ctypes.c_float),  # v_grad
            ctypes.POINTER(ctypes.c_float),  # debug_buffer
            ctypes.c_int,                    # seq_len
            ctypes.c_int,                    # head_dim
            ctypes.c_int                     # num_heads
        )
        func.restype = None
    elif "attention_backprop_tracking" in kernel_symbol:
        prototype = ctypes.CFUNCTYPE(ctypes.c_int,  # Return type is c_int for error codes
            ctypes.c_void_p,  # output gradient
            ctypes.c_void_p,  # Q matrix
            ctypes.c_void_p,  # K matrix
            ctypes.c_void_p,  # V matrix
            ctypes.c_void_p,  # attention scores
            ctypes.c_void_p,  # Q gradient
            ctypes.c_void_p,  # K gradient
            ctypes.c_void_p,  # V gradient
            ctypes.c_void_p,  # debug buffer
            ctypes.c_int,     # seq_len
            ctypes.c_int,     # head_dim
            ctypes.c_int      # num_heads
        )
        func.restype = None
    elif kernel_symbol == "_transformer_kernel":
        prototype = ctypes.CFUNCTYPE(None,
            ctypes.POINTER(ctypes.c_float),  # Q matrix
            ctypes.POINTER(ctypes.c_float),  # K matrix
            ctypes.POINTER(ctypes.c_float),  # V matrix
            ctypes.POINTER(ctypes.c_float),  # Output matrix
            ctypes.c_int,                    # num_heads
            ctypes.c_int                     # head_dim
        )
        func.restype = None
    elif kernel_symbol == "_dot_product":
        prototype = ctypes.CFUNCTYPE(None,
            ctypes.POINTER(ctypes.c_float),  # vector A
            ctypes.POINTER(ctypes.c_float),  # vector B
            ctypes.POINTER(ctypes.c_float),  # result
            ctypes.c_int                     # length
        )
        func.restype = None
    elif kernel_symbol == "_transpose":
        prototype = ctypes.CFUNCTYPE(None,
            ctypes.c_void_p,  # input matrix
            ctypes.c_void_p,  # output matrix
            ctypes.c_int,     # rows
            ctypes.c_int      # columns
        )
    elif kernel_symbol == "_tokenizer_kernel":
        prototype = ctypes.CFUNCTYPE(None,
            ctypes.c_void_p,  # input text (char*)
            ctypes.c_void_p,  # token IDs output (int*)
            ctypes.c_void_p,  # token boundaries (int*)
            ctypes.c_void_p,  # vocabulary (char**)
            ctypes.c_int,     # vocabulary size
            ctypes.c_int,     # max tokens
            ctypes.c_void_p,  # special tokens (char**)
            ctypes.c_int      # number of special tokens
        )
        func.restype = None
    elif kernel_symbol == "_position_embedding":
        prototype = ctypes.CFUNCTYPE(None, 
            ctypes.POINTER(ctypes.c_float),  # output
            ctypes.c_int,                    # max_seq_len
            ctypes.c_int,                    # embedding_dim
            ctypes.c_float                   # base
        )
        func.restype = None
    elif kernel_symbol == "_transformer_forward":
        prototype = ctypes.CFUNCTYPE(None,
            ctypes.POINTER(ctypes.c_float),  # input tensor
            ctypes.POINTER(ctypes.c_float),  # output tensor
            ctypes.c_int,                    # batch_size
            ctypes.c_int,                    # seq_len
            ctypes.c_int,                    # d_model
            ctypes.c_int,                    # n_heads
            ctypes.c_int,                    # n_layers
            ctypes.POINTER(ctypes.POINTER(ctypes.c_float)),  # layer weights
            ctypes.POINTER(ctypes.c_float)   # position embeddings
        )
        func.restype = None
    elif kernel_symbol == "_softmax":
        prototype = ctypes.CFUNCTYPE(None,
            ctypes.POINTER(ctypes.c_float),  # input array
            ctypes.POINTER(ctypes.c_float),  # output array
            ctypes.c_int                     # length
        )
        func.restype = None
    elif kernel_symbol == "_fully_fused_transformer_layer":
        prototype = ctypes.CFUNCTYPE(None,
            ctypes.POINTER(ctypes.c_float),  # input tensor
            ctypes.POINTER(ctypes.c_float),  # output tensor
            ctypes.POINTER(ctypes.c_float),  # layer_norm1_gamma
            ctypes.POINTER(ctypes.c_float),  # layer_norm1_beta
            ctypes.POINTER(ctypes.c_float),  # qkv_weights
            ctypes.POINTER(ctypes.c_float),  # attn_output_weights
            ctypes.POINTER(ctypes.c_float),  # layer_norm2_gamma
            ctypes.POINTER(ctypes.c_float),  # layer_norm2_beta
            ctypes.POINTER(ctypes.c_float),  # ff1_weights
            ctypes.POINTER(ctypes.c_float),  # ff2_weights
            ctypes.c_int,                    # batch_size
            ctypes.c_int,                    # seq_len
            ctypes.c_int,                    # d_model
            ctypes.c_int                     # n_heads
        )
        func.restype = None
    else:
        # Default to no-argument function
        prototype = ctypes.CFUNCTYPE(None)
    
    func = prototype(addr)
    func._mem = mem  # Prevent garbage collection
    
    print(f"JIT kernel {kernel_symbol} loaded at 0x{addr:x}")
    return func

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
                extern void {kernel_symbol}();
                
                // Create a C function that can be called without leading underscore
                void wrapper_{kernel_symbol[1:]}() {{
                    // Just forward the call to the assembly function
                    {kernel_symbol}();
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
            
            # Link everything together
            subprocess.check_call([
                "ld", 
                "-dylib", 
                "-o", lib_path, 
                obj_path,
                c_obj_path,
                "-lSystem", 
                "-syslibroot", sdk_path
            ])
            
            # Load the library
            lib = ctypes.CDLL(lib_path)
            
            # Get the wrapper function
            wrapper_func = getattr(lib, f"wrapper_{kernel_symbol[1:]}")
            wrapper_func.argtypes = [
                ctypes.POINTER(ctypes.c_float),
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_float
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