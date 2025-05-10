"""
ARM64 assembly builder and JIT compiler.
"""

import os
import tempfile
import subprocess
import ctypes
import sys
from typing import Optional, Callable

def build_and_jit(asm_code: str, entry_point: str) -> Optional[Callable]:
    """
    Build and JIT compile ARM64 assembly code.
    
    Args:
        asm_code: The assembly code as a string
        entry_point: The name of the entry point function
        
    Returns:
        The compiled function if successful, None otherwise
    """
    try:
        # Create temporary files
        with tempfile.NamedTemporaryFile(suffix='.s', delete=False) as asm_file, \
             tempfile.NamedTemporaryFile(suffix='.o', delete=False) as obj_file, \
             tempfile.NamedTemporaryFile(suffix='.dylib' if sys.platform == 'darwin' else '.so', delete=False) as lib_file:
            
            # Write assembly code
            asm_file.write(asm_code.encode())
            asm_file.flush()
            
            # Assemble the code
            subprocess.run(['as', '-arch', 'arm64', '-o', obj_file.name, asm_file.name], check=True)
            
            # Create dynamic library
            if sys.platform == 'darwin':
                # Get SDK path for macOS
                sdk_path = subprocess.check_output([
                    "xcrun",
                    "--show-sdk-path"
                ]).decode().strip()
                
                # Use clang to create the dynamic library
                subprocess.run([
                    'clang',
                    '-dynamiclib',
                    '-arch', 'arm64',
                    '-o', lib_file.name,
                    obj_file.name,
                    '-lSystem',
                    '-isysroot', sdk_path
                ], check=True)
            else:
                # For Linux/Unix, use gcc
                subprocess.run([
                    'gcc',
                    '-shared',
                    '-o', lib_file.name,
                    obj_file.name
                ], check=True)
            
            # Load the dynamic library
            lib = ctypes.CDLL(lib_file.name)
            
            # Try to get the function with the exact symbol name
            try:
                # On macOS, the linker adds a leading underscore to all symbols
                # So we need to try both with and without the underscore
                symbol_to_try = entry_point.lstrip('_')  # Remove any leading underscores
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
            
            return func
            
    except Exception as e:
        print(f"Error building assembly: {e}")
        return None
    finally:
        # Clean up temporary files
        try:
            os.unlink(asm_file.name)
            os.unlink(obj_file.name)
            os.unlink(lib_file.name)
        except:
            pass 