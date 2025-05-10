"""
Layer normalization kernel for ARM64.
"""

import numpy as np
import ctypes
from OPENtransformer.core.asm.assembler.builder import build_and_jit
import logging

logger = logging.getLogger("OPENtransformer.core.asm.layer_norm")

layer_norm_code = """
    .section __TEXT,__text
    .global _layer_norm_asm
    .align 2
    
    _layer_norm_asm:
        // x0: input ptr
        // x1: output ptr
        // x2: gamma ptr
        // x3: beta ptr
        // x4: size
        // x5: epsilon
        
        // Save frame and registers
        stp x29, x30, [sp, #-16]!
        mov x29, sp
        stp x19, x20, [sp, #-16]!
        stp x21, x22, [sp, #-16]!
        
        // Check for null pointers
        cbz x0, error
        cbz x1, error
        cbz x2, error
        cbz x3, error
        
        // Check size
        cbz x4, error
        
        // Calculate mean
        mov x6, #0           // i = 0
        eor v0.16b, v0.16b, v0.16b  // sum = 0
        
    mean_loop:
        cmp x6, x4
        b.ge mean_loop_end
        
        // Load element
        ldr s1, [x0, x6, lsl #2]
        fadd s0, s0, s1      // sum += element
        
        add x6, x6, #1
        b mean_loop
        
    mean_loop_end:
        // Calculate mean
        scvtf s2, x4         // Convert size to float
        fdiv s3, s0, s2      // mean = sum / size
        
        // Calculate variance
        mov x6, #0           // i = 0
        eor v4.16b, v4.16b, v4.16b  // variance_sum = 0
        
    var_loop:
        cmp x6, x4
        b.ge var_loop_end
        
        // Load element
        ldr s6, [x0, x6, lsl #2]
        
        // Calculate (x - mean)^2
        fsub s7, s6, s3      // x - mean
        fmul s8, s7, s7      // (x - mean)^2
        
        // Add to variance sum
        fadd s4, s4, s8      // variance_sum += (x - mean)^2
        
        add x6, x6, #1
        b var_loop
        
    var_loop_end:
        // Calculate standard deviation
        fdiv s9, s4, s2      // variance = variance_sum / size
        fadd s10, s9, s5     // variance + epsilon
        fsqrt s11, s10       // std = sqrt(variance + epsilon)
        
        // Normalize and apply gamma/beta
        mov x6, #0           // i = 0
        
    norm_loop:
        cmp x6, x4
        b.ge norm_loop_end
        
        // Load input element
        ldr s12, [x0, x6, lsl #2]
        
        // Normalize: (x - mean) / std
        fsub s13, s12, s3    // x - mean
        fdiv s14, s13, s11   // (x - mean) / std
        
        // Load gamma and beta
        ldr s15, [x2, x6, lsl #2]  // gamma
        ldr s16, [x3, x6, lsl #2]  // beta
        
        // Apply scale and shift
        fmul s17, s14, s15   // normalized * gamma
        fadd s18, s17, s16   // (normalized * gamma) + beta
        
        // Store output
        str s18, [x1, x6, lsl #2]
        
        add x6, x6, #1
        b norm_loop
        
    norm_loop_end:
        // Return success
        mov x0, #1
        
        // Restore registers
        ldp x21, x22, [sp], #16
        ldp x19, x20, [sp], #16
        ldp x29, x30, [sp], #16
        ret
        
    error:
        // Return error
        mov x0, #0
        
        // Restore registers
        ldp x21, x22, [sp], #16
        ldp x19, x20, [sp], #16
        ldp x29, x30, [sp], #16
        ret
    """

def get_kernel_code() -> str:
    """Get the assembly code for the layer normalization kernel."""
    return layer_norm_code

class LayerNorm:
    def __init__(self, size: int, epsilon: float = 1e-5):
        """Initialize layer normalization kernel."""
        self._size = size
        self._epsilon = epsilon
        self._kernel = None
        
        try:
            self._kernel = build_and_jit(get_kernel_code(), "_layer_norm_asm")
            if self._kernel:
                self._kernel.argtypes = [
                    ctypes.POINTER(ctypes.c_float),  # input
                    ctypes.POINTER(ctypes.c_float),  # output
                    ctypes.POINTER(ctypes.c_float),  # gamma
                    ctypes.POINTER(ctypes.c_float),  # beta
                    ctypes.c_int,                    # size
                    ctypes.c_float                   # epsilon
                ]
                self._kernel.restype = ctypes.c_int
        except Exception as e:
            print(f"Warning: Failed to compile layer norm kernel: {e}")
            print("Falling back to NumPy implementation")
            
    def __call__(self, x: np.ndarray, gamma: np.ndarray, beta: np.ndarray) -> np.ndarray:
        """Apply layer normalization."""
        if not isinstance(x, np.ndarray) or not isinstance(gamma, np.ndarray) or not isinstance(beta, np.ndarray):
            raise TypeError("Inputs must be numpy arrays")
            
        if x.dtype != np.float32 or gamma.dtype != np.float32 or beta.dtype != np.float32:
            x = x.astype(np.float32)
            gamma = gamma.astype(np.float32)
            beta = beta.astype(np.float32)
            
        if x.size != self._size or gamma.size != self._size or beta.size != self._size:
            raise ValueError("Input arrays must match size")
            
        output = np.empty_like(x)
        
        if self._kernel is None:
            return self._numpy_layer_norm(x, gamma, beta)
            
        try:
            result = self._kernel(
                x.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                output.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                gamma.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                beta.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                ctypes.c_int(self._size),
                ctypes.c_float(self._epsilon)
            )
            if result != 1:
                raise RuntimeError("Layer norm kernel failed")
        except Exception as e:
            print(f"Warning: Layer norm kernel failed: {e}")
            return self._numpy_layer_norm(x, gamma, beta)
            
        return output
        
    def _numpy_layer_norm(self, x: np.ndarray, gamma: np.ndarray, beta: np.ndarray) -> np.ndarray:
        """NumPy implementation of layer normalization."""
        mean = np.mean(x)
        variance = np.var(x)
        std = np.sqrt(variance + self._epsilon)
        normalized = (x - mean) / std
        return normalized * gamma + beta

def execute_kernel(input_ptr, output_ptr, gamma_ptr=None, beta_ptr=None, num_elements=None):
    """Execute the layer normalization kernel with safety checks."""
    try:
        # Build kernel if not already built
        kernel = build_and_jit(get_kernel_code(), "_layer_norm_asm")
        if not kernel:
            raise RuntimeError("Failed to build layer norm kernel")
            
        # Validate input pointers
        if not input_ptr or not output_ptr:
            raise ValueError("Invalid input pointers")
            
        # Validate dimensions
        if num_elements is None and isinstance(input_ptr, ctypes.Array):
            # Try to infer num_elements if not provided
            num_elements = len(input_ptr)
        
        if num_elements is None or num_elements <= 0:
            raise ValueError("Invalid number of elements, must be provided")
        
        # Create default gamma and beta if not provided (1s and 0s respectively)
        if gamma_ptr is None:
            gamma = np.ones(num_elements, dtype=np.float32)
            gamma_ptr = gamma.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        
        if beta_ptr is None:
            beta = np.zeros(num_elements, dtype=np.float32)
            beta_ptr = beta.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        
        # Ensure num_elements is the correct type
        num_elements_c = ctypes.c_int(num_elements)
        num_dims_c = ctypes.c_int(1)  # Always 1 for simplicity
            
        # Set function types
        kernel.argtypes = [
            ctypes.POINTER(ctypes.c_float),  # input
            ctypes.POINTER(ctypes.c_float),  # output
            ctypes.POINTER(ctypes.c_float),  # gamma
            ctypes.POINTER(ctypes.c_float),  # beta
            ctypes.c_int,                    # num_elements (size)
            ctypes.c_int                     # num_dims (set to 1 for simplicity)
        ]
        kernel.restype = None
        
        # Log argument details for debugging
        logger.debug(f"Executing layer_norm kernel with:")
        logger.debug(f"input_ptr: {type(input_ptr)}, output_ptr: {type(output_ptr)}")
        logger.debug(f"gamma_ptr: {type(gamma_ptr)}, beta_ptr: {type(beta_ptr)}")
        logger.debug(f"num_elements: {num_elements_c.value}, num_dims: {num_dims_c.value}")
        
        # Execute kernel with proper argument types
        kernel(input_ptr, output_ptr, gamma_ptr, beta_ptr, num_elements_c, num_dims_c)
        
    except Exception as e:
        logger.error(f"Error executing layer norm kernel: {str(e)}")
        raise

def test_layer_norm_kernel():
    """
    Test function to verify that the layer_norm kernel works correctly
    """
    try:
        # Set up logging
        logging.basicConfig(level=logging.DEBUG)
        logger.setLevel(logging.DEBUG)
        
        logger.info("Testing layer_norm kernel")
        
        # Create test data with proper alignment
        feature_dim = 64  # Test with typical feature dimension
        test_data = np.random.normal(0, 1, feature_dim).astype(np.float32)
        test_data = np.ascontiguousarray(test_data)
        output = np.zeros_like(test_data)
        output = np.ascontiguousarray(output)
        
        # Create gamma and beta parameters
        gamma = np.ones(feature_dim, dtype=np.float32)  # Scaling factor
        beta = np.zeros(feature_dim, dtype=np.float32)  # Shift factor
        
        logger.debug(f"Test input shape: {test_data.shape}")
        logger.debug(f"Test input mean: {np.mean(test_data):.6f}, std: {np.std(test_data):.6f}")
        
        # Get numpy result for comparison
        mean = np.mean(test_data)
        std = np.std(test_data)
        expected = (test_data - mean) / (std + 1e-5)
        
        # Execute kernel with safety wrapper
        input_ptr = test_data.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        output_ptr = output.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        gamma_ptr = gamma.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        beta_ptr = beta.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        
        execute_kernel(input_ptr, output_ptr, gamma_ptr, beta_ptr, feature_dim)
        
        # Compare results
        max_diff = np.max(np.abs(output - expected))
        mean_diff = np.mean(np.abs(output - expected))
        
        logger.debug(f"Max difference from numpy: {max_diff}")
        logger.debug(f"Mean difference from numpy: {mean_diff}")
        logger.debug(f"First few input values: {test_data[:5]}")
        logger.debug(f"First few ASM output values: {output[:5]}")
        logger.debug(f"First few NumPy output values: {expected[:5]}")
        
        # Test if normalized output has correct properties
        output_mean = np.mean(output)
        output_std = np.std(output)
        expected_mean = np.mean(expected)
        expected_std = np.std(expected)
        
        logger.debug(f"ASM output stats - Mean: {output_mean:.6f}, Std: {output_std:.6f}")
        logger.debug(f"NumPy output stats - Mean: {expected_mean:.6f}, Std: {expected_std:.6f}")
        
        # Use a reasonable tolerance for floating point comparisons
        tolerance = 1e-3
        is_mean_ok = abs(output_mean) < tolerance
        is_std_ok = abs(output_std - 1.0) < tolerance
        is_close = max_diff < tolerance
        
        if not is_mean_ok:
            logger.error(f"Mean test failed: {output_mean} is not close to 0")
        if not is_std_ok:
            logger.error(f"Std test failed: {output_std} is not close to 1")
        if not is_close:
            logger.error(f"Max difference test failed: {max_diff} exceeds tolerance")
        
        success = is_mean_ok and is_std_ok and is_close
        if success:
            logger.info("All tests passed!")
        else:
            logger.error("Some tests failed!")
        
        return success
        
    except Exception as e:
        logger.error(f"Error in layer norm kernel test: {str(e)}")
        return False

if __name__ == "__main__":
    # Run test
    success = test_layer_norm_kernel()
    logger.info(f"Layer norm kernel test {'passed' if success else 'failed'}")