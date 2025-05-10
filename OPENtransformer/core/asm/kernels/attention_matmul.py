from OPENtransformer.core.asm.assembler.builder import build_and_jit
import ctypes
import numpy as np
import logging
import time

logger = logging.getLogger("assembly_test.attention_matmul")

attention_matmul_code = """
.section __TEXT,__text,regular,pure_instructions

// Data section for constants
.section __DATA,__data
.align 4
.float_neg_max: .float -3.402823466e+38  // Negative FLT_MAX
.const_0: .float 0.0
.const_1: .float 1.0
.const_4: .float 4.0

.section __TEXT,__text
.globl _attention_matmul
.align 2
_attention_matmul:
    // Save registers
    stp x29, x30, [sp, -32]!
    mov x29, sp

    // Parameters:
    // x0: pointer to Q matrix (seq_len x head_dim)
    // x1: pointer to K matrix (seq_len x head_dim)
    // x2: pointer to output matrix (seq_len x seq_len)
    // w3: seq_len
    // w4: head_dim
    // w5: num_heads (unused in this kernel)

    // Save non-volatile registers
    stp x19, x20, [sp, -16]!
    stp x21, x22, [sp, -16]!
    stp x23, x24, [sp, -16]!
    stp x25, x26, [sp, -16]!
    stp x27, x28, [sp, -16]!
    stp d8, d9, [sp, -16]!
    stp d10, d11, [sp, -16]!
    stp d12, d13, [sp, -16]!
    stp d14, d15, [sp, -16]!

    // Load constants
    adrp x9, .const_1@PAGE
    add x9, x9, .const_1@PAGEOFF
    ldr s8, [x9]              // 1.0
    dup v8.4s, v8.s[0]       // Broadcast 1.0

    adrp x9, .const_0@PAGE
    add x9, x9, .const_0@PAGEOFF
    ldr s9, [x9]              // 0.0
    dup v9.4s, v9.s[0]       // Broadcast 0.0

    // Calculate 1/sqrt(head_dim)
    scvtf s10, w4             // Convert head_dim to float
    fsqrt s10, s10           // sqrt(head_dim)
    fdiv s10, s8, s10        // 1/sqrt(head_dim)
    dup v10.4s, v10.s[0]     // Broadcast scaling factor

    // For each row of Q
    mov w19, #0              // i = 0
1:  cmp w19, w3             // if i >= seq_len, exit
    b.ge 4f

    // For each row of K
    mov w20, #0              // j = 0
2:  cmp w20, w3             // if j >= seq_len, exit
    b.ge 3f

    // Initialize accumulator
    movi v11.4s, #0         // acc = 0.0

    // For each element in the row (process 4 elements at a time)
    mov w21, #0              // k = 0
5:  add w22, w21, #4        // Check if we can process 4 more elements
    cmp w22, w4             // if k + 4 > head_dim, handle remaining elements
    b.gt 7f

    // Calculate base addresses for Q and K
    mul w22, w19, w4        // i * head_dim
    add w22, w22, w21       // i * head_dim + k
    lsl w22, w22, #2        // byte offset
    add x24, x0, x22        // Q base address

    mul w23, w20, w4        // j * head_dim
    add w23, w23, w21       // j * head_dim + k
    lsl w23, w23, #2        // byte offset
    add x25, x1, x23        // K base address

    // Load 4 elements from Q and K using NEON
    ld1 {v12.4s}, [x24]     // Load 4 elements from Q[i][k:k+4]
    ld1 {v13.4s}, [x25]     // Load 4 elements from K[j][k:k+4]

    // Multiply and accumulate using NEON
    fmla v11.4s, v12.4s, v13.4s

    add w21, w21, #4        // k += 4
    b 5b

7:  // Handle remaining elements
    cmp w21, w4             // if k >= head_dim, store result
    b.ge 6f

    // Load and process one element at a time for remainder
    mul w22, w19, w4        // i * head_dim
    add w22, w22, w21       // i * head_dim + k
    lsl w22, w22, #2        // byte offset
    add x24, x0, x22        // Q base address
    ldr s12, [x24]          // Q[i][k]

    mul w23, w20, w4        // j * head_dim
    add w23, w23, w21       // j * head_dim + k
    lsl w23, w23, #2        // byte offset
    add x25, x1, x23        // K base address
    ldr s13, [x25]          // K[j][k]

    // Multiply and accumulate
    fmul s14, s12, s13      // Q[i][k] * K[j][k]
    fadd s11, s11, s14      // acc += Q[i][k] * K[j][k]

    add w21, w21, #1        // k++
    b 7b

6:  // Reduce vector sum and scale
    addv s11, v11.4s        // Sum all elements in vector
    fmul s11, s11, s10      // Scale by 1/sqrt(head_dim)

    // Store result
    mul w22, w19, w3        // i * seq_len
    add w22, w22, w20       // i * seq_len + j
    lsl w22, w22, #2        // byte offset
    add x24, x2, x22        // Output base address
    str s11, [x24]          // Store result

    add w20, w20, #1        // j++
    b 2b

3:  add w19, w19, #1        // i++
    b 1b

4:  // Restore registers and return
    ldp d14, d15, [sp], #16
    ldp d12, d13, [sp], #16
    ldp d10, d11, [sp], #16
    ldp d8, d9, [sp], #16
    ldp x27, x28, [sp], #16
    ldp x25, x26, [sp], #16
    ldp x23, x24, [sp], #16
    ldp x21, x22, [sp], #16
    ldp x19, x20, [sp], #16
    ldp x29, x30, [sp], #32
    ret
"""

def get_kernel_code():
    logger.debug("Returning attention_matmul assembly code")
    return attention_matmul_code

def build_kernel():
    """Build the attention matmul kernel."""
    kernel = build_and_jit(attention_matmul_code, "_attention_matmul")
    if kernel:
        kernel.argtypes = [
            ctypes.POINTER(ctypes.c_float),  # Q matrix
            ctypes.POINTER(ctypes.c_float),  # K matrix
            ctypes.POINTER(ctypes.c_float),  # Output matrix
            ctypes.c_int,     # seq_len
            ctypes.c_int,     # head_dim
            ctypes.c_int      # num_heads
        ]
        kernel.restype = None
    return kernel

def execute_kernel(q_ptr, k_ptr, output_ptr, seq_len, head_dim, num_heads):
    """Execute the attention matmul kernel with safety checks."""
    try:
        # Build kernel if not already built
        kernel = build_kernel()
        if not kernel:
            raise RuntimeError("Failed to build attention kernel")
            
        # Validate input pointers
        if not q_ptr or not k_ptr or not output_ptr:
            raise ValueError("Invalid input pointers")
            
        # Validate dimensions
        if seq_len <= 0 or head_dim <= 0 or num_heads <= 0:
            raise ValueError("Invalid dimensions")
            
        # Ensure numpy arrays are contiguous and of correct dtype
        q_array = np.ascontiguousarray(q_ptr, dtype=np.float32)
        k_array = np.ascontiguousarray(k_ptr, dtype=np.float32)
        output_array = np.ascontiguousarray(output_ptr, dtype=np.float32)

        # Convert to ctypes pointers
        q_ptr = q_array.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        k_ptr = k_array.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        output_ptr = output_array.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        # Log pointer types and addresses
        logger.debug(f"Q pointer type: {type(q_ptr)}, address: {ctypes.addressof(q_ptr.contents) if q_ptr else 'None'}")
        logger.debug(f"K pointer type: {type(k_ptr)}, address: {ctypes.addressof(k_ptr.contents) if k_ptr else 'None'}")
        logger.debug(f"Output pointer type: {type(output_ptr)}, address: {ctypes.addressof(output_ptr.contents) if output_ptr else 'None'}")

        # Verify memory alignment using base numpy array if available
        if isinstance(q_ptr, np.ndarray):
            logger.debug(f"Q alignment: {q_ptr.ctypes.data % 16}")
        if isinstance(k_ptr, np.ndarray):
            logger.debug(f"K alignment: {k_ptr.ctypes.data % 16}")
        if isinstance(output_ptr, np.ndarray):
            logger.debug(f"Output alignment: {output_ptr.ctypes.data % 16}")

        # Execute kernel
        kernel(q_ptr, k_ptr, output_ptr, seq_len, head_dim, num_heads)
        
        # Validate output
        output_shape = (seq_len, seq_len)
        output_arr = np.ctypeslib.as_array(output_ptr, shape=output_shape)
        if np.any(np.isnan(output_arr)) or np.any(np.isinf(output_arr)):
            raise ValueError("Output contains NaN or infinite values")
        
    except Exception as e:
        logger.error(f"Error executing attention kernel: {str(e)}")
        raise

def test_attention_matmul_kernel():
    """Test the attention matmul kernel."""
    try:
        # Parameters
        seq_len = 4
        head_dim = 64
        num_heads = 8
        
        # Create test matrices with proper alignment
        Q = np.ascontiguousarray(np.random.normal(0, 0.1, (seq_len, head_dim)).astype(np.float32))
        K = np.ascontiguousarray(np.random.normal(0, 0.1, (seq_len, head_dim)).astype(np.float32))
        output = np.ascontiguousarray(np.zeros((seq_len, seq_len), dtype=np.float32))
        
        # Get pointers
        Q_ptr = Q.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        K_ptr = K.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        output_ptr = output.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        
        # Execute kernel with safety wrapper
        execute_kernel(Q_ptr, K_ptr, output_ptr, seq_len, head_dim, num_heads)
        
        # Compute reference result
        ref_output = np.matmul(Q, K.T) / np.sqrt(head_dim)
        
        # Compare results
        max_diff = np.max(np.abs(output - ref_output))
        logger.info(f"Max difference from reference: {max_diff}")
        
        # Check for NaN or inf values
        if np.isnan(output).any():
            logger.error("Output contains NaN values")
            return False
        if np.isinf(output).any():
            logger.error("Output contains infinite values")
            return False
            
        return max_diff < 1e-5
        
    except Exception as e:
        logger.error(f"Error in attention kernel test: {str(e)}")
        return False

if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.DEBUG)
    
    # Run test
    success = test_attention_matmul_kernel()
    logger.info(f"Attention kernel test {'passed' if success else 'failed'}")
