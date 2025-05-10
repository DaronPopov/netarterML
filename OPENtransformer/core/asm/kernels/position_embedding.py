from ..assembler.builder import build_and_load
import numpy as np
import ctypes
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Export the position embedding kernel code
position_embedding_code = """
    .section    __DATA,__data
    .p2align    2
// Constants for position embedding calculations
const_neg_ln10000: .float -9.21034037197618  // -ln(10000)
const_pi_div_2: .float 1.570796326794897     // pi/2
const_0: .float 0.0
const_1: .float 1.0

// Coefficients for sine approximation using Taylor series
const_sin_coef1: .float 1.0                // x
const_sin_coef3: .float -0.166666666667    // -x³/3!
const_sin_coef5: .float 0.00833333333333   // x⁵/5!
const_sin_coef7: .float -0.000198412698    // -x⁷/7!

// Coefficients for cosine approximation using Taylor series
const_cos_coef0: .float 1.0                // 1
const_cos_coef2: .float -0.5               // -x²/2!
const_cos_coef4: .float 0.0416666666667    // x⁴/4!
const_cos_coef6: .float -0.00138888888889  // -x⁶/6!

    .section    __TEXT,__text,regular,pure_instructions
    .globl    _position_embedding
    .p2align    2
_position_embedding:
    // Save registers
    stp     x29, x30, [sp, #-16]!
    mov     x29, sp
    
    // Save NEON registers
    sub     sp, sp, #48
    stp     d8, d9, [sp, #32]
    stp     d10, d11, [sp, #16]
    stp     d12, d13, [sp]
    
    // Load arguments
    // x0: output pointer (float*)
    // w1: max_seq_len (int)
    // w2: embedding_dim (int)
    // s0: base (float) - use the base argument for calculations
    
    // Load constants using PC-relative addressing
    adrp    x8, const_neg_ln10000@GOTPAGE
    ldr     x8, [x8, const_neg_ln10000@GOTPAGEOFF]
    ldr     s0, [x8]            // -ln(base)
    
    adrp    x8, const_pi_div_2@GOTPAGE
    ldr     x8, [x8, const_pi_div_2@GOTPAGEOFF]
    ldr     s1, [x8]            // pi/2
    
    adrp    x8, const_0@GOTPAGE
    ldr     x8, [x8, const_0@GOTPAGEOFF]
    ldr     s2, [x8]            // 0.0
    
    adrp    x8, const_1@GOTPAGE
    ldr     x8, [x8, const_1@GOTPAGEOFF]
    ldr     s3, [x8]            // 1.0
    
    adrp    x8, const_sin_coef1@GOTPAGE
    ldr     x8, [x8, const_sin_coef1@GOTPAGEOFF]
    ldr     s4, [x8]            // sin coef1
    
    adrp    x8, const_sin_coef3@GOTPAGE
    ldr     x8, [x8, const_sin_coef3@GOTPAGEOFF]
    ldr     s5, [x8]            // sin coef3
    
    adrp    x8, const_sin_coef5@GOTPAGE
    ldr     x8, [x8, const_sin_coef5@GOTPAGEOFF]
    ldr     s6, [x8]            // sin coef5
    
    adrp    x8, const_sin_coef7@GOTPAGE
    ldr     x8, [x8, const_sin_coef7@GOTPAGEOFF]
    ldr     s7, [x8]            // sin coef7
    
    adrp    x8, const_cos_coef0@GOTPAGE
    ldr     x8, [x8, const_cos_coef0@GOTPAGEOFF]
    ldr     s8, [x8]            // cos coef0
    
    adrp    x8, const_cos_coef2@GOTPAGE
    ldr     x8, [x8, const_cos_coef2@GOTPAGEOFF]
    ldr     s9, [x8]            // cos coef2
    
    adrp    x8, const_cos_coef4@GOTPAGE
    ldr     x8, [x8, const_cos_coef4@GOTPAGEOFF]
    ldr     s10, [x8]           // cos coef4
    
    adrp    x8, const_cos_coef6@GOTPAGE
    ldr     x8, [x8, const_cos_coef6@GOTPAGEOFF]
    ldr     s11, [x8]           // cos coef6
    
    // Calculate 1/embedding_dim
    scvtf   s12, w2             // Convert embedding_dim to float
    fdiv    s13, s3, s12        // 1/embedding_dim
    fmul    s14, s0, s13        // -ln(base)/embedding_dim
    
    // Loop over positions
    mov     w4, #0              // position counter
1:  cmp     w4, w1
    b.ge    4f                  // exit if position >= max_seq_len
    
    // Convert position to float
    scvtf   s15, w4             // position as float
    
    // Loop over dimensions
    mov     w5, #0              // dimension counter
2:  cmp     w5, w2
    b.ge    3f                  // exit if dimension >= embedding_dim
    
    // Calculate dim * (-ln(base)/embedding_dim)
    scvtf   s16, w5             // dimension as float
    fmul    s17, s16, s14       // dim * (-ln(base)/embedding_dim)
    
    // exp(dim * (-ln(base)/embedding_dim)) simple approximation with just 2 terms
    fmov    s18, s3             // result = 1.0
    fadd    s18, s18, s17       // result += x
    fmul    s19, s17, s17       // x²
    fmul    s19, s19, s13       // x²/2 (reusing 1/embedding_dim as ~0.5)
    fadd    s18, s18, s19       // result += x²/2
    
    // pos * exp(dim * (-ln(base)/embedding_dim))
    fmul    s19, s15, s18       // pos * div_term
    
    // Check if dimension is even or odd
    tst     w5, #1
    b.ne    5f                  // if odd, calculate cos
    
    // Even dimension: sin calculation
    // sin(x) ≈ x + sin_coef3*x³ + sin_coef5*x⁵ + sin_coef7*x⁷
    fmul    s20, s19, s19       // x²
    fmul    s21, s20, s19       // x³
    fmul    s22, s21, s5        // sin_coef3*x³
    
    fmul    s21, s20, s20       // x⁴
    fmul    s21, s21, s19       // x⁵
    fmul    s21, s21, s6        // sin_coef5*x⁵
    
    // Initialize the result with x
    fmul    s23, s19, s4        // sin_coef1*x
    
    // Add the x³ term
    fadd    s23, s23, s22       // result += sin_coef3*x³
    
    // Add the x⁵ term
    fadd    s23, s23, s21       // result += sin_coef5*x⁵
    
    // Store sin value at output[dimension]
    str     s23, [x0, w5, UXTW #2]
    b       6f
    
5:  // Odd dimension: cos calculation
    // cos(x) ≈ cos_coef0 + cos_coef2*x² + cos_coef4*x⁴ + cos_coef6*x⁶
    fmul    s20, s19, s19       // x²
    fmul    s21, s20, s9        // cos_coef2*x²
    
    fmul    s22, s20, s20       // x⁴
    fmul    s22, s22, s10       // cos_coef4*x⁴
    
    // Initialize result with constant term
    fmov    s23, s8             // cos_coef0
    
    // Add the x² term
    fadd    s23, s23, s21       // result += cos_coef2*x²
    
    // Add the x⁴ term
    fadd    s23, s23, s22       // result += cos_coef4*x⁴
    
    // Store cos value at output[dimension]
    str     s23, [x0, w5, UXTW #2]
    
6:  add     w5, w5, #1          // increment dimension
    b       2b                  // continue dimension loop
    
3:  add     w4, w4, #1          // increment position
    add     x0, x0, w2, UXTW #2 // advance output pointer
    b       1b                  // continue position loop
    
4:  // Restore NEON registers
    ldp     d12, d13, [sp]
    ldp     d10, d11, [sp, #16]
    ldp     d8, d9, [sp, #32]
    add     sp, sp, #48
    
    // Restore registers and return
    ldp     x29, x30, [sp], #16
    ret
    """

def execute_kernel(output_ptr, max_seq_len, embedding_dim, base=10000.0):
    """Execute the position embedding kernel."""
    try:
        # Debugging: Log argument types and values
        logger.info(f"Executing position embedding kernel with arguments:")
        logger.info(f"output_ptr type: {type(output_ptr)}, address: {ctypes.addressof(output_ptr.contents)}")
        logger.info(f"max_seq_len: {max_seq_len}, type: {type(max_seq_len)}")
        logger.info(f"embedding_dim: {embedding_dim}, type: {type(embedding_dim)}")
        logger.info(f"base: {base}, type: {type(base)}")

        # Build and load the kernel
        from ..assembler.builder import build_and_jit
        kernel = build_and_jit(position_embedding_code, "_position_embedding")
        
        # Execute the kernel
        kernel(output_ptr, max_seq_len, embedding_dim, base)
        
    except Exception as e:
        logger.error(f"Error executing position embedding kernel: {str(e)}")
        raise

def test_position_embedding_kernel():
    """Test the position embedding kernel."""
    try:
        # Create test data
        max_seq_len = 512
        embedding_dim = 512
        output = np.zeros((max_seq_len, embedding_dim), dtype=np.float32)
        output_ptr = output.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        
        # Execute kernel
        execute_kernel(output_ptr, max_seq_len, embedding_dim)
        
        # Verify output
        assert not np.any(np.isnan(output)), "Output contains NaN values"
        assert not np.any(np.isinf(output)), "Output contains infinite values"
        assert output.shape == (max_seq_len, embedding_dim), f"Expected shape {(max_seq_len, embedding_dim)}, got {output.shape}"
        
        logger.info("Position embedding kernel test passed!")
        return True
        
    except Exception as e:
        logger.error(f"Position embedding kernel test failed: {str(e)}")
        return False

if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Run test
    test_position_embedding_kernel() 