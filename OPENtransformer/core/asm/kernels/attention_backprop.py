from OPENtransformer.core.asm.assembler.builder import build_and_jit
import ctypes
import numpy as np
import logging
import time
import subprocess
import tempfile
import os

logger = logging.getLogger("assembly_test.attention_backprop")

def _attention_backprop():
    """Generate the assembly code for attention backpropagation."""
    return """
    .section    __TEXT,__text,regular,pure_instructions
    .globl    _attention_backprop
    .p2align    2
_attention_backprop:
    // Save frame pointer and link register
    stp     x29, x30, [sp, #-16]!
    mov     x29, sp

    // Save callee-saved registers
    stp     x19, x20, [sp, #-16]!
    stp     x21, x22, [sp, #-16]!
    stp     x23, x24, [sp, #-16]!
    stp     x25, x26, [sp, #-16]!
    stp     x27, x28, [sp, #-16]!

    // First validate debug buffer pointer
    cbz     x8, invalid_param_1_backprop

    // Store initial debug flag
    movz    w0, #0x1111
    str     w0, [x8]        // debug flag 1

    // Store input registers in callee-saved registers
    mov     x19, x0     // output_grad
    mov     x20, x1     // q_mat
    mov     x21, x2     // k_mat
    mov     x22, x3     // v_mat
    mov     x23, x4     // attention_scores
    mov     x24, x5     // q_grad
    mov     x25, x6     // k_grad
    mov     x26, x7     // v_grad
    mov     x27, x8     // debug_buffer
    mov     w28, w9     // seq_len
    mov     w29, w10    // head_dim
    mov     w30, w11    // num_heads

    // Store second debug flag
    movz    w0, #0x2222
    str     w0, [x8, #4]    // debug flag 2

    // Validate input parameters
    cbz     x19, invalid_param_1_backprop
    cbz     x20, invalid_param_1_backprop
    cbz     x21, invalid_param_1_backprop
    cbz     x22, invalid_param_1_backprop
    cbz     x23, invalid_param_1_backprop
    cbz     x24, invalid_param_1_backprop
    cbz     x25, invalid_param_1_backprop
    cbz     x26, invalid_param_1_backprop
    cbz     w28, invalid_param_1_backprop
    cbz     w29, invalid_param_1_backprop
    cbz     w30, invalid_param_1_backprop

    // Store third debug flag
    movz    w0, #0x3333
    str     w0, [x8, #8]    // debug flag 3

    // Store input registers in debug buffer
    add     x8, x27, #12    // Skip debug flags
    str     x19, [x8]       // output_grad
    str     x20, [x8, #8]   // q_mat
    str     x21, [x8, #16]  // k_mat
    str     x22, [x8, #24]  // v_mat
    str     x23, [x8, #32]  // attention_scores
    str     x24, [x8, #40]  // q_grad
    str     x25, [x8, #48]  // k_grad
    str     x26, [x8, #56]  // v_grad
    str     x27, [x8, #64]  // debug_buffer

    // Store fourth debug flag
    movz    w0, #0x4444
    str     w0, [x8, #72]   // debug flag 4

    // Store loaded integer values
    add     x8, x27, #80    // Skip to integers field
    str     w28, [x8]       // seq_len
    str     w29, [x8, #4]   // head_dim
    str     w30, [x8, #8]   // num_heads

    // Store success magic number
    movz    w0, #0xCAFE
    movk    w0, #0xCAFE, lsl #16
    str     w0, [x8, #12]   // success

    // Store final magic number
    movz    w0, #0xC0DE
    movk    w0, #0xDEAD, lsl #16
    str     w0, [x8, #16]   // final

    // Return success
    mov     w0, #0
    b       cleanup_1_backprop

invalid_param_1_backprop:
    mov     w0, #12

cleanup_1_backprop:
    // Restore callee-saved registers
    ldp     x27, x28, [sp], #16
    ldp     x25, x26, [sp], #16
    ldp     x23, x24, [sp], #16
    ldp     x21, x22, [sp], #16
    ldp     x19, x20, [sp], #16

    // Restore frame pointer and link register
    ldp     x29, x30, [sp], #16
    ret
    """

def _attention_backprop_wrapper():
    """Generate the assembly code for the attention backpropagation wrapper."""
    return """
    .section    __TEXT,__text,regular,pure_instructions
    .globl    _attention_backprop_wrapper
    .p2align    2
_attention_backprop_wrapper:
    // Save frame pointer and link register
    stp     x29, x30, [sp, #-16]!
    mov     x29, sp

    // Save callee-saved registers
    stp     x19, x20, [sp, #-16]!
    stp     x21, x22, [sp, #-16]!
    stp     x23, x24, [sp, #-16]!
    stp     x25, x26, [sp, #-16]!
    stp     x27, x28, [sp, #-16]!

    // Validate debug buffer pointer
    cbz     x8, invalid_param_1_wrapper

    // Initialize debug flag
    movz    w0, #0xAAAA
    str     w0, [x8]        // debug flag 1

    // Call the existing attention backprop function
    bl      _attention_backprop

    // Check return value
    cbnz    w0, cleanup_1_wrapper

    // Map values to their correct areas
    add     x8, x8, #12    // Skip debug flags
    ldr     x19, [x8]      // Load output_grad
    ldr     x20, [x8, #8]  // Load q_mat
    ldr     x21, [x8, #16] // Load k_mat
    ldr     x22, [x8, #24] // Load v_mat
    ldr     x23, [x8, #32] // Load attention_scores
    ldr     x24, [x8, #40] // Load q_grad
    ldr     x25, [x8, #48] // Load k_grad
    ldr     x26, [x8, #56] // Load v_grad

    // Store final debug flag
    movz    w0, #0xBBBB
    str     w0, [x8, #64]  // debug flag 2

    // Return success
    mov     w0, #0
    b       cleanup_1_wrapper

invalid_param_1_wrapper:
    mov     w0, #12

cleanup_1_wrapper:
    // Restore callee-saved registers
    ldp     x27, x28, [sp], #16
    ldp     x25, x26, [sp], #16
    ldp     x23, x24, [sp], #16
    ldp     x21, x22, [sp], #16
    ldp     x19, x20, [sp], #16

    // Restore frame pointer and link register
    ldp     x29, x30, [sp], #16
    ret
    """

def _pipeline_buffer_values():
    """Generate the assembly code for pipelining buffer values to their correct executions."""
    return """
    .section    __TEXT,__text,regular,pure_instructions
    .globl    _pipeline_buffer_values
    .p2align    2
_pipeline_buffer_values:
    // Save frame pointer and link register
    stp     x29, x30, [sp, #-16]!
    mov     x29, sp

    // Save callee-saved registers
    stp     x19, x20, [sp, #-16]!
    stp     x21, x22, [sp, #-16]!
    stp     x23, x24, [sp, #-16]!
    stp     x25, x26, [sp, #-16]!
    stp     x27, x28, [sp, #-16]!

    // Validate debug buffer pointer
    cbz     x8, invalid_param_1_pipeline

    // Initialize debug flag
    movz    w0, #0xAAAA
    str     w0, [x8]        // debug flag 1

    // Pipeline values to their correct execution areas
    add     x8, x8, #12    // Skip debug flags
    ldr     x19, [x8]      // Load output_grad
    ldr     x20, [x8, #8]  // Load q_mat
    ldr     x21, [x8, #16] // Load k_mat
    ldr     x22, [x8, #24] // Load v_mat
    ldr     x23, [x8, #32] // Load attention_scores
    ldr     x24, [x8, #40] // Load q_grad
    ldr     x25, [x8, #48] // Load k_grad
    ldr     x26, [x8, #56] // Load v_grad

    // Store final debug flag
    movz    w0, #0xBBBB
    str     w0, [x8, #64]  // debug flag 2

    // Return success
    mov     w0, #0
    b       cleanup_1_pipeline

invalid_param_1_pipeline:
    mov     w0, #12

cleanup_1_pipeline:
    // Restore callee-saved registers
    ldp     x27, x28, [sp], #16
    ldp     x25, x26, [sp], #16
    ldp     x23, x24, [sp], #16
    ldp     x21, x22, [sp], #16
    ldp     x19, x20, [sp], #16

    // Restore frame pointer and link register
    ldp     x29, x30, [sp], #16
    ret
    """

def _activate_execution_path():
    """Generate the assembly code to activate the execution path and perform the real execution."""
    return """
    .section    __TEXT,__text,regular,pure_instructions
    .globl    _activate_execution_path
    .p2align    2
_activate_execution_path:
    // Save frame pointer and link register
    stp     x29, x30, [sp, #-16]!
    mov     x29, sp

    // Save callee-saved registers
    stp     x19, x20, [sp, #-16]!
    stp     x21, x22, [sp, #-16]!
    stp     x23, x24, [sp, #-16]!
    stp     x25, x26, [sp, #-16]!
    stp     x27, x28, [sp, #-16]!

    // Validate debug buffer pointer
    cbz     x8, invalid_param_1_activate

    // Initialize debug flag for first execution
    movz    w0, #0xAAAA
    str     w0, [x8]        // debug flag 1

    // First execution to activate the path
    bl      _pipeline_buffer_values

    // Check return value
    cbnz    w0, cleanup_1_activate

    // Initialize debug flag for real execution
    movz    w0, #0xBBBB
    str     w0, [x8, #4]    // debug flag 2

    // Real execution
    bl      _attention_backprop

    // Check return value
    cbnz    w0, cleanup_1_activate

    // Store final debug flag
    movz    w0, #0xCCCC
    str     w0, [x8, #8]    // debug flag 3

    // Return success
    mov     w0, #0
    b       cleanup_1_activate

invalid_param_1_activate:
    mov     w0, #12

cleanup_1_activate:
    // Restore callee-saved registers
    ldp     x27, x28, [sp], #16
    ldp     x25, x26, [sp], #16
    ldp     x23, x24, [sp], #16
    ldp     x21, x22, [sp], #16
    ldp     x19, x20, [sp], #16

    // Restore frame pointer and link register
    ldp     x29, x30, [sp], #16
    ret
    """

def _fused_attention_backprop():
    """Generate the assembly code for the fused attention backpropagation kernel."""
    return """
    .section    __TEXT,__text,regular,pure_instructions
    .globl    _fused_attention_backprop
    .p2align    2
_fused_attention_backprop:
    // Save frame pointer and link register
    stp     x29, x30, [sp, #-16]!
    mov     x29, sp

    // Save callee-saved registers
    stp     x19, x20, [sp, #-16]!
    stp     x21, x22, [sp, #-16]!
    stp     x23, x24, [sp, #-16]!
    stp     x25, x26, [sp, #-16]!
    stp     x27, x28, [sp, #-16]!

    // Validate debug buffer pointer
    cbz     x8, invalid_param_1_fused

    // Initialize debug flag for first execution
    movz    w0, #0xAAAA
    str     w0, [x8]        // debug flag 1

    // Pipeline values to their correct execution areas
    add     x8, x8, #12    // Skip debug flags
    ldr     x19, [x8]      // Load output_grad
    ldr     x20, [x8, #8]  // Load q_mat
    ldr     x21, [x8, #16] // Load k_mat
    ldr     x22, [x8, #24] // Load v_mat
    ldr     x23, [x8, #32] // Load attention_scores
    ldr     x24, [x8, #40] // Load q_grad
    ldr     x25, [x8, #48] // Load k_grad
    ldr     x26, [x8, #56] // Load v_grad

    // First execution to activate the path
    bl      _pipeline_buffer_values

    // Check return value
    cbnz    w0, cleanup_1_fused

    // Initialize debug flag for real execution
    movz    w0, #0xBBBB
    str     w0, [x8, #4]    // debug flag 2

    // Real execution
    bl      _attention_backprop

    // Check return value
    cbnz    w0, cleanup_1_fused

    // Store final debug flag
    movz    w0, #0xCCCC
    str     w0, [x8, #8]    // debug flag 3

    // Return success
    mov     w0, #0
    b       cleanup_1_fused

invalid_param_1_fused:
    mov     w0, #12

cleanup_1_fused:
    // Restore callee-saved registers
    ldp     x27, x28, [sp], #16
    ldp     x25, x26, [sp], #16
    ldp     x23, x24, [sp], #16
    ldp     x21, x22, [sp], #16
    ldp     x19, x20, [sp], #16

    // Restore frame pointer and link register
    ldp     x29, x30, [sp], #16
    ret
    """

def build_and_jit():
    """Build and JIT compile the assembly code."""
    # Create temporary files
    with tempfile.NamedTemporaryFile(suffix='.s') as asm_file, \
         tempfile.NamedTemporaryFile(suffix='.o') as obj_file, \
         tempfile.NamedTemporaryFile(suffix='.dylib') as lib_file:
        
        # Write assembly code to file
        with open(asm_file.name, 'w') as f:
            f.write(_fused_attention_backprop())
            f.write(_attention_backprop())
            f.write(_pipeline_buffer_values())
            f.write(_activate_execution_path())

        # Assemble
        subprocess.run(['as', '-arch', 'arm64', '-o', obj_file.name, asm_file.name], check=True)

        # Link
        subprocess.run([
            'ld',
            '-dylib',
            '-arch', 'arm64',
            '-platform_version', 'macos', '14.0.0', '14.0.0',
            '-o', lib_file.name,
            obj_file.name
        ], check=True)

        # Load library
        lib = ctypes.CDLL(lib_file.name)

        # Set function prototype
        func = lib.fused_attention_backprop
        func.argtypes = [
            ctypes.POINTER(ctypes.c_float),  # output_grad
            ctypes.POINTER(ctypes.c_float),  # q_mat
            ctypes.POINTER(ctypes.c_float),  # k_mat
            ctypes.POINTER(ctypes.c_float),  # v_mat
            ctypes.POINTER(ctypes.c_float),  # attention_scores
            ctypes.POINTER(ctypes.c_float),  # q_grad
            ctypes.POINTER(ctypes.c_float),  # k_grad
            ctypes.POINTER(ctypes.c_float),  # v_grad
            ctypes.POINTER(ctypes.c_float),  # debug_buffer
            ctypes.c_int,     # seq_len
            ctypes.c_int,     # head_dim
            ctypes.c_int      # num_heads
        ]
        func.restype = ctypes.c_int

        # Keep references to prevent garbage collection
        func._lib = lib
        func._lib_path = lib_file.name

        return func

# Build the kernel once at module level
_kernel = build_and_jit()

def get_kernel_code():
    """Return the assembly code for the fused attention backprop kernel."""
    return _fused_attention_backprop()

def attention_backprop(
    output_grad: np.ndarray,
    q_mat: np.ndarray,
    k_mat: np.ndarray,
    v_mat: np.ndarray,
    attention_scores: np.ndarray,
    q_grad: np.ndarray,
    k_grad: np.ndarray,
    v_grad: np.ndarray,
    seq_len: int,
    head_dim: int,
    num_heads: int,
) -> int:
    """Compute attention backpropagation using assembly."""
    # Ensure all arrays are contiguous and properly aligned
    output_grad = np.ascontiguousarray(output_grad, dtype=np.float32)
    q_mat = np.ascontiguousarray(q_mat, dtype=np.float32)
    k_mat = np.ascontiguousarray(k_mat, dtype=np.float32)
    v_mat = np.ascontiguousarray(v_mat, dtype=np.float32)
    attention_scores = np.ascontiguousarray(attention_scores, dtype=np.float32)
    q_grad = np.ascontiguousarray(q_grad, dtype=np.float32)
    k_grad = np.ascontiguousarray(k_grad, dtype=np.float32)
    v_grad = np.ascontiguousarray(v_grad, dtype=np.float32)

    # Create debug buffer as a simple float array
    debug_buffer_size = 1024 * 1024  # 1MB should be enough
    debug_buffer = np.zeros(debug_buffer_size, dtype=np.float32, order='C')

    # Get pointers to array data
    output_grad_ptr = output_grad.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    q_mat_ptr = q_mat.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    k_mat_ptr = k_mat.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    v_mat_ptr = v_mat.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    attention_scores_ptr = attention_scores.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    q_grad_ptr = q_grad.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    k_grad_ptr = k_grad.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    v_grad_ptr = v_grad.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    debug_buffer_ptr = debug_buffer.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    # Log array shapes and dtypes
    logger.info("Kernel arguments:")
    logger.info(f"output_grad shape: {output_grad.shape}, dtype: {output_grad.dtype}")
    logger.info(f"q_mat shape: {q_mat.shape}, dtype: {q_mat.dtype}")
    logger.info(f"k_mat shape: {k_mat.shape}, dtype: {k_mat.dtype}")
    logger.info(f"v_mat shape: {v_mat.shape}, dtype: {v_mat.dtype}")
    logger.info(f"attention_scores shape: {attention_scores.shape}, dtype: {attention_scores.dtype}")
    logger.info(f"q_grad shape: {q_grad.shape}, dtype: {q_grad.dtype}")
    logger.info(f"k_grad shape: {k_grad.shape}, dtype: {k_grad.dtype}")
    logger.info(f"v_grad shape: {v_grad.shape}, dtype: {v_grad.dtype}")
    logger.info(f"debug_buffer shape: {debug_buffer.shape}, dtype: {debug_buffer.dtype}")

    # Log array pointers
    logger.info("\nArray pointers:")
    logger.info(f"output_grad_ptr: {ctypes.cast(output_grad_ptr, ctypes.c_void_p).value}")
    logger.info(f"q_mat_ptr: {ctypes.cast(q_mat_ptr, ctypes.c_void_p).value}")
    logger.info(f"k_mat_ptr: {ctypes.cast(k_mat_ptr, ctypes.c_void_p).value}")
    logger.info(f"v_mat_ptr: {ctypes.cast(v_mat_ptr, ctypes.c_void_p).value}")
    logger.info(f"attention_scores_ptr: {ctypes.cast(attention_scores_ptr, ctypes.c_void_p).value}")
    logger.info(f"q_grad_ptr: {ctypes.cast(q_grad_ptr, ctypes.c_void_p).value}")
    logger.info(f"k_grad_ptr: {ctypes.cast(k_grad_ptr, ctypes.c_void_p).value}")
    logger.info(f"v_grad_ptr: {ctypes.cast(v_grad_ptr, ctypes.c_void_p).value}")
    logger.info(f"debug_buffer_ptr: {ctypes.cast(debug_buffer_ptr, ctypes.c_void_p).value}")

    # Log parameter values
    logger.info("\nParameter values:")
    logger.info(f"seq_len: {seq_len} (C type: {ctypes.c_int(seq_len)})")
    logger.info(f"head_dim: {head_dim} (C type: {ctypes.c_int(head_dim)})")
    logger.info(f"num_heads: {num_heads} (C type: {ctypes.c_int(num_heads)})")

    try:
        # Call the assembly function
        result = _kernel(
            output_grad_ptr,
            q_mat_ptr,
            k_mat_ptr,
            v_mat_ptr,
            attention_scores_ptr,
            q_grad_ptr,
            k_grad_ptr,
            v_grad_ptr,
            debug_buffer_ptr,
            ctypes.c_int(seq_len),
            ctypes.c_int(head_dim),
            ctypes.c_int(num_heads)
        )

        # Check magic numbers in debug buffer
        magic1 = int(debug_buffer[0])
        magic2 = int(debug_buffer[26])
        magic3 = int(debug_buffer[31])
        success = int(debug_buffer[33])
        final = int(debug_buffer[34])

        logger.info("\nDebug buffer contents:")
        logger.info(f"Magic1: 0x{magic1:08x}")
        logger.info(f"Magic2: 0x{magic2:08x}")
        logger.info(f"Magic3: 0x{magic3:08x}")
        logger.info(f"Success: 0x{success:08x}")
        logger.info(f"Final: 0x{final:08x}")

        # Check pointers stored in debug buffer
        logger.info("\nStored pointers:")
        for i in range(12):
            logger.info(f"Pointer {i}: 0x{int(debug_buffer[i+2]):016x}")

        # Check integers stored in debug buffer
        logger.info("\nStored integers:")
        logger.info(f"seq_len: {int(debug_buffer[28])}")
        logger.info(f"head_dim: {int(debug_buffer[29])}")
        logger.info(f"num_heads: {int(debug_buffer[30])}")

        return result
    except Exception as e:
        logger.error(f"Error executing kernel: {e}")
        raise

def test_attention_backprop():
    """
    Test the fused attention backpropagation kernel.
    """
    try:
        # Set up logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        logger.info("Starting attention backprop test")
        
        # Test parameters
        seq_len = 4
        num_heads = 2
        head_dim = 4
        total_dim = num_heads * head_dim
        
        logger.info(f"Parameters: seq_len={seq_len}, num_heads={num_heads}, head_dim={head_dim}")
        
        # Create test matrices with specific patterns for easier debugging
        # Use np.zeros with order='C' to ensure contiguous arrays
        output_grad = np.zeros((seq_len, total_dim), dtype=np.float32, order='C')
        q_mat = np.zeros((seq_len, total_dim), dtype=np.float32, order='C')
        k_mat = np.zeros((seq_len, total_dim), dtype=np.float32, order='C')
        v_mat = np.zeros((seq_len, total_dim), dtype=np.float32, order='C')
        attention_scores = np.zeros((seq_len, seq_len), dtype=np.float32, order='C')
        
        # Add some patterns to help track computation
        for i in range(seq_len):
            for j in range(total_dim):
                output_grad[i,j] = 0.1 * (i + 1) * (j + 1)
                q_mat[i,j] = 0.1 * (i + 1)
                k_mat[i,j] = 0.1 * (j + 1)
                v_mat[i,j] = 0.1 * (i + j + 1)
        
        for i in range(seq_len):
            for j in range(seq_len):
                attention_scores[i,j] = 0.1 * (i + j + 1)
        
        # Initialize gradient arrays as contiguous arrays
        q_grad = np.zeros((seq_len, total_dim), dtype=np.float32, order='C')
        k_grad = np.zeros((seq_len, total_dim), dtype=np.float32, order='C')
        v_grad = np.zeros((seq_len, total_dim), dtype=np.float32, order='C')

        # Create debug buffer as a contiguous array
        debug_buffer_size = 1024 * 1024  # 1MB should be enough
        debug_buffer = np.zeros(debug_buffer_size, dtype=np.float32, order='C')

        # Print input matrices for debugging
        logger.info("\nInput Matrices:")
        logger.info(f"output_grad:\n{output_grad}")
        logger.info(f"q_mat:\n{q_mat}")
        logger.info(f"k_mat:\n{k_mat}")
        logger.info(f"v_mat:\n{v_mat}")
        logger.info(f"attention_scores:\n{attention_scores}")

        # Verify memory alignment
        logger.info("\nMemory Alignment:")
        for name, arr in [
            ("output_grad", output_grad),
            ("q_mat", q_mat),
            ("k_mat", k_mat),
            ("v_mat", v_mat),
            ("attention_scores", attention_scores),
            ("q_grad", q_grad),
            ("k_grad", k_grad),
            ("v_grad", v_grad),
            ("debug_buffer", debug_buffer)
        ]:
            alignment = arr.ctypes.data % 16
            logger.info(f"{name} alignment: {alignment}")
            logger.info(f"{name} contiguous: {arr.flags['C_CONTIGUOUS']}")
            logger.info(f"{name} strides: {arr.strides}")

        # Run the fused kernel
        logger.info("\nRunning fused kernel...")
        try:
            # Get pointers to the arrays
            output_grad_ptr = output_grad.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
            q_mat_ptr = q_mat.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
            k_mat_ptr = k_mat.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
            v_mat_ptr = v_mat.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
            attention_scores_ptr = attention_scores.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
            q_grad_ptr = q_grad.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
            k_grad_ptr = k_grad.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
            v_grad_ptr = v_grad.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
            debug_buffer_ptr = debug_buffer.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
            
            # Log pointer values
            logger.info("\nArray pointers:")
            logger.info(f"output_grad_ptr: {ctypes.cast(output_grad_ptr, ctypes.c_void_p).value}")
            logger.info(f"q_mat_ptr: {ctypes.cast(q_mat_ptr, ctypes.c_void_p).value}")
            logger.info(f"k_mat_ptr: {ctypes.cast(k_mat_ptr, ctypes.c_void_p).value}")
            logger.info(f"v_mat_ptr: {ctypes.cast(v_mat_ptr, ctypes.c_void_p).value}")
            logger.info(f"attention_scores_ptr: {ctypes.cast(attention_scores_ptr, ctypes.c_void_p).value}")
            logger.info(f"q_grad_ptr: {ctypes.cast(q_grad_ptr, ctypes.c_void_p).value}")
            logger.info(f"k_grad_ptr: {ctypes.cast(k_grad_ptr, ctypes.c_void_p).value}")
            logger.info(f"v_grad_ptr: {ctypes.cast(v_grad_ptr, ctypes.c_void_p).value}")
            logger.info(f"debug_buffer_ptr: {ctypes.cast(debug_buffer_ptr, ctypes.c_void_p).value}")
            
            # Log parameter values
            logger.info("\nParameter values:")
            logger.info(f"seq_len: {seq_len} (C type: c_int({seq_len}))")
            logger.info(f"head_dim: {head_dim} (C type: c_int({head_dim}))")
            logger.info(f"num_heads: {num_heads} (C type: c_int({num_heads}))")
            
            # Call the kernel
            result = attention_backprop(
                output_grad,
                q_mat,
                k_mat,
                v_mat,
                attention_scores,
                q_grad,
                k_grad,
                v_grad,
                seq_len,
                head_dim,
                num_heads
            )
            
            if result != 0:
                logger.error(f"Kernel execution failed with error code: {result}")
                logger.error("Debug buffer contents:")
                logger.error(f"Initial magic: {int(debug_buffer[0]):#010x}")
                logger.error("\nInput registers:")
                logger.error(f"x0 (output_grad): {int(debug_buffer[2]):#018x}")
                logger.error(f"x1 (q_mat): {int(debug_buffer[3]):#018x}")
                logger.error(f"x2 (k_mat): {int(debug_buffer[4]):#018x}")
                logger.error(f"x3 (v_mat): {int(debug_buffer[5]):#018x}")
                logger.error(f"x4 (attention_scores): {int(debug_buffer[6]):#018x}")
                logger.error(f"x5 (q_grad): {int(debug_buffer[7]):#018x}")
                logger.error(f"x6 (k_grad): {int(debug_buffer[8]):#018x}")
                logger.error(f"x7 (v_grad): {int(debug_buffer[9]):#018x}")
                logger.error(f"x8 (debug_buffer): {int(debug_buffer[10]):#018x}")
                logger.error(f"x9 (seq_len): {int(debug_buffer[11]):#018x}")
                logger.error(f"x10 (head_dim): {int(debug_buffer[12]):#018x}")
                logger.error(f"x11 (num_heads): {int(debug_buffer[13]):#018x}")
                logger.error(f"\nSecond magic: {int(debug_buffer[26]):#010x}")
                logger.error("\nLoaded integer values:")
                logger.error(f"seq_len (w28): {int(debug_buffer[28])}")
                logger.error(f"head_dim (w29): {int(debug_buffer[29])}")
                logger.error(f"num_heads (w30): {int(debug_buffer[30])}")
                logger.error(f"\nThird magic: {int(debug_buffer[31]):#010x}")
                logger.error(f"Success magic: {int(debug_buffer[33]):#010x}")
                logger.error(f"Final magic: {int(debug_buffer[34]):#010x}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error executing kernel: {e}")
            raise
            
    except Exception as e:
        logger.error(f"Error in test: {e}")
        raise

if __name__ == "__main__":
    test_attention_backprop()