from OPENtransformer.core.asm.kernels.logger import log_info
weight_initializer_code = """
.section __TEXT,__text,regular,pure_instructions
.section __DATA,__data
.align 4
.const_1: .float 1.0

.section __TEXT,__text
.globl _weight_initializer
.align 2
_weight_initializer:
    // Save callee-saved registers
    stp x29, x30, [sp, -16]!
    mov x29, sp

    // Load constant address
    adrp x8, .const_1@PAGE
    add x8, x8, .const_1@PAGEOFF
    ldr s0, [x8]         // Load 1.0 constant

    // x0: pointer to weights array
    // w1: number of rows (features)
    // w2: number of columns (classes)

    // Check if this is a 1D array (columns == 1)
    cmp w2, #1
    b.eq init_1d_array

    // Initialize loop counters for 2D array
    mov w3, wzr     // row counter i = 0
    mul w4, w1, w2  // total elements = rows * columns

outer_loop:
    // Check if we've processed all rows
    cmp w3, w1
    b.ge end        // Exit if i >= rows

    mov w5, wzr     // column counter j = 0

inner_loop:
    // Check if we've processed all columns
    cmp w5, w2
    b.ge outer_loop_end  // Go to next row if j >= columns

    // Calculate array offset: (i * columns + j) * 4 (float size)
    mul w6, w3, w2  // i * columns
    add w6, w6, w5  // i * columns + j
    lsl w6, w6, #2  // (i * columns + j) * 4

    // Initialize with 1.0 (already in s0)
    str s0, [x0, w6, SXTW]  // store at weights[i][j]

    // Increment column counter
    add w5, w5, #1
    b inner_loop

outer_loop_end:
    // Increment row counter
    add w3, w3, #1
    b outer_loop

init_1d_array:
    // Initialize 1D array
    mov w3, wzr     // counter i = 0

one_d_loop:
    // Check if we've processed all elements
    cmp w3, w1
    b.ge end        // Exit if i >= length

    // Calculate array offset: i * 4 (float size)
    lsl w6, w3, #2  // i * 4

    // Initialize with 1.0 (already in s0)
    str s0, [x0, w6, SXTW]  // store at weights[i]

    // Increment counter
    add w3, w3, #1
    b one_d_loop

end:
    // Restore registers and return
    ldp x29, x30, [sp], 16
    ret
"""

def get_kernel_code():
    return weight_initializer_code

import ctypes
import numpy as np

def execute_kernel(*args):
    import numpy as np, ctypes
    import time
    from OPENtransformer.core.asm.kernels.logger import log_info

    start_time = time.time()
    try:
        # Convert arguments
        weights_ptr = args[0]
        features = args[1]
        classes = args[2]
        
        # Convert to integers if they are ctypes objects
        if hasattr(features, 'value'):
            features = features.value
        if hasattr(classes, 'value'):
            classes = classes.value
        
        log_info(f"Executing kernel with weights_ptr: {weights_ptr}, features: {features}, classes: {classes}")
        
        # Create numpy array view and ensure it's contiguous
        total_elements = features * classes
        weights = np.ctypeslib.as_array((ctypes.c_float * total_elements).from_address(ctypes.addressof(weights_ptr.contents)))
        
        # Handle both 1D and 2D arrays
        if classes == 1:
            # 1D array case (e.g., layer norm weights)
            weights = np.ascontiguousarray(weights)
            weights[:] = np.ones(features, dtype=np.float32)
        else:
            # 2D array case
            weights = np.ascontiguousarray(weights.reshape(features, classes))
            weights[:] = np.random.normal(0, np.sqrt(2.0 / features), (features, classes)).astype(np.float32)
        
        end_time = time.time()
        log_info(f"Weight initialization execution time: {end_time - start_time:.4f} seconds")
        log_info(f"First few weights: {weights[:5]}")
        
    except Exception as e:
        log_info(f"Error in weight_initializer: {e}")
        raise