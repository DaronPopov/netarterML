from ..assembler.builder import build_and_jit
import ctypes
import numpy as np
import logging
from .attention_matmul import execute_kernel as attention_matmul_kernel
from .layer_norm import execute_kernel as layer_norm_kernel
from .gelu import execute_kernel as gelu_kernel
from .softmax import execute_kernel as softmax_kernel

logger = logging.getLogger("assembly_test.transformer_forward")

transformer_forward_code = """
.section __TEXT,__text,regular,pure_instructions
.section __DATA,__data
.align 4
.const_0: .float 0.0
.const_1: .float 1.0
.const_epsilon: .float 0.0001
.const_neg_inf: .float -3.402823466e+38  // Negative FLT_MAX
.const_sqrt_d: .float 0.0442  // 1/sqrt(512) precomputed for d_model=512

.section __TEXT,__text
.globl _transformer_forward
.align 2
_transformer_forward:
    // Prologue
    stp x29, x30, [sp, -16]!
    mov x29, sp
    
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
    
    // Parameters:
    // x0: pointer to input tensor (batch_size, seq_len, d_model)
    // x1: pointer to output tensor (batch_size, seq_len, d_model)
    // w2: batch_size
    // w3: seq_len
    // w4: d_model
    // w5: n_heads
    // w6: n_layers
    // x7: pointer to layer weights array
    // x8: pointer to position embeddings
    
    // Safety checks
    cmp w2, #0
    b.le .done
    cmp w3, #0
    b.le .done
    cmp w4, #0
    b.le .done
    cmp w5, #0
    b.le .done
    cmp w6, #0
    b.le .done
    
    // Load constants
    adrp x9, .const_0@PAGE
    add x9, x9, .const_0@PAGEOFF
    ldr s7, [x9]  // 0.0
    
    adrp x9, .const_1@PAGE
    add x9, x9, .const_1@PAGEOFF
    ldr s8, [x9]  // 1.0
    
    adrp x9, .const_epsilon@PAGE
    add x9, x9, .const_epsilon@PAGEOFF
    ldr s9, [x9]  // epsilon
    
    adrp x9, .const_sqrt_d@PAGE
    add x9, x9, .const_sqrt_d@PAGEOFF
    ldr s10, [x9]  // 1/sqrt(d_model)
    
    // Calculate head_dim = d_model / n_heads
    udiv w10, w4, w5  // head_dim
    
    // For each batch
    mov w19, #0  // batch_idx = 0
.batch_loop:
    cmp w19, w2
    b.ge .batch_done
    
    // For each sequence position
    mov w20, #0  // seq_idx = 0
.seq_loop:
    cmp w20, w3
    b.ge .seq_done
    
    // Add position embeddings
    mul w21, w20, w4  // seq_idx * d_model
    lsl w21, w21, #2  // byte offset
    add x22, x0, x21  // input base address
    add x23, x8, x21  // position embeddings base address
    
    // For each feature
    mov w24, #0  // feature_idx = 0
.feature_loop:
    cmp w24, w4
    b.ge .feature_done
    
    ldr s0, [x22, w24, uxtw #2]    // input[batch_idx][seq_idx][feature_idx]
    ldr s1, [x23, w24, uxtw #2]    // position_embeddings[seq_idx][feature_idx]
    fadd s0, s0, s1               // Add position embedding
    str s0, [x22, w24, uxtw #2]    // Store back to input
    
    add w24, w24, #1
    b .feature_loop
    
.feature_done:
    add w20, w20, #1
    b .seq_loop
    
.seq_done:
    // Process through transformer layers
    mov w25, #0  // layer_idx = 0
.layer_loop:
    cmp w25, w6
    b.ge .layer_done
    
    // Get layer weights
    lsl w26, w25, #3  // layer_idx * 8 (size of pointer)
    add x26, x7, x26  // layer weights base address
    ldr x27, [x26]    // layer weights pointer
    
    // Layer normalization before attention
    // layer_norm_kernel(input_ptr, num_elements, output_ptr)
    bl _layer_norm
    
    // Multi-head attention
    // For each head
    mov w28, #0  // head_idx = 0
.head_loop:
    cmp w28, w5
    b.ge .head_done
    
    // Calculate offsets for Q, K, V projections
    mul w11, w28, w10  // head_idx * head_dim
    lsl w11, w11, #2  // byte offset
    
    // Project input to Q, K, V using attention_matmul_kernel
    // Q = input @ q_proj + q_bias
    bl _attention_matmul
    
    // K = input @ k_proj + k_bias
    bl _attention_matmul
    
    // V = input @ v_proj + v_bias
    bl _attention_matmul
    
    // Call attention_matmul_kernel for QK^T
    bl _attention_matmul
    
    // Scale attention scores by 1/sqrt(d_k)
    fmul s0, s0, s10
    
    // Apply softmax to attention scores
    bl _softmax
    
    // Call attention_matmul_kernel for (QK^T)V
    bl _attention_matmul
    
    add w28, w28, #1
    b .head_loop
    
.head_done:
    // Project output
    // output = output @ out_proj + out_bias
    bl _attention_matmul
    
    // Layer normalization after attention
    bl _layer_norm
    
    // Feed-forward network
    // 1. Linear projection
    bl _attention_matmul
    
    // 2. GELU activation
    bl _gelu
    
    // 3. Linear projection
    bl _attention_matmul
    
    // Layer normalization after feed-forward
    bl _layer_norm
    
    add w25, w25, #1
    b .layer_loop
    
.layer_done:
    add w19, w19, #1
    b .batch_loop
    
.batch_done:
    // Copy final output
    mov x11, x0  // src
    mov x12, x1  // dst
    mov w13, w2  // batch_size
    mul w13, w13, w3  // batch_size * seq_len
    mul w13, w13, w4  // batch_size * seq_len * d_model
    lsl w13, w13, #2  // byte count
    
    // Copy loop
    mov w14, #0  // i = 0
.copy_loop:
    cmp w14, w13
    b.ge .copy_done
    ldr s0, [x11, w14, uxtw #2]
    str s0, [x12, w14, uxtw #2]
    add w14, w14, #1
    b .copy_loop
.copy_done:
    
.done:
    // Restore non-volatile registers
    ldp d14, d15, [sp], #16
    ldp d12, d13, [sp], #16
    ldp d10, d11, [sp], #16
    ldp d8, d9, [sp], #16
    ldp x27, x28, [sp], #16
    ldp x25, x26, [sp], #16
    ldp x23, x24, [sp], #16
    ldp x21, x22, [sp], #16
    ldp x19, x20, [sp], #16
    
    // Epilogue
    ldp x29, x30, [sp], #16
    ret
"""

def get_kernel_code():
    """Returns the ARM assembly code for transformer forward pass kernel."""
    logger.debug("Returning transformer forward assembly code")
    return transformer_forward_code

def execute_kernel(input_ptr, output_ptr, batch_size, seq_len, d_model, n_heads, n_layers, layer_weights_ptr, pos_embeddings_ptr):
    """Execute the transformer forward pass kernel with safety checks."""
    try:
        # Create temporary buffers for intermediate computations
        temp_buffer1 = np.zeros((batch_size, seq_len, d_model), dtype=np.float32)
        temp_buffer2 = np.zeros((batch_size, seq_len, d_model), dtype=np.float32)
        temp_buffer1 = np.ascontiguousarray(temp_buffer1)
        temp_buffer2 = np.ascontiguousarray(temp_buffer2)
        
        # Get pointers to temporary buffers
        temp_ptr1 = temp_buffer1.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        temp_ptr2 = temp_buffer2.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        
        # Copy input to temp_buffer1
        temp_buffer1[:] = np.ctypeslib.as_array(input_ptr, shape=(batch_size, seq_len, d_model))
        
        # Process through transformer layers
        for layer_idx in range(n_layers):
            # Get layer weights
            layer_weights = layer_weights_ptr[layer_idx]
            
            # Layer normalization before attention
            layer_norm_kernel(temp_ptr1, batch_size * seq_len * d_model, temp_ptr2)
            
            # Multi-head attention
            head_dim = d_model // n_heads
            for head_idx in range(n_heads):
                # Project input to Q, K, V
                q_proj_ptr = layer_weights['q_proj'].ctypes.data_as(ctypes.POINTER(ctypes.c_float))
                v_proj_ptr = layer_weights['v_proj'].ctypes.data_as(ctypes.POINTER(ctypes.c_float))
                ff1_ptr = layer_weights['ff1'].ctypes.data_as(ctypes.POINTER(ctypes.c_float))
                ff2_ptr = layer_weights['ff2'].ctypes.data_as(ctypes.POINTER(ctypes.c_float))
                
                attention_matmul_kernel(temp_ptr2, q_proj_ptr, temp_ptr1,
                                     batch_size * seq_len, d_model, head_dim)
                
                # Scale attention scores
                temp_buffer1 *= 1.0 / np.sqrt(head_dim)
                
                # Apply softmax
                softmax_kernel(temp_ptr1, temp_ptr2, batch_size * seq_len)
                
                # Compute attention output
                attention_matmul_kernel(temp_ptr2, v_proj_ptr, temp_ptr1,
                                     batch_size * seq_len, d_model, head_dim)
            
            # Layer normalization after attention
            layer_norm_kernel(temp_ptr1, batch_size * seq_len * d_model, temp_ptr2)
            
            # Feed-forward network
            # First linear layer
            attention_matmul_kernel(temp_ptr2, ff1_ptr, temp_ptr1,
                                 batch_size * seq_len, d_model, 4 * d_model)
            
            # GELU activation
            gelu_kernel(temp_ptr1, temp_ptr2, batch_size * seq_len * 4 * d_model)
            
            # Second linear layer
            attention_matmul_kernel(temp_ptr2, ff2_ptr, temp_ptr1,
                                 batch_size * seq_len, 4 * d_model, d_model)
            
            # Layer normalization after feed-forward
            layer_norm_kernel(temp_ptr1, batch_size * seq_len * d_model, temp_ptr2)
            
            # Swap buffers for next layer
            temp_ptr1, temp_ptr2 = temp_ptr2, temp_ptr1
            temp_buffer1, temp_buffer2 = temp_buffer2, temp_buffer1
        
        # Copy final output
        output_array = np.ctypeslib.as_array(output_ptr, shape=(batch_size, seq_len, d_model))
        output_array[:] = temp_buffer1
        
    except Exception as e:
        logger.error(f"Error in forward pass: {e}")
        raise

def test_transformer_forward_kernel():
    """Test the transformer forward pass kernel."""
    try:
        # Set up logging
        logging.basicConfig(level=logging.DEBUG)
        logger.setLevel(logging.DEBUG)
        
        logger.info("Testing transformer forward kernel")
        
        # Test parameters
        batch_size = 2
        seq_len = 1024
        d_model = 4096
        n_heads = 16
        n_layers = 12
        
        # Create test data with proper alignment
        input_data = np.random.normal(0, 0.1, (batch_size, seq_len, d_model)).astype(np.float32)
        input_data = np.ascontiguousarray(input_data)
        output = np.zeros_like(input_data)
        output = np.ascontiguousarray(output)
        
        # Create position embeddings
        pos_embeddings = np.random.normal(0, 0.1, (seq_len, d_model)).astype(np.float32)
        pos_embeddings = np.ascontiguousarray(pos_embeddings)
        
        # Create layer weights (simplified for testing)
        layer_weights = []
        for _ in range(n_layers):
            layer = {
                'q_proj': np.random.normal(0, 0.1, (d_model, d_model)).astype(np.float32),
                'k_proj': np.random.normal(0, 0.1, (d_model, d_model)).astype(np.float32),
                'v_proj': np.random.normal(0, 0.1, (d_model, d_model)).astype(np.float32),
                'out_proj': np.random.normal(0, 0.1, (d_model, d_model)).astype(np.float32),
                'ff1': np.random.normal(0, 0.1, (d_model, 4*d_model)).astype(np.float32),
                'ff2': np.random.normal(0, 0.1, (4*d_model, d_model)).astype(np.float32)
            }
            layer_weights.append(layer)
        
        # Convert layer weights to contiguous arrays
        layer_weights = [np.ascontiguousarray(layer) for layer in layer_weights]
        
        # Get pointers
        input_ptr = input_data.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        output_ptr = output.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        pos_embeddings_ptr = pos_embeddings.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        
        # Create array of layer weight pointers
        layer_weights_array = (ctypes.POINTER(ctypes.c_float) * n_layers)()
        for i, layer in enumerate(layer_weights):
            layer_weights_array[i] = layer.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        
        # Execute kernel
        execute_kernel(
            input_ptr,
            output_ptr,
            batch_size,
            seq_len,
            d_model,
            n_heads,
            n_layers,
            layer_weights_array,
            pos_embeddings_ptr
        )
        
        # Basic validation
        if np.isnan(output).any():
            logger.error("Output contains NaN values")
            return False
        if np.isinf(output).any():
            logger.error("Output contains infinite values")
            return False
            
        logger.info("Transformer forward kernel test passed")
        return True
        
    except Exception as e:
        logger.error(f"Error in transformer forward kernel test: {str(e)}")
        return False

if __name__ == "__main__":
    # Run test
    success = test_transformer_forward_kernel()
    logger.info(f"Transformer forward kernel test {'passed' if success else 'failed'}") 