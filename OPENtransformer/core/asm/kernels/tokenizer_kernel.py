from OPENtransformer.core.asm.assembler.builder import build_and_load
import ctypes
import numpy as np
import logging
import time

logger = logging.getLogger("OPENtransformer.core.asm.tokenizer_kernel")

tokenizer_kernel_code = """
.section __TEXT,__text,regular,pure_instructions

// Data section for constants
.section __DATA,__data
.align 4
.const_0x80: .byte 0x80  // UTF-8 continuation byte mask
.const_0xE0: .byte 0xE0  // UTF-8 2-byte mask
.const_0xF0: .byte 0xF0  // UTF-8 3-byte mask
.const_0xF8: .byte 0xF8  // UTF-8 4-byte mask
.const_0xC0: .byte 0xC0  // UTF-8 continuation byte check
.const_0x3F: .byte 0x3F  // UTF-8 payload mask

.section __TEXT,__text
.globl _tokenizer_kernel
.align 2
_tokenizer_kernel:
    // Save registers
    stp x29, x30, [sp, -32]!
    mov x29, sp

    // Parameters:
    // x0: pointer to input text (char*)
    // x1: pointer to token IDs output (int*)
    // x2: pointer to token boundaries (int*)
    // x3: pointer to vocabulary (char**)
    // w4: vocabulary size
    // w5: max tokens
    // w6: pointer to special tokens (char**)
    // w7: number of special tokens

    // Save non-volatile registers
    stp x19, x20, [sp, -16]!
    stp x21, x22, [sp, -16]!
    stp x23, x24, [sp, -16]!
    stp x25, x26, [sp, -16]!
    stp x27, x28, [sp, -16]!

    // Initialize counters and pointers
    mov w19, #0  // token_count = 0
    mov x20, x0  // current_text_ptr = input_text
    mov x21, x1  // current_token_ptr = token_ids
    mov x22, x2  // current_boundary_ptr = token_boundaries

    // Main tokenization loop
tokenize_loop:
    // Check if we've reached max tokens
    cmp w19, w5
    b.ge tokenize_done

    // Load next byte
    ldrb w23, [x20]
    cbz w23, tokenize_done  // If byte is 0, we're done

    // Check if it's a continuation byte
    tst w23, #0x80
    b.ne check_utf8_start

    // ASCII character - process as single byte
    mov w25, #1  // char_length = 1
    b process_char

check_utf8_start:
    // Check UTF-8 sequence length
    and w24, w23, #0xE0
    cmp w24, #0xC0
    b.eq two_byte_char
    and w24, w23, #0xF0
    cmp w24, #0xE0
    b.eq three_byte_char
    and w24, w23, #0xF8
    cmp w24, #0xF0
    b.eq four_byte_char
    b invalid_char

two_byte_char:
    mov w25, #2
    b process_char

three_byte_char:
    mov w25, #3
    b process_char

four_byte_char:
    mov w25, #4
    b process_char

invalid_char:
    // Handle invalid UTF-8 sequence
    mov w25, #1  // Skip invalid byte
    b process_char

process_char:
    // Store character length in boundary array
    sxtw x24, w19  // Sign extend w19 to x24
    str w25, [x22, x24, lsl #2]
    add w19, w19, #1

    // Check if we've reached max tokens
    cmp w19, w5
    b.ge tokenize_done

    // Advance text pointer
    sxtw x24, w25  // Sign extend w25 to x24
    add x20, x20, x24
    b tokenize_loop

tokenize_done:
    // Store final token count at the start of token_ids array
    str w19, [x21]

    // Restore non-volatile registers
    ldp x27, x28, [sp], #16
    ldp x25, x26, [sp], #16
    ldp x23, x24, [sp], #16
    ldp x21, x22, [sp], #16
    ldp x19, x20, [sp], #16

    // Restore frame pointer and link register
    ldp x29, x30, [sp], #32
    ret
"""

def execute_kernel(input_text_ptr, token_ids_ptr, token_boundaries_ptr, vocab_ptr, vocab_size, max_tokens, special_tokens_ptr, num_special_tokens):
    """Execute the tokenizer kernel."""
    try:
        # Debugging: Log argument types and values
        logger.info(f"Executing tokenizer kernel with arguments:")
        logger.info(f"input_text_ptr type: {type(input_text_ptr)}, address: {ctypes.addressof(input_text_ptr.contents)}")
        logger.info(f"token_ids_ptr type: {type(token_ids_ptr)}, address: {ctypes.addressof(token_ids_ptr.contents)}")
        logger.info(f"token_boundaries_ptr type: {type(token_boundaries_ptr)}, address: {ctypes.addressof(token_boundaries_ptr.contents)}")
        logger.info(f"vocab_size: {vocab_size}, type: {type(vocab_size)}")
        logger.info(f"max_tokens: {max_tokens}, type: {type(max_tokens)}")
        logger.info(f"num_special_tokens: {num_special_tokens}, type: {type(num_special_tokens)}")

        # Build and load the kernel
        kernel = build_and_load(tokenizer_kernel_code, "_tokenizer_kernel")
        
        # Execute the kernel
        kernel(input_text_ptr, token_ids_ptr, token_boundaries_ptr, vocab_ptr, vocab_size, max_tokens, special_tokens_ptr, num_special_tokens)
        
    except Exception as e:
        logger.error(f"Error executing tokenizer kernel: {str(e)}")
        raise

def test_tokenizer_kernel():
    """Test the tokenizer kernel."""
    try:
        # Test with realistic input
        test_text = "Hello, world! 你好，世界！"
        max_tokens = 100
        
        # Allocate memory for output
        token_ids = np.zeros(max_tokens, dtype=np.int32)
        token_boundaries = np.zeros(max_tokens, dtype=np.int32)
        
        # Convert input text to bytes and create a ctypes array
        text_bytes = test_text.encode('utf-8')
        text_array = (ctypes.c_char * len(text_bytes))(*text_bytes)
        
        # Get pointers for kernel
        text_ptr = ctypes.cast(text_array, ctypes.c_void_p)
        token_ids_ptr = token_ids.ctypes.data_as(ctypes.c_void_p)
        boundaries_ptr = token_boundaries.ctypes.data_as(ctypes.c_void_p)
        
        # Dummy vocabulary and special tokens for testing
        vocab = np.array([""], dtype=np.object_)
        special_tokens = np.array([""], dtype=np.object_)
        
        vocab_ptr = vocab.ctypes.data_as(ctypes.c_void_p)
        special_tokens_ptr = special_tokens.ctypes.data_as(ctypes.c_void_p)

        logger.info("\nRunning tokenizer kernel test...")
        logger.info(f"Input text: {test_text}")
        logger.info(f"Max tokens: {max_tokens}")

        # Execute kernel
        execute_kernel(text_ptr, token_ids_ptr, boundaries_ptr, vocab_ptr, 
                      len(vocab), max_tokens, special_tokens_ptr, len(special_tokens))

        # Print results
        logger.info("\nTokenization results:")
        logger.info(f"Number of tokens: {token_ids[0]}")
        logger.info("Token boundaries:")
        for i in range(int(token_ids[0])):
            logger.info(f"Token {i}: {token_boundaries[i]} bytes")

        return True

    except Exception as e:
        logger.error(f"Error during kernel execution: {e}")
        return False

if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Run test
    test_tokenizer_kernel() 