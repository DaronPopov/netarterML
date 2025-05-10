from OPENtransformer.core.asm.assembler.builder import build_and_jit
import numpy as np
import ctypes

kv_cache_update_code = """
.section __TEXT,__text,regular,pure_instructions
.globl _kv_cache_update
.align 2

// Function signature:
// void kv_cache_update(float* cache, float* new_kv, int cache_size, int new_size, int head_dim)
// x0: cache pointer
// x1: new_kv pointer
// w2: cache_size (number of elements in cache)
// w3: new_size (number of elements in new_kv)
// w4: head_dim (dimension of each head)

_kv_cache_update:
    // Save registers
    stp x29, x30, [sp, -16]!
    mov x29, sp
    
    // Save non-volatile registers
    stp x19, x20, [sp, -16]!
    stp x21, x22, [sp, -16]!
    stp x23, x24, [sp, -16]!
    stp d8, d9, [sp, -16]!
    stp d10, d11, [sp, -16]!
    
    // Calculate offsets
    lsl w5, w2, #2  // cache_offset = cache_size * 4 (float size)
    lsl w6, w3, #2  // new_offset = new_size * 4 (float size)
    
    // Calculate number of elements to copy
    mov w7, w3  // num_elements = new_size
    
    // Check if we have enough space in cache
    add w8, w2, w3  // total_size = cache_size + new_size
    cmp w8, w4
    b.hi cache_overflow
    
    // Main copy loop using NEON SIMD
    lsr w9, w7, #3  // num_vectors = num_elements / 8
    and w10, w7, #7  // remaining = num_elements % 8
    
    // Add offset to cache pointer
    add x0, x0, x5
    
    // Copy vectors of 8 floats at a time
    cbz w9, skip_vector_copy
vector_copy_loop:
    ldp q0, q1, [x1], #32  // Load 8 floats from new_kv
    stp q0, q1, [x0], #32  // Store 8 floats to cache
    subs w9, w9, #1
    b.ne vector_copy_loop
skip_vector_copy:
    
    // Copy remaining elements
    cbz w10, copy_done
remaining_copy_loop:
    ldr s0, [x1], #4  // Load single float
    str s0, [x0], #4  // Store single float
    subs w10, w10, #1
    b.ne remaining_copy_loop
copy_done:
    
    // Restore registers and return
    ldp d10, d11, [sp], #16
    ldp d8, d9, [sp], #16
    ldp x23, x24, [sp], #16
    ldp x21, x22, [sp], #16
    ldp x19, x20, [sp], #16
    ldp x29, x30, [sp], #16
    ret

cache_overflow:
    // Handle cache overflow error
    // You might want to implement proper error handling here
    // For now, we'll just return
    ldp d10, d11, [sp], #16
    ldp d8, d9, [sp], #16
    ldp x23, x24, [sp], #16
    ldp x21, x22, [sp], #16
    ldp x19, x20, [sp], #16
    ldp x29, x30, [sp], #16
    ret
"""

def get_kernel_code():
    return kv_cache_update_code

def execute_kernel(cache, new_kv, cache_size, new_size, head_dim):
    """
    Execute the KV cache update kernel.
    
    Args:
        cache: Pointer to the cache array (float*)
        new_kv: Pointer to the new key/value tensor (float*)
        cache_size: Current size of the cache (number of elements)
        new_size: Size of the new key/value tensor (number of elements)
        head_dim: Dimension of each attention head
    
    Returns:
        None
    """
    # Get the kernel function
    kernel = get_kernel()
    
    # Execute the kernel
    kernel(cache, new_kv, cache_size, new_size, head_dim)

def get_kernel():
    """Get the JIT-compiled kernel function."""
    return build_and_jit(get_kernel_code(), "_kv_cache_update")

def update_kv_cache(cache, new_kv, head_dim):
    """
    High-level wrapper to update the KV cache.
    
    Args:
        cache: NumPy array containing the cache
        new_kv: NumPy array containing new key/value data
        head_dim: Dimension of each attention head
    
    Returns:
        Updated cache array
    """
    # Convert inputs to ctypes pointers
    cache_ptr = cache.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    new_kv_ptr = new_kv.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    
    # Get sizes
    cache_size = cache.size
    new_size = new_kv.size
    
    # Execute the kernel
    execute_kernel(cache_ptr, new_kv_ptr, cache_size, new_size, head_dim)
    
    return cache

def test_kv_cache_update():
    """
    Test the KV cache update kernel against NumPy operations.
    """
    # Test parameters
    num_heads = 4
    head_dim = 64
    max_seq_len = 1024
    new_seq_len = 1
    
    # Create test data
    cache = np.zeros((max_seq_len, num_heads, head_dim), dtype=np.float32)
    new_kv = np.random.randn(new_seq_len, num_heads, head_dim).astype(np.float32)
    
    # Create a copy for NumPy comparison
    cache_numpy = cache.copy()
    
    # Update using our kernel
    cache_kernel = update_kv_cache(cache, new_kv, head_dim)
    
    # Update using NumPy
    cache_numpy[:new_seq_len] = new_kv
    
    # Compare results
    max_diff = np.max(np.abs(cache_kernel - cache_numpy))
    mean_diff = np.mean(np.abs(cache_kernel - cache_numpy))
    
    print(f"KV Cache Update Test Results:")
    print(f"Max difference: {max_diff:.6f}")
    print(f"Mean difference: {mean_diff:.6f}")
    print(f"Test {'passed' if max_diff < 1e-6 else 'failed'}")
    
    # Additional validation
    print("\nAdditional Validation:")
    print(f"Cache shape: {cache_kernel.shape}")
    print(f"New KV shape: {new_kv.shape}")
    print(f"First few elements of cache after update:")
    print(cache_kernel[:new_seq_len, 0, :5])  # Show first 5 elements of first head
    print("\nFirst few elements of new KV:")
    print(new_kv[0, 0, :5])  # Show first 5 elements of first head
    
    return max_diff < 1e-6

if __name__ == "__main__":
    test_kv_cache_update() 