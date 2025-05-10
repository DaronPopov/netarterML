#ifndef ASM_KERNEL_ORCHESTRATOR_H
#define ASM_KERNEL_ORCHESTRATOR_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>  // For size_t

// Forward declarations
typedef struct ASMKernelContext ASMKernelContext;

// Buffer types
typedef enum {
    BUFFER_FP32,
    BUFFER_INT8,
    BUFFER_INT16,
    BUFFER_INT32
} BufferType;

// Kernel types
typedef enum {
    KERNEL_ATTENTION,
    KERNEL_CONVOLUTION,
    KERNEL_MATMUL,
    KERNEL_ACTIVATION
} KernelType;

// Buffer structure
typedef struct ASMKernelBuffer {
    void* data;
    size_t size;
    BufferType type;
} ASMKernelBuffer;

// Create and manage kernel context
ASMKernelContext* asm_create_context(void);
void asm_destroy_context(ASMKernelContext* ctx);

// Buffer management
ASMKernelBuffer* asm_create_buffer(ASMKernelContext* ctx, size_t size, BufferType type);
void asm_free_buffer(ASMKernelBuffer* buffer);

// Kernel execution
int asm_execute_kernel(
    ASMKernelContext* ctx,
    KernelType kernel_type,
    ASMKernelBuffer* input,
    ASMKernelBuffer* output,
    void* params);

// Memory management
void* asm_alloc_aligned(size_t size, size_t alignment);
void asm_free_aligned(void* ptr);

// SIMD operations
bool asm_is_simd_available(void);
void asm_enable_simd(ASMKernelContext* ctx);

// Performance monitoring
typedef struct {
    float execution_time_ms;
    uint64_t cycles;
    uint64_t flops;
} KernelMetrics;

void asm_get_kernel_metrics(ASMKernelContext* ctx, KernelMetrics* metrics);

// Initialize ASM kernels
ASMKernelContext* init_asm_kernels(void);

// Free ASM kernels
void free_asm_kernels(ASMKernelContext* ctx);

// Run text encoder
int run_text_encoder(ASMKernelContext* ctx, float* weights, const char* text, float** embeddings, size_t* embeddings_size);

// Run UNet
int run_unet(ASMKernelContext* ctx, float* weights, float* text_embeddings, size_t text_embeddings_size, float** latents, size_t* latents_size);

// Run VAE decoder
int run_vae_decoder(ASMKernelContext* ctx, float* weights, float* latents, size_t latents_size, float* output, size_t output_size);

#endif // ASM_KERNEL_ORCHESTRATOR_H 