#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "c_inference_engine.h"
#include "c_inference_internal.h"
#include "stb_image_write.h"
#include "asm_kernel_orchestrator.h"

// Internal context structure
struct ASMKernelContext {
    KernelMetrics last_metrics;
    uint64_t total_cycles;
    uint64_t total_flops;
    bool simd_available;
};

// Create a new inference context
InferenceContext* inference_create_context(const char* model_path) {
    InferenceContext* ctx = (InferenceContext*)malloc(sizeof(InferenceContext));
    if (!ctx) {
        return NULL;
    }

    // Initialize context
    memset(ctx, 0, sizeof(InferenceContext));
    
    // Allocate and initialize ASM kernel context
    ctx->asm_ctx = (ASMKernelContext*)malloc(sizeof(ASMKernelContext));
    if (!ctx->asm_ctx) {
        free(ctx);
        return NULL;
    }
    memset(ctx->asm_ctx, 0, sizeof(ASMKernelContext));

    // Allocate and initialize model weights
    ctx->weights = (ModelWeights*)malloc(sizeof(ModelWeights));
    if (!ctx->weights) {
        free(ctx->asm_ctx);
        free(ctx);
        return NULL;
    }
    memset(ctx->weights, 0, sizeof(ModelWeights));

    // Store model path
    if (model_path) {
        ctx->model_path = strdup(model_path);
        if (!ctx->model_path) {
            free(ctx->weights);
            free(ctx->asm_ctx);
            free(ctx);
            return NULL;
        }
    }

    return ctx;
}

// Save image data as PNG
int save_image_as_png(const char* filename, uint8_t* data, int width, int height, int channels) {
    if (!filename || !data || width <= 0 || height <= 0 || channels <= 0) {
        return -1;
    }

    // Ensure channels is 3 or 4
    if (channels != 3 && channels != 4) {
        return -1;
    }

    // Save the image using stb_image_write
    int result = stbi_write_png(filename, width, height, channels, data, width * channels);
    return result ? 0 : -1;
} 