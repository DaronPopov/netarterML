/**
 * @file diffusion_wrapper.c
 * @brief Implementation of the C wrapper for Stable Diffusion inference
 */

#include "diffusion_wrapper.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>  // For nanosleep on POSIX systems

// Define the opaque context structure
struct DiffusionContext {
    void* model_data;          // Pointer to model data
    char* model_path;          // Path to the loaded model
    DiffusionPerfCallback perf_callback;  // Performance callback
    void* user_data;           // User data for callback
    bool initialized;          // Whether the context is initialized
};

// Implemented in python_bridge.c
extern int call_python_inference(
    const char* model_path,
    const char* prompt,
    int num_inference_steps,
    float guidance_scale,
    int width,
    int height,
    uint32_t seed,
    bool use_memory_optimizations,
    uint8_t** out_data,
    int* out_width,
    int* out_height,
    int* out_channels,
    DiffusionPerfCallback callback,
    void* user_data);

// Error message mapping
static const char* error_messages[] = {
    "Success",
    "Invalid argument",
    "Failed to load model",
    "Memory allocation failed",
    "Inference failed",
    "Feature not implemented",
    "Unknown error"
};

DiffusionError diffusion_initialize(void) {
    // Initialize any global resources
    // This is where we would initialize Python interpreter if needed
    // For now, we'll assume it's handled externally
    return DIFFUSION_SUCCESS;
}

DiffusionError diffusion_cleanup(void) {
    // Clean up any global resources
    return DIFFUSION_SUCCESS;
}

DiffusionError diffusion_create_context(const char* model_path, DiffusionContextHandle* context) {
    if (!model_path || !context) {
        return DIFFUSION_ERROR_INVALID_ARGUMENT;
    }
    
    // Allocate context
    DiffusionContext* ctx = (DiffusionContext*)malloc(sizeof(DiffusionContext));
    if (!ctx) {
        return DIFFUSION_ERROR_MEMORY_ALLOCATION;
    }
    
    // Initialize context
    memset(ctx, 0, sizeof(DiffusionContext));
    
    // Copy model path
    ctx->model_path = strdup(model_path);
    if (!ctx->model_path) {
        free(ctx);
        return DIFFUSION_ERROR_MEMORY_ALLOCATION;
    }
    
    // TODO: Preload model here if needed
    
    ctx->initialized = true;
    *context = ctx;
    
    return DIFFUSION_SUCCESS;
}

DiffusionError diffusion_destroy_context(DiffusionContextHandle context) {
    if (!context) {
        return DIFFUSION_ERROR_INVALID_ARGUMENT;
    }
    
    // Free resources
    if (context->model_path) {
        free(context->model_path);
    }
    
    // TODO: Free model data if loaded
    
    // Free context
    free(context);
    
    return DIFFUSION_SUCCESS;
}

DiffusionError diffusion_generate_image(
    DiffusionContextHandle context,
    const DiffusionParams* params,
    DiffusionImage* image) {
    
    if (!context || !params || !image) {
        return DIFFUSION_ERROR_INVALID_ARGUMENT;
    }
    
    if (!context->initialized) {
        return DIFFUSION_ERROR_MODEL_LOAD_FAILED;
    }
    
    // Initialize output image
    memset(image, 0, sizeof(DiffusionImage));
    
    // Generate seed if needed
    uint32_t seed = params->seed;
    if (seed == 0) {
        // Use time-based seed if not specified
        seed = (uint32_t)time(NULL);
    }
    
    // Call Python implementation
    int result = call_python_inference(
        context->model_path,
        params->prompt,
        params->num_inference_steps,
        params->guidance_scale,
        params->width,
        params->height,
        seed,
        params->use_memory_optimizations,
        &image->data,
        &image->width,
        &image->height,
        &image->channels,
        context->perf_callback,
        context->user_data);
    
    if (result != 0) {
        // Handle error
        return DIFFUSION_ERROR_INFERENCE_FAILED;
    }
    
    return DIFFUSION_SUCCESS;
}

void diffusion_free_image(DiffusionImage* image) {
    if (image && image->data) {
        free(image->data);
        image->data = NULL;
        image->width = 0;
        image->height = 0;
        image->channels = 0;
    }
}

const char* diffusion_error_string(DiffusionError error) {
    if (error < 0 || error > DIFFUSION_ERROR_UNKNOWN) {
        error = DIFFUSION_ERROR_UNKNOWN;
    }
    return error_messages[error];
}

DiffusionError diffusion_set_perf_callback(
    DiffusionContextHandle context,
    DiffusionPerfCallback callback,
    void* user_data) {
    
    if (!context) {
        return DIFFUSION_ERROR_INVALID_ARGUMENT;
    }
    
    context->perf_callback = callback;
    context->user_data = user_data;
    
    return DIFFUSION_SUCCESS;
} 