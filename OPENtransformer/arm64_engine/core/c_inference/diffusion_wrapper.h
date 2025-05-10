/**
 * @file diffusion_wrapper.h
 * @brief C wrapper for Stable Diffusion inference without Python overhead
 * 
 * This header provides a C API for running Stable Diffusion inference
 * directly from C code, avoiding Python interpreter overhead.
 */

#ifndef DIFFUSION_WRAPPER_H
#define DIFFUSION_WRAPPER_H

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Inference context handle (opaque pointer)
 */
typedef struct DiffusionContext DiffusionContext;
typedef struct DiffusionContext* DiffusionContextHandle;

/**
 * Image data structure
 */
typedef struct {
    uint8_t* data;        // RGB image data (HxWx3)
    int width;            // Image width
    int height;           // Image height
    int channels;         // Number of channels (typically 3 for RGB)
} DiffusionImage;

/**
 * Inference parameters
 */
typedef struct {
    const char* prompt;           // Text prompt for image generation
    int num_inference_steps;      // Number of denoising steps (e.g., 7-50)
    float guidance_scale;         // How closely to follow the prompt (e.g., 7.5)
    int width;                    // Output image width
    int height;                   // Output image height
    uint32_t seed;                // Random seed for reproducibility (0 = random)
    bool use_memory_optimizations; // Whether to use memory buffer optimizations
} DiffusionParams;

/**
 * Error codes
 */
typedef enum {
    DIFFUSION_SUCCESS = 0,
    DIFFUSION_ERROR_INVALID_ARGUMENT,
    DIFFUSION_ERROR_MODEL_LOAD_FAILED,
    DIFFUSION_ERROR_MEMORY_ALLOCATION,
    DIFFUSION_ERROR_INFERENCE_FAILED,
    DIFFUSION_ERROR_NOT_IMPLEMENTED,
    DIFFUSION_ERROR_UNKNOWN
} DiffusionError;

/**
 * Initialize the diffusion inference library
 * 
 * @return DIFFUSION_SUCCESS on success, error code otherwise
 */
DiffusionError diffusion_initialize(void);

/**
 * Clean up and release resources used by the diffusion inference library
 * 
 * @return DIFFUSION_SUCCESS on success, error code otherwise
 */
DiffusionError diffusion_cleanup(void);

/**
 * Create a new diffusion context by loading a model
 * 
 * @param model_path Path to the model directory or model ID
 * @param context Output pointer to receive the context handle
 * @return DIFFUSION_SUCCESS on success, error code otherwise
 */
DiffusionError diffusion_create_context(const char* model_path, DiffusionContextHandle* context);

/**
 * Release resources associated with a diffusion context
 * 
 * @param context Context handle to destroy
 * @return DIFFUSION_SUCCESS on success, error code otherwise
 */
DiffusionError diffusion_destroy_context(DiffusionContextHandle context);

/**
 * Generate an image using the provided diffusion context and parameters
 * 
 * @param context Diffusion context handle
 * @param params Generation parameters
 * @param image Output image (caller must free data with diffusion_free_image)
 * @return DIFFUSION_SUCCESS on success, error code otherwise
 */
DiffusionError diffusion_generate_image(
    DiffusionContextHandle context,
    const DiffusionParams* params,
    DiffusionImage* image);

/**
 * Free resources associated with an image
 * 
 * @param image Image to free
 */
void diffusion_free_image(DiffusionImage* image);

/**
 * Get a string description of an error code
 * 
 * @param error Error code
 * @return Human-readable error description
 */
const char* diffusion_error_string(DiffusionError error);

/**
 * Set a performance callback to receive metrics during inference
 * 
 * @param context Diffusion context handle
 * @param callback Function pointer to call with performance metrics
 * @param user_data User data pointer passed to callback
 * @return DIFFUSION_SUCCESS on success, error code otherwise
 */
typedef void (*DiffusionPerfCallback)(
    int step, int total_steps, float step_time, void* user_data);

DiffusionError diffusion_set_perf_callback(
    DiffusionContextHandle context,
    DiffusionPerfCallback callback,
    void* user_data);

#ifdef __cplusplus
}
#endif

#endif /* DIFFUSION_WRAPPER_H */ 