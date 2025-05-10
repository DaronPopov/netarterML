#ifndef PYTHON_BRIDGE_H
#define PYTHON_BRIDGE_H

#include <Python.h>
#include "c_inference_engine.h"
#include "diffusion_types.h"

// Initialize Python interpreter and required modules
int init_python_environment(void);

// Initialize Python bridge
int init_python_bridge(void);

// Load model using diffusers library
PyObject* load_model_with_diffusers(const char* model_path);

// Prepare model for C inference by extracting weights
int prepare_model_for_c_inference(PyObject* pipeline, InferenceContext* ctx);

// Clean up Python environment
void cleanup_python_environment(void);

// Clean up Python bridge
void cleanup_python_bridge(void);

// Call Python inference function
int call_python_inference(
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

#endif // PYTHON_BRIDGE_H 