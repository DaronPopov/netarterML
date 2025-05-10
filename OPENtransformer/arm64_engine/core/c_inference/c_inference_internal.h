#ifndef C_INFERENCE_INTERNAL_H
#define C_INFERENCE_INTERNAL_H

#include <stdbool.h>
#include <Python.h>
#include "asm_kernel_orchestrator.h"

// Structure to hold tensor metadata
typedef struct {
    char name[256];
    int dims[4];
    int num_dims;
    size_t offset;
    size_t size;
} TensorMetadata;

// Structure to hold model weights
typedef struct ModelWeights {
    PyObject* unet_weights;
    PyObject* vae_weights;
    PyObject* text_encoder_weights;
    float* unet_data;
    float* vae_data;
    float* text_encoder_data;
    size_t unet_size;
    size_t vae_size;
    size_t text_encoder_size;
    float* data;
    size_t total_size;
    TensorMetadata* metadata;
    int num_tensors;
} ModelWeights;

// Structure to hold inference metrics
typedef struct {
    int num_steps;
    float total_time_ms;
    float min_step_time_ms;
    float max_step_time_ms;
    float avg_step_time_ms;
} InferenceMetrics;

// Structure to hold inference context
typedef struct InferenceContext {
    ASMKernelContext* asm_ctx;
    ModelWeights* weights;
    bool weights_loaded;
    InferenceMetrics metrics;
    char* model_path;
} InferenceContext;

#endif // C_INFERENCE_INTERNAL_H 