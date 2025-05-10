#ifndef C_INFERENCE_ENGINE_H
#define C_INFERENCE_ENGINE_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>
#include "c_inference_internal.h"

// Model configuration structures
typedef struct {
    int in_channels;
    int out_channels;
    int attention_head_dim;
} UNetConfig;

typedef struct {
    int latent_channels;
    float scaling_factor;
} VAEConfig;

typedef struct {
    int hidden_size;
    int intermediate_size;
    int num_attention_heads;
    int num_hidden_layers;
} TextEncoderConfig;

typedef struct {
    int num_train_timesteps;
    float beta_start;
    float beta_end;
} SchedulerConfig;

typedef struct {
    UNetConfig unet;
    VAEConfig vae;
    TextEncoderConfig text_encoder;
    SchedulerConfig scheduler;
} ModelConfig;

// Inference configuration
typedef struct {
    int width;
    int height;
    int num_inference_steps;
    float guidance_scale;
    uint32_t seed;
    bool use_memory_optimizations;
    const char* model_path;  // Path to model directory
} InferenceConfig;

// Create and manage inference context
InferenceContext* inference_create_context(const char* model_path);
void inference_destroy_context(InferenceContext* ctx);

// Load model weights
int inference_load_weights(InferenceContext* ctx, const char* weights_path);

// Run inference
int inference_generate_image(
    InferenceContext* ctx,
    const char* prompt,
    const InferenceConfig* config,
    uint8_t** out_data,
    int* out_width,
    int* out_height,
    int* out_channels
);

// Memory management
void inference_free_image(uint8_t* data);

// Performance monitoring
void inference_get_metrics(InferenceContext* ctx, InferenceMetrics* metrics);

#endif // C_INFERENCE_ENGINE_H 