#ifndef WEIGHT_CONVERTER_H
#define WEIGHT_CONVERTER_H

#include <stddef.h>
#include "c_inference_internal.h"

// Convert safetensors/pt to raw binary format
int convert_weights_to_binary(const char* input_path, const char* output_path);

// Load weights from binary file
int load_weights_from_binary(const char* path, ModelWeights* weights);

// Get tensor data by name
float* get_tensor_data(ModelWeights* weights, const char* name);

// Free model weights
void free_model_weights(ModelWeights* weights);

#endif // WEIGHT_CONVERTER_H 