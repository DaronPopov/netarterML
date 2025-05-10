#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>
#include "weight_converter.h"
#include "c_inference_internal.h"

// Convert tensor to binary format
static int convert_tensor_to_binary(PyObject* tensor, FILE* fp) {
    if (!tensor) {
        fprintf(stderr, "Invalid tensor object\n");
        return -1;
    }

    // Get tensor name and shape
    PyObject* name = PyObject_GetAttrString(tensor, "name");
    if (!name) {
        fprintf(stderr, "Failed to get tensor name\n");
        return -1;
    }

    PyObject* shape = PyObject_GetAttrString(tensor, "shape");
    if (!shape) {
        fprintf(stderr, "Failed to get tensor shape\n");
        Py_DECREF(name);
        return -1;
    }

    // Convert tensor to numpy array
    PyObject* array = PyObject_GetAttrString(tensor, "numpy");
    if (!array) {
        fprintf(stderr, "Failed to convert tensor to numpy array\n");
        Py_DECREF(name);
        Py_DECREF(shape);
        return -1;
    }

    // Get array data
    PyArrayObject* arr = (PyArrayObject*)array;
    void* tensor_data = PyArray_DATA(arr);
    size_t data_size = PyArray_NBYTES(arr);

    // Write tensor metadata
    const char* tensor_name = PyUnicode_AsUTF8(name);
    fwrite(tensor_name, strlen(tensor_name) + 1, 1, fp);
    
    int ndim = PyArray_NDIM(arr);
    fwrite(&ndim, sizeof(int), 1, fp);
    
    npy_intp* dims = PyArray_DIMS(arr);
    fwrite(dims, sizeof(npy_intp), ndim, fp);
    
    // Write tensor data
    fwrite(tensor_data, 1, data_size, fp);

    // Cleanup
    Py_DECREF(name);
    Py_DECREF(shape);
    Py_DECREF(array);

    return 0;
}

// Load weights from binary file
int load_weights_from_binary(const char* path, ModelWeights* weights) {
    FILE* fp = fopen(path, "rb");
    if (!fp) {
        fprintf(stderr, "Failed to open weights file: %s\n", path);
        return -1;
    }

    // Read UNet weights
    size_t unet_size;
    fread(&unet_size, sizeof(size_t), 1, fp);
    weights->unet_data = (float*)malloc(unet_size);
    if (!weights->unet_data) {
        fclose(fp);
        return -1;
    }
    weights->unet_size = unet_size;
    fread(weights->unet_data, 1, unet_size, fp);

    // Read VAE weights
    size_t vae_size;
    fread(&vae_size, sizeof(size_t), 1, fp);
    weights->vae_data = (float*)malloc(vae_size);
    if (!weights->vae_data) {
        free(weights->unet_data);
        fclose(fp);
        return -1;
    }
    weights->vae_size = vae_size;
    fread(weights->vae_data, 1, vae_size, fp);

    // Read text encoder weights
    size_t text_encoder_size;
    fread(&text_encoder_size, sizeof(size_t), 1, fp);
    weights->text_encoder_data = (float*)malloc(text_encoder_size);
    if (!weights->text_encoder_data) {
        free(weights->unet_data);
        free(weights->vae_data);
        fclose(fp);
        return -1;
    }
    weights->text_encoder_size = text_encoder_size;
    fread(weights->text_encoder_data, 1, text_encoder_size, fp);

    fclose(fp);
    return 0;
}

// Convert weights to binary format
int convert_weights_to_binary(const char* input_path, const char* output_path) {
    // Import numpy
    if (_import_array() < 0) {
        fprintf(stderr, "Failed to import numpy array API\n");
        return -1;
    }

    // Open output file
    FILE* fp = fopen(output_path, "wb");
    if (!fp) {
        fprintf(stderr, "Failed to open output file: %s\n", output_path);
        return -1;
    }

    // Load state dict
    PyObject* torch = PyImport_ImportModule("torch");
    if (!torch) {
        fprintf(stderr, "Failed to import torch module\n");
        fclose(fp);
        return -1;
    }

    PyObject* load_fn = PyObject_GetAttrString(torch, "load");
    if (!load_fn) {
        fprintf(stderr, "Failed to get torch.load function\n");
        Py_DECREF(torch);
        fclose(fp);
        return -1;
    }

    PyObject* state_dict = PyObject_CallFunction(load_fn, "s", input_path);
    if (!state_dict) {
        fprintf(stderr, "Failed to load state dict from: %s\n", input_path);
        Py_DECREF(load_fn);
        Py_DECREF(torch);
        fclose(fp);
        return -1;
    }

    // Convert each tensor
    PyObject* key;
    PyObject* value;
    Py_ssize_t pos = 0;

    while (PyDict_Next(state_dict, &pos, &key, &value)) {
        if (convert_tensor_to_binary(value, fp) != 0) {
            fprintf(stderr, "Failed to convert tensor\n");
            Py_DECREF(state_dict);
            Py_DECREF(load_fn);
            Py_DECREF(torch);
            fclose(fp);
            return -1;
        }
    }

    // Cleanup
    Py_DECREF(state_dict);
    Py_DECREF(load_fn);
    Py_DECREF(torch);
    fclose(fp);

    return 0;
}

// Load binary weights into memory
ModelWeights* load_binary_weights(const char* path) {
    FILE* fp = fopen(path, "rb");
    if (!fp) return NULL;

    ModelWeights* weights = (ModelWeights*)malloc(sizeof(ModelWeights));
    if (!weights) {
        fclose(fp);
        return NULL;
    }

    // Read number of tensors
    fread(&weights->num_tensors, sizeof(Py_ssize_t), 1, fp);

    // Allocate metadata array
    weights->metadata = (TensorMetadata*)malloc(weights->num_tensors * sizeof(TensorMetadata));
    if (!weights->metadata) {
        free(weights);
        fclose(fp);
        return NULL;
    }

    // Calculate total size and read metadata
    size_t total_size = 0;
    for (int i = 0; i < weights->num_tensors; i++) {
        TensorMetadata* meta = &weights->metadata[i];
        
        // Read dimensions
        fread(&meta->num_dims, sizeof(int), 1, fp);
        fread(meta->dims, sizeof(int), meta->num_dims, fp);

        // Read name
        size_t name_len;
        fread(&name_len, sizeof(size_t), 1, fp);
        fread(meta->name, 1, name_len, fp);
        meta->name[name_len] = '\0';

        // Calculate tensor size
        size_t tensor_size = sizeof(float);
        for (int j = 0; j < meta->num_dims; j++) {
            tensor_size *= meta->dims[j];
        }
        meta->size = tensor_size;
        meta->offset = total_size;
        total_size += tensor_size;
    }

    // Allocate and read weight data
    weights->total_size = total_size;
    weights->data = (float*)malloc(total_size);
    if (!weights->data) {
        free(weights->metadata);
        free(weights);
        fclose(fp);
        return NULL;
    }

    // Read all tensor data
    fread(weights->data, 1, total_size, fp);
    fclose(fp);

    return weights;
}

// Get tensor data by name
float* get_tensor_data(ModelWeights* weights, const char* name) {
    for (int i = 0; i < weights->num_tensors; i++) {
        if (strcmp(weights->metadata[i].name, name) == 0) {
            return weights->data + weights->metadata[i].offset;
        }
    }
    return NULL;
}

// Free model weights
void free_model_weights(ModelWeights* weights) {
    if (weights) {
        if (weights->data) free(weights->data);
        if (weights->metadata) free(weights->metadata);
        if (weights->unet_data) free(weights->unet_data);
        if (weights->vae_data) free(weights->vae_data);
        if (weights->text_encoder_data) free(weights->text_encoder_data);
        free(weights);
    }
} 