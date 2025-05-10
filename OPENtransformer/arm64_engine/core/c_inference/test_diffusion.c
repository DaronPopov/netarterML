/**
 * @file test_diffusion.c
 * @brief Test program for the C wrapper for Stable Diffusion inference
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include "c_inference_engine.h"
#include "python_bridge.h"

// Function to save PPM image
static int save_ppm(const char* filename, const uint8_t* data, int width, int height) {
    FILE* fp = fopen(filename, "wb");
    if (!fp) return -1;
    
    // Write PPM header
    fprintf(fp, "P6\n%d %d\n255\n", width, height);
    
    // Write image data
    fwrite(data, 1, width * height * 3, fp);
    fclose(fp);
    return 0;
}

// Function to display image using system viewer
static void display_image(const char* filename) {
    char cmd[1024];
#ifdef __APPLE__
    snprintf(cmd, sizeof(cmd), "open %s", filename);
#else
    snprintf(cmd, sizeof(cmd), "xdg-open %s", filename);
#endif
    system(cmd);
}

int main(int argc, char** argv) {
    // Initialize Python environment
    if (init_python_environment() != 0) {
        fprintf(stderr, "Failed to initialize Python environment\n");
        return 1;
    }

    if (argc < 2) {
        fprintf(stderr, "Usage: %s --prompt \"your prompt\" [--steps N] [--output filename.ppm] [--model MODEL_PATH]\n", argv[0]);
        return 1;
    }
    
    // Parse arguments
    const char* prompt = NULL;
    int steps = 7; // Use fewer steps by default for speed
    const char* output_file = "output.ppm";
    const char* model_path = "runwayml/stable-diffusion-v1-5"; // Default to Hugging Face model
    float guidance_scale = 7.5f;
    int width = 512;
    int height = 512;
    unsigned int seed = 0;
    
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--prompt") == 0 && i + 1 < argc) {
            prompt = argv[++i];
        } else if (strcmp(argv[i], "--steps") == 0 && i + 1 < argc) {
            steps = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--output") == 0 && i + 1 < argc) {
            output_file = argv[++i];
        } else if (strcmp(argv[i], "--model") == 0 && i + 1 < argc) {
            model_path = argv[++i];
        } else if (strcmp(argv[i], "--guidance") == 0 && i + 1 < argc) {
            guidance_scale = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "--width") == 0 && i + 1 < argc) {
            width = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--height") == 0 && i + 1 < argc) {
            height = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc) {
            seed = (unsigned int)atoi(argv[++i]);
        }
    }
    
    if (!prompt) {
        fprintf(stderr, "Error: No prompt provided\n");
        return 1;
    }
    
    // Create inference context
    InferenceContext* ctx = inference_create_context(model_path);
    if (!ctx) {
        fprintf(stderr, "Failed to create inference context\n");
        return 1;
    }
    
    // Load model weights (this will handle Hugging Face models correctly)
    if (inference_load_weights(ctx, NULL) != 0) {
        fprintf(stderr, "Failed to load model weights\n");
        inference_destroy_context(ctx);
        return 1;
    }
    
    // Configure inference
    InferenceConfig config = {
        .width = width,
        .height = height,
        .num_inference_steps = steps,
        .guidance_scale = guidance_scale,
        .seed = seed,
        .model_path = model_path
    };
    
    // Generate image
    uint8_t* image_data = NULL;
    int out_width, out_height, out_channels;
    
    printf("Generating image for prompt: %s\n", prompt);
    if (inference_generate_image(ctx, prompt, &config, &image_data, &out_width, &out_height, &out_channels) != 0) {
        fprintf(stderr, "Failed to generate image\n");
        inference_destroy_context(ctx);
        return 1;
    }
    
    // Save image
    if (save_ppm(output_file, image_data, out_width, out_height) != 0) {
        fprintf(stderr, "Failed to save image\n");
        inference_free_image(image_data);
        inference_destroy_context(ctx);
        return 1;
    }
    
    // Get and print metrics
    InferenceMetrics metrics;
    inference_get_metrics(ctx, &metrics);
    printf("\nInference metrics:\n");
    printf("Total steps: %d\n", metrics.num_steps);
    printf("Total time: %.2f ms\n", metrics.total_time_ms);
    printf("Average step time: %.2f ms\n", metrics.avg_step_time_ms);
    
    // Clean up
    inference_free_image(image_data);
    inference_destroy_context(ctx);
    
    // Display the image
    printf("\nImage saved to: %s\n", output_file);
    printf("Displaying generated image...\n");
    display_image(output_file);
    
    // Clean up Python environment before exiting
    cleanup_python_environment();
    return 0;
} 