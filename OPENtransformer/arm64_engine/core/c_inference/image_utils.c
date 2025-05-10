#include <stdio.h>
#include <stdlib.h>

// Define STB_IMAGE_WRITE_IMPLEMENTATION before including to create implementation
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

// Save image data as PNG
int save_image_as_png(const char* filename, unsigned char* data, int width, int height, int channels) {
    if (!data || width <= 0 || height <= 0 || (channels != 1 && channels != 3 && channels != 4)) {
        fprintf(stderr, "Invalid image parameters for PNG saving\n");
        return 0; // Failed
    }
    
    int stride_in_bytes = width * channels;
    
    // Use stb_image_write to save PNG
    int result = stbi_write_png(filename, width, height, channels, data, stride_in_bytes);
    
    return result != 0; // Return true if successful
} 