# C Inference Engine for Image Generation

This directory contains a lightweight C-based inference engine for Stable Diffusion image generation. It provides high performance with lower memory usage compared to PyTorch-based implementations.

## Overview

The C inference engine consists of:

1. Core C implementation for key diffusion operations
2. Python bridge for easy integration with Python code  
3. Optimized versions of critical kernels for ARM64 architecture

## Building the Engine

To build the C inference engine:

```bash
# From the c_inference directory
make clean
make
```

This will build the shared library and Python extension module that can be used by Python applications.

## Using the Engine

There are two ways to use the C inference engine:

### 1. Direct C API

```c
#include "c_inference_engine.h"

// Create inference context
InferenceContext* ctx = inference_create_context("/path/to/model");

// Configure inference parameters
InferenceConfig config = {
    .width = 512,
    .height = 512,
    .num_inference_steps = 25,
    .guidance_scale = 7.5,
    .seed = 42,
    .use_memory_optimizations = true,
    .model_path = "/path/to/model"
};

// Generate image
uint8_t* img_data;
int out_width, out_height, out_channels;
inference_generate_image(
    ctx, 
    "A beautiful sunset over the mountains", 
    &config,
    &img_data,
    &out_width,
    &out_height,
    &out_channels
);

// Use the image data...

// Free memory
inference_free_image(img_data);
inference_destroy_context(ctx);
```

### 2. Python Interface

```python
from OPENtransformer.arm64_engine.core.c_inference.py_diffusion_interface import run_inference

# Generate image
width, height, channels, img_data = run_inference(
    model_path="runwayml/stable-diffusion-v1-5",
    prompt="A beautiful sunset over the mountains",
    num_inference_steps=25,
    width=512,
    height=512,
    guidance_scale=7.5,
    seed=42,
    use_memory_optimizations=True,
    callback_ptr=None,
    user_data_ptr=None
)

# Convert to PIL Image
from io import BytesIO
from PIL import Image
image = Image.open(BytesIO(img_data))
image.save("output.png")
```

## Integration with UI

The C inference engine is integrated into the UI through the `image_gen_engine.py` module in the `ui/engines` directory. You can use it by running:

```bash
python image_gen_engine.py --prompt "Your prompt here" --engine c
```

Or in interactive mode:

```bash
python image_gen_engine.py --interactive --engine c
```

## Performance Considerations

- The C inference engine uses significantly less memory than PyTorch-based implementations
- Inference is faster on ARM64 architecture due to optimized kernels
- For best performance, set the `use_memory_optimizations` flag to `True`

## Troubleshooting

If you encounter issues with the C inference engine:

1. Make sure the engine is properly built (run `make clean && make`)
2. Verify that the model path is correct
3. Check for missing dependencies

If the C engine fails, the system will automatically fall back to the HuggingFace implementation.

# OPENtransformer Inference Scripts

Quick reference for running inference scripts in the OPENtransformer engine.

## Download Model Scripts

Download Stable Diffusion v1.5:
```bash
python download_model.py
# download_model.py
```

Download Dreamlike PhotoReal 2.0:
```bash
python download_realistic.py
# download_realistic.py
```

Download Dreamshaper:
```bash
python download_dreamshaper.py
# download_dreamshaper.py
```

Download Stable Diffusion v2:
```bash
python download_sdv2.py
# download_sdv2.py
```

Download SDXL models:
```bash
# Download the official SDXL base model
python download_sdxl.py

# Download a specific SDXL model
python download_sdxl.py --model "lykon/dreamshaper-xl-1-0" --name "dreamshaper-xl"
# download_sdxl.py
```

All download scripts now automatically convert models to binary format for optimized C/ASM inference. If you need to reconvert an existing model:

```bash
# Force reconversion of a standard SD model
python download_model.py --model "runwayml/stable-diffusion-v1-5" --force-reconvert

# Force reconversion of an SDXL model
python download_sdxl.py --model "stabilityai/stable-diffusion-xl-base-1.0" --force-reconvert
```

## Image Generation Commands

Generate using Python with custom parameters:
```bash
python test_py_direct.py --prompt "your prompt here" --model "runwayml/stable-diffusion-v1-5" --steps 25 --guidance 7.5 --size 512 --output "output.png"
# test_py_direct.py
```

Generate using shell script with token:
```bash
./run_with_token.sh "your prompt here" "runwayml/stable-diffusion-v1-5"
# run_with_token.sh
```

Run with Python and token:
```bash
python run_with_token_python.py --prompt "your prompt here" --model "runwayml/stable-diffusion-v1-5" --steps 25 --guidance 7.5 --size 512
# run_with_token_python.py
```

Generate multiple portraits:
```bash
python generate_portraits.py
# generate_portraits.py
```

Run standard test:
```bash
./run_test.sh "your prompt here" --model="runwayml/stable-diffusion-v1-5" --steps=25 --guidance=7.5 --size=512
# run_test.sh
```

## Easy Diffusion API

A simple API for hot-swapping diffusion models, allowing easy model management and inference.

### Usage

```bash
# List registered models
python easy_diffusion_api.py list

# Register a model
python easy_diffusion_api.py register "dreamlike-photoreal" --path "dreamlike-art/dreamlike-photoreal-2.0"

# Set active model
python easy_diffusion_api.py activate "dreamlike-photoreal"

# Generate an image
python easy_diffusion_api.py generate "photo, a beautiful landscape" --steps 25 --guidance 7.5 --output "output.png"

# Download a model from Hugging Face
python easy_diffusion_api.py download "dreamlike-art/dreamlike-photoreal-2.0"
```

### Supported Models

#### Dreamlike Photoreal 2.0
For best results with this model:
- Add "photo" to your prompt to enhance photorealism
- Use 768x768 resolution (the model was trained on this size)
- Try different aspect ratios for different compositions (portrait/landscape)

#### Stable Diffusion XL Models
The API now supports SDXL models, which produce higher quality images at larger resolutions:

```bash
# Register SDXL models
python easy_diffusion_api.py register "sdxl-base" --path "stabilityai/stable-diffusion-xl-base-1.0"
python easy_diffusion_api.py register "dreamshaper-xl" --path "lykon/dreamshaper-xl-1-0"

# Generate with SDXL
python easy_diffusion_api.py generate "an astronaut riding a horse on mars, highly detailed" --model "sdxl-base" --width 1024 --height 1024 --steps 30 --guidance 9.0 --output "sdxl_output.png"
```

For best results with SDXL models:
- Use 1024x1024 resolution or larger aspect ratios (1024x768, 768x1024)
- Increase the number of steps (25-50) for better quality
- Use slightly higher guidance scale (7.5-9.0) for more prompt adherence
- No need to add "photo" as SDXL is already optimized for high-quality outputs

### API Examples

Run the example file to see different usage scenarios:

```bash
# Run all examples
python easy_diffusion_example.py

# Run specific examples
python easy_diffusion_example.py basic       # Basic Dreamlike Photoreal example
python easy_diffusion_example.py multiple    # Multiple models comparison
python easy_diffusion_example.py download    # Download and use models
python easy_diffusion_example.py programmatic # Programmatic API usage
python easy_diffusion_example.py sdxl        # SDXL models example
```

### Programmatic Usage

```python
from easy_diffusion_api import EasyDiffusionAPI

# Create API instance
api = EasyDiffusionAPI()

# Register and activate an SDXL model
api.register_model("sdxl-base", "stabilityai/stable-diffusion-xl-base-1.0")
api.set_active_model("sdxl-base")

# Generate a high-quality image with SDXL
result = api.generate_image(
    prompt="a majestic mountain landscape with a lake, epic lighting, detailed, 8k",
    steps=30,
    guidance=8.5,
    width=1024,
    height=1024,
    output_path="sdxl_landscape.png"
)
```

## Binary Format for C/ASM Inference

All models are automatically converted to an optimized binary format for C/ASM inference. This conversion:

1. Extracts all model weights and converts them to float32 numpy arrays
2. Saves them in a compressed binary format (.bin files)
3. Creates a model_config.json file with all necessary parameters
4. Organizes files in a way that's optimized for the C inference engine

The binary conversion provides several advantages:
- Faster loading times in the C/ASM backend
- Reduced memory usage during inference
- Optimized format for ARM64 architectures
- Consistent layout for both standard SD and SDXL models

You can check if a model has been properly converted by looking for the `binary_conversion_complete` file in the model directory.

## Utilities

Set Hugging Face token:
```bash
source ./set_hf_token.sh
# set_hf_token.sh
```

Build and test integration:
```bash
./build_and_test_integration.sh
# build_and_test_integration.sh
```

Convert model weights:
```bash
python convert_model.py
# convert_model.py
```

Run benchmark:
```bash
./benchmark.sh
# benchmark.sh
``` 