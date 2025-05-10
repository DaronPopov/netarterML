# C Inference Engine Integration Guide

This guide explains how the C inference engine is integrated into the UI for image generation.

## Overview

The C inference engine provides a high-performance, memory-efficient implementation of Stable Diffusion for image generation. It's designed to run efficiently on ARM64 architecture, making it ideal for Apple Silicon Macs.

Key benefits:
- Lower memory usage than PyTorch-based implementations
- Faster inference on ARM64 architecture
- Simple Python interface for easy integration

## Setup Instructions

### 1. Build the C Inference Engine

First, build the C inference engine:

```bash
cd OPENtransformer/arm64_engine/core/c_inference
make clean
make
pip install -e .
```

Alternatively, you can use the provided build script:

```bash
cd OPENtransformer/arm64_engine/core/c_inference
./build_and_test_integration.sh
```

### 2. Generate Images

There are two ways to generate images:

#### A. Using the simplified script:

```bash
cd ui/engines
./run_image_gen.sh "Your prompt here"
```

Options:
- `-m, --model MODEL`: Specify model ID (default: runwayml/stable-diffusion-v1-5)
- `-s, --steps STEPS`: Number of inference steps (default: 25)
- `-w, --width WIDTH`: Image width (default: 512)
- `--height HEIGHT`: Image height (default: 512)
- `-g, --guidance GUIDANCE`: Guidance scale (default: 7.5)
- `--seed SEED`: Random seed (default: 0 = random)

Example:
```bash
./run_image_gen.sh -s 50 -w 768 --height 512 "A beautiful sunset over the ocean"
```

#### B. Using the Python script directly:

```bash
cd ui/engines
python image_gen_engine.py --prompt "Your prompt here"
```

Options:
- `--prompt`: Text prompt for image generation (required)
- `--model`: Model ID or path (default: runwayml/stable-diffusion-v1-5)
- `--height`: Image height (default: 512)
- `--width`: Image width (default: 512)
- `--steps`: Number of inference steps (default: 25)
- `--guidance`: Guidance scale (default: 7.5)
- `--seed`: Random seed (0 for random)

## How It Works

The integration works as follows:

1. `image_gen_engine.py` imports the Python interface from the C inference engine
2. When generating an image, it calls `run_inference` from the Python interface
3. The Python interface communicates with the C inference engine through a bridge
4. The C engine performs the actual inference using optimized kernels
5. The resulting image is returned to Python and saved/displayed

## Troubleshooting

If you encounter issues:

1. **Import errors**: Make sure the C inference engine is built correctly and installed with `pip install -e .`

2. **Module not found**: Check that the path to the C inference directory is correct in `image_gen_engine.py`

3. **Missing libraries**: Install required dependencies with:
   ```bash
   pip install diffusers torch numpy pillow
   ```

4. **Build errors**: Make sure you have the necessary build tools installed (gcc, make)

5. **Memory errors**: Try reducing the image dimensions or inference steps

## Performance Tips

For best performance:

1. Use 25-50 inference steps for a good balance of quality and speed
2. Set `use_memory_optimizations` to True (default in the integration)
3. For quick drafts, reduce steps to 10-15
4. Keep the model loaded between generations by using the Python API directly

## Advanced Usage

For more advanced usage, you can directly use the C inference Python API in your own scripts:

```python
from OPENtransformer.arm64_engine.core.c_inference.py_diffusion_interface import run_inference
from PIL import Image
from io import BytesIO

# Generate image
width, height, channels, img_data = run_inference(
    model_path="runwayml/stable-diffusion-v1-5",
    prompt="Your prompt here",
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
image = Image.open(BytesIO(img_data))
image.save("output.png")
``` 