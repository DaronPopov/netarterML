# Diffusion Models

The diffusion module provides a high-level interface for working with state-of-the-art image generation models. This module is built on top of the Stable Diffusion framework and provides additional utilities for model management and optimization.

## Quick Start

```python
from OPENtransformer import EasyDiffusionAPI

# Initialize the API
diffusion = EasyDiffusionAPI()

# Register a model
diffusion.register_model("sd-v1-5", "runwayml/stable-diffusion-v1-5")

# Generate an image
image = diffusion.generate_image(
    prompt="A beautiful sunset over mountains",
    negative_prompt="blurry, low quality",
    num_inference_steps=30
)
```

## API Reference

### EasyDiffusionAPI

The main class for interacting with diffusion models.

#### Methods

##### `__init__(self, config=None)`
Initialize the diffusion API with optional configuration.

Parameters:
- `config` (dict, optional): Configuration dictionary for the API

##### `register_model(self, model_id, model_path)`
Register a new model for use.

Parameters:
- `model_id` (str): Unique identifier for the model
- `model_path` (str): Path to the model or Hugging Face model ID

##### `generate_image(self, prompt, negative_prompt=None, num_inference_steps=30, guidance_scale=7.5)`
Generate an image from a text prompt.

Parameters:
- `prompt` (str): Text description of the desired image
- `negative_prompt` (str, optional): Text description of what to avoid
- `num_inference_steps` (int, optional): Number of denoising steps
- `guidance_scale` (float, optional): How closely to follow the prompt

Returns:
- `PIL.Image`: Generated image

##### `transform_image(self, image, prompt, negative_prompt=None)`
Transform an existing image based on a prompt.

Parameters:
- `image` (PIL.Image): Input image to transform
- `prompt` (str): Text description of the desired transformation
- `negative_prompt` (str, optional): Text description of what to avoid

Returns:
- `PIL.Image`: Transformed image

## Advanced Usage

### Custom Pipelines

```python
from OPENtransformer import EasyDiffusionAPI
from diffusers import StableDiffusionPipeline

# Create custom pipeline
pipeline = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16
)

# Initialize API with custom pipeline
diffusion = EasyDiffusionAPI(pipeline=pipeline)
```

### Model Optimization

```python
# Enable memory optimization
diffusion = EasyDiffusionAPI(
    config={
        "enable_attention_slicing": True,
        "enable_model_cpu_offload": True
    }
)
```

### Batch Processing

```python
# Generate multiple images
images = diffusion.generate_images(
    prompts=[
        "A beautiful sunset",
        "A mountain landscape",
        "An ocean view"
    ],
    batch_size=3
)
```

## Best Practices

1. **Prompt Engineering**
   - Be specific and detailed in your prompts
   - Use negative prompts to avoid unwanted elements
   - Experiment with different prompt structures

2. **Performance Optimization**
   - Use appropriate batch sizes for your hardware
   - Enable memory optimization for large models
   - Consider using half-precision (float16) for faster inference

3. **Error Handling**
   - Handle out-of-memory errors gracefully
   - Implement retry logic for API calls
   - Validate input parameters

4. **Resource Management**
   - Clean up resources when done
   - Monitor GPU memory usage
   - Use context managers for automatic cleanup

## Examples

### Basic Image Generation

```python
from OPENtransformer import EasyDiffusionAPI

diffusion = EasyDiffusionAPI()
diffusion.register_model("sd-v1-5", "runwayml/stable-diffusion-v1-5")

image = diffusion.generate_image(
    prompt="A serene lake at sunset with mountains in the background",
    negative_prompt="people, buildings, cars",
    num_inference_steps=50,
    guidance_scale=8.5
)
image.save("serene_lake.png")
```

### Image-to-Image Transformation

```python
from PIL import Image
from OPENtransformer import EasyDiffusionAPI

diffusion = EasyDiffusionAPI()
original_image = Image.open("input.jpg")

transformed = diffusion.transform_image(
    image=original_image,
    prompt="Convert to watercolor painting style",
    negative_prompt="photorealistic, detailed"
)
transformed.save("watercolor.jpg")
```

### Custom Pipeline with ControlNet

```python
from OPENtransformer import EasyDiffusionAPI
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel

# Load ControlNet model
controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/sd-controlnet-canny",
    torch_dtype=torch.float16
)

# Create custom pipeline
pipeline = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    controlnet=controlnet,
    torch_dtype=torch.float16
)

# Initialize API with custom pipeline
diffusion = EasyDiffusionAPI(pipeline=pipeline)
```

## Troubleshooting

Common issues and solutions:

1. **Out of Memory Errors**
   - Reduce batch size
   - Enable memory optimization
   - Use model offloading

2. **Slow Generation**
   - Reduce number of inference steps
   - Use half-precision
   - Enable attention slicing

3. **Quality Issues**
   - Adjust guidance scale
   - Improve prompt quality
   - Increase inference steps

4. **Model Loading Errors**
   - Check model path
   - Verify model compatibility
   - Ensure sufficient disk space

For more information and examples, please refer to the [examples directory](../examples/README.md). 