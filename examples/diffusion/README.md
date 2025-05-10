# Diffusion Model Examples

This directory contains examples for using the Easy Diffusion API to generate images using various diffusion models.

## Files
- `easy_diffusion_api.py`: The main API implementation
- `easy_diffusion_example.py`: Example usage of the API

## Usage Examples

### Basic Usage
```python
from easy_diffusion_api import EasyDiffusionAPI

# Create API instance
api = EasyDiffusionAPI()

# Register and activate a model
api.register_model("sd-v1-5", "runwayml/stable-diffusion-v1-5")
api.set_active_model("sd-v1-5")

# Generate an image
result = api.generate_image(
    prompt="A beautiful landscape with mountains and a lake at sunset",
    steps=20,
    output_path="output.png"
)
```

### Multiple Models
```python
# Register multiple models
models = [
    ("sd-v1-5", "runwayml/stable-diffusion-v1-5"),
    ("dreamlike", "dreamlike-art/dreamlike-photoreal-2.0"),
    ("dreamshaper", "dreamshaper/dreamshaper-xl-turbo")
]

for model_id, model_path in models:
    api.register_model(model_id, model_path)
```

### Download and Use
```python
# Download a model from Hugging Face
api.download_model("runwayml/stable-diffusion-v1-5")

# Register with a shorter name
api.register_model("sd-v1-5", model_id)
```

## Supported Models
- Stable Diffusion v1.5
- Dreamlike Photoreal 2.0
- Dreamshaper XL
- SDXL Base
- SDXL Turbo

## Best Practices
1. Use appropriate resolution for each model
2. Adjust steps and guidance scale based on model
3. Add "photo" to prompts for photorealistic models
4. Use higher steps (25-50) for SDXL models
5. Use guidance scale 7.5-9.0 for better prompt adherence 