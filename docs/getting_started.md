# Getting Started with OPENtransformer

This guide will help you get up and running with OPENtransformer quickly.

## Installation

You can install OPENtransformer using pip:

```bash
pip install OPENtransformer
```

Or install from source:

```bash
git clone https://github.com/yourusername/OPENtransformer.git
cd OPENtransformer
pip install -e .
```

## Basic Usage

### Diffusion Models

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

# Save the image
image.save("sunset.png")
```

### Chat Models

```python
from OPENtransformer import LLMAPI, ModelConversation

# Initialize the API
chat = LLMAPI(model_name="")
chat.load_model()

# Create a conversation
conversation = ModelConversation(chat)

# Chat with the model
response = conversation.chat("What is machine learning?")
print(response)
```

### Vision Models

```python
from OPENtransformer import EasyImage

# Initialize the vision model
vision = EasyImage()

# Load and analyze an image
image = vision.load_image("path/to/image.jpg")
analysis = vision.analyze_image(image)

# Get image features
features = vision.get_image_features(image)
```

### Multimodal Analysis

```python
from OPENtransformer import MultimodalAnalysis

# Initialize multimodal analysis
multimodal = MultimodalAnalysis()

# Analyze image with text
result = multimodal.analyze(
    image_path="path/to/image.jpg",
    text="Describe what you see in this image"
)
```

## Configuration

OPENtransformer can be configured using environment variables or a configuration file:

```python
# Using environment variables
export OPENTRANSFORMER_MODEL_PATH="/path/to/models"
export OPENTRANSFORMER_CACHE_DIR="/path/to/cache"

# Or using configuration file
from OPENtransformer.config import Config

config = Config(
    model_path="/path/to/models",
    cache_dir="/path/to/cache",
    device="cuda"  # or "cpu"
)
```

## Next Steps

- Check out the [Examples](examples/README.md) for more detailed usage
- Read the [API Reference](api/README.md) for detailed documentation
- Explore [Advanced Topics](advanced/README.md) for optimization and customization 