# Local AI Engine Examples

This directory contains examples for using various AI models and capabilities through the Local AI Engine.

## Directory Structure

- `diffusion/`: Examples for image generation using diffusion models
- `chat/`: Examples for text generation and conversation using language models
- `multimodal/`: Examples combining vision and language models
- `vision/`: Examples for vision-only tasks (coming soon)

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set your Hugging Face token as an environment variable:
```bash
export HF_TOKEN=your_token_here
```

3. Run the examples:
```bash
python examples/vision/multimodal_pipeline.py
```

## Getting Started

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Set up your Hugging Face token:
```bash
export HF_TOKEN=your_token_here
```

3. Choose the example you want to try and follow its README.

## Available Examples

### Diffusion Models
- Basic image generation
- Multiple model comparison
- Model downloading and management
- SDXL model usage

### Chat Models
- Basic chat interface
- Model-to-model conversation
- Streaming responses
- System prompt usage

### Multimodal Analysis
- Image analysis with chat
- Interactive mode
- Medical imaging support
- Real-time analysis

## Common Features

1. Easy model loading and initialization
2. Support for multiple model architectures
3. Configurable parameters
4. Error handling
5. Resource management

## Best Practices

1. Use appropriate model size for your hardware
2. Monitor resource usage
3. Implement proper error handling
4. Use system prompts when available
5. Follow model-specific guidelines

## Contributing

Feel free to contribute new examples or improve existing ones. Please follow the established patterns and include proper documentation. 