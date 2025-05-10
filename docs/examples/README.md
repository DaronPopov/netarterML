# OPENtransformer Examples

This directory contains examples demonstrating how to use various features of the OPENtransformer library. Each example is designed to showcase specific functionality and best practices.

## Directory Structure

```
examples/
├── diffusion/
│   ├── basic_generation.py
│   ├── image_transformation.py
│   └── custom_pipeline.py
├── chat/
│   ├── basic_chat.py
│   ├── conversation_management.py
│   └── streaming_responses.py
├── vision/
│   ├── image_analysis.py
│   ├── object_detection.py
│   └── feature_extraction.py
└── multimodal/
    ├── image_text_analysis.py
    ├── visual_qa.py
    └── cross_modal_search.py
```

## Running Examples

To run an example, navigate to the appropriate directory and execute the Python script:

```bash
# Run a diffusion example
python examples/diffusion/basic_generation.py

# Run a chat example
python examples/chat/basic_chat.py

# Run a vision example
python examples/vision/image_analysis.py

# Run a multimodal example
python examples/multimodal/image_text_analysis.py
```

## Example Categories

### 1. Diffusion Examples

- **Basic Generation**: Simple text-to-image generation
- **Image Transformation**: Converting images to different styles
- **Custom Pipeline**: Using custom diffusion pipelines

### 2. Chat Examples

- **Basic Chat**: Simple conversation with a language model
- **Conversation Management**: Handling multi-turn conversations
- **Streaming Responses**: Real-time text generation

### 3. Vision Examples

- **Image Analysis**: Basic image processing and analysis
- **Object Detection**: Detecting objects in images
- **Feature Extraction**: Extracting image features for similarity

### 4. Multimodal Examples

- **Image-Text Analysis**: Combining vision and language
- **Visual QA**: Answering questions about images
- **Cross-modal Search**: Searching images with text queries

## Example Code

### Basic Image Generation

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

### Basic Chat

```python
from OPENtransformer import LLMAPI, ModelConversation

# Initialize the API
chat = LLMAPI(model_name="meta-llama/Llama-2-7b-chat-hf")
chat.load_model()

# Create a conversation
conversation = ModelConversation(chat)

# Chat with the model
response = conversation.chat("What is machine learning?")
print(response)
```

### Image Analysis

```python
from OPENtransformer import EasyImage

# Initialize the vision model
vision = EasyImage()

# Load and analyze an image
image = vision.load_image("path/to/image.jpg")
analysis = vision.analyze_image(image)

# Print analysis results
print("Detected objects:", analysis["objects"])
print("Scene type:", analysis["scene"])
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

# Print results
print("Understanding:", result["understanding"])
print("Confidence:", result["confidence"])
```

## Best Practices

1. **Error Handling**
   - Always implement proper error handling
   - Use try-except blocks for model operations
   - Validate input parameters

2. **Resource Management**
   - Clean up resources when done
   - Use context managers
   - Monitor memory usage

3. **Performance**
   - Use appropriate batch sizes
   - Enable GPU acceleration when available
   - Implement caching for frequent operations

4. **Code Organization**
   - Keep examples focused and simple
   - Include comments and documentation
   - Follow PEP 8 style guidelines

## Contributing

We welcome contributions to the examples! If you'd like to add a new example:

1. Create a new file in the appropriate directory
2. Follow the existing code style
3. Include comments and documentation
4. Add a description to this README
5. Submit a pull request

## Support

For help with the examples:
1. Check the [documentation](../README.md)
2. Search [existing issues](https://github.com/yourusername/OPENtransformer/issues)
3. Create a new issue if needed 