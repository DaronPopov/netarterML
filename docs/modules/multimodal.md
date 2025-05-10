# Multimodal Models

The multimodal module provides a unified interface for working with models that can process both images and text simultaneously. This module enables advanced tasks such as image-text understanding, visual question answering, and cross-modal retrieval.

## Quick Start

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

## API Reference

### MultimodalAnalysis

The main class for interacting with multimodal models.

#### Methods

##### `__init__(self, config=None)`
Initialize the multimodal API with optional configuration.

Parameters:
- `config` (dict, optional): Configuration dictionary for the API

##### `analyze(self, image_path, text)`
Analyze an image with accompanying text.

Parameters:
- `image_path` (str): Path to the image file
- `text` (str): Text to analyze with the image

Returns:
- `dict`: Analysis results including image-text understanding

##### `visual_question_answering(self, image_path, question)`
Answer questions about an image.

Parameters:
- `image_path` (str): Path to the image file
- `question` (str): Question about the image

Returns:
- `str`: Answer to the question

##### `get_joint_embedding(self, image_path, text)`
Get joint embedding of image and text.

Parameters:
- `image_path` (str): Path to the image file
- `text` (str): Text to embed with the image

Returns:
- `numpy.ndarray`: Joint embedding vector

## Advanced Usage

### Visual Question Answering

```python
from OPENtransformer import MultimodalAnalysis

multimodal = MultimodalAnalysis()

# Answer questions about an image
answer = multimodal.visual_question_answering(
    image_path="image.jpg",
    question="What is the main subject in this image?"
)
print(f"Answer: {answer}")
```

### Cross-modal Retrieval

```python
from OPENtransformer import MultimodalAnalysis
import numpy as np

multimodal = MultimodalAnalysis()

# Get embeddings
image_embedding = multimodal.get_image_embedding("image.jpg")
text_embedding = multimodal.get_text_embedding("A beautiful sunset")

# Calculate similarity
similarity = np.dot(image_embedding, text_embedding)
print(f"Similarity score: {similarity:.2f}")
```

### Custom Model Integration

```python
from OPENtransformer import MultimodalAnalysis
from transformers import CLIPProcessor, CLIPModel

# Load custom CLIP model
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Initialize multimodal API with custom model
multimodal = MultimodalAnalysis(
    config={
        "model": model,
        "processor": processor,
        "device": "cuda"
    }
)
```

## Best Practices

1. **Model Selection**
   - Choose appropriate model for your task
   - Consider model size and performance
   - Balance between speed and accuracy

2. **Performance Optimization**
   - Use batch processing when possible
   - Enable GPU acceleration
   - Implement caching for frequent operations

3. **Error Handling**
   - Validate input formats
   - Handle model loading errors
   - Implement proper error messages

4. **Resource Management**
   - Clean up resources when done
   - Monitor memory usage
   - Use context managers for automatic cleanup

## Examples

### Basic Image-Text Analysis

```python
from OPENtransformer import MultimodalAnalysis

multimodal = MultimodalAnalysis()

# Analyze image with text
result = multimodal.analyze(
    image_path="landscape.jpg",
    text="Describe the scene in detail"
)

# Print analysis results
print("Image-text understanding:", result["understanding"])
print("Confidence score:", result["confidence"])
```

### Visual Question Answering with Context

```python
from OPENtransformer import MultimodalAnalysis

multimodal = MultimodalAnalysis()

# Set context
multimodal.set_context("This is a nature photography collection.")

# Answer questions
answer1 = multimodal.visual_question_answering(
    "image1.jpg",
    "What type of landscape is shown?"
)

answer2 = multimodal.visual_question_answering(
    "image2.jpg",
    "Are there any animals in the scene?"
)
```

### Cross-modal Search

```python
from OPENtransformer import MultimodalAnalysis
import numpy as np

multimodal = MultimodalAnalysis()

# Get embeddings for multiple images
image_embeddings = []
for image_path in ["image1.jpg", "image2.jpg", "image3.jpg"]:
    embedding = multimodal.get_image_embedding(image_path)
    image_embeddings.append(embedding)

# Search with text query
query_embedding = multimodal.get_text_embedding("A mountain landscape")
similarities = [np.dot(query_embedding, img_emb) for img_emb in image_embeddings]

# Find best match
best_match_idx = np.argmax(similarities)
print(f"Best matching image: image{best_match_idx + 1}.jpg")
```

## Troubleshooting

Common issues and solutions:

1. **Model Loading Errors**
   - Check model path
   - Verify model compatibility
   - Ensure sufficient disk space

2. **Memory Issues**
   - Reduce batch size
   - Enable model quantization
   - Use model offloading

3. **Performance Issues**
   - Use appropriate model size
   - Enable GPU acceleration
   - Implement caching

4. **Input Validation**
   - Check image formats
   - Validate text input
   - Handle edge cases

For more information and examples, please refer to the [examples directory](../examples/README.md). 