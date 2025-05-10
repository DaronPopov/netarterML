# Vision Models

The vision module provides a comprehensive interface for image processing and analysis, offering various capabilities from basic image operations to advanced computer vision tasks. This module is built on top of popular computer vision libraries and provides a unified API for different vision tasks.

## Quick Start

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

## API Reference

### EasyImage

The main class for interacting with vision models.

#### Methods

##### `__init__(self, config=None)`
Initialize the vision API with optional configuration.

Parameters:
- `config` (dict, optional): Configuration dictionary for the API

##### `load_image(self, image_path)`
Load an image from a file path.

Parameters:
- `image_path` (str): Path to the image file

Returns:
- `PIL.Image`: Loaded image

##### `analyze_image(self, image)`
Perform comprehensive analysis on an image.

Parameters:
- `image` (PIL.Image): Input image to analyze

Returns:
- `dict`: Analysis results including objects, scenes, and attributes

##### `get_image_features(self, image)`
Extract feature vectors from an image.

Parameters:
- `image` (PIL.Image): Input image

Returns:
- `numpy.ndarray`: Feature vector

## Advanced Usage

### Object Detection

```python
from OPENtransformer import EasyImage

vision = EasyImage()

# Load image
image = vision.load_image("image.jpg")

# Detect objects
detections = vision.detect_objects(
    image,
    confidence_threshold=0.5,
    model="yolov5"
)

# Visualize detections
annotated_image = vision.visualize_detections(image, detections)
annotated_image.save("detections.jpg")
```

### Image Classification

```python
from OPENtransformer import EasyImage

vision = EasyImage()

# Classify image
predictions = vision.classify_image(
    image,
    top_k=5,
    model="resnet50"
)

# Print results
for label, score in predictions:
    print(f"{label}: {score:.2f}")
```

### Feature Extraction

```python
from OPENtransformer import EasyImage
import numpy as np

vision = EasyImage()

# Extract features
features = vision.get_image_features(
    image,
    model="clip",
    normalize=True
)

# Use features for similarity search
similarity = np.dot(features1, features2)
```

## Best Practices

1. **Image Preprocessing**
   - Resize images to appropriate dimensions
   - Normalize pixel values
   - Handle different color spaces

2. **Performance Optimization**
   - Use batch processing when possible
   - Enable GPU acceleration
   - Implement caching for frequent operations

3. **Error Handling**
   - Validate image formats
   - Handle corrupted images
   - Implement proper error messages

4. **Resource Management**
   - Clean up resources when done
   - Monitor memory usage
   - Use context managers for automatic cleanup

## Examples

### Basic Image Analysis

```python
from OPENtransformer import EasyImage

vision = EasyImage()

# Load and analyze image
image = vision.load_image("landscape.jpg")
analysis = vision.analyze_image(image)

# Print analysis results
print("Detected objects:", analysis["objects"])
print("Scene type:", analysis["scene"])
print("Image attributes:", analysis["attributes"])
```

### Object Detection with Custom Model

```python
from OPENtransformer import EasyImage
from ultralytics import YOLO

# Load custom YOLO model
model = YOLO("path/to/custom/model.pt")

# Initialize vision API with custom model
vision = EasyImage(
    config={
        "detection_model": model,
        "confidence_threshold": 0.6
    }
)

# Perform detection
detections = vision.detect_objects(image)
```

### Image Similarity Search

```python
from OPENtransformer import EasyImage
import numpy as np

vision = EasyImage()

# Load images
image1 = vision.load_image("image1.jpg")
image2 = vision.load_image("image2.jpg")

# Extract features
features1 = vision.get_image_features(image1)
features2 = vision.get_image_features(image2)

# Calculate similarity
similarity = np.dot(features1, features2)
print(f"Similarity score: {similarity:.2f}")
```

## Troubleshooting

Common issues and solutions:

1. **Image Loading Errors**
   - Check file path
   - Verify image format
   - Handle corrupted images

2. **Memory Issues**
   - Reduce image size
   - Use batch processing
   - Enable memory optimization

3. **Performance Issues**
   - Use appropriate model size
   - Enable GPU acceleration
   - Implement caching

4. **Model Loading Errors**
   - Check model path
   - Verify model compatibility
   - Ensure sufficient disk space

For more information and examples, please refer to the [examples directory](../examples/README.md). 