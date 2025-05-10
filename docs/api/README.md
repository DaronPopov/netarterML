# OPENtransformer API Reference

This document provides detailed API reference for all OPENtransformer modules and their components.

## Table of Contents

1. [Diffusion API](diffusion.md)
   - EasyDiffusionAPI
   - Model Management
   - Image Generation
   - Image Transformation

2. [Chat API](chat.md)
   - LLMAPI
   - ModelConversation
   - Response Generation
   - Context Management

3. [Vision API](vision.md)
   - EasyImage
   - Image Analysis
   - Object Detection
   - Feature Extraction

4. [Multimodal API](multimodal.md)
   - MultimodalAnalysis
   - Image-Text Understanding
   - Visual Question Answering
   - Cross-modal Retrieval

## Common Patterns

### Initialization

All API classes follow a similar initialization pattern:

```python
from OPENtransformer import EasyDiffusionAPI, LLMAPI, EasyImage, MultimodalAnalysis

# Initialize with default configuration
api = ClassName()

# Initialize with custom configuration
api = ClassName(config={
    "key": "value",
    "option": "setting"
})
```

### Configuration

Common configuration options:

```python
config = {
    # Model settings
    "model_name": "model-name",
    "model_path": "path/to/model",
    
    # Device settings
    "device": "cuda",  # or "cpu"
    
    # Performance settings
    "batch_size": 32,
    "num_workers": 4,
    
    # Memory settings
    "enable_attention_slicing": True,
    "enable_model_cpu_offload": True,
    
    # Cache settings
    "cache_dir": "path/to/cache",
    "use_cache": True
}
```

### Error Handling

All API methods follow a consistent error handling pattern:

```python
try:
    result = api.method(params)
except ModelError as e:
    # Handle model-specific errors
    print(f"Model error: {e}")
except ResourceError as e:
    # Handle resource-related errors
    print(f"Resource error: {e}")
except Exception as e:
    # Handle other errors
    print(f"Unexpected error: {e}")
```

### Resource Management

Use context managers for proper resource cleanup:

```python
from OPENtransformer import EasyDiffusionAPI

with EasyDiffusionAPI() as diffusion:
    # Use the API
    image = diffusion.generate_image("prompt")
    image.save("output.png")
# Resources are automatically cleaned up
```

## Type Hints

All API methods include type hints for better IDE support:

```python
from typing import Dict, List, Optional, Union
from PIL import Image
import numpy as np

class ExampleAPI:
    def method(
        self,
        param1: str,
        param2: Optional[int] = None,
        param3: Union[str, List[str]] = "default"
    ) -> Dict[str, Union[str, float]]:
        pass
```

## Return Types

Common return types across APIs:

```python
# Image generation
Image  # PIL.Image

# Text generation
str    # Generated text

# Analysis results
Dict[str, Any]  # Analysis results

# Feature vectors
np.ndarray  # Feature vectors

# Batch results
List[Any]  # List of results
```

## Best Practices

1. **API Usage**
   - Always check return values
   - Handle errors appropriately
   - Use type hints for better IDE support

2. **Resource Management**
   - Use context managers
   - Clean up resources properly
   - Monitor memory usage

3. **Performance**
   - Use appropriate batch sizes
   - Enable caching when possible
   - Utilize GPU acceleration

4. **Error Handling**
   - Implement proper error handling
   - Use specific exception types
   - Provide meaningful error messages

## Version Compatibility

API versions and compatibility:

```python
import OPENtransformer

# Check version
print(OPENtransformer.__version__)

# Check compatibility
if OPENtransformer.is_compatible("1.0.0"):
    # Use features
    pass
```

## Deprecation Policy

Deprecated features are marked with warnings:

```python
import warnings

# Deprecated method
def old_method():
    warnings.warn(
        "This method is deprecated. Use new_method() instead.",
        DeprecationWarning,
        stacklevel=2
    )
```

## Contributing

To contribute to the API:

1. Follow the [coding style guide](../contributing.md#coding-style)
2. Add type hints to all methods
3. Include docstrings and examples
4. Update this documentation

## Support

For API support:
1. Check the [documentation](../README.md)
2. Search [existing issues](https://github.com/yourusername/OPENtransformer/issues)
3. Create a new issue if needed 