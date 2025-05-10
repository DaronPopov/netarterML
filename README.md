# OPENtransformer

A powerful transformer-based machine learning framework with optimized inference engines for various tasks including diffusion models, vision transformers, and LLMs.

## Quick Start

1. Clone the repository:
```bash
git clone https://github.com/DaronPopov/netarterML.git
cd netarterML
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up your Hugging Face token (required for model downloads):
```bash
export HF_TOKEN=your_token_here  # On Windows use: set HF_TOKEN=your_token_here
```

## Model Downloads

The framework supports various models that can be downloaded as needed:

### Diffusion Models
```bash
# Download Stable Diffusion v1.5
python OPENtransformer/arm64_engine/core/c_inference/download_model.py

# Download Dreamlike PhotoReal 2.0
python OPENtransformer/arm64_engine/core/c_inference/download_realistic.py

# Download Dreamshaper
python OPENtransformer/arm64_engine/core/c_inference/download_dreamshaper.py

# Download Stable Diffusion v2
python OPENtransformer/arm64_engine/core/c_inference/download_sdv2.py
```

### Vision Models
```bash
# Download vision model weights
python OPENtransformer/vision/download_weights.py
```

### LLM Models
```bash
# Download LLM weights
python OPENtransformer/chat/download_weights.py
```

## Usage Examples

### Image Generation
```python
from OPENtransformer.diffusion.easy_diffusion_api import EasyDiffusionAPI

# Initialize API
api = EasyDiffusionAPI()

# Generate an image
result = api.generate_image(
    prompt="Your prompt here",
    steps=25,
    guidance=7.5,
    output_path="output.png"
)
```

### Vision Tasks
```python
from OPENtransformer.vision.easy_image import EasyVisionAPI

# Initialize API
api = EasyVisionAPI()

# Process an image
result = api.process_image("input.jpg")
```

### LLM Chat
```python
from OPENtransformer.chat.llm_api import LLMAPI

# Initialize API
api = LLMAPI()

# Generate response
response = api.generate("Your prompt here")
```

## Development Setup

For development work:

1. Install development dependencies:
```bash
pip install -r requirements-dev.txt
```

2. Run tests:
```bash
./scripts/run_tests.sh
```

## Project Structure

- `OPENtransformer/` - Main package directory
  - `arm64_engine/` - Optimized inference engine
  - `chat/` - LLM and chat functionality
  - `diffusion/` - Image generation models
  - `vision/` - Computer vision models
  - `core/` - Core transformer implementations
  - `ui/` - User interface components

## Contributing

Please read [CONTRIBUTING.md](docs/contributing.md) for details on our code of conduct and the process for submitting pull requests.

## License

This project is licensed under the MIT License - see the [LICENSE](docs/test_and_deps/LICENSE.txt) file for details. 