# OPENtransformer Inference Scripts

Quick reference for running inference scripts in the OPENtransformer engine.

## Download Model Scripts

Download Stable Diffusion v1.5:
```bash
python download_model.py
# download_model.py
```

Download Dreamlike PhotoReal 2.0:
```bash
python download_realistic.py
# download_realistic.py
```

Download Dreamshaper:
```bash
python download_dreamshaper.py
# download_dreamshaper.py
```

Download Stable Diffusion v2:
```bash
python download_sdv2.py
# download_sdv2.py
```

## Image Generation Commands

Generate using Python with custom parameters:
```bash
python test_py_direct.py --prompt "your prompt here" --model "runwayml/stable-diffusion-v1-5" --steps 25 --guidance 7.5 --size 512 --output "output.png"
# test_py_direct.py
```

Generate using shell script with token:
```bash
./run_with_token.sh "your prompt here" "runwayml/stable-diffusion-v1-5"
# run_with_token.sh
```

Run with Python and token:
```bash
python run_with_token_python.py --prompt "your prompt here" --model "runwayml/stable-diffusion-v1-5" --steps 25 --guidance 7.5 --size 512
# run_with_token_python.py
```

Generate multiple portraits:
```bash
python generate_portraits.py
# generate_portraits.py
```

Run standard test:
```bash
./run_test.sh "your prompt here" --model="runwayml/stable-diffusion-v1-5" --steps=25 --guidance=7.5 --size=512
# run_test.sh
```

## Easy Diffusion API

A simple API for hot-swapping diffusion models, allowing easy model management and inference.

### Usage

```bash
# List registered models
python easy_diffusion_api.py list

# Register a model
python easy_diffusion_api.py register "sd-v1-5" --path "runwayml/stable-diffusion-v1-5"

# Set active model
python easy_diffusion_api.py activate "sd-v1-5"

# Generate an image
python easy_diffusion_api.py generate "a beautiful landscape" --steps 25 --guidance 7.5 --output "output.png"

# Download a model from Hugging Face
python easy_diffusion_api.py download "runwayml/stable-diffusion-v1-5"
```

### API Examples

Run the example file to see different usage scenarios:

```bash
# Run all examples
python easy_diffusion_example.py

# Run specific examples
python easy_diffusion_example.py basic
python easy_diffusion_example.py multiple
python easy_diffusion_example.py download
python easy_diffusion_example.py programmatic
```

### Programmatic Usage

```python
from easy_diffusion_api import EasyDiffusionAPI

# Create API instance
api = EasyDiffusionAPI()

# Register and activate a model
api.register_model("sd-v1-5", "runwayml/stable-diffusion-v1-5")
api.set_active_model("sd-v1-5")

# Generate an image
result = api.generate_image(
    prompt="Your prompt here",
    steps=25,
    guidance=7.5,
    output_path="output.png"
)
```

## Utilities

Set Hugging Face token:
```bash
export HF_TOKEN=your_token_here
```

Build and test integration:
```bash
./build_and_test_integration.sh
# build_and_test_integration.sh
```

Convert model weights:
```bash
python convert_model.py
# convert_model.py
```

Run benchmark:
```bash
./benchmark.sh
# benchmark.sh
```

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set your Hugging Face token as an environment variable:
```bash
export HF_TOKEN=your_token_here
```

3. Run the application:
```bash
python ui/app.py
``` 