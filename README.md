# Local AI Engine Examples

This directory contains runnable examples for using various AI models and capabilities through the Local AI Engine. Follow these instructions to ensure all examples and tests work smoothly.

## One-Command Quick Start

Copy and paste this block into your terminal to set up and verify your environment:

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set your Hugging Face token (replace with your actual token)
export HF_TOKEN=your_token_here

# 3. Set the Python path for core inference modules
export PYTHONPATH=OPENtransformer/arm64_engine/core/c_inference

# 4. Run the system check to verify all imports and readiness
python tests/test_vision_download.py
```

If you see all ✓ messages and no errors, your environment is ready for model downloading and inference!

---

## One-Command Quick Start for Each Modality

After verifying your environment, you can run a full quick start for each modality with a single copy-paste command block:

### Diffusion (Image Generation)
```bash
pip install -r requirements.txt
export HF_TOKEN=your_token_here
export PYTHONPATH=OPENtransformer/arm64_engine/core/c_inference
python examples/diffusion/easy_diffusion_example.py
```

### Vision (Image Classification, Captioning, etc.)
```bash
pip install -r requirements.txt
export HF_TOKEN=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
export PYTHONPATH=OPENtransformer/arm64_engine/core/c_inference
python examples/vision/vision_api.py
```

### Chat (Language Model Conversation)
```bash
pip install -r requirements.txt
export HF_TOKEN=your_token_here
export PYTHONPATH=OPENtransformer/arm64_engine/core/c_inference
python examples/chat/tinyllama_chat.py
```

### Multimodal (Vision + Language)
```bash
pip install -r requirements.txt
export HF_TOKEN=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
export PYTHONPATH=OPENtransformer/arm64_engine/core/c_inference
python examples/multimodal/multimodal_pipeline.py
```

---

## Directory Structure

- `diffusion/`: Image generation using diffusion models
- `chat/`: Text generation and conversation using language models
- `multimodal/`: Vision and language model combinations
- `vision/`: Vision-only tasks (e.g., classification, captioning)

## Quick Setup

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
2. **Set your Hugging Face token as an environment variable:**
   ```bash
   export HF_TOKEN=your_token_here
   ```
3. **Set the Python path for examples that require core inference modules:**
   ```bash
   export PYTHONPATH=OPENtransformer/arm64_engine/core/c_inference
   ```

## Running Example Scripts

- **Diffusion Example:**
  ```bash
  python examples/diffusion/easy_diffusion_example.py
  ```
- **Vision Example:**
  ```bash
  python examples/vision/vision_api.py
  ```
- **Chat Example:**
  ```bash
  python examples/chat/tinyllama_chat.py
  ```
- **Multimodal Example:**
  ```bash
  python examples/multimodal/multimodal_pipeline.py
  ```

> **Tip:** Always ensure you have set the `PYTHONPATH` as above before running any example that depends on custom C/C++/Python modules in `OPENtransformer/arm64_engine/core/c_inference`.

## Running Tests

To verify your installation and imports, you can run the provided test scripts:

```bash
python tests/test_vision_download.py
```

This script checks that all major modules (diffusion, vision, chat, core kernels) can be imported and initialized.

## Troubleshooting

- **ImportError: No module named 'py_diffusion_interface'**
  - Make sure you have set the `PYTHONPATH` as shown above.
- **Hugging Face Token Errors**
  - Ensure you have set your `HF_TOKEN` environment variable with a valid token.
- **Missing dependencies**
  - Run `pip install -r requirements.txt` again to ensure all dependencies are installed.
- **GPU/CPU Warnings**
  - Some warnings (e.g., bitsandbytes, CUDA) are informational and do not prevent CPU-based inference.

## Best Practices

- Use appropriate model size for your hardware
- Monitor resource usage
- Implement proper error handling
- Use system prompts when available
- Follow model-specific guidelines

## Contributing

Feel free to contribute new examples or improve existing ones. Please follow the established patterns and include proper documentation. 