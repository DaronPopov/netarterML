# Local AI Engine Examples

This directory contains runnable examples for using various AI models and capabilities through the Local AI Engine. Follow these instructions to ensure all examples and tests work smoothly.

## Quick Start: Run Any Example with One Command

After setup, you can run any example with a single bash script:

### Vision (Image Captioning, Classification, etc.)
```bash
bash scripts/run_vision.sh
```

### Diffusion (Image Generation)
```bash
bash scripts/run_c_diffusion.sh
```

### Chat (Language Model)
```bash
bash scripts/run_chat.sh
```

### Multimodal (Vision + Language)
```bash
bash scripts/run_multimodal.sh
```

---

## Setup

1. **Create and activate virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```

2. **Install the package with all dependencies:**
   ```bash
   pip install -e .
   ```

3. **Build the C Inference Engine (for Diffusion and LLM):**
   ```bash
   bash scripts/build_c_inference.sh
   ```

4. **Set up environment variables:**
   ```bash
   # Set Hugging Face token (required for model downloads)
   export HUGGINGFACE_TOKEN=your_token_here
   
   # Set Python path for C/ASM backend
   export PYTHONPATH="/path/to/your/project/OPENtransformer/arm64_engine/core/c_inference:$PYTHONPATH"
   ```

---

## Scripts Directory
- `scripts/run_vision.sh`: Runs the vision example
- `scripts/run_c_diffusion.sh`: Runs the diffusion example with C backend
- `scripts/run_chat.sh`: Runs the chat example
- `scripts/run_multimodal.sh`: Runs the multimodal example
- `scripts/build_c_inference.sh`: Builds the C backend for diffusion and LLM

---

## Available Examples

### Vision Examples
```

## Interactive Demos

### LLM Chat Demo with C/ASM Backend
Run the LLM chat demo with optimized C/ASM backend using TinyLlama 1.1B:

1. **Get your Hugging Face token:**
   - Go to https://huggingface.co/
   - Create an account or log in
   - Go to your profile settings
   - Create a new token

2. **Set up the environment:**
   ```bash
   # Set your Hugging Face token
   export HUGGINGFACE_TOKEN=your_token_here
   
   # Set Python path for C/ASM backend
   export PYTHONPATH="/path/to/your/project/OPENtransformer/arm64_engine/core/c_inference:$PYTHONPATH"
   ```

3. **Build the C/ASM backend:**
   ```bash
   bash scripts/build_c_inference.sh
   ```

4. **Run the chat example:**
   ```bash
   cd examples/chat
   python llm_chat_example.py
   ```

The chat interface will start and you can interact with the TinyLlama 1.1B model. Type 'quit' to exit.

### Vision Webcam Demo
Run the vision webcam demo interactively:
```bash
bash scripts/run_vision_webcam.sh
```

### Diffusion Demo
Run the diffusion demo interactively:
```bash
bash scripts/run_c_diffusion.sh
```