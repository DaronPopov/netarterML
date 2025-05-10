# Installation Guide

This guide will help you set up the Local AI Engine on your system.

## Prerequisites

- Python 3.8 or higher
- pip (Python package installer)
- Git
- A Hugging Face account and API token
- C/C++ compiler (gcc/clang)
- Make

## Quick Installation

1. **Clone the Repository**
   ```bash
   git clone <repository-url>
   cd netarteryML
   ```

2. **Create a Virtual Environment (Recommended)**
   ```bash
   # Create virtual environment
   python -m venv venv

   # Activate virtual environment
   # On macOS/Linux:
   source venv/bin/activate
   # On Windows:
   .\venv\Scripts\activate
   ```

3. **Install Everything**
   ```bash
   # This single command installs all dependencies and build tools
   pip install -r requirements.txt
   ```

4. **Build C/C++ Modules**
   ```bash
   # Navigate to the core inference directory
   cd OPENtransformer/arm64_engine/core/c_inference
   
   # Build the C/C++ modules
   make clean
   make
   
   # Return to root directory
   cd ../../../../..
   ```

5. **Set Up Environment Variables**
   ```bash
   # Set your Hugging Face token (replace with your actual token)
   export HF_TOKEN=your_token_here

   # Set the Python path for core inference modules
   export PYTHONPATH=OPENtransformer/arm64_engine/core/c_inference
   ```

6. **Verify Installation**
   ```bash
   python tests/test_vision_download.py
   ```

## Running Examples

After installation, you can try the example scripts:

- **Chat Example:**
  ```bash
  python examples/chat/tinyllama_chat.py
  ```

- **Vision Example:**
  ```bash
  python examples/vision/vision_api.py
  ```

- **Diffusion Example:**
  ```bash
  python examples/diffusion/easy_diffusion_example.py
  ```

- **Multimodal Example:**
  ```bash
  python examples/multimodal/multimodal_pipeline.py
  ```

## Troubleshooting

### Common Issues

1. **ImportError: No module named 'py_diffusion_interface'**
   - Make sure you have set the `PYTHONPATH` environment variable correctly
   - Verify you're in the correct directory
   - Ensure the C/C++ modules were built successfully

2. **Build Errors**
   - Make sure you have a C/C++ compiler installed
   - On macOS: `xcode-select --install`
   - On Ubuntu/Debian: `sudo apt-get install build-essential`
   - On Windows: Install Visual Studio Build Tools

3. **Hugging Face Token Errors**
   - Ensure you have set your `HF_TOKEN` environment variable
   - Verify your token is valid and has the necessary permissions

4. **Missing Dependencies**
   - Try running `pip install -r requirements.txt` again
   - Check if you're using the correct Python version

5. **GPU/CPU Warnings**
   - Some warnings about CUDA or GPU are informational
   - The system will fall back to CPU if GPU is not available

### Getting Help

If you encounter any issues not covered here:
1. Check the [README.md](README.md) for additional information
2. Review the [PHILOSOPHY.md](PHILOSOPHY.md) for project context
3. Open an issue on the repository's issue tracker

## System Requirements

- **Minimum:**
  - 8GB RAM
  - 10GB free disk space
  - Python 3.8+
  - C/C++ compiler (gcc/clang)
  - Make

- **Recommended:**
  - 16GB+ RAM
  - 20GB+ free disk space
  - GPU with CUDA support (for faster inference) 