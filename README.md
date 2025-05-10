# NetArteryML

A comprehensive machine learning toolkit for medical imaging, vision, and language models.

## Version Information
- Current Version: 0.1.0
- Python Compatibility: 3.8 - 3.11
- CUDA Compatibility: 11.7 - 12.1

## Features

- Medical Image Analysis
- Vision API (Image Classification)
- Language Chat Interface
- Image Generation (Multiple Options)
  - Simple Diffusion
  - Standard Diffusion
  - Multimodal Pipeline

## Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (recommended for faster processing)
- Git
- C/ASM Backend (for diffusion and LLM features)
  - Required for optimal performance
  - See "C/ASM Backend Setup" section below

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/netarteryML.git
cd netarteryML
```

2. Create and activate a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the package and dependencies:
```bash
pip install -r requirements.txt
pip install -e .
```

4. Set up Hugging Face token (required for model downloads):
```bash
export HUGGINGFACE_TOKEN="your_token_here"  # On Windows: set HUGGINGFACE_TOKEN=your_token_here
```

5. Build the C/ASM backend (required for diffusion and LLM features):
```bash
./scripts/build_c_inference.sh
```

## C/ASM Backend Setup

The C/ASM backend is required for optimal performance of diffusion and LLM features. To set it up:

1. Ensure you have the required build tools:
   - GCC/G++ (Linux/Mac)
   - Visual Studio Build Tools (Windows)
   - CMake 3.15 or higher

2. Build the backend:
```bash
./scripts/build_c_inference.sh
```

3. Set the Python path (add to your shell profile):
```bash
export PYTHONPATH="/path/to/your/project/OPENtransformer/arm64_engine/core/c_inference:$PYTHONPATH"
```

## Model Management

### Model Caching
- Models are cached in `~/.cache/huggingface/hub/` by default
- First run will download required models
- Subsequent runs will use cached models
- Clear cache with: `rm -rf ~/.cache/huggingface/hub/`

### Storage Requirements
- Base installation: ~500MB
- Full model cache: ~10GB
- Temporary files: ~2GB during generation

## Available Demos

### 1. Medical Image Analysis
Analyzes medical images using advanced AI models.
```bash
./scripts/run_demo.sh medical
```

### 2. Vision API
Performs image classification using the vision API.
```bash
./scripts/run_demo.sh vision
```

### 3. Chat Interface
Interactive chat interface powered by language models.
```bash
./scripts/run_demo.sh chat
```

### 4. Image Generation
Multiple options for image generation:

#### a. Simple Diffusion
Fast image generation using Stable Diffusion with minimal steps.
```bash
./scripts/run_demo.sh simple-diffusion --prompt "your prompt here"
```

#### b. Standard Diffusion
Full-featured diffusion model for high-quality image generation.
```bash
./scripts/run_demo.sh diffusion
```

#### c. Multimodal Pipeline
Advanced image generation with multimodal capabilities.
```bash
./scripts/run_demo.sh generate --prompt "your prompt here"
```

## Demo Options

Most demos support the following options:
- `--offline`: Run in offline mode (if supported)
- `--help`: Show help message for the specific demo

Example:
```bash
./scripts/run_demo.sh simple-diffusion --prompt "a beautiful sunset" --output "my_image.png"
```

## Output

- Generated images are saved in the `generated_images` directory by default
- Medical analysis results are displayed in the terminal
- Chat interface runs interactively in the terminal
- Model cache is stored in `~/.cache/huggingface/hub/`

## Troubleshooting

1. **ModuleNotFoundError**: Ensure you've installed the package with `pip install -e .`

2. **CUDA/GPU Issues**: 
   - Check if CUDA is properly installed
   - Verify GPU compatibility
   - The system will fall back to CPU if GPU is not available

3. **Model Download Issues**:
   - Verify your Hugging Face token is set correctly
   - Check internet connection
   - Ensure sufficient disk space

4. **Permission Issues**:
   - Make scripts executable: `chmod +x scripts/*.sh`
   - Ensure proper permissions for output directories

5. **C/ASM Backend Issues**:
   - Ensure build tools are installed
   - Check PYTHONPATH is set correctly
   - Verify backend was built successfully
   - Check system architecture compatibility

6. **Model Cache Issues**:
   - Clear cache if models are corrupted
   - Check available disk space
   - Verify Hugging Face token permissions

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

MIT License

## Contact

- GitHub Issues: [Create an issue](https://github.com/yourusername/netarteryML/issues)
- Email: your.email@example.com