# NetArterML

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
export HUGGINGFACE_TOKEN="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"  # On Windows: set HUGGINGFACE_TOKEN=your_token_here
```



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
you can setup ur own normal image gen pipeline with my system but i setup a multimodal for the sake of the demo to 

#### a  Diffusion
Fast image generation using Stable Diffusion with minimal steps.
```bash
./scripts/run_demo.sh simple-diffusion --prompt "time annhilation" --steps 15
```

Note: This demo uses the ostris/Flex.2-preview model by default, which provides high-quality image generation.

#### c. Multimodal Pipeline
Advanced image generation with multimodal capabilities.
```bash
./scripts/run_demo.sh generate --prompt "fat tensor" 
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

- GitHub Issues: [Create an issue](https://github.com/yourusername/netarterML/issues)
- Email: daron94545@gmail.com