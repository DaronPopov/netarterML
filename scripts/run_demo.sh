#!/bin/bash

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." &> /dev/null && pwd )"

# Function to show help
show_help() {
    echo "Usage: $0 [demo_type] [options]"
    echo
    echo "Available demos:"
    echo "  chat        - Language chat demo (LLM)"
    echo "  medical     - Medical imaging analysis demo"
    echo "  vision      - Vision API demo (image classification)"
    echo "  generate    - Image generation demo (diffusion)"
    echo "  diffusion   - Standard diffusion image generation demo"
    echo "  simple-diffusion    - Simple diffusion image generation demo"
    echo
    echo "Options:"
    echo "  --offline   - Run in offline mode (if supported)"
    echo "  --help      - Show this help message"
    echo
    echo "Examples:"
    echo "  $0 chat"
    echo "  $0 medical"
    echo "  $0 vision"
    echo "  $0 generate --prompt \"your prompt\""
    echo "  $0 diffusion"
    echo "  $0 simple-diffusion --prompt \"your prompt\""
}

# Function to run Python script with proper path
run_python_script() {
    local script_path="$1"
    shift
    cd "$PROJECT_ROOT" && python "$script_path" "$@"
}

# Check if demo type is provided
if [ $# -eq 0 ]; then
    show_help
    exit 1
fi

DEMO_TYPE="$1"
shift

case "$DEMO_TYPE" in
    "chat")
        run_python_script "examples/chat/llm_chat_example.py" "$@"
        ;;
    "medical")
        run_python_script "examples/multimodal/multimodal_analysis.py" "$@"
        ;;
    "vision")
        run_python_script "examples/vision/vision_api.py" "$@"
        ;;
    "generate")
        run_python_script "examples/multimodal/multimodal_pipeline.py" "$@"
        ;;
    "diffusion")
        run_python_script "examples/diffusion/easy_diffusion_example.py" "$@"
        ;;
    "simple-diffusion")
        run_python_script "examples/diffusion/simple_diffusion.py" "$@"
        ;;
    "--help"|"-h")
        show_help
        ;;
    *)
        echo "Error: Unknown demo type '$DEMO_TYPE'"
        show_help
        exit 1
        ;;
esac 