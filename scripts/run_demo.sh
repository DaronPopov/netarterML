#!/bin/bash

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." &> /dev/null && pwd )"

# Function to show help
show_help() {
    echo "Usage: $0 [demo_type] [options]"
    echo
    echo "Available demos:"
    echo "  generate    - Image generation demo"
    echo "  medical     - Medical imaging analysis demo"
    echo "  chat        - Language chat demo"
    echo "  webcam      - Webcam captioning demo"
    echo "  code        - Code generation demo"
    echo
    echo "Options:"
    echo "  --offline   - Run in offline mode"
    echo "  --help      - Show this help message"
    echo
    echo "Examples:"
    echo "  $0 generate --prompt \"your prompt\" --offline"
    echo "  $0 medical --image path/to/image.jpg"
    echo "  $0 webcam --offline"
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
    "generate")
        run_python_script "examples/vision/multimodal_pipeline.py" "$@"
        ;;
    "medical")
        run_python_script "examples/medical/medical_analysis.py" "$@"
        ;;
    "chat")
        run_python_script "examples/chat/chat_interface.py" "$@"
        ;;
    "webcam")
        run_python_script "examples/vision/webcam_caption.py" "$@"
        ;;
    "code")
        run_python_script "examples/code/code_generator.py" "$@"
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