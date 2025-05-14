#!/bin/bash

# Ensure the Hugging Face token is set as an environment variable
if [ -z "$HF_TOKEN" ]; then
  echo "Error: HF_TOKEN environment variable is not set."
  echo "Please set it before running the script: export HF_TOKEN='your_token_here'"
  exit 1
fi

# The script will use the HF_TOKEN from the environment

APP_PATH="$(dirname "$0")/app.py"
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

echo "Hugging Face token set: $HF_TOKEN"

# Path to c_inference directory
C_INFERENCE_PATH="../OPENtransformer/arm64_engine/core/c_inference"

# Default prompt
PROMPT="prime matrix transform"
MODEL="runwayml/stable-diffusion-v1-5"
STEPS=25
GUIDANCE=7.5
SIZE=512

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --prompt=*)
      PROMPT="${1#*=}"
      shift
      ;;
    --model=*)
      MODEL="${1#*=}"
      shift
      ;;
    --steps=*)
      STEPS="${1#*=}"
      shift
      ;;
    --guidance=*)
      GUIDANCE="${1#*=}"
      shift
      ;;
    --size=*)
      SIZE="${1#*=}"
      shift
      ;;
    *)
      # If first positional argument and doesn't start with --, assume it's the prompt
      if [[ $1 != --* ]] && [[ -z "$PROMPT_SET" ]]; then
        PROMPT="$1"
        PROMPT_SET=1
        shift
      else
        echo "Unknown option: $1"
        exit 1
      fi
      ;;
  esac
done

echo "Generating image with prompt: '$PROMPT'"
echo "Using model: $MODEL"
echo "Steps: $STEPS, Guidance: $GUIDANCE, Size: $SIZE"

# Run the script in c_inference directory
cd "$C_INFERENCE_PATH" || exit 1

# Create a safe filename from the prompt
SAFE_FILENAME=$(echo "$PROMPT" | tr ' ' '_' | tr -d '[:punct:]')
OUTPUT_PATH="generated_images/${SAFE_FILENAME}.png"

# Run the Python test script with virtual environment
./venv/bin/python test_py_direct.py \
  --prompt "$PROMPT" \
  --model "$MODEL" \
  --steps "$STEPS" \
  --guidance "$GUIDANCE" \
  --size "$SIZE" \
  --output "$OUTPUT_PATH"

# Check if image was generated successfully
if [ -f "$OUTPUT_PATH" ]; then
  echo "Image generated successfully: $OUTPUT_PATH"
  # Copy the image to the ui directory
  UI_OUTPUT_DIR="../../ui/generated_images"
  mkdir -p "$UI_OUTPUT_DIR"
  cp "$OUTPUT_PATH" "$UI_OUTPUT_DIR/"
  echo "Image copied to UI directory: $UI_OUTPUT_DIR/${SAFE_FILENAME}.png"
  
  # Open the image on macOS
  if [[ "$OSTYPE" == "darwin"* ]]; then
    open "$OUTPUT_PATH"
  fi
else
  echo "Failed to generate image"
fi 