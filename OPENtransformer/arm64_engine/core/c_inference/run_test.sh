#!/bin/bash

# Ensure working in the correct directory
cd "$(dirname "$0")"

# Ensure the Hugging Face token is set as an environment variable
if [ -z "$HF_TOKEN" ]; then
  echo "Error: HF_TOKEN environment variable is not set."
  echo "Please set it before running the script: export HF_TOKEN='your_token_here'"
  exit 1
fi

# The HF_TOKEN environment variable should be used by Hugging Face libraries.
echo "Using HF_TOKEN from environment: $HF_TOKEN"

# Ensure the generated_images directory exists
mkdir -p generated_images

# Default model
MODEL_ID="runwayml/stable-diffusion-v1-5"

# Parse command line arguments
ARGS=()
for i in "$@"; do
  case $i in
    --model=*)
      MODEL_ID="${i#*=}"
      # Skip this argument as we'll add it later in the proper format
      ;;
    *)
      # Pass other arguments through
      ARGS+=("$i")
      ;;
  esac
done

# Pass prompt as an argument
prompt="a beautiful landscape with mountains, trees, and a lake"
if [ "${ARGS[0]}" != "" ] && [[ ! "${ARGS[0]}" == --* ]]; then
    prompt="${ARGS[0]}"
    # Remove the prompt from the arguments
    ARGS=("${ARGS[@]:1}")
fi

# Create a safe filename from the prompt
safe_filename=$(echo "$prompt" | tr ' ' '_' | tr -d '[:punct:]')

# Create the output path
output_path="generated_images/${safe_filename}.png"

# Add the model parameter
ARGS+=("--model" "$MODEL_ID")

# Run the Python test script directly with the virtual environment Python
./venv/bin/python test_py_direct.py --prompt "$prompt" --output "$output_path" "${ARGS[@]}" 