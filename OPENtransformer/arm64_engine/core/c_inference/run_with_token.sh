#!/bin/bash

# Change to the script's directory
cd "$(dirname "$0")"

# Source the token script
source ./set_hf_token.sh

# Default prompt
PROMPT="prime matrix transform"
MODEL="runwayml/stable-diffusion-v1-5"

# Parse command line arguments
if [ $# -gt 0 ]; then
    PROMPT="$1"
fi

if [ $# -gt 1 ]; then
    MODEL="$2"
fi

# Make the script executable if not already
chmod +x ./run_test.sh

# Run the image generation script
./run_test.sh "$PROMPT" --model="$MODEL"

echo "Image generation complete!" 