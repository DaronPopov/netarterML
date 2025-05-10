#!/bin/bash
# Quick inference with Dreamshaper XL Lightning model at 728x728 resolution and 3 steps
# This script does NOT download or convert the model - it uses the existing model

# Check if a prompt was provided
if [ $# -lt 1 ]; then
    echo "Usage: $0 \"your prompt here\" [output_filename]"
    echo "Example: $0 \"a cyberpunk city at night\" cyberpunk.png"
    exit 1
fi

# Get the prompt from the first argument
PROMPT="$1"

# Set the output filename (either from second argument or generated from prompt)
if [ -n "$2" ]; then
    # Use provided output filename
    OUTPUT_FILENAME="$2"
else
    # Generate filename from prompt (first 4 words)
    OUTPUT_FILENAME=$(echo "$PROMPT" | tr ' ' '_' | cut -d'_' -f1-4)
    OUTPUT_FILENAME="${OUTPUT_FILENAME}.png"
fi

# Ensure the output path is in the generated_images directory
OUTPUT_PATH="generated_images/$OUTPUT_FILENAME"

# Create output directory if it doesn't exist
mkdir -p generated_images

# Run the inference script with provided arguments
echo "Generating image with:"
echo "  - Resolution: 728x728"
echo "  - Steps: 3"
echo "  - Prompt: $PROMPT"
echo "  - Output: $OUTPUT_PATH"
echo ""

# Run the actual inference
python inference_dreamshaper_xl_lightning.py \
    --prompt "$PROMPT" \
    --output "$OUTPUT_PATH" \
    --steps 3 \
    --width 728 \
    --height 728 \
    --guidance 7.0

# Open the image on macOS if the generation was successful
if [ -f "$OUTPUT_PATH" ] && [ "$(uname)" = "Darwin" ]; then
    echo "Opening generated image..."
    open "$OUTPUT_PATH"
fi

echo "Done!" 