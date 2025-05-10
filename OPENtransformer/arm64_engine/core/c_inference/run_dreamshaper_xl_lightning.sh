#!/bin/bash
# Run the Dreamshaper XL Lightning model with appropriate settings for fast, high-quality generations

# Set the Hugging Face token if you have one in an environment variable
if [ -n "$HF_TOKEN" ]; then
    export HF_TOKEN="$HF_TOKEN"
elif [ -f "$HOME/.hf_token" ]; then
    export HF_TOKEN=$(cat "$HOME/.hf_token")
fi

# Check if model already exists instead of re-downloading
MODEL_PATH="models/dreamshaper-xl-lightning"
if [ -d "$MODEL_PATH" ] && [ -f "$MODEL_PATH/binary_conversion_complete" ]; then
    echo "Dreamshaper XL Lightning model already exists, skipping download..."
else
    echo "Downloading Dreamshaper XL Lightning model..."
    python download_dreamshaper_xl_lightning.py
fi

# Make sure the output directory exists
mkdir -p generated_images

# Run the dedicated example script with modified resolution/steps
echo "Running Dreamshaper XL Lightning example (728x728, 6 steps)..."
python dreamshaper_xl_lightning_example.py

# Create a customizable prompt for quick testing
echo "Running with custom prompt..."
PROMPT="a detailed photo of a fantasy landscape with mountains, lakes, and magical creatures, ultra realistic, 8k, trending on artstation"
OUTPUT_PATH="generated_images/custom_dreamshaper_xl_lightning.png"

# Run with Python directly for greater flexibility
python -c "
from easy_diffusion_api import EasyDiffusionAPI
import os

# Create API instance
api = EasyDiffusionAPI()

# Register and set active model
api.register_model('dreamshaper-xl-lightning', 'Lykon/dreamshaper-xl-lightning')
api.set_active_model('dreamshaper-xl-lightning')

# Generate with ultra-fast settings for the Lightning model
result = api.generate_image(
    prompt='$PROMPT',
    steps=6,           # Ultra-fast generation with only 6 steps
    width=728,         # Lower resolution for faster generation
    height=728,
    guidance=7.0,      # Slightly lower guidance for more creative results
    output_path='$OUTPUT_PATH'
)

if result:
    print(f'Successfully generated: $OUTPUT_PATH')
    # Show the image on macOS
    if os.path.exists('$OUTPUT_PATH') and os.system('which open') == 0:
        os.system('open \"$OUTPUT_PATH\"')
else:
    print('Generation failed')
"

echo "Done!" 