#!/bin/bash

# Set your Hugging Face token here. Replace 'your_actual_token_here' with your real token.
export HF_TOKEN="your_actual_token_here"

if [ -z "$HF_TOKEN" ] || [ "$HF_TOKEN" == "your_actual_token_here" ]; then
  echo "Please edit this script (set_hf_token.sh) and replace 'your_actual_token_here' with your actual Hugging Face token."
else
  echo "Hugging Face token set to: $HF_TOKEN"
fi

# Source this script with:
# source set_hf_token.sh 