#!/bin/bash

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." &> /dev/null && pwd )"

# Color output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Default values
MODEL="runwayml/stable-diffusion-v1-5"
STEPS=25
WIDTH=512
HEIGHT=512
GUIDANCE=7.5
SEED=0
ENGINE="c"  # Default to C engine

# Function to show help
show_help() {
    echo -e "${BLUE}Image Generation Script${NC}"
    echo -e "Usage: $0 [options] \"your prompt here\""
    echo
    echo "Options:"
    echo "  -h, --help          Show this help message"
    echo "  -m, --model MODEL   Specify model ID (default: $MODEL)"
    echo "  -s, --steps STEPS   Number of inference steps (default: $STEPS)"
    echo "  -w, --width WIDTH   Image width (default: $WIDTH)"
    echo "  --height HEIGHT     Image height (default: $HEIGHT)"
    echo "  -g, --guidance G    Guidance scale (default: $GUIDANCE)"
    echo "  --seed SEED         Random seed (default: $SEED = random)"
    echo "  -e, --engine E      Engine to use (c/pytorch) (default: $ENGINE)"
    echo "  --offline          Run in offline mode"
    echo
    echo "Examples:"
    echo "  $0 \"A beautiful sunset over the ocean\""
    echo "  $0 -s 50 -w 768 --height 512 \"A fantasy landscape\""
    echo "  $0 -e pytorch --offline \"A portrait of a cat\""
}

# Parse arguments
PARAMS=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help)
            show_help
            exit 0
            ;;
        -m|--model)
            MODEL="$2"
            shift 2
            ;;
        -s|--steps)
            STEPS="$2"
            shift 2
            ;;
        -w|--width)
            WIDTH="$2"
            shift 2
            ;;
        --height)
            HEIGHT="$2"
            shift 2
            ;;
        -g|--guidance)
            GUIDANCE="$2"
            shift 2
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        -e|--engine)
            ENGINE="$2"
            shift 2
            ;;
        --offline)
            PARAMS="$PARAMS --offline"
            shift
            ;;
        *)
            if [[ -z "$PROMPT" ]]; then
                PROMPT="$1"
            else
                PARAMS="$PARAMS $1"
            fi
            shift
            ;;
    esac
done

# Check if prompt is provided
if [[ -z "$PROMPT" ]]; then
    echo -e "${YELLOW}Error: No prompt provided${NC}"
    show_help
    exit 1
fi

# Run the image generation
cd "$PROJECT_ROOT" && python ui/engines/image_gen_engine.py \
    --prompt "$PROMPT" \
    --model "$MODEL" \
    --steps "$STEPS" \
    --width "$WIDTH" \
    --height "$HEIGHT" \
    --guidance "$GUIDANCE" \
    --seed "$SEED" \
    --engine "$ENGINE" \
    $PARAMS 