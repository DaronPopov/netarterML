#!/bin/bash
# Quick script to run the C inference engine for image generation

# Color output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Default model
MODEL="runwayml/stable-diffusion-v1-5"

# Check if Python is available
if ! command -v python3 &> /dev/null && ! command -v python &> /dev/null; then
    echo "Error: Python not found. Please install Python 3.7 or higher"
    exit 1
fi

# Determine Python command
PYTHON="python"
if command -v python3 &> /dev/null; then
    PYTHON="python3"
fi

# Help message
show_help() {
    echo -e "${BLUE}C Inference Engine Image Generator${NC}"
    echo -e "Usage: $0 [options] \"your prompt here\""
    echo
    echo "Options:"
    echo "  -h, --help          Show this help message"
    echo "  -m, --model MODEL   Specify model ID (default: $MODEL)"
    echo "  -s, --steps STEPS   Number of inference steps (default: 25)"
    echo "  -w, --width WIDTH   Image width (default: 512)"
    echo "  -h, --height HEIGHT Image height (default: 512)"
    echo "  -g, --guidance GUIDANCE  Guidance scale (default: 7.5)"
    echo "  --seed SEED         Random seed (default: 0 = random)"
    echo
    echo "Example:"
    echo "  $0 \"A beautiful sunset over the ocean\""
    echo "  $0 -s 50 -w 768 -h 512 \"A fantasy landscape with mountains and waterfalls\""
}

# Parse arguments
PARAMS=""
while (( "$#" )); do
    case "$1" in
        -h|--help)
            show_help
            exit 0
            ;;
        -m|--model)
            if [ -n "$2" ] && [ ${2:0:1} != "-" ]; then
                MODEL=$2
                shift 2
            else
                echo "Error: Argument for $1 is missing" >&2
                exit 1
            fi
            ;;
        -s|--steps)
            if [ -n "$2" ] && [ ${2:0:1} != "-" ]; then
                STEPS=$2
                shift 2
            else
                echo "Error: Argument for $1 is missing" >&2
                exit 1
            fi
            ;;
        -w|--width)
            if [ -n "$2" ] && [ ${2:0:1} != "-" ]; then
                WIDTH=$2
                shift 2
            else
                echo "Error: Argument for $1 is missing" >&2
                exit 1
            fi
            ;;
        --height)
            if [ -n "$2" ] && [ ${2:0:1} != "-" ]; then
                HEIGHT=$2
                shift 2
            else
                echo "Error: Argument for $1 is missing" >&2
                exit 1
            fi
            ;;
        -g|--guidance)
            if [ -n "$2" ] && [ ${2:0:1} != "-" ]; then
                GUIDANCE=$2
                shift 2
            else
                echo "Error: Argument for $1 is missing" >&2
                exit 1
            fi
            ;;
        --seed)
            if [ -n "$2" ] && [ ${2:0:1} != "-" ]; then
                SEED=$2
                shift 2
            else
                echo "Error: Argument for $1 is missing" >&2
                exit 1
            fi
            ;;
        --) # end argument parsing
            shift
            break
            ;;
        -*|--*=) # unsupported flags
            echo "Error: Unsupported flag $1" >&2
            exit 1
            ;;
        *) # preserve positional arguments
            PARAMS="$PARAMS $1"
            shift
            ;;
    esac
done

# Set positional arguments in their proper place
eval set -- "$PARAMS"

# Check if prompt is provided
if [ -z "$1" ]; then
    echo -e "${YELLOW}Error: No prompt provided${NC}" >&2
    show_help
    exit 1
fi

PROMPT="$1"
echo -e "${GREEN}Generating image for prompt:${NC} $PROMPT"

# Build command
CMD="$PYTHON image_gen_engine.py --prompt \"$PROMPT\" --model \"$MODEL\""

if [ -n "$STEPS" ]; then
    CMD="$CMD --steps $STEPS"
fi

if [ -n "$WIDTH" ]; then
    CMD="$CMD --width $WIDTH"
fi

if [ -n "$HEIGHT" ]; then
    CMD="$CMD --height $HEIGHT"
fi

if [ -n "$GUIDANCE" ]; then
    CMD="$CMD --guidance $GUIDANCE"
fi

if [ -n "$SEED" ]; then
    CMD="$CMD --seed $SEED"
fi

# Run the command
echo -e "${BLUE}Running:${NC} $CMD"
eval $CMD 