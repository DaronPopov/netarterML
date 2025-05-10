#!/bin/bash
# Build and test C inference engine integration

set -e  # Exit on any error

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}==== C Inference Engine Integration Build & Test ====${NC}"

# Get the absolute path to the c_inference directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../../../" && pwd)"
UI_ENGINE_DIR="$PROJECT_ROOT/ui/engines"

# Check Python
echo -e "\n${BLUE}Checking Python environment...${NC}"
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}Python3 is not installed. Please install Python 3.7 or higher.${NC}"
    exit 1
fi

PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
echo -e "Python version: ${GREEN}$PYTHON_VERSION${NC}"

# Check for required packages
echo -e "\n${BLUE}Checking required packages...${NC}"
python3 -c "import diffusers, torch, numpy, PIL" 2>/dev/null || {
    echo -e "${YELLOW}Installing required packages...${NC}"
    pip install -r "$SCRIPT_DIR/requirements.txt"
}
echo -e "${GREEN}Required packages are installed${NC}"

# Build the C inference engine
echo -e "\n${BLUE}Building C inference engine...${NC}"
cd "$SCRIPT_DIR"
make clean
make

if [ $? -ne 0 ]; then
    echo -e "${RED}Build failed. Please check the error messages above.${NC}"
    exit 1
fi
echo -e "${GREEN}Build successful${NC}"

# Install the Python package in development mode
echo -e "\n${BLUE}Installing Python package...${NC}"
pip install -e "$SCRIPT_DIR"

# Run a basic test
echo -e "\n${BLUE}Running basic test...${NC}"
cd "$SCRIPT_DIR"
python3 -c "from py_diffusion_interface import run_inference; print('Module imported successfully')" || {
    echo -e "${RED}Failed to import module. Check the build output above.${NC}"
    exit 1
}

# Test the integration with the UI
echo -e "\n${BLUE}Testing UI integration...${NC}"
cd "$UI_ENGINE_DIR"

if [ -f "image_gen_engine.py" ]; then
    echo -e "${GREEN}Found image_gen_engine.py${NC}"
    echo -e "\n${YELLOW}Running a quick test with the C inference engine...${NC}"
    python image_gen_engine.py --prompt "A test image" --steps 2 --engine c
else
    echo -e "${RED}Cannot find image_gen_engine.py in $UI_ENGINE_DIR${NC}"
    exit 1
fi

echo -e "\n${GREEN}==== C Inference Engine Integration Complete ====${NC}"
echo -e "You can now use the C inference engine through the UI with:"
echo -e "${BLUE}cd $UI_ENGINE_DIR && python image_gen_engine.py --interactive --engine c${NC}"
echo -e "or"
echo -e "${BLUE}cd $UI_ENGINE_DIR && python image_gen_engine.py --prompt \"Your prompt\" --engine c${NC}" 