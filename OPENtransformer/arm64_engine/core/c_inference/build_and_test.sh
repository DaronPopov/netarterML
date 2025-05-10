#!/bin/bash
# Build and test the C wrapper for Stable Diffusion

set -e  # Exit on error

echo "==== Building C wrapper for Stable Diffusion ===="

# Create __init__.py file for Python module directory
mkdir -p $(dirname "$0")
touch $(dirname "$0")/__init__.py

# Install Python dependencies if needed
echo "Checking Python dependencies..."
pip install -e .

# Build the C wrapper
echo "Building C wrapper..."
make clean
make

# Check if build succeeded
if [ $? -ne 0 ]; then
    echo "Build failed!"
    exit 1
fi

echo "Build succeeded!"

# Run the Python test directly
echo "==== Testing Python interface ===="
python py_diffusion_interface.py

# Run the C test with default parameters
echo "==== Testing C wrapper ===="
./test_diffusion --steps 5

echo "All tests completed successfully!" 