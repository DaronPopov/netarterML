#!/bin/bash
# Benchmark script for comparing standard vs. ARM64 optimized performance

echo "=== Building Stable Diffusion C Wrapper ==="
make clean
make

echo ""
echo "=== Running ARM64 ASM Kernel Benchmark ==="

# Standard benchmark with memory optimization and ASM kernels (default)
./test_diffusion \
  --prompt "A digital painting of a futuristic city with flying cars" \
  --steps 10 \
  --benchmark 3 \
  --warmup 1 \
  --output optimized_output.png

echo ""
echo "=== Running Standard Implementation Benchmark (No Optimizations) ==="

# Benchmark without memory optimizations or ASM kernels
./test_diffusion \
  --prompt "A digital painting of a futuristic city with flying cars" \
  --steps 10 \
  --benchmark 3 \
  --warmup 1 \
  --no-optimize \
  --output standard_output.png

echo ""
echo "=== Benchmarks Complete ==="
echo "You can compare the generated images for quality differences:"
echo "  - optimized_output.ppm: Generated with ASM kernels and memory optimizations"
echo "  - standard_output.ppm: Generated without optimizations"
echo ""
echo "To convert PPM to PNG, install ImageMagick and run:"
echo "  convert optimized_output.ppm optimized_output.png"
echo "  convert standard_output.ppm standard_output.png" 