"""
Comprehensive test script for all implemented kernels in the text-to-video pipeline.
Tests and benchmarks each kernel's performance with various input sizes.
"""

import numpy as np
import time
from video_decoder_kernel_asm import VideoDecoderKernelASM
from frame_interpolation_kernel_asm import FrameInterpolationKernelASM
from video_post_processing_kernel_asm import VideoPostProcessingKernelASM
from memory_efficient_attention_kernel_asm import MemoryEfficientAttentionKernelASM
from latent_space_projection_kernel_asm import LatentSpaceProjectionKernelASM
from cross_attention_kernel_asm import CrossAttentionKernelASM

def calculate_gflops(flops: int, time_ms: float) -> float:
    """Calculate GFLOPS from FLOPS count and execution time in milliseconds."""
    return (flops / (time_ms / 1000)) / 1e9

def count_kernel_flops(kernel_name: str, config: dict) -> int:
    """Calculate theoretical FLOPS for each kernel operation."""
    if kernel_name == "Video Decoder":
        # Each pixel requires 4 FMA operations for RGBA channels
        return (config["batch_size"] * config["num_frames"] * 
                config["height"] * config["width"] * config["channels"] * 8)
    
    elif kernel_name == "Frame Interpolation":
        # Each pixel requires 6 FMA operations for motion estimation and blending
        return (config["height"] * config["width"] * config["channels"] * 12)
    
    elif kernel_name == "Video Post-Processing":
        # Each pixel requires 8 FMA operations for enhancement and filtering
        return (config["batch_size"] * config["num_frames"] * 
                config["height"] * config["width"] * config["channels"] * 16)
    
    elif kernel_name == "Memory-Efficient Attention":
        # Q*K^T + softmax + V operations
        seq_len = config["seq_length"]
        return (config["batch_size"] * config["num_heads"] * 
                (2 * seq_len * seq_len * config["head_dim"] + seq_len * seq_len))
    
    elif kernel_name == "Latent Space Projection":
        # Matrix multiplication operations
        return (2 * config["batch_size"] * config["input_dim"] * config["output_dim"])
    
    elif kernel_name == "Cross-Attention":
        # Similar to Memory-Efficient Attention but with text and video tokens
        return (config["batch_size"] * 
                (2 * config["text_len"] * config["video_len"] * config["hidden_size"] + 
                 config["text_len"] * config["video_len"]))
    
    return 0

def run_benchmarks():
    """Run benchmarks for all implemented kernels."""
    print("Running comprehensive kernel benchmarks with GFLOPS measurements...\n")
    
    # Initialize all kernels
    kernels = {
        "Video Decoder": VideoDecoderKernelASM(),
        "Frame Interpolation": FrameInterpolationKernelASM(),
        "Video Post-Processing": VideoPostProcessingKernelASM(),
        "Memory-Efficient Attention": MemoryEfficientAttentionKernelASM(),
        "Latent Space Projection": LatentSpaceProjectionKernelASM(),
        "Cross-Attention": CrossAttentionKernelASM()
    }
    
    # Define test configurations for realistic video generation workload
    test_configs = {
        "Video Decoder": {
            "batch_size": 1,
            "num_frames": 16,
            "height": 256,
            "width": 256,
            "channels": 3
        },
        "Frame Interpolation": {
            "height": 256,
            "width": 256,
            "channels": 3
        },
        "Video Post-Processing": {
            "batch_size": 1,
            "num_frames": 16,
            "height": 256,
            "width": 256,
            "channels": 3
        },
        "Memory-Efficient Attention": {
            "batch_size": 1,
            "num_heads": 8,
            "seq_length": 256,
            "head_dim": 64
        },
        "Latent Space Projection": {
            "batch_size": 4,
            "input_dim": 512,
            "output_dim": 256
        },
        "Cross-Attention": {
            "batch_size": 2,
            "text_len": 77,
            "video_len": 256,
            "hidden_size": 768
        }
    }
    
    # Run benchmarks
    results = {}
    num_runs = 100  # Increased for more accurate measurements
    warmup_runs = 10  # Warmup runs to stabilize performance
    
    for kernel_name, kernel in kernels.items():
        print(f"\nBenchmarking {kernel_name}...")
        config = test_configs[kernel_name]
        
        # Calculate theoretical FLOPS
        flops = count_kernel_flops(kernel_name, config)
        
        # Generate test inputs based on kernel type
        if kernel_name == "Video Decoder":
            input_data = np.random.randn(
                config["batch_size"],
                config["num_frames"],
                config["height"],
                config["width"],
                config["channels"]
            ).astype(np.float32)
            
            # Warmup runs
            for _ in range(warmup_runs):
                kernel.decode_video(input_data)
            
            # Benchmark
            start_time = time.time()
            for _ in range(num_runs):
                kernel.decode_video(input_data)
            end_time = time.time()
            
        elif kernel_name == "Frame Interpolation":
            frame1 = np.random.randn(
                config["height"],
                config["width"],
                config["channels"]
            ).astype(np.float32)
            frame2 = np.random.randn(
                config["height"],
                config["width"],
                config["channels"]
            ).astype(np.float32)
            
            # Warmup runs
            for _ in range(warmup_runs):
                kernel.interpolate(frame1, frame2)
            
            # Benchmark
            start_time = time.time()
            for _ in range(num_runs):
                kernel.interpolate(frame1, frame2)
            end_time = time.time()
            
        elif kernel_name == "Video Post-Processing":
            video = np.random.randn(
                config["batch_size"],
                config["num_frames"],
                config["height"],
                config["width"],
                config["channels"]
            ).astype(np.float32)
            
            # Warmup runs
            for _ in range(warmup_runs):
                kernel.process_video(video)
            
            # Benchmark
            start_time = time.time()
            for _ in range(num_runs):
                kernel.process_video(video)
            end_time = time.time()
            
        elif kernel_name == "Memory-Efficient Attention":
            query = np.random.randn(
                config["batch_size"],
                config["num_heads"],
                config["seq_length"],
                config["head_dim"]
            ).astype(np.float32)
            key = query.copy()
            value = query.copy()
            
            # Warmup runs
            for _ in range(warmup_runs):
                kernel.compute_attention(query, key, value)
            
            # Benchmark
            start_time = time.time()
            for _ in range(num_runs):
                kernel.compute_attention(query, key, value)
            end_time = time.time()
            
        elif kernel_name == "Latent Space Projection":
            input_latent = np.random.randn(
                config["batch_size"],
                config["input_dim"]
            ).astype(np.float32)
            projection_matrix = np.random.randn(
                config["output_dim"],
                config["input_dim"]
            ).astype(np.float32)
            
            # Warmup runs
            for _ in range(warmup_runs):
                kernel.project_latent_space(input_latent, projection_matrix)
            
            # Benchmark
            start_time = time.time()
            for _ in range(num_runs):
                kernel.project_latent_space(input_latent, projection_matrix)
            end_time = time.time()
            
        elif kernel_name == "Cross-Attention":
            text_embeddings = np.random.randn(
                config["batch_size"],
                config["text_len"],
                config["hidden_size"]
            ).astype(np.float32)
            video_features = np.random.randn(
                config["batch_size"],
                config["video_len"],
                config["hidden_size"]
            ).astype(np.float32)
            
            # Warmup runs
            for _ in range(warmup_runs):
                kernel.apply_cross_attention(text_embeddings, video_features)
            
            # Benchmark
            start_time = time.time()
            for _ in range(num_runs):
                kernel.apply_cross_attention(text_embeddings, video_features)
            end_time = time.time()
        
        # Calculate and store results
        avg_time = (end_time - start_time) * 1000 / num_runs  # Convert to milliseconds
        gflops = calculate_gflops(flops, avg_time)
        results[kernel_name] = {
            "avg_time_ms": avg_time,
            "gflops": gflops,
            "config": config,
            "total_flops": flops
        }
        
        # Print results
        print(f"Average time: {avg_time:.2f} ms")
        print(f"GFLOPS: {gflops:.2f}")
        print(f"Total FLOPS: {flops:,}")
        print(f"Configuration: {config}")
    
    # Print summary
    print("\nBenchmark Summary:")
    print("=================")
    for kernel_name, result in results.items():
        print(f"\n{kernel_name}:")
        print(f"  Average time: {result['avg_time_ms']:.2f} ms")
        print(f"  GFLOPS: {result['gflops']:.2f}")
        print(f"  Total FLOPS: {result['total_flops']:,}")
        print(f"  Configuration: {result['config']}")
    
    print("\nAll benchmarks completed successfully!")

if __name__ == "__main__":
    run_benchmarks() 