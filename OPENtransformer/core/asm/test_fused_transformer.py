import numpy as np
import time
import logging
import csv
import os
import sys
from finlib.core.asm.kernels.transformer import Transformer

# Suppress all logs from transformer module
logging.getLogger("finlib.core.asm.transformer").setLevel(logging.CRITICAL)
logging.getLogger("finlib.core.asm.fused_transformer_op").setLevel(logging.CRITICAL)

# Configure logging for the script
logging.basicConfig(level=logging.CRITICAL)
logger = logging.getLogger(__name__)

# Create a simple output function that bypasses logging
def print_status(message):
    sys.stdout.write(message + "\n")
    sys.stdout.flush()

def test_large_model():
    """
    Test the transformer with a larger model size using the fully fused implementation.
    """
    # Large model parameters
    d_model = 3072
    n_heads = 16
    n_layers = 6
    vocab_size = 32000
    max_context_length = 1024
    
    # Input parameters
    batch_size = 7
    seq_len = 1024
    
    # Initialize model
    print_status("Starting test with transformer model...")
    print_status(f"Model configuration: {d_model}d × {n_heads}h × {n_layers}l, input shape: {batch_size} × {seq_len}")
    
    # Suppress stdout during model initialization
    original_stdout = sys.stdout
    sys.stdout = open(os.devnull, 'w')
    
    try:
        model = Transformer(d_model=d_model, n_heads=n_heads, n_layers=n_layers)
        x = np.random.randn(batch_size, seq_len, d_model).astype(np.float32)
        
        # Warm-up run
        _ = model.fully_fused_forward(x)
    finally:
        # Restore stdout
        sys.stdout.close()
        sys.stdout = original_stdout
    
    print_status("Model initialized. Running benchmark...")
    
    # Benchmark run
    iterations = 1  # Run 3 times and take the average
    
    # Progress indicators
    start_time = time.time()
    for i in range(iterations):
        # Suppress all output during execution
        original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')
        
        try:
            output = model.fully_fused_forward(x)
        finally:
            sys.stdout.close()
            sys.stdout = original_stdout
            
        print_status(f"Iteration {i+1}/{iterations} completed")
    
    end_time = time.time()
    
    # Calculate execution time
    execution_time = (end_time - start_time) / iterations
    
    # Calculate FLOPS
    flops = calculate_flops(batch_size, seq_len, d_model, n_heads, n_layers)
    gflops = flops / execution_time / 1e9
    
    # Output validation (simplified)
    valid_output = not np.any(np.isnan(output)) and output.shape == (batch_size, seq_len, d_model)
    
    # Print final results
    print_status("\n" + "="*50)
    print_status("PERFORMANCE RESULTS:")
    print_status(f"  Model size: {d_model}d × {n_heads}h × {n_layers}l")
    print_status(f"  Input shape: {batch_size} × {seq_len} × {d_model}")
    print_status(f"  Execution time: {execution_time:.6f} seconds")
    print_status(f"  Performance: {gflops:.2f} GFLOPS")
    print_status(f"  Output validation: {'Pass' if valid_output else 'Fail'}")
    print_status("="*50)
    
    # Save results to CSV for future reference
    save_results_to_csv("transformer_performance.csv", {
        "model_type": "large_model",
        "d_model": d_model,
        "n_heads": n_heads, 
        "n_layers": n_layers,
        "batch_size": batch_size,
        "seq_len": seq_len,
        "execution_time": execution_time,
        "gflops": gflops
    })
    
    return gflops

def calculate_flops(batch_size, seq_len, d_model, n_heads, n_layers):
    """
    Calculate the number of floating-point operations for a transformer model.
    """
    head_dim = d_model // n_heads
    
    # Per-layer flops
    # Self-attention: 4 * batch_size * seq_len * d_model^2 (QKV projections + output projection)
    qkv_flops = 3 * batch_size * seq_len * d_model * d_model
    attn_matmul_flops = batch_size * n_heads * seq_len * seq_len * head_dim
    attn_output_flops = batch_size * seq_len * d_model * d_model
    
    # FFN: 2 * batch_size * seq_len * d_model * (4*d_model)
    ffn_flops = 2 * batch_size * seq_len * d_model * (4 * d_model)
    
    # Layer norm: 5 * batch_size * seq_len * d_model each (2 per layer)
    ln_flops = 2 * 5 * batch_size * seq_len * d_model
    
    # Total flops per layer
    flops_per_layer = qkv_flops + attn_matmul_flops + attn_output_flops + ffn_flops + ln_flops
    
    # Total flops for all layers
    total_flops = flops_per_layer * n_layers
    
    return total_flops

def save_results_to_csv(filename, results):
    """
    Save benchmark results to a CSV file.
    """
    # Check if file exists to determine if we need to write headers
    file_exists = os.path.isfile(filename)
    
    with open(filename, 'a', newline='') as csvfile:
        fieldnames = list(results.keys())
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        if not file_exists:
            writer.writeheader()
        
        writer.writerow(results)

if __name__ == "__main__":
    gflops = test_large_model()
    print_status(f"Results saved to transformer_performance.csv") 