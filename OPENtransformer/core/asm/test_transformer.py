import numpy as np
import logging
import ctypes
import time
from finlib.core.asm.kernels.transformer import Transformer
from finlib.core.asm.kernels.position_embedding import execute_kernel as position_embedding_kernel

# Set logging to INFO level to reduce verbosity
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def calculate_transformer_flops(batch_size, seq_len, d_model, n_heads, n_layers):
    """Calculate theoretical FLOPs for one forward pass of the transformer."""
    # Self-attention FLOPs
    attention_flops = (3 * batch_size * seq_len * d_model * d_model + 
                      batch_size * n_heads * seq_len * seq_len * (d_model // n_heads) +
                      batch_size * seq_len * d_model * d_model)
    
    # FFN FLOPs (per layer)
    ffn_flops_per_layer = (batch_size * seq_len * d_model * (4 * d_model) +
                          batch_size * seq_len * (4 * d_model) * d_model)
    
    # Total FLOPs for all layers
    total_flops = n_layers * (attention_flops + ffn_flops_per_layer)
    
    return total_flops

def test_transformer():
    """Test the transformer with all custom kernels."""
    try:
        # Model parameters
        d_model = 512
        n_heads = 8
        n_layers = 6
        vocab_size = 32000
        max_seq_len = 1024
        
        # Initialize transformer
        model = Transformer(
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            vocab_size=vocab_size,
            max_context_length=max_seq_len
        )
        
        # Create test input with proper alignment
        batch_size = 2
        seq_len = 4
        x = np.random.normal(0, 0.1, (batch_size, seq_len, d_model)).astype(np.float32)
        x = np.ascontiguousarray(x)
        
        # Initialize token embeddings
        model.token_embeddings = np.ascontiguousarray(
            np.random.normal(0, 0.1, (vocab_size, d_model)).astype(np.float32)
        )
        
        # Initialize position embeddings
        model.position_embeddings = np.ascontiguousarray(
            np.zeros((max_seq_len, d_model), dtype=np.float32)
        )
        position_embeddings_ptr = model.position_embeddings.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        
        try:
            position_embedding_kernel(
                position_embeddings_ptr,
                max_seq_len,
                d_model,
                10000.0
            )
        except Exception as e:
            logger.error(f"Error executing position embedding kernel: {str(e)}")
            raise
        
        # Run forward pass with smaller batch size and sequence length for testing
        try:
            # Use smaller dimensions for testing
            test_batch_size = 1
            test_seq_len = 2
            test_input = np.array([[[0.1] * d_model] * test_seq_len] * test_batch_size, dtype=np.float32)
            test_input = np.ascontiguousarray(test_input)
            
            # Warm-up run
            _ = model.forward(test_input)
            
            # Timing measurement
            num_runs = 10
            times = []
            for _ in range(num_runs):
                start_time = time.perf_counter()
                output = model.forward(test_input)
                end_time = time.perf_counter()
                times.append(end_time - start_time)
            
            avg_time = np.mean(times)
            std_time = np.std(times)
            
            # Calculate GFLOPS
            total_flops = calculate_transformer_flops(test_batch_size, test_seq_len, d_model, n_heads, n_layers)
            gflops = (total_flops / avg_time) / 1e9
            
            # Print only important metrics
            print("\n=== Transformer Performance ===")
            print(f"Model: {n_layers} layers, {n_heads} heads, d_model={d_model}")
            print(f"Input shape: ({test_batch_size}, {test_seq_len}, {d_model})")
            print(f"Time: {avg_time*1000:.2f} ms (±{std_time*1000:.2f} ms)")
            print(f"GFLOPS: {gflops:.2f}")
            print(f"Output shape: {output.shape}")
            print(f"Output stats: mean={np.mean(output):.3f}, std={np.std(output):.3f}")
            print("=============================\n")
            
            # Check for NaN values
            if np.isnan(output).any():
                logger.error("NaN values detected in output!")
                return False
                
            # Check for infinite values
            if np.isinf(output).any():
                logger.error("Infinite values detected in output!")
                return False
                
            # Check output shape
            expected_shape = (test_batch_size, test_seq_len, d_model)
            if output.shape != expected_shape:
                logger.error(f"Unexpected output shape: got {output.shape}, expected {expected_shape}")
                return False
                
            # Check output statistics
            mean = np.mean(output)
            std = np.std(output)
            if abs(mean) > 1.0:
                logger.warning(f"Output mean ({mean:.3f}) is outside expected range [-1, 1]")
            if abs(std - 1.0) > 0.1:
                logger.warning(f"Output std ({std:.3f}) is not close to 1.0")
            
            return True
            
        except Exception as e:
            logger.error(f"Error during forward pass: {str(e)}")
            return False
            
    except Exception as e:
        logger.error(f"Error during test setup: {str(e)}")
        return False

if __name__ == "__main__":
    # Run transformer test
    transformer_success = test_transformer()
    
    # Print summary
    print(f"\nTest Status: {'✓ PASSED' if transformer_success else '✗ FAILED'}")
    
    # Exit with appropriate status code
    exit(0 if transformer_success else 1) 