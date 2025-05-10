import numpy as np
import time
import logging
from finlib.core.asm.kernels.transformer import Transformer

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_transformer_single_inference():
    """
    Test a single forward inference pass with large model parameters.
    This test uses parameters similar to a large transformer model.
    """
    # Model parameters (similar to a large transformer)
    d_model = 12288      # Hidden size (12K)
    n_heads = 96        # Number of attention heads
    n_layers = 96       # Number of transformer layers
    d_ff = 49152       # Feed-forward dimension (4 * d_model)
    max_context_length = 2048  # Maximum sequence length
    
    # Test parameters
    batch_size = 4      # Reasonable batch size
    seq_len = 2048      # Full context length
    
    logger.info("Initializing transformer with large parameters:")
    logger.info(f"d_model={d_model}, n_heads={n_heads}, n_layers={n_layers}")
    logger.info(f"d_ff={d_ff}, max_context_length={max_context_length}")
    logger.info(f"Testing with batch_size={batch_size}, seq_len={seq_len}")
    
    # Create transformer
    transformer = Transformer(
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        d_ff=d_ff,
        max_context_length=max_context_length
    )
    
    # Create random input
    x = np.random.normal(0, 1, (batch_size, seq_len, d_model)).astype(np.float32)
    
    # Run single inference
    logger.info("\nRunning single forward inference...")
    start_time = time.time()
    output = transformer.forward(x)
    end_time = time.time()
    inference_time = (end_time - start_time) * 1000  # Convert to ms
    
    # Calculate FLOPs
    flops = (
        # Self-attention
        2 * batch_size * seq_len * seq_len * d_model +  # QK^T
        2 * batch_size * seq_len * seq_len * d_model +  # (QK^T)V
        # Feed-forward
        2 * batch_size * seq_len * d_model * d_ff +  # FF1
        2 * batch_size * seq_len * d_ff * d_model    # FF2
    ) * n_layers
    
    gflops = (flops / (inference_time / 1000)) / 1e9
    
    # Print results
    logger.info("\nInference Results:")
    logger.info(f"Input shape: {x.shape}")
    logger.info(f"Output shape: {output.shape}")
    logger.info(f"Inference time: {inference_time:.2f} ms")
    logger.info(f"Throughput: {gflops:.2f} GFLOPS")
    logger.info(f"Tokens per second: {batch_size * seq_len / (inference_time / 1000):.2f}")
    
    # Verify output
    if np.any(np.isnan(output)):
        logger.warning("Warning: Output contains NaN values!")
    if np.any(np.isinf(output)):
        logger.warning("Warning: Output contains infinite values!")
    
    return output

if __name__ == "__main__":
    test_transformer_single_inference() 