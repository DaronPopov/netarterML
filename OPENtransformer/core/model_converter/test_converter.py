import os
import numpy as np
import torch
import logging
import ctypes
from pathlib import Path
from finlib.core.model_converter.converter import ModelConverter
from finlib.core.model_converter.loader import ConvertedModelLoader
from finlib.core.asm.kernels.fused_transformer_op import create_fully_fused_transformer_op
from finlib.core.asm.assembler.builder import build_and_jit

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MultiHeadAttention(torch.nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        
        # Combined QKV projection
        self.qkv = torch.nn.Linear(d_model, 3 * d_model, bias=False)
        # Output projection
        self.proj = torch.nn.Linear(d_model, d_model, bias=False)

class DummyTransformerLayer(torch.nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.norm1 = torch.nn.LayerNorm(d_model)
        self.norm2 = torch.nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(d_model, 4 * d_model),
            torch.nn.GELU(),
            torch.nn.Linear(4 * d_model, d_model)
        )

class DummyTransformer(torch.nn.Module):
    def __init__(self, d_model: int, n_heads: int, n_layers: int):
        super().__init__()
        self.transformer = torch.nn.ModuleList([
            DummyTransformerLayer(d_model, n_heads)
            for _ in range(n_layers)
        ])

def create_dummy_transformer_model(d_model: int = 512, n_heads: int = 8, n_layers: int = 6) -> torch.nn.Module:
    """
    Create a dummy transformer model for testing.
    
    Args:
        d_model: Model dimension
        n_heads: Number of attention heads
        n_layers: Number of transformer layers
        
    Returns:
        Dummy transformer model
    """
    return DummyTransformer(d_model, n_heads, n_layers)

def test_model_conversion():
    """Test the model conversion and loading process with ASM backend."""
    try:
        # Create test directory
        test_dir = Path("test_converted_model")
        test_dir.mkdir(exist_ok=True)
        
        # Create dummy model
        logger.info("Creating dummy transformer model...")
        model = create_dummy_transformer_model(d_model=512, n_heads=8, n_layers=6)
        
        # Save dummy model
        model_path = test_dir / "dummy_model.pt"
        torch.save(model, model_path)
        
        # Convert model
        logger.info("Converting model...")
        converter = ModelConverter()
        converter.convert_model(
            str(model_path),
            str(test_dir / "converted_model"),
            model_config={"backend": "asm", "test": True}
        )
        
        # Load converted model
        logger.info("Loading converted model...")
        loader = ConvertedModelLoader(str(test_dir / "converted_model"))
        
        # Load weights and metadata first
        loader.load_weights()
        loader.load_metadata()
        
        # Get model dimensions before initializing fused op
        d_model, n_heads, n_layers = loader.get_model_dimensions()
        logger.info(f"Model dimensions: d_model={d_model}, n_heads={n_heads}, n_layers={n_layers}")
        
        # Initialize the fused operation with proper builder function
        logger.info("Initializing fused transformer operation...")
        loader.initialize_fused_op(build_and_jit)
        
        if loader.fused_op is None:
            raise RuntimeError("Failed to initialize fused transformer operation")
        
        # Create dummy input with proper alignment for ASM
        batch_size = 4
        seq_len = 64
        x = np.ascontiguousarray(np.random.randn(batch_size, seq_len, d_model).astype(np.float32))
        
        # Test each layer with proper error handling
        logger.info("Testing each transformer layer...")
        for i in range(n_layers):
            try:
                # Get layer weights
                weights = loader.get_layer_weights(i)
                
                # Ensure weights are contiguous and properly aligned
                weights = [np.ascontiguousarray(w) for w in weights]
                
                # Create output tensor with proper alignment
                output = np.ascontiguousarray(np.zeros_like(x))
                
                logger.info(f"Processing layer {i}...")
                logger.info(f"Input shape: {x.shape}")
                logger.info(f"QKV weights shape: {weights[0].shape}")
                logger.info(f"Attention output weights shape: {weights[1].shape}")
                logger.info(f"FF1 weights shape: {weights[2].shape}")
                logger.info(f"FF2 weights shape: {weights[3].shape}")
                
                # Apply fused operation with proper error checking
                try:
                    result = loader.fused_op(
                        x.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                        output.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                        *[w.ctypes.data_as(ctypes.POINTER(ctypes.c_float)) for w in weights],
                        batch_size,
                        seq_len,
                        d_model,
                        n_heads
                    )
                    
                    if result is not None:  # Some implementations might return status
                        logger.info(f"Layer {i} operation returned: {result}")
                        
                except Exception as e:
                    logger.error(f"Error in fused operation for layer {i}: {e}")
                    raise
                
                # Validate output
                if not np.all(np.isfinite(output)):
                    logger.warning(f"Non-finite values detected in layer {i} output")
                    # Print statistics about the output
                    logger.info(f"Output stats - min: {np.min(output)}, max: {np.max(output)}, mean: {np.mean(output)}")
                else:
                    logger.info(f"Layer {i} processed successfully")
                    
                # Update input for next layer
                x = output.copy()
                
            except Exception as e:
                logger.error(f"Error processing layer {i}: {e}")
                raise
        
        logger.info("Model conversion and loading test completed successfully")
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        raise
    finally:
        # Cleanup
        if test_dir.exists():
            import shutil
            shutil.rmtree(test_dir)

if __name__ == "__main__":
    test_model_conversion() 