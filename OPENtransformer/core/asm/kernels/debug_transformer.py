import os
import sys
import logging
import numpy as np
import ctypes

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from OPENtransformer.core.asm.kernels.transformer import Transformer

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def test_transformer():
    """Run a simple test of the Transformer model."""
    try:
        # Create a small transformer model
        model = Transformer(
            d_model=512,
            n_heads=8,
            n_layers=4,
            vocab_size=32000
        )
        
        # Initialize weights
        model.initialize_weights()
        
        # Create test input
        batch_size = 1
        seq_len = 2
        input_shape = (batch_size, seq_len, model.d_model)
        logger.debug(f"Creating test input with shape {input_shape}")
        
        # Create input with small random values 
        x = np.random.randn(*input_shape).astype(np.float32) * 0.1
        
        # Log input statistics
        logger.debug(f"Input statistics - mean: {np.mean(x):.6f}, std: {np.std(x):.6f}")
        
        # Track intermediate states for debugging
        def debug_forward_pass():
            """Run a debug version of the forward pass with diagnostics at each step."""
            # Get input shape
            batch_size, seq_len, _ = x.shape
            
            # Create output tensor
            output = np.zeros_like(x)
            
            # Create temporary buffers for intermediate computations
            temp_buffer1 = np.zeros_like(x)
            temp_buffer2 = np.zeros_like(x)
            
            # Get initial input for the first layer
            current_input = x.copy()
            
            # Clip input values to a reasonable range
            np.clip(current_input, -100.0, 100.0, out=current_input)
            logger.debug(f"Initial input - mean: {np.mean(current_input):.6f}, std: {np.std(current_input):.6f}")
            
            # Process through transformer layers
            for layer_idx in range(model.n_layers):
                logger.debug(f"Processing layer {layer_idx+1}/{model.n_layers}")
                
                # Get layer weights
                layer_weights = model.weights[layer_idx]
                
                # Layer normalization before attention
                norm1_input = current_input.copy()
                norm1_output = np.zeros_like(norm1_input)
                
                # Apply layer norm across the feature dimension for each position
                for batch_idx in range(batch_size):
                    for seq_idx in range(seq_len):
                        # Get the feature vector for this position
                        feature_vec = norm1_input[batch_idx, seq_idx]
                        output_vec = norm1_output[batch_idx, seq_idx]
                        
                        # Get pointers
                        input_ptr = feature_vec.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
                        output_ptr = output_vec.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
                        gamma_ptr = layer_weights['norm1'].ctypes.data_as(ctypes.POINTER(ctypes.c_float))
                        beta_ptr = layer_weights['norm1_bias'].ctypes.data_as(ctypes.POINTER(ctypes.c_float))
                        
                        # Apply layer normalization
                        model.layer_norm_kernel(
                            input_ptr,
                            output_ptr,
                            gamma_ptr,
                            beta_ptr,
                            ctypes.c_int(model.d_model),
                            ctypes.c_int(1)
                        )
                
                # Check for NaNs after norm1
                if np.any(np.isnan(norm1_output)):
                    logger.error(f"Layer {layer_idx+1} - NaNs detected after norm1")
                else:
                    logger.debug(f"Layer {layer_idx+1} - After norm1 - mean: {np.mean(norm1_output):.6f}, std: {np.std(norm1_output):.6f}")
                
                # Apply attention manually for debugging
                attn_output = np.zeros_like(norm1_output)
                
                # Reshape for matrix multiplication
                x_reshaped = norm1_output.reshape(batch_size * seq_len, model.d_model)
                attn_output_reshaped = attn_output.reshape(batch_size * seq_len, model.d_model)
                
                # Get pointers
                input_ptr = x_reshaped.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
                output_ptr = attn_output_reshaped.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
                q_weights_ptr = layer_weights['q_proj'].ctypes.data_as(ctypes.POINTER(ctypes.c_float))
                
                # Apply Q projection
                model.attention_kernel(
                    input_ptr,
                    q_weights_ptr,
                    output_ptr,
                    batch_size * seq_len,
                    model.d_model,
                    model.d_model
                )
                
                # Scale by sqrt(d_k)
                attn_output_reshaped *= 1.0 / np.sqrt(model.d_model / model.n_heads)
                
                # Clip values
                np.clip(attn_output_reshaped, -100.0, 100.0, out=attn_output_reshaped)
                
                # Check for NaNs after attention
                if np.any(np.isnan(attn_output)):
                    logger.error(f"Layer {layer_idx+1} - NaNs detected after attention projection")
                else:
                    logger.debug(f"Layer {layer_idx+1} - After attention - mean: {np.mean(attn_output):.6f}, std: {np.std(attn_output):.6f}")
                
                # Residual connection
                post_attn = norm1_input + attn_output
                
                # Clip values
                np.clip(post_attn, -100.0, 100.0, out=post_attn)
                
                # Check for NaNs after residual
                if np.any(np.isnan(post_attn)):
                    logger.error(f"Layer {layer_idx+1} - NaNs detected after attention residual")
                else:
                    logger.debug(f"Layer {layer_idx+1} - After attn residual - mean: {np.mean(post_attn):.6f}, std: {np.std(post_attn):.6f}")
                
                # Layer normalization after attention
                norm2_input = post_attn.copy()
                norm2_output = np.zeros_like(norm2_input)
                
                # Apply layer norm across the feature dimension for each position
                for batch_idx in range(batch_size):
                    for seq_idx in range(seq_len):
                        # Get the feature vector for this position
                        feature_vec = norm2_input[batch_idx, seq_idx]
                        output_vec = norm2_output[batch_idx, seq_idx]
                        
                        # Get pointers
                        input_ptr = feature_vec.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
                        output_ptr = output_vec.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
                        gamma_ptr = layer_weights['norm2'].ctypes.data_as(ctypes.POINTER(ctypes.c_float))
                        beta_ptr = layer_weights['norm2_bias'].ctypes.data_as(ctypes.POINTER(ctypes.c_float))
                        
                        # Apply layer normalization
                        model.layer_norm_kernel(
                            input_ptr,
                            output_ptr,
                            gamma_ptr,
                            beta_ptr,
                            ctypes.c_int(model.d_model),
                            ctypes.c_int(1)
                        )
                
                # Check for NaNs after norm2
                if np.any(np.isnan(norm2_output)):
                    logger.error(f"Layer {layer_idx+1} - NaNs detected after norm2")
                else:
                    logger.debug(f"Layer {layer_idx+1} - After norm2 - mean: {np.mean(norm2_output):.6f}, std: {np.std(norm2_output):.6f}")
                
                # Create zero tensors for FFN
                intermediate = np.zeros((batch_size * seq_len, 4 * model.d_model), dtype=np.float32)
                ff_output = np.zeros_like(norm2_output)
                ff_output_reshaped = ff_output.reshape(batch_size * seq_len, model.d_model)
                
                # Reshape for matrix multiplication
                x_reshaped = norm2_output.reshape(batch_size * seq_len, model.d_model)
                
                # Get pointers
                input_ptr = x_reshaped.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
                intermediate_ptr = intermediate.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
                output_ptr = ff_output_reshaped.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
                ff1_weights_ptr = layer_weights['ff1'].ctypes.data_as(ctypes.POINTER(ctypes.c_float))
                ff2_weights_ptr = layer_weights['ff2'].ctypes.data_as(ctypes.POINTER(ctypes.c_float))
                
                # Apply first linear layer
                model.attention_kernel(
                    input_ptr,
                    ff1_weights_ptr,
                    intermediate_ptr,
                    batch_size * seq_len,
                    model.d_model,
                    4 * model.d_model
                )
                
                # Clip values
                np.clip(intermediate, -100.0, 100.0, out=intermediate)
                
                # Check for NaNs after FF1
                if np.any(np.isnan(intermediate)):
                    logger.error(f"Layer {layer_idx+1} - NaNs detected after FF1")
                else:
                    logger.debug(f"Layer {layer_idx+1} - After FF1 - mean: {np.mean(intermediate):.6f}, std: {np.std(intermediate):.6f}")
                
                # Apply GELU - replace with simple activation for testing
                intermediate = np.tanh(intermediate)
                
                # Clip values
                np.clip(intermediate, -100.0, 100.0, out=intermediate)
                
                # Apply second linear layer
                model.attention_kernel(
                    intermediate_ptr,
                    ff2_weights_ptr,
                    output_ptr,
                    batch_size * seq_len,
                    4 * model.d_model,
                    model.d_model
                )
                
                # Clip values
                np.clip(ff_output_reshaped, -100.0, 100.0, out=ff_output_reshaped)
                
                # Check for NaNs after FF2
                if np.any(np.isnan(ff_output)):
                    logger.error(f"Layer {layer_idx+1} - NaNs detected after FF2")
                else:
                    logger.debug(f"Layer {layer_idx+1} - After FF2 - mean: {np.mean(ff_output):.6f}, std: {np.std(ff_output):.6f}")
                
                # Residual connection
                current_input = norm2_input + ff_output
                
                # Clip values
                np.clip(current_input, -100.0, 100.0, out=current_input)
                
                # Check for NaNs after residual
                if np.any(np.isnan(current_input)):
                    logger.error(f"Layer {layer_idx+1} - NaNs detected after FF residual")
                else:
                    logger.debug(f"Layer {layer_idx+1} - After FF residual - mean: {np.mean(current_input):.6f}, std: {np.std(current_input):.6f}")
            
            # Copy final output
            output[:] = current_input
            
            # Final clip
            np.clip(output, -100.0, 100.0, out=output)
            
            return output
        
        # Run the debug forward pass
        debug_output = debug_forward_pass()
        
        # Check if the debug output contains NaN values
        if np.any(np.isnan(debug_output)):
            logger.error("NaN values detected in debug output!")
        else:
            logger.debug(f"Debug output statistics - mean: {np.mean(debug_output):.6f}, std: {np.std(debug_output):.6f}")
        
        # Now also run the actual forward pass for comparison
        logger.debug("Running official forward pass...")
        output = model.forward(x)
        
        # Check output shape
        logger.debug(f"Forward pass successful! Output shape: {output.shape}")
        
        # Log output statistics
        logger.debug("Output statistics:")
        logger.debug(f"  Mean: {np.mean(output)}")
        logger.debug(f"  Std: {np.std(output)}")
        logger.debug(f"  Min: {np.min(output)}")
        logger.debug(f"  Max: {np.max(output)}")
        
        # Check for NaN values in output
        if np.any(np.isnan(output)) or np.any(np.isinf(output)):
            logger.error("NaN values detected in output!")
            test_success = False
        else:
            test_success = True
        
        # Log test summary
        logger.info("\nTest Summary:")
        logger.info(f"Transformer test: {'✓' if test_success else '✗'}")
        
        return test_success
        
    except Exception as e:
        logger.error(f"Error in transformer test: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

if __name__ == "__main__":
    test_transformer() 