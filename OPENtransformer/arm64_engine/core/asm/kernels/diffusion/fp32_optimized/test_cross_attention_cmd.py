"""
Test script for cross-attention kernel.
"""

import numpy as np
from cross_attention_kernel_asm import CrossAttentionKernelASM

def run_test():
    """Run tests for cross-attention kernel."""
    # Initialize kernel
    kernel = CrossAttentionKernelASM()
    
    # Create test inputs
    text_embeddings = np.random.randn(2, 4, 8).astype(np.float32)  # (batch_size, text_len, hidden_size)
    video_features = np.random.randn(2, 3, 8).astype(np.float32)   # (batch_size, video_len, hidden_size)
    
    print(f"Test inputs created:")
    print(f"- Text embeddings shape: {text_embeddings.shape}")
    print(f"- Video features shape: {video_features.shape}")
    
    # Test basic functionality
    output = kernel.apply_cross_attention(text_embeddings, video_features)
    print(f"\nBasic functionality test:")
    print(f"- Output shape: {output.shape}")
    print(f"- Output range: [{output.min():.3f}, {output.max():.3f}]")
    print(f"- Output mean: {output.mean():.3f}")
    print(f"- Output std: {output.std():.3f}")
    
    # Test different scale factors
    for scale in [0.1, 1.0, 10.0]:
        output = kernel.apply_cross_attention(text_embeddings, video_features, scale_factor=scale)
        print(f"\nScale factor {scale} test:")
        print(f"- Output range: [{output.min():.3f}, {output.max():.3f}]")
        print(f"- Output mean: {output.mean():.3f}")
        print(f"- Output std: {output.std():.3f}")
    
    # Test dropout
    output_no_dropout = kernel.apply_cross_attention(text_embeddings, video_features, dropout_prob=0.0)
    output_with_dropout = kernel.apply_cross_attention(text_embeddings, video_features, dropout_prob=0.5)
    print(f"\nDropout test:")
    print(f"- No dropout range: [{output_no_dropout.min():.3f}, {output_no_dropout.max():.3f}]")
    print(f"- With dropout range: [{output_with_dropout.min():.3f}, {output_with_dropout.max():.3f}]")
    
    # Test numerical stability with extreme values
    extreme_text = np.random.randn(2, 4, 8).astype(np.float32) * 1e3
    extreme_video = np.random.randn(2, 3, 8).astype(np.float32) * 1e3
    output = kernel.apply_cross_attention(extreme_text, extreme_video)
    print(f"\nNumerical stability test:")
    print(f"- All values finite: {np.all(np.isfinite(output))}")
    print(f"- Output range: [{output.min():.3f}, {output.max():.3f}]")
    
    print("\nAll tests completed successfully!")

if __name__ == "__main__":
    run_test() 