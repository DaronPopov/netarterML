#ifndef DIFFUSION_KERNELS_H
#define DIFFUSION_KERNELS_H

// NEON-optimized attention computation
void compute_attention_neon(
    float* query,      // [batch_size, num_heads, seq_len, head_dim]
    float* key,        // [batch_size, num_heads, seq_len, head_dim]
    float* value,      // [batch_size, num_heads, seq_len, head_dim]
    float* output,     // [batch_size, num_heads, seq_len, head_dim]
    int batch_size,
    int num_heads,
    int seq_len,
    int head_dim);

// NEON-optimized cross attention computation
void compute_cross_attention_neon(
    float* hidden_states,  // [batch_size, seq_len, hidden_size]
    float* encoder_hidden_states,  // [batch_size, encoder_seq_len, encoder_hidden_size]
    float* output,  // [batch_size, seq_len, hidden_size]
    int batch_size,
    int seq_len,
    int hidden_size,
    int encoder_seq_len);

// NEON-optimized feed forward network computation
void compute_ffn_neon(
    float* input,      // [batch_size, seq_len, hidden_size]
    float* weights_1,  // [hidden_size, intermediate_size]
    float* weights_2,  // [intermediate_size, hidden_size]
    float* output,     // [batch_size, seq_len, hidden_size]
    int batch_size,
    int seq_len,
    int hidden_size,
    int intermediate_size);

// NEON-optimized convolution computation
void compute_conv_neon(
    float* input,      // [batch_size, channels, height, width]
    float* weights,    // [out_channels, in_channels, kernel_size, kernel_size]
    float* output,     // [batch_size, out_channels, out_height, out_width]
    int batch_size,
    int in_channels,
    int out_channels,
    int height,
    int width,
    int kernel_size,
    int stride,
    int padding);

#endif // DIFFUSION_KERNELS_H 