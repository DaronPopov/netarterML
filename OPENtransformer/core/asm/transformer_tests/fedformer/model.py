import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class FrequencyDecomposition(nn.Module):
    def __init__(self, d_model, n_freq_bands=4):
        super().__init__()
        self.n_freq_bands = n_freq_bands
        self.d_model = d_model
        
        # Frequency decomposition layers
        self.freq_decomp = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(d_model, d_model, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv1d(d_model, d_model, kernel_size=3, padding=1)
            ) for _ in range(n_freq_bands)
        ])
        
        # Frequency-specific attention
        self.freq_attention = nn.ModuleList([
            nn.MultiheadAttention(d_model, num_heads=8, dropout=0.1)
            for _ in range(n_freq_bands)
        ])
        
        # Frequency fusion
        self.freq_fusion = nn.Sequential(
            nn.Linear(d_model * n_freq_bands, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
    
    def forward(self, x):
        # Input shape: [batch_size, seq_len, d_model]
        batch_size, seq_len, _ = x.shape
        
        # Reshape for convolution
        x = x.transpose(1, 2)  # [batch_size, d_model, seq_len]
        
        # Frequency decomposition
        freq_components = []
        for i in range(self.n_freq_bands):
            # Apply frequency-specific convolution
            freq_comp = self.freq_decomp[i](x)
            
            # Reshape for attention
            freq_comp = freq_comp.transpose(1, 2)  # [batch_size, seq_len, d_model]
            
            # Apply frequency-specific attention
            freq_comp, _ = self.freq_attention[i](
                freq_comp.transpose(0, 1),
                freq_comp.transpose(0, 1),
                freq_comp.transpose(0, 1)
            )
            freq_comp = freq_comp.transpose(0, 1)  # [batch_size, seq_len, d_model]
            
            freq_components.append(freq_comp)
        
        # Concatenate frequency components
        freq_concat = torch.cat(freq_components, dim=-1)  # [batch_size, seq_len, d_model * n_freq_bands]
        
        # Fuse frequency components
        fused = self.freq_fusion(freq_concat)  # [batch_size, seq_len, d_model]
        
        return fused

class FEDformer(nn.Module):
    def __init__(self, 
                 enc_in, 
                 dec_in, 
                 c_out, 
                 seq_len, 
                 label_len, 
                 out_len,
                 d_model=512, 
                 n_heads=8, 
                 e_layers=3, 
                 d_layers=2, 
                 d_ff=512,
                 dropout=0.1,
                 activation='gelu',
                 device=torch.device('cuda:0')):
        super().__init__()
        self.pred_len = out_len
        
        # Ensure sequence length is sufficient for lags
        min_seq_len = max(seq_len, 1000)  # Minimum sequence length for proper lag computation
        self.seq_len = min_seq_len
        self.label_len = label_len
        
        # Data embedding
        self.enc_embedding = nn.Sequential(
            nn.Linear(enc_in, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        self.dec_embedding = nn.Sequential(
            nn.Linear(dec_in, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        
        # Position embedding
        self.pos_embedding = nn.Parameter(torch.randn(1, self.seq_len, d_model))
        
        # Frequency decomposition
        self.freq_decomp = FrequencyDecomposition(d_model)
        
        # Encoder layers
        self.encoder_layers = nn.ModuleList([
            nn.Sequential(
                nn.MultiheadAttention(d_model, n_heads, dropout=dropout),
                nn.LayerNorm(d_model),
                nn.Linear(d_model, d_ff),
                nn.GELU() if activation == 'gelu' else nn.ReLU(),
                nn.Linear(d_ff, d_model),
                nn.LayerNorm(d_model)
            ) for _ in range(e_layers)
        ])
        
        # Decoder layers
        self.decoder_layers = nn.ModuleList([
            nn.Sequential(
                nn.MultiheadAttention(d_model, n_heads, dropout=dropout),
                nn.LayerNorm(d_model),
                nn.Linear(d_model, d_ff),
                nn.GELU() if activation == 'gelu' else nn.ReLU(),
                nn.Linear(d_ff, d_model),
                nn.LayerNorm(d_model)
            ) for _ in range(d_layers)
        ])
        
        # Output projection
        self.projection = nn.Linear(d_model, c_out)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Move to device
        self.to(device)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights."""
        for name, param in self.named_parameters():
            if 'weight' in name:
                if 'norm' in name:
                    nn.init.normal_(param, mean=1.0, std=0.02)
                else:
                    nn.init.xavier_normal_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
            elif 'pos_embedding' in name:
                nn.init.trunc_normal_(param, mean=0.0, std=0.02)
    
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        """Forward pass of the model."""
        batch_size = x_enc.shape[0]
        
        # Ensure input sequence length matches model requirements
        if x_enc.shape[1] < self.seq_len:
            # Pad the input sequence if needed
            pad_length = self.seq_len - x_enc.shape[1]
            x_enc = F.pad(x_enc, (0, 0, pad_length, 0), mode='replicate')
            x_mark_enc = F.pad(x_mark_enc, (0, 0, pad_length, 0), mode='replicate')
        
        # Data embedding
        enc_embed = self.enc_embedding(x_enc)
        dec_embed = self.dec_embedding(x_dec)
        
        # Add position embedding
        enc_embed = enc_embed + self.pos_embedding[:, :enc_embed.shape[1], :]
        
        # Frequency decomposition
        enc_freq = self.freq_decomp(enc_embed)
        
        # Encoder layers
        enc_out = enc_freq
        for layer in self.encoder_layers:
            # Self-attention
            attn_out, _ = layer[0](enc_out, enc_out, enc_out, key_padding_mask=enc_self_mask)
            enc_out = layer[1](enc_out + attn_out)
            
            # Feed-forward
            ff_out = layer[2](enc_out)
            ff_out = layer[3](ff_out)
            ff_out = layer[4](ff_out)
            enc_out = layer[5](enc_out + ff_out)
        
        # Decoder layers
        dec_out = dec_embed
        for layer in self.decoder_layers:
            # Self-attention
            attn_out, _ = layer[0](dec_out, dec_out, dec_out, key_padding_mask=dec_self_mask)
            dec_out = layer[1](dec_out + attn_out)
            
            # Cross-attention with encoder
            cross_attn_out, _ = layer[0](dec_out, enc_out, enc_out, key_padding_mask=dec_enc_mask)
            dec_out = layer[1](dec_out + cross_attn_out)
            
            # Feed-forward
            ff_out = layer[2](dec_out)
            ff_out = layer[3](ff_out)
            ff_out = layer[4](ff_out)
            dec_out = layer[5](dec_out + ff_out)
        
        # Output projection
        output = self.projection(dec_out)
        
        return output 