import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Optional, Tuple

class GatedLinearUnit(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, dropout: float = 0.1):
        super().__init__()
        self.fc = nn.Linear(input_size, hidden_size * 2)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc(x)
        x = self.dropout(x)
        return x[:, :x.size(1)//2] * torch.sigmoid(x[:, x.size(1)//2:])

class VariableSelectionNetwork(nn.Module):
    def __init__(self, input_sizes: List[int], hidden_size: int, dropout: float = 0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.input_sizes = input_sizes
        
        # GRN for each input
        self.grn_vars = nn.ModuleList([
            GatedLinearUnit(size, hidden_size, dropout)
            for size in input_sizes
        ])
        
        # GRN for combined inputs
        self.grn_combined = GatedLinearUnit(hidden_size * len(input_sizes), hidden_size, dropout)
        
        # Attention weights
        self.attention = nn.Linear(hidden_size, 1)
        
    def forward(self, x: List[torch.Tensor]) -> torch.Tensor:
        # Process each input through its GRN
        processed = []
        for i, var in enumerate(x):
            processed.append(self.grn_vars[i](var))
        
        # Combine all processed inputs
        combined = torch.cat(processed, dim=-1)
        combined = self.grn_combined(combined)
        
        # Calculate attention weights
        attention_weights = F.softmax(self.attention(combined), dim=-1)
        
        # Apply attention
        return torch.sum(attention_weights * combined, dim=-1)

class TemporalSelfAttention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_size = hidden_size // num_heads
        
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, _ = x.size()
        
        # Project queries, keys, and values
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_size).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_size).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_size).transpose(1, 2)
        
        # Calculate attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.head_size)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention to values
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
        
        return self.out_proj(out)

class TemporalFusionTransformer(nn.Module):
    def __init__(self,
                 num_static_features: int,
                 num_time_varying_features: int,
                 num_targets: int,
                 hidden_size: int = 64,
                 num_heads: int = 4,
                 num_encoder_layers: int = 3,
                 num_decoder_layers: int = 2,
                 dropout: float = 0.1,
                 max_seq_len: int = 96):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len
        
        # Static feature processing
        self.static_encoder = nn.Sequential(
            nn.Linear(num_static_features, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)
        )
        
        # Time-varying feature processing
        self.time_varying_encoder = VariableSelectionNetwork(
            [num_time_varying_features],
            hidden_size,
            dropout
        )
        
        # Position encoding
        self.pos_encoder = nn.Parameter(torch.randn(1, max_seq_len, hidden_size))
        
        # Encoder layers
        self.encoder_layers = nn.ModuleList([
            nn.Sequential(
                TemporalSelfAttention(hidden_size, num_heads, dropout),
                nn.LayerNorm(hidden_size),
                nn.Linear(hidden_size, hidden_size * 4),
                nn.ReLU(),
                nn.Linear(hidden_size * 4, hidden_size),
                nn.LayerNorm(hidden_size)
            )
            for _ in range(num_encoder_layers)
        ])
        
        # Decoder layers
        self.decoder_layers = nn.ModuleList([
            nn.Sequential(
                TemporalSelfAttention(hidden_size, num_heads, dropout),
                nn.LayerNorm(hidden_size),
                nn.Linear(hidden_size, hidden_size * 4),
                nn.ReLU(),
                nn.Linear(hidden_size * 4, hidden_size),
                nn.LayerNorm(hidden_size)
            )
            for _ in range(num_decoder_layers)
        ])
        
        # Output projection
        self.output_proj = nn.Linear(hidden_size, num_targets)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
    def forward(self,
                static_features: torch.Tensor,
                time_varying_features: torch.Tensor,
                target_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size = static_features.size(0)
        
        # Process static features
        static_encoded = self.static_encoder(static_features)
        
        # Process time-varying features
        time_varying_encoded = self.time_varying_encoder([time_varying_features])
        
        # Add position encoding
        time_varying_encoded = time_varying_encoded + self.pos_encoder[:, :time_varying_encoded.size(1)]
        
        # Apply encoder layers
        x = time_varying_encoded
        for layer in self.encoder_layers:
            x = layer(x)
        
        # Apply decoder layers
        for layer in self.decoder_layers:
            x = layer(x)
        
        # Project to output space
        output = self.output_proj(x)
        
        if target_mask is not None:
            output = output.masked_fill(target_mask == 0, 0)
        
        return output

class EnhancedTFT(nn.Module):
    def __init__(self,
                 num_static_features: int,
                 num_time_varying_features: int,
                 num_targets: int,
                 hidden_size: int = 64,
                 num_heads: int = 4,
                 num_encoder_layers: int = 3,
                 num_decoder_layers: int = 2,
                 dropout: float = 0.1,
                 max_seq_len: int = 96):
        super().__init__()
        self.tft = TemporalFusionTransformer(
            num_static_features,
            num_time_varying_features,
            num_targets,
            hidden_size,
            num_heads,
            num_encoder_layers,
            num_decoder_layers,
            dropout,
            max_seq_len
        )
        
        # Additional feature extractors
        self.technical_indicators = nn.Sequential(
            nn.Linear(num_time_varying_features, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)
        )
        
        self.market_sentiment = nn.Sequential(
            nn.Linear(num_static_features, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)
        )
        
        # Feature fusion
        self.feature_fusion = nn.Sequential(
            nn.Linear(hidden_size * 3, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)
        )
        
        # Output layers
        self.price_prediction = nn.Linear(hidden_size, 1)
        self.volatility_prediction = nn.Linear(hidden_size, 1)
        self.trend_prediction = nn.Linear(hidden_size, 3)  # Up, Down, Sideways
        
    def forward(self,
                static_features: torch.Tensor,
                time_varying_features: torch.Tensor,
                target_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        # Get base TFT predictions
        tft_output = self.tft(static_features, time_varying_features, target_mask)
        
        # Extract technical indicators
        tech_features = self.technical_indicators(time_varying_features)
        
        # Extract market sentiment
        sentiment_features = self.market_sentiment(static_features)
        
        # Fuse all features
        fused_features = torch.cat([tft_output, tech_features, sentiment_features], dim=-1)
        fused_features = self.feature_fusion(fused_features)
        
        # Generate multiple predictions
        predictions = {
            'price': self.price_prediction(fused_features),
            'volatility': self.volatility_prediction(fused_features),
            'trend': F.softmax(self.trend_prediction(fused_features), dim=-1)
        }
        
        return predictions 