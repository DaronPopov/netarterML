import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

class TransformerModel(nn.Module):
    def __init__(self, input_dim: int = 1, d_model: int = 512, nhead: int = 8, num_layers: int = 6):
        super().__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        # Initial embedding to convert input to d_model dimensions
        self.embedding = nn.Sequential(
            nn.Linear(input_dim, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, d_model)
        )
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.decoder = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, input_dim)
        )
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Ensure input is 3D: [batch_size, seq_len, input_dim]
        if x.dim() == 2:
            x = x.unsqueeze(-1)
        elif x.dim() == 1:
            x = x.unsqueeze(0).unsqueeze(-1)
            
        # Reshape if needed
        batch_size, seq_len = x.shape[0], x.shape[1]
        if x.shape[-1] != self.input_dim:
            x = x.view(batch_size, seq_len, self.input_dim)
            
        x = self.embedding(x)  # [batch_size, seq_len, d_model]
        x = self.transformer_encoder(x, mask)  # [batch_size, seq_len, d_model]
        x = self.decoder(x)  # [batch_size, seq_len, input_dim]
        return x

class TransformerInference:
    def __init__(self, model_path: Optional[str] = None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = TransformerModel().to(self.device)
        
        if model_path:
            try:
                # Load model weights with weights_only=False for compatibility
                state_dict = torch.load(model_path, map_location=self.device, weights_only=False)
                if isinstance(state_dict, dict) and 'state_dict' in state_dict:
                    state_dict = state_dict['state_dict']
                self.model.load_state_dict(state_dict, strict=False)
                logger.info(f"Loaded model from {model_path}")
            except Exception as e:
                logger.error(f"Error loading model: {e}")
        
        self.model.eval()
        
    def preprocess_data(self, prices: List[float], dates: List[str]) -> Tuple[torch.Tensor, List[datetime]]:
        """Preprocess price data for transformer input."""
        # Convert dates to datetime objects
        dates = [datetime.fromisoformat(d.replace('Z', '+00:00')) for d in dates]
        
        # Create price sequence
        prices = np.array(prices)
        
        # Normalize prices
        mean_price = np.mean(prices)
        std_price = np.std(prices)
        normalized_prices = (prices - mean_price) / std_price
        
        # Create input tensor
        x = torch.FloatTensor(normalized_prices).unsqueeze(-1).to(self.device)
        
        return x, dates, mean_price, std_price
    
    def generate_future_dates(self, last_date: datetime, time_horizon: int) -> List[datetime]:
        """Generate future dates for prediction."""
        return [last_date + timedelta(minutes=i) for i in range(1, time_horizon + 1)]
    
    def predict(
        self,
        symbol: str,
        prices: List[float],
        dates: List[str],
        current_price: float,
        time_horizon: int = 60,
        confidence_threshold: float = 0.7
    ) -> Dict:
        """
        Generate price predictions using the transformer model.
        
        Args:
            symbol: Stock symbol
            prices: Historical price data
            dates: Historical dates
            current_price: Current price
            time_horizon: Number of minutes to predict
            confidence_threshold: Minimum confidence for predictions
            
        Returns:
            Dictionary containing predictions and confidence metrics
        """
        try:
            # Preprocess data
            x, dates, mean_price, std_price = self.preprocess_data(prices, dates)
            
            # Generate predictions
            with torch.no_grad():
                predictions = self.model(x)
                predictions = predictions.squeeze(-1).cpu().numpy()
                
                # Denormalize predictions
                predictions = predictions * std_price + mean_price
                
                # Calculate confidence scores
                confidence_scores = self.calculate_confidence(predictions, current_price)
                
                # Filter predictions by confidence
                valid_predictions = predictions[confidence_scores >= confidence_threshold]
                valid_dates = self.generate_future_dates(dates[-1], len(valid_predictions))
                
                # Calculate trend and momentum
                trend_direction = "up" if valid_predictions[-1] > current_price else "down"
                price_change = ((valid_predictions[-1] - current_price) / current_price) * 100
                momentum = self.calculate_momentum(predictions)
                volatility = self.calculate_volatility(predictions)
                
                # Prepare prediction data for visualization
                prediction_data = [
                    {
                        "time": d.isoformat(),
                        "price": float(p)
                    }
                    for d, p in zip(valid_dates, valid_predictions)
                ]
                
                return {
                    "trend_direction": trend_direction,
                    "confidence": float(np.mean(confidence_scores)),
                    "predicted_price": float(valid_predictions[-1]),
                    "price_change": float(price_change),
                    "volatility": float(volatility),
                    "momentum": float(momentum),
                    "time_horizon": time_horizon,
                    "prediction_data": prediction_data
                }
                
        except Exception as e:
            logger.error(f"Error generating predictions for {symbol}: {e}")
            return None
    
    def calculate_confidence(self, predictions: np.ndarray, current_price: float) -> np.ndarray:
        """Calculate confidence scores for predictions."""
        # Simple confidence calculation based on price stability
        price_changes = np.abs(np.diff(predictions))
        confidence = 1 - (price_changes / current_price)
        return np.clip(confidence, 0, 1)
    
    def calculate_momentum(self, prices: np.ndarray) -> float:
        """Calculate price momentum."""
        if len(prices) < 2:
            return 0.0
        return (prices[-1] - prices[0]) / prices[0]
    
    def calculate_volatility(self, prices: np.ndarray) -> float:
        """Calculate price volatility."""
        if len(prices) < 2:
            return 0.0
        returns = np.diff(prices) / prices[:-1]
        return float(np.std(returns) * np.sqrt(252))  # Annualized volatility 