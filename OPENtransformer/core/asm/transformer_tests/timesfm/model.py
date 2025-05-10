import torch
import torch.nn as nn
import logging
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoConfig
from .architecture import TimesFMConfig, TimesFMModel

logger = logging.getLogger(__name__)

class TimesFM:
    def __init__(self, 
                 seq_len: int = 1000,
                 pred_len: int = 24,
                 device: str = None):
        """Initialize TimesFM model for time series forecasting.
        
        Args:
            seq_len: Length of input sequence
            pred_len: Length of prediction sequence
            device: Device to use (cuda/cpu)
        """
        self.seq_len = seq_len
        self.pred_len = pred_len
        
        # Set device - force CPU for now to avoid MPS issues
        self.device = torch.device('cpu')
        
        # Load model
        self._load_model()
        
        logger.info(f"TimesFM initialized on device: {self.device}")
    
    def _load_model(self):
        """Load the TimesFM model."""
        try:
            # Get model path
            model_dir = Path(__file__).parent / 'weights' / 'timesfm'
            
            if not model_dir.exists():
                raise ValueError(
                    "Model weights not found. Please run download_weights.py first."
                )
            
            # Load configuration
            self.config = TimesFMConfig.from_pretrained(model_dir)
            
            # Load model - force CPU for stability
            self.model = TimesFMModel.from_pretrained(
                model_dir,
                config=self.config,
                trust_remote_code=True,
                device_map="cpu"  # Force CPU usage
            )
            
            # Move model to CPU explicitly
            self.model.to('cpu')
            
            # Set to evaluation mode
            self.model.eval()
            
            logger.info("Successfully loaded TimesFM model on CPU")
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
    
    def _preprocess_input(self, x: torch.Tensor) -> torch.Tensor:
        """Preprocess input data.
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, n_features]
        
        Returns:
            Preprocessed tensor
        """
        # Ensure correct shape
        if len(x.shape) == 2:
            x = x.unsqueeze(-1)
        
        # Move to device
        x = x.to(self.device)
        
        # Just normalize the input for now to avoid token conversion issues
        # Scale to [0, 1] range for each feature
        x_min = x.min(dim=1, keepdim=True)[0]
        x_max = x.max(dim=1, keepdim=True)[0]
        x_scaled = (x - x_min) / (x_max - x_min + 1e-6)
        
        return x_scaled
    
    def forecast(self, x: torch.Tensor) -> torch.Tensor:
        """Generate forecasts.
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, n_features]
        
        Returns:
            Forecasted values of shape [batch_size, pred_len, n_features]
        """
        try:
            # Preprocess input
            x_scaled = self._preprocess_input(x)
            
            # Generate predictions
            with torch.no_grad():
                # Generate predictions using the model's generate method
                output = self.model.generate(
                    x_scaled,
                    max_new_tokens=self.pred_len
                )
                
                # Extract only the prediction part (last self.pred_len timesteps)
                predictions = output[:, -self.pred_len:, :]
                
                # Rescale back to original range
                x_min = x.min(dim=1, keepdim=True)[0].to(self.device)
                x_max = x.max(dim=1, keepdim=True)[0].to(self.device)
                predictions = predictions * (x_max - x_min) + x_min
            
            logger.info(f"Generated predictions with shape: {predictions.shape}")
            
            return predictions
            
        except Exception as e:
            logger.error(f"Error generating forecast: {str(e)}")
            raise
    
    def forecast_sequence(self, 
                         x: torch.Tensor, 
                         n_steps: int = 1,
                         stride: int = None) -> torch.Tensor:
        """Generate sequential forecasts with sliding window.
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, n_features]
            n_steps: Number of forecast steps
            stride: Stride for sliding window (default: pred_len)
        
        Returns:
            Sequence of forecasts
        """
        if stride is None:
            stride = self.pred_len
        
        try:
            forecasts = []
            current_input = x.clone()
            
            for _ in range(n_steps):
                # Generate forecast
                pred = self.forecast(current_input)
                forecasts.append(pred)
                
                # Update input sequence
                if len(current_input.shape) == 2:
                    current_input = torch.cat([
                        current_input[:, stride:],
                        pred[:, :stride]
                    ], dim=1)
                else:
                    current_input = torch.cat([
                        current_input[:, stride:, :],
                        pred[:, :stride, :]
                    ], dim=1)
            
            # Concatenate all forecasts
            forecasts = torch.cat(forecasts, dim=1)
            
            logger.info(f"Generated sequence forecast with shape: {forecasts.shape}")
            
            return forecasts
            
        except Exception as e:
            logger.error(f"Error generating sequence forecast: {str(e)}")
            raise 