import torch
from transformers import TimeSeriesTransformerForPrediction, TimeSeriesTransformerConfig
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def download_model():
    """Download and cache the Time Series Transformer model."""
    try:
        logger.info("Initializing model configuration...")
        config = TimeSeriesTransformerConfig(
            prediction_length=12,
            context_length=200,  # Increased from 96 to accommodate lag features
            input_size=10,  # For 10 stocks
            scaling=True,
            num_parallel_samples=100,
            d_model=64,
            num_attention_heads=4,
            num_trainable_samples=100,
            dropout=0.1,
            encoder_layers=3,
            decoder_layers=3,
            activation_function="gelu",
            use_cache=True,
            lags_sequence=[1, 2, 3, 4, 5]  # Simplified lag sequence
        )
        
        logger.info("Creating model...")
        model = TimeSeriesTransformerForPrediction(config)
        
        # Move model to appropriate device
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        model.eval()  # Set to evaluation mode
        
        logger.info(f"Model initialized on device: {device}")
        logger.info("Model parameters loaded and cached successfully")
        
        # Test the model with dummy data
        logger.info("Testing model with dummy data...")
        batch_size = 1
        seq_len = 200  # Match context_length
        n_features = 10
        
        # Create dummy input data
        past_values = torch.randn(batch_size, seq_len, n_features).to(device)
        past_time_features = torch.zeros(batch_size, seq_len, 5).to(device)  # Time features
        future_time_features = torch.zeros(batch_size, 12, 5).to(device)  # Future time features
        past_observed_mask = torch.ones(batch_size, seq_len, n_features).bool().to(device)  # All values observed
        
        with torch.no_grad():
            outputs = model(
                past_values=past_values,
                past_time_features=past_time_features,
                future_time_features=future_time_features,
                past_observed_mask=past_observed_mask,
                return_dict=True
            )
        
        logger.info(f"Test successful! Output shape: {outputs.sequences.shape if hasattr(outputs, 'sequences') else outputs.predictions.shape}")
        logger.info("Model is ready for use")
        return True
        
    except Exception as e:
        logger.error(f"Error downloading/initializing model: {e}")
        logger.error("Full traceback:", exc_info=True)
        return False

if __name__ == "__main__":
    success = download_model()
    if success:
        logger.info("Model setup completed successfully")
    else:
        logger.error("Model setup failed") 