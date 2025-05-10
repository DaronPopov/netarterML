import yfinance as yf
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple
import logging
from datetime import datetime, timedelta
import torch
import asyncio
import websockets
import json
from transformers import PreTrainedModel, GenerationMixin
from OPENtransformer.core.asm.kernels.transformer import Transformer
from OPENtransformer.core.asm.kernels.tokenizer import BasicTokenizer
import torch.nn as nn

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TimesFMConfig:
    def __init__(
        self,
        vocab_size=32000,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=1024,
        layer_norm_eps=1e-12,
        pad_token_id=0,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.layer_norm_eps = layer_norm_eps
        self.pad_token_id = pad_token_id

class TransformerLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=config.hidden_size,
            num_heads=config.num_attention_heads,
            dropout=config.attention_probs_dropout_prob,
            batch_first=True
        )
        self.feed_forward = nn.Sequential(
            nn.Linear(config.hidden_size, config.intermediate_size),
            nn.GELU(),
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(config.intermediate_size, config.hidden_size),
            nn.Dropout(config.hidden_dropout_prob)
        )
        self.layer_norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.layer_norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
    def forward(self, x, attention_mask=None):
        # Ensure x is 3D: [batch_size, seq_len, hidden_size]
        if len(x.shape) > 3:
            # If 4D or more, reshape to 3D by combining dimensions
            batch_size, seq_len = x.shape[0], x.shape[1]
            x = x.reshape(batch_size, seq_len, -1)
        
        # Self-attention
        residual = x
        x = self.layer_norm1(x)
        
        # PyTorch MultiheadAttention needs specific format for attention mask
        # It should be able to handle various mask formats
        attn_mask = None
        if attention_mask is not None:
            # Assume attention_mask is already in the right format from TimesFMModel
            attn_mask = attention_mask
        
        x, _ = self.attention(x, x, x, attn_mask=attn_mask)
        x = x + residual
        
        # Feed-forward
        residual = x
        x = self.layer_norm2(x)
        x = self.feed_forward(x)
        x = x + residual
        
        return x

class TimesFMModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Embeddings
        self.embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        
        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerLayer(config) for _ in range(config.num_hidden_layers)
        ])
        
        # Output projection
        self.output_projection = nn.Linear(config.hidden_size, config.vocab_size)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    
    def forward(self, input_ids, attention_mask=None, use_cache=False):
        batch_size, seq_length = input_ids.size()
        
        # Create position IDs
        position_ids = torch.arange(seq_length, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
        
        # Get embeddings and ensure they're float32
        embeddings = self.embeddings(input_ids).float()
        position_embeddings = self.position_embeddings(position_ids).float()
        
        # Add position embeddings
        hidden_states = embeddings + position_embeddings
        
        # Create proper attention mask if provided
        attn_mask = None
        if attention_mask is not None:
            # Convert attention mask [batch_size, seq_length] -> [batch_size, 1, 1, seq_length]
            # 1.0 -> no mask, 0.0 -> mask
            extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
            attn_mask = extended_attention_mask
        
        # Apply transformer layers
        for layer in self.layers:
            hidden_states = layer(hidden_states, attn_mask)
        
        # Get logits
        logits = self.output_projection(hidden_states)
        
        return logits

class FeatureProjection(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.projection = nn.Linear(input_dim, output_dim)
        self.layer_norm = nn.LayerNorm(output_dim)
    
    def forward(self, x):
        x = self.projection(x)
        x = self.layer_norm(x)
        return x

class StockTrendAnalyzer:
    def __init__(
        self,
        symbols: List[str],
        context_length: int = 200,
        prediction_length: int = 12,
        d_model: int = 200,
        n_heads: int = 4,
        n_layers: int = 3,
        update_interval: float = 1.0,  # seconds
        websocket_port: int = 8766,
        data_buffer_size: int = 1000  # Buffer size for historical data
    ):
        """
        Initialize the stock trend analyzer.
        
        Args:
            symbols: List of stock symbols to analyze
            context_length: Number of historical data points to use
            prediction_length: Number of future points to predict
            d_model: Model dimension for the transformer (200 to match ASM)
            n_heads: Number of attention heads
            n_layers: Number of transformer layers
            update_interval: How often to update predictions (in seconds)
            websocket_port: Port for real-time updates
            data_buffer_size: Size of the historical data buffer
        """
        self.symbols = symbols
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.update_interval = update_interval
        self.websocket_port = websocket_port
        self.data_buffer_size = data_buffer_size
        
        # Initialize TimesFM model with matching dimensions
        self.timesfm_config = TimesFMConfig(
            vocab_size=32000,
            hidden_size=d_model,
            num_hidden_layers=n_layers,
            num_attention_heads=n_heads,
            max_position_embeddings=context_length + prediction_length,
            intermediate_size=d_model * 4,
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1,
            layer_norm_eps=1e-12,
            pad_token_id=0
        )
        self.timesfm_model = TimesFMModel(self.timesfm_config)
        
        # Initialize ASM transformer
        self.asm_transformer = Transformer(
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            max_context_length=prediction_length
        )
        
        # Initialize feature projection layer
        self.feature_projection = FeatureProjection(input_dim=5, output_dim=d_model)
        
        # Initialize data storage with ring buffer for real-time updates
        self.historical_data: Dict[str, pd.DataFrame] = {}
        self.predictions: Dict[str, np.ndarray] = {}
        self.last_update: Dict[str, datetime] = {}
        self.websocket_clients: List[websockets.WebSocketServerProtocol] = []
        self.data_buffer: Dict[str, pd.DataFrame] = {}
        
        # Initialize tokenizer
        self.tokenizer = BasicTokenizer()
        
        # Move models to appropriate device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.timesfm_model = self.timesfm_model.to(self.device)
        self.feature_projection = self.feature_projection.to(self.device)
        
        # Set models to evaluation mode
        self.timesfm_model.eval()
        self.feature_projection.eval()
        
        # Initialize data buffers
        self._initialize_data_buffers()
        
        logger.info(f"StockTrendAnalyzer initialized for symbols: {symbols}")
        logger.info(f"Using device: {self.device}")
    
    def _initialize_data_buffers(self):
        """Initialize data buffers for all symbols."""
        for symbol in self.symbols:
            try:
                # Fetch initial historical data
                df = self._fetch_historical_data(symbol)
                if df is not None and not df.empty:
                    self.data_buffer[symbol] = df
                    self.historical_data[symbol] = df.tail(self.context_length)
                    logger.info(f"Initialized data buffer for {symbol} with {len(df)} records")
                else:
                    logger.error(f"Failed to initialize data buffer for {symbol}")
            except Exception as e:
                logger.error(f"Error initializing data buffer for {symbol}: {e}")
    
    def _fetch_historical_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """Fetch historical data for a symbol with retries."""
        max_retries = 3
        retry_delay = 1  # seconds
        
        for attempt in range(max_retries):
            try:
                # Get data for a longer period to ensure we have enough
                end_date = datetime.now()
                start_date = end_date - timedelta(days=self.data_buffer_size)
                
                ticker = yf.Ticker(symbol)
                df = ticker.history(start=start_date, end=end_date, interval='1d')
                
                if df.empty:
                    logger.warning(f"No data received for {symbol} on attempt {attempt + 1}")
                    if attempt < max_retries - 1:
                        asyncio.sleep(retry_delay)
                        continue
                    return None
                
                # Select relevant features
                features = ['Open', 'High', 'Low', 'Close', 'Volume']
                df = df[features]
                
                # Handle missing values
                df = df.ffill().bfill()
                
                # Normalize the data
                df = (df - df.mean()) / df.std()
                
                return df
                
            except Exception as e:
                logger.error(f"Error fetching data for {symbol} on attempt {attempt + 1}: {e}")
                if attempt < max_retries - 1:
                    asyncio.sleep(retry_delay)
                    continue
                return None
    
    async def update_data_buffer(self, symbol: str):
        """Update the data buffer with new data."""
        try:
            new_data = self._fetch_historical_data(symbol)
            if new_data is not None and not new_data.empty:
                # Update the buffer
                if symbol in self.data_buffer:
                    self.data_buffer[symbol] = pd.concat([
                        self.data_buffer[symbol],
                        new_data
                    ]).tail(self.data_buffer_size)
                else:
                    self.data_buffer[symbol] = new_data
                
                # Update historical data for predictions
                self.historical_data[symbol] = self.data_buffer[symbol].tail(self.context_length)
                
                logger.info(f"Updated data buffer for {symbol} with {len(new_data)} new records")
            else:
                logger.warning(f"No new data available for {symbol}")
        except Exception as e:
            logger.error(f"Error updating data buffer for {symbol}: {e}")
    
    async def start_websocket_server(self):
        """Start the websocket server for real-time updates."""
        async def handler(websocket, path):
            self.websocket_clients.append(websocket)
            try:
                async for message in websocket:
                    # Handle incoming messages if needed
                    pass
            finally:
                self.websocket_clients.remove(websocket)
        
        server = await websockets.serve(handler, "localhost", self.websocket_port)
        logger.info(f"WebSocket server started on port {self.websocket_port}")
        return server
    
    async def broadcast_update(self, analysis: Dict):
        """Broadcast analysis updates to all connected clients."""
        if self.websocket_clients:
            message = json.dumps(analysis)
            await asyncio.gather(
                *[client.send(message) for client in self.websocket_clients]
            )
    
    def prepare_input_tensor(self, data: pd.DataFrame) -> Tuple[np.ndarray, torch.Tensor]:
        """Prepare input tensors for both models."""
        try:
            # Get the last context_length rows
            recent_data = data.tail(self.context_length)
            
            # Prepare features for ASM transformer
            features = recent_data[['Open', 'High', 'Low', 'Close', 'Volume']].values
            features = (features - features.mean()) / (features.std() + 1e-8)
            
            # Project to model dimension for ASM transformer
            # features shape: [context_length, 5]
            # projection weight shape: [d_model, 5]
            # Need to transpose projection weight for correct matrix multiplication
            projection_weight = self.feature_projection.projection.weight.detach().cpu().numpy()
            asm_tensor = np.matmul(features, projection_weight.T)  # Result: [context_length, d_model]
            
            # Add batch dimension and ensure correct shape for ASM transformer
            # We need to take only the last prediction_length rows
            asm_tensor = asm_tensor[-self.prediction_length:, :]  # Take last prediction_length rows
            asm_tensor = asm_tensor.reshape(1, self.prediction_length, -1)  # Add batch dimension
            
            # Prepare input for TimesFM model
            # Convert to tensor and add batch dimension
            timesfm_tensor = torch.tensor(features, dtype=torch.float32, device=self.device)
            timesfm_tensor = timesfm_tensor.unsqueeze(0)  # Add batch dimension
            
            # Convert float values to discrete token IDs for TimesFM
            # Scale and shift to get positive values, then clamp to vocab size
            timesfm_tensor = ((timesfm_tensor + 3) * 5000).long().clamp(0, self.timesfm_config.vocab_size - 1)
            
            # Take mean across feature dimension to get single token ID per timestep
            # Convert to float before taking mean, then back to long for token IDs
            timesfm_tensor = timesfm_tensor.float().mean(dim=2).long()
            
            return asm_tensor, timesfm_tensor
            
        except Exception as e:
            logger.error(f"Error preparing input tensors: {e}")
            raise
    
    async def update_predictions(self, symbol: str) -> None:
        """Update predictions for a symbol using both models."""
        try:
            # Update data buffer first
            await self.update_data_buffer(symbol)
            
            # Check if we have enough data
            if symbol not in self.historical_data or len(self.historical_data[symbol]) < self.context_length:
                logger.warning(f"Insufficient data for {symbol}, skipping prediction update")
                return
            
            # Prepare input tensors
            asm_tensor, timesfm_tensor = self.prepare_input_tensor(self.historical_data[symbol])
            
            # Get TimesFM predictions with optimized inference
            with torch.no_grad():
                # TimesFM expects [batch_size, seq_len] for input_ids
                # timesfm_tensor should already be in this shape from prepare_input_tensor
                
                # Project input to hidden dimension and get predictions
                timesfm_output = self.timesfm_model(
                    timesfm_tensor,
                    attention_mask=None,
                    use_cache=True
                )
                
                # timesfm_output shape: [batch_size, seq_len, vocab_size]
                
                # Extract the last prediction_length predictions
                timesfm_predictions = timesfm_output[:, -self.prediction_length:, :]
                
                # Scale predictions back to original range
                timesfm_predictions = (timesfm_predictions / 5000) - 3
                
                # Project back to feature dimension using transpose of feature projection
                # First, ensure we're working with float tensors for mean operation
                timesfm_predictions = timesfm_predictions.float()
                # Get average embedding for each position  
                timesfm_predictions = timesfm_predictions.mean(dim=2, keepdim=True).expand(-1, -1, self.timesfm_config.hidden_size)
                
                # Project from hidden_size to feature dimension (5)
                # We need to transpose properly for matrix multiplication
                projection_weight = self.feature_projection.projection.weight  # shape: [d_model, input_dim]
                timesfm_predictions = torch.matmul(timesfm_predictions, projection_weight)  # result: [batch, seq, input_dim]
            
            # Convert TimesFM output to numpy (with proper detachment)
            timesfm_predictions = timesfm_predictions.detach().cpu().numpy()
            
            # Get ASM transformer predictions using the projected tensor
            asm_predictions = self.asm_transformer.fully_fused_forward(
                asm_tensor
            )
            
            # Project ASM predictions back to feature dimension
            # ASM predictions are in shape [batch, seq, d_model]
            # We need to project back to [batch, seq, 5]
            # Use the feature projection weight directly (not transposed)
            asm_predictions = np.matmul(
                asm_predictions,
                self.feature_projection.projection.weight.detach().cpu().numpy()
            )
            
            # Both models should now output predictions with shape [1, prediction_length, 5]
            # Combine predictions with weighted average
            # Give more weight to ASM predictions for real-time data
            combined_predictions = 0.7 * asm_predictions + 0.3 * timesfm_predictions
            
            # Store predictions
            self.predictions[symbol] = combined_predictions
            self.last_update[symbol] = datetime.now()
            
            logger.info(f"Updated predictions for {symbol}")
            
        except Exception as e:
            logger.error(f"Error updating predictions for {symbol}: {e}")
            raise
    
    async def get_trend_analysis(self, symbol: str) -> Dict:
        """Get trend analysis for a symbol."""
        try:
            # Check if we need to update predictions
            if (symbol not in self.last_update or 
                (datetime.now() - self.last_update[symbol]).total_seconds() >= self.update_interval):
                await self.update_predictions(symbol)
            
            # Check if we have predictions
            if symbol not in self.predictions:
                return {
                    "symbol": symbol,
                    "error": "No predictions available",
                    "last_updated": datetime.now().isoformat()
                }
            
            # Get the latest predictions
            predictions = self.predictions[symbol]
            
            # Calculate trend metrics
            current_price = self.historical_data[symbol]['Close'].iloc[-1]
            predicted_prices = predictions[0, -self.prediction_length:, 3]  # Close price predictions
            
            # Calculate trend direction and strength
            price_change = predicted_prices[-1] - current_price
            trend_direction = "up" if price_change > 0 else "down"
            trend_strength = abs(price_change) / current_price
            
            # Calculate confidence score based on prediction consistency
            confidence = np.mean(np.abs(np.diff(predicted_prices))) / current_price
            
            return {
                "symbol": symbol,
                "current_price": current_price,
                "predicted_prices": predicted_prices.tolist(),
                "trend_direction": trend_direction,
                "trend_strength": float(trend_strength),
                "confidence": float(confidence),
                "last_updated": self.last_update[symbol].isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting trend analysis for {symbol}: {e}")
            return {
                "symbol": symbol,
                "error": str(e),
                "last_updated": datetime.now().isoformat()
            }
    
    async def run_real_time_analysis(self):
        """Run real-time analysis loop."""
        server = await self.start_websocket_server()
        
        try:
            while True:
                for symbol in self.symbols:
                    try:
                        analysis = await self.get_trend_analysis(symbol)
                        await self.broadcast_update(analysis)
                    except Exception as e:
                        logger.error(f"Error in real-time analysis for {symbol}: {e}")
                
                await asyncio.sleep(self.update_interval)
                
        except Exception as e:
            logger.error(f"Error in real-time analysis loop: {e}")
        finally:
            server.close()
            await server.wait_closed()

async def main():
    # Example usage
    symbols = ["AAPL", "GOOGL", "MSFT"]
    analyzer = StockTrendAnalyzer(
        symbols=symbols,
        update_interval=1.0,  # Update every second
        websocket_port=8766,
        data_buffer_size=1000  # Store 1000 days of historical data
    )
    
    # Start real-time analysis
    await analyzer.run_real_time_analysis()

if __name__ == "__main__":
    asyncio.run(main()) 