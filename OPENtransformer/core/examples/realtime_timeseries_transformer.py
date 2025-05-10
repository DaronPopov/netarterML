import numpy as np
import torch
import pandas as pd
from transformers import TimeSeriesTransformerForPrediction, TimeSeriesTransformerConfig
import yfinance as yf
from datetime import datetime, timedelta
import logging
from typing import List, Dict, Optional
import time
from dataclasses import dataclass
from statistics import mean, stdev
import argparse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='timeseries_transformer_debug.log',
    filemode='w'
)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console.setFormatter(formatter)
logging.getLogger('').addHandler(console)

logger = logging.getLogger(__name__)

@dataclass
class BenchmarkStats:
    """Statistics for benchmarking operations."""
    operation: str
    times: List[float]
    success_rate: float
    avg_time: float
    std_time: float
    min_time: float
    max_time: float

class Benchmarker:
    def __init__(self):
        self.stats = {}
        self.operation_start_times = {}
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        pass
    
    def measure(self, operation: str):
        """Measure execution time of an operation."""
        start_time = time.time()
        success = True
        try:
            yield
        except Exception:
            success = False
            raise
        finally:
            end_time = time.time()
            duration = end_time - start_time
            
            if operation not in self.stats:
                self.stats[operation] = {
                    'times': [],
                    'successes': 0,
                    'total': 0
                }
            
            self.stats[operation]['times'].append(duration)
            self.stats[operation]['total'] += 1
            if success:
                self.stats[operation]['successes'] += 1
    
    def get_stats(self, operation: str) -> BenchmarkStats:
        """Get statistics for an operation."""
        if operation not in self.stats:
            return None
        
        data = self.stats[operation]
        times = data['times']
        success_rate = data['successes'] / data['total'] if data['total'] > 0 else 0
        
        return BenchmarkStats(
            operation=operation,
            times=times,
            success_rate=success_rate,
            avg_time=mean(times) if times else 0,
            std_time=stdev(times) if len(times) > 1 else 0,
            min_time=min(times) if times else 0,
            max_time=max(times) if times else 0
        )
    
    def print_stats(self):
        """Print statistics for all operations."""
        print("\n" + "="*50)
        print("PERFORMANCE METRICS")
        print("="*50)
        
        for operation in self.stats:
            stats = self.get_stats(operation)
            print(f"\n{operation}:")
            print(f"  Success Rate: {stats.success_rate*100:.1f}%")
            print(f"  Avg Time: {stats.avg_time*1000:.1f}ms")
            print(f"  Std Dev: {stats.std_time*1000:.1f}ms")
            print(f"  Min Time: {stats.min_time*1000:.1f}ms")
            print(f"  Max Time: {stats.max_time*1000:.1f}ms")

class RealtimeTimeSeriesTransformer:
    def __init__(self, 
                 symbols: List[str],
                 prediction_length: int = 12,
                 context_length: int = 96,
                 d_model: int = 64,
                 n_heads: int = 4,
                 n_layers: int = 3,
                 dropout: float = 0.1):
        """Initialize the Time Series Transformer model."""
        self.symbols = symbols
        self.prediction_length = prediction_length
        self.context_length = context_length
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
        # Initialize model configuration
        config = TimeSeriesTransformerConfig(
            prediction_length=prediction_length,
            context_length=context_length,
            input_size=len(symbols),
            scaling=True,
            num_parallel_samples=100,
            d_model=d_model,
            num_attention_heads=n_heads,
            num_trainable_samples=100,
            dropout=dropout,
            encoder_layers=n_layers,
            decoder_layers=n_layers,
            activation_function="gelu",
            use_cache=True
        )
        
        # Initialize model
        self.model = TimeSeriesTransformerForPrediction(config).to(self.device)
        self.model.eval()
        
        # Initialize benchmarker
        self.benchmarker = Benchmarker()
        
        logger.info(f"Initialized Time Series Transformer on device: {self.device}")
    
    def prepare_data(self, data: pd.DataFrame) -> torch.Tensor:
        """Prepare data for the model."""
        # Convert data to tensor and add batch dimension
        data_tensor = torch.FloatTensor(data.values).unsqueeze(0)
        return data_tensor.to(self.device)
    
    def predict(self, data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Generate predictions for the input data."""
        try:
            with torch.no_grad():
                # Prepare input data
                input_data = self.prepare_data(data)
                
                # Generate predictions
                outputs = self.model.generate(
                    input_data,
                    num_beams=5,
                    max_length=self.context_length + self.prediction_length,
                    num_parallel_samples=100
                )
                
                # Convert predictions to numpy array
                predictions = outputs.sequences.cpu().numpy()
                
                # Extract predictions for each symbol
                result = {}
                for i, symbol in enumerate(self.symbols):
                    result[symbol] = predictions[0, -self.prediction_length:, i]
                
                return result
                
        except Exception as e:
            logger.error(f"Error in prediction: {e}")
            # Return zero predictions in case of error
            return {symbol: np.zeros(self.prediction_length) for symbol in self.symbols}
    
    def fetch_data(self, lookback: str = '1d', interval: str = '1m') -> pd.DataFrame:
        """Fetch real-time data for all symbols."""
        data = {}
        
        for symbol in self.symbols:
            try:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period=lookback, interval=interval)
                
                if not hist.empty:
                    # Use closing prices
                    data[symbol] = hist['Close']
                else:
                    logger.warning(f"No data received for {symbol}")
                    data[symbol] = pd.Series([0] * self.context_length)
                    
            except Exception as e:
                logger.error(f"Error fetching data for {symbol}: {e}")
                data[symbol] = pd.Series([0] * self.context_length)
        
        # Convert to DataFrame
        df = pd.DataFrame(data)
        
        # Handle missing values
        df = df.fillna(method='ffill').fillna(method='bfill')
        
        # Ensure we have enough data points
        if len(df) < self.context_length:
            # Pad with zeros if we don't have enough data
            pad_length = self.context_length - len(df)
            df = pd.concat([pd.DataFrame([[0] * len(self.symbols)] * pad_length, columns=self.symbols), df])
        
        # Take only the last context_length points
        df = df.tail(self.context_length)
        
        return df

def process_callback(predictions: Dict[str, np.ndarray], symbols: List[str], transformer: RealtimeTimeSeriesTransformer):
    """Process and display predictions."""
    try:
        # Clear screen and move cursor to top
        print("\033[2J\033[H", end="")
        logger.info("Screen cleared")
        
        # Get current timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Print header
        print("\033[1;36m" + "="*100)
        print(f"AI-POWERED FINANCIAL MARKET ANALYSIS - {timestamp}")
        print("="*100 + "\033[0m\n")
        
        # Print column headers
        print("\033[1;33m" + f"{'Symbol':<8} {'Current':<10} {'Predicted':<10} {'Change':<10} {'Trend':<8}" + "\033[0m")
        print("-" * 70)
        
        # Print predictions for each symbol
        for symbol in symbols:
            try:
                # Get current price
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period='1d', interval='1m')
                current_price = hist['Close'].iloc[-1]
                
                # Get predictions
                pred = predictions[symbol][-1]
                
                # Calculate metrics
                price_change = pred - current_price
                price_change_pct = (price_change / current_price) * 100
                
                # Determine trend
                trend = "UP" if price_change > 0 else "DOWN" if price_change < 0 else "SIDE"
                trend_color = "\033[92m" if trend == "UP" else "\033[91m" if trend == "DOWN" else "\033[93m"
                
                # Print formatted line
                print(f"{symbol:<8} "
                      f"${current_price:,.2f} "
                      f"${pred:,.2f} "
                      f"{price_change_pct:+.2f}% "
                      f"{trend_color}{trend:<8}\033[0m")
                
            except Exception as e:
                logger.error(f"Error processing symbol {symbol}: {e}")
                print(f"{symbol:<8} {'ERROR':<10} {'ERROR':<10} {'ERROR':<10} {'ERROR':<8}")
        
        # Print footer
        print("-" * 70)
        current_time = datetime.now()
        prediction_time = current_time + timedelta(minutes=transformer.prediction_length)
        print(f"\nPrediction Window: {current_time.strftime('%H:%M:%S')} → {prediction_time.strftime('%H:%M:%S')}")
        print("\033[1;36m" + "="*100 + "\033[0m")
        
        # Hide cursor
        print("\033[?25l", end="")
        logger.info("Display completed successfully")
        
    except Exception as e:
        logger.error(f"Error in callback: {e}")
        logger.error("Full traceback:", exc_info=True)

def get_top_symbols(n: int = 10) -> List[str]:
    """Get the top n most actively traded symbols."""
    try:
        # Pre-filtered list of major stocks
        major_stocks = [
            'AAPL',  # Apple
            'MSFT',  # Microsoft
            'GOOGL', # Google
            'AMZN',  # Amazon
            'NVDA',  # NVIDIA
            'META',  # Meta (Facebook)
            'TSLA',  # Tesla
            'BRK-B', # Berkshire Hathaway
            'JPM',   # JPMorgan Chase
            'V',     # Visa
            'XOM',   # Exxon Mobil
            'WMT',   # Walmart
            'JNJ',   # Johnson & Johnson
            'MA',    # Mastercard
            'PG'     # Procter & Gamble
        ]
        
        # Get market data for pre-filtered list
        market_data = {}
        for symbol in major_stocks:
            try:
                ticker = yf.Ticker(symbol)
                info = ticker.info
                market_data[symbol] = {
                    'market_cap': info.get('marketCap', 0),
                    'volume': info.get('averageVolume', 0)
                }
                logger.info(f"Got market data for {symbol}")
            except Exception as e:
                logger.warning(f"Could not get market data for {symbol}: {e}")
                continue
        
        # Sort by market cap and volume
        sorted_symbols = sorted(
            market_data.items(),
            key=lambda x: (x[1]['market_cap'], x[1]['volume']),
            reverse=True
        )
        
        # Get top n symbols
        top_symbols = [symbol for symbol, _ in sorted_symbols[:n]]
        logger.info(f"Selected top {n} symbols: {top_symbols}")
        return top_symbols
        
    except Exception as e:
        logger.error(f"Error getting top symbols: {e}")
        # Fallback to default symbols if there's an error
        return ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'NVDA', 'META', 'TSLA', 'BRK-B', 'JPM', 'V']

def run_benchmark(model: RealtimeTimeSeriesTransformer, symbols: List[str], num_iterations: int = 10):
    """Run a short benchmark test."""
    print("\n" + "="*50)
    print("RUNNING BENCHMARK TEST")
    print("="*50)
    print(f"Number of iterations: {num_iterations}")
    print(f"Number of symbols: {len(symbols)}")
    print("="*50 + "\n")
    
    try:
        for i in range(num_iterations):
            try:
                logger.info(f"Benchmark iteration {i+1}/{num_iterations}")
                
                # Fetch data
                with model.benchmarker.measure("data_fetch"):
                    data = model.fetch_data()
                
                # Generate predictions
                with model.benchmarker.measure("prediction"):
                    predictions = model.predict(data)
                
                # Print intermediate stats every iteration
                model.benchmarker.print_stats()
                
            except Exception as e:
                logger.error(f"Error in benchmark iteration {i+1}: {e}")
                logger.error("Full traceback:", exc_info=True)
                continue
            
    except KeyboardInterrupt:
        print("\nBenchmark interrupted by user")
    finally:
        print("\n" + "="*50)
        print("BENCHMARK COMPLETE")
        print("="*50)
        model.benchmarker.print_stats()

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run real-time time series transformer')
    parser.add_argument('--benchmark', action='store_true', help='Run in benchmark mode')
    parser.add_argument('--iterations', type=int, default=10, help='Number of benchmark iterations')
    args = parser.parse_args()
    
    # Get top 10 symbols
    logger.info("Fetching top 10 symbols...")
    symbols = get_top_symbols(10)
    
    # Initialize model
    logger.info("Initializing Time Series Transformer...")
    model = RealtimeTimeSeriesTransformer(
        symbols=symbols,
        prediction_length=12,  # Predict next 12 minutes
        context_length=96,     # Use last 96 minutes of data
        d_model=64,           # Model dimension
        n_heads=4,            # Number of attention heads
        n_layers=3,           # Number of transformer layers
        dropout=0.1           # Dropout rate
    )
    
    if args.benchmark:
        # Run benchmark mode
        run_benchmark(model, symbols, args.iterations)
    else:
        # Run normal mode
        logger.info("Starting real-time processing...")
        
        try:
            while True:
                try:
                    # Fetch data
                    start_time = time.time()
                    data = model.fetch_data()
                    duration = time.time() - start_time
                    model.benchmarker.stats.setdefault("data_fetch", {'times': [], 'successes': 0, 'total': 0})
                    model.benchmarker.stats["data_fetch"]['times'].append(duration)
                    model.benchmarker.stats["data_fetch"]['total'] += 1
                    model.benchmarker.stats["data_fetch"]['successes'] += 1
                    
                    # Generate predictions
                    start_time = time.time()
                    predictions = model.predict(data)
                    duration = time.time() - start_time
                    model.benchmarker.stats.setdefault("prediction", {'times': [], 'successes': 0, 'total': 0})
                    model.benchmarker.stats["prediction"]['times'].append(duration)
                    model.benchmarker.stats["prediction"]['total'] += 1
                    model.benchmarker.stats["prediction"]['successes'] += 1
                    
                    # Process and display results
                    process_callback(predictions, symbols, model)
                    
                    # Wait for next update
                    time.sleep(0.9)  # Wait just under 1 minute
                    
                except Exception as e:
                    logger.error(f"Error in main loop: {e}")
                    logger.error("Full traceback:", exc_info=True)
                    time.sleep(5)  # Wait 5 seconds before retrying
                    
        except KeyboardInterrupt:
            print("\033[?25h")  # Show cursor
            logger.info("Stopping real-time processing...")

if __name__ == "__main__":
    main() 