import asyncio
import yfinance as yf
import time
import logging
import sys
import os
import pandas as pd
from finlib.core.examples.stock_trend_analyzer import StockTrendAnalyzer
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
import csv

#####this is a test script to see how long it takes to run the model inference on incoming stock data######



# Suppress all logging
logging.getLogger().setLevel(logging.CRITICAL)
for logger_name in logging.root.manager.loggerDict:
    logging.getLogger(logger_name).setLevel(logging.CRITICAL)
    logging.getLogger(logger_name).propagate = False

# Configure logging for the script
logging.basicConfig(level=logging.CRITICAL)
logger = logging.getLogger(__name__)

# Create a simple output function that bypasses logging
def print_status(message):
    sys.stdout.write(message + "\n")
    sys.stdout.flush()

class SuppressOutput:
    def __enter__(self):
        self._original_stdout = sys.stdout
        self._original_stderr = sys.stderr
        sys.stdout = open(os.devnull, 'w')
        sys.stderr = open(os.devnull, 'w')
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stderr.close()
        sys.stdout = self._original_stdout
        sys.stderr = self._original_stderr

class RateLimiter:
    def __init__(self, max_requests_per_min=300):
        self.max_requests = max_requests_per_min
        self.requests = []
        self.lock = asyncio.Lock()
    
    async def acquire(self):
        async with self.lock:
            now = time.time()
            self.requests = [req for req in self.requests if now - req < 60]
            if len(self.requests) >= self.max_requests:
                sleep_time = 1 - (now - self.requests[0])
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)
                self.requests = self.requests[1:]
            self.requests.append(now)

class TimingStats:
    def __init__(self):
        self.fetch_times = []
        self.analysis_times = []
        self.total_times = []
        self.inference_times = []
        self.stocks_processed = 0
        self.start_time = time.time()
        self.total_flops = 0
    
    def add_fetch_time(self, time_taken):
        self.fetch_times.append(time_taken)
    
    def add_analysis_time(self, time_taken):
        self.analysis_times.append(time_taken)
    
    def add_total_time(self, time_taken):
        self.total_times.append(time_taken)
    
    def add_inference_time(self, time_taken):
        self.inference_times.append(time_taken)
    
    def add_flops(self, flops):
        self.total_flops += flops
    
    def calculate_flops(self, batch_size, seq_len, d_model, n_heads, n_layers):
        """Calculate the number of floating-point operations for a transformer model."""
        head_dim = d_model // n_heads
        
        # Per-layer flops
        # Self-attention: 4 * batch_size * seq_len * d_model^2 (QKV projections + output projection)
        qkv_flops = 3 * batch_size * seq_len * d_model * d_model
        attn_matmul_flops = batch_size * n_heads * seq_len * seq_len * head_dim
        attn_output_flops = batch_size * seq_len * d_model * d_model
        
        # FFN: 2 * batch_size * seq_len * d_model * (4*d_model)
        ffn_flops = 2 * batch_size * seq_len * d_model * (4 * d_model)
        
        # Layer norm: 5 * batch_size * seq_len * d_model each (2 per layer)
        ln_flops = 2 * 5 * batch_size * seq_len * d_model
        
        # Total flops per layer
        flops_per_layer = qkv_flops + attn_matmul_flops + attn_output_flops + ffn_flops + ln_flops
        
        # Total flops for all layers
        total_flops = flops_per_layer * n_layers
        
        return total_flops
    
    def get_stats(self):
        if not self.fetch_times:
            return None
        
        # Calculate GFLOPS
        total_time = sum(self.total_times)
        gflops = (self.total_flops / total_time) / 1e9 if total_time > 0 else 0
        
        # Calculate average inference time
        avg_inference_time = sum(self.inference_times) / len(self.inference_times) if self.inference_times else 0
        
        return {
            "avg_fetch_time": sum(self.fetch_times) / len(self.fetch_times),
            "avg_analysis_time": sum(self.analysis_times) / len(self.analysis_times),
            "avg_total_time": sum(self.total_times) / len(self.total_times),
            "avg_inference_time": avg_inference_time,
            "total_stocks": self.stocks_processed,
            "elapsed_time": time.time() - self.start_time,
            "stocks_per_second": self.stocks_processed / (time.time() - self.start_time),
            "gflops": gflops,
            "total_flops": self.total_flops
        }

class StockTerminal:
    def __init__(self, batch_size=10):
        # Initialize with suppressed output
        with SuppressOutput():
            self.batch_size = batch_size
            self.yf_limiter = RateLimiter(max_requests_per_min=300)
            self.thread_pool = ThreadPoolExecutor(max_workers=10)
            
            self.analyzer = StockTrendAnalyzer(
                symbols=[],
                update_interval=0.1,
                data_buffer_size=1000
            )
            
            self.last_update = {}
            self.predictions = {}
            self.current_prices = {}
            self.price_changes = {}
            self.stock_details = {}
            self.processed_stocks = set()
            self.total_stocks = 0
            self.current_batch = []
            self.last_processed_time = {}
            self.cache = {}
            self.cache_timeout = 30
            self.prediction_cache = {}
            self.prediction_cache_timeout = 60
            
            # Initialize timing stats
            self.timing_stats = TimingStats()
            
            # Load S&P 500 stocks
            self.load_sp500_stocks()
        
        # Clear terminal and setup display
        os.system('clear' if os.name == 'posix' else 'cls')
        print("\033[?25l")  # Hide cursor
        
        # Print header
        print("\033[1;34m" + "=" * 80)
        print("S&P 500 Stock Market Analysis".center(80))
        print("=" * 80 + "\033[0m")
        print("\033[1;37m" + "Symbol".ljust(8) + "Price".rjust(12) + "Change".rjust(10) + 
              "Trend".rjust(10) + "Vol".rjust(10) + "Momentum".rjust(10) + 
              "Volume".rjust(12) + "Conf".rjust(10) + "Updated".rjust(10) + "\033[0m")
        print("-" * 80)

    def print_stock_row(self, symbol, current_price, price_change, trend, volatility, momentum, volume, confidence, last_update):
        """Print a single stock row with colors."""
        # Format price change
        change_color = "\033[92m" if price_change >= 0 else "\033[91m"  # Green or Red
        change_text = f"{price_change:+.2f}%"
        
        # Format trend
        trend_color = "\033[92m" if trend == "up" else "\033[91m"  # Green or Red
        trend_symbol = "↑" if trend == "up" else "↓"
        
        # Format momentum
        momentum_color = "\033[92m" if momentum > 0 else "\033[91m"  # Green or Red
        
        # Format volume with K/M/B
        if volume >= 1e9:
            volume_text = f"{volume/1e9:.1f}B"
        elif volume >= 1e6:
            volume_text = f"{volume/1e6:.1f}M"
        elif volume >= 1e3:
            volume_text = f"{volume/1e3:.1f}K"
        else:
            volume_text = f"{volume:,.0f}"
        
        # Print the row
        print(f"{symbol:<8}${current_price:>11.2f}{change_color}{change_text:>10}\033[0m"
              f"{trend_color}{trend_symbol:>10}\033[0m{volatility:>10.1%}"
              f"{momentum_color}{momentum:>+10.1%}\033[0m{volume_text:>12}"
              f"{confidence:>10.1%}{last_update:>10}")

    def update_display(self):
        """Update the terminal display with current stock data."""
        # Build the display buffer
        display_buffer = []
        
        # Sort stocks by last update time
        all_stocks = sorted(
            list(self.processed_stocks) + [s for s in self.current_batch if s not in self.processed_stocks],
            key=lambda x: self.last_processed_time.get(x, 0),
            reverse=True
        )
        
        # Build each row in the buffer
        for symbol in all_stocks:
            current_price = self.current_prices.get(symbol, 0)
            price_change = self.price_changes.get(symbol, 0)
            details = self.stock_details.get(symbol, {})
            last_update = self.last_processed_time.get(symbol, 0)
            
            if symbol in self.predictions:
                pred = self.predictions[symbol]
                volatility = details.get('volatility', 0)
                momentum = details.get('momentum', 0)
                avg_volume = details.get('avg_volume', 0)
                update_time = datetime.fromtimestamp(last_update).strftime('%H:%M:%S')
                
                # Format the row
                change_color = "\033[92m" if price_change >= 0 else "\033[91m"
                change_text = f"{price_change:+.2f}%"
                trend_color = "\033[92m" if pred["trend_direction"] == "up" else "\033[91m"
                trend_symbol = "↑" if pred["trend_direction"] == "up" else "↓"
                momentum_color = "\033[92m" if momentum > 0 else "\033[91m"
                
                if avg_volume >= 1e9:
                    volume_text = f"{avg_volume/1e9:.1f}B"
                elif avg_volume >= 1e6:
                    volume_text = f"{avg_volume/1e6:.1f}M"
                elif avg_volume >= 1e3:
                    volume_text = f"{avg_volume/1e3:.1f}K"
                else:
                    volume_text = f"{avg_volume:,.0f}"
                
                row = (f"{symbol:<8}${current_price:>11.2f}{change_color}{change_text:>10}\033[0m"
                      f"{trend_color}{trend_symbol:>10}\033[0m{volatility:>10.1%}"
                      f"{momentum_color}{momentum:>+10.1%}\033[0m{volume_text:>12}"
                      f"{pred['confidence']:>10.1%}{update_time:>10}")
            else:
                row = f"{symbol:<8}Processing...{' ' * 70}"
            
            display_buffer.append(row)
        
        # Get current stats if available
        stats = self.timing_stats.get_stats()
        stats_text = ""
        if stats:
            stats_text = (
                f"\n\033[1;34mPERFORMANCE BENCHMARKS:\033[0m\n"
                f"  Average Fetch Time: {stats['avg_fetch_time']*1000:.2f}ms\n"
                f"  Average Analysis Time: {stats['avg_analysis_time']*1000:.2f}ms\n"
                f"  Average Total Time: {stats['avg_total_time']*1000:.2f}ms\n"
                f"  Average Inference Time: {stats['avg_inference_time']*1000:.2f}ms\n"
                f"  Total Stocks Processed: {stats['total_stocks']}\n"
                f"  Processing Rate: {stats['stocks_per_second']:.2f} stocks/second\n"
                f"  Total Elapsed Time: {stats['elapsed_time']:.2f}s\n"
                f"  Performance: {stats['gflops']:.2f} GFLOPS\n"
                f"  Total FLOPS: {stats['total_flops']/1e9:.2f} GFLOPS"
            )
        
        # Build footer
        footer = [
            "\033[1;34m" + "-" * 80,
            f"Processed: {len(self.processed_stocks)}/{self.total_stocks} stocks | "
            f"Current Batch: {', '.join(self.current_batch)} | "
            f"Press Ctrl+C to exit".center(80),
            "=" * 80 + "\033[0m"
        ]
        
        # Clear screen and update all at once
        print("\033[4;0H")  # Move to line 4 (after header)
        print("\n".join(display_buffer))
        print("\n".join(footer))
        
        # Print stats if available
        if stats_text:
            print(stats_text)
        
        # Flush output to ensure immediate display
        sys.stdout.flush()

    async def fetch_stock_data_parallel(self, symbols: list):
        """Fetch all stock data from yfinance in parallel."""
        fetch_start = time.time()
        async def fetch_single(symbol):
            try:
                await self.yf_limiter.acquire()
                
                # Check cache first
                cache_key = f"{symbol}_data"
                if cache_key in self.cache:
                    cache_time, cache_data = self.cache[cache_key]
                    if time.time() - cache_time < self.cache_timeout:
                        return symbol, cache_data
                
                ticker = yf.Ticker(symbol)
                data = await asyncio.get_event_loop().run_in_executor(
                    self.thread_pool,
                    lambda: ticker.history(period='30d', interval='1d')
                )
                
                if not data.empty:
                    # Calculate metrics
                    closes = data['Close'].values
                    volumes = data['Volume'].values
                    daily_returns = pd.Series(closes).pct_change().dropna()
                    volatility = daily_returns.std() * (252 ** 0.5)
                    avg_volume = volumes.mean()
                    momentum = (closes[-1] - closes[0]) / closes[0] if len(closes) > 20 else 0
                    
                    result = {
                        'current_price': closes[-1],
                        'prev_price': closes[-2],
                        'price_change': ((closes[-1] - closes[-2]) / closes[-2]) * 100,
                        'volatility': volatility,
                        'avg_volume': avg_volume,
                        'momentum': momentum,
                        'last_close': closes[-1],
                        'price_range': {'min': closes.min(), 'max': closes.max()}
                    }
                    
                    self.cache[cache_key] = (time.time(), result)
                    return symbol, result
                    
            except Exception:
                pass
            return symbol, None

        # Process symbols in chunks
        chunk_size = 10
        results = []
        for i in range(0, len(symbols), chunk_size):
            chunk = symbols[i:i + chunk_size]
            chunk_results = await asyncio.gather(*[fetch_single(symbol) for symbol in chunk])
            results.extend(chunk_results)
        
        # Update all data at once
        for symbol, data in results:
            if data:
                self.current_prices[symbol] = data['current_price']
                self.price_changes[symbol] = data['price_change']
                self.stock_details[symbol] = {
                    'volatility': data['volatility'],
                    'avg_volume': data['avg_volume'],
                    'momentum': data['momentum'],
                    'last_close': data['last_close'],
                    'price_range': data['price_range']
                }
        
        fetch_time = time.time() - fetch_start
        self.timing_stats.add_fetch_time(fetch_time)

    async def process_batch(self, symbols):
        """Process a batch of symbols efficiently with suppressed output."""
        batch_start = time.time()
        with SuppressOutput():
            # Fetch all data from yfinance
            await self.fetch_stock_data_parallel(symbols)
            
            # Update predictions with caching
            analysis_start = time.time()
            for symbol in symbols:
                if symbol not in self.processed_stocks:
                    cache_key = f"{symbol}_prediction"
                    if cache_key in self.prediction_cache:
                        cache_time, cache_data = self.prediction_cache[cache_key]
                        if time.time() - cache_time < self.prediction_cache_timeout:
                            self.predictions[symbol] = cache_data
                            self.processed_stocks.add(symbol)
                            self.last_processed_time[symbol] = time.time()
                            continue
                    
                    # Track model inference time
                    inference_start = time.time()
                    analysis = await self.analyzer.get_trend_analysis(symbol)
                    inference_time = time.time() - inference_start
                    self.timing_stats.add_inference_time(inference_time)
                    
                    # Calculate and add FLOPS for this inference
                    # Assuming model parameters from StockTrendAnalyzer
                    flops = self.timing_stats.calculate_flops(
                        batch_size=1,  # Single stock analysis
                        seq_len=30,     # 30 days of data
                        d_model=256,    # Model dimension
                        n_heads=8,      # Number of attention heads
                        n_layers=6      # Number of transformer layers
                    )
                    self.timing_stats.add_flops(flops)
                    
                    if "error" not in analysis:
                        self.predictions[symbol] = analysis
                        self.prediction_cache[cache_key] = (time.time(), analysis)
                        self.processed_stocks.add(symbol)
                        self.last_processed_time[symbol] = time.time()
            
            analysis_time = time.time() - analysis_start
            self.timing_stats.add_analysis_time(analysis_time)
            
            batch_time = time.time() - batch_start
            self.timing_stats.add_total_time(batch_time)
            self.timing_stats.stocks_processed += len(symbols)

    async def run(self):
        """Run the terminal UI with optimized execution flow."""
        try:
            last_display_update = 0
            display_update_interval = 0.1
            last_benchmark_time = time.time()
            benchmark_interval = 60  # Show benchmarks every 60 seconds
            
            while True:
                current_time = time.time()
                
                if self.current_batch:
                    # Process current batch with suppressed output
                    await self.process_batch(self.current_batch)
                    
                    # Check if batch is complete and get next batch
                    if all(symbol in self.processed_stocks for symbol in self.current_batch):
                        self.current_batch = self.get_next_batch()
                        if self.current_batch:
                            self.analyzer.symbols = self.current_batch
                
                # Update display at fixed intervals
                if current_time - last_display_update >= display_update_interval:
                    self.update_display()
                    last_display_update = current_time
                
                # Save benchmarks to CSV every 60 seconds
                if current_time - last_benchmark_time >= benchmark_interval:
                    stats = self.timing_stats.get_stats()
                    if stats:
                        self.save_benchmarks_to_csv("stock_terminal_benchmarks.csv", stats)
                    last_benchmark_time = current_time
                
                await asyncio.sleep(0.9)
                
        except KeyboardInterrupt:
            print("\033[?25h")  # Show cursor
            print_status("\nShutting down...")
        finally:
            await self.cleanup()
    
    def save_benchmarks_to_csv(self, filename, stats):
        """Save benchmark results to a CSV file."""
        file_exists = os.path.isfile(filename)
        
        with open(filename, 'a', newline='') as csvfile:
            fieldnames = list(stats.keys())
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            if not file_exists:
                writer.writeheader()
            
            writer.writerow(stats)

    def load_sp500_stocks(self):
        """Load S&P 500 stocks from Wikipedia."""
        try:
            url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
            tables = pd.read_html(url)
            df = tables[0]
            self.all_stocks = df['Symbol'].tolist()
            self.total_stocks = len(self.all_stocks)
            self.current_batch = self.all_stocks[:self.batch_size]
            self.analyzer.symbols = self.current_batch
        except Exception:
            self.all_stocks = ["AAPL", "GOOGL", "MSFT"]  # Fallback to default stocks
            self.total_stocks = len(self.all_stocks)
            self.current_batch = self.all_stocks
            self.analyzer.symbols = self.current_batch
    
    def get_next_batch(self):
        """Get the next batch of stocks to process."""
        processed_count = len(self.processed_stocks)
        if processed_count >= self.total_stocks:
            return []
        
        start_idx = processed_count
        end_idx = min(start_idx + self.batch_size, self.total_stocks)
        return self.all_stocks[start_idx:end_idx]
    
    async def cleanup(self):
        """Cleanup resources."""
        self.thread_pool.shutdown()

async def main():
    with SuppressOutput():
        terminal = StockTerminal(batch_size=10)
    await terminal.run()

if __name__ == "__main__":
    asyncio.run(main()) 