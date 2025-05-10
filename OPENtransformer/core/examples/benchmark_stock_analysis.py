import asyncio
import yfinance as yf
import time
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from stock_trend_analyzer import StockTrendAnalyzer
from concurrent.futures import ThreadPoolExecutor
import os
from typing import List, Dict, Tuple
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

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

class BenchmarkStockAnalysis:
    def __init__(self, top_n: int = 50):
        self.top_n = top_n
        self.thread_pool = ThreadPoolExecutor(max_workers=10)
        self.yf_limiter = RateLimiter(max_requests_per_min=300)
        self.analyzer = StockTrendAnalyzer(
            symbols=[],
            update_interval=0.1,
            data_buffer_size=1000
        )
        self.results = []
        self.timing_stats = {
            'data_fetching': [],
            'ai_inference': [],
            'total_processing': []
        }

    async def get_top_stocks(self) -> List[str]:
        """Get the top N stocks by market cap."""
        try:
            # Get S&P 500 stocks
            url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
            tables = pd.read_html(url)
            df = tables[0]
            symbols = df['Symbol'].tolist()
            
            # Get market caps for all symbols
            market_caps = {}
            for symbol in symbols:
                try:
                    await self.yf_limiter.acquire()
                    ticker = yf.Ticker(symbol)
                    info = ticker.info
                    market_cap = info.get('marketCap', 0)
                    market_caps[symbol] = market_cap
                except Exception as e:
                    logger.warning(f"Error getting market cap for {symbol}: {e}")
                    market_caps[symbol] = 0
            
            # Sort by market cap and get top N
            sorted_symbols = sorted(market_caps.items(), key=lambda x: x[1], reverse=True)
            top_symbols = [symbol for symbol, _ in sorted_symbols[:self.top_n]]
            
            logger.info(f"Selected top {len(top_symbols)} stocks by market cap")
            return top_symbols
            
        except Exception as e:
            logger.error(f"Error getting top stocks: {e}")
            # Fallback to some major stocks if there's an error
            return ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "JPM", "V", "WMT"]

    async def fetch_stock_data(self, symbol: str) -> Dict:
        """Fetch historical data for a stock using yfinance."""
        try:
            await self.yf_limiter.acquire()
            ticker = yf.Ticker(symbol)
            data = await asyncio.get_event_loop().run_in_executor(
                self.thread_pool,
                lambda: ticker.history(period='30d', interval='1d')
            )
            
            if not data.empty:
                closes = data['Close'].values
                volumes = data['Volume'].values
                daily_returns = pd.Series(closes).pct_change().dropna()
                
                return {
                    'symbol': symbol,
                    'closes': closes,
                    'volumes': volumes,
                    'daily_returns': daily_returns,
                    'last_close': closes[-1]
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {e}")
            return None

    async def process_stock(self, symbol: str) -> Dict:
        """Process a single stock with timing measurements."""
        start_time = time.time()
        
        # Fetch data
        data_fetch_start = time.time()
        stock_data = await self.fetch_stock_data(symbol)
        data_fetch_time = time.time() - data_fetch_start
        
        if not stock_data:
            return None
        
        # AI inference
        inference_start = time.time()
        analysis = await self.analyzer.get_trend_analysis(symbol)
        inference_time = time.time() - inference_start
        
        # Calculate metrics
        total_time = time.time() - start_time
        
        # Store timing stats
        self.timing_stats['data_fetching'].append(data_fetch_time)
        self.timing_stats['ai_inference'].append(inference_time)
        self.timing_stats['total_processing'].append(total_time)
        
        return {
            'symbol': symbol,
            'data_fetch_time': data_fetch_time,
            'inference_time': inference_time,
            'total_time': total_time,
            'analysis': analysis,
            'stock_data': stock_data
        }

    async def run_benchmark(self):
        """Run the benchmark on top N stocks."""
        logger.info("Starting benchmark...")
        
        # Get top stocks
        symbols = await self.get_top_stocks()
        self.analyzer.symbols = symbols
        
        # Process stocks in parallel
        chunk_size = 5
        for i in range(0, len(symbols), chunk_size):
            chunk = symbols[i:i + chunk_size]
            chunk_results = await asyncio.gather(*[self.process_stock(symbol) for symbol in chunk])
            self.results.extend([r for r in chunk_results if r is not None])
            
            # Log progress
            logger.info(f"Processed {len(self.results)}/{len(symbols)} stocks")
        
        # Calculate statistics
        stats = {
            'total_stocks': len(self.results),
            'avg_data_fetch_time': np.mean(self.timing_stats['data_fetching']),
            'avg_inference_time': np.mean(self.timing_stats['ai_inference']),
            'avg_total_time': np.mean(self.timing_stats['total_processing']),
            'max_data_fetch_time': max(self.timing_stats['data_fetching']),
            'max_inference_time': max(self.timing_stats['ai_inference']),
            'max_total_time': max(self.timing_stats['total_processing'])
        }
        
        # Print results
        logger.info("\nBenchmark Results:")
        logger.info(f"Total stocks processed: {stats['total_stocks']}")
        logger.info(f"Average data fetch time: {stats['avg_data_fetch_time']:.2f}s")
        logger.info(f"Average AI inference time: {stats['avg_inference_time']:.2f}s")
        logger.info(f"Average total processing time: {stats['avg_total_time']:.2f}s")
        logger.info(f"Max data fetch time: {stats['max_data_fetch_time']:.2f}s")
        logger.info(f"Max AI inference time: {stats['max_inference_time']:.2f}s")
        logger.info(f"Max total processing time: {stats['max_total_time']:.2f}s")
        
        return stats

    async def cleanup(self):
        """Cleanup resources."""
        self.thread_pool.shutdown()

async def main():
    benchmark = BenchmarkStockAnalysis(top_n=50)
    try:
        await benchmark.run_benchmark()
    finally:
        await benchmark.cleanup()

if __name__ == "__main__":
    asyncio.run(main()) 