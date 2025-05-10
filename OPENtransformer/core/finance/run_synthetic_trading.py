#!/usr/bin/env python3
import argparse
import logging
import time
import sys
import os
from typing import List, Dict, Any
import datetime
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Set specific loggers to DEBUG level
logging.getLogger('finlib.finance.stocks_api').setLevel(logging.DEBUG)
logging.getLogger('finlib.finance.synthetic_data_provider').setLevel(logging.DEBUG)

logger = logging.getLogger("synthetic_trading")

# Add parent directory to path to allow imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import our modules
from finlib.finance.trading_strategy import TradingStrategy
from finlib.finance.synthetic_data_provider import fetch_realtime_data, fetch_historical_data, preload_historical_data

class SyntheticTradingRunner:
    """
    Runner for trading strategy using synthetic data.
    """
    
    def __init__(self, assets: List[str], iterations: int = 100, sleep_time: float = 0.0):
        """
        Initialize the synthetic trading runner.
        
        Args:
            assets: List of asset symbols to trade
            iterations: Number of iterations to run
            sleep_time: Time to sleep between iterations (seconds)
        """
        self.assets = [asset.upper() for asset in assets]
        self.iterations = iterations
        self.sleep_time = sleep_time
        self.strategy = TradingStrategy(self.assets)
        
        # Performance metrics
        self.execution_times = []
        self.analysis_times = []
        self.data_fetch_times = []
        
        logger.info(f"Initialized synthetic trading runner with assets: {self.assets}")
        logger.info(f"Will run for {self.iterations} iterations with {self.sleep_time}s sleep between iterations")
    
    def preload_data(self) -> None:
        """
        Preload historical data for all assets.
        """
        logger.info("Preloading historical data for all assets...")
        start_time = time.time()
        
        # Preload data for all assets
        results = preload_historical_data(self.assets, limit=1000)
        
        # Check results
        success_count = sum(results.values())
        if success_count < len(self.assets):
            logger.warning(f"Failed to preload data for {len(self.assets) - success_count} assets")
        
        elapsed = time.time() - start_time
        logger.info(f"Preloaded historical data in {elapsed:.4f} seconds")
    
    def fetch_latest_prices(self) -> Dict[str, float]:
        """
        Fetch the latest prices for all assets.
        
        Returns:
            dict: Dictionary mapping asset symbols to prices
        """
        start_time = time.time()
        
        prices = {}
        for asset in self.assets:
            try:
                data = fetch_realtime_data(asset)
                prices[asset] = data["rate"]
            except Exception as e:
                logger.error(f"Error fetching price for {asset}: {e}")
                prices[asset] = None
        
        elapsed = time.time() - start_time
        self.data_fetch_times.append(elapsed)
        
        return prices
    
    def run_iteration(self, iteration: int) -> Dict[str, Any]:
        """
        Run a single iteration of the trading strategy.
        
        Args:
            iteration: The iteration number
            
        Returns:
            dict: Results of the iteration
        """
        logger.info(f"Running iteration {iteration}/{self.iterations}")
        
        # Fetch latest prices
        prices = self.fetch_latest_prices()
        
        # Run the strategy
        start_time = time.time()
        trades = self.strategy.run(prices)
        elapsed = time.time() - start_time
        
        # Record execution time
        self.execution_times.append(elapsed)
        
        # Extract analysis time from strategy
        if hasattr(self.strategy, 'last_analysis_time'):
            self.analysis_times.append(self.strategy.last_analysis_time)
        
        # Log results
        logger.info(f"Iteration {iteration} completed in {elapsed:.4f} seconds")
        if trades:
            for trade in trades:
                logger.info(f"Trade: {trade}")
        else:
            logger.info("No trades generated")
        
        return {
            "iteration": iteration,
            "execution_time": elapsed,
            "trades": trades,
            "prices": prices
        }
    
    def run(self) -> Dict[str, Any]:
        """
        Run the trading strategy for the specified number of iterations.
        
        Returns:
            dict: Results of the run
        """
        logger.info(f"Starting synthetic trading run with {len(self.assets)} assets")
        
        # Preload data
        self.preload_data()
        
        # Initialize the strategy
        logger.info("Initializing trading strategy...")
        self.strategy.initialize()
        
        # Run iterations
        results = []
        total_start_time = time.time()
        
        for i in range(1, self.iterations + 1):
            result = self.run_iteration(i)
            results.append(result)
            
            # Sleep between iterations if specified
            if i < self.iterations and self.sleep_time > 0:
                time.sleep(self.sleep_time)
        
        total_elapsed = time.time() - total_start_time
        
        # Calculate statistics
        avg_execution_time = np.mean(self.execution_times)
        max_execution_time = np.max(self.execution_times)
        min_execution_time = np.min(self.execution_times)
        
        avg_analysis_time = np.mean(self.analysis_times) if self.analysis_times else 0
        avg_data_fetch_time = np.mean(self.data_fetch_times)
        
        # Log summary
        logger.info(f"Completed {self.iterations} iterations in {total_elapsed:.4f} seconds")
        logger.info(f"Average execution time: {avg_execution_time:.4f} seconds")
        logger.info(f"Min/Max execution time: {min_execution_time:.4f}/{max_execution_time:.4f} seconds")
        logger.info(f"Average analysis time: {avg_analysis_time:.4f} seconds")
        logger.info(f"Average data fetch time: {avg_data_fetch_time:.4f} seconds")
        
        # Calculate theoretical max throughput
        if avg_execution_time > 0:
            theoretical_throughput = 1.0 / avg_execution_time
            logger.info(f"Theoretical max throughput: {theoretical_throughput:.2f} iterations/second")
        else:
            logger.info("Theoretical max throughput: N/A (execution time too small to measure)")
        
        # Return summary
        return {
            "total_time": total_elapsed,
            "iterations": self.iterations,
            "avg_execution_time": avg_execution_time,
            "max_execution_time": max_execution_time,
            "min_execution_time": min_execution_time,
            "avg_analysis_time": avg_analysis_time,
            "avg_data_fetch_time": avg_data_fetch_time,
            "results": results
        }

def parse_args():
    """
    Parse command line arguments.
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(description="Run trading strategy with synthetic data")
    
    parser.add_argument("--assets", type=str, nargs="+", default=["BTC"],
                        help="Asset symbols to trade (default: BTC)")
    
    parser.add_argument("--iterations", type=int, default=100,
                        help="Number of iterations to run (default: 100)")
    
    parser.add_argument("--sleep", type=float, default=0.0,
                        help="Time to sleep between iterations in seconds (default: 0.0)")
    
    parser.add_argument("--verbose", action="store_true",
                        help="Enable verbose logging")
    
    return parser.parse_args()

def main():
    """
    Main entry point.
    """
    args = parse_args()
    
    # Set log level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Create and run the synthetic trading runner
    runner = SyntheticTradingRunner(
        assets=args.assets,
        iterations=args.iterations,
        sleep_time=args.sleep
    )
    
    # Run the strategy
    results = runner.run()
    
    # Calculate theoretical max throughput
    if results['avg_execution_time'] > 0:
        theoretical_throughput = 1.0 / results['avg_execution_time']
    else:
        theoretical_throughput = float('inf')
    
    # Print summary
    print("\n" + "=" * 50)
    print("SYNTHETIC TRADING BENCHMARK SUMMARY")
    print("=" * 50)
    print(f"Assets: {args.assets}")
    print(f"Iterations: {args.iterations}")
    print(f"Total time: {results['total_time']:.4f} seconds")
    print(f"Average execution time: {results['avg_execution_time']:.4f} seconds")
    print(f"Min/Max execution time: {results['min_execution_time']:.4f}/{results['max_execution_time']:.4f} seconds")
    print(f"Average analysis time: {results['avg_analysis_time']:.4f} seconds")
    print(f"Average data fetch time: {results['avg_data_fetch_time']:.4f} seconds")
    
    if theoretical_throughput == float('inf'):
        print(f"Theoretical max throughput: N/A (execution time too small to measure)")
    else:
        print(f"Theoretical max throughput: {theoretical_throughput:.2f} iterations/second")
    
    print("=" * 50)

if __name__ == "__main__":
    main() 