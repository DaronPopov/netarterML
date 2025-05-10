#!/usr/bin/env python
"""
Live Cryptocurrency Trading Signal Generator

This script runs the trading signal generator in live mode, using cached historical data
and only fetching the latest price updates. It's designed to be run periodically
(e.g., via a cron job) to generate up-to-date trading signals without hitting API rate limits.
"""

import os
import sys
import time
import logging
import argparse
import shutil
from datetime import datetime, timedelta

# Add the current directory to the path to ensure imports work correctly
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Import our modules
from run_trading_signals import run_trading_signals

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s.%(msecs)03d - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler("live_signals.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("live_signals")

class Timer:
    """
    Simple timer class to measure execution time with millisecond precision
    """
    def __init__(self, name):
        self.name = name
        self.start_time = None
        
    def __enter__(self):
        self.start_time = time.time()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        end_time = time.time()
        elapsed_time = end_time - self.start_time
        logger.info(f"TIMING: {self.name} completed in {elapsed_time:.3f}s ({int(elapsed_time * 1000)}ms)")

def parse_arguments():
    """
    Parse command line arguments
    
    Returns:
        Parsed arguments
    """
    with Timer("Parsing arguments"):
        parser = argparse.ArgumentParser(description="Live Cryptocurrency Trading Signal Generator")
        
        parser.add_argument("--num-coins", type=int, default=5,
                            help="Number of top cryptocurrencies to analyze (default: 5)")
        
        parser.add_argument("--output", type=str, default="live_signals.json",
                            help="Output JSON file path (default: live_signals.json)")
        
        parser.add_argument("--specific-coins", type=str, nargs="+",
                            help="Specific cryptocurrencies to analyze (e.g., bitcoin ethereum)")
        
        parser.add_argument("--cache-dir", type=str, default="./data_cache",
                            help="Directory to store cached data (default: ./data_cache)")
        
        parser.add_argument("--interval", type=int, default=0,
                            help="Interval in minutes to run the signal generator (0 for once)")
        
        parser.add_argument("--cache-ttl", type=int, default=24,
                            help="Cache time-to-live in hours (default: 24)")
        
        parser.add_argument("--simulations", type=int, default=10000,
                            help="Number of Monte Carlo simulations for Black-Scholes VaR (default: 10000)")
        
        parser.add_argument("--clear-cache", action="store_true",
                            help="Force clearing the cache before running")
        
        return parser.parse_args()

class RunArgs:
    """
    Class to hold parameters for run_trading_signals function
    """
    def __init__(self, args):
        self.num_coins = args.num_coins
        self.days = 365  # Increased from 180 to 365 to ensure enough data for all indicators
        self.output = args.output
        self.confidence = 0.95
        self.simulations = args.simulations  # Use the value from command line arguments
        self.specific_coins = args.specific_coins
        self.max_retries = 3
        self.retry_delay = 5
        self.test = False
        self.force_refresh = True  # Force refresh to get more historical data
        self.live = True
        self.cache_dir = args.cache_dir
        self.cache_ttl = args.cache_ttl  # Use the value from command line arguments

def clear_cache(cache_dir):
    """
    Clear the cache directory to force fresh data fetch
    
    Args:
        cache_dir: Path to the cache directory
    """
    if os.path.exists(cache_dir):
        logger.info(f"Clearing cache directory: {cache_dir}")
        try:
            for file in os.listdir(cache_dir):
                file_path = os.path.join(cache_dir, file)
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            logger.info("Cache cleared successfully")
        except Exception as e:
            logger.error(f"Error clearing cache: {e}")
    else:
        logger.info(f"Cache directory does not exist: {cache_dir}")
        os.makedirs(cache_dir, exist_ok=True)
        logger.info(f"Created cache directory: {cache_dir}")

def main():
    """
    Main function to run the live trading signal generator
    """
    start_time = time.time()
    logger.info(f"Starting live trading signal generator at {datetime.now().isoformat()}")
    
    with Timer("Setting up arguments"):
        args = parse_arguments()
        
        run_args = RunArgs(args)
    
    # Clear the cache if force_refresh is enabled or --clear-cache is specified
    if run_args.force_refresh or args.clear_cache:
        clear_cache(run_args.cache_dir)
    
    if args.interval > 0:
        # Ensure interval is not too frequent to avoid rate limiting
        if args.interval < 1:
            logger.warning(f"Interval of {args.interval} minutes is too frequent and may cause rate limiting.")
            logger.warning(f"Setting interval to 1 minute to avoid API rate limits.")
            args.interval = 1
            
        logger.info(f"Running live signal generator every {args.interval} minutes")
        
        run_count = 0
        while True:
            run_count += 1
            iteration_start_time = time.time()
            
            try:
                logger.info(f"Running signal generator at {datetime.now().isoformat()} (iteration {run_count})")
                with Timer(f"Signal generation iteration {run_count}"):
                    run_trading_signals(run_args)
                logger.info(f"Signal generation completed successfully")
            except Exception as e:
                logger.error(f"Error running signal generator: {e}")
            
            # Calculate time to sleep
            elapsed_time = time.time() - iteration_start_time
            sleep_time = max(0, args.interval * 60 - elapsed_time)
            
            if sleep_time > 0:
                logger.info(f"Sleeping for {sleep_time:.1f} seconds until next run (iteration {run_count+1})")
                time.sleep(sleep_time)
            else:
                logger.warning(f"Signal generation took longer than interval ({elapsed_time:.1f}s > {args.interval*60}s)")
                logger.warning(f"Starting next iteration immediately")
    else:
        # Run once
        logger.info("Running live signal generator once")
        try:
            with Timer("Single run of signal generator"):
                run_trading_signals(run_args)
            logger.info("Signal generation completed successfully")
        except Exception as e:
            logger.error(f"Error running signal generator: {e}")
    
    # Log total execution time
    end_time = time.time()
    total_time = end_time - start_time
    logger.info(f"TIMING: Total execution time: {total_time:.3f}s ({int(total_time * 1000)}ms)")

if __name__ == "__main__":
    main() 