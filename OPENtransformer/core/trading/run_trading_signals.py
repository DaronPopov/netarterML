#!/usr/bin/env python
"""
Cryptocurrency Trading Signal Generator - Main Runner

This script combines all the functionality from our other scripts to provide
a complete trading signal solution. It fetches cryptocurrency data, analyzes it,
generates trading signals, and outputs them in a format suitable for trading bots.
"""

import os
import sys
import json
import logging
import argparse
import time
from datetime import datetime
import numpy as np
import pickle

# Add the current directory to the path to ensure imports work correctly
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Import our modules
from crypto_trading_signals import CryptoDataFetcher, SignalGenerator
from finlib_extensions import ExtendedStocksAPI
from black_scholes import compute_black_scholes_var

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s.%(msecs)03d - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("trading_signals")

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
    parser = argparse.ArgumentParser(description="Cryptocurrency Trading Signal Generator")
    
    parser.add_argument("--num-coins", type=int, default=3,
                        help="Number of top cryptocurrencies to analyze (default: 3)")
    
    parser.add_argument("--days", type=int, default=30,
                        help="Number of days of historical data to fetch (default: 30)")
    
    parser.add_argument("--output", type=str, default="crypto_signals.json",
                        help="Output JSON file path (default: crypto_signals.json)")
    
    parser.add_argument("--confidence", type=float, default=0.95,
                        help="Confidence level for risk metrics (default: 0.95)")
    
    parser.add_argument("--simulations", type=int, default=10000,
                        help="Number of Monte Carlo simulations for Black-Scholes VaR (default: 10000)")
    
    parser.add_argument("--specific-coins", type=str, nargs="+",
                        help="Specific cryptocurrencies to analyze (e.g., bitcoin ethereum)")
    
    parser.add_argument("--max-retries", type=int, default=5,
                        help="Maximum number of retries for API calls")
    
    parser.add_argument("--retry-delay", type=int, default=5,
                        help="Delay in seconds between retry attempts (reduced from 120)")
    
    parser.add_argument("--test", action="store_true",
                        help="Run in test mode with synthetic data")
    
    parser.add_argument("--force-refresh", action="store_true",
                        help="Force refresh of historical data (ignore cache)")
    
    parser.add_argument("--live", action="store_true",
                        help="Use live data updates instead of full historical data fetch")
    
    parser.add_argument("--cache-dir", type=str, default="./data_cache",
                        help="Directory to store cached data (default: ./data_cache)")
    
    parser.add_argument("--cache-ttl", type=int, default=24,
                        help="Cache time-to-live in hours (default: 24)")
    
    return parser.parse_args()

def run_trading_signals(args):
    """
    Run the trading signal generation process
    
    Args:
        args: Command line arguments
    """
    start_time = time.time()
    logger.info(f"Starting trading signal generation at {datetime.now().isoformat()}")
    
    # Initialize the data fetcher and signal generator
    with Timer("Initializing data fetcher and signal generator"):
        data_fetcher = CryptoDataFetcher(cache_dir=args.cache_dir)
        signal_generator = SignalGenerator()
    
    # Test mode with synthetic data
    if args.test:
        logger.info("Running in test mode with synthetic data")
        
        with Timer("Creating synthetic data"):
            # Create synthetic coins
            coins = ["test_coin_1", "test_coin_2"]
            logger.info(f"Using synthetic coins: {', '.join(coins)}")
            
            # Create synthetic price data
            days = 100
            assets = len(coins)
            prices = np.random.rand(days, assets) * 1000 + 100  # Random prices between 100 and 1100
            prices = np.sort(prices, axis=0)  # Sort to make them generally increasing
            prices = prices.astype(np.float32)
            
            logger.info(f"Created synthetic price data with shape: {prices.shape}")
        
        # Generate signals
        logger.info("Generating trading signals from synthetic data...")
        with Timer("Generating signals from synthetic data"):
            signals = signal_generator.generate_signals(prices, coins)
        
    else:
        # Get the cryptocurrencies to analyze
        with Timer("Getting cryptocurrencies to analyze"):
            if args.specific_coins:
                coins = args.specific_coins
                # Limit the number of specific coins to avoid rate limiting
                if len(coins) > 3:
                    logger.warning(f"Limiting analysis to the first 3 coins to avoid rate limiting")
                    coins = coins[:3]
                logger.info(f"Analyzing specific coins: {', '.join(coins)}")
            else:
                # Get the top N cryptocurrencies
                logger.info(f"Fetching top {args.num_coins} cryptocurrencies...")
                
                # Retry logic for fetching top coins
                retries = 0
                coins = []
                while retries < args.max_retries and not coins:
                    with Timer(f"Fetching top coins (attempt {retries+1})"):
                        coins = data_fetcher.get_top_coins(limit=args.num_coins)
                    if not coins:
                        retries += 1
                        if retries < args.max_retries:
                            logger.warning(f"Failed to fetch top coins. Retrying in {args.retry_delay} seconds... (Attempt {retries+1}/{args.max_retries})")
                            time.sleep(args.retry_delay)
                
                if not coins:
                    logger.error("Failed to fetch top coins after multiple attempts. Exiting.")
                    return
                
                logger.info(f"Top coins: {', '.join(coins)}")
        
        # Get historical data for the coins
        historical_data = {}
        
        if args.live:
            # Live mode - use cached data and update with latest prices
            logger.info("Running in live mode - using cached data with latest updates")
            
            # First, check if we have cached data for these coins
            with Timer("Checking cached data"):
                cached_data = data_fetcher.get_historical_data(coins, days=args.days, force_refresh=False)
            
            if not cached_data or len(cached_data) < len(coins):
                missing_coins = set(coins) - set(cached_data.keys())
                logger.warning(f"Missing cached data for {len(missing_coins)} coins: {', '.join(missing_coins)}")
                
                # Fetch historical data for missing coins
                if missing_coins:
                    logger.info(f"Fetching historical data for missing coins: {', '.join(missing_coins)}")
                    with Timer(f"Fetching historical data for {len(missing_coins)} missing coins"):
                        missing_data = data_fetcher.get_historical_data(list(missing_coins), days=args.days, force_refresh=True)
                    cached_data.update(missing_data)
            
            # Update with latest data
            logger.info("Updating historical data with latest prices...")
            with Timer("Updating historical data with latest prices"):
                historical_data = data_fetcher.update_historical_data(coins)
        else:
            # Regular mode - fetch or use cached historical data
            logger.info(f"Fetching historical data for the past {args.days} days...")
            
            # Retry logic for fetching historical data
            retries = 0
            while retries < args.max_retries and not historical_data:
                with Timer(f"Fetching historical data (attempt {retries+1})"):
                    historical_data = data_fetcher.get_historical_data(coins, days=args.days, force_refresh=args.force_refresh)
                if not historical_data:
                    retries += 1
                    if retries < args.max_retries:
                        logger.warning(f"Failed to fetch historical data. Retrying in {args.retry_delay} seconds... (Attempt {retries+1}/{args.max_retries})")
                        time.sleep(args.retry_delay)
        
        if not historical_data:
            logger.error("Failed to fetch historical data after multiple attempts. Exiting.")
            return
        
        # Log the historical data we received
        logger.info(f"Received historical data for {len(historical_data)} coins:")
        for coin_id, prices in historical_data.items():
            logger.info(f"  {coin_id}: {len(prices)} data points")
        
        # Check which coins we actually have data for
        with Timer("Validating available coins"):
            available_coins = list(historical_data.keys())
            if len(available_coins) < len(coins):
                missing_coins = set(coins) - set(available_coins)
                logger.warning(f"Missing data for {len(missing_coins)} coins: {', '.join(missing_coins)}")
                logger.info(f"Proceeding with analysis for {len(available_coins)} coins: {', '.join(available_coins)}")
                
                # Update the coins list to only include those with data
                coins = available_coins
            
            if not coins:
                logger.error("No coins with data available for analysis. Exiting.")
                return
        
        # Format data for analysis
        with Timer("Formatting data for analysis"):
            prices = data_fetcher.format_data_for_analysis(historical_data)
        
        # Check if we have valid price data
        if prices.size == 0 or prices.shape[1] == 0:
            logger.error("No valid price data available for analysis. Exiting.")
            return
        
        logger.info(f"Formatted price data shape: {prices.shape}")
        
        # Generate signals
        logger.info("Generating trading signals...")
        with Timer(f"Generating trading signals for {len(coins)} coins"):
            signals = signal_generator.generate_signals(prices, coins)
    
    # Check if we got any signals
    if not signals:
        logger.error("No trading signals were generated. Check the logs for errors.")
        
        # Try to diagnose the issue
        logger.info("Diagnostic information:")
        logger.info(f"  Price data shape: {prices.shape}")
        logger.info(f"  Coins: {coins}")
        
        # Try to compute some basic metrics to see if the data is valid
        try:
            with Timer("Computing diagnostic metrics"):
                # Calculate simple moving average as a test
                window = min(5, prices.shape[0])
                sma = np.mean(prices[-window:], axis=0)
                logger.info(f"  Simple {window}-day moving average: {sma}")
                
                # Calculate daily returns as a test
                returns = np.zeros((prices.shape[0] - 1, prices.shape[1]), dtype=np.float32)
                for i in range(1, prices.shape[0]):
                    returns[i-1] = (prices[i] / prices[i-1]) - 1
                logger.info(f"  Average daily returns: {np.mean(returns, axis=0)}")
        except Exception as e:
            logger.error(f"  Error computing diagnostic metrics: {e}")
    
    # Compute Black-Scholes VaR separately (with more simulations)
    if not args.test:  # Skip in test mode
        logger.info(f"Computing Black-Scholes VaR with {args.simulations} simulations...")
        try:
            with Timer(f"Computing Black-Scholes VaR for {len(coins)} coins"):
                # Use cached data for Black-Scholes VaR if available
                bs_var_cache_file = os.path.join(args.cache_dir, "bs_var_results.pkl")
                bs_var_results = {}
                
                # Check if we have cached BS VaR results that are less than cache_ttl hours old
                if os.path.exists(bs_var_cache_file):
                    try:
                        with open(bs_var_cache_file, "rb") as f:
                            cached_data = pickle.load(f)
                        
                        if "timestamp" in cached_data and "results" in cached_data:
                            cache_age = (datetime.now() - cached_data["timestamp"]).total_seconds() / 3600
                            
                            if cache_age < args.cache_ttl:  # Use cache if less than cache_ttl hours old
                                bs_var_results = cached_data["results"]
                                logger.info(f"Using cached Black-Scholes VaR results ({cache_age:.1f} hours old)")
                    except Exception as e:
                        logger.error(f"Error loading cached BS VaR results: {e}")
                
                # Compute BS VaR for coins not in cache
                coins_to_compute = [coin for coin in coins if coin not in bs_var_results]
                
                if coins_to_compute:
                    logger.info(f"Computing Black-Scholes VaR for {len(coins_to_compute)} coins")
                    with Timer(f"Computing Black-Scholes VaR for {len(coins_to_compute)} new coins"):
                        new_results = compute_black_scholes_var(coins_to_compute, num_simulations=args.simulations)
                    bs_var_results.update(new_results)
                    
                    # Save updated cache
                    try:
                        with open(bs_var_cache_file, "wb") as f:
                            pickle.dump({"timestamp": datetime.now(), "results": bs_var_results}, f)
                        logger.info("Saved Black-Scholes VaR results to cache")
                    except Exception as e:
                        logger.error(f"Error saving BS VaR cache: {e}")
                
                # Add Black-Scholes VaR results to signals
                for coin_id, var in bs_var_results.items():
                    if coin_id in signals:
                        signals[coin_id]["bs_var_95_detailed"] = float(var)
                        
                        # Update risk signal based on detailed BS VaR
                        risk_signal = "SELL" if var > 0.05 else "BUY" if var < 0.02 else "HOLD"
                        signals[coin_id]["risk_signal_detailed"] = risk_signal
        except Exception as e:
            logger.error(f"Error computing detailed Black-Scholes VaR: {e}")
    
    # Print signals
    if signals:
        logger.info("Trading signals generated:")
        for coin_id, signal_data in signals.items():
            logger.info(f"{coin_id.upper()}: {signal_data['final_signal']} (Confidence: {signal_data['confidence']:.2f})")
            logger.info(f"  Price: ${signal_data['price']:.2f}")
            logger.info(f"  RSI: {signal_data['rsi']:.2f} ({signal_data['rsi_signal']})")
            logger.info(f"  MACD: {signal_data['macd']:.6f} vs Signal: {signal_data['signal_line']:.6f} ({signal_data['macd_signal']})")
            logger.info(f"  Bollinger Bands: {signal_data['lower_band']:.2f} < {signal_data['middle_band']:.2f} < {signal_data['upper_band']:.2f} ({signal_data['bb_signal']})")
            logger.info(f"  VaR (95%): {signal_data['var_95']:.2f}%")
            logger.info(f"  ES (95%): {signal_data['es_95']:.2f}%")
            
            if "bs_var_95" in signal_data:
                logger.info(f"  Black-Scholes VaR (95%): {signal_data['bs_var_95']:.2f}%")
            
            if "bs_var_95_detailed" in signal_data:
                logger.info(f"  Detailed BS VaR (95%): {signal_data['bs_var_95_detailed']:.2f}%")
                logger.info(f"  Detailed Risk Signal: {signal_data['risk_signal_detailed']}")
                
            logger.info("")
    else:
        logger.warning("No trading signals were generated.")
    
    # Output JSON for potential bot consumption
    with Timer("Saving output to JSON"):
        output = {
            "timestamp": datetime.now().isoformat(),
            "signals": signals,
            "metadata": {
                "num_coins": len(coins),
                "days_analyzed": args.days if not args.test else days,
                "confidence_level": args.confidence,
                "simulations": args.simulations,
                "test_mode": args.test,
                "live_mode": args.live,
                "force_refresh": args.force_refresh,
                "cache_ttl": args.cache_ttl,
                "execution_time_ms": int((time.time() - start_time) * 1000)
            }
        }
        
        with open(args.output, "w") as f:
            json.dump(output, f, indent=2)
        
        logger.info(f"Signals saved to {args.output}")
    
    # Log total execution time
    end_time = time.time()
    total_time = end_time - start_time
    logger.info(f"TIMING: Total execution time: {total_time:.3f}s ({int(total_time * 1000)}ms)")

def main():
    """
    Main function
    """
    start_time = time.time()
    
    with Timer("Parsing arguments"):
        args = parse_arguments()
    
    run_trading_signals(args)
    
    end_time = time.time()
    total_time = end_time - start_time
    logger.info(f"TIMING: Total program execution time: {total_time:.3f}s ({int(total_time * 1000)}ms)")

if __name__ == "__main__":
    main() 