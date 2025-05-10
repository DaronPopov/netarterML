#!/usr/bin/env python
"""
Example script demonstrating how to use the Alpaca paper trading integration.

This script shows how to set up and run the Alpaca paper trader with different
configurations for testing your cryptocurrency trading signals.
"""

import os
import sys
import logging
from datetime import datetime

# Import our Alpaca paper trader
from alpaca_paper_trader import AlpacaPaperTrader

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("example_alpaca_trading")

def example_1_basic_usage():
    """
    Example 1: Basic usage with environment variables
    
    Make sure to set ALPACA_API_KEY and ALPACA_API_SECRET environment variables before running.
    """
    logger.info("Example 1: Basic usage with environment variables")
    
    # Check if environment variables are set
    if not os.environ.get("ALPACA_API_KEY") or not os.environ.get("ALPACA_API_SECRET"):
        logger.error("ALPACA_API_KEY and ALPACA_API_SECRET environment variables must be set")
        logger.info("You can set them with:")
        logger.info("export ALPACA_API_KEY='your_api_key'")
        logger.info("export ALPACA_API_SECRET='your_api_secret'")
        return
    
    # Initialize the paper trader
    paper_trader = AlpacaPaperTrader()
    
    # Run a single trading cycle with default settings
    paper_trader.run_trading_cycle()
    
    logger.info("Example 1 completed")

def example_2_specific_symbols():
    """
    Example 2: Trading specific cryptocurrencies
    
    Make sure to set ALPACA_API_KEY and ALPACA_API_SECRET environment variables before running.
    """
    logger.info("Example 2: Trading specific cryptocurrencies")
    
    # Check if environment variables are set
    if not os.environ.get("ALPACA_API_KEY") or not os.environ.get("ALPACA_API_SECRET"):
        logger.error("ALPACA_API_KEY and ALPACA_API_SECRET environment variables must be set")
        return
    
    # Initialize the paper trader
    paper_trader = AlpacaPaperTrader()
    
    # Run a trading cycle with specific symbols
    paper_trader.run_trading_cycle(
        symbols_to_trade=["BTCUSD", "ETHUSD", "SOLUSD"],
        max_position_value=2000.0,
        max_positions=2
    )
    
    logger.info("Example 2 completed")

def example_3_direct_api_keys():
    """
    Example 3: Using direct API keys instead of environment variables
    
    Replace 'your_api_key' and 'your_api_secret' with your actual Alpaca paper trading API keys.
    """
    logger.info("Example 3: Using direct API keys")
    
    # Replace these with your actual API keys
    api_key = "your_api_key"
    api_secret = "your_api_secret"
    
    # Check if API keys are provided
    if api_key == "your_api_key" or api_secret == "your_api_secret":
        logger.error("Please replace the placeholder API keys with your actual Alpaca paper trading API keys")
        return
    
    # Initialize the paper trader with direct API keys
    paper_trader = AlpacaPaperTrader(
        api_key=api_key,
        api_secret=api_secret
    )
    
    # Run a trading cycle with custom settings
    paper_trader.run_trading_cycle(
        max_position_value=1500.0,
        max_positions=3
    )
    
    logger.info("Example 3 completed")

def example_4_manual_steps():
    """
    Example 4: Manual control of the trading process
    
    This example shows how to manually control each step of the trading process.
    """
    logger.info("Example 4: Manual control of the trading process")
    
    # Check if environment variables are set
    if not os.environ.get("ALPACA_API_KEY") or not os.environ.get("ALPACA_API_SECRET"):
        logger.error("ALPACA_API_KEY and ALPACA_API_SECRET environment variables must be set")
        return
    
    # Initialize the paper trader
    paper_trader = AlpacaPaperTrader()
    
    # 1. Get historical data for specific symbols
    symbols = ["BTCUSD", "ETHUSD"]
    historical_data = paper_trader.get_historical_data(symbols)
    
    if not historical_data:
        logger.error("Failed to get historical data")
        return
    
    # 2. Format data for analysis
    formatted_data = paper_trader.format_data_for_analysis(historical_data)
    
    if not formatted_data:
        logger.error("Failed to format data for analysis")
        return
    
    # 3. Generate signals
    signals = paper_trader.generate_signals(formatted_data)
    
    if not signals:
        logger.error("Failed to generate signals")
        return
    
    # 4. Print signals
    logger.info("Trading signals generated:")
    for coin_id, signal_data in signals.items():
        logger.info(f"{coin_id.upper()}: {signal_data['final_signal']} (Confidence: {signal_data['confidence']:.2f})")
    
    # 5. Execute trades based on signals
    paper_trader.execute_trades(signals, max_position_value=1000.0, max_positions=2)
    
    logger.info("Example 4 completed")

def main():
    """
    Main function to run the examples
    """
    logger.info(f"Starting Alpaca paper trading examples at {datetime.now().isoformat()}")
    
    # Uncomment the example you want to run
    example_1_basic_usage()
    # example_2_specific_symbols()
    # example_3_direct_api_keys()
    # example_4_manual_steps()
    
    logger.info("All examples completed")

if __name__ == "__main__":
    main() 