#!/usr/bin/env python
"""
Simple test script for the ExtendedStocksAPI
"""

import os
import sys
import numpy as np
import logging

# Add the current directory to the path to ensure imports work correctly
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from finlib_extensions import ExtendedStocksAPI

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("test_api")

def main():
    """
    Main function to test the ExtendedStocksAPI
    """
    logger.info("Initializing ExtendedStocksAPI...")
    api = ExtendedStocksAPI()
    
    # Create some test data
    logger.info("Creating test data...")
    days = 100
    assets = 2
    prices = np.random.rand(days, assets) * 1000 + 100  # Random prices between 100 and 1100
    prices = np.sort(prices, axis=0)  # Sort to make them generally increasing
    prices = prices.astype(np.float32)
    
    logger.info(f"Test data shape: {prices.shape}")
    logger.info(f"First few prices: {prices[:5, 0]}")
    
    # Calculate daily returns
    logger.info("Calculating daily returns...")
    returns = api.compute_daily_returns(prices)
    logger.info(f"Returns shape: {returns.shape}")
    logger.info(f"First few returns: {returns[:5, 0]}")
    
    # Calculate RSI
    logger.info("Calculating RSI...")
    rsi = api.relative_strength_index(prices, window=14)
    logger.info(f"RSI shape: {rsi.shape}")
    logger.info(f"Last few RSI values: {rsi[-5:, 0]}")
    
    # Calculate MACD
    logger.info("Calculating MACD...")
    macd_line, signal_line, histogram = api.macd(prices, fast_period=12, slow_period=26, signal_period=9)
    logger.info(f"MACD shape: {macd_line.shape}")
    logger.info(f"Last few MACD values: {macd_line[-5:, 0]}")
    
    # Calculate Bollinger Bands
    logger.info("Calculating Bollinger Bands...")
    upper_band, middle_band, lower_band = api.bollinger_bands(prices, window=20, num_std=2)
    logger.info(f"Bollinger Bands shape: {upper_band.shape}")
    logger.info(f"Last few upper band values: {upper_band[-5:, 0]}")
    
    # Calculate VaR and ES
    logger.info("Calculating VaR and ES...")
    var, es = api.calculate_var_and_es(returns, confidence_level=0.95)
    logger.info(f"VaR: {var}")
    logger.info(f"ES: {es}")
    
    # Calculate Black-Scholes VaR
    logger.info("Calculating Black-Scholes VaR...")
    bs_var = api.calculate_black_scholes_var(returns, confidence_level=0.95, num_simulations=10000)
    logger.info(f"Black-Scholes VaR: {bs_var}")
    
    logger.info("All tests completed successfully!")

if __name__ == "__main__":
    main() 