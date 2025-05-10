#!/usr/bin/env python
"""
Test script for the Cryptocurrency Trading Signal Generator.

This script tests the main components of the trading signal generator
to ensure they work correctly.
"""

import unittest
import numpy as np
import logging
from finlib_extensions import ExtendedStocksAPI
from crypto_trading_signals import CryptoDataFetcher, SignalGenerator

# Disable logging for tests
logging.basicConfig(level=logging.ERROR)

class TestTradingSignals(unittest.TestCase):
    """
    Test cases for the trading signal generator
    """
    
    def setUp(self):
        """
        Set up test fixtures
        """
        self.api = ExtendedStocksAPI()
        self.data_fetcher = CryptoDataFetcher()
        self.signal_generator = SignalGenerator()
        
        # Create sample price data
        days = 100
        assets = 3
        self.sample_prices = np.random.normal(100, 10, (days, assets)).cumsum(axis=0) + 1000
        self.sample_prices = np.abs(self.sample_prices).astype(np.float32)  # Ensure positive prices
        
        # Create sample returns data
        self.sample_returns = np.random.normal(0, 0.02, (days - 1, assets)).astype(np.float32)
        
        # Sample coin IDs
        self.sample_coins = ["bitcoin", "ethereum", "solana"]
    
    def test_compute_daily_returns(self):
        """
        Test the compute_daily_returns function
        """
        returns = self.signal_generator.compute_daily_returns(self.sample_prices)
        
        # Check shape
        self.assertEqual(returns.shape, (self.sample_prices.shape[0] - 1, self.sample_prices.shape[1]))
        
        # Check values (manually calculate for first day)
        expected_return = (self.sample_prices[1, 0] / self.sample_prices[0, 0]) - 1
        self.assertAlmostEqual(returns[0, 0], expected_return, places=5)
    
    def test_calculate_black_scholes_var(self):
        """
        Test the calculate_black_scholes_var function
        """
        var = self.api.calculate_black_scholes_var(self.sample_returns, num_simulations=10000)
        
        # Check shape
        self.assertEqual(var.shape, (self.sample_returns.shape[1],))
        
        # Check values (should be positive)
        self.assertTrue(np.all(var > 0))
    
    def test_generate_signals(self):
        """
        Test the generate_signals function
        """
        # Generate signals
        signals = self.signal_generator.generate_signals(self.sample_prices, self.sample_coins)
        
        # Check that we have signals for all coins
        self.assertEqual(len(signals), len(self.sample_coins))
        
        # Check that each coin has the expected signal keys
        expected_keys = [
            "price", "rsi", "macd", "signal_line", "upper_band", "middle_band", "lower_band",
            "var_95", "es_95", "bs_var_95", "rsi_signal", "macd_signal", "bb_signal",
            "risk_signal", "final_signal", "confidence"
        ]
        
        for coin in self.sample_coins:
            self.assertIn(coin, signals)
            for key in expected_keys:
                self.assertIn(key, signals[coin])
            
            # Check that signals are one of BUY, SELL, or HOLD
            self.assertIn(signals[coin]["rsi_signal"], ["BUY", "SELL", "HOLD"])
            self.assertIn(signals[coin]["macd_signal"], ["BUY", "SELL"])
            self.assertIn(signals[coin]["bb_signal"], ["BUY", "SELL", "HOLD"])
            self.assertIn(signals[coin]["risk_signal"], ["BUY", "SELL", "HOLD"])
            self.assertIn(signals[coin]["final_signal"], ["BUY", "SELL", "HOLD"])
            
            # Check that confidence is between 0 and 1
            self.assertTrue(0 <= signals[coin]["confidence"] <= 1)
    
    def test_api_connection(self):
        """
        Test the connection to the CoinGecko API
        """
        # This test will be skipped if the API is not available
        try:
            coins = self.data_fetcher.get_top_coins(limit=3)
            self.assertTrue(len(coins) > 0)
        except Exception as e:
            self.skipTest(f"API connection failed: {e}")
    
    def test_technical_indicators(self):
        """
        Test the technical indicators
        """
        # Test RSI
        rsi = self.api.relative_strength_index(self.sample_prices, window=14)
        self.assertEqual(rsi.shape, self.sample_prices.shape)
        self.assertTrue(np.all((rsi >= 0) & (rsi <= 100)))
        
        # Test MACD
        macd_line, signal_line, histogram = self.api.macd(self.sample_prices)
        self.assertEqual(macd_line.shape, self.sample_prices.shape)
        self.assertEqual(signal_line.shape, self.sample_prices.shape)
        self.assertEqual(histogram.shape, self.sample_prices.shape)
        
        # Test Bollinger Bands
        upper, middle, lower = self.api.bollinger_bands(self.sample_prices)
        self.assertEqual(upper.shape, self.sample_prices.shape)
        self.assertEqual(middle.shape, self.sample_prices.shape)
        self.assertEqual(lower.shape, self.sample_prices.shape)
        self.assertTrue(np.all(upper >= middle))
        self.assertTrue(np.all(middle >= lower))

if __name__ == "__main__":
    unittest.main() 