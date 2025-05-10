#!/usr/bin/env python
"""
Cryptocurrency Trading Signal Generator

This script fetches cryptocurrency data from a public API, analyzes it using
the ExtendedStocksAPI from finlib_extensions, and generates buy/sell signals based on technical indicators.
"""

import os
import sys
import time
import json
import logging
import numpy as np
import requests
from datetime import datetime, timedelta
import pandas as pd
from typing import List, Dict, Tuple, Any, Optional
import random
import pickle

# Import ExtendedStocksAPI instead of StocksAPI
from finlib_extensions import ExtendedStocksAPI

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s.%(msecs)03d - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("crypto_signals")

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

class CryptoDataFetcher:
    """
    Class to fetch cryptocurrency data from public APIs with caching
    """
    
    def __init__(self, cache_dir="./data_cache"):
        """
        Initialize the CryptoDataFetcher
        
        Args:
            cache_dir: Directory to store cached data
        """
        with Timer("CryptoDataFetcher initialization"):
            self.api_key = os.environ.get("COIN_API_KEY", "")
            self.base_url = "https://api.coingecko.com/api/v3"  # Using CoinGecko as it has a free tier
            self.cache_dir = cache_dir
            self.historical_data_cache = {}
            self.last_update_time = {}
            self.last_api_call_time = 0  # Track the last API call time for rate limiting
            self.min_request_interval = 0.1  # Minimum seconds between API calls (reduced from 1.5)
            
            # Create cache directory if it doesn't exist
            if not os.path.exists(cache_dir):
                os.makedirs(cache_dir)
            
            # Load cached data if available
            self._load_cache()
        
    def _load_cache(self):
        """
        Load cached data from disk
        """
        cache_file = os.path.join(self.cache_dir, "historical_data.pkl")
        last_update_file = os.path.join(self.cache_dir, "last_update_time.pkl")
        
        try:
            if os.path.exists(cache_file):
                with open(cache_file, "rb") as f:
                    self.historical_data_cache = pickle.load(f)
                logger.info(f"Loaded historical data cache for {len(self.historical_data_cache)} coins")
            
            if os.path.exists(last_update_file):
                with open(last_update_file, "rb") as f:
                    self.last_update_time = pickle.load(f)
                logger.info(f"Loaded last update times for {len(self.last_update_time)} coins")
        except Exception as e:
            logger.error(f"Error loading cache: {e}")
            # Reset cache if there's an error
            self.historical_data_cache = {}
            self.last_update_time = {}
    
    def _save_cache(self):
        """
        Save cached data to disk
        """
        cache_file = os.path.join(self.cache_dir, "historical_data.pkl")
        last_update_file = os.path.join(self.cache_dir, "last_update_time.pkl")
        
        try:
            with open(cache_file, "wb") as f:
                pickle.dump(self.historical_data_cache, f)
            
            with open(last_update_file, "wb") as f:
                pickle.dump(self.last_update_time, f)
            
            logger.info(f"Saved historical data cache for {len(self.historical_data_cache)} coins")
        except Exception as e:
            logger.error(f"Error saving cache: {e}")
        
    def _api_request(self, url, params=None, max_retries=3):
        """
        Make an API request with rate limiting and retries
        
        Args:
            url: API endpoint URL
            params: Query parameters
            max_retries: Maximum number of retries
            
        Returns:
            JSON response or None if failed
        """
        request_start_time = time.time()
        
        if params is None:
            params = {}
            
        # Implement rate limiting
        current_time = time.time()
        time_since_last_call = current_time - self.last_api_call_time
        
        if time_since_last_call < self.min_request_interval:
            sleep_time = self.min_request_interval - time_since_last_call
            logger.debug(f"Rate limiting: Sleeping for {sleep_time:.2f}s")
            time.sleep(sleep_time)
        
        # Add a random component to the User-Agent to reduce chance of being blocked
        headers = {
            "User-Agent": f"CryptoTradingSignals/1.0 (Research Project; {random.randint(1000, 9999)})"
        }
        
        # Try the request with exponential backoff
        retry_count = 0
        while retry_count < max_retries:
            try:
                response = requests.get(url, params=params, headers=headers)
                self.last_api_call_time = time.time()  # Update last call time
                
                # Check for rate limiting response
                if response.status_code == 429:
                    retry_count += 1
                    wait_time = 60 * (2 ** retry_count)  # Exponential backoff: 60s, 120s, 240s
                    logger.warning(f"Rate limit exceeded. Waiting {wait_time}s before retry. (Attempt {retry_count}/{max_retries})")
                    time.sleep(wait_time)
                    continue
                
                response.raise_for_status()
                request_end_time = time.time()
                request_duration = request_end_time - request_start_time
                logger.info(f"TIMING: API request to {url.split('/')[-1]} completed in {request_duration:.3f}s ({int(request_duration * 1000)}ms)")
                return response.json()
                
            except requests.exceptions.RequestException as e:
                retry_count += 1
                if retry_count < max_retries:
                    wait_time = 5 * (2 ** retry_count)  # Exponential backoff for other errors
                    logger.warning(f"Request failed: {e}. Retrying in {wait_time}s. (Attempt {retry_count}/{max_retries})")
                    time.sleep(wait_time)
                else:
                    logger.error(f"Request failed after {max_retries} attempts: {e}")
                    return None
        
        return None

    def get_top_coins(self, limit: int = 10) -> List[str]:
        """
        Get the top cryptocurrencies by market cap
        
        Args:
            limit: Number of top coins to return
            
        Returns:
            List of coin IDs
        """
        with Timer("get_top_coins"):
            # Check if we have a cached list of top coins that's less than 6 hours old
            cache_file = os.path.join(self.cache_dir, "top_coins.pkl")
            
            if os.path.exists(cache_file):
                try:
                    with open(cache_file, "rb") as f:
                        cached_data = pickle.load(f)
                    
                    if "timestamp" in cached_data and "coins" in cached_data:
                        cache_age = (datetime.now() - cached_data["timestamp"]).total_seconds() / 3600
                        
                        if cache_age < 6:  # Use cache if less than 6 hours old
                            logger.info(f"Using cached top coins list ({cache_age:.1f} hours old)")
                            return cached_data["coins"][:limit]
                except Exception as e:
                    logger.error(f"Error loading cached top coins: {e}")
            
            # Fetch from API if no valid cache
            url = f"{self.base_url}/coins/markets"
            params = {
                "vs_currency": "usd",
                "order": "market_cap_desc",
                "per_page": min(limit, 25),  # CoinGecko limit is 250, but we'll be conservative
                "page": 1,
                "sparkline": False
            }
            
            data = self._api_request(url, params)
            
            if data:
                coins = [coin["id"] for coin in data]
                
                # Save to cache
                try:
                    with open(cache_file, "wb") as f:
                        pickle.dump({"timestamp": datetime.now(), "coins": coins}, f)
                    logger.info(f"Cached top coins list")
                except Exception as e:
                    logger.error(f"Error saving top coins cache: {e}")
                    
                return coins
            
            return []
    
    def get_historical_data(self, coin_ids: List[str], days: int = 365, force_refresh: bool = False) -> Dict[str, np.ndarray]:
        """
        Get historical price data for the specified coins, using cache when available
        
        Args:
            coin_ids: List of coin IDs to fetch data for
            days: Number of days of historical data to fetch
            force_refresh: Whether to force refresh the data from the API
            
        Returns:
            Dictionary mapping coin IDs to price arrays
        """
        with Timer(f"get_historical_data for {len(coin_ids)} coins"):
            historical_data = {}
            coins_to_fetch = []
            
            # Check which coins we need to fetch
            for coin_id in coin_ids:
                if not force_refresh and coin_id in self.historical_data_cache:
                    # Use cached data
                    historical_data[coin_id] = self.historical_data_cache[coin_id]
                    logger.info(f"Using cached historical data for {coin_id} ({len(historical_data[coin_id])} data points)")
                else:
                    # Need to fetch this coin
                    coins_to_fetch.append(coin_id)
            
            if coins_to_fetch:
                logger.info(f"Fetching historical data for {len(coins_to_fetch)} coins: {', '.join(coins_to_fetch)}")
                with Timer(f"_fetch_historical_data for {len(coins_to_fetch)} coins"):
                    fetched_data = self._fetch_historical_data(coins_to_fetch, days)
                
                # Update cache and result
                for coin_id, prices in fetched_data.items():
                    self.historical_data_cache[coin_id] = prices
                    historical_data[coin_id] = prices
                    self.last_update_time[coin_id] = datetime.now()
                
                # Save updated cache
                self._save_cache()
            
            return historical_data
    
    def _fetch_historical_data(self, coin_ids: List[str], days: int = 365) -> Dict[str, np.ndarray]:
        """
        Fetch historical price data from the API
        
        Args:
            coin_ids: List of coin IDs to fetch data for
            days: Number of days of historical data to fetch
            
        Returns:
            Dictionary mapping coin IDs to price arrays
        """
        historical_data = {}
        
        # Limit the number of coins to fetch to avoid excessive API calls
        if len(coin_ids) > 5:
            logger.warning(f"Limiting historical data fetch to 5 coins to avoid rate limiting")
            coin_ids = coin_ids[:5]
        
        for coin_id in coin_ids:
            url = f"{self.base_url}/coins/{coin_id}/market_chart"
            params = {
                "vs_currency": "usd",
                "days": days,
                "interval": "daily"
            }
            
            data = self._api_request(url, params)
            
            if data and "prices" in data:
                prices = np.array([price[1] for price in data["prices"]], dtype=np.float32)
                historical_data[coin_id] = prices
                logger.info(f"Successfully fetched historical data for {coin_id} ({len(prices)} data points)")
            else:
                logger.error(f"Failed to fetch historical data for {coin_id}")
        
        return historical_data
    
    def get_latest_data(self, coin_ids: List[str]) -> Dict[str, Dict[str, float]]:
        """
        Get the latest price data for the specified coins
        
        Args:
            coin_ids: List of coin IDs to fetch data for
            
        Returns:
            Dictionary mapping coin IDs to latest prices
        """
        with Timer(f"get_latest_data for {len(coin_ids)} coins"):
            latest_data = {}
            
            # Limit the number of coins to fetch to avoid excessive API calls
            if len(coin_ids) > 10:
                logger.warning(f"Limiting latest data fetch to 10 coins to avoid rate limiting")
                coin_ids = coin_ids[:10]
            
            # Check if we have a cached latest data that's less than 15 minutes old
            cache_file = os.path.join(self.cache_dir, "latest_data.pkl")
            
            if os.path.exists(cache_file):
                try:
                    with open(cache_file, "rb") as f:
                        cached_data = pickle.load(f)
                    
                    if "timestamp" in cached_data and "data" in cached_data:
                        cache_age = (datetime.now() - cached_data["timestamp"]).total_seconds() / 60
                        
                        if cache_age < 15:  # Use cache if less than 15 minutes old
                            logger.info(f"Using cached latest data ({cache_age:.1f} minutes old)")
                            
                            # Filter for requested coins
                            for coin_id in coin_ids:
                                if coin_id in cached_data["data"]:
                                    latest_data[coin_id] = cached_data["data"][coin_id]
                        
                            # If we have all requested coins, return the data
                            if len(latest_data) == len(coin_ids):
                                return latest_data
                except Exception as e:
                    logger.error(f"Error loading cached latest data: {e}")
            
            # Fetch data in a single batch to minimize API calls
            url = f"{self.base_url}/simple/price"
            params = {
                "ids": ",".join(coin_ids),
                "vs_currencies": "usd",
                "include_24hr_change": "true"
            }
            
            data = self._api_request(url, params)
            
            if data:
                all_latest_data = {}
                
                for coin_id in coin_ids:
                    if coin_id in data:
                        latest_data[coin_id] = {
                            "price": data[coin_id]["usd"],
                            "change_24h": data[coin_id].get("usd_24h_change", 0)
                        }
                        all_latest_data[coin_id] = latest_data[coin_id]
            
            # Save to cache
            try:
                with open(cache_file, "wb") as f:
                    pickle.dump({"timestamp": datetime.now(), "data": all_latest_data}, f)
                logger.info(f"Cached latest data for {len(all_latest_data)} coins")
            except Exception as e:
                logger.error(f"Error saving latest data cache: {e}")
        
            return latest_data
    
    def update_historical_data(self, coin_ids: List[str]) -> Dict[str, np.ndarray]:
        """
        Update historical data with the latest prices
        
        Args:
            coin_ids: List of coin IDs to update
            
        Returns:
            Dictionary mapping coin IDs to updated price arrays
        """
        with Timer(f"update_historical_data for {len(coin_ids)} coins"):
            # Get the latest data
            latest_data = self.get_latest_data(coin_ids)
            
            # Update historical data
            updated_data = {}
            for coin_id in coin_ids:
                if coin_id in self.historical_data_cache and coin_id in latest_data:
                    # Get the latest price
                    latest_price = latest_data[coin_id]["price"]
                    
                    # Get the historical data
                    historical_prices = self.historical_data_cache[coin_id]
                    
                    # Check if we need to append the latest price
                    # Only append if the last update was more than 12 hours ago
                    if coin_id in self.last_update_time:
                        last_update = self.last_update_time[coin_id]
                        hours_since_update = (datetime.now() - last_update).total_seconds() / 3600
                        
                        if hours_since_update >= 12:
                            # Append the latest price
                            updated_prices = np.append(historical_prices, latest_price)
                            updated_data[coin_id] = updated_prices
                            
                            # Update cache
                            self.historical_data_cache[coin_id] = updated_prices
                            self.last_update_time[coin_id] = datetime.now()
                            
                            logger.info(f"Updated historical data for {coin_id} with latest price: ${latest_price:.2f}")
                        else:
                            # Use existing data
                            updated_data[coin_id] = historical_prices
                            logger.info(f"Using existing data for {coin_id} (last updated {hours_since_update:.1f} hours ago)")
                    else:
                        # No last update time, just use the existing data
                        updated_data[coin_id] = historical_prices
                elif coin_id in latest_data:
                    # No historical data, just use the latest price
                    updated_data[coin_id] = np.array([latest_data[coin_id]["price"]], dtype=np.float32)
                    logger.warning(f"No historical data for {coin_id}, using only latest price: ${latest_data[coin_id]['price']:.2f}")
            
            # Save updated cache
            self._save_cache()
            
            return updated_data
    
    def format_data_for_analysis(self, historical_data: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Format the historical data for analysis with ExtendedStocksAPI
        
        Args:
            historical_data: Dictionary mapping coin IDs to price arrays
            
        Returns:
            Numpy array with shape (days, assets) containing price data
        """
        # Check if we have any data
        if not historical_data:
            logger.error("No historical data available for analysis")
            return np.array([])
            
        # Find the minimum length across all price arrays
        min_length = min(len(prices) for prices in historical_data.values())
        
        # Create a 2D array with shape (days, assets)
        coins = list(historical_data.keys())
        prices_array = np.zeros((min_length, len(coins)), dtype=np.float32)
        
        for i, coin_id in enumerate(coins):
            prices_array[:, i] = historical_data[coin_id][-min_length:]
        
        logger.info(f"Formatted data for analysis: {prices_array.shape[0]} days, {prices_array.shape[1]} assets")
        return prices_array


class SignalGenerator:
    """
    Class to generate trading signals based on technical indicators
    """
    
    def __init__(self):
        """
        Initialize the SignalGenerator with ExtendedStocksAPI
        """
        with Timer("SignalGenerator initialization"):
            self.api = ExtendedStocksAPI()
        
    def compute_daily_returns(self, prices: np.ndarray) -> np.ndarray:
        """
        Compute daily returns from price data
        
        Args:
            prices: Numpy array with shape (days, assets) containing price data
            
        Returns:
            Numpy array with shape (days-1, assets) containing daily returns
        """
        with Timer("compute_daily_returns"):
            returns = np.zeros((prices.shape[0] - 1, prices.shape[1]), dtype=np.float32)
            
            for i in range(1, prices.shape[0]):
                returns[i-1] = (prices[i] / prices[i-1]) - 1
            
            return returns
    
    def generate_signals(self, prices: np.ndarray, coin_ids: List[str]) -> Dict[str, Dict[str, Any]]:
        """
        Generate trading signals based on technical indicators
        
        Args:
            prices: Numpy array with shape (days, assets) containing price data
            coin_ids: List of coin IDs corresponding to the assets in prices
            
        Returns:
            Dictionary mapping coin IDs to signal dictionaries
        """
        with Timer(f"generate_signals for {len(coin_ids)} coins"):
            signals = {}
            
            # Validate inputs
            if prices.size == 0 or prices.shape[1] == 0:
                logger.error("Empty price data provided")
                return signals
                
            if len(coin_ids) == 0:
                logger.error("No coin IDs provided")
                return signals
            
            # Ensure the number of coins matches the number of assets
            if len(coin_ids) != prices.shape[1]:
                logger.warning(f"Number of coins ({len(coin_ids)}) doesn't match number of assets in price data ({prices.shape[1]})")
                
                if len(coin_ids) > prices.shape[1]:
                    # Truncate coin_ids to match the number of assets
                    logger.warning(f"Truncating coin list to match price data: {coin_ids[:prices.shape[1]]}")
                    coin_ids = coin_ids[:prices.shape[1]]
                else:
                    # Truncate price data to match the number of coins
                    logger.warning(f"Truncating price data to match coin list")
                    prices = prices[:, :len(coin_ids)]
            
            logger.info(f"Generating signals for {len(coin_ids)} coins with price data shape {prices.shape}")
            
            try:
                # Compute daily returns
                with Timer("Computing daily returns"):
                    returns = self.compute_daily_returns(prices)
                
                # Calculate RSI
                with Timer("Calculating RSI"):
                    rsi = self.api.relative_strength_index(prices, window=14)
                
                # Calculate MACD
                with Timer("Calculating MACD"):
                    macd_line, signal_line, histogram = self.api.macd(prices, fast_period=12, slow_period=26, signal_period=9)
                
                # Calculate Bollinger Bands
                with Timer("Calculating Bollinger Bands"):
                    upper_band, middle_band, lower_band = self.api.bollinger_bands(prices, window=20, num_std=2)
                
                # Calculate VaR and Expected Shortfall
                with Timer("Calculating VaR and ES"):
                    var, es = self.api.calculate_var_and_es(returns, confidence_level=0.95)
                
                # Calculate Black-Scholes VaR
                with Timer("Calculating Black-Scholes VaR"):
                    bs_var = self.api.calculate_black_scholes_var(returns, confidence_level=0.95, num_simulations=100000)
                
                # Generate signals for each coin
                with Timer("Generating final signals"):
                    for i, coin_id in enumerate(coin_ids):
                        # Get the most recent values
                        current_price = prices[-1, i]
                        current_rsi = rsi[-1, i]
                        current_macd = macd_line[-1, i]
                        current_signal = signal_line[-1, i]
                        current_upper = upper_band[-1, i]
                        current_middle = middle_band[-1, i]
                        current_lower = lower_band[-1, i]
                        current_var = var[i]
                        current_es = es[i]
                        current_bs_var = bs_var[i]
                        
                        # Determine signals
                        rsi_signal = "SELL" if current_rsi > 70 else "BUY" if current_rsi < 30 else "HOLD"
                        macd_signal = "BUY" if current_macd > current_signal else "SELL"
                        bb_signal = "BUY" if current_price < current_lower else "SELL" if current_price > current_upper else "HOLD"
                        
                        # Risk-based signal
                        risk_signal = "SELL" if current_bs_var > 0.05 else "BUY" if current_bs_var < 0.02 else "HOLD"
                        
                        # Combine signals (simple majority voting)
                        signals_list = [rsi_signal, macd_signal, bb_signal, risk_signal]
                        buy_count = signals_list.count("BUY")
                        sell_count = signals_list.count("SELL")
                        hold_count = signals_list.count("HOLD")
                        
                        if buy_count > sell_count and buy_count > hold_count:
                            final_signal = "BUY"
                        elif sell_count > buy_count and sell_count > hold_count:
                            final_signal = "SELL"
                        else:
                            final_signal = "HOLD"
                        
                        # Store signals and metrics
                        signals[coin_id] = {
                            "price": float(current_price),
                            "rsi": float(current_rsi),
                            "macd": float(current_macd),
                            "signal_line": float(current_signal),
                            "upper_band": float(current_upper),
                            "middle_band": float(current_middle),
                            "lower_band": float(current_lower),
                            "var_95": float(current_var),
                            "es_95": float(current_es),
                            "bs_var_95": float(current_bs_var),
                            "rsi_signal": rsi_signal,
                            "macd_signal": macd_signal,
                            "bb_signal": bb_signal,
                            "risk_signal": risk_signal,
                            "final_signal": final_signal,
                            "confidence": max(buy_count, sell_count, hold_count) / 4.0  # Updated for 4 signals
                        }
            except Exception as e:
                logger.error(f"Error generating signals: {e}")
                # Return any signals we were able to generate
            
            return signals


def main():
    """
    Main function to run the crypto trading signal generator
    """
    start_time = time.time()
    logger.info(f"Starting crypto trading signal generator at {datetime.now().isoformat()}")
    
    # Initialize the data fetcher and signal generator
    data_fetcher = CryptoDataFetcher()
    signal_generator = SignalGenerator()
    
    # Get the top 10 cryptocurrencies
    logger.info("Fetching top cryptocurrencies...")
    top_coins = data_fetcher.get_top_coins(limit=10)
    
    if not top_coins:
        logger.error("Failed to fetch top coins. Exiting.")
        return
    
    logger.info(f"Top coins: {', '.join(top_coins)}")
    
    # Get historical data for the top coins
    logger.info("Fetching historical data...")
    historical_data = data_fetcher.get_historical_data(top_coins)
    
    if not historical_data:
        logger.error("Failed to fetch historical data. Exiting.")
        return
    
    # Format data for analysis
    logger.info("Formatting data for analysis...")
    with Timer("format_data_for_analysis"):
        prices = data_fetcher.format_data_for_analysis(historical_data)
    
    # Generate signals
    logger.info("Generating trading signals...")
    signals = signal_generator.generate_signals(prices, top_coins)
    
    # Print signals
    logger.info("Trading signals generated:")
    for coin_id, signal_data in signals.items():
        logger.info(f"{coin_id.upper()}: {signal_data['final_signal']} (Confidence: {signal_data['confidence']:.2f})")
        logger.info(f"  Price: ${signal_data['price']:.2f}")
        logger.info(f"  RSI: {signal_data['rsi']:.2f} ({signal_data['rsi_signal']})")
        logger.info(f"  MACD: {signal_data['macd']:.6f} vs Signal: {signal_data['signal_line']:.6f} ({signal_data['macd_signal']})")
        logger.info(f"  Bollinger Bands: {signal_data['lower_band']:.2f} < {signal_data['middle_band']:.2f} < {signal_data['upper_band']:.2f} ({signal_data['bb_signal']})")
        logger.info(f"  VaR (95%): {signal_data['var_95']:.2f}%")
        logger.info(f"  ES (95%): {signal_data['es_95']:.2f}%")
        logger.info(f"  Black-Scholes VaR (95%): {signal_data['bs_var_95']:.2f}%")
        logger.info(f"  Risk Signal: {signal_data['risk_signal']}")
        logger.info("")
    
    # Output JSON for potential bot consumption
    output = {
        "timestamp": datetime.now().isoformat(),
        "signals": signals
    }
    
    with open("crypto_signals.json", "w") as f:
        json.dump(output, f, indent=2)
    
    logger.info("Signals saved to crypto_signals.json")
    
    # Log total execution time
    end_time = time.time()
    total_time = end_time - start_time
    logger.info(f"TIMING: Total execution time: {total_time:.3f}s ({int(total_time * 1000)}ms)")


if __name__ == "__main__":
    main() 