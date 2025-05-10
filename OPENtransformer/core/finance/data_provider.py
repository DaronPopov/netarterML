import requests
import numpy as np
import pandas as pd
import datetime
import time
import logging
import yfinance as yf
from functools import lru_cache
import os
from dotenv import load_dotenv
import random
from typing import Dict, List, Any, Optional, Tuple, Union
import threading
import json

# Load environment variables
load_dotenv()

# Import our cache system
from finlib.finance.data_cache import get_cache_instance

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("data_provider")

# Define asset types
CRYPTO_ASSETS = ["BTC", "ETH", "LTC", "XRP", "ADA", "DOT", "SOL", "BNB", "AVAX", "DOGE"]
STOCK_ASSETS = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "NVDA", "JPM", "V", "WMT"]

# Improved Rate Limiting Settings
# We'll set a per-service rate limit configuration with burst capacity
RATE_LIMIT_CONFIG = {
    "requests_per_hour": 1800,  # Slightly under the 2000 limit to be safe
}

# Calculate the interval in seconds for each service
for service, config in RATE_LIMIT_CONFIG.items():
    config["interval"] = 3600 / config["requests_per_hour"]  # seconds per request

# Track last API call time and burst capacity for each service
last_api_call_time = {}
current_burst_capacity = {
    service: config["burst_capacity"] 
    for service, config in RATE_LIMIT_CONFIG.items()
}
burst_last_refill_time = {
    service: time.time() 
    for service in RATE_LIMIT_CONFIG
}

# Thread lock for rate limiting
rate_limit_lock = threading.Lock()

# Initialize the cache
data_cache = get_cache_instance()

def is_crypto(asset_symbol):
    """
    Check if an asset is a cryptocurrency.
    
    Args:
        asset_symbol: The asset symbol
        
    Returns:
        bool: True if the asset is a cryptocurrency, False otherwise
    """
    return asset_symbol.upper() in CRYPTO_ASSETS

def rate_limit(service: str = "default") -> None:
    """
    Rate limit API calls with burst capacity and jitter.
    
    Args:
        service: Service identifier for rate limiting
    """
    with rate_limit_lock:
        config = RATE_LIMIT_CONFIG.get(service, RATE_LIMIT_CONFIG["default"])
        current_time = time.time()
        
        # Check if we need to refill burst capacity
        time_since_refill = current_time - burst_last_refill_time.get(service, 0)
        if time_since_refill >= config["burst_refill_time"]:
            refills = int(time_since_refill / config["burst_refill_time"])
            current_burst_capacity[service] = min(
                config["burst_capacity"],
                current_burst_capacity.get(service, 0) + refills
            )
            burst_last_refill_time[service] = current_time
            
        # If we have burst capacity available, use it and return immediately
        if current_burst_capacity.get(service, 0) > 0:
            current_burst_capacity[service] -= 1
            last_api_call_time[service] = current_time
            return
            
        # Otherwise, we need to wait
        last_call_time = last_api_call_time.get(service, 0)
        time_since_last_call = current_time - last_call_time
        
        # Minimum wait time based on rate limit, with jitter to avoid synchronized requests
        min_wait_time = config["interval"]
        actual_wait_time = max(0, min_wait_time - time_since_last_call)
        
        # Add small random jitter (up to 10% of wait time) to avoid synchronized requests
        if actual_wait_time > 0:
            jitter = random.uniform(0, min(0.1 * actual_wait_time, 0.5))
            actual_wait_time += jitter
            time.sleep(actual_wait_time)
            
        # Update last call time after waiting
        last_api_call_time[service] = time.time()

# Cache real-time data for 5 seconds to avoid excessive API calls
@lru_cache(maxsize=128)
def fetch_realtime_data_cached(asset_symbol, quote_currency, timestamp):
    """
    Cached version of fetch_realtime_data to avoid excessive API calls.
    The timestamp parameter is used to invalidate the cache every 5 seconds.
    
    Args:
        asset_symbol: The asset symbol
        quote_currency: The quote currency
        timestamp: Current timestamp rounded to the nearest 5 seconds
        
    Returns:
        dict: Real-time data for the asset
    """
    return fetch_realtime_data_internal(asset_symbol, quote_currency)

def fetch_realtime_data(asset_symbol, quote_currency="USD"):
    """
    Fetch real-time data for an asset from the appropriate data provider.
    Uses caching to avoid excessive API calls.
    
    Args:
        asset_symbol: The asset symbol
        quote_currency: The quote currency
        
    Returns:
        dict: Real-time data for the asset
    """
    # Round timestamp to the nearest 5 seconds for cache invalidation
    current_timestamp = int(time.time() / 5) * 5
    return fetch_realtime_data_cached(asset_symbol, quote_currency, current_timestamp)

def fetch_realtime_data_internal(asset_symbol, quote_currency="USD"):
    """
    Internal function to fetch real-time data for an asset from the appropriate data provider.
    Uses yfinance for real-time data for all assets.
    
    Args:
        asset_symbol: The asset symbol
        quote_currency: The quote currency
        
    Returns:
        dict: Real-time data for the asset
    """
    try:
        # Adjust symbol for crypto assets in yfinance format
        if is_crypto(asset_symbol):
            ticker_symbol = f"{asset_symbol}-{quote_currency}"
        else:
            ticker_symbol = asset_symbol
            
        # Use Yahoo Finance for all real-time data
        return fetch_realtime_data_yfinance(ticker_symbol)
    except Exception as e:
        logger.error(f"Error fetching real-time data for {asset_symbol}: {e}")
        return None

def fetch_historical_data(asset_symbol, quote_currency="USD", period_id='1MIN', limit=500):
    """
    Fetch historical data for an asset using YFinance.
    Uses disk-based caching to avoid excessive API calls.
    
    Args:
        asset_symbol: The asset symbol
        quote_currency: The quote currency
        period_id: The time period (e.g., '1MIN', '1D')
        limit: The number of data points to fetch
        
    Returns:
        list: Historical data for the asset
    """
    is_crypto_asset = is_crypto(asset_symbol)
    
    # Try to get data from cache first
    cached_data = data_cache.get_cached_data(
        asset_symbol, 
        quote_currency, 
        period_id, 
        is_crypto_asset
    )
    
    if cached_data is not None:
        logger.info(f"Cache hit for {asset_symbol} ({period_id})")
        return cached_data
    
    # If not in cache, fetch from YFinance API
    try:
        # Adjust symbol for crypto assets in yfinance format
        if is_crypto_asset:
            ticker_symbol = f"{asset_symbol}-{quote_currency}"
        else:
            ticker_symbol = asset_symbol
            
        # Convert period_id to format expected by yfinance
        if period_id == '1MIN':
            yf_period = "1d"
            yf_interval = "1m"
        elif period_id == '1HRS' or period_id == '1h':
            yf_period = "7d"
            yf_interval = "1h"
        elif period_id == '1DAY' or period_id == '1D' or period_id == '1d':
            yf_period = "60d"
            yf_interval = "1d"
        else:
            # Default to daily data for 60 days
            yf_period = "60d"
            yf_interval = "1d"
        
        # Apply rate limiting for Yahoo Finance
        rate_limit("yahoo")
        logger.info(f"Fetching historical data for {ticker_symbol} (period: {yf_period}, interval: {yf_interval})")
        
        # Fetch historical data from yfinance
        ticker = yf.Ticker(ticker_symbol)
        df = ticker.history(period=yf_period, interval=yf_interval)
        
        if df.empty:
            logger.warning(f"No data returned from Yahoo Finance for {ticker_symbol}")
            return generate_synthetic_historical_data(asset_symbol, quote_currency, period_id)
            
        # Convert to list of dictionaries
        results = []
        for timestamp, row in df.iterrows():
            # Format date based on period_id
            if period_id == '1MIN':
                date_str = timestamp.strftime('%Y-%m-%dT%H:%M:%S.%fZ')
            else:
                date_str = timestamp.strftime('%Y-%m-%d')
                
            result = {
                "asset_id_base": asset_symbol,
                "asset_id_quote": quote_currency,
                "time": date_str,
                "date": date_str.split('T')[0],
                "open": float(row["Open"]),
                "high": float(row["High"]),
                "low": float(row["Low"]),
                "close": float(row["Close"]),
                "volume": int(row["Volume"]),
                "price": float(row["Close"]),
                "data_source": "yfinance"
            }
            results.append(result)
            
        # Limit the number of results if necessary
        if limit and len(results) > limit:
            results = results[-limit:]
            
        # Cache the results
        data_cache.cache_data(
            asset_symbol, 
            quote_currency, 
            period_id, 
            is_crypto_asset, 
            results
        )
        
        logger.info(f"Fetched {len(results)} data points for {asset_symbol} using YFinance")
        return results
            
    except Exception as e:
        logger.error(f"Error fetching historical data for {asset_symbol}: {e}")
        return generate_synthetic_historical_data(asset_symbol, quote_currency, period_id)

def generate_synthetic_historical_data(asset_symbol, quote_currency="USD", period_id='1d'):
    """
    Generate synthetic historical data when API calls fail.
    
    Args:
        asset_symbol: The asset symbol
        quote_currency: The quote currency
        period_id: The time period (e.g., '1MIN', '1D')
        
    Returns:
        list: Synthetic historical data
    """
    logger.info(f"Generating synthetic historical data for {asset_symbol}")
    
    # Get base price for the asset
    base_price = 100.0
    volatility = 0.02  # 2% daily volatility
    
    if asset_symbol in CRYPTO_ASSETS:
        if asset_symbol == "BTC":
            base_price = 80000.0
            volatility = 0.03
        elif asset_symbol == "ETH":
            base_price = 3800.0
            volatility = 0.025
        elif asset_symbol == "SOL":
            base_price = 180.0
            volatility = 0.035
        else:
            base_price = 10.0
            volatility = 0.04
    elif asset_symbol in STOCK_ASSETS:
        if asset_symbol == "AAPL":
            base_price = 170.0
            volatility = 0.015
        elif asset_symbol == "MSFT":
            base_price = 400.0
            volatility = 0.015
        elif asset_symbol == "GOOGL":
            base_price = 150.0
            volatility = 0.02
        else:
            base_price = 100.0
            volatility = 0.02
            
    # Determine number of data points and interval based on period_id
    if period_id == '1MIN':
        num_points = 60  # 1 hour
        interval_str = "minute"
        date_format = '%Y-%m-%dT%H:%M:%S.%fZ'
        
        # Calculate dates: start from now and go back by num_points minutes
        end_date = datetime.datetime.now()
        dates = [end_date - datetime.timedelta(minutes=i) for i in range(num_points)]
        dates.reverse()  # Put in chronological order
    elif period_id == '1HRS' or period_id == '1h':
        num_points = 168  # 7 days of hourly data
        interval_str = "hour"
        date_format = '%Y-%m-%dT%H:%M:%S.%fZ'
        
        # Calculate dates: start from now and go back by num_points hours
        end_date = datetime.datetime.now()
        dates = [end_date - datetime.timedelta(hours=i) for i in range(num_points)]
        dates.reverse()  # Put in chronological order
    else:  # Daily data
        num_points = 60  # 60 days
        interval_str = "day"
        date_format = '%Y-%m-%d'
        
        # Calculate dates: start from now and go back by num_points days
        end_date = datetime.datetime.now()
        dates = [end_date - datetime.timedelta(days=i) for i in range(num_points)]
        dates.reverse()  # Put in chronological order
    
    # Generate synthetic data with random walk
    results = []
    current_price = base_price
    
    for date in dates:
        # Simulate daily price movement using geometric Brownian motion
        daily_return = np.random.normal(0.0005, volatility)  # Small positive drift
        current_price *= (1 + daily_return)
        
        # Simulate OHLC
        daily_volatility = current_price * volatility
        open_price = current_price * (1 + np.random.normal(0, 0.002))
        high_price = max(open_price, current_price) * (1 + abs(np.random.normal(0, 0.003)))
        low_price = min(open_price, current_price) * (1 - abs(np.random.normal(0, 0.003)))
        volume = int(np.random.normal(1000000, 500000))
        
        # Format date based on period_id
        if period_id == '1MIN' or period_id == '1HRS' or period_id == '1h':
            date_str = date.strftime(date_format)
        else:
            date_str = date.strftime(date_format)
            
        result = {
            "asset_id_base": asset_symbol,
            "asset_id_quote": quote_currency,
            "time": date_str,
            "date": date_str.split('T')[0] if 'T' in date_str else date_str,
            "open": float(open_price),
            "high": float(high_price),
            "low": float(low_price),
            "close": float(current_price),
            "volume": volume,
            "price": float(current_price),
            "data_source": "synthetic"
        }
        results.append(result)
    
    logger.info(f"Generated {len(results)} synthetic data points for {asset_symbol}")
    return results

def fetch_realtime_data_yfinance(ticker_symbol):
    """
    Fetch real-time data from Yahoo Finance.
    
    Args:
        ticker_symbol: The ticker symbol (e.g., 'AAPL' or 'BTC-USD')
        
    Returns:
        dict: Real-time data for the asset
    """
    try:
        # Apply rate limiting for Yahoo Finance
        rate_limit("yahoo")
        
        # Extract base asset from ticker symbol
        if "-" in ticker_symbol:
            asset = ticker_symbol.split("-")[0]
            quote = ticker_symbol.split("-")[1]
        else:
            asset = ticker_symbol
            quote = "USD"
        
        # Create a yfinance Ticker object
        ticker = yf.Ticker(ticker_symbol)
        
        # Get the latest data
        data = ticker.history(period="1d")
        
        if data.empty:
            logger.warning(f"No data available for {ticker_symbol}")
            return None
            
        # Get the latest price
        latest = data.iloc[-1]
        
        # Calculate price change
        if len(data) > 1:
            prev_close = data.iloc[-2]['Close']
            change = latest['Close'] - prev_close
            percent_change = (change / prev_close) * 100 if prev_close > 0 else 0
        else:
            # If we only have one data point, use Open price for comparison
            change = latest['Close'] - latest['Open']
            percent_change = (change / latest['Open']) * 100 if latest['Open'] > 0 else 0
            
        # Format the result to match our API format
        result = {
            "asset_id_base": asset,
            "asset_id_quote": quote,
            "rate": float(latest['Close']),
            "volume": int(latest['Volume']),
            "change_24h": float(percent_change),
            "time": datetime.datetime.now().isoformat(),
            "price": float(latest['Close']),
            "high_24h": float(data['High'].max()),
            "low_24h": float(data['Low'].min()),
            "open_24h": float(data.iloc[0]['Open']),
            "data_source": "yfinance"
        }
        
        return result
    except Exception as e:
        logger.error(f"Error fetching data from Yahoo Finance for {ticker_symbol}: {e}")
        return None

def analyze_buy_sell(historical_rates, short_window=5, long_window=20):
    """
    Analyze buy/sell signals based on moving averages.
    
    Args:
        historical_rates: List of historical rates
        short_window: Short moving average window
        long_window: Long moving average window
        
    Returns:
        str: Buy/sell signal
    """
    if len(historical_rates) < long_window:
        return "Insufficient data for analysis"

    df = pd.DataFrame({'rate': historical_rates})
    df['short_ma'] = df['rate'].rolling(window=short_window).mean()
    df['long_ma'] = df['rate'].rolling(window=long_window).mean()

    # Generate buy/sell signals
    if df['short_ma'].iloc[-1] > df['long_ma'].iloc[-1] and df['short_ma'].iloc[-2] <= df['long_ma'].iloc[-2]:
        return "Buy"
    elif df['short_ma'].iloc[-1] < df['long_ma'].iloc[-1] and df['short_ma'].iloc[-2] >= df['long_ma'].iloc[-2]:
        return "Sell"
    else:
        return "Hold"

# Test function
def test_data_providers():
    """
    Test the data providers by fetching data for a stock and a cryptocurrency.
    """
    # Test with a stock
    stock_symbol = "AAPL"
    logger.info(f"Testing with stock: {stock_symbol}")
    
    # Fetch real-time data
    stock_realtime = fetch_realtime_data(stock_symbol)
    if stock_realtime:
        logger.info(f"Real-time data for {stock_symbol}: {stock_realtime['rate']}")
    else:
        logger.error(f"Failed to fetch real-time data for {stock_symbol}")
    
    # Fetch historical data
    stock_historical = fetch_historical_data(stock_symbol, period_id='1d', limit=30)
    if stock_historical:
        logger.info(f"Historical data for {stock_symbol}: {len(stock_historical)} data points")
        logger.info(f"First data point: {stock_historical[0]}")
    else:
        logger.error(f"Failed to fetch historical data for {stock_symbol}")
    
    # Test with a cryptocurrency
    crypto_symbol = "BTC"
    logger.info(f"Testing with cryptocurrency: {crypto_symbol}")
    
    # Fetch real-time data
    crypto_realtime = fetch_realtime_data(crypto_symbol)
    if crypto_realtime:
        logger.info(f"Real-time data for {crypto_symbol}: {crypto_realtime['price' if 'price' in crypto_realtime else 'rate']}")
    else:
        logger.error(f"Failed to fetch real-time data for {crypto_symbol}")
    
    # Fetch historical data
    crypto_historical = fetch_historical_data(crypto_symbol, period_id='1d', limit=30)
    if crypto_historical:
        logger.info(f"Historical data for {crypto_symbol}: {len(crypto_historical)} data points")
        logger.info(f"First data point: {crypto_historical[0]}")
    else:
        logger.error(f"Failed to fetch historical data for {crypto_symbol}")

# Add a new function to preload data for multiple assets
def preload_historical_data(assets, quote_currency="USD", period_id=None, limit=500):
    """
    Preload historical data for all assets to ensure it's available when needed.
    Uses the caching system to avoid repeated API calls.
    
    Args:
        assets: List of asset symbols
        quote_currency: Quote currency for all assets
        period_id: The time period (e.g., '1MIN', '1D'). If None, uses appropriate default for asset type.
        limit: The number of data points to fetch
        
    Returns:
        dict: Dictionary mapping asset symbols to success/failure status
    """
    logger.info(f"Preloading historical data for {len(assets)} assets...")
    results = {}
    
    for asset in assets:
        is_crypto_asset = is_crypto(asset)
        
        # Use appropriate period_id if not specified
        asset_period_id = period_id
        if asset_period_id is None:
            asset_period_id = '1d' if is_crypto_asset else '1d'
        
        logger.info(f"Preloading data for {asset} ({asset_period_id})...")
        
        try:
            # Use the fetch_historical_data function which handles caching
            data = fetch_historical_data(asset, quote_currency, asset_period_id, limit)
            
            if data and len(data) > 0:
                logger.info(f"Successfully preloaded {len(data)} data points for {asset}")
                results[asset] = True
            else:
                logger.warning(f"Failed to preload data for {asset}")
                results[asset] = False
        except Exception as e:
            logger.error(f"Error preloading data for {asset}: {e}")
            results[asset] = False
    
    # Log cache statistics
    stats = data_cache.get_cache_stats()
    logger.info(f"Cache statistics: {stats}")
    
    return results

if __name__ == "__main__":
    test_data_providers() 