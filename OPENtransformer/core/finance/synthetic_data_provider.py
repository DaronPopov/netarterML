import numpy as np
import pandas as pd
import datetime
import time
import logging
import random
from typing import Dict, List, Optional, Union

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("synthetic_data_provider")

# Define asset types
CRYPTO_ASSETS = ["BTC", "ETH", "LTC", "XRP", "ADA", "DOT", "SOL", "BNB", "AVAX", "DOGE"]
STOCK_ASSETS = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "NVDA", "JPM", "V", "WMT"]

# Base prices for synthetic data generation
BASE_PRICES = {
    "BTC": 85000.0,
    "ETH": 3500.0,
    "LTC": 80.0,
    "XRP": 0.5,
    "ADA": 0.4,
    "DOT": 7.0,
    "SOL": 150.0,
    "BNB": 600.0,
    "AVAX": 35.0,
    "DOGE": 0.15,
    "AAPL": 210.0,
    "MSFT": 420.0,
    "GOOGL": 170.0,
    "AMZN": 180.0,
    "META": 500.0,
    "TSLA": 180.0,
    "NVDA": 950.0,
    "JPM": 200.0,
    "V": 280.0,
    "WMT": 60.0
}

# Volatility parameters for each asset
VOLATILITY = {
    "BTC": 0.02,    # 2% volatility for Bitcoin
    "ETH": 0.025,   # 2.5% for Ethereum
    "LTC": 0.03,    # 3% for Litecoin
    "XRP": 0.035,   # 3.5% for Ripple
    "ADA": 0.04,    # 4% for Cardano
    "DOT": 0.035,   # 3.5% for Polkadot
    "SOL": 0.045,   # 4.5% for Solana
    "BNB": 0.025,   # 2.5% for Binance Coin
    "AVAX": 0.04,   # 4% for Avalanche
    "DOGE": 0.05,   # 5% for Dogecoin
    "AAPL": 0.015,  # 1.5% for Apple
    "MSFT": 0.015,  # 1.5% for Microsoft
    "GOOGL": 0.015, # 1.5% for Google
    "AMZN": 0.02,   # 2% for Amazon
    "META": 0.02,   # 2% for Meta
    "TSLA": 0.025,  # 2.5% for Tesla
    "NVDA": 0.025,  # 2.5% for NVIDIA
    "JPM": 0.015,   # 1.5% for JPMorgan
    "V": 0.015,     # 1.5% for Visa
    "WMT": 0.01     # 1% for Walmart
}

# Cached synthetic data to ensure consistency between calls
_synthetic_data_cache = {}

def is_crypto(asset_symbol: str) -> bool:
    """
    Check if an asset is a cryptocurrency.
    
    Args:
        asset_symbol: The asset symbol to check
        
    Returns:
        bool: True if the asset is a cryptocurrency, False otherwise
    """
    return asset_symbol.upper() in CRYPTO_ASSETS

def get_asset_ticker(asset_symbol: str, quote_currency: str = "USD") -> str:
    """
    Get the ticker symbol for an asset.
    
    Args:
        asset_symbol: The asset symbol
        quote_currency: The quote currency
        
    Returns:
        str: The ticker symbol
    """
    if is_crypto(asset_symbol):
        return f"{asset_symbol.upper()}/{quote_currency}"
    else:
        return asset_symbol.upper()

def generate_synthetic_price(asset_symbol: str, base_price: float, volatility: float) -> float:
    """
    Generate a synthetic price for an asset.
    
    Args:
        asset_symbol: The asset symbol
        base_price: The base price for the asset
        volatility: The volatility parameter for the asset
        
    Returns:
        float: The synthetic price
    """
    # Use a random walk model with mean reversion and momentum
    random_component = np.random.normal(0, volatility)
    momentum = np.random.choice([-1, 1]) * volatility * 0.5  # Add momentum effect
    mean_reversion = 0.1 * (base_price - base_price * (1 + random_component)) / base_price
    
    # Combine effects
    total_change = random_component + momentum + mean_reversion
    
    # Ensure price doesn't go too far from base price (within ±30%)
    total_change = np.clip(total_change, -0.3, 0.3)
    
    return base_price * (1 + total_change)

def fetch_realtime_data(asset_symbol: str, quote_currency: str = "USD") -> Dict:
    """
    Fetch synthetic real-time data for an asset.
    
    Args:
        asset_symbol: The asset symbol
        quote_currency: The quote currency
        
    Returns:
        dict: The synthetic real-time data
    """
    asset_symbol = asset_symbol.upper()
    
    # Get base price and volatility
    base_price = BASE_PRICES.get(asset_symbol, 100.0)
    vol = VOLATILITY.get(asset_symbol, 0.03)
    
    # Generate synthetic price
    price = generate_synthetic_price(asset_symbol, base_price, vol)
    
    # Create response
    response = {
        "asset_id_base": asset_symbol,
        "asset_id_quote": quote_currency,
        "rate": price,
        "time": datetime.datetime.now().isoformat()
    }
    
    logger.debug(f"Generated synthetic real-time data for {asset_symbol}: {response}")
    return response

def generate_historical_data(asset_symbol: str, period_id: str, limit: int) -> List[Dict]:
    """
    Generate synthetic historical data for an asset.
    
    Args:
        asset_symbol: The asset symbol
        period_id: The period ID (e.g., '1MIN', '1d')
        limit: The number of data points to generate
        
    Returns:
        list: The synthetic historical data
    """
    asset_symbol = asset_symbol.upper()
    
    # Check if we already have cached data for this asset and period
    cache_key = f"{asset_symbol}_{period_id}_{limit}"
    if cache_key in _synthetic_data_cache:
        return _synthetic_data_cache[cache_key]
    
    # Get base price and volatility
    base_price = BASE_PRICES.get(asset_symbol, 100.0)
    vol = VOLATILITY.get(asset_symbol, 0.03)
    
    # Determine time delta based on period_id
    if period_id == '1MIN':
        time_delta = datetime.timedelta(minutes=1)
    elif period_id == '5MIN':
        time_delta = datetime.timedelta(minutes=5)
    elif period_id == '15MIN':
        time_delta = datetime.timedelta(minutes=15)
    elif period_id == '30MIN':
        time_delta = datetime.timedelta(minutes=30)
    elif period_id == '1HRS':
        time_delta = datetime.timedelta(hours=1)
    elif period_id == '1d':
        time_delta = datetime.timedelta(days=1)
    else:
        time_delta = datetime.timedelta(days=1)
    
    # Generate synthetic data
    data = []
    current_time = datetime.datetime.now()
    
    # Set seed for reproducibility
    np.random.seed(hash(asset_symbol) % 10000)
    
    # Generate initial price
    current_price = base_price
    
    # Generate data points
    for i in range(limit):
        # Calculate time for this data point
        point_time = current_time - (limit - i) * time_delta
        
        # Generate price using geometric Brownian motion
        if i > 0:
            # Mean reversion component
            mean_reversion = 0.05 * (base_price - current_price) / base_price
            
            # Random component with volatility
            random_component = np.random.normal(0, vol)
            
            # Update price
            current_price = current_price * (1 + mean_reversion + random_component)
        
        # Ensure price is positive
        current_price = max(current_price, 0.01)
        
        # Calculate high, low, and open prices
        high_price = current_price * (1 + abs(np.random.normal(0, vol * 0.5)))
        low_price = current_price * (1 - abs(np.random.normal(0, vol * 0.5)))
        
        # Ensure low price is less than high price and positive
        low_price = min(low_price, high_price * 0.99)
        low_price = max(low_price, 0.01)
        
        # Previous close becomes the open for this candle
        if i > 0:
            open_price = data[i-1]["rate_close"]
        else:
            open_price = current_price * (1 + np.random.normal(0, vol * 0.3))
        
        # Create data point
        data_point = {
            "time_period_start": point_time.isoformat(),
            "time_period_end": (point_time + time_delta).isoformat(),
            "time_open": point_time.isoformat(),
            "time_close": (point_time + time_delta).isoformat(),
            "rate_open": float(open_price),
            "rate_high": float(high_price),
            "rate_low": float(low_price),
            "rate_close": float(current_price),
            "volume": float(np.random.randint(1000, 100000))
        }
        
        data.append(data_point)
    
    # Cache the generated data
    _synthetic_data_cache[cache_key] = data
    
    return data

def fetch_historical_data(asset_symbol: str, quote_currency: str = "USD", period_id: str = '1MIN', limit: int = 500) -> List[Dict]:
    """
    Fetch synthetic historical data for an asset.
    
    Args:
        asset_symbol: The asset symbol
        quote_currency: The quote currency
        period_id: The period ID (e.g., '1MIN', '1d')
        limit: The number of data points to generate
        
    Returns:
        list: The synthetic historical data
    """
    asset_symbol = asset_symbol.upper()
    
    # Generate synthetic historical data
    data = generate_historical_data(asset_symbol, period_id, limit)
    
    logger.debug(f"Generated synthetic historical data for {asset_symbol}: {len(data)} data points")
    return data

def test_data_providers() -> bool:
    """
    Test the synthetic data providers.
    
    Returns:
        bool: True if the test is successful, False otherwise
    """
    logger.info("Testing synthetic data providers...")
    
    # Test real-time data
    try:
        btc_data = fetch_realtime_data("BTC", "USD")
        logger.info(f"Synthetic BTC/USD real-time data: {btc_data['rate']:.2f}")
        
        aapl_data = fetch_realtime_data("AAPL", "USD")
        logger.info(f"Synthetic AAPL real-time data: {aapl_data['rate']:.2f}")
    except Exception as e:
        logger.error(f"Error testing synthetic real-time data: {e}")
        return False
    
    # Test historical data
    try:
        btc_hist = fetch_historical_data("BTC", "USD", period_id='1MIN', limit=100)
        logger.info(f"Synthetic BTC/USD historical data: {len(btc_hist)} data points")
        
        aapl_hist = fetch_historical_data("AAPL", "USD", period_id='1d', limit=100)
        logger.info(f"Synthetic AAPL historical data: {len(aapl_hist)} data points")
    except Exception as e:
        logger.error(f"Error testing synthetic historical data: {e}")
        return False
    
    logger.info("Synthetic data providers test completed successfully")
    return True

def preload_historical_data(assets: List[str], quote_currency: str = "USD", period_id: Optional[str] = None, limit: int = 500) -> Dict[str, bool]:
    """
    Preload synthetic historical data for a list of assets.
    
    Args:
        assets: List of asset symbols
        quote_currency: The quote currency
        period_id: The period ID (e.g., '1MIN', '1d')
        limit: The number of data points to generate
        
    Returns:
        dict: Dictionary mapping asset symbols to success/failure status
    """
    logger.info(f"Preloading synthetic historical data for {len(assets)} assets...")
    
    results = {}
    
    for asset in assets:
        asset = asset.upper()
        
        # Determine period_id based on asset type if not specified
        if period_id is None:
            if is_crypto(asset):
                asset_period_id = '1MIN'
            else:
                asset_period_id = '1d'
        else:
            asset_period_id = period_id
        
        try:
            data = fetch_historical_data(asset, quote_currency, period_id=asset_period_id, limit=limit)
            if data and len(data) > 0:
                logger.info(f"Successfully preloaded synthetic data for {asset}: {len(data)} data points")
                results[asset] = True
            else:
                logger.warning(f"Failed to preload synthetic data for {asset}")
                results[asset] = False
        except Exception as e:
            logger.error(f"Error preloading synthetic data for {asset}: {e}")
            results[asset] = False
    
    logger.info(f"Preloaded synthetic data for {sum(results.values())}/{len(assets)} assets")
    return results

if __name__ == "__main__":
    # Test the synthetic data providers
    test_data_providers() 