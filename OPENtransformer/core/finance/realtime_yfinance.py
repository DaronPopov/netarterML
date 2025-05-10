import time
import logging
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
from typing import Dict, List, Any, Optional

from finlib.finance.realtime import RealTimeAlgorithm
from finlib.finance.data_provider import STOCK_ASSETS, CRYPTO_ASSETS

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("realtime_yfinance")

class YFinanceRealtimeAlgorithm(RealTimeAlgorithm):
    """
    Algorithm that fetches real-time data from Yahoo Finance and computes
    technical indicators for display on the dashboard.
    """
    
    def __init__(self, short_window: int = 10, long_window: int = 30, name: str = "YFinanceRealtime"):
        super().__init__(name)
        self.short_window = short_window
        self.long_window = long_window
        self.last_fetch_time = {}
        self.cache = {}
        # Set cache timeout to 300 seconds (5 minutes) to avoid hitting API rate limits
        self.cache_timeout = 300
        # Add retry mechanism properties
        self.max_retries = 3
        self.retry_delay = 5  # seconds
        
    def fetch_latest_data(self, asset: str) -> Optional[Dict[str, Any]]:
        """Fetch the latest data from yfinance"""
        current_time = time.time()
        
        # Only fetch new data if it's been more than cache_timeout seconds since last fetch
        if asset in self.last_fetch_time and current_time - self.last_fetch_time[asset] < self.cache_timeout:
            if asset in self.cache:
                logger.info(f"Using cached data for {asset} (age: {current_time - self.last_fetch_time[asset]:.1f}s)")
                # Update the timestamp to show current time
                if self.cache[asset]:
                    self.cache[asset]['timestamp'] = datetime.now().isoformat()
                return self.cache[asset]
            return None
            
        # Format symbol for yahoo finance (add USD suffix for crypto)
        ticker_symbol = asset
        if asset in CRYPTO_ASSETS:
            ticker_symbol = f"{asset}-USD"
        
        # Implement retry logic    
        for retry in range(self.max_retries):
            try:
                logger.info(f"Fetching new data from Yahoo Finance for {asset} (attempt {retry+1}/{self.max_retries})")
                # Fetch data from yfinance
                ticker = yf.Ticker(ticker_symbol)
                data = ticker.history(period="1d", interval="1m")
                
                if data.empty:
                    logger.warning(f"No data returned from Yahoo Finance for {ticker_symbol}")
                    if retry < self.max_retries - 1:
                        logger.info(f"Retrying in {self.retry_delay} seconds...")
                        time.sleep(self.retry_delay)
                        continue
                    return None
                    
                # Extract latest price and compute basic metrics
                latest = data.iloc[-1]
                close_price = latest['Close']
                open_price = data.iloc[0]['Open']
                high_price = data['High'].max()
                low_price = data['Low'].min()
                volume = int(latest['Volume'])
                
                # Calculate percent change
                day_change = close_price - open_price
                percent_change = (day_change / open_price) * 100
                
                # Compute RSI if we have enough data
                rsi = None
                if len(data) > 14:
                    delta = data['Close'].diff()
                    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
                    loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
                    # Handle division by zero
                    rs = gain / loss.replace(0, np.nan).fillna(1)
                    rsi = 100 - (100 / (1 + rs)).iloc[-1]
                
                # Compute moving averages
                short_ma = data['Close'].rolling(window=self.short_window).mean().iloc[-1] if len(data) >= self.short_window else None
                long_ma = data['Close'].rolling(window=self.long_window).mean().iloc[-1] if len(data) >= self.long_window else None
                
                # Determine trend signal based on moving averages
                signal = "Hold"
                if short_ma is not None and long_ma is not None:
                    if short_ma > long_ma:
                        signal = "Buy"
                    elif short_ma < long_ma:
                        signal = "Sell"
                
                # Store results
                result = {
                    'asset': asset,
                    'price': close_price,
                    'open': open_price,
                    'high': high_price,
                    'low': low_price,
                    'volume': volume,
                    'change': day_change,
                    'percent_change': percent_change,
                    'rsi': rsi,
                    'short_ma': short_ma,
                    'long_ma': long_ma,
                    'signal': signal,
                    'timestamp': datetime.now().isoformat(),
                    'data_source': 'yfinance'
                }
                
                # Add volatility calculation
                result['volatility'] = (high_price - low_price) / close_price * 100
                
                # Update cache and last fetch time
                self.cache[asset] = result
                self.last_fetch_time[asset] = current_time
                
                # Successfully fetched data, no need to retry
                return result
                
            except Exception as e:
                logger.error(f"Error fetching data from Yahoo Finance for {asset}: {e}")
                if retry < self.max_retries - 1:
                    logger.info(f"Retrying in {self.retry_delay} seconds...")
                    time.sleep(self.retry_delay)
                else:
                    logger.warning(f"All {self.max_retries} attempts failed for {asset}")
            
        # If we get here, all retries failed
        # If we have cached data, use it even if it's expired
        if asset in self.cache and self.cache[asset]:
            logger.info(f"Using expired cache for {asset} after {self.max_retries} failed attempts")
            # Mark as using expired data
            self.cache[asset]['data_source'] = 'yfinance_cached'
            # Update the timestamp to show current time
            self.cache[asset]['timestamp'] = datetime.now().isoformat()
            return self.cache[asset]
            
        return None
    
    def process(self, asset: str, realtime_data: Dict[str, Any], historical_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Process real-time and historical data for an asset.
        
        Args:
            asset: Asset symbol being processed
            realtime_data: Current real-time data for the asset
            historical_data: Historical data for the asset
            
        Returns:
            dict: Results of the algorithm
        """
        # Fetch the latest data directly from yfinance
        result = self.fetch_latest_data(asset)
        
        if not result:
            # If we couldn't get data from yfinance, use the provided realtime_data
            if realtime_data and 'rate' in realtime_data:
                price = realtime_data['rate']
                result = {
                    'asset': asset,
                    'price': price,
                    'open': price,  # Fallback
                    'high': price,  # Fallback
                    'low': price,   # Fallback
                    'volume': 0,    # Fallback
                    'change': 0,    # Fallback
                    'percent_change': 0,  # Fallback
                    'signal': "Hold",  # Fallback
                    'timestamp': datetime.now().isoformat(),
                    'data_source': 'realtime_fallback'
                }
            else:
                # No data available, create synthetic data
                base_price = 100
                if asset in CRYPTO_ASSETS:
                    if asset == 'BTC':
                        base_price = 80000
                    elif asset == 'ETH':
                        base_price = 1800
                    else:
                        base_price = 100
                elif asset in STOCK_ASSETS:
                    if asset == 'AAPL':
                        base_price = 170
                    elif asset == 'MSFT':
                        base_price = 400
                    elif asset == 'GOOGL':
                        base_price = 150
                    else:
                        base_price = 100
                        
                # Add some randomness
                price = base_price * (1 + (np.random.random() - 0.5) * 0.01)
                
                # Return synthetic data
                result = {
                    'asset': asset,
                    'price': price,
                    'open': price * 0.995,
                    'high': price * 1.005,
                    'low': price * 0.99,
                    'volume': int(np.random.random() * 10000),
                    'change': price * 0.005,
                    'percent_change': 0.5,
                    'rsi': 50,
                    'signal': "Hold",
                    'timestamp': datetime.now().isoformat(),
                    'data_source': 'synthetic'
                }
        
        # Calculate additional analytics
        # Simple volatility estimation based on high-low range
        if 'high' in result and 'low' in result and 'price' in result:
            result['volatility'] = (result['high'] - result['low']) / result['price'] * 100
            
        # Add a confidence metric based on RSI
        if 'rsi' in result and result['rsi'] is not None:
            rsi = result['rsi']
            # RSI above 70 is overbought, below 30 is oversold
            if rsi > 70:
                result['confidence'] = 0.8  # High confidence for selling
                result['market_condition'] = "Overbought"
            elif rsi < 30:
                result['confidence'] = 0.8  # High confidence for buying
                result['market_condition'] = "Oversold"
            else:
                result['confidence'] = 0.5  # Moderate confidence
                result['market_condition'] = "Neutral"
        else:
            result['confidence'] = 0.5  # Default
            result['market_condition'] = "Unknown"
        
        # Add time since last update
        result['seconds_since_update'] = time.time() - self.last_fetch_time.get(asset, time.time())
        
        # Track successful algorithm run
        self.run_count += 1
        self.last_run_time = time.time()
        self.last_results = result
        
        return result


# Helper functions
def register_yfinance_algorithm():
    """Register the YFinance algorithm with the monitoring system"""
    from finlib.finance.realtime import register_algorithm
    algorithm = YFinanceRealtimeAlgorithm()
    register_algorithm(algorithm)
    logger.info(f"Registered {algorithm.name} for real-time processing")
    return algorithm 