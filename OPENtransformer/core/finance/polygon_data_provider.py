from polygon import RESTClient
from polygon.rest.models import Exchange
from typing import Dict, Optional, List
import logging
import datetime
import time
import threading
import random
import numpy as np
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("polygon_data_provider")

# Get API key from environment variables
POLYGON_API_KEY = os.getenv('POLYGON_API_KEY')
if not POLYGON_API_KEY:
    logger.warning("POLYGON_API_KEY not found in environment variables. Using default key.")
    POLYGON_API_KEY = "Eic9kerUsx5VfmfW4h2pJCZiZimkPWU8"

# Default values for assets in case API fails
DEFAULT_PRICES = {
    "BTC": 84000.0,
    "ETH": 3800.0,
    "SOL": 180.0,
    "AVAX": 40.0,
    "MATIC": 0.85,
    "LINK": 18.0,
    "DOT": 8.0,
    "ADA": 0.55,
    "DOGE": 0.15,
    "XRP": 0.60
}

DEFAULT_VOLATILITY = {
    "BTC": 0.02,  # 2% daily volatility
    "ETH": 0.025,
    "SOL": 0.035,
    "AVAX": 0.03,
    "MATIC": 0.04,
    "LINK": 0.03,
    "DOT": 0.035,
    "ADA": 0.04,
    "DOGE": 0.05,
    "XRP": 0.035
}

class PolygonDataProvider:
    """Data provider using Polygon.io REST API with daily OHLC data."""
    
    def __init__(self, api_key: str):
        """
        Initialize the Polygon data provider.
        
        Args:
            api_key: Polygon.io API key
        """
        self.api_key = api_key
        self.client = RESTClient(api_key)
        self.latest_prices = {}
        self.historical_data = {}
        self.running = False
        self.update_thread = None
        self.update_interval = 5.0  # Update every 5 seconds
        
        # For simulating intraday movements based on daily data
        self.last_update_time = {}
        self.daily_volatility = {}
        
        # Rate limiting
        self.last_api_call = 0
        self.min_api_call_interval = 0.5  # Minimum time between API calls (seconds)
        
        logger.info("PolygonDataProvider initialized with REST client")
    
    def start(self):
        """Start the price update thread."""
        if self.running:
            return
            
        # Initialize with default values first
        for asset, price in DEFAULT_PRICES.items():
            self.latest_prices[asset] = {
                "asset_id_base": asset,
                "asset_id_quote": "USD",
                "price": price,
                "time": datetime.datetime.now().isoformat()
            }
            self.last_update_time[asset] = time.time()
            self.daily_volatility[asset] = DEFAULT_VOLATILITY.get(asset, 0.02)
        
        # Try to load historical data in a separate thread to avoid blocking
        threading.Thread(target=self._load_initial_data, daemon=True).start()
        
        def update_prices():
            while self.running:
                try:
                    # Update prices for all assets in the latest_prices dictionary
                    for asset in list(self.latest_prices.keys()):
                        self._simulate_price_movement(asset)
                    time.sleep(self.update_interval)
                except Exception as e:
                    logger.error(f"Error updating prices: {e}")
                    time.sleep(1)  # Wait before retrying
        
        self.running = True
        self.update_thread = threading.Thread(target=update_prices)
        self.update_thread.daemon = True
        self.update_thread.start()
        logger.info("Polygon price update thread started")
    
    def _load_initial_data(self):
        """Load initial historical data for common assets."""
        # Preload historical data for common crypto assets
        common_assets = ["BTC", "ETH", "SOL", "AVAX", "MATIC"]
        for asset in common_assets:
            try:
                self._load_historical_data(asset)
                # Add delay between API calls to avoid rate limiting
                time.sleep(2)
            except Exception as e:
                logger.error(f"Error loading initial data for {asset}: {e}")
    
    def stop(self):
        """Stop the price update thread."""
        if self.running:
            self.running = False
            if self.update_thread:
                logger.info("Waiting for update thread to stop...")
                self.update_thread.join(timeout=5)
            logger.info("Polygon price update thread stopped")
    
    def _respect_rate_limit(self):
        """Respect rate limits by adding delay between API calls."""
        current_time = time.time()
        elapsed = current_time - self.last_api_call
        if elapsed < self.min_api_call_interval:
            sleep_time = self.min_api_call_interval - elapsed
            time.sleep(sleep_time)
        self.last_api_call = time.time()
    
    def _load_historical_data(self, asset: str):
        """
        Load historical data for an asset.
        
        Args:
            asset: Asset symbol (e.g., 'BTC')
        """
        try:
            # Format for crypto tickers
            ticker = f"X:{asset}USD"
            
            # Get data for the last 7 days (to avoid hitting rate limits)
            end_date = datetime.datetime.now().strftime("%Y-%m-%d")
            start_date = (datetime.datetime.now() - datetime.timedelta(days=7)).strftime("%Y-%m-%d")
            
            logger.info(f"Loading historical data for {asset} from {start_date} to {end_date}")
            
            # Respect rate limits
            self._respect_rate_limit()
            
            # Get aggregates (daily bars)
            aggs = self.client.get_aggs(
                ticker=ticker,
                multiplier=1,
                timespan="day",
                from_=start_date,
                to=end_date
            )
            
            # Convert to our format
            result = []
            for agg in aggs:
                result.append({
                    "asset_id_base": asset,
                    "asset_id_quote": "USD",
                    "time": datetime.datetime.fromtimestamp(agg.timestamp / 1000.0).isoformat(),
                    "open": agg.open,
                    "high": agg.high,
                    "low": agg.low,
                    "close": agg.close,
                    "volume": agg.volume
                })
            
            # Store historical data
            self.historical_data[asset] = result
            
            # Calculate daily volatility (as percentage)
            if result:
                daily_returns = []
                for i in range(1, len(result)):
                    daily_return = (result[i]['close'] - result[i-1]['close']) / result[i-1]['close']
                    daily_returns.append(daily_return)
                
                if daily_returns:
                    # Calculate standard deviation of daily returns
                    self.daily_volatility[asset] = np.std(daily_returns)
                    logger.info(f"{asset} daily volatility: {self.daily_volatility[asset]:.2%}")
                else:
                    # Use default volatility
                    self.daily_volatility[asset] = DEFAULT_VOLATILITY.get(asset, 0.02)
                
                # Initialize latest price with the most recent close
                self.latest_prices[asset] = {
                    "asset_id_base": asset,
                    "asset_id_quote": "USD",
                    "price": result[-1]['close'],
                    "time": datetime.datetime.now().isoformat()
                }
                self.last_update_time[asset] = time.time()
                
                logger.info(f"Loaded {len(result)} days of historical data for {asset}")
                logger.info(f"Latest price for {asset}: ${result[-1]['close']:.2f}")
            else:
                logger.warning(f"No historical data found for {asset}, using default values")
                # Use default values
                self.latest_prices[asset] = {
                    "asset_id_base": asset,
                    "asset_id_quote": "USD",
                    "price": DEFAULT_PRICES.get(asset, 1000.0),
                    "time": datetime.datetime.now().isoformat()
                }
                self.daily_volatility[asset] = DEFAULT_VOLATILITY.get(asset, 0.02)
        except Exception as e:
            logger.error(f"Error loading historical data for {asset}: {e}")
            # Use default values
            self.latest_prices[asset] = {
                "asset_id_base": asset,
                "asset_id_quote": "USD",
                "price": DEFAULT_PRICES.get(asset, 1000.0),
                "time": datetime.datetime.now().isoformat()
            }
            self.daily_volatility[asset] = DEFAULT_VOLATILITY.get(asset, 0.02)
    
    def _simulate_price_movement(self, asset: str):
        """
        Simulate intraday price movement based on historical volatility.
        
        Args:
            asset: Asset symbol (e.g., 'BTC')
        """
        if asset not in self.latest_prices:
            # Use default values
            self.latest_prices[asset] = {
                "asset_id_base": asset,
                "asset_id_quote": "USD",
                "price": DEFAULT_PRICES.get(asset, 1000.0),
                "time": datetime.datetime.now().isoformat()
            }
            self.daily_volatility[asset] = DEFAULT_VOLATILITY.get(asset, 0.02)
            self.last_update_time[asset] = time.time()
            return
        
        if asset not in self.daily_volatility:
            # Default volatility if we couldn't calculate from historical data
            self.daily_volatility[asset] = DEFAULT_VOLATILITY.get(asset, 0.02)
        
        current_time = time.time()
        if asset not in self.last_update_time:
            self.last_update_time[asset] = current_time
        
        # Calculate time elapsed since last update (in days)
        time_elapsed_days = (current_time - self.last_update_time[asset]) / (24 * 60 * 60)
        
        # Scale volatility by square root of time (random walk property)
        scaled_volatility = self.daily_volatility[asset] * (time_elapsed_days ** 0.5)
        
        # Generate random price movement
        # Use a normal distribution for price changes
        price_change_pct = random.normalvariate(0, scaled_volatility)
        
        # Add some mean reversion and momentum
        if asset in self.historical_data and self.historical_data[asset]:
            latest_historical = self.historical_data[asset][-1]['close']
            current_price = self.latest_prices[asset]['price']
            
            # Mean reversion component (pull towards the latest daily close)
            mean_reversion = 0.1 * (latest_historical - current_price) / current_price
            
            # Momentum component (random direction with persistence)
            momentum = 0.05 * random.choice([-1, 1]) * scaled_volatility
            
            # Combine components
            price_change_pct = price_change_pct + mean_reversion + momentum
        
        # Update price
        current_price = self.latest_prices[asset]['price']
        new_price = current_price * (1 + price_change_pct)
        
        # Update the price in our cache
        self.latest_prices[asset] = {
            "asset_id_base": asset,
            "asset_id_quote": "USD",
            "price": new_price,
            "time": datetime.datetime.now().isoformat()
        }
        
        # Update last update time
        self.last_update_time[asset] = current_time
        
        logger.debug(f"Updated price for {asset}: ${new_price:.2f} (change: {price_change_pct:.4%})")
    
    def get_latest_price(self, asset: str) -> Optional[Dict]:
        """
        Get the latest price for an asset.
        
        Args:
            asset: Asset symbol (e.g., 'BTC')
            
        Returns:
            dict: Latest price data or None if not available
        """
        # If we don't have this asset yet, initialize with default values
        if asset not in self.latest_prices:
            self.latest_prices[asset] = {
                "asset_id_base": asset,
                "asset_id_quote": "USD",
                "price": DEFAULT_PRICES.get(asset, 1000.0),
                "time": datetime.datetime.now().isoformat()
            }
            self.daily_volatility[asset] = DEFAULT_VOLATILITY.get(asset, 0.02)
            self.last_update_time[asset] = time.time()
            
            # Try to load historical data in background
            threading.Thread(target=lambda: self._load_historical_data(asset), daemon=True).start()
            
        # Simulate a price movement
        self._simulate_price_movement(asset)
            
        return self.latest_prices.get(asset)
    
    def get_historical_data(self, asset: str, from_date: str, to_date: str) -> List[Dict]:
        """
        Get historical data for an asset.
        
        Args:
            asset: Asset symbol (e.g., 'BTC')
            from_date: Start date in ISO format (YYYY-MM-DD)
            to_date: End date in ISO format (YYYY-MM-DD)
            
        Returns:
            list: List of historical price data
        """
        # If we don't have this asset yet, load its historical data
        if asset not in self.historical_data:
            try:
                self._load_historical_data(asset)
            except Exception as e:
                logger.error(f"Error loading historical data for {asset}: {e}")
                # Return empty list if we can't load historical data
                return []
            
        # Filter historical data by date range
        result = []
        for data in self.historical_data.get(asset, []):
            data_date = data['time'].split('T')[0]
            if from_date <= data_date <= to_date:
                result.append(data)
                
        return result

# Global instance
_polygon_provider = None

def initialize_polygon(api_key: str):
    """Initialize the global Polygon data provider instance."""
    global _polygon_provider
    if _polygon_provider is None:
        _polygon_provider = PolygonDataProvider(api_key)
        _polygon_provider.start()
    return _polygon_provider

def get_polygon_provider() -> Optional[PolygonDataProvider]:
    """Get the global Polygon data provider instance."""
    return _polygon_provider

def fetch_realtime_data(asset: str, quote_currency: str = "USD") -> Optional[Dict]:
    """
    Fetch simulated real-time data for an asset based on historical data from Polygon.
    
    Args:
        asset: Asset symbol
        quote_currency: Quote currency (only USD supported for now)
        
    Returns:
        dict: Simulated real-time price data or None if not available
    """
    provider = get_polygon_provider()
    if provider is None:
        raise RuntimeError("Polygon data provider not initialized")
    
    # Get the latest price
    return provider.get_latest_price(asset)