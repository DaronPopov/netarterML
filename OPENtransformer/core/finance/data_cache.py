import os
import json
import time
import logging
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta

# Configure logging
logger = logging.getLogger("data_cache")

class HistoricalDataCache:
    """
    A cache for historical financial data to avoid repeated API calls.
    
    This class provides functionality to:
    1. Cache historical data to disk
    2. Load cached data when available
    3. Check if cached data is still valid (not expired)
    4. Manage the cache directory structure
    """
    
    def __init__(self, cache_dir=None, expiry_hours=24):
        """
        Initialize the historical data cache.
        
        Args:
            cache_dir: Directory to store cached data (default: ~/.finlib/cache)
            expiry_hours: Number of hours before cached data expires (default: 24)
        """
        if cache_dir is None:
            # Default to ~/.finlib/cache
            home_dir = os.path.expanduser("~")
            cache_dir = os.path.join(home_dir, ".finlib", "cache")
        
        self.cache_dir = Path(cache_dir)
        self.expiry_hours = expiry_hours
        self.cache_stats = {
            "hits": 0,
            "misses": 0,
            "saves": 0,
            "errors": 0
        }
        
        # Create cache directory if it doesn't exist
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Create subdirectories for different asset types
        os.makedirs(os.path.join(self.cache_dir, "crypto"), exist_ok=True)
        os.makedirs(os.path.join(self.cache_dir, "stocks"), exist_ok=True)
        
        logger.info(f"Initialized historical data cache at {self.cache_dir}")
    
    def get_cache_path(self, asset_symbol, quote_currency, period_id, is_crypto):
        """
        Get the path to the cached data file.
        
        Args:
            asset_symbol: The asset symbol (e.g., 'BTC', 'AAPL')
            quote_currency: The quote currency (e.g., 'USD')
            period_id: The time period (e.g., '1MIN', '1D')
            is_crypto: Whether the asset is a cryptocurrency
            
        Returns:
            Path: Path to the cached data file
        """
        asset_type = "crypto" if is_crypto else "stocks"
        filename = f"{asset_symbol}_{quote_currency}_{period_id}.json"
        return self.cache_dir / asset_type / filename
    
    def is_cache_valid(self, cache_path):
        """
        Check if the cached data is still valid (not expired).
        
        Args:
            cache_path: Path to the cached data file
            
        Returns:
            bool: True if the cache is valid, False otherwise
        """
        if not cache_path.exists():
            return False
        
        # Check if the file was modified within the expiry period
        mtime = cache_path.stat().st_mtime
        mtime_datetime = datetime.fromtimestamp(mtime)
        expiry_time = datetime.now() - timedelta(hours=self.expiry_hours)
        
        return mtime_datetime > expiry_time
    
    def get_cached_data(self, asset_symbol, quote_currency, period_id, is_crypto):
        """
        Get cached historical data if available and valid.
        
        Args:
            asset_symbol: The asset symbol (e.g., 'BTC', 'AAPL')
            quote_currency: The quote currency (e.g., 'USD')
            period_id: The time period (e.g., '1MIN', '1D')
            is_crypto: Whether the asset is a cryptocurrency
            
        Returns:
            list: Cached historical data if available and valid, None otherwise
        """
        cache_path = self.get_cache_path(asset_symbol, quote_currency, period_id, is_crypto)
        
        if self.is_cache_valid(cache_path):
            try:
                with open(cache_path, 'r') as f:
                    data = json.load(f)
                
                self.cache_stats["hits"] += 1
                logger.info(f"Cache hit for {asset_symbol} ({period_id})")
                return data
            except Exception as e:
                self.cache_stats["errors"] += 1
                logger.warning(f"Error reading cache for {asset_symbol}: {e}")
                return None
        else:
            self.cache_stats["misses"] += 1
            logger.info(f"Cache miss for {asset_symbol} ({period_id})")
            return None
    
    def save_to_cache(self, asset_symbol, quote_currency, period_id, is_crypto, data):
        """
        Save historical data to the cache.
        
        Args:
            asset_symbol: The asset symbol (e.g., 'BTC', 'AAPL')
            quote_currency: The quote currency (e.g., 'USD')
            period_id: The time period (e.g., '1MIN', '1D')
            is_crypto: Whether the asset is a cryptocurrency
            data: The historical data to cache
            
        Returns:
            bool: True if the data was successfully cached, False otherwise
        """
        if not data:
            return False
        
        cache_path = self.get_cache_path(asset_symbol, quote_currency, period_id, is_crypto)
        
        try:
            with open(cache_path, 'w') as f:
                json.dump(data, f)
            
            self.cache_stats["saves"] += 1
            logger.info(f"Saved {asset_symbol} ({period_id}) data to cache")
            return True
        except Exception as e:
            self.cache_stats["errors"] += 1
            logger.warning(f"Error saving {asset_symbol} data to cache: {e}")
            return False
    
    # Add an alias for save_to_cache to handle the method name used in data_provider.py
    def cache_data(self, asset_symbol, quote_currency, period_id, is_crypto, data):
        """
        Cache historical data. This is an alias for save_to_cache.
        
        Args:
            asset_symbol: The asset symbol (e.g., 'BTC', 'AAPL')
            quote_currency: The quote currency (e.g., 'USD')
            period_id: The time period (e.g., '1MIN', '1D')
            is_crypto: Whether the asset is a cryptocurrency
            data: The historical data to cache
            
        Returns:
            bool: True if the data was successfully cached, False otherwise
        """
        return self.save_to_cache(asset_symbol, quote_currency, period_id, is_crypto, data)
    
    def preload_data(self, asset_symbol, quote_currency, period_id, is_crypto, data_fetcher):
        """
        Preload data into the cache if it's not already cached or is expired.
        
        Args:
            asset_symbol: The asset symbol (e.g., 'BTC', 'AAPL')
            quote_currency: The quote currency (e.g., 'USD')
            period_id: The time period (e.g., '1MIN', '1D')
            is_crypto: Whether the asset is a cryptocurrency
            data_fetcher: Function to fetch the data if not cached
            
        Returns:
            list: The historical data (either from cache or freshly fetched)
        """
        # First try to get from cache
        cached_data = self.get_cached_data(asset_symbol, quote_currency, period_id, is_crypto)
        
        if cached_data is not None:
            return cached_data
        
        # If not in cache or expired, fetch the data
        logger.info(f"Preloading data for {asset_symbol} ({period_id})")
        try:
            fresh_data = data_fetcher(asset_symbol, quote_currency, period_id)
            
            if fresh_data:
                # Save to cache
                self.save_to_cache(asset_symbol, quote_currency, period_id, is_crypto, fresh_data)
                return fresh_data
            else:
                logger.warning(f"Failed to preload data for {asset_symbol}")
                return None
        except Exception as e:
            logger.error(f"Error preloading data for {asset_symbol}: {e}")
            return None
    
    def get_cache_stats(self):
        """
        Get statistics about cache usage.
        
        Returns:
            dict: Cache statistics
        """
        total_requests = self.cache_stats["hits"] + self.cache_stats["misses"]
        hit_rate = (self.cache_stats["hits"] / total_requests * 100) if total_requests > 0 else 0
        
        stats = {
            **self.cache_stats,
            "hit_rate": hit_rate,
            "total_requests": total_requests
        }
        
        return stats
    
    def clear_expired_cache(self):
        """
        Clear expired cache entries.
        
        Returns:
            int: Number of cache entries cleared
        """
        cleared_count = 0
        
        for asset_type in ["crypto", "stocks"]:
            asset_dir = self.cache_dir / asset_type
            if not asset_dir.exists():
                continue
                
            for cache_file in asset_dir.glob("*.json"):
                if not self.is_cache_valid(cache_file):
                    try:
                        os.remove(cache_file)
                        cleared_count += 1
                        logger.info(f"Cleared expired cache: {cache_file.name}")
                    except Exception as e:
                        logger.warning(f"Error clearing cache {cache_file.name}: {e}")
        
        return cleared_count

# Global instance of the cache
_cache_instance = None

def get_cache_instance(cache_dir=None, expiry_hours=24):
    """
    Get the global cache instance (singleton pattern).
    
    Args:
        cache_dir: Directory to store cached data
        expiry_hours: Number of hours before cached data expires
        
    Returns:
        HistoricalDataCache: The global cache instance
    """
    global _cache_instance
    
    if _cache_instance is None:
        _cache_instance = HistoricalDataCache(cache_dir, expiry_hours)
    
    return _cache_instance 