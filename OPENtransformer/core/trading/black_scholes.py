from finlib_extensions import ExtendedStocksAPI
import numpy as np
import requests
import time
import logging
import os
import pickle
import random
from datetime import datetime
from typing import List, Dict, Any, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s.%(msecs)03d - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("black_scholes")

# Initialize ExtendedStocksAPI instead of StocksAPI
api = ExtendedStocksAPI()

# Cache directory
CACHE_DIR = os.environ.get("CRYPTO_CACHE_DIR", "./data_cache")
if not os.path.exists(CACHE_DIR):
    os.makedirs(CACHE_DIR)

# Rate limiting variables
last_api_call_time = 0
min_request_interval = 0.1  # Minimum seconds between API calls (reduced from 1.5)

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

def _api_request(url, params=None, max_retries=3):
    """
    Make an API request with rate limiting and retries
    
    Args:
        url: API endpoint URL
        params: Query parameters
        max_retries: Maximum number of retries
        
    Returns:
        JSON response or None if failed
    """
    global last_api_call_time
    request_start_time = time.time()
    
    if params is None:
        params = {}
        
    # Implement rate limiting
    current_time = time.time()
    time_since_last_call = current_time - last_api_call_time
    
    if time_since_last_call < min_request_interval:
        sleep_time = min_request_interval - time_since_last_call
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
            last_api_call_time = time.time()  # Update last call time
            
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

def get_crypto_data(tickers: List[str], days: int = 365, use_cache: bool = True) -> Dict[str, np.ndarray]:
    """
    Get historical cryptocurrency data from CoinGecko API with caching
    
    Args:
        tickers: List of cryptocurrency IDs
        days: Number of days of historical data
        use_cache: Whether to use cached data if available
        
    Returns:
        Dictionary mapping cryptocurrency IDs to price arrays
    """
    with Timer(f"get_crypto_data for {len(tickers)} tickers"):
        historical_data = {}
        tickers_to_fetch = []
        
        # Limit the number of tickers to avoid excessive API calls
        if len(tickers) > 5:
            logger.warning(f"Limiting historical data fetch to 5 cryptocurrencies to avoid rate limiting")
            tickers = tickers[:5]
        
        # Check cache first if use_cache is True
        if use_cache:
            cache_file = os.path.join(CACHE_DIR, "crypto_historical_data.pkl")
            last_update_file = os.path.join(CACHE_DIR, "crypto_last_update.pkl")
            
            cached_data = {}
            last_update = {}
            
            # Load cached data if available
            with Timer("Loading cached data"):
                if os.path.exists(cache_file) and os.path.exists(last_update_file):
                    try:
                        with open(cache_file, "rb") as f:
                            cached_data = pickle.load(f)
                        
                        with open(last_update_file, "rb") as f:
                            last_update = pickle.load(f)
                        
                        logger.info(f"Loaded cached data for {len(cached_data)} cryptocurrencies")
                        
                        # Check which tickers we have cached data for
                        for ticker in tickers:
                            if ticker in cached_data:
                                # Check if the cached data is less than 24 hours old
                                if ticker in last_update:
                                    cache_age = (datetime.now() - last_update[ticker]).total_seconds() / 3600
                                    
                                    if cache_age < 24:  # Use cache if less than 24 hours old
                                        historical_data[ticker] = cached_data[ticker]
                                        logger.info(f"Using cached data for {ticker} ({cache_age:.1f} hours old)")
                                        continue
                                
                                # Cache is too old or no timestamp, need to fetch
                                tickers_to_fetch.append(ticker)
                            else:
                                # No cached data, need to fetch
                                tickers_to_fetch.append(ticker)
                    except Exception as e:
                        logger.error(f"Error loading cached data: {e}")
                        tickers_to_fetch = tickers
                else:
                    # No cache files, need to fetch all tickers
                    tickers_to_fetch = tickers
        else:
            # Not using cache, fetch all tickers
            tickers_to_fetch = tickers
        
        # Fetch data for tickers not in cache
        if tickers_to_fetch:
            logger.info(f"Fetching data for {len(tickers_to_fetch)} cryptocurrencies")
            
            base_url = "https://api.coingecko.com/api/v3"
            
            with Timer(f"Fetching data for {len(tickers_to_fetch)} tickers"):
                for ticker in tickers_to_fetch:
                    url = f"{base_url}/coins/{ticker}/market_chart"
                    params = {
                        "vs_currency": "usd",
                        "days": days,
                        "interval": "daily"
                    }
                    
                    data = _api_request(url, params)
                    
                    if data and "prices" in data:
                        prices = np.array([price[1] for price in data["prices"]], dtype=np.float32)
                        historical_data[ticker] = prices
                        
                        # Update cache
                        if use_cache:
                            cached_data[ticker] = prices
                            last_update[ticker] = datetime.now()
                        
                        logger.info(f"Successfully fetched data for {ticker} ({len(prices)} data points)")
                    else:
                        logger.error(f"Failed to fetch data for {ticker}")
            
            # Save updated cache
            if use_cache and tickers_to_fetch:
                with Timer("Saving cache"):
                    try:
                        with open(cache_file, "wb") as f:
                            pickle.dump(cached_data, f)
                        
                        with open(last_update_file, "wb") as f:
                            pickle.dump(last_update, f)
                        
                        logger.info(f"Saved cached data for {len(cached_data)} cryptocurrencies")
                    except Exception as e:
                        logger.error(f"Error saving cached data: {e}")
        
        return historical_data

def format_data_for_analysis(historical_data: Dict[str, np.ndarray]) -> np.ndarray:
    """
    Format the historical data for analysis with StocksAPI
    
    Args:
        historical_data: Dictionary mapping cryptocurrency IDs to price arrays
        
    Returns:
        Numpy array with shape (days, assets) containing price data
    """
    with Timer("format_data_for_analysis"):
        # Find the minimum length across all price arrays
        min_length = min(len(prices) for prices in historical_data.values())
        
        # Create a 2D array with shape (days, assets)
        tickers = list(historical_data.keys())
        prices_array = np.zeros((min_length, len(tickers)), dtype=np.float32)
        
        for i, ticker in enumerate(tickers):
            prices_array[:, i] = historical_data[ticker][-min_length:]
        
        return prices_array

def compute_daily_returns(prices: np.ndarray) -> np.ndarray:
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

def compute_black_scholes_var(tickers: List[str], num_simulations: int = 1000000, use_cache: bool = True) -> Dict[str, float]:
    """
    Compute Black-Scholes Value at Risk (VaR) for a list of cryptocurrencies
    
    Args:
        tickers: List of cryptocurrency IDs
        num_simulations: Number of Monte Carlo simulations
        use_cache: Whether to use cached data if available
        
    Returns:
        Dictionary mapping cryptocurrency IDs to VaR values
    """
    start_time = time.time()
    logger.info(f"Starting Black-Scholes VaR calculation for {len(tickers)} tickers")
    
    # Limit the number of tickers to avoid excessive computation
    if len(tickers) > 10:
        logger.warning(f"Limiting Black-Scholes VaR calculation to 10 cryptocurrencies")
        tickers = tickers[:10]
    
    # Check if we have cached VaR results
    var_cache_file = os.path.join(CACHE_DIR, "bs_var_cache.pkl")
    var_cache_timestamp_file = os.path.join(CACHE_DIR, "bs_var_timestamp.pkl")
    
    var_dict = {}
    cached_var = {}
    cached_timestamp = {}
    tickers_to_compute = []
    
    # Try to load cached VaR results
    with Timer("Loading cached VaR results"):
        if use_cache and os.path.exists(var_cache_file) and os.path.exists(var_cache_timestamp_file):
            try:
                with open(var_cache_file, "rb") as f:
                    cached_var = pickle.load(f)
                
                with open(var_cache_timestamp_file, "rb") as f:
                    cached_timestamp = pickle.load(f)
                
                logger.info(f"Loaded cached VaR results for {len(cached_var)} cryptocurrencies")
                
                # Check which tickers we have cached VaR results for
                for ticker in tickers:
                    if ticker in cached_var and ticker in cached_timestamp:
                        # Check if the cached VaR is less than 24 hours old
                        cache_age = (datetime.now() - cached_timestamp[ticker]).total_seconds() / 3600
                        
                        if cache_age < 24:  # Use cache if less than 24 hours old
                            var_dict[ticker] = cached_var[ticker]
                            logger.info(f"Using cached VaR for {ticker} ({cache_age:.1f} hours old)")
                            continue
                    
                    # Cache is too old or no timestamp, need to compute
                    tickers_to_compute.append(ticker)
            except Exception as e:
                logger.error(f"Error loading cached VaR results: {e}")
                tickers_to_compute = tickers
        else:
            # No cache files, need to compute all tickers
            tickers_to_compute = tickers
    
    # Compute VaR for tickers not in cache
    if tickers_to_compute:
        logger.info(f"Computing VaR for {len(tickers_to_compute)} cryptocurrencies")
        
        # Get historical data for the tickers
        logger.info(f"Fetching historical data for {len(tickers_to_compute)} cryptocurrencies...")
        historical_data = get_crypto_data(tickers=tickers_to_compute, days=365, use_cache=use_cache)
        
        if not historical_data:
            logger.error("Failed to fetch historical data")
            return var_dict  # Return any cached results we have
        
        # Format data for analysis
        logger.info("Formatting data for analysis...")
        prices = format_data_for_analysis(historical_data)
        
        # Compute daily returns
        logger.info("Computing daily returns...")
        returns = compute_daily_returns(prices)
        
        # Compute Black-Scholes VaR
        logger.info(f"Computing Black-Scholes VaR with {num_simulations} simulations...")
        
        with Timer(f"Black-Scholes VaR calculation for {len(tickers_to_compute)} tickers"):
            # Calculate mean and standard deviation of returns
            mu = np.mean(returns, axis=0)
            sigma = np.std(returns, axis=0)
            
            # Calculate VaR for each asset
            for i, ticker in enumerate(tickers_to_compute):
                ticker_start_time = time.time()
                
                if i >= len(mu) or i >= len(sigma):
                    logger.error(f"Index out of bounds for ticker {ticker}. Skipping.")
                    continue
                    
                # Generate random returns using Black-Scholes model
                with Timer(f"Monte Carlo simulation for {ticker}"):
                    random_returns = np.random.normal(
                        mu[i],
                        sigma[i],
                        num_simulations
                    ).astype(np.float32)
                
                # Sort returns
                with Timer(f"Sorting returns for {ticker}"):
                    sorted_returns = np.sort(random_returns)
                
                # Calculate VaR at 95% confidence level
                var_index = int(0.05 * num_simulations)
                var = -sorted_returns[var_index]
                
                var_dict[ticker] = float(var)
                
                # Update cache
                if use_cache:
                    cached_var[ticker] = float(var)
                    cached_timestamp[ticker] = datetime.now()
                
                ticker_end_time = time.time()
                ticker_duration = ticker_end_time - ticker_start_time
                logger.info(f"TIMING: VaR calculation for {ticker} completed in {ticker_duration:.3f}s ({int(ticker_duration * 1000)}ms)")
        
        # Save updated cache
        if use_cache and tickers_to_compute:
            with Timer("Saving VaR cache"):
                try:
                    with open(var_cache_file, "wb") as f:
                        pickle.dump(cached_var, f)
                    
                    with open(var_cache_timestamp_file, "wb") as f:
                        pickle.dump(cached_timestamp, f)
                    
                    logger.info(f"Saved cached VaR results for {len(cached_var)} cryptocurrencies")
                except Exception as e:
                    logger.error(f"Error saving cached VaR results: {e}")
    
    end_time = time.time()
    total_time = end_time - start_time
    logger.info(f"TIMING: Total Black-Scholes VaR calculation completed in {total_time:.3f}s ({int(total_time * 1000)}ms)")
    
    return var_dict

def main():
    """
    Main function to demonstrate Black-Scholes VaR calculation
    """
    start_time = time.time()
    logger.info(f"Starting Black-Scholes VaR calculation at {datetime.now().isoformat()}")
    
    # List of top 5 cryptocurrencies by market cap (reduced from 10 to 5)
    top_cryptos = [
        "bitcoin",
        "ethereum",
        "tether",
        "binancecoin",
        "solana"
    ]
    
    # Compute Black-Scholes VaR
    logger.info("Computing Black-Scholes VaR for top 5 cryptocurrencies...")
    var_results = compute_black_scholes_var(top_cryptos, num_simulations=100000)  # Reduced simulations
    
    # Print results
    logger.info("Black-Scholes VaR (95% confidence level):")
    for ticker, var in var_results.items():
        logger.info(f"{ticker.upper()}: {var:.2%}")
    
    # Generate trading signals based on VaR
    logger.info("Generating trading signals based on VaR:")
    for ticker, var in var_results.items():
        if var > 0.05:  # High risk
            signal = "SELL"
        elif var < 0.02:  # Low risk
            signal = "BUY"
        else:  # Medium risk
            signal = "HOLD"
        
        logger.info(f"{ticker.upper()}: {signal} (VaR: {var:.2%})")
    
    end_time = time.time()
    total_time = end_time - start_time
    logger.info(f"TIMING: Total execution time: {total_time:.3f}s ({int(total_time * 1000)}ms)")

if __name__ == "__main__":
    main()





