import requests
from finlib.APIS.stocks_api import StocksAPI
import numpy as np
import time
import datetime
import pandas as pd
import matplotlib.pyplot as plt
from finlib.finance.data_provider import (
    fetch_realtime_data, 
    fetch_historical_data, 
    analyze_buy_sell,
    CRYPTO_ASSETS,
    STOCK_ASSETS
)
import logging
import inspect
import traceback
import threading
from typing import Dict, List, Any, Callable, Optional, Union

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("realtime")

# Define a base class for custom algorithms
class RealTimeAlgorithm:
    """Base class for custom real-time data processing algorithms."""
    
    def __init__(self, name: str = None):
        """
        Initialize the algorithm.
        
        Args:
            name: Optional custom name for the algorithm
        """
        self.name = name or self.__class__.__name__
        self.enabled = True
        self.last_run_time = 0
        self.run_count = 0
        self.error_count = 0
        self.last_results = None
    
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
        raise NotImplementedError("Subclasses must implement process()")
    
    def on_error(self, asset: str, error: Exception) -> None:
        """
        Called when an error occurs during processing.
        
        Args:
            asset: Asset symbol that was being processed
            error: The exception that was raised
        """
        self.error_count += 1
        logger.error(f"Error in algorithm {self.name} for {asset}: {error}")
        logger.debug(traceback.format_exc())

# Registry for custom algorithms
registered_algorithms: Dict[str, RealTimeAlgorithm] = {}

def register_algorithm(algorithm: RealTimeAlgorithm) -> None:
    """
    Register a custom algorithm for real-time data processing.
    
    Args:
        algorithm: The algorithm instance to register
    """
    registered_algorithms[algorithm.name] = algorithm
    logger.info(f"Registered real-time algorithm: {algorithm.name}")

def unregister_algorithm(name: str) -> bool:
    """
    Unregister a custom algorithm.
    
    Args:
        name: Name of the algorithm to unregister
        
    Returns:
        bool: True if the algorithm was unregistered, False if it wasn't found
    """
    if name in registered_algorithms:
        del registered_algorithms[name]
        logger.info(f"Unregistered real-time algorithm: {name}")
        return True
    return False

def get_algorithm(name: str) -> Optional[RealTimeAlgorithm]:
    """
    Get a registered algorithm by name.
    
    Args:
        name: Name of the algorithm to retrieve
        
    Returns:
        RealTimeAlgorithm: The algorithm instance, or None if not found
    """
    return registered_algorithms.get(name)

def get_registered_algorithms() -> Dict[str, RealTimeAlgorithm]:
    """
    Get all registered algorithms.
    
    Returns:
        dict: Dictionary of algorithm name to algorithm instance
    """
    return registered_algorithms.copy()

# Global store for algorithm results
algorithm_results: Dict[str, Dict[str, Any]] = {}

# Thread lock for updating algorithm results
results_lock = threading.Lock()

def run_algorithms(asset: str, realtime_data: Dict[str, Any], historical_data: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """
    Run all registered algorithms for an asset.
    
    Args:
        asset: Asset symbol being processed
        realtime_data: Current real-time data for the asset
        historical_data: Historical data for the asset
        
    Returns:
        dict: Results from all algorithms for this asset
    """
    asset_results = {}
    
    for name, algorithm in registered_algorithms.items():
        if not algorithm.enabled:
            continue
            
        try:
            start_time = time.time()
            results = algorithm.process(asset, realtime_data, historical_data)
            run_time = time.time() - start_time
            
            algorithm.last_run_time = run_time
            algorithm.run_count += 1
            algorithm.last_results = results
            
            asset_results[name] = {
                "results": results,
                "run_time": run_time,
                "timestamp": datetime.datetime.now().isoformat()
            }
            
            logger.debug(f"Algorithm {name} for {asset} completed in {run_time:.4f}s")
        except Exception as e:
            algorithm.on_error(asset, e)
    
    # Update global results store
    with results_lock:
        algorithm_results[asset] = asset_results
    
    return asset_results

def get_algorithm_results(asset: Optional[str] = None) -> Union[Dict[str, Dict[str, Any]], Dict[str, Any]]:
    """
    Get results from algorithms.
    
    Args:
        asset: Optional asset symbol to get results for, or None for all assets
        
    Returns:
        dict: Results for the specified asset or all assets
    """
    with results_lock:
        if asset is not None:
            return algorithm_results.get(asset, {})
        return algorithm_results.copy()

# Sample implementation of a simple algorithm
class SimpleMovingAverageAlgorithm(RealTimeAlgorithm):
    """Simple algorithm that calculates moving averages."""
    
    def __init__(self, short_window: int = 5, long_window: int = 20):
        """
        Initialize with window sizes.
        
        Args:
            short_window: Short moving average window size
            long_window: Long moving average window size
        """
        super().__init__("SimpleMovingAverage")
        self.short_window = short_window
        self.long_window = long_window
    
    def process(self, asset: str, realtime_data: Dict[str, Any], historical_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate short and long moving averages."""
        if not historical_data or len(historical_data) < self.long_window:
            return {"error": "Insufficient historical data"}
            
        # Extract price data
        if 'rate_close' in historical_data[0]:
            prices = [item['rate_close'] for item in historical_data]
        elif 'close' in historical_data[0]:
            prices = [item['close'] for item in historical_data]
        else:
            return {"error": "Unsupported data format"}
            
        # Calculate moving averages
        short_ma = sum(prices[-self.short_window:]) / self.short_window
        long_ma = sum(prices[-self.long_window:]) / self.long_window
        
        # Get current price
        if 'rate' in realtime_data:
            current_price = realtime_data['rate']
        elif 'price' in realtime_data:
            current_price = realtime_data['price']
        else:
            current_price = prices[-1]
            
        # Determine signal
        signal = "HOLD"
        if short_ma > long_ma:
            signal = "BUY"
        elif short_ma < long_ma:
            signal = "SELL"
            
        return {
            "current_price": current_price,
            "short_ma": short_ma,
            "long_ma": long_ma,
            "signal": signal,
            "short_window": self.short_window,
            "long_window": self.long_window
        }

# Register the default algorithm
register_algorithm(SimpleMovingAverageAlgorithm())

class RealTimeMonitor:
    """Class for monitoring real-time data and running algorithms."""
    
    def __init__(self, callback: Optional[Callable[[str, str, Dict[str, Any], Optional[str]], None]] = None):
        """
        Initialize the monitor.
        
        Args:
            callback: Optional callback function to be called when algorithm results are available.
                     The callback receives (asset, algorithm_name, results, error).
        """
        self.assets = set()
        self.running = False
        self.thread = None
        self.callback = callback
        self.lock = threading.Lock()
        
        # Initialize StocksAPI
        self.stocks_api = StocksAPI()
        
        logger.info("RealTimeMonitor initialized")
    
    def add_asset(self, asset: str) -> None:
        """
        Add an asset to monitor.
        
        Args:
            asset: Asset symbol to monitor
        """
        with self.lock:
            self.assets.add(asset)
            logger.info(f"Added asset to monitor: {asset}")
    
    def remove_asset(self, asset: str) -> None:
        """
        Remove an asset from monitoring.
        
        Args:
            asset: Asset symbol to remove
        """
        with self.lock:
            self.assets.discard(asset)
            logger.info(f"Removed asset from monitoring: {asset}")
    
    def start(self) -> None:
        """Start the monitoring thread."""
        if self.running:
            logger.warning("Monitor is already running")
            return
            
        self.running = True
        self.thread = threading.Thread(target=self._monitor_loop)
        self.thread.daemon = True
        self.thread.start()
        logger.info("Started real-time monitoring thread")
    
    def stop(self) -> None:
        """Stop the monitoring thread."""
        self.running = False
        if self.thread:
            self.thread.join()
            self.thread = None
        logger.info("Stopped real-time monitoring")
    
    def _monitor_loop(self) -> None:
        """Main monitoring loop."""
        while self.running:
            try:
                with self.lock:
                    assets = list(self.assets)
                
                for asset in assets:
                    try:
                        # Fetch real-time and historical data
                        realtime_data = fetch_realtime_data(asset)
                        historical_data = fetch_historical_data(asset)
                        
                        if realtime_data and historical_data:
                            # Run algorithms
                            results = run_algorithms(asset, realtime_data, historical_data)
                            
                            # Call callback if provided
                            if self.callback:
                                for algo_name, result in results.items():
                                    self.callback(asset, algo_name, result["results"], None)
                        else:
                            logger.warning(f"Failed to fetch data for {asset}")
                            
                    except Exception as e:
                        logger.error(f"Error processing asset {asset}: {e}")
                        if self.callback:
                            self.callback(asset, "", {}, str(e))
                
                # Sleep for a short time to avoid overwhelming the API
                time.sleep(1)
                
            except Exception as e:
                logger.error(f"Error in monitor loop: {e}")
                time.sleep(5)  # Sleep longer on error

def main():
    """Main function to continuously fetch data, calculate risk, and run custom algorithms for multiple assets."""
    
    # Use predefined asset lists from data_provider
    # Combine some crypto and stock assets for testing
    mixed_assets = CRYPTO_ASSETS[:5] + STOCK_ASSETS[:5]
    asset_id_quote = "USD"
    
    logger.info(f"Starting real-time monitoring for {len(mixed_assets)} assets: {', '.join(mixed_assets)}")
    start_time = time.time()  # Capture the script's start time

    # Initialize StocksAPI once outside the loop
    stocks_api = StocksAPI()
    
    # Predefine factor exposures, factor covariance matrix, and specific risk
    num_assets = len(mixed_assets)
    num_factors = 100  # Example number of factors
    factor_exposures = np.random.randn(num_assets, num_factors).astype(np.float32)
    factor_covariance_matrix = np.random.randn(num_factors, num_factors).astype(np.float32)
    factor_covariance_matrix = factor_covariance_matrix @ factor_covariance_matrix.T  # Ensure PSD
    specific_risk = np.random.rand(num_assets).astype(np.float32)

    iteration = 0
    while True:
        iteration += 1
        logger.info(f"Iteration {iteration} - Processing {len(mixed_assets)} assets")
        
        # Track performance metrics
        fetch_times = []
        risk_times = []
        model_times = []
        algo_times = []
        
        for i, asset_id_base in enumerate(mixed_assets):
            try:
                fetch_start = time.time()
                data = fetch_realtime_data(asset_id_base, asset_id_quote)
                historical_data = fetch_historical_data(asset_id_base, asset_id_quote)
                fetch_time = time.time() - fetch_start
                fetch_times.append(fetch_time)

                if data and historical_data:
                    # Handle different response formats between Yahoo Finance and Polygon
                    if 'rate' in data:
                        rate = data['rate']
                    elif 'price' in data:
                        rate = data['price']
                    else:
                        logger.warning(f"Unexpected data format for {asset_id_base}: {data}")
                        continue

                    logger.info(f"Asset {i+1}/{len(mixed_assets)}: {asset_id_base}/{asset_id_quote} = {rate:.2f} (fetched in {fetch_time:.4f}s)")

                    # Process historical data
                    # Handle different response formats for historical data
                    if 'rate_close' in historical_data[0]:
                        historical_rates = [item['rate_close'] for item in historical_data]
                    elif 'close' in historical_data[0]:
                        historical_rates = [item['close'] for item in historical_data]
                    else:
                        logger.warning(f"Unexpected historical data format for {asset_id_base}")
                        continue

                    returns_matrix = np.array([historical_rates]).astype(np.float32)

                    # Analyze buy/sell signals
                    signal = analyze_buy_sell(historical_rates)
                    logger.info(f"  Signal: {signal}")

                    # Risk Calculation using StocksAPI
                    risk_start_time = time.time()
                    var, es = stocks_api.calculate_var_and_es(returns_matrix, confidence_level=0.05)
                    risk_time = time.time() - risk_start_time
                    risk_times.append(risk_time)
                    
                    # Handle NaN values
                    var = np.nan_to_num(var, nan=0)
                    es = np.nan_to_num(es, nan=0)
                    
                    # Convert numpy arrays to scalar values for logging
                    var_value = float(var[0]) if isinstance(var, np.ndarray) and var.size > 0 else float(var)
                    es_value = float(es[0]) if isinstance(es, np.ndarray) and es.size > 0 else float(es)
                    
                    logger.info(f"  Risk: VaR={var_value:.4f}, ES={es_value:.4f} (calc in {risk_time:.4f}s)")
                    
                    # Run custom algorithms
                    algo_start_time = time.time()
                    asset_results = run_algorithms(asset_id_base, data, historical_data)
                    algo_time = time.time() - algo_start_time
                    algo_times.append(algo_time)
                    
                    # Log algorithm results
                    for algo_name, result in asset_results.items():
                        if 'results' in result and 'signal' in result['results']:
                            logger.info(f"  Algorithm {algo_name}: Signal={result['results']['signal']} (calc in {result['run_time']:.4f}s)")
                    
                    # Risk Model Calculation - only do this every 10 iterations to save processing time
                    if iteration % 10 == 0:
                        model_start_time = time.time()
                        covariance_matrix = stocks_api.risk_model(factor_exposures, factor_covariance_matrix, specific_risk)
                        model_time = time.time() - model_start_time
                        model_times.append(model_time)
                        
                        # Numerical checks for covariance matrix
                        if not np.isreal(covariance_matrix).all() or not np.isfinite(covariance_matrix).all():
                            logger.warning(f"  Warning: Covariance matrix for {asset_id_base} is not real or finite.")
                            
                        # Some more checks and processing on the covariance matrix
                        if np.any(np.diag(covariance_matrix) < 0):
                            logger.warning(f"  Warning: Negative variance detected for {asset_id_base}.")
                            
                        # Compute the correlation matrix using the covariance matrix
                        # This is a diagonal matrix where each element is 1/sqrt(cov[i,i])
                        std_devs = np.sqrt(np.diag(covariance_matrix))
                        inv_std_devs = np.zeros_like(std_devs)
                        inv_std_devs[std_devs > 0] = 1 / std_devs[std_devs > 0]
                        inv_std_matrix = np.diag(inv_std_devs)
                        
                        # Compute the correlation matrix
                        correlation_matrix = inv_std_matrix @ covariance_matrix @ inv_std_matrix
                        
                        # Count how many are high correlations (>0.7)
                        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool), k=1)
                        high_correlations = np.sum(np.abs(correlation_matrix[mask]) > 0.7)
                        
                        logger.info(f"  Risk Model: matrix shape={covariance_matrix.shape}, high correlations={high_correlations}")
                        logger.info(f"  Model calculated in {model_time:.4f}s")
                else:
                    if not data:
                        logger.warning(f"Failed to retrieve real-time data for {asset_id_base}.")
                    if not historical_data:
                        logger.warning(f"Failed to retrieve historical data for {asset_id_base}.")
            except Exception as e:
                logger.error(f"Error processing asset {asset_id_base}: {e}")

        # Print performance summary for this iteration
        if fetch_times:
            logger.info(f"Performance: avg fetch={np.mean(fetch_times):.4f}s, avg risk={np.mean(risk_times) if risk_times else 0:.4f}s, avg algo={np.mean(algo_times) if algo_times else 0:.4f}s")
        
        # Add a delay between iterations (10 seconds - reduces API calls)
        elapsed = time.time() - start_time
        logger.info(f"Sleeping for 10 seconds... (total runtime: {elapsed:.1f}s)")
        time.sleep(10)

if __name__ == "__main__":
    main()




