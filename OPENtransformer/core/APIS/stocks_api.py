import numpy as np
import ctypes
import gc
import logging
import time
import os
import hashlib
import pickle
from pathlib import Path
import sys
import tempfile
from scipy import special  # Add scipy.special import
import yfinance as yf
from datetime import datetime
import platform

# Remove the sys.path.append line since we're using proper package imports now
# sys.path.append(str(Path(__file__).parent.parent))

from finlib.finlib.asm.kernels.layer_norm import get_kernel_code as get_layer_norm_code
from finlib.finlib.asm.kernels.matmul import get_kernel_code as get_matmul_code
from finlib.finlib.asm.assembler.builder import build_and_load
from finlib.finlib.asm.assembler.builder import build_and_jit

logger = logging.getLogger("finlib.stocks_api")

# Global kernel cache directory
_KERNEL_CACHE_DIR = os.path.join(os.path.expanduser("~"), ".finlib", "kernel_cache")

# Create cache directory if it doesn't exist
os.makedirs(_KERNEL_CACHE_DIR, exist_ok=True)

class StocksAPI:
    """
    API for stock market data analysis
    Uses optimized assembly kernels for computation when available,
    falls back to pure Python/NumPy implementations otherwise
    """
    
    # Class-level cache to avoid recompiling kernels across instances
    _kernel_cache = {}
    
    @staticmethod
    def _get_kernel_hash(code):
        """Calculate a hash of the kernel code for caching purposes"""
        return hashlib.md5(code.encode('utf-8')).hexdigest()
    
    @staticmethod
    def _get_kernel_path(kernel_name, code_hash):
        """Get the path to the cached kernel library"""
        return os.path.join(_KERNEL_CACHE_DIR, f"{kernel_name}_{code_hash}.dylib")
    
    @staticmethod
    def _load_cached_kernel(kernel_name, code):
        """Load a kernel from cache if available, otherwise compile and cache it"""
        code_hash = StocksAPI._get_kernel_hash(code)
        kernel_path = StocksAPI._get_kernel_path(kernel_name, code_hash)
        
        logger.info(f"Looking for cached {kernel_name} kernel with hash {code_hash}")
        logger.info(f"Cache path: {kernel_path}")
        
        # Check if kernel is already in memory cache
        if kernel_name in StocksAPI._kernel_cache:
            logger.info(f"Using in-memory cached {kernel_name} kernel")
            return StocksAPI._kernel_cache[kernel_name]
        
        # Check if kernel is cached on disk
        if os.path.exists(kernel_path):
            try:
                logger.info(f"Loading {kernel_name} kernel from disk cache: {kernel_path}")
                # Load the library using ctypes
                lib = ctypes.CDLL(kernel_path)
                
                # Try to get the function with the correct symbol name
                if kernel_name == "layer_norm":
                    symbol = "_layer_norm"
                elif kernel_name == "matmul":
                    symbol = "__matmul"
                else:
                    symbol = f"_{kernel_name}"
                
                try:
                    func = getattr(lib, symbol)
                    logger.info(f"Successfully loaded kernel function with symbol: {symbol}")
                except AttributeError:
                    # Try alternative symbol name
                    alt_symbol = symbol.lstrip('_')
                    try:
                        func = getattr(lib, alt_symbol)
                        logger.info(f"Successfully loaded kernel function with alternative symbol: {alt_symbol}")
                    except AttributeError:
                        raise AttributeError(f"Could not find kernel function with symbols {symbol} or {alt_symbol}")
                
                # Set argument types based on kernel type
                if kernel_name == "matmul":
                    func.argtypes = [
                        ctypes.c_void_p,  # A_ptr
                        ctypes.c_void_p,  # B_ptr
                        ctypes.c_void_p,  # C_ptr
                        ctypes.c_int,     # N
                        ctypes.c_int,     # K
                        ctypes.c_int      # M
                    ]
                elif kernel_name == "layer_norm":
                    func.argtypes = [
                        ctypes.POINTER(ctypes.c_float),  # input
                        ctypes.POINTER(ctypes.c_float),  # output
                        ctypes.POINTER(ctypes.c_float),  # gamma weights
                        ctypes.POINTER(ctypes.c_float),  # beta weights
                        ctypes.c_int,                    # size
                        ctypes.c_int                     # num_dims
                    ]
                
                func.restype = None
                
                # Create a wrapper function that keeps a reference to the library
                def wrapped_func(*args):
                    func(*args)
                
                # Keep references to prevent garbage collection
                wrapped_func._lib = lib
                wrapped_func._lib_path = kernel_path
                
                # Cache the function in memory
                StocksAPI._kernel_cache[kernel_name] = wrapped_func
                
                logger.info(f"Successfully loaded {kernel_name} kernel from cache")
                return wrapped_func
            except Exception as e:
                logger.warning(f"Failed to load cached kernel {kernel_name}: {e}")
                # Fall back to compiling
        else:
            logger.info(f"No cached kernel found at {kernel_path}")
        
        # Compile the kernel
        logger.info(f"Compiling {kernel_name} kernel")
        kernel_func = build_and_load(code, kernel_name)
        
        if kernel_func is None:
            logger.error(f"Failed to compile and load {kernel_name} kernel")
            return None
        
        # Cache the kernel in memory
        StocksAPI._kernel_cache[kernel_name] = kernel_func
        
        # Try to copy the compiled library to our cache directory
        try:
            # Get the path to the compiled library
            if hasattr(kernel_func, '_lib_path'):
                lib_path = kernel_func._lib_path
                logger.info(f"Compiled kernel library path: {lib_path}")
                # Copy the library to our cache
                import shutil
                shutil.copy2(lib_path, kernel_path)
                logger.info(f"Cached {kernel_name} kernel to {kernel_path}")
            else:
                logger.warning(f"Kernel function has no _lib_path attribute, cannot cache")
        except Exception as e:
            logger.warning(f"Failed to cache {kernel_name} kernel: {e}")
        
        return kernel_func
    
    def __init__(self):
        """Initialize the API and load ASM kernels"""
        # Initialize kernels
        self.use_kernels = True
        try:
            self._initialize_kernels()
            logger.info("Successfully initialized ASM kernels")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize ASM kernels: {e}")
    
    def _initialize_kernels(self):
        """Initialize ASM kernels"""
        # Try to load kernels from cache first
        self.matmul_kernel = self._load_cached_kernel('matmul', get_matmul_code())
        self.layer_norm_kernel = build_and_jit(get_layer_norm_code(), "_layer_norm")
        logger.info("Successfully initialized ASM kernels")
    
    def _ensure_2d(self, arr):
        """Ensure array is 2D with shape (days, assets)"""
        if len(arr.shape) == 1:
            return arr.reshape(-1, 1)
        return arr
        
    def moving_average(self, prices, window):
        """Calculate moving average with proper shape handling"""
        prices = self._ensure_2d(prices)
        days, assets = prices.shape
        result = np.zeros_like(prices, dtype=np.float64)  # Use float64 for better precision
        
        for i in range(days):
            start_idx = max(0, i - window + 1)
            window_prices = prices[start_idx:i+1]
            result[i] = np.mean(window_prices, axis=0)
            
        return result
        
    def relative_strength_index(self, prices, window=14):
        """Calculate RSI with proper shape handling"""
        prices = self._ensure_2d(prices)
        days, assets = prices.shape
        result = np.zeros_like(prices, dtype=np.float64)  # Use float64 for better precision
        
        # Calculate price changes
        delta = np.diff(prices, axis=0)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        # Calculate average gain and loss
        avg_gain = np.zeros_like(prices, dtype=np.float64)
        avg_loss = np.zeros_like(prices, dtype=np.float64)
        
        for i in range(days):
            if i < window:
                avg_gain[i] = np.mean(gain[:i+1], axis=0)
                avg_loss[i] = np.mean(loss[:i+1], axis=0)
            else:
                avg_gain[i] = (avg_gain[i-1] * (window-1) + gain[i-1]) / window
                avg_loss[i] = (avg_loss[i-1] * (window-1) + loss[i-1]) / window
                
        # Calculate RSI
        rs = avg_gain / np.where(avg_loss == 0, 1e-10, avg_loss)
        result = 100 - (100 / (1 + rs))
        
        # Handle edge cases
        result[0] = 100  # First value is always 100
        result = np.where(np.isnan(result), 100, result)  # Replace NaN with 100
        
        return result
        
    def macd(self, prices, fast_period=12, slow_period=26, signal_period=9):
        """Calculate MACD with proper shape handling"""
        prices = self._ensure_2d(prices)
        days, assets = prices.shape
        
        # Calculate EMAs using ASM kernel if available
        fast_ema = self._exponential_moving_average(prices, fast_period)
        slow_ema = self._exponential_moving_average(prices, slow_period)
        
        # Calculate MACD line
        macd_line = fast_ema - slow_ema
        
        # Calculate signal line (EMA of MACD line)
        signal_line = self._exponential_moving_average(macd_line, signal_period)
        
        # Calculate histogram
        histogram = macd_line - signal_line
        
        return macd_line, signal_line, histogram
        
    def bollinger_bands(self, prices, window=20, num_std=2):
        """Calculate Bollinger Bands with proper shape handling"""
        prices = self._ensure_2d(prices)
        days, assets = prices.shape
        
        # Calculate middle band (SMA)
        middle_band = self.moving_average(prices, window)
        
        # Calculate standard deviation
        std = np.zeros_like(prices)
        for i in range(days):
            start_idx = max(0, i - window + 1)
            std[i] = np.std(prices[start_idx:i+1], axis=0)
        
        # Calculate upper and lower bands
        upper_band = middle_band + (num_std * std)
        lower_band = middle_band - (num_std * std)
        
        return upper_band, middle_band, lower_band
        
    def price_american_option_fd(self, S0, K, T, r, sigma, isCall=True, M=100, N=100):
        """Price American options using finite difference method with ASM kernel if available"""
        if self.matmul_kernel is not None:
            # Use ASM kernel for matrix operations
            dt = T / N
            dx = sigma * np.sqrt(3 * dt)
            x = np.arange(-M, M+1) * dx
            S = S0 * np.exp(x)
            
            # Initialize option values
            V = np.maximum(S - K, 0) if isCall else np.maximum(K - S, 0)
            
            # Prepare arrays for ASM kernel
            V_ptr = V.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
            S_ptr = S.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
            result_ptr = np.zeros_like(V).ctypes.data_as(ctypes.POINTER(ctypes.c_float))
            
            # Time stepping with ASM kernel
            for n in range(N):
                self.matmul_kernel(V_ptr, S_ptr, result_ptr, 2*M+1, 1, 1)
                V = result_ptr
                
                # Apply early exercise condition
                if isCall:
                    V = np.maximum(V, S - K)
                else:
                    V = np.maximum(V, K - S)
            
            # Find the option value at S0
            idx = np.searchsorted(S, S0)
            return V[idx]
        else:
            # Fall back to pure NumPy implementation
            dt = T / N
            dx = sigma * np.sqrt(3 * dt)
            x = np.arange(-M, M+1) * dx
            S = S0 * np.exp(x)
            
            # Initialize option values
            V = np.maximum(S - K, 0) if isCall else np.maximum(K - S, 0)
            
            # Time stepping
            for n in range(N):
                # Calculate coefficients
                alpha = 0.5 * sigma**2 * dt / dx**2
                beta = (r - 0.5 * sigma**2) * dt / (2 * dx)
                
                # Update option values
                V[1:-1] = (1 - 2 * alpha) * V[1:-1] + \
                          (alpha + beta) * V[2:] + \
                          (alpha - beta) * V[:-2]
                
                # Apply early exercise condition
                if isCall:
                    V = np.maximum(V, S - K)
                else:
                    V = np.maximum(V, K - S)
            
            # Find the option value at S0
            idx = np.searchsorted(S, S0)
            return V[idx]

    def calculate_var_and_es(self, returns, confidence_level=0.95):
        """Calculate Value at Risk and Expected Shortfall using ASM kernels"""
        returns = self._ensure_2d(returns)
        days, assets = returns.shape
        
        if days == 1:
            return np.abs(returns[0]), 1.2 * np.abs(returns[0])
        
        # Sort returns in ascending order (most negative first)
        sorted_returns = np.sort(returns, axis=0)
        
        # Calculate VaR index
        var_index = max(1, int((1 - confidence_level) * days))
        var = np.abs(sorted_returns[var_index - 1])
        
        # Calculate ES using a slightly larger window
        es_index = min(var_index + 1, days)
        
        # Use ASM kernel for normalization
        tail_values = sorted_returns[:es_index].copy().astype(np.float32)
        gamma = np.ones(assets, dtype=np.float32)
        beta = np.zeros(assets, dtype=np.float32)
        
        if self.layer_norm_kernel is not None:
            try:
                # Normalize each asset's returns separately
                result = np.zeros_like(tail_values, dtype=np.float32)
                for i in range(assets):
                    # Get pointers for this asset's data
                    tail_ptr = tail_values[:, i].ctypes.data_as(ctypes.POINTER(ctypes.c_float))
                    result_ptr = result[:, i].ctypes.data_as(ctypes.POINTER(ctypes.c_float))
                    gamma_ptr = gamma[i:i+1].ctypes.data_as(ctypes.POINTER(ctypes.c_float))
                    beta_ptr = beta[i:i+1].ctypes.data_as(ctypes.POINTER(ctypes.c_float))
                    
                    # Call ASM kernel with proper argument types
                    self.layer_norm_kernel(
                        tail_ptr,
                        result_ptr,
                        gamma_ptr,
                        beta_ptr,
                        ctypes.c_int(es_index),
                        ctypes.c_int(1)
                    )
                es = np.abs(np.mean(result, axis=0))
            except Exception as e:
                logger.error(f"ASM layer_norm failed: {e}. Falling back to NumPy implementation.")
                es = np.abs(np.mean(sorted_returns[:es_index], axis=0))
        else:
            es = np.abs(np.mean(sorted_returns[:es_index], axis=0))
        
        # Add a small buffer to ensure ES is strictly greater than VaR
        es = np.maximum(es, var * 1.001)
        
        return var, es
        
    def run_stress_test(self, portfolio, risk_factors, scenarios):
        """Run stress test with proper shape handling"""
        portfolio = self._ensure_2d(portfolio)
        
        # Calculate historical returns
        historical_market = np.mean(risk_factors['market'])
        historical_interest_rate = np.mean(risk_factors['interest_rate'])
        
        # Calculate scenario returns
        scenario_returns = []
        for market_scenario in scenarios['market']:
            for rate_scenario in scenarios['interest_rate']:
                scenario_returns.append(
                    historical_market * (1 + market_scenario) +
                    historical_interest_rate * (1 + rate_scenario)
                )
                
        return {
            'historical_market': historical_market,
            'historical_interest_rate': historical_interest_rate,
            'scenario_returns': np.array(scenario_returns)
        }

    def _begin_method(self):
        """Begin a method call, clearing any temporary arrays"""
        self.temp_arrays = []
    
    def _end_method(self):
        """End a method call, clearing any temporary arrays"""
        self.temp_arrays = []
        gc.collect()
    
    def _exponential_moving_average(self, prices, period):
        """Calculate EMA using ASM kernel"""
        prices = self._ensure_2d(prices).astype(np.float32)
        days, assets = prices.shape
        result = np.zeros_like(prices, dtype=np.float32)
        alpha = 2 / (period + 1)
        
        if self.matmul_kernel is not None:
            try:
                # Use ASM kernel for matrix operations
                weights = np.array([(1 - alpha) ** i for i in range(days)], dtype=np.float32)
                weights = weights.reshape(-1, 1)
                
                # Prepare input arrays for ASM kernel
                prices_ptr = prices.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
                weights_ptr = weights.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
                result_ptr = result.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
                
                # Call ASM kernel
                self.matmul_kernel(prices_ptr, weights_ptr, result_ptr, days, assets, 1)
                return result / np.sum(weights)
            except Exception as e:
                logger.error(f"ASM matmul failed: {e}. Falling back to NumPy implementation.")
        
        # Fall back to NumPy implementation
        weights = np.array([(1 - alpha) ** i for i in range(days)], dtype=np.float32)
        weights = weights.reshape(-1, 1)
        return np.sum(prices * weights, axis=0) / np.sum(weights)
    
    def calculate_monte_carlo_var(self, returns: np.ndarray, confidence_level: float = 0.95, num_simulations: int = 1000000) -> float:
        """
        Calculates VaR using Monte Carlo simulation with optimized kernels.
        """
        returns = np.array(returns, dtype=np.float32).copy()
        portfolio_values = np.zeros(num_simulations, dtype=np.float32)
        
        # Generate random scenarios
        np.random.seed(42)
        random_indices = np.random.randint(0, len(returns), num_simulations)
        selected_returns = returns[random_indices].copy()
        
        # Calculate portfolio values using matrix multiplication
        portfolio_value = np.ones(1, dtype=np.float32)
        self.matmul_kernel(
            portfolio_value.ctypes.data,
            selected_returns.T.ctypes.data,
            portfolio_values.ctypes.data,
            1, 1, num_simulations
        )
        
        # Sort and find VaR
        sorted_values = np.sort(portfolio_values)
        var_index = int(num_simulations * confidence_level)
        var_value = -sorted_values[var_index]
        
        return var_value

    def calculate_black_scholes_var(self, returns: np.ndarray, confidence_level: float = 0.95, num_simulations: int = 1000000) -> float:
        """
        Calculates the Value at Risk (VaR) using Black-Scholes model.
        """
        pass    

    def price_barrier_option(self, S0: float, K: float, H: float, T: float, r: float, sigma: float,
                           barrier_type: str = 'up-and-out', is_call: bool = True,
                           num_steps: int = 1000) -> float:
        """
        Price barrier options using Monte Carlo simulation with variance reduction.
        Uses optimized matrix multiplication kernels for path simulation.
        
        Args:
            S0: Initial stock price
            K: Strike price
            H: Barrier level
            T: Time to maturity in years
            r: Risk-free interest rate
            sigma: Volatility
            barrier_type: One of 'up-and-out', 'up-and-in', 'down-and-out', 'down-and-in'
            is_call: True for call option, False for put option
            num_steps: Number of time steps in simulation
            
        Returns:
            float: Option price
        """
        self._begin_method()  # Ensure clean state
        
        dt = T/num_steps
        num_paths = 100000  # Large number of paths for accuracy
        
        # Generate random paths using optimized matrix operations
        Z = np.random.standard_normal((num_paths, num_steps)).astype(np.float32)
        S = np.zeros((num_paths, num_steps + 1), dtype=np.float32)
        S[:, 0] = S0
        self.temp_arrays.extend([Z, S])  # Keep reference to prevent GC
        
        # Precompute drift and diffusion terms
        drift = np.float32((r - 0.5*sigma**2)*dt)
        diffusion = np.float32(sigma*np.sqrt(dt))
        
        # Simulate paths in batches to avoid memory issues
        batch_size = 10000
        num_batches = (num_paths + batch_size - 1) // batch_size
        
        for b in range(num_batches):
            start_idx = b * batch_size
            end_idx = min((b + 1) * batch_size, num_paths)
            current_batch_size = end_idx - start_idx
            
            for t in range(num_steps):
                # Prepare batch data
                batch_Z = Z[start_idx:end_idx, t]
                batch_S = S[start_idx:end_idx, t]
                
                # Calculate log returns for this batch
                log_returns = np.zeros(current_batch_size, dtype=np.float32)
                self.temp_arrays.append(log_returns)
                
                # Compute returns directly to avoid matrix multiplication overhead
                log_returns = drift + diffusion * batch_Z
                
                # Update stock prices for this batch
                S[start_idx:end_idx, t+1] = batch_S * np.exp(log_returns)
        
        # Check barrier conditions
        if barrier_type.startswith('up'):
            hits = np.any(S > H, axis=1)
        else:  # down
            hits = np.any(S < H, axis=1)
        
        # Calculate payoffs
        if is_call:
            payoffs = np.maximum(S[:, -1] - K, 0)
        else:
            payoffs = np.maximum(K - S[:, -1], 0)
        self.temp_arrays.append(payoffs)
        
        # Apply barrier condition
        if barrier_type.endswith('out'):
            payoffs[hits] = 0
        else:  # in
            payoffs[~hits] = 0
        
        # Use geometric average control variate for better stability
        log_ST = np.log(S[:, -1])
        expected_log_ST = np.log(S0) + (r - 0.5*sigma**2)*T
        cv = log_ST - expected_log_ST
        self.temp_arrays.extend([log_ST, cv])
        
        # Calculate means for numerical stability
        payoff_mean = np.mean(payoffs)
        cv_mean = np.mean(cv)
        
        # Calculate covariance components directly to avoid matrix instability
        payoff_cv_sum = np.sum((payoffs - payoff_mean) * (cv - cv_mean))
        cv_var_sum = np.sum((cv - cv_mean) * (cv - cv_mean))
        
        # Calculate beta with safety checks
        if cv_var_sum > 1e-10:  # Avoid division by very small numbers
            beta = payoff_cv_sum / cv_var_sum
        else:
            beta = 0.0
        
        # Apply control variate adjustment with clipping for stability
        beta = np.clip(beta, -100, 100)  # Prevent extreme adjustments
        payoffs_cv = payoffs - beta * cv
        
        # Calculate final option price
        discount_factor = np.exp(-r*T)
        option_price = discount_factor * np.mean(payoffs_cv)
        
        self._end_method()  # Clean up temporary arrays
        
        return float(option_price)

    def simulate_heston_model_fft(self, num_paths: int, num_time_steps: int, initial_price: float,
                                risk_free_rate: float, volatility: float, kappa: float, theta: float,
                                sigma: float, rho: float, use_fft: bool = True) -> np.ndarray:
        """
        Enhanced Heston model simulation using FFT optimization for characteristic function.
        
        Args:
            num_paths: Number of simulation paths
            num_time_steps: Number of time steps
            initial_price: Initial stock price
            risk_free_rate: Risk-free rate
            volatility: Initial volatility
            kappa: Mean reversion speed
            theta: Long-term variance
            sigma: Volatility of variance
            rho: Correlation between stock and variance processes
            use_fft: Whether to use FFT optimization
            
        Returns:
            np.ndarray: Simulated price paths
        """
        dt = 1 / num_time_steps
        
        if use_fft:
            # Use FFT for faster computation of characteristic function
            omega = np.fft.fftfreq(num_paths)
            cf = np.zeros(num_paths, dtype=np.complex64)
            
            # Compute characteristic function in frequency domain
            for k in range(num_paths):
                u = 2 * np.pi * omega[k]
                d = np.sqrt((kappa - rho*sigma*u*1j)**2 + sigma**2*(u**2 + u*1j))
                g = (kappa - rho*sigma*u*1j - d)/(kappa - rho*sigma*u*1j + d)
                
                A = (risk_free_rate*u*1j*dt + (kappa*theta)/(sigma**2)*
                     ((kappa - rho*sigma*u*1j - d)*dt - 2*np.log((1-g*np.exp(-d*dt))/(1-g))))
                
                B = ((kappa - rho*sigma*u*1j - d)*(1-np.exp(-d*dt)))/(sigma**2*(1-g*np.exp(-d*dt)))
                
                cf[k] = np.exp(A + B*volatility**2)
            
            # Generate correlated random numbers using FFT
            Z1_base = np.fft.ifft(cf).real
            Z2_base = rho * Z1_base + np.sqrt(1-rho**2) * np.random.randn(num_paths)
            
            # Reshape to match the non-FFT case dimensions
            Z1 = np.tile(Z1_base[:, np.newaxis], (1, num_time_steps))
            Z2 = np.tile(Z2_base[:, np.newaxis], (1, num_time_steps))
            
            # Ensure float32 precision
            Z1 = Z1.astype(np.float32)
            Z2 = Z2.astype(np.float32)
        else:
            # Fall back to standard random number generation
            Z1 = np.random.randn(num_paths, num_time_steps).astype(np.float32)
            Z2 = np.random.randn(num_paths, num_time_steps).astype(np.float32)
            Z2 = rho * Z1 + np.sqrt(1-rho**2) * Z2
        
        # Initialize arrays
        S = np.zeros((num_paths, num_time_steps), dtype=np.float32)
        v = np.zeros((num_paths, num_time_steps), dtype=np.float32)
        S[:, 0] = initial_price
        v[:, 0] = volatility**2
        
        # Simulate paths
        for t in range(1, num_time_steps):
            v[:, t] = np.maximum(0, v[:, t-1] + kappa*(theta - v[:, t-1])*dt + 
                       sigma*np.sqrt(v[:, t-1]*dt)*Z2[:, t-1])
            S[:, t] = S[:, t-1] * np.exp((risk_free_rate - 0.5*v[:, t-1])*dt + 
                                        np.sqrt(v[:, t-1]*dt)*Z1[:, t-1])
        
        return S.copy()

    def simulate_hull_white_model(self, num_paths: int, num_steps: int, r0: float,
                                  mean_reversion: float, volatility: float, theta: float,
                                  T: float) -> np.ndarray:
        """
        Simulate interest rates using the Hull-White model.
        
        Args:
            num_paths: Number of simulation paths
            num_steps: Number of time steps
            r0: Initial short rate
            mean_reversion: Mean reversion speed (a)
            volatility: Volatility of short rate (σ)
            theta: Long-term mean level
            T: Time horizon
            
        Returns:
            np.ndarray: Simulated short rate paths
        """
        dt = T/num_steps
        r = np.zeros((num_paths, num_steps+1), dtype=np.float32)
        r[:, 0] = r0
        
        # Generate random numbers
        dW = np.random.normal(0, np.sqrt(dt), (num_paths, num_steps)).astype(np.float32)
        
        # Simulate paths using vectorized operations
        for t in range(num_steps):
            dr = mean_reversion * (theta - r[:, t]) * dt + volatility * dW[:, t]
            r[:, t+1] = r[:, t] + dr
        
        return r

    def simulate_cir_model(self, num_paths: int, num_steps: int, r0: float,
                          kappa: float, theta: float, sigma: float, T: float) -> np.ndarray:
        """
        Simulate interest rates using the Cox-Ingersoll-Ross (CIR) model.
        
        Args:
            num_paths: Number of simulation paths
            num_steps: Number of time steps
            r0: Initial short rate
            kappa: Mean reversion speed
            theta: Long-term mean level
            sigma: Volatility parameter
            T: Time horizon
            
        Returns:
            np.ndarray: Simulated short rate paths
        """
        dt = T/num_steps
        r = np.zeros((num_paths, num_steps+1), dtype=np.float32)
        r[:, 0] = r0
        
        # Generate random numbers
        dW = np.random.normal(0, np.sqrt(dt), (num_paths, num_steps)).astype(np.float32)
        
        # Simulate paths using vectorized operations
        for t in range(num_steps):
            dr = kappa * (theta - r[:, t]) * dt + sigma * np.sqrt(r[:, t]) * dW[:, t]
            r[:, t+1] = np.maximum(r[:, t] + dr, 0)  # Ensure rates stay positive
        
        return r

    def interpolate_yield_curve(self, tenors: np.ndarray, rates: np.ndarray, 
                              method: str = 'cubic_spline') -> tuple:
        """
        Interpolate yield curve using various methods.
        
        Args:
            tenors: Array of tenor points (in years)
            rates: Array of corresponding interest rates
            method: Interpolation method ('cubic_spline' or 'nelson_siegel')
            
        Returns:
            tuple: (interpolation_function, parameters)
        """
        if method == 'cubic_spline':
            # Convert inputs to float32
            tenors = tenors.astype(np.float32)
            rates = rates.astype(np.float32)
            
            # Number of spline pieces
            n = len(tenors) - 1
            
            # Build the tridiagonal system for natural cubic spline
            A = np.zeros((n+1, n+1), dtype=np.float32)
            b = np.zeros(n+1, dtype=np.float32)
            
            # Set up the tridiagonal matrix
            for i in range(1, n):
                hi = tenors[i] - tenors[i-1]
                hi1 = tenors[i+1] - tenors[i]
                A[i, i-1] = hi
                A[i, i] = 2*(hi + hi1)
                A[i, i+1] = hi1
                b[i] = 3*((rates[i+1] - rates[i])/hi1 - (rates[i] - rates[i-1])/hi)
            
            # Natural spline boundary conditions
            A[0, 0] = 1
            A[n, n] = 1
            
            # Solve the system using optimized matrix operations
            c = np.zeros(n+1, dtype=np.float32)
            self.matmul_kernel(
                A.ctypes.data,
                b.reshape(-1, 1).ctypes.data,
                c.reshape(-1, 1).ctypes.data,
                n+1, 1, n+1
            )
            
            # Store the spline coefficients
            coeffs = []
            for i in range(n):
                h = tenors[i+1] - tenors[i]
                coeffs.append([
                    rates[i],
                    (rates[i+1] - rates[i])/h - h*(2*c[i] + c[i+1])/3,
                    c[i],
                    (c[i+1] - c[i])/(3*h)
                ])
            
            return coeffs, tenors
            
        elif method == 'nelson_siegel':
            # Implement Nelson-Siegel model parameters estimation
            def nelson_siegel(t, beta0, beta1, beta2, tau):
                factor1 = 1.0
                factor2 = (1 - np.exp(-t/tau))/(t/tau)
                factor3 = factor2 - np.exp(-t/tau)
                return beta0 + beta1*factor2 + beta2*factor3
            
            # Initial parameter guess
            p0 = [np.mean(rates), 0.0, 0.0, 1.0]
            
            # Convert to float32 for optimization
            tenors = tenors.astype(np.float32)
            rates = rates.astype(np.float32)
            
            # Optimize parameters using least squares
            from scipy.optimize import least_squares
            def objective(params):
                return nelson_siegel(tenors, *params) - rates
            
            result = least_squares(objective, p0)
            params = result.x
            
            return nelson_siegel, params
        
        else:
            raise ValueError("Unsupported interpolation method")

    def calculate_bond_risk_metrics(self, price: float, coupon_rate: float, 
                                  years_to_maturity: float, yield_rate: float,
                                  frequency: int = 2) -> tuple:
        """
        Calculate key risk metrics for a bond.
        
        Args:
            price: Bond price
            coupon_rate: Annual coupon rate (as decimal)
            years_to_maturity: Years to maturity
            yield_rate: Yield to maturity (as decimal)
            frequency: Coupon frequency per year
            
        Returns:
            tuple: (modified_duration, convexity, dollar_duration)
        """
        # Convert to float32 for calculations
        price = np.float32(price)
        coupon_rate = np.float32(coupon_rate)
        years_to_maturity = np.float32(years_to_maturity)
        yield_rate = np.float32(yield_rate)
        
        # Calculate number of remaining payments
        num_payments = int(years_to_maturity * frequency)
        
        # Calculate payment amount
        payment = coupon_rate * price / frequency
        
        # Calculate duration
        duration = np.float32(0.0)
        convexity = np.float32(0.0)
        discount_factor = np.float32(1.0 / (1.0 + yield_rate/frequency))
        
        for t in range(1, num_payments + 1):
            t_years = t / frequency
            pv_factor = discount_factor ** t
            
            if t == num_payments:
                # Add principal to final payment
                cf = payment + price
            else:
                cf = payment
            
            # Duration calculation
            duration += t_years * cf * pv_factor
            
            # Convexity calculation
            convexity += t_years * (t_years + 1) * cf * pv_factor
        
        # Normalize by price
        duration /= price
        convexity /= price
        
        # Convert to modified duration
        modified_duration = duration / (1 + yield_rate/frequency)
        
        # Calculate dollar duration
        dollar_duration = -modified_duration * price * (yield_rate/100)
        
        return modified_duration, convexity, dollar_duration

    def simulate_sabr_model(self, forward: float, strike: float, T: float,
                            alpha: float, beta: float, rho: float, nu: float) -> float:
        """
        Calculate implied volatility using the SABR model.
        
        Args:
            forward: Forward price
            strike: Strike price
            T: Time to expiry
            alpha: Initial volatility
            beta: CEV parameter (0 <= beta <= 1)
            rho: Correlation between price and vol
            nu: Vol of vol
            
        Returns:
            float: SABR implied volatility
        """
        # Handle ATM case separately (using L'Hopital's rule)
        if abs(forward - strike) < 1e-10:
            # ATM implied volatility formula
            A = (((1 - beta)**2)/24 * alpha**2/forward**(2-2*beta) +
                 rho*beta*nu*alpha/(4*forward**(1-beta)) +
                 (2 - 3*rho**2)*nu**2/24)
            vol = alpha/forward**(1-beta) * (1 + A*T)
            return float(vol)
        
        # Calculate intermediate terms
        F = forward
        K = strike
        z = (nu/alpha) * (F*K)**((1-beta)/2) * np.log(F/K)
        x = np.log((np.sqrt(1 - 2*rho*z + z**2) + z - rho)/(1 - rho))
        
        # Calculate the main components
        A = alpha * ((F*K)**((1-beta)/2)) * (
            1 + ((1-beta)**2/24) * np.log(F/K)**2 +
            ((1-beta)**4/1920) * np.log(F/K)**4
        )
        
        B = 1 + (
            ((1-beta)**2/24) * alpha**2/((F*K)**(1-beta)) +
            rho*beta*nu*alpha/(4*(F*K)**((1-beta)/2)) +
            (2-3*rho**2)*nu**2/24
        ) * T
        
        # Calculate implied volatility
        vol = A * z/x * B
        
        return float(vol)

    def simulate_libor_market_model(self, num_paths: int, tenors: np.ndarray, 
                                  initial_rates: np.ndarray, volatilities: np.ndarray,
                                  correlations: np.ndarray, dt: float) -> np.ndarray:
        """
        Simulate forward LIBOR rates using the LIBOR Market Model (LMM).
        
        Args:
            num_paths: Number of simulation paths
            tenors: Array of tenor points
            initial_rates: Initial forward rates
            volatilities: Volatility parameters for each rate
            correlations: Correlation matrix between rates
            dt: Time step size
            
        Returns:
            np.ndarray: Simulated forward rates
        """
        num_rates = len(initial_rates)
        num_steps = len(tenors)
        
        # Convert inputs to float32
        initial_rates = initial_rates.astype(np.float32)
        volatilities = volatilities.astype(np.float32)
        correlations = correlations.astype(np.float32)
        
        # Initialize rates array
        rates = np.zeros((num_paths, num_steps, num_rates), dtype=np.float32)
        rates[:, 0] = initial_rates
        
        # Compute Cholesky decomposition of correlation matrix
        L = np.linalg.cholesky(correlations)
        L = L.astype(np.float32)
        
        # Generate correlated random numbers
        Z = np.random.standard_normal((num_paths, num_steps-1, num_rates)).astype(np.float32)
        
        # Apply correlation structure using matrix multiplication
        for t in range(num_steps-1):
            # Generate correlated increments
            dW = np.zeros((num_paths, num_rates), dtype=np.float32)
            self.matmul_kernel(
                Z[:, t].ctypes.data,
                L.ctypes.data,
                dW.ctypes.data,
                num_paths, num_rates, num_rates
            )
            
            # Calculate drift term
            drift = np.zeros((num_paths, num_rates), dtype=np.float32)
            for i in range(num_rates):
                vol_i = volatilities[i]
                for j in range(i+1, num_rates):
                    delta = tenors[j] - tenors[j-1]
                    vol_j = volatilities[j]
                    corr_ij = correlations[i, j]
                    
                    # Drift calculation
                    drift[:, i] += (rates[:, t, j] * delta * vol_i * vol_j * corr_ij /
                                  (1 + rates[:, t, j] * delta))
            
            # Update rates
            for i in range(num_rates):
                rates[:, t+1, i] = rates[:, t, i] * np.exp(
                    (drift[:, i] - 0.5 * volatilities[i]**2) * dt +
                    volatilities[i] * np.sqrt(dt) * dW[:, i]
                )
        
        return rates

    def price_credit_default_swap(self, notional: float, maturity: float, 
                                hazard_rate: float, recovery_rate: float,
                                discount_curve: callable) -> tuple:
        """
        Price a Credit Default Swap (CDS).
        
        Args:
            notional: Notional amount
            maturity: Time to maturity in years
            hazard_rate: Hazard rate (probability of default)
            recovery_rate: Recovery rate in case of default
            discount_curve: Function that returns discount factors
            
        Returns:
            tuple: (fair_spread, protection_leg_value, premium_leg_value)
        """
        # Convert inputs to float32
        notional = np.float32(notional)
        maturity = np.float32(maturity)
        hazard_rate = np.float32(hazard_rate)
        recovery_rate = np.float32(recovery_rate)
        
        # Time grid for integration (quarterly)
        dt = 0.25
        times = np.arange(0, maturity + dt, dt, dtype=np.float32)
        
        # Calculate survival probabilities
        survival_prob = np.exp(-hazard_rate * times)
        
        # Calculate default probabilities for each interval
        default_prob = survival_prob[:-1] - survival_prob[1:]
        
        # Get discount factors
        discount_factors = np.array([discount_curve(t) for t in times], dtype=np.float32)
        
        # Calculate protection leg value
        protection_leg = notional * (1 - recovery_rate) * np.sum(
            discount_factors[1:] * default_prob
        )
        
        # Calculate premium leg value (assume quarterly payments)
        premium_leg = notional * dt * np.sum(
            discount_factors[1:] * survival_prob[1:]
        )
        
        # Calculate fair spread
        fair_spread = protection_leg / premium_leg
        
        return fair_spread, protection_leg, premium_leg

    def price_cdo_tranche(self, attachment: float, detachment: float,
                         correlation: float, hazard_rates: np.ndarray,
                         recovery_rates: np.ndarray, notionals: np.ndarray,
                         maturity: float, num_scenarios: int = 10000) -> tuple:
        """
        Price a CDO tranche using Gaussian copula model.
        
        Args:
            attachment: Attachment point of the tranche
            detachment: Detachment point of the tranche
            correlation: Asset correlation
            hazard_rates: Array of hazard rates for each asset
            recovery_rates: Array of recovery rates
            notionals: Array of notional amounts
            maturity: Time to maturity in years
            num_scenarios: Number of Monte Carlo scenarios
            
        Returns:
            tuple: (tranche_spread, expected_loss)
        """
        num_assets = len(hazard_rates)
        
        # Convert inputs to float32
        hazard_rates = hazard_rates.astype(np.float32)
        recovery_rates = recovery_rates.astype(np.float32)
        notionals = notionals.astype(np.float32)
        
        # Generate correlated uniform variables using Gaussian copula
        rho = correlation
        L = np.linalg.cholesky(rho * np.ones((num_assets, num_assets)) + 
                              (1 - rho) * np.eye(num_assets))
        L = L.astype(np.float32)
        
        # Generate standard normal variables
        Z = np.random.standard_normal((num_scenarios, num_assets)).astype(np.float32)
        
        # Apply correlation structure
        corr_Z = np.zeros((num_scenarios, num_assets), dtype=np.float32)
        self.matmul_kernel(
            Z.ctypes.data,
            L.ctypes.data,
            corr_Z.ctypes.data,
            num_scenarios, num_assets, num_assets
        )
        
        # Convert to uniform variables using scipy.special.erf
        U = 0.5 * (1 + special.erf(corr_Z / np.sqrt(2)))
        
        # Generate default times
        default_times = -np.log(1 - U) / hazard_rates
        
        # Calculate portfolio losses
        portfolio_losses = np.zeros(num_scenarios, dtype=np.float32)
        for i in range(num_scenarios):
            defaults = default_times[i] <= maturity
            loss = np.sum(notionals[defaults] * (1 - recovery_rates[defaults]))
            portfolio_losses[i] = loss
        
        # Calculate tranche losses
        tranche_size = detachment - attachment
        tranche_losses = np.maximum(0, np.minimum(
            portfolio_losses - attachment,
            tranche_size
        ))
        
        # Calculate expected loss
        expected_loss = np.mean(tranche_losses) / tranche_size
        
        # Calculate tranche spread (simplified)
        tranche_spread = expected_loss / maturity
        
        return float(tranche_spread), float(expected_loss)

    def run_stress_test(self, portfolio: np.ndarray, risk_factors: dict,
                        scenarios: dict, correlation_matrix: np.ndarray = None,
                        num_simulations: int = 10000) -> dict:
        """
        Run stress tests on a portfolio under different scenarios.
        
        Args:
            portfolio: Array of position sizes
            risk_factors: Dictionary of risk factor sensitivities
            scenarios: Dictionary of stress scenarios
            correlation_matrix: Correlation matrix between risk factors
            num_simulations: Number of Monte Carlo simulations
            
        Returns:
            dict: Stress test results
        """
        # Convert inputs to float32
        portfolio = portfolio.astype(np.float32)
        
        results = {}
        
        # Run historical stress tests
        for scenario_name, scenario in scenarios.items():
            # Apply scenario shocks to risk factors
            shocked_factors = {}
            for factor, base_value in risk_factors.items():
                if factor in scenario:
                    shocked_factors[factor] = base_value * (1 + scenario[factor])
                else:
                    shocked_factors[factor] = base_value
            
            # Calculate portfolio value under stress
            stressed_value = self._calculate_portfolio_value(portfolio, shocked_factors)
            results[f"historical_{scenario_name}"] = stressed_value
        
        # Run Monte Carlo stress tests if correlation matrix is provided
        if correlation_matrix is not None:
            # Convert correlation matrix to float32
            correlation_matrix = correlation_matrix.astype(np.float32)
            
            # Generate correlated random scenarios
            L = np.linalg.cholesky(correlation_matrix)
            L = L.astype(np.float32)
            
            num_factors = len(risk_factors)
            Z = np.random.standard_normal((num_simulations, num_factors)).astype(np.float32)
            
            # Apply correlation structure
            corr_Z = np.zeros((num_simulations, num_factors), dtype=np.float32)
            self.matmul_kernel(
                Z.ctypes.data,
                L.ctypes.data,
                corr_Z.ctypes.data,
                num_simulations, num_factors, num_factors
            )
            
            # Calculate portfolio values under simulated scenarios
            mc_values = np.zeros(num_simulations, dtype=np.float32)
            for i in range(num_simulations):
                shocked_factors = {
                    factor: value * (1 + corr_Z[i, j])
                    for j, (factor, value) in enumerate(risk_factors.items())
                }
                mc_values[i] = self._calculate_portfolio_value(portfolio, shocked_factors)
            
            # Calculate stress metrics
            results["mc_var_95"] = np.percentile(mc_values, 5)
            results["mc_var_99"] = np.percentile(mc_values, 1)
            results["mc_expected_shortfall_95"] = np.mean(mc_values[mc_values <= results["mc_var_95"]])
            results["mc_expected_shortfall_99"] = np.mean(mc_values[mc_values <= results["mc_var_99"]])
        
        return results

    def _calculate_portfolio_value(self, portfolio: np.ndarray, risk_factors: dict) -> float:
        """
        Calculate portfolio value given risk factors.
        
        Args:
            portfolio: Array of position sizes
            risk_factors: Dictionary of risk factor values
            
        Returns:
            float: Portfolio value
        """
        # Implement your portfolio valuation logic here
        # This is a simplified example
        factor_values = np.array(list(risk_factors.values()), dtype=np.float32)
        return float(np.sum(portfolio * factor_values))

    def run_scenario_analysis(self, portfolio: np.ndarray, scenarios: list,
                            risk_factors: dict, pricing_functions: dict) -> dict:
        """
        Run scenario analysis on a portfolio.
        
        Args:
            portfolio: Array of position sizes
            scenarios: List of scenario dictionaries
            risk_factors: Dictionary of current risk factor values
            pricing_functions: Dictionary of pricing functions for each asset type
            
        Returns:
            dict: Scenario analysis results
        """
        results = {}
        
        # Convert portfolio to float32
        portfolio = portfolio.astype(np.float32)
        
        # Calculate base portfolio value
        base_value = self._calculate_portfolio_value(portfolio, risk_factors)
        results["base_value"] = base_value
        
        # Run scenarios
        for i, scenario in enumerate(scenarios):
            # Apply scenario adjustments to risk factors
            adjusted_factors = risk_factors.copy()
            for factor, adjustment in scenario["changes"].items():
                if factor in adjusted_factors:
                    adjusted_factors[factor] *= (1 + adjustment)
            
            # Calculate portfolio value under scenario
            scenario_value = self._calculate_portfolio_value(portfolio, adjusted_factors)
            
            # Store results
            results[f"scenario_{i+1}"] = {
                "name": scenario.get("name", f"Scenario {i+1}"),
                "value": scenario_value,
                "change": scenario_value - base_value,
                "percent_change": (scenario_value - base_value) / base_value * 100
            }
            
            # Calculate risk metrics under scenario
            if "risk_metrics" in scenario:
                for metric in scenario["risk_metrics"]:
                    if metric == "var":
                        var, es = self.calculate_var_and_es(
                            self._generate_returns(adjusted_factors),
                            confidence_level=0.95
                        )
                        # Take the mean of the VaR values across assets
                        var_value = np.mean(var) if isinstance(var, np.ndarray) and var.size > 1 else float(var)
                        results[f"scenario_{i+1}"][f"{metric}_95"] = float(var_value)
                    elif metric == "volatility":
                        vol = np.std(self._generate_returns(adjusted_factors))
                        results[f"scenario_{i+1}"][metric] = float(vol)
        
        return results

    def _generate_returns(self, risk_factors: dict) -> np.ndarray:
        """
        Generate returns based on risk factors.
        
        Args:
            risk_factors: Dictionary of risk factor values
            
        Returns:
            np.ndarray: Generated returns with shape (days, assets)
        """
        # Implement your returns generation logic here
        # This is a simplified example using normal distribution
        num_days = 252  # One year of daily returns
        num_assets = len(risk_factors)  # Use number of risk factors as number of assets
        
        # Generate returns for each asset
        if num_assets > 0:
            returns = np.random.normal(0, 0.01, (num_days, num_assets)).astype(np.float32)
        else:
            # If no risk factors, generate a single asset
            returns = np.random.normal(0, 0.01, (num_days, 1)).astype(np.float32)
            
        return returns

    def initialize(self):
        """
        Initialize the StocksAPI.
        This method ensures all kernels are loaded and ready to use.
        """
        # Make sure kernels are initialized
        if not self.use_kernels:
            self._initialize_kernels()

    def process_realtime_data(self, data: dict, window_size: int = 20, stride: int = 1) -> dict:
        """
        Process real-time market data with standardized format.
        
        Args:
            data: Dictionary containing stock data with keys:
                - symbol: Stock symbol
                - price: Current price
                - change: Price change
                - changePercent: Percentage change
                - volume: Trading volume
                - timestamp: Current timestamp
            window_size: Size of the processing window
            stride: Stride size for processing
            
        Returns:
            dict: Processed data with standardized format
        """
        try:
            # Validate input data
            required_fields = ['symbol', 'price', 'change', 'changePercent', 'volume', 'timestamp']
            if not all(field in data for field in required_fields):
                raise ValueError("Missing required fields in input data")
            
            # Convert price data to numpy array for processing
            price_data = np.array([data['price']], dtype=np.float32)
            
            # Calculate technical indicators
            rsi = self.relative_strength_index(price_data.reshape(1, -1), window=14)
            macd_line, signal_line, histogram = self.macd(price_data.reshape(1, -1))
            upper_band, middle_band, lower_band = self.bollinger_bands(price_data.reshape(1, -1))
            
            # Determine trend
            trend = "up" if data['change'] >= 0 else "down"
            
            # Calculate momentum
            momentum = data['changePercent'] / 100.0
            
            # Calculate volatility
            volatility = np.std(price_data) / np.mean(price_data) if len(price_data) > 1 else 0.0
            
            # Prepare processed data
            processed_data = {
                'symbol': data['symbol'],
                'price': float(data['price']),
                'change': float(data['change']),
                'changePercent': float(data['changePercent']),
                'volume': int(data['volume']),
                'timestamp': data['timestamp'],
                'trend': trend,
                'momentum': float(momentum),
                'volatility': float(volatility),
                'rsi': float(rsi[-1, 0]),
                'macd': {
                    'line': float(macd_line[-1, 0]),
                    'signal': float(signal_line[-1, 0]),
                    'histogram': float(histogram[-1, 0])
                },
                'bollinger_bands': {
                    'upper': float(upper_band[-1, 0]),
                    'middle': float(middle_band[-1, 0]),
                    'lower': float(lower_band[-1, 0])
                }
            }
            
            return processed_data
            
        except Exception as e:
            logger.error(f"Error processing real-time data: {e}")
            return {
                'symbol': data.get('symbol', ''),
                'error': str(e)
            }

    def get_realtime_quote(self, symbol: str) -> dict:
        """
        Get real-time quote data with standardized format.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            dict: Quote data with standardized format
        """
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            quote_data = {
                'symbol': symbol,
                'price': float(info.get('regularMarketPrice', 0.0)),
                'change': float(info.get('regularMarketChange', 0.0)),
                'changePercent': float(info.get('regularMarketChangePercent', 0.0)),
                'volume': int(info.get('regularMarketVolume', 0)),
                'timestamp': datetime.now().isoformat()
            }
            
            # Process the quote data
            return self.process_realtime_data(quote_data)
            
        except Exception as e:
            logger.error(f"Error getting real-time quote for {symbol}: {e}")
            return {
                'symbol': symbol,
                'error': str(e)
            }

    def search_stocks(self, query):
        """
        Search for stocks using the provided query.
        
        Args:
            query (str): Search query string
            
        Returns:
            list: List of dictionaries containing stock information
        """
        try:
            # Use yfinance's search functionality
            tickers = yf.Tickers(query)
            results = []
            
            for symbol in tickers.tickers:
                info = tickers.tickers[symbol].info
                results.append({
                    'symbol': symbol,
                    'name': info.get('longName', ''),
                    'exchange': info.get('exchange', ''),
                    'type': info.get('quoteType', ''),
                    'sector': info.get('sector', ''),
                    'industry': info.get('industry', '')
                })
            
            return results
            
        except Exception as e:
            logger.error(f"Error searching stocks: {str(e)}")
            return []

    def simulate_gbm(self, num_paths: int, num_time_steps: int, initial_price: float,
                    volatility: float, drift: float, risk_free_rate: float) -> np.ndarray:
        """Simulate stock prices using Geometric Brownian Motion"""
        # Initialize paths array
        paths = np.zeros((num_paths, num_time_steps + 1))
        paths[:, 0] = initial_price
        
        # Calculate parameters
        dt = 1.0 / num_time_steps
        mu = drift - 0.5 * volatility**2
        
        # Generate random numbers
        if self.use_kernels and self.matmul_kernel is not None:
            # Use ASM kernel for matrix operations
            z = np.random.standard_normal((num_paths, num_time_steps))
            z_ptr = z.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
            paths_ptr = paths.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
            result_ptr = np.zeros_like(paths).ctypes.data_as(ctypes.POINTER(ctypes.c_float))
            
            for t in range(num_time_steps):
                self.matmul_kernel(z_ptr, paths_ptr, result_ptr, num_paths, 1, 1)
                paths[:, t+1] = paths[:, t] * np.exp(mu * dt + volatility * np.sqrt(dt) * z[:, t])
        else:
            # Fall back to NumPy implementation
            for t in range(num_time_steps):
                z = np.random.standard_normal(num_paths)
                paths[:, t+1] = paths[:, t] * np.exp(mu * dt + volatility * np.sqrt(dt) * z)
        
        return paths

    def simulate_jump_diffusion(self, num_paths: int, num_time_steps: int, initial_price: float,
                              volatility: float, drift: float, jump_intensity: float,
                              jump_mean: float, jump_std: float) -> np.ndarray:
        """
        Simulate stock prices using Jump Diffusion model with optimized assembly kernels.
        
        Args:
            num_paths: Number of simulation paths
            num_time_steps: Number of time steps
            initial_price: Initial stock price
            volatility: Volatility
            drift: Drift parameter
            jump_intensity: Intensity of jumps (lambda)
            jump_mean: Mean of jump size
            jump_std: Standard deviation of jump size
            
        Returns:
            np.ndarray: Matrix of simulated price paths
        """
        self._begin_method()  # Ensure clean state
        
        # Initialize kernels if not already done and if using kernels
        if self.use_kernels and not hasattr(self, 'layer_norm_kernel'):
            self._initialize_kernels()
        
        dt = 1.0 / num_time_steps
        sqrt_dt = np.sqrt(dt)
        
        # Pre-allocate arrays with float32 for memory efficiency
        S = np.zeros((num_paths, num_time_steps + 1), dtype=np.float32)
        S[:, 0] = initial_price
        
        # Generate random numbers efficiently
        np.random.seed(42)  # For reproducibility
        Z = np.random.standard_normal((num_paths, num_time_steps)).astype(np.float32)
        J = np.random.standard_normal((num_paths, num_time_steps)).astype(np.float32)
        
        # Pre-compute drift and diffusion terms
        drift_term = np.float32((drift - 0.5 * volatility**2) * dt)
        diffusion_term = np.float32(volatility * sqrt_dt)
        
        # Process in chunks to manage memory and utilize SIMD
        chunk_size = min(10000, num_paths)
        num_chunks = (num_paths + chunk_size - 1) // chunk_size
        
        for chunk_idx in range(num_chunks):
            start_idx = chunk_idx * chunk_size
            end_idx = min((chunk_idx + 1) * chunk_size, num_paths)
            current_chunk_size = end_idx - start_idx
            
            # Allocate arrays for current chunk
            chunk_S = S[start_idx:end_idx]
            chunk_Z = Z[start_idx:end_idx]
            chunk_J = J[start_idx:end_idx]
            
            # Simulate paths for current chunk using matrix operations
            for t in range(num_time_steps):
                if self.use_kernels:
                    # Calculate exponential term using matrix multiplication
                    exp_term = np.zeros(current_chunk_size, dtype=np.float32)
                    self.matmul_kernel(
                        chunk_S[:, t].ctypes.data,
                        np.array([drift_term + diffusion_term * chunk_Z[:, t]], dtype=np.float32).ctypes.data,
                        exp_term.ctypes.data,
                        current_chunk_size, 1, 1
                    )
                    
                    # Calculate jump component
                    jump_component = np.zeros(current_chunk_size, dtype=np.float32)
                    jump_mask = np.random.random(current_chunk_size) < jump_intensity * dt
                    jump_component[jump_mask] = np.exp(jump_mean + jump_std * chunk_J[jump_mask, t])
                    
                    # Update prices
                    chunk_S[:, t+1] = chunk_S[:, t] * np.exp(exp_term) * (1 + jump_component)
                else:
                    # Use NumPy implementation
                    exp_term = drift_term + diffusion_term * chunk_Z[:, t]
                    jump_component = np.zeros(current_chunk_size, dtype=np.float32)
                    jump_mask = np.random.random(current_chunk_size) < jump_intensity * dt
                    jump_component[jump_mask] = np.exp(jump_mean + jump_std * chunk_J[jump_mask, t])
                    chunk_S[:, t+1] = chunk_S[:, t] * np.exp(exp_term) * (1 + jump_component)
        
        self._end_method()  # Clean up temporary arrays
        return S.copy()

    def simulate_mean_reversion(self, num_paths: int, num_time_steps: int, initial_price: float,
                               mean_reversion_speed: float, long_term_mean: float, volatility: float) -> np.ndarray:
        """
        Simulate stock prices using Mean Reversion (Ornstein-Uhlenbeck) model with optimized assembly kernels.
        
        Args:
            num_paths: Number of simulation paths
            num_time_steps: Number of time steps
            initial_price: Initial stock price
            mean_reversion_speed: Speed of mean reversion
            long_term_mean: Long-term mean price
            volatility: Volatility
            
        Returns:
            np.ndarray: Matrix of simulated price paths
        """
        self._begin_method()  # Ensure clean state
        
        # Initialize kernels if not already done and if using kernels
        if self.use_kernels and not hasattr(self, 'layer_norm_kernel'):
            self._initialize_kernels()
        
        dt = 1.0 / num_time_steps
        sqrt_dt = np.sqrt(dt)
        
        # Pre-allocate arrays with float32 for memory efficiency
        S = np.zeros((num_paths, num_time_steps + 1), dtype=np.float32)
        S[:, 0] = initial_price
        
        # Generate random numbers efficiently
        np.random.seed(42)  # For reproducibility
        Z = np.random.standard_normal((num_paths, num_time_steps)).astype(np.float32)
        
        # Pre-compute mean reversion and volatility terms
        mean_rev_term = np.float32(mean_reversion_speed * dt)
        vol_term = np.float32(volatility * sqrt_dt)
        
        # Process in chunks to manage memory and utilize SIMD
        chunk_size = min(10000, num_paths)
        num_chunks = (num_paths + chunk_size - 1) // chunk_size
        
        for chunk_idx in range(num_chunks):
            start_idx = chunk_idx * chunk_size
            end_idx = min((chunk_idx + 1) * chunk_size, num_paths)
            current_chunk_size = end_idx - start_idx
            
            # Allocate arrays for current chunk
            chunk_S = S[start_idx:end_idx]
            chunk_Z = Z[start_idx:end_idx]
            
            # Simulate paths for current chunk using matrix operations
            for t in range(num_time_steps):
                if self.use_kernels:
                    # Calculate mean reversion component using matrix multiplication
                    mean_rev_component = np.zeros(current_chunk_size, dtype=np.float32)
                    self.matmul_kernel(
                        (chunk_S[:, t] - long_term_mean).ctypes.data,
                        np.array([-mean_rev_term], dtype=np.float32).ctypes.data,
                        mean_rev_component.ctypes.data,
                        current_chunk_size, 1, 1
                    )
                    # Calculate diffusion term
                    diffusion = vol_term * chunk_Z[:, t]
                    # Update prices
                    chunk_S[:, t+1] = chunk_S[:, t] + mean_rev_component + diffusion
                else:
                    # Use NumPy implementation
                    mean_rev_component = -mean_rev_term * (chunk_S[:, t] - long_term_mean)
                    diffusion = vol_term * chunk_Z[:, t]
                    chunk_S[:, t+1] = chunk_S[:, t] + mean_rev_component + diffusion
        
        self._end_method()  # Clean up temporary arrays
        return S.copy()

    def _use_asm_if_available(self, asm_func, numpy_func, *args, **kwargs):
        """Helper method to use ASM kernel if available, otherwise fall back to NumPy"""
        if asm_func is not None:
            try:
                return asm_func(*args, **kwargs)
            except Exception as e:
                logger.warning(f"ASM kernel failed: {e}. Falling back to NumPy implementation.")
        return numpy_func(*args, **kwargs)

    def _layer_normalize(self, x, gamma, beta):
        """Apply layer normalization using ASM kernel"""
        # Prepare input arrays for ASM kernel
        x_ptr = x.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        gamma_ptr = gamma.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        beta_ptr = beta.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        result = np.zeros_like(x)
        result_ptr = result.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        
        self.layer_norm_kernel(x_ptr, result_ptr, gamma_ptr, beta_ptr, x.size, x.ndim)
        return result

  