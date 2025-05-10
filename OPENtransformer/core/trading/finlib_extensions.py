#!/usr/bin/env python
"""
Extensions to the StocksAPI class from finlib.

This module provides additional methods for the StocksAPI class,
including the Black-Scholes VaR calculation that was missing.
"""

import numpy as np
import logging
from scipy import stats
from finlib.APIS.stocks_api import StocksAPI

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("finlib_extensions")

class ExtendedStocksAPI(StocksAPI):
    """
    Extended version of the StocksAPI class with additional methods.
    """
    
    def __init__(self):
        """
        Initialize the ExtendedStocksAPI
        """
        super().__init__()
        logger.info("Initialized ExtendedStocksAPI")
    
    def get_historical_data(self, tickers=None, start_date=None, end_date=None):
        """
        Placeholder for getting historical data.
        In a real implementation, this would fetch data from a data provider.
        
        Args:
            tickers: List of ticker symbols
            start_date: Start date for historical data
            end_date: End date for historical data
            
        Returns:
            Numpy array with shape (days, assets) containing price data
        """
        logger.warning("Using placeholder get_historical_data method")
        # This is a placeholder - in a real implementation, this would fetch data
        # Return the input data if provided
        if isinstance(tickers, np.ndarray):
            return tickers
        
        # Otherwise, return a dummy array
        return np.random.rand(100, len(tickers) if tickers else 1).astype(np.float32)
    
    def compute_daily_returns(self, prices):
        """
        Compute daily returns from price data
        
        Args:
            prices: Numpy array with shape (days, assets) containing price data
            
        Returns:
            Numpy array with shape (days-1, assets) containing daily returns
        """
        if not isinstance(prices, np.ndarray):
            prices = np.array(prices, dtype=np.float32)
        
        returns = np.zeros((prices.shape[0] - 1, prices.shape[1]), dtype=np.float32)
        
        for i in range(1, prices.shape[0]):
            returns[i-1] = (prices[i] / prices[i-1]) - 1
        
        return returns
    
    def calculate_black_scholes_var(self, returns, confidence_level=0.95, num_simulations=1000000, time_horizon=1):
        """
        Calculate Value at Risk (VaR) using the Black-Scholes model
        
        Args:
            returns: Numpy array with shape (days, assets) containing historical returns
            confidence_level: Confidence level for VaR calculation (default: 0.95)
            num_simulations: Number of Monte Carlo simulations (default: 1000000)
            time_horizon: Time horizon in days (default: 1)
            
        Returns:
            Numpy array with VaR values for each asset
        """
        # Ensure returns is a numpy array
        if not isinstance(returns, np.ndarray):
            returns = np.array(returns, dtype=np.float32)
        
        # Handle 1D arrays
        if len(returns.shape) == 1:
            returns = returns.reshape(-1, 1)
        
        # Calculate mean and standard deviation of returns
        mu = np.mean(returns, axis=0)
        sigma = np.std(returns, axis=0)
        
        # Number of assets
        num_assets = returns.shape[1]
        
        # Initialize VaR array
        var = np.zeros(num_assets, dtype=np.float32)
        
        # Calculate VaR for each asset
        for i in range(num_assets):
            # Generate random returns using Black-Scholes model
            random_returns = np.random.normal(
                mu[i] * time_horizon,
                sigma[i] * np.sqrt(time_horizon),
                num_simulations
            ).astype(np.float32)
            
            # Sort returns
            sorted_returns = np.sort(random_returns)
            
            # Calculate VaR
            var_index = int((1 - confidence_level) * num_simulations)
            var[i] = -sorted_returns[var_index]
        
        return var
    
    def compute_black_scholes_option_price(self, S, K, T, r, sigma, option_type='call'):
        """
        Calculate option price using the Black-Scholes formula
        
        Args:
            S: Current stock price
            K: Strike price
            T: Time to maturity in years
            r: Risk-free interest rate
            sigma: Volatility
            option_type: 'call' or 'put'
            
        Returns:
            Option price
        """
        # Calculate d1 and d2
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        # Calculate option price
        if option_type.lower() == 'call':
            option_price = S * stats.norm.cdf(d1) - K * np.exp(-r * T) * stats.norm.cdf(d2)
        else:  # put
            option_price = K * np.exp(-r * T) * stats.norm.cdf(-d2) - S * stats.norm.cdf(-d1)
        
        return option_price
    
    def compute_portfolio_var(self, returns, weights, confidence_level=0.95, method='historical'):
        """
        Calculate portfolio Value at Risk (VaR)
        
        Args:
            returns: Numpy array with shape (days, assets) containing historical returns
            weights: Numpy array with shape (assets,) containing portfolio weights
            confidence_level: Confidence level for VaR calculation (default: 0.95)
            method: Method for VaR calculation ('historical', 'parametric', or 'monte_carlo')
            
        Returns:
            Portfolio VaR
        """
        # Ensure returns and weights are numpy arrays
        if not isinstance(returns, np.ndarray):
            returns = np.array(returns, dtype=np.float32)
        
        if not isinstance(weights, np.ndarray):
            weights = np.array(weights, dtype=np.float32)
        
        # Handle 1D arrays
        if len(returns.shape) == 1:
            returns = returns.reshape(-1, 1)
        
        # Calculate portfolio returns
        portfolio_returns = np.dot(returns, weights)
        
        if method == 'historical':
            # Historical VaR
            sorted_returns = np.sort(portfolio_returns)
            var_index = int((1 - confidence_level) * len(sorted_returns))
            var = -sorted_returns[var_index]
            
        elif method == 'parametric':
            # Parametric VaR
            mu = np.mean(portfolio_returns)
            sigma = np.std(portfolio_returns)
            z_score = stats.norm.ppf(1 - confidence_level)
            var = -(mu + z_score * sigma)
            
        elif method == 'monte_carlo':
            # Monte Carlo VaR
            mu = np.mean(portfolio_returns)
            sigma = np.std(portfolio_returns)
            num_simulations = 1000000
            random_returns = np.random.normal(mu, sigma, num_simulations).astype(np.float32)
            sorted_returns = np.sort(random_returns)
            var_index = int((1 - confidence_level) * num_simulations)
            var = -sorted_returns[var_index]
            
        else:
            raise ValueError(f"Unknown method: {method}")
        
        return var 