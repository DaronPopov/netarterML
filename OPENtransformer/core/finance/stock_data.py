import numpy as np
import yfinance as yf
from typing import Dict, List, Union, Any
import json

def get_stock_data(symbol: str, period: str = "1y", interval: str = "1d") -> Dict[str, Any]:
    """Get historical stock data for a given symbol."""
    try:
        stock = yf.Ticker(symbol)
        hist = stock.history(period=period, interval=interval)
        
        if hist.empty:
            return {"error": f"No data available for {symbol}"}
            
        # Convert to list of dictionaries
        results = []
        for timestamp, row in hist.iterrows():
            result = {
                "date": timestamp.strftime('%Y-%m-%d'),
                "price": float(row["Close"]),
                "open": float(row["Open"]),
                "high": float(row["High"]),
                "low": float(row["Low"]),
                "volume": int(row["Volume"]),
                "symbol": symbol
            }
            results.append(result)
            
        return results
    except Exception as e:
        return {"error": f"Error fetching stock data: {str(e)}"}

def calculate_var_and_es(returns: List[float], confidence_level: float = 0.95) -> Dict[str, float]:
    """Calculate Value at Risk (VaR) and Expected Shortfall (ES)."""
    try:
        returns_array = np.array(returns)
        var = np.percentile(returns_array, (1 - confidence_level) * 100)
        es = returns_array[returns_array <= var].mean()
        return {
            "var": float(var),
            "es": float(es)
        }
    except Exception as e:
        return {"error": f"Error calculating VaR/ES: {str(e)}"}

def simulate_heston_model(
    S0: float = 100.0,
    K: float = 100.0,
    T: float = 1.0,
    r: float = 0.05,
    kappa: float = 2.0,
    theta: float = 0.04,
    sigma: float = 0.2,
    rho: float = -0.7,
    n_steps: int = 1000,
    n_simulations: int = 10000
) -> Dict[str, List[float]]:
    """Simulate stock prices using the Heston model."""
    try:
        dt = T / n_steps
        S = np.zeros((n_simulations, n_steps + 1))
        v = np.zeros((n_simulations, n_steps + 1))
        
        # Initialize
        S[:, 0] = S0
        v[:, 0] = theta
        
        # Generate correlated Brownian motions
        dW1 = np.random.normal(0, np.sqrt(dt), (n_simulations, n_steps))
        dW2 = rho * dW1 + np.sqrt(1 - rho**2) * np.random.normal(0, np.sqrt(dt), (n_simulations, n_steps))
        
        # Simulate paths
        for t in range(n_steps):
            v[:, t+1] = np.maximum(v[:, t] + kappa * (theta - v[:, t]) * dt + 
                                 sigma * np.sqrt(v[:, t]) * dW2[:, t], 0)
            S[:, t+1] = S[:, t] * np.exp((r - 0.5 * v[:, t]) * dt + 
                                        np.sqrt(v[:, t]) * dW1[:, t])
        
        return {
            "times": [float(t * dt) for t in range(n_steps + 1)],
            "prices": [float(price) for price in S.mean(axis=0)]
        }
    except Exception as e:
        return {"error": f"Error simulating Heston model: {str(e)}"}

def calculate_technical_indicators(prices: List[float], window: int = 20) -> Dict[str, List[float]]:
    """Calculate technical indicators from price data."""
    try:
        prices_array = np.array(prices)
        
        # Calculate SMA
        sma = np.convolve(prices_array, np.ones(window)/window, mode='valid')
        
        # Calculate EMA
        ema = np.zeros_like(prices_array)
        multiplier = 2 / (window + 1)
        ema[0] = prices_array[0]
        for i in range(1, len(prices_array)):
            ema[i] = (prices_array[i] - ema[i-1]) * multiplier + ema[i-1]
        
        # Calculate RSI
        delta = np.diff(prices_array)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = np.convolve(gain, np.ones(window)/window, mode='valid')
        avg_loss = np.convolve(loss, np.ones(window)/window, mode='valid')
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return {
            "sma": [float(x) for x in sma],
            "ema": [float(x) for x in ema],
            "rsi": [float(x) for x in rsi]
        }
    except Exception as e:
        return {"error": f"Error calculating technical indicators: {str(e)}"}

def calculate_portfolio_metrics(returns: List[float], weights: List[float]) -> Dict[str, float]:
    """Calculate portfolio metrics based on returns and weights."""
    try:
        returns_array = np.array(returns)
        weights_array = np.array(weights)
        
        # Calculate portfolio return
        portfolio_return = np.sum(returns_array * weights_array)
        
        # Calculate portfolio volatility
        portfolio_volatility = np.sqrt(np.sum(weights_array**2 * np.var(returns_array)))
        
        # Calculate Sharpe ratio (assuming risk-free rate of 0.02)
        risk_free_rate = 0.02
        sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility
        
        return {
            "return": float(portfolio_return),
            "volatility": float(portfolio_volatility),
            "sharpe_ratio": float(sharpe_ratio)
        }
    except Exception as e:
        return {"error": f"Error calculating portfolio metrics: {str(e)}"}

def handle_request(request: Dict[str, Any]) -> Dict[str, Any]:
    """Handle incoming API requests."""
    try:
        function = request.get("function")
        params = request.get("params", {})
        
        if function == "get_stock_data":
            return get_stock_data(params.get("symbol"), params.get("period", "1y"), params.get("interval", "1d"))
        elif function == "calculate_var_and_es":
            return calculate_var_and_es(params.get("returns", []), params.get("confidence_level", 0.95))
        elif function == "simulate_heston_model":
            return simulate_heston_model(**params)
        elif function == "calculate_technical_indicators":
            return calculate_technical_indicators(params.get("prices", []), params.get("window", 20))
        elif function == "calculate_portfolio_metrics":
            return calculate_portfolio_metrics(params.get("returns", []), params.get("weights", []))
        else:
            return {"error": f"Unsupported function: {function}"}
    except Exception as e:
        return {"error": f"Error handling request: {str(e)}"}
