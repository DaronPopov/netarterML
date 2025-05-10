import numpy as np
import yfinance as yf
from typing import Tuple, List, Dict, Any

def calculate_var_es(returns: np.ndarray, var: np.ndarray, es: np.ndarray, 
                    num_assets: int, num_days: int, confidence_level: float) -> None:
    """
    Calculate Value at Risk (VaR) and Expected Shortfall (ES) for a portfolio.

    Args:
        returns: A 2D numpy array of asset returns (num_days x num_assets)
        var: A 1D numpy array to store VaR values
        es: A 1D numpy array to store ES values
        num_assets: Number of assets
        num_days: Number of days
        confidence_level: Confidence level for VaR and ES calculations
    """
    for i in range(num_assets):
        sorted_returns = np.sort(returns[:, i])
        var_index = int((1 - confidence_level) * num_days)
        var[i] = -sorted_returns[var_index]
        es[i] = -np.mean(sorted_returns[:var_index])

class StocksAPI:
    def __init__(self):
        self._cache = {}
        
    def get_stock_data(self, symbol: str, period: str = "1d") -> Dict[str, Any]:
        """Get real-time stock data from Yahoo Finance."""
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period=period)
            
            if hist.empty:
                return {"error": f"No data available for {symbol}"}
                
            # Format historical data consistently
            historical_data = []
            for index, row in hist.iterrows():
                historical_data.append({
                    "date": index.strftime("%Y-%m-%d"),
                    "price": float(row['Close']),
                    "open": float(row['Open']),
                    "high": float(row['High']),
                    "low": float(row['Low']),
                    "volume": float(row['Volume']),
                    "symbol": symbol
                })
            
            # Get the latest data point for current price info
            last_row = hist.iloc[-1]
            return {
                "symbol": symbol,
                "price": float(last_row['Close']),
                "volume": float(last_row['Volume']),
                "change": float((last_row['Close'] / last_row['Open'] - 1) * 100),
                "high": float(last_row['High']),
                "low": float(last_row['Low']),
                "timestamp": last_row.name.isoformat(),
                "historical_data": historical_data
            }
        except Exception as e:
            return {"error": str(e)}

    def calculate_var_and_es(self, returns: np.ndarray, confidence_level: float = 0.95) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate Value at Risk (VaR) and Expected Shortfall (ES)."""
        try:
            num_assets = returns.shape[1]
            var = np.zeros(num_assets)
            es = np.zeros(num_assets)
            
            for i in range(num_assets):
                sorted_returns = np.sort(returns[:, i])
                var_index = int((1 - confidence_level) * len(sorted_returns))
                var[i] = -sorted_returns[var_index]
                es[i] = -np.mean(sorted_returns[:var_index])
                
            return var, es
        except Exception as e:
            raise ValueError(f"Error calculating VaR/ES: {str(e)}")

    def simulate_heston_model(self, num_paths: int, num_time_steps: int, initial_price: float,
                            risk_free_rate: float, volatility: float, kappa: float, theta: float,
                            sigma: float, rho: float) -> np.ndarray:
        """Simulate stock prices using the Heston model."""
        try:
            dt = 1 / num_time_steps
            s = np.zeros((num_paths, num_time_steps))
            v = np.zeros((num_paths, num_time_steps))
            
            # Initialize
            s[:, 0] = initial_price
            v[:, 0] = volatility**2
            
            # Generate correlated random numbers
            np.random.seed(42)
            z1 = np.random.randn(num_paths, num_time_steps)
            z2 = np.random.randn(num_paths, num_time_steps)
            z2 = rho * z1 + np.sqrt(1 - rho**2) * z2
            
            # Simulate paths
            for t in range(1, num_time_steps):
                v[:, t] = np.maximum(0, v[:, t-1] + kappa * (theta - v[:, t-1]) * dt + 
                                   sigma * np.sqrt(v[:, t-1] * dt) * z2[:, t-1])
                s[:, t] = s[:, t-1] * np.exp((risk_free_rate - 0.5 * v[:, t-1]) * dt + 
                                            np.sqrt(v[:, t-1] * dt) * z1[:, t-1])
            
            return s
        except Exception as e:
            raise ValueError(f"Error in Heston model simulation: {str(e)}")

    def calculate_technical_indicators(self, prices: np.ndarray, window: int = 14) -> Dict[str, np.ndarray]:
        """Calculate technical indicators including RSI, MACD, and Bollinger Bands."""
        try:
            # Calculate RSI
            delta = np.diff(prices)
            gain = np.where(delta > 0, delta, 0)
            loss = np.where(delta < 0, -delta, 0)
            avg_gain = np.mean(gain[:window])
            avg_loss = np.mean(loss[:window])
            
            for i in range(window, len(delta)):
                avg_gain = (avg_gain * (window - 1) + gain[i]) / window
                avg_loss = (avg_loss * (window - 1) + loss[i]) / window
            
            rs = avg_gain / avg_loss if avg_loss != 0 else 0
            rsi = 100 - (100 / (1 + rs))
            
            # Calculate MACD
            ema12 = self._calculate_ema(prices, 12)
            ema26 = self._calculate_ema(prices, 26)
            macd = ema12 - ema26
            signal = self._calculate_ema(macd, 9)
            histogram = macd - signal
            
            # Calculate Bollinger Bands
            sma = np.mean(prices[-window:])
            std = np.std(prices[-window:])
            upper_band = sma + (2 * std)
            lower_band = sma - (2 * std)
            
            return {
                "rsi": rsi,
                "macd": macd,
                "signal": signal,
                "histogram": histogram,
                "bollinger_upper": upper_band,
                "bollinger_middle": sma,
                "bollinger_lower": lower_band
            }
        except Exception as e:
            raise ValueError(f"Error calculating technical indicators: {str(e)}")

    def _calculate_ema(self, prices: np.ndarray, period: int) -> np.ndarray:
        """Calculate Exponential Moving Average."""
        multiplier = 2 / (period + 1)
        ema = np.zeros_like(prices)
        ema[0] = prices[0]
        
        for i in range(1, len(prices)):
            ema[i] = (prices[i] - ema[i-1]) * multiplier + ema[i-1]
            
        return ema

    def calculate_portfolio_metrics(self, returns: np.ndarray, weights: np.ndarray) -> Dict[str, float]:
        """Calculate portfolio metrics including return, volatility, and Sharpe ratio."""
        try:
            portfolio_returns = np.sum(returns * weights, axis=1)
            portfolio_return = np.mean(portfolio_returns) * 252  # Annualized
            portfolio_volatility = np.std(portfolio_returns) * np.sqrt(252)  # Annualized
            sharpe_ratio = portfolio_return / portfolio_volatility if portfolio_volatility != 0 else 0
            
            return {
                "return": float(portfolio_return),
                "volatility": float(portfolio_volatility),
                "sharpe_ratio": float(sharpe_ratio)
            }
        except Exception as e:
            raise ValueError(f"Error calculating portfolio metrics: {str(e)}")

# Create a singleton instance
stocks_api = StocksAPI()